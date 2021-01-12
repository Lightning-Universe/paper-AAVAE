import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import numpy as np
from torch.optim import Adam
from einops import repeat, rearrange
import math

import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as FM
from pytorch_lightning.callbacks import LearningRateMonitor

from pl_bolts.callbacks import LatentDimInterpolator
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    stl10_normalization,
    imagenet_normalization,
)
from resnet import (
    resnet18_encoder,
    resnet18_decoder,
    resnet50_encoder,
    resnet50_decoder,
)

from transforms import (
    LocalTransform,
    OriginalTransform,
    LinearEvalTrainTransform,
    LinearEvalValidTransform,
)
from online_eval import SSLOnlineEvaluator

encoders = {"resnet18": resnet18_encoder, "resnet50": resnet50_encoder}
decoders = {"resnet18": resnet18_decoder, "resnet50": resnet50_decoder}


def gaussian_likelihood(mean, logscale, sample):
    scale = torch.exp(logscale)
    dist = torch.distributions.Normal(mean, scale)
    log_pxz = dist.log_prob(sample)

    # sum over dimensions
    return log_pxz.sum(dim=(1, 2, 3))


def linear_warmup_decay(warmup_steps, total_steps, cosine=True, linear=False):
    """
    Linear warmup for warmup_steps, optionally with cosine annealing or
    linear decay to 0 at total_steps
    """
    assert not (linear and cosine)

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        if not (cosine or linear):
            # no decay
            return 1.0

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        if cosine:
            # cosine decay
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # linear decay
        return 1.0 - progress

    return fn


class TrainTransform:
    """
    TrainTransform returns a transformed image along with the original
    """

    def __init__(
        self,
        input_height: int = 224,
        dataset="cifar10",
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
        normalize=None,
    ):
        self.input_transform = LocalTransform(
            input_height=input_height,
            jitter_strength=jitter_strength,
            gaussian_blur=gaussian_blur,
            normalize=normalize,
        )
        self.original_transform = OriginalTransform(
            dataset=dataset, normalize=normalize
        )
        self.finetune_transform = LinearEvalTrainTransform(
            dataset=dataset, normalize=normalize
        )

    def __call__(self, x):
        return (
            self.input_transform(x),
            self.original_transform(x),
            self.finetune_transform(x),
        )


class EvalTransform:
    """
    EvalTransform returns the orginial image twice
    """

    def __init__(
        self, input_height: int = 224, dataset="cifar10", normalize=None
    ) -> None:
        self.original_transform = OriginalTransform(
            dataset=dataset, normalize=normalize
        )
        self.finetune_transform = LinearEvalValidTransform(
            dataset=dataset, normalize=normalize
        )

    def __call__(self, x):
        out = self.original_transform(x)
        return out, out, self.finetune_transform(x)


class ProjectionEncoder(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super(ProjectionEncoder, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        self.mu = nn.Linear(hidden_dim, output_dim, bias=False)
        self.logvar = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.first_layer(x)
        return self.mu(x), self.logvar(x)


class VAE(pl.LightningModule):
    def __init__(
        self,
        input_height,
        num_samples,
        gpus=1,
        batch_size=32,
        kl_coeff=0.1,
        h_dim=2048,
        latent_dim=128,
        learning_rate=1e-4,
        encoder="resnet18",
        decoder="resnet18",
        first_conv=False,
        maxpool1=False,
        dataset="cifar10",
        max_epochs=100,
        warmup_epochs=10,
        analytic=False,
        val_samples=1,
        cosine_decay=0,
        linear_decay=0,
        **kwargs,
    ):
        super(VAE, self).__init__()
        self.save_hyperparameters()

        self.input_height = input_height
        self.kl_coeff = kl_coeff
        self.analytic = analytic

        self.h_dim = h_dim
        self.latent_dim = latent_dim

        self.dataset = dataset
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.cosine_decay = cosine_decay
        self.linear_decay = linear_decay

        self.gpus = gpus
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.val_samples = val_samples

        global_batch_size = (
            self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        )
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        self.in_channels = 3

        self.encoder = encoders[encoder](self.first_conv, self.maxpool1)
        self.decoder = decoders[decoder](
            self.latent_dim, self.input_height, self.first_conv, self.maxpool1
        )

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.projection = ProjectionEncoder(
            input_dim=self.h_dim, hidden_dim=self.h_dim, output_dim=self.latent_dim
        )

    def forward(self, x):
        return self.decoder(x)

    def sample(self, z_mu, z_var, eps=1e-6):
        # add ep to prevent 0 variance
        std = torch.exp(z_var / 2) + eps

        p = torch.distributions.Normal(torch.zeros_like(z_mu), torch.ones_like(std))
        q = torch.distributions.Normal(z_mu, std)
        z = q.rsample()

        return p, q, z

    @staticmethod
    def kl_divergence_mc(p, q, z):
        log_pz = p.log_prob(z)
        log_qz = q.log_prob(z)

        kl = (log_qz - log_pz).sum(dim=-1)
        log_pz = log_pz.sum(dim=-1)
        log_qz = log_qz.sum(dim=-1)

        return kl, log_pz, log_qz

    @staticmethod
    def kl_divergence_analytic(p, q, z):
        log_pz = p.log_prob(z)
        log_qz = q.log_prob(z)

        # kl, log_pz, log_qz should be (batch * samples)
        kl = torch.distributions.kl.kl_divergence(q, p).sum(dim=-1)
        log_pz = p.log_prob(z).sum(dim=-1)
        log_qz = q.log_prob(z).sum(dim=-1)
        return kl, log_pz, log_qz

    def step(self, batch, samples=1):
        if self.dataset == "stl10":
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        (x, original, _), y = batch

        x = repeat(x, "b c h w -> (b samples) c h w", samples=samples)
        original = repeat(original, "b c h w -> (b samples) c h w", samples=samples)

        batch_size, c, h, w = x.shape
        pixels = c * h * w

        x_enc = self.encoder(x)
        mu, log_var = self.projection(x_enc)
        p, q, z = self.sample(mu, log_var)

        if self.analytic:
            kl, log_pz, log_qz = self.kl_divergence_analytic(p, q, z)
        else:
            kl, log_pz, log_qz = self.kl_divergence_mc(p, q, z)

        x_hat = self.decoder(z)

        log_pxz = gaussian_likelihood(x_hat, self.log_scale, original)

        elbo = (kl - log_pxz).mean()
        loss = (self.kl_coeff * kl - log_pxz).mean()

        # add samples dimension back
        log_pxz = rearrange(log_pxz, "(b samples) -> b samples", samples=samples)
        log_pz = rearrange(log_pz, "(b samples) -> b samples", samples=samples)
        log_qz = rearrange(log_qz, "(b samples) -> b samples", samples=samples)

        # marginal likelihood, logsumexp over sample dim, mean ove batch dim
        log_px = torch.logsumexp(log_pxz + log_pz - log_qz, dim=1).mean(dim=0) - np.log(
            samples
        )
        bpd = -log_px / (pixels * np.log(2))  # need log_px in base 2

        logs = {
            "kl": kl.mean(),
            "elbo": elbo,
            "bpd": bpd,
            "log_pxz": log_pxz.mean(),
            "log_px": log_px,
            "log_scale": self.log_scale.item(),
        }

        return loss, logs, z

    def training_step(self, batch, batch_idx):
        loss, logs, z = self.step(batch, 1)  # use only one sample for train
        self.log_dict({f"train_{k}": v for k, v in logs.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs, z = self.step(batch, self.val_samples)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        if self.warmup_epochs < 0:
            # no lr schedule
            return optimizer

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        # linear warmup with optional cosine decay
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(
                    warmup_steps, total_steps, self.cosine_decay, self.linear_decay
                ),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    # encoder/decoder params
    parser.add_argument("--encoder", default="resnet50", choices=encoders.keys())
    parser.add_argument("--decoder", default="resnet50", choices=decoders.keys())
    parser.add_argument("--h_dim", type=int, default=2048)
    parser.add_argument("--first_conv", type=bool, default=True)
    parser.add_argument("--maxpool1", type=bool, default=True)

    # vae params
    parser.add_argument("--kl_coeff", type=float, default=0.1)
    parser.add_argument("--latent_dim", type=int, default=128)
    # use analytic KL
    parser.add_argument("--analytic", type=int, default=0)

    # number of samples to use for validation
    parser.add_argument("--val_samples", type=int, default=1)

    # optimizer param
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=800)
    parser.add_argument("--cosine_decay", type=int, default=0)
    parser.add_argument("--linear_decay", type=int, default=0)

    # training params
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")

    # datamodule params
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument(
        "--dataset", type=str, default="cifar10"
    )  # cifar10, stl10, imagenet
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # transforms param
    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument("--gaussian_blur", type=bool, default=True)
    parser.add_argument("--jitter_strength", type=float, default=1.0)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # set hidden dim for resnet18
    if args.encoder == "resnet18":
        args.h_dim = 512

    if args.dataset == "cifar10":
        dm = CIFAR10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]

        args.maxpool1 = False
        args.first_conv = False
        normalization = cifar10_normalization()

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == "stl10":
        dm = STL10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples
        args.input_height = dm.size()[-1]

        args.maxpool1 = False
        args.first_conv = True
        normalization = stl10_normalization()
    elif args.dataset == "imagenet":
        dm = ImagenetDataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]

        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = TrainTransform(
        input_height=args.input_height,
        dataset=args.dataset,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
    )

    dm.val_transforms = EvalTransform(
        input_height=args.input_height, dataset=args.dataset, normalize=normalization
    )

    # model init
    model = VAE(**args.__dict__)

    online_evaluator = SSLOnlineEvaluator(
        z_dim=model.encoder.out_dim,
        num_classes=dm.num_classes,
        hidden_dim=None,
        drop_p=0.0,
        dataset=args.dataset,
    )

    interpolator = LatentDimInterpolator(
        interpolate_epoch_interval=20, range_start=-3, range_end=3, normalize=True
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        distributed_backend="ddp" if args.gpus > 1 else None,
        precision=16 if args.fp16 else 32,
        callbacks=[online_evaluator, interpolator, lr_monitor],
    )

    trainer.fit(model, dm)
