import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
import numpy as np

import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as FM

from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    stl10_normalization,
    imagenet_normalization
)

from resnet import (
    resnet18_encoder,
    resnet18_decoder,
    resnet50_encoder,
    resnet50_decoder,
)
from online_eval import SSLOnlineEvaluator
from metrics import gini_score, KurtosisScore
from transforms import Transforms

distributions = {
    "laplace": torch.distributions.Laplace,
    "normal": torch.distributions.Normal,
}

encoders = {"resnet18": resnet18_encoder, "resnet50": resnet50_encoder}
decoders = {"resnet18": resnet18_decoder, "resnet50": resnet50_decoder}


def discretized_logistic(mean, logscale, sample, binsize=1 / 256):
    mean = mean.clamp(min=-0.5 + 1 / 512, max=0.5 - 1 / 512)
    scale = torch.exp(logscale)
    sample = (torch.floor(sample / binsize) * binsize - mean) / scale
    log_pxz = torch.log(
        torch.sigmoid(sample + binsize / scale) - torch.sigmoid(sample) + 1e-7
    )
    return log_pxz.sum(dim=(1, 2, 3))


def gaussian_likelihood(mean, logscale, sample):
    scale = torch.exp(logscale)
    dist = torch.distributions.Normal(mean, scale)
    log_pxz = dist.log_prob(sample)
    return log_pxz.sum(dim=(1, 2, 3))


class ProjectionEncoder(nn.Module):
    def __init__(
        self,
        input_dim=2048,
        hidden_dim=2048,
        output_dim=128
    ):
        super(ProjectionEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.first_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU()
        )

        self.mu = nn.Linear(self.hidden_dim, self.output_dim, bias=False)
        self.logvar = nn.Linear(self.hidden_dim, self.output_dim, bias=False)

    def forward(self, x):
        x = self.first_layer(x)
        return self.mu(x), self.logvar(x)


"""
TODOs:
1. take a look at kl computation, elbo, log_pxz from discretized logistic and marg_log_px
2. fix kurtosis for epoch
3. take a look at transform for the latest set of runs
(separete views for kl, input and reconstruction, confirm this)
"""
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
        prior="normal",
        posterior="normal",
        first_conv=False,
        maxpool1=False,
        dataset='cifar10',
        max_epochs=100,
        warmup_epochs=10,
        warmup_start_lr=0.,
        eta_min=1e-6,
        **kwargs
    ):
        super(VAE, self).__init__()

        self.input_height = input_height
        self.kl_coeff = kl_coeff
        self.prior = prior
        self.posterior = posterior

        self.h_dim = h_dim
        self.latent_dim = latent_dim

        self.dataset = dataset
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        self.gpus = gpus
        self.batch_size = batch_size
        self.num_samples = num_samples

        global_batch_size = self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        self.in_channels = 3

        self.encoder = encoders[encoder](self.first_conv, self.maxpool1)
        self.decoder = decoders[decoder](
            self.latent_dim, self.input_height, self.first_conv, self.maxpool1
        )

        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        self.projection = ProjectionEncoder(
            input_dim=self.h_dim,
            hidden_dim=self.h_dim,
            output_dim=self.latent_dim
        )

        #self.train_kurtosis = KurtosisScore()
        #self.val_kurtosis = KurtosisScore()

    def forward(self, x):
        x = self.encoder(x)
        mu, logvar = self.projection(x)

        p, q, z = self.sample(mu, logvar)
        return z, self.decoder(z), p, q

    def sample(self, mu, logvar):
        std = torch.exp(logvar / 2)

        p = distributions[self.prior](torch.zeros_like(mu), torch.ones_like(std))
        q = distributions[self.posterior](mu, std)

        z = q.rsample()
        return p, q, z

    # TODO: verify kl computation
    def kl_divergence_mc(self, p, q, num_samples=1):
        z = p.rsample([num_samples])

        log_pz = p.log_prob(z)
        log_qz = q.log_prob(z)

        # mean over num_samples, sum over z_dim
        return (log_pz - log_qz).mean(dim=0).sum(dim=(1))

    def step(self, batch, batch_idx):
        if self.unlabeled_batch:
            batch = batch[0]

        (x1, x2, _), y = batch

        z, x1_hat, p, q = self.forward(x1)

        log_pxz = discretized_logistic(x1_hat, self.log_scale, x2)
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = self.kl_coeff * self.kl_divergence_mc(p, q)

        elbo = (kl - log_pxz).mean()
        bpd = elbo / (
            self.input_height * self.input_height * self.in_channels * np.log(2.0)
        )

        gini = gini_score(z)

        # TODO: this should be epoch metric
        #kurt = kurtosis_score(z)

        n = torch.tensor(x1.size(0)).type_as(x1)
        marg_log_px = torch.logsumexp(log_pxz + log_pz.sum(dim=-1) - log_qz.sum(dim=-1), dim=0) - torch.log(n)

        logs = {
            "kl": kl.mean(),
            "elbo": elbo,
            "gini": gini.mean(),
            #"kurtosis": kurt,
            "bpd": bpd,
            "log_pxz": log_pxz.mean(),
            "marginal_log_px": marg_log_px.mean(),
        }

        return elbo, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)

        # takes z sampled from latent
        #self.train_kurtosis.update(z)
        #self.log("train_kurtosis_score", self.train_kurtosis, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})

        # takes z sampled from latent
        #self.val_kurtosis.update(z)
        #self.log("val_kurtosis_score", self.val_kurtosis, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)

        # this needs to be per step
        self.warmup_epochs = self.train_iters_per_epoch * self.warmup_epochs
        self.max_epochs = self.train_iters_per_epoch * self.max_epochs

        linear_warmup_cosine_decay = LinearWarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=self.warmup_epochs,
            max_epochs=self.max_epochs,
            warmup_start_lr=self.warmup_start_lr,
            eta_min=self.eta_min,
        )

        scheduler = {
            'scheduler': linear_warmup_cosine_decay,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]


if __name__ == "__main__":
    pl.seed_everything(0)
    parser = argparse.ArgumentParser()

    # encoder/decoder params
    parser.add_argument("--encoder", default="resnet50", choices=encoders.keys())
    parser.add_argument("--decoder", default="resnet50", choices=decoders.keys())
    parser.add_argument('--h_dim', type=int, default=2048)
    parser.add_argument('--first_conv', type=bool, default=True)
    parser.add_argument('--maxpool1', type=bool, default=True)

    # vae params
    parser.add_argument('--kl_coeff', type=float, default=1.)  # try 10., 1., 0.1, 0.01
    parser.add_argument('--latent_dim', type=int, default=128)  # try 64, 128, 256, 512
    parser.add_argument('--prior', type=str, default='normal')  # normal/laplace
    parser.add_argument('--posterior', type=str, default='normal')  # normal/laplace

    # optimizer param
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # try both 1e-3/1e-4
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--warmup_start_lr', type=float, default=0.)
    parser.add_argument('--eta_min', type=float, default=1e-6)

    # training params
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument("--fp32", action='store_true')

    # datamodule params
    parser.add_argument('--data_path', type=str, default='.')
    parser.add_argument('--dataset', type=str, default="stl10")  # cifar10, stl10, imagenet
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)

    # transforms param
    parser.add_argument('--input_height', type=int, default=32)
    parser.add_argument('--gaussian_blur', type=bool, default=True)
    parser.add_argument('--jitter_strength', type=float, default=1.)
    parser.add_argument("--flip", action='store_true')

    tf_choices = ["original", "global", "local"]
    parser.add_argument("--input_transform", default="original", choices=tf_choices)
    parser.add_argument("--recon_transform", default="original", choices=tf_choices)

    args = parser.parse_args()

    dm = None
    if args.dataset == 'cifar10':
        dm = CIFAR10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]

        args.maxpool1 = False
        args.first_conv = False
        normalization = cifar10_normalization()

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == 'stl10':
        dm = STL10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples
        args.input_height = dm.size()[-1]

        args.maxpool1 = False
        args.first_conv = True
        normalization = stl10_normalization()
    elif args.dataset == 'imagenet':
        dm = ImagenetDataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]

        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = Transforms(
        size=args.input_height,
        input_transform=args.input_transform,
        recon_transform=args.recon_transform,
        normalize_fn=lambda x: x - 0.5,
        flip=args.flip,
        jitter_strength=args.jitter_strength,
    )
    dm.test_transforms = Transforms(
        size=args.input_height, normalize_fn=lambda x: x - 0.5
    )
    dm.val_transforms = Transforms(
        size=args.input_height, normalize_fn=lambda x: x - 0.5
    )

    # model init
    model = VAE(**args.__dict__)
    model = VAE(
        input_height=args.input_height,
        latent_dim=args.latent_dim,
        lr=args.learning_rate,
        kl_coeff=args.kl_coeff,
        prior=args.prior,
        posterior=args.posterior,
        projection=args.projection,
        encoder=args.encoder,
        decoder=args.decoder,
        first_conv=args.first_conv,
        maxpool1=args.maxpool1,
        unlabeled_batch=(args.dataset == "stl10"),
        max_epochs=args.max_epochs,
        scheduler=args.scheduler,
    )

    online_eval = SSLOnlineEvaluator(
        z_dim=model.encoder.out_dim,
        num_classes=dm.num_classes,
        drop_p=0.0,
        hidden_dim=None,
        dataset=args.dataset
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        distributed_backend='ddp' if args.gpus > 1 else None,
        sync_batchnorm=True if args.gpus > 1 else False,
        precision=32 if args.fp32 else 16,
        callbacks=[online_evaluator],
    )

    trainer.fit(model, dm)
