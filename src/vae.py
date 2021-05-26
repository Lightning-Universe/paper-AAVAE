import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np

from torch.optim import Adam
from typing import Union, List, Optional, Sequence, Dict, Iterator, Tuple, Callable

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import ProjectionHeadVAE
from src.models import resnet18, resnet50, resnet50w2, resnet50w4
from src.models import decoder18, decoder50, decoder50w2, decoder50w4

from src.optimizers import LAMB, linear_warmup_decay
from src.transforms import TrainTransform, EvalTransform
from src.callbacks import OnlineFineTuner, EarlyStopping
from src.datamodules import CIFAR10DataModule, STL10DataModule
from src.datamodules import cifar10_normalization, stl10_normalization


ENCODERS = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet50w2": resnet50w2,
    "resnet50w4": resnet50w4,
}
DECODERS = {
    "resnet18": decoder18,
    "resnet50": decoder50,
    "resnet50w2": decoder50w2,
    "resnet50w4": decoder50w4,
}


class VAE(pl.LightningModule):
    def __init__(
        self,
        input_height,
        num_samples,
        gpus,
        batch_size,
        kl_coeff,
        h_dim,
        latent_dim,
        optimizer,
        learning_rate,
        encoder_name,
        first_conv3x3,
        remove_first_maxpool,
        dataset,
        max_epochs,
        warmup_epochs,
        cosine_decay,
        linear_decay,
        learn_scale,
        log_scale,
        val_samples,
        weight_decay,
        exclude_bn_bias,
        online_ft,
        **kwargs,
    ) -> None:
        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.input_height = input_height
        self.num_samples = num_samples
        self.dataset = dataset
        self.batch_size = batch_size

        self.gpus = gpus
        self.online_ft = online_ft

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exclude_bn_bias = exclude_bn_bias
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.cosine_decay = cosine_decay
        self.linear_decay = linear_decay

        self.encoder_name = encoder_name
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.first_conv3x3 = first_conv3x3
        self.remove_first_maxpool = remove_first_maxpool

        self.kl_coeff = kl_coeff
        self.learn_scale = learn_scale
        self.log_scale = log_scale
        self.val_samples = val_samples

        global_batch_size = (
            self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        )
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        self.encoder = ENCODERS[self.encoder_name](
            first_conv3x3=self.first_conv3x3,
            remove_first_maxpool=self.remove_first_maxpool,
        )
        self.decoder = DECODERS[self.encoder_name](
            input_height=self.input_height,
            latent_dim=self.latent_dim,
            h_dim=self.h_dim,
            first_conv3x3=self.first_conv3x3,
            remove_first_maxpool=self.remove_first_maxpool,
        )

        self.projection = ProjectionHeadVAE(
            input_dim=self.h_dim, hidden_dim=self.h_dim, output_dim=self.latent_dim
        )

        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

        # start log-scale with a specific value
        self.log_scale = nn.Parameter(torch.Tensor([self.log_scale]))
        self.log_scale.requires_grad = bool(self.learn_scale)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, x):
        return self.encoder(x)

    def sample(self, z_mu, z_var, eps=1e-6):
        """
        z_mu and z_var is (batch, dim)
        """
        # add eps to prevent 0 variance
        std = torch.exp(z_var / 2.) + eps

        p = torch.distributions.Normal(torch.zeros_like(z_mu), torch.ones_like(std))
        q = torch.distributions.Normal(z_mu, std)
        z = q.rsample()

        return p, q, z

    @staticmethod
    def kl_divergence_mc(p, q, z):
        """
        z is (batch, dim)
        """
        log_pz = p.log_prob(z)
        log_qz = q.log_prob(z)

        kl = (log_qz - log_pz).sum(dim=-1)
        log_pz = log_pz.sum(dim=-1)
        log_qz = log_qz.sum(dim=-1)

        return kl, log_pz, log_qz

    @staticmethod
    def kl_divergence_analytic(p, q, z):
        """
        z is (batch, dim)
        """
        kl = torch.distributions.kl.kl_divergence(q, p).sum(dim=-1)
        log_pz = p.log_prob(z).sum(dim=-1)
        log_qz = q.log_prob(z).sum(dim=-1)

        return kl, log_pz, log_qz

    @staticmethod
    def gaussian_likelihood(mean, logscale, sample, eps=1e-6):
        scale = torch.exp(logscale / 2.) + eps
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(sample)

        # sum over dimensions
        return log_pxz.sum(dim=(1, 2, 3))

    def step(self, batch, samples=1):
        if self.dataset == "stl10":
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        if self.online_ft:
            (x, original, _), y = batch
        else:
            (x, original), y = batch

        batch_size, c, h, w = x.shape
        pixels = c * h * w

        # get representation of original image
        with torch.no_grad():
            x_original = self.encoder(original).clone().detach()
            mu_orig, log_var_orig = self.projection(x_original)

        x_enc = self.encoder(x)
        mu, log_var = self.projection(x_enc)

        log_pzs = []
        log_qzs = []
        log_pxzs = []

        kls = []
        elbos = []
        losses = []
        cos_sims = []
        kl_augmentations = []

        for _ in range(samples):
            p, q, z = self.sample(mu, log_var)
            kl, log_pz, log_qz = self.kl_divergence_analytic(p, q, z)

            with torch.no_grad():
                _, q_orig, z_orig = self.sample(mu_orig, log_var_orig)

            # kl between original image and augmented image
            kl_aug = torch.distributions.kl.kl_divergence(q, q_orig).sum(dim=-1)
            kl_augmentations.append(kl_aug)

            cos_sims.append(self.cosine_similarity(z_orig, z))

            x_hat = self.decoder(z)
            log_pxz = self.gaussian_likelihood(x_hat, self.log_scale, original)

            # plot reconstructions
            img_grid = torchvision.utils.make_grid(x_hat)

            elbo = kl - log_pxz
            loss = self.kl_coeff * kl - log_pxz

            log_pzs.append(log_pz)
            log_qzs.append(log_qz)
            log_pxzs.append(log_pxz)

            kls.append(kl)
            elbos.append(elbo)
            losses.append(loss)

        # all of these will be of shape [batch, samples, ... ]
        log_pz = torch.stack(log_pzs, dim=1)
        log_qz = torch.stack(log_qzs, dim=1)
        log_pxz = torch.stack(log_pxzs, dim=1)

        kl = torch.stack(kls, dim=1)
        elbo = torch.stack(elbos, dim=1).mean()
        loss = torch.stack(losses, dim=1).mean()

        cos_sim = torch.stack(cos_sims, dim=1).mean()
        kl_augmentation = torch.stack(kl_augmentations, dim=1).mean()

        # marginal likelihood, logsumexp over sample dim, mean over batch dim
        log_px = torch.logsumexp(log_pxz + log_pz - log_qz, dim=1).mean(dim=0) - np.log(
            samples
        )
        bpd = -log_px / (pixels * np.log(2))  # need log_px in base 2

        logs = {
            "kl": kl.mean(),
            "elbo": elbo,
            "loss": loss,
            "bpd": bpd,
            "cos_sim": cos_sim,
            "kl_augmentation": kl_augmentation,
            "log_pxz": log_pxz.mean(),
            "log_pz": log_pz.mean(),
            "log_px": log_px,
            "log_scale": self.log_scale.item(),
        }

        return loss, logs, img_grid

    def training_step(self, batch, batch_idx):
        loss, logs, img_grid = self.step(batch, samples=1)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})

        if self.global_step % 1000 == 0:
            self.logger.experiment.add_image('train_reconstructions', img_grid, global_step=self.global_step)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs, img_grid = self.step(batch, samples=self.val_samples)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})

        if self.global_step % 1000 == 0:
            self.logger.experiment.add_image('val_reconstructions', img_grid, global_step=self.global_step)

        return loss

    def exclude_from_wt_decay_and_layer_adaptation(
        self,
        named_params: Iterator[Tuple[str, torch.Tensor]],
        weight_decay: float,
        skip_list: List[str] = ['bias', 'bn'],
    ) -> List[Dict]:
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{'params': params, 'weight_decay': weight_decay, 'exclude_from_layer_adaptation': False},
                {'params': excluded_params, 'weight_decay': 0., 'exclude_from_layer_adaptation': True}]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optimizer == 'adam':
            optimizer = Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'lamb':
            optimizer = LAMB(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)  # added to launch 2 ddp script on same node

    # ae params
    parser.add_argument("--denoising", action="store_true")
    parser.add_argument("--encoder_name", default="resnet50", choices=ENCODERS.keys())
    parser.add_argument("--h_dim", type=int, default=2048)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--first_conv3x3", type=bool, default=True)  # default for cifar-10
    parser.add_argument("--remove_first_maxpool", type=bool, default=True)  # default for cifar-10

    # vae params
    parser.add_argument('--kl_coeff', type=float, default=0.1)
    parser.add_argument("--log_scale", type=float, default=0.)
    parser.add_argument("--learn_scale", type=int, default=0)  # default keep fixed log-scale
    parser.add_argument("--val_samples", type=int, default=1)

    # optimizer param
    parser.add_argument("--optimizer", type=str, default="adam")  # adam/lamb
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--exclude_bn_bias", action="store_true")

    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--max_epochs", type=int, default=3200)
    parser.add_argument("--cosine_decay", type=int, default=0)
    parser.add_argument("--linear_decay", type=int, default=0)

    # training params
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--online_ft", action="store_true")

    # datamodule params
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="cifar10")  # cifar10, stl10
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # transforms param
    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument("--gaussian_blur", type=bool, default=True)
    parser.add_argument("--jitter_strength", type=float, default=1.0)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # set hidden dim for resnets
    if args.encoder_name == "resnet18":
        args.h_dim = 512
    elif args.encoder_name == "resnet50w2":
        args.h_dim = 4096
    elif args.encoder_name == "resnet50w4":
        args.h_dim = 8192

    if args.dataset == "cifar10":
        dm = CIFAR10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]

        args.first_conv3x3 = True
        args.remove_first_maxpool = True
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

        args.first_conv3x3 = False  # first conv is 7x7 for stl-10
        args.remove_first_maxpool = True  # we still remove maxpool1
        normalization = stl10_normalization()
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = TrainTransform(
        denoising=args.denoising,
        input_height=args.input_height,
        dataset=args.dataset,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
        online_ft=args.online_ft,
    )

    dm.val_transforms = EvalTransform(
        denoising=args.denoising,
        input_height=args.input_height,
        dataset=args.dataset,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
        online_ft=args.online_ft,
    )

    # model init
    model = VAE(**args.__dict__)

    # TODO: add early stopping
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(save_last=True, save_top_k=3, monitor='val_elbo'),
        # ModelCheckpoint(every_n_val_epochs=20, save_top_k=-1),
    ]

    if args.online_ft:
        online_finetuner = OnlineFineTuner(
            encoder_output_dim=args.h_dim,
            num_classes=dm.num_classes,
            dataset=args.dataset
        )
        callbacks.append(online_finetuner)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        distributed_backend="ddp" if args.gpus > 1 else None,
        precision=16 if args.fp16 else 32,
        callbacks=callbacks,
    )

    trainer.fit(model, dm)
