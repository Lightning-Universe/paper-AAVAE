import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from pl_bolts.callbacks import LatentDimInterpolator
import numpy as np
from torch.optim import Adam
from src import utils

import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as FM
from src.transforms import MultiViewEvalTransform, MultiViewTrainTransform

from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from pl_bolts.datamodules import CIFAR10DataModule, STL10DataModule, ImagenetDataModule
from pl_bolts.transforms.dataset_normalizations import (
    cifar10_normalization,
    stl10_normalization,
    imagenet_normalization
)

from src.resnet import (
    resnet18_encoder,
    resnet18_decoder,
    resnet50_encoder,
    resnet50_decoder,
)
from src.online_eval import SSLOnlineEvaluator
from src.metrics import gini_score, KurtosisScore
from src.transforms import Transforms

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
4. normalization not used for discretized_logistic?
"""
class VAE(pl.LightningModule):
    def __init__(
        self,
        input_height,
        num_train_samples,
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
        num_mc_samples=7,
        unlabeled_batch=False,
        **kwargs
    ):
        super(VAE, self).__init__()
        self.save_hyperparameters()

        self.input_height = input_height
        self.kl_coeff = kl_coeff
        self.num_mc_samples = num_mc_samples
        self.prior = prior
        self.posterior = posterior
        self.unlabeled_batch = unlabeled_batch

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

        self.batch_size = batch_size
        self.num_train_samples = num_train_samples

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
        return self.decoder(x)

    def sample(self, z_mu, z_var):
        num_train_samples = self.num_mc_samples

        # expand dims to sample all at once
        # (batch, z_dim) -> (batch, num_train_samples, z_dim)
        z_mu = z_mu.unsqueeze(1)
        z_mu = utils.tile(z_mu, 1, num_train_samples)

        # (batch, z_dim) -> (batch, num_train_samples, z_dim)
        z_var = z_var.unsqueeze(1)
        z_var = utils.tile(z_var, 1, num_train_samples)

        std = torch.exp(z_var/2)

        p = distributions[self.prior](torch.zeros_like(z_mu), torch.ones_like(std))
        q = distributions[self.posterior](z_mu, std)

        z = q.rsample()
        return p, q, z

    def kl_divergence_mc(self, p, q, z):
        log_pz = p.log_prob(z)
        log_qz = q.log_prob(z)

        # mean over num_train_samples, sum over z_dim
        return -(log_pz - log_qz).sum(-1).mean(-1)

    def step(self, batch, batch_idx):
        if self.unlabeled_batch:
            unlabeled, labeled = batch
            batch = unlabeled

        (x1, x2, x3), y = batch

        # --------------------------
        # use x1 for KL divergence
        # --------------------------
        x1 = self.encoder(x1)
        x1_mu, x1_logvar = self.projection(x1)
        x1_P, x1_Q, x1_z = self.sample(x1_mu, x1_logvar)

        # kl
        kl = self.kl_coeff * self.kl_divergence_mc(x1_P, x1_Q, x1_z)

        # (batch, num_mc_samples) -> (batch * num_mc_samples)
        kl = kl.view(-1)

        # --------------------------
        # use x2 for reconstruction
        # --------------------------
        with torch.no_grad():
            x2 = self.encoder(x2)
            x2_mu, x2_logvar = self.projection(x2)
            x2_P, x2_Q, x2_z = self.sample(x2_mu, x2_logvar)

        # since we use MC sampling, the latent will have (b, num_mc_samples, hidden_dim)
        # convert (b, num_mc_samples, hidden_dim) -> (b * num_mc_samples, hidden_dim)
        x2_z = x2_z.view(-1, x2_z.size(-1))
        x2_hat = self.decoder(x2_z)

        # --------------------------
        # use x2_hat and x3 for log likelihood
        # --------------------------
        # since we use MC sampling, x3 also needs to be duplicated across num samples
        # (batch, channels, width, height) -> (batch * num_mc_samples, channels, width, height)
        x3 = utils.tile(x3, 0, self.num_mc_samples)
        log_pxz = gaussian_likelihood(x2_hat, self.log_scale, x3)
        
        # --------------------------
        # ELBO
        # --------------------------
        elbo = (kl - log_pxz).mean()
        
        # --------------------------
        # ADDITIONAL METRICS
        # --------------------------
        bpd = elbo / (
            self.input_height * self.input_height * self.in_channels * np.log(2.0)
        )

        gini = gini_score(x1_z)

        # TODO: this should be epoch metric
        #kurt = kurtosis_score(z)

        # n = torch.tensor(x1.size(0)).type_as(x1)
        # marg_log_px = torch.logsumexp(log_pxz + log_pz.sum(dim=-1) - log_qz.sum(dim=-1), dim=0) - torch.log(n)

        logs = {
            "kl": kl.mean(),
            "elbo": elbo,
            "gini": gini.mean(),
            #"kurtosis": kurt,
            "bpd": bpd,
            "log_pxz": log_pxz.mean(),
            # "marginal_log_px": marg_log_px.mean(),
        }

        return elbo, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, prog_bar=True)

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

    def setup(self, stage: str):
        gpus = 0 if not isinstance(self.trainer.gpus, int) else self.trainer.gpus
        global_batch_size = gpus * self.batch_size if gpus > 0 else self.batch_size
        self.train_iters_per_epoch = self.num_train_samples // global_batch_size

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
    parser = pl.Trainer.add_argparse_args(parser)

    # encoder/decoder params
    parser.add_argument("--encoder", default="resnet50", choices=encoders.keys())
    parser.add_argument("--decoder", default="resnet50", choices=decoders.keys())
    parser.add_argument('--h_dim', type=int, default=2048)
    parser.add_argument('--first_conv', type=bool, default=True)
    parser.add_argument('--maxpool1', type=bool, default=True)

    # src params
    parser.add_argument('--kl_coeff', type=float, default=1.)  # try 10., 1., 0.1, 0.01
    parser.add_argument('--latent_dim', type=int, default=128)  # try 64, 128, 256, 512
    parser.add_argument('--prior', type=str, default='normal')  # normal/laplace
    parser.add_argument('--posterior', type=str, default='normal')  # normal/laplace

    # optimizer param
    parser.add_argument('--learning_rate', type=float, default=1e-3)  # try both 1e-3/1e-4
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--num_mc_samples', type=int, default=1)
    parser.add_argument('--warmup_start_lr', type=float, default=0.)
    parser.add_argument('--eta_min', type=float, default=1e-6)

    # datamodule params
    parser.add_argument('--data_path', type=str, default='.')
    parser.add_argument('--dataset', type=str, default="cifar10")  # cifar10, stl10, imagenet
    parser.add_argument('--num_train_samples', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)

    # transforms param
    parser.add_argument('--input_height', type=int, default=32)
    parser.add_argument('--gaussian_blur', type=int, default=1)

    args = parser.parse_args()

    dm = None
    to_device = None
    if args.dataset == 'cifar10':
        dm = CIFAR10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        args.num_train_samples = dm.num_samples
        args.input_height = dm.size()[-1]

        args.maxpool1 = False
        args.first_conv = False
        normalization = cifar10_normalization()

    elif args.dataset == 'stl10':
        dm = STL10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_train_samples = dm.num_unlabeled_samples
        args.input_height = dm.size()[-1]

        args.maxpool1 = False
        args.first_conv = True
        normalization = stl10_normalization()

        def to_device(batch, device):
            unlabeled, labeled = batch
            (_, _, x), y = labeled
            x = x.to(device)
            y = y.to(device)

            return x, y
    elif args.dataset == 'imagenet':
        dm = ImagenetDataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        args.num_train_samples = dm.num_samples
        args.input_height = dm.size()[-1]

        args.maxpool1 = True
        args.first_conv = True
        normalization = imagenet_normalization()
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = MultiViewTrainTransform(
        normalization,
        gaussian_blur=args.gaussian_blur,
        num_views=3,
        input_height=args.input_height
    )
    dm.val_transforms = MultiViewEvalTransform(normalization, num_views=3, input_height=args.input_height)
    dm.test_transforms = MultiViewEvalTransform(normalization, num_views=3, input_height=args.input_height)

    # model init
    model = VAE(
        batch_size=args.batch_size,
        num_mc_samples=args.num_mc_samples,
        num_train_samples=args.num_train_samples,
        input_height=args.input_height,
        latent_dim=args.latent_dim,
        lr=args.learning_rate,
        kl_coeff=args.kl_coeff,
        prior=args.prior,
        posterior=args.posterior,
        encoder=args.encoder,
        decoder=args.decoder,
        first_conv=args.first_conv,
        maxpool1=args.maxpool1,
        unlabeled_batch=(args.dataset == "stl10"),
        max_epochs=args.max_epochs,
    )

    interpolator = LatentDimInterpolator(interpolate_epoch_interval=10)
    online_evaluator = SSLOnlineEvaluator(
        z_dim=model.encoder.out_dim,
        num_classes=dm.num_classes,
        drop_p=0.0,
        hidden_dim=None,
    )
    if to_device:
        online_evaluator.to_device = to_device

    trainer = pl.Trainer.from_argparse_args(args, callbacks=[online_evaluator, interpolator])

    trainer.fit(model, dm)
