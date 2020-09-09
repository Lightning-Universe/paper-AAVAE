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

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from resnet import resnet18_encoder, resnet18_decoder
from online_eval import SSLOnlineEvaluator
from metrics import gini_score
from transforms import Transforms

distributions = {
    "laplace": torch.distributions.Laplace,
    "normal": torch.distributions.Normal,
}


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


class VAE(pl.LightningModule):
    def __init__(
        self, kl_coeff=0.1, latent_dim=256, lr=1e-4, prior="normal", posterior="normal"
    ):
        super(VAE, self).__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.kl_coeff = kl_coeff
        self.encoder = resnet18_encoder()
        self.decoder = resnet18_decoder(latent_dim=latent_dim)
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)
        self.prior = prior
        self.posterior = posterior

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        p, q, z = self.sample(mu, log_var)
        return z, self.decoder(z), p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = distributions[self.prior](torch.zeros_like(mu), torch.ones_like(std))
        q = distributions[self.posterior](mu, std)
        z = q.rsample()
        return p, q, z

    def step(self, batch, batch_idx):
        (x1, x2, _), y = batch

        z, x1_hat, p, q = self.forward(x1)

        log_pxz = discretized_logistic(x1_hat, self.log_scale, x2)
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.sum(dim=(1))  # sum all dims except batch

        elbo = (kl - log_pxz).mean()
        bpd = elbo / (32 * 32 * 3 * np.log(2.0))

        gini = gini_score(z)

        # marginal log p(x) using importance sampling
        # TODO: is this N batch size or number of elements (e.g. 3 * 32 * 32 for CIFAR)
        n = torch.tensor(x1.size(0)).type_as(x1)
        marg_log_px = torch.logsumexp(log_pxz + log_pz - log_qz, dim=0) - torch.log(n)

        logs = {
            "kl": kl.mean(),
            "elbo": elbo,
            "gini": gini.mean(),
            "bpd": bpd,
            "log_pxz": log_pxz.mean(),
            "marginal_log_px": marg_log_px.mean(),
        }
        return elbo, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        result = pl.TrainResult(minimize=loss)
        result.log_dict(
            {f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False
        )
        return result

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({f"val_{k}": v for k, v in logs.items()})
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    # TODO: model specific args and stuff

    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--kl_coeff", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--prior", default="normal")
    parser.add_argument("--posterior", default="normal")

    parser.add_argument("--batch_size", type=int, default=256)

    tf_choices = ["orginal", "global", "local"]
    parser.add_argument("--input_transform", default="original", choices=tf_choices)
    parser.add_argument("--recon_transform", default="original", choices=tf_choices)

    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--gpus", default="1")

    args = parser.parse_args()

    dm = CIFAR10DataModule(data_dir="data", batch_size=args.batch_size, num_workers=6)
    dm.train_transforms = Transforms(
        input_transform=args.input_transform,
        recon_transform=args.recon_transform,
        normalize_fn=lambda x: x - 0.5,
    )
    dm.test_transforms = Transforms(normalize_fn=lambda x: x - 0.5)
    dm.val_transforms = Transforms(normalize_fn=lambda x: x - 0.5)

    model = VAE(
        latent_dim=args.latent_dim,
        lr=args.learning_rate,
        kl_coeff=args.kl_coeff,
        prior=args.prior,
        posterior=args.posterior,
    )

    online_eval = SSLOnlineEvaluator(z_dim=512, num_classes=dm.num_classes, drop_p=0.0)

    trainer = pl.Trainer(
        gpus=args.gpus, max_epochs=args.max_epochs, callbacks=[online_eval]
    )
    trainer.fit(model, dm)
