import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as FM

from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.models.self_supervised.simclr.simclr_transforms import (
    SimCLRTrainDataTransform,
    SimCLREvalDataTransform,
)

from resnet import resnet18_encoder, resnet18_decoder


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


def kl_divergence(mu, log_var):
    return -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp())


class VAE(pl.LightningModule):
    def __init__(
        self, kl_coeff: float, latent_dim=256, max_hidden=256, lr=1e-3, cosine=False
    ):
        super(VAE, self).__init__()
        self.save_hyperparameters()
        self.cosine = cosine
        self.kl_coeff = kl_coeff
        self.lr = lr
        self.encoder = resnet18_encoder()
        self.decoder = resnet18_decoder(latent_dim=latent_dim)

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        z = reparameterize(mu, log_var)
        return z, self.decoder(z), mu, log_var

    def step(self, batch, batch_idx):
        (x1, x2), y = batch

        z1, x1_hat, mu, log_var = self.forward(x1)
        recon = F.binary_cross_entropy(x1_hat, x1)
        kl = kl_divergence(mu, log_var)
        loss = recon + self.kl_coeff * kl
        logs = {"kl": kl, "recon": recon}

        if self.cosine:
            z2, _, _, _ = self.forward(x2)
            cosine = F.cosine_similarity(z1, z2, dim=1).mean()
            logs["cosine"] = cosine
            loss += cosine

        logs["loss"] = loss
        return loss, logs

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
    parser = argparse.ArgumentParser()
    # TODO organize args
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--cosine", action="store_true")

    args = parser.parse_args()

    dm = CIFAR10DataModule(data_dir="data", batch_size=args.batch_size, num_workers=6)
    dm.train_transforms = SimCLRTrainDataTransform(input_height=32)
    dm.test_transforms = SimCLREvalDataTransform(input_height=32)
    dm.val_transforms = SimCLREvalDataTransform(input_height=32)

    kl_coeff = args.batch_size / dm.num_samples

    model = VAE(
        latent_dim=args.latent_dim,
        lr=args.learning_rate,
        kl_coeff=kl_coeff,
        cosine=args.cosine,
    )

    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epochs)
    trainer.fit(model, dm)
