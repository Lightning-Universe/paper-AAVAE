import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

import pytorch_lightning as pl
from pl_bolts.datamodules import CIFAR10DataModule

from components import Encoder, Decoder


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


# def kl_divergence(mu, log_var): return torch.mean(
#        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
#    )


def kl_divergence(mu, log_var):
    return -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp())


class VAE(pl.LightningModule):
    def __init__(self, kl_coeff: float, latent_dim=128, max_hidden=128, lr=1e-3):
        super(VAE, self).__init__()
        self.save_hyperparameters()
        self.kl_coeff = kl_coeff
        self.lr = lr
        self.encode = Encoder(latent_dim, max_channels=max_hidden)
        self.decode = Decoder(latent_dim, max_channels=max_hidden)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = reparameterize(mu, log_var)
        return z, self.decode(z), mu, log_var

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z, x_hat, mu, log_var = self.forward(x)

        reconst_loss = F.binary_cross_entropy(x_hat, x)
        kl_loss = kl_divergence(mu, log_var)

        loss = reconst_loss + self.kl_coeff * kl_loss

        result = pl.TrainResult(minimize=loss)
        result.log("loss", loss)
        result.log("kl", kl_loss)
        result.log("reconst", reconst_loss)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    batch_size = 64
    learning_rate = 3e-4
    max_epochs = 20

    transform = T.Compose(
        [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()]
    )

    dataset = CIFAR10("data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)

    kl_coeff = batch_size / len(dataset)  # important

    model = VAE(lr=learning_rate, kl_coeff=kl_coeff)
    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs)
    trainer.fit(model, dataloader)
