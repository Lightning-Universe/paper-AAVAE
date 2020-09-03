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
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from resnet import resnet18_encoder, resnet18_decoder

distributions = {
    "laplace": torch.distributions.Laplace,
    "normal": torch.distributions.Normal,
}


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
        x, y = batch

        z, x_hat, p, q = self.forward(x)

        # reconstruction
        recon = F.mse_loss(x_hat, x)

        # KL divergence
        kl = torch.sum(q.log_prob(z) - p.log_prob(z))
        kl /= torch.numel(x)  # normalize kl by number of elements in reconstruction

        loss = recon + self.kl_coeff * kl

        logs = {"kl": kl, "recon": recon, "loss": loss}
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
    # TODO: model specific args and stuff

    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--kl_coeff", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--prior", default="normal")
    parser.add_argument("--posterior", default="normal")

    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--gpus", default="1")

    args = parser.parse_args()

    dm = CIFAR10DataModule(data_dir="data", batch_size=args.batch_size, num_workers=6)
    dm.train_transforms = T.Compose(
        [
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            cifar10_normalization(),
        ]
    )
    dm.test_transforms = T.Compose([T.ToTensor(), cifar10_normalization()])
    dm.val_transforms = T.Compose([T.ToTensor(), cifar10_normalization()])

    model = VAE(
        latent_dim=args.latent_dim,
        lr=args.learning_rate,
        kl_coeff=args.kl_coeff,
        prior=args.prior,
        posterior=args.posterior,
    )

    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.max_epochs)
    trainer.fit(model, dm)
