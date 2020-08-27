import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

import pytorch_lightning as pl


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def vae_loss(x, x_hat, mu, log_var):
    recon_loss = F.mse_loss(x_hat, x)
    kld_loss = torch.mean(
        -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0
    )
    return recon_loss + kld_loss


class VAE(pl.LightningModule):
    def __init__(self, latent_dim=128, hidden_dims=[32, 64], lr=1e-3):

        super().__init__()
        self.save_hyperparameters()

        self.lr = lr

        # construct encoder
        encoder = []
        in_channels = 3
        for h_dim in hidden_dims:
            encoder.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
        self.encoder = nn.Sequential(*encoder)

        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, latent_dim)

        self.decoder_in = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        # construct decoder
        decoder = []
        for i in range(len(hidden_dims) - 1):
            decoder.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        def encode(self, x):
            x = self.encoder(x)
            x = x.flatten(x, start_dim=1)

            mu = self.fc_mu(x)
            log_var = self.fc_var(x)
            return mu, log_var

        def decode(self, z):
            x_hat = self.decoder_in(z)
            x_hat = x_hat.view(-1, 64, 2, 2)
            x_hat = self.decoder(x_hat)
            return self.final_layer(x_hat)

        def forward(self, x):
            mu, log_var = self.encode(x)
            z = reparameterize(mu, log_var)
            return self.decode(z), mu, log_var

        def training_step(self, batch, batch_idx):
            x, _ = batch
            x_hat, mu, log_var = self.forward(x)
            loss = vae_loss(x, x_hat, mu, log_var)
            return pl.TrainResult(loss)

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    model = VAE()
    transform = T.Compose(
        [
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )

    dataset = CIFAR10("data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=8)
    trainer = pl.Trainer(gpus=1, max_epochs=20)
    trainer.fit(model, dataloader)
