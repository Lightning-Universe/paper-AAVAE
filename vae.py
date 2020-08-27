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

from components import Encoder, Decoder
from simclr_transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


def kl_divergence(mu, log_var):
    return -0.5 * torch.mean(1 + log_var - mu ** 2 - log_var.exp())


class VAE(pl.LightningModule):
    def __init__(
        self,
        kl_coeff: float,
        latent_dim=128,
        max_hidden=256,
        lr=1e-3,
        finetune=False,
        num_classes=10,
    ):
        super(VAE, self).__init__()
        self.save_hyperparameters()
        self.finetune = finetune
        self.kl_coeff = kl_coeff
        self.lr = lr
        self.encode = Encoder(
            latent_dim, max_channels=max_hidden, num_classes=num_classes
        )
        self.decode = Decoder(latent_dim, max_channels=max_hidden)

    def forward(self, x):
        mu, log_var, _ = self.encode(x)
        z = reparameterize(mu, log_var)
        return z, self.decode(z), mu, log_var

    def step(self, batch, batch_idx):
        (x1, x2), y = batch

        if self.finetune:
            _, _, logits = self.encode(x1)
            loss = F.cross_entropy(logits, y)
            acc = FM.accuracy(logits, y)
            return loss, {"ce_loss": loss, "acc": acc}

        z1, x1_hat, mu, log_var = self.forward(x1)
        recon = F.binary_cross_entropy(x1_hat, x1)
        kl = kl_divergence(mu, log_var)
        loss = recon + self.kl_coeff * kl

        # TODO: experiment with cosine similarity
        # something like
        # if self.cosine:
        #     z2, _, _, _ = self.forward(x2)
        #     loss += F.cosine_similarity(z1, z2, dim=1)

        return loss, {"loss": loss, "kl": kl, "recon": recon}

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
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--finetune", action="store_true")

    args = parser.parse_args()

    dm = CIFAR10DataModule(batch_size=args.batch_size, num_workers=6)
    dm.train_transforms = SimCLRTrainDataTransform(input_height=32)
    dm.test_transforms = SimCLREvalDataTransform(input_height=32)
    dm.val_transforms = SimCLREvalDataTransform(input_height=32)

    kl_coeff = args.batch_size / dm.num_samples

    if args.pretrained is not None:
        model = VAE.load_from_checkpoint(
            args.pretrained, lr=args.learning_rate, finetune=args.finetune
        )
    model = VAE(lr=args.learning_rate, kl_coeff=kl_coeff, finetune=args.finetune)

    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epochs)
    trainer.fit(model, dm)
