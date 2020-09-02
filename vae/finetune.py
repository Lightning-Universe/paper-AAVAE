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

from vae import VAE


class FineTuner(pl.LightningModule):
    def __init__(self, backbone, in_features=512, lr=1e-3, num_classes=10, p=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.lr = lr

        self.mlp = nn.Sequential(nn.Dropout(p=p), nn.Linear(in_features, num_classes))

    def on_train_epoch_start(self):
        self.backbone.eval()

    def step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            feats = self.backbone(x)
        logits = self.mlp(feats)
        loss = F.cross_entropy(logits, y)
        acc = FM.accuracy(logits, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
        result = pl.TrainResult(minimize=loss)
        result.log("train_acc", acc)
        return result

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({"valid_acc": acc, "valid_loss": loss})
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pretrained")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--gpus", default="1")
    args = parser.parse_args()

    vae = VAE.load_from_checkpoint(args.pretrained)
    model = FineTuner(backbone=vae.encoder)

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

    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.max_epochs)
    trainer.fit(model, dm)
