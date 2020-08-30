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


class FineTuner(pl.LightningModule):
    def __init__(self, backbone, in_features, lr=1e-3, num_classes=10, p=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone
        self.lr = lr

        self.mlp = nn.Sequential(nn.Dropout(p=p), nn.Linear(in_features, num_classes))

    def step(self, batch, batch_idx):
        (x1, x2), y = batch
        with torch.no_grad():
            feats = self.backbone(x1)
        logits = self.mlp(feats)
        loss = F.cross_entropy(logits, y)
        acc = FM.accuracy(logits, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch, batch_idx)
        result = pl.TrainResult(minimize=loss)
        return result

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        result = pl.EvalResult(checkpoint_on=loss)
        return result

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_epochs", type=int, default=200)

    args = parser.parse_args()

    dm = CIFAR10DataModule(batch_size=args.batch_size, num_workers=6)
    dm.train_transforms = SimCLRTrainDataTransform(input_height=32)
    dm.test_transforms = SimCLREvalDataTransform(input_height=32)
    dm.val_transforms = SimCLREvalDataTransform(input_height=32)

    trainer = pl.Trainer(gpus=1, max_epochs=args.max_epochs)
    trainer.fit(model, dm)
