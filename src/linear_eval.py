import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.optim import SGD
from torch.nn import functional as F
from pytorch_lightning.metrics import Accuracy

from typing import Union, List, Optional, Sequence, Dict, Iterator, Tuple


# TODO: check logging and metrics
# TODO: check if acc needs softmax over logits or not
class LinearEvaluation(pl.LightningModule):

    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_dim: int,
        num_classes: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        nesterov: bool,
        momentum: float = 0.9,
    ) -> None:

        super().__init__()
        assert isinstance(encoder, nn.Module)

        self.encoder = encoder
        self.linear_layer = nn.Linear(encoder_output_dim, num_classes)

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum = momentum

        # metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy(compute_on_step=False)
        self.test_acc = Accuracy(compute_on_step=False)

    def on_train_epoch_start(self) -> None:
        self.encoder.eval()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.linear_layer.parameters(),
            lr=self.learning_rate,
            nesterov=self.nesterov,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

        # TODO: check decay with epoch vs per-step (simclr vs swav)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            self.epochs,
        )

        return [optimizer], [scheduler]

    def shared_step(self, x: torch.Tensor):
        with torch.no_grad():
            feats = self.encoder(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        loss, logits, y = self.shared_step(x)
        acc = self.train_acc(F.softmax(logits, dim=1), y)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_acc_step', acc, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_acc_epoch', self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, x: torch.Tensor) -> torch.Tensor:
        loss, logits, y = self.shared_step(x)
        self.val_acc(F.softmax(logits, dim=1), y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

        return loss

    def test_step(self, x: torch.Tensor) -> torch.Tensor:
        loss, logits, y = self.shared_step(x)
        self.test_acc(F.softmax(logits, dim=1), y)

        self.log('test_loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.val_acc, on_step=False, on_epoch=True)

        return loss
