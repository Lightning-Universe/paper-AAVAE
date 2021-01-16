import torch
import math
import pytorch_lightning as pl
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.metrics import Accuracy
from typing import Optional


class SSLOnlineEvaluator(pl.Callback):
    def __init__(
        self,
        dataset: str,
        drop_p: float = 0.2,
        hidden_dim: Optional[int] = None,
        z_dim: int = None,
        num_classes: int = None,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.drop_p = drop_p
        self.optimizer = None

        self.z_dim = z_dim
        self.num_classes = num_classes
        self.dataset = dataset

    def on_pretrain_routine_start(self, trainer, pl_module):
        from pl_bolts.models.self_supervised.evaluator import SSLEvaluator

        pl_module.non_linear_evaluator = SSLEvaluator(
            n_input=self.z_dim,
            n_classes=self.num_classes,
            p=self.drop_p,
            n_hidden=self.hidden_dim,
        ).to(pl_module.device)

        self.valid_acc = Accuracy().to(pl_module.device)

        self.optimizer = torch.optim.Adam(
            pl_module.non_linear_evaluator.parameters(), lr=1e-4
        )

    def get_representations(self, pl_module, x):
        """
        Override this to customize for the particular model
        Args:
            pl_module:
            x:
        """
        representations = pl_module(x)
        representations = representations.reshape(representations.size(0), -1)
        return representations

    def to_device(self, batch, device):
        # get the labeled batch
        if self.dataset == 'stl10':
            labeled_batch = batch[1]
            batch = labeled_batch

        inputs, y = batch

        # last input is for online eval
        x = inputs[-1]
        x = x.to(device)
        y = y.to(device)

        return x, y

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module.encoder, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = F.cross_entropy(mlp_preds, y)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # log metrics
        acc = accuracy(mlp_preds, y)
        pl_module.log('online_train_acc', acc, on_step=True, on_epoch=False)
        pl_module.log('online_train_loss', mlp_loss, on_step=True, on_epoch=False)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        x, y = self.to_device(batch, pl_module.device)

        with torch.no_grad():
            representations = self.get_representations(pl_module.encoder, x)

        representations = representations.detach()

        # forward pass
        mlp_preds = pl_module.non_linear_evaluator(representations)
        mlp_loss = F.cross_entropy(mlp_preds, y)

        self.valid_acc(mlp_preds, y)

        # log loss
        pl_module.log('online_val_loss', mlp_loss, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        # compute accuracy, synced over nodes and batches
        val_acc = self.valid_acc.compute()
        pl_module.log('online_val_acc', val_acc)
        self.valid_acc.reset()
