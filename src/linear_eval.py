import torch
import torch.nn as nn
import argparse
import pytorch_lightning as pl
import torchvision.transforms as transforms

from torch.optim import SGD
from torch.nn import functional as F
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import LearningRateMonitor

from src.optimizers import linear_warmup_decay
from src.models import resnet18, resnet50, resnet50w2, resnet50w4

from src.datamodules import CIFAR10DataModule, STL10DataModule
from src.datamodules import cifar10_normalization, stl10_normalization

from typing import Union, List, Optional, Sequence, Dict, Iterator, Tuple


ENCODERS = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet50w2": resnet50w2,
    "resnet50w4": resnet50w4,
}


class LinearEvaluation(pl.LightningModule):

    def __init__(
        self,
        encoder: nn.Module,
        encoder_output_dim: int,
        num_classes: int,
        num_samples: int,
        batch_size: int,
        gpus: int,
        max_epochs: int,
        learning_rate: float,
        weight_decay: float,
        nesterov: bool,
        momentum: float,
    ) -> None:

        super().__init__()
        assert isinstance(encoder, nn.Module)

        self.encoder = encoder
        self.linear_layer = nn.Linear(encoder_output_dim, num_classes)

        self.batch_size = batch_size
        self.num_samples = num_samples
        self.gpus = gpus

        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum = momentum

        global_batch_size = (
            self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        )
        self.train_iters_per_epoch = self.num_samples // global_batch_size

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

        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(0, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    def shared_step(self, batch):
        x, y = batch

        with torch.no_grad():
            feats = self.encoder(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(F.softmax(logits, dim=1), y)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log('train_acc_step', acc, on_step=True, on_epoch=False)
        self.log('train_acc_epoch', self.train_acc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, y = self.shared_step(batch)
        self.val_acc(F.softmax(logits, dim=1), y)

        self.log('val_loss', loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        loss, logits, y = self.shared_step(batch)
        self.test_acc(F.softmax(logits, dim=1), y)

        self.log('test_loss', loss, sync_dist=True, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)  # added to launch 2 ddp script on same node

    # encoder params
    parser.add_argument("--encoder_name", default="resnet50", choices=ENCODERS.keys())
    parser.add_argument('--encoder_output_dim', type=int, default=2048)
    parser.add_argument("--first_conv3x3", type=bool, default=True)  # default for cifar-10
    parser.add_argument("--remove_first_maxpool", type=bool, default=True)  # default for cifar-10

    # eval params
    parser.add_argument('--dataset', type=str, help='cifar10, stl10', default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes in classification dataset")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt')
    parser.add_argument("--data_path", type=str, default=".")

    parser.add_argument("--batch_size", default=256, type=int, help="batch size per gpu")
    parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
    parser.add_argument("--gpus", default=1, type=int, help="number of GPUs")
    parser.add_argument('--max_epochs', default=90, type=int, help="number of epochs")

    # fine-tuner params
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--momentum', type=float, default=0.9)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    args.learning_rate = 0.1 * int(args.batch_size / 256)

    # set hidden dim for resnet18
    if args.encoder_name == "resnet18":
        args.encoder_output_dim = 512

    # initialize datamodules
    train_transforms = None
    eval_transforms = None

    if args.dataset == 'cifar10':
        dm = CIFAR10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        args.num_samples = dm.num_samples
        args.input_height = dm.size()[-1]

        args.first_conv3x3 = True
        args.remove_first_maxpool = True
        normalization = cifar10_normalization()

        train_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization
        ])

        eval_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalization
        ])
    elif args.dataset == 'stl10':
        dm = STL10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        dm.train_dataloader = dm.train_dataloader_labeled
        dm.val_dataloader = dm.val_dataloader_labeled
        args.num_samples = dm.num_labeled_samples
        args.input_height = dm.size()[-1]

        args.first_conv3x3 = False  # first conv is 7x7 for stl-10
        args.remove_first_maxpool = True  # we still remove maxpool1
        normalization = stl10_normalization()

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(96),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization
        ])

        eval_transforms = transforms.Compose([
            transforms.Resize(int(args.input_height * 1.1)),
            transforms.CenterCrop(args.input_height),
            transforms.ToTensor(),
            normalization
        ])
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = train_transforms
    dm.val_transforms = eval_transforms
    dm.test_transforms = eval_transforms

    encoder = ENCODERS[args.encoder_name](first_conv3x3=args.first_conv3x3, remove_first_maxpool=args.remove_first_maxpool)

    # load encoder weights from ckpt
    device = torch.device(encoder.conv1.weight.device)
    ckpt_model = torch.load(args.ckpt_path, map_location=device)
    encoder_dict = {}

    for k in ckpt_model['state_dict'].keys():
        if 'encoder' in k:
            encoder_dict[k.replace('encoder.', '')] = ckpt_model['state_dict'][k]
    encoder.load_state_dict(encoder_dict, strict=True)

    linear_eval = LinearEvaluation(
        encoder=encoder,
        encoder_output_dim=args.encoder_output_dim,
        num_classes=args.num_classes,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
        momentum=args.momentum,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        distributed_backend="ddp" if args.gpus > 1 else None,
        precision=16,
        callbacks=[LearningRateMonitor(logging_interval="step")]
    )

    trainer.fit(linear_eval, dm)
    trainer.test(datamodule=dm)
