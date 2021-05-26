import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np

from torch.optim import Adam
from typing import Union, List, Optional, Sequence, Dict, Iterator, Tuple, Callable

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import ProjectionHeadAE
from src.models import resnet18, resnet50, resnet50w2, resnet50w4
from src.models import decoder18, decoder50, decoder50w2, decoder50w4

from src.optimizers import LAMB, linear_warmup_decay
from src.transforms import TrainTransform, EvalTransform
from src.callbacks import OnlineFineTuner, EarlyStopping
from src.datamodules import CIFAR10DataModule, STL10DataModule
from src.datamodules import cifar10_normalization, stl10_normalization


ENCODERS = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet50w2": resnet50w2,
    "resnet50w4": resnet50w4,
}
DECODERS = {
    "resnet18": decoder18,
    "resnet50": decoder50,
    "resnet50w2": decoder50w2,
    "resnet50w4": decoder50w4,
}


class AE(pl.LightningModule):
    def __init__(
        self,
        input_height,
        num_samples,
        gpus,
        batch_size,
        h_dim,
        latent_dim,
        optimizer,
        learning_rate,
        encoder_name,
        first_conv3x3,
        remove_first_maxpool,
        dataset,
        max_epochs,
        warmup_epochs,
        cosine_decay,
        linear_decay,
        weight_decay,
        exclude_bn_bias,
        online_ft,
        **kwargs,
    ) -> None:
        super(AE, self).__init__()

        self.save_hyperparameters()

        self.input_height = input_height
        self.num_samples = num_samples
        self.dataset = dataset
        self.batch_size = batch_size

        self.gpus = gpus
        self.online_ft = online_ft

        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.exclude_bn_bias = exclude_bn_bias
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.cosine_decay = cosine_decay
        self.linear_decay = linear_decay

        self.encoder_name = encoder_name
        self.h_dim = h_dim
        self.latent_dim = latent_dim
        self.first_conv3x3 = first_conv3x3
        self.remove_first_maxpool = remove_first_maxpool

        global_batch_size = (
            self.gpus * self.batch_size if self.gpus > 0 else self.batch_size
        )
        self.train_iters_per_epoch = self.num_samples // global_batch_size

        self.encoder = ENCODERS[self.encoder_name](
            first_conv3x3=self.first_conv3x3,
            remove_first_maxpool=self.remove_first_maxpool,
        )
        self.decoder = DECODERS[self.encoder_name](
            input_height=self.input_height,
            latent_dim=self.latent_dim,
            h_dim=self.h_dim,
            first_conv3x3=self.first_conv3x3,
            remove_first_maxpool=self.remove_first_maxpool,
        )

        self.projection = ProjectionHeadAE(
            input_dim=self.h_dim, hidden_dim=self.h_dim, output_dim=self.latent_dim
        )

        self.cosine_similarity = nn.CosineSimilarity(dim=1, eps=1e-6)

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def forward(self, x):
        return self.encoder(x)

    def step(self, batch):
        if self.dataset == "stl10":
            unlabeled_batch = batch[0]
            batch = unlabeled_batch

        if self.online_ft:
            (x, original, _), y = batch
        else:
            (x, original), y = batch

        # get representations of original image
        with torch.no_grad():
            h_original = self.encoder(original).clone().detach()
            z_original = self.projection(h_original)

        # TODO: add val reconstructions here
        h = self.encoder(x)
        z = self.projection(h)
        x_hat = self.decoder(z)

        loss = F.mse_loss(x_hat, x, reduction='mean')
        cos_sim = self.cosine_similarity(z_original, z).mean()

        logs = {
            "mse": loss,
            "cos_sim": cos_sim,
        }

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})

        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})

        return loss

    def exclude_from_wt_decay_and_layer_adaptation(
        self,
        named_params: Iterator[Tuple[str, torch.Tensor]],
        weight_decay: float,
        skip_list: List[str] = ['bias', 'bn'],
    ) -> List[Dict]:
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [{'params': params, 'weight_decay': weight_decay, 'exclude_from_layer_adaptation': False},
                {'params': excluded_params, 'weight_decay': 0., 'exclude_from_layer_adaptation': True}]

    def configure_optimizers(self):
        if self.exclude_bn_bias:
            params = self.exclude_from_wt_decay(self.named_parameters(), weight_decay=self.weight_decay)
        else:
            params = self.parameters()

        if self.optimizer == 'adam':
            optimizer = Adam(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'lamb':
            optimizer = LAMB(params, lr=self.learning_rate, weight_decay=self.weight_decay)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(
                    warmup_steps, total_steps, self.cosine_decay, self.linear_decay
                ),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=0)  # added to launch 2 ddp script on same node

    # ae params
    parser.add_argument("--denoising", action="store_true")
    parser.add_argument("--encoder_name", default="resnet50", choices=ENCODERS.keys())
    parser.add_argument("--h_dim", type=int, default=2048)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--first_conv3x3", type=bool, default=True)  # default for cifar-10
    parser.add_argument("--remove_first_maxpool", type=bool, default=True)  # default for cifar-10

    # optimizer param
    parser.add_argument("--optimizer", type=str, default="adam")  # adam/lamb
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--exclude_bn_bias", action="store_true")

    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=3200)
    parser.add_argument("--cosine_decay", type=int, default=0)
    parser.add_argument("--linear_decay", type=int, default=0)

    # training params
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--online_ft", action="store_true")

    # datamodule params
    parser.add_argument("--data_path", type=str, default=".")
    parser.add_argument("--dataset", type=str, default="cifar10")  # cifar10, stl10
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # transforms param
    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument("--gaussian_blur", type=bool, default=True)
    parser.add_argument("--jitter_strength", type=float, default=1.0)

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    # set hidden dim for resnets
    if args.encoder_name == "resnet18":
        args.h_dim = 512
    elif args.encoder_name == "resnet50w2":
        args.h_dim = 4096
    elif args.encoder_name == "resnet50w4":
        args.h_dim = 8192

    if args.dataset == "cifar10":
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

        args.gaussian_blur = False
        args.jitter_strength = 0.5
    elif args.dataset == "stl10":
        dm = STL10DataModule(
            data_dir=args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        dm.train_dataloader = dm.train_dataloader_mixed
        dm.val_dataloader = dm.val_dataloader_mixed
        args.num_samples = dm.num_unlabeled_samples
        args.input_height = dm.size()[-1]

        args.first_conv3x3 = False  # first conv is 7x7 for stl-10
        args.remove_first_maxpool = True  # we still remove maxpool1
        normalization = stl10_normalization()
    else:
        raise NotImplementedError("other datasets have not been implemented till now")

    dm.train_transforms = TrainTransform(
        denoising=args.denoising,
        input_height=args.input_height,
        dataset=args.dataset,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
        online_ft=args.online_ft,
    )

    dm.val_transforms = EvalTransform(
        denoising=args.denoising,
        input_height=args.input_height,
        dataset=args.dataset,
        gaussian_blur=args.gaussian_blur,
        jitter_strength=args.jitter_strength,
        normalize=normalization,
        online_ft=args.online_ft,
    )

   # model init
    model = AE(**args.__dict__)

    # TODO: add early stopping
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(save_last=True, save_top_k=3, monitor='val_cos_sim', mode='max'),
        # ModelCheckpoint(every_n_val_epochs=20, save_top_k=-1),
    ]

    if args.online_ft:
        online_finetuner = OnlineFineTuner(
            encoder_output_dim=args.h_dim,
            num_classes=dm.num_classes,
            dataset=args.dataset
        )
        callbacks.append(online_finetuner)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        gpus=args.gpus,
        distributed_backend="ddp" if args.gpus > 1 else None,
        precision=16 if args.fp16 else 32,
        callbacks=callbacks,
    )

    trainer.fit(model, dm)
