import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from typing import Union, List, Optional, Sequence, Dict, Iterator, Tuple, Callable

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import ProjectionEncoder
from src.models import resnet18, resnet50, resnet50w2, resnet50w4
from src.models import decoder18, decoder50, decoder50w2, decoder50w4

from src.callbacks import OnlineFineTuner, EarlyStopping
from src.datamodules import CIFAR10DataModule, STL10DataModule
from src.datamodules import cifar10_normalization, stl10_normalization


encoders = {
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet50w2": resnet50w2,
    "resnet50w4": resnet50w4,
}
decoders = {
    "resnet18": decoder18,
    "resnet50": decoder50,
    "resnet50w2": decoder50w2,
    "resnet50w4": decoder50w4,
}

# TODO: add non-agumented baseline
# TODO: first conv and maxpool options flipped bool value
class VAE(pl.LightningModule):
    def __init__(self) -> None:
        super(VAE, self).__init__()
