import torch
import torch.nn as nn

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


def decoder18(**kwargs):
    return Decoder(BasicBlock, [2, 2, 2, 2], **kwargs)


def decoder50(**kwargs):
    return Decoder(Bottleneck, [3, 4, 6, 3], **kwargs)


def decoder50w2(**kwargs):
    return Decoder(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


def decoder50w4(**kwargs):
    return Decoder(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)
