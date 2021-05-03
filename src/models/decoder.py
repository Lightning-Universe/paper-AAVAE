import torch
import torch.nn as nn

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


class Decoder(nn.Module):

    def __init__(
        self,
        latent_dim: int = 128,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        widen: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        first_conv3x3: bool = False,
        remove_first_maxpool: bool = False,
    ) -> None:

        super(Decoder, self).__init__()

        self.linear_projection = nn.Linear(latent_dim, )

    def forward(self, x: Tensor) -> Tensor:
        # latent-dim -> channel count

        return x


def decoder18(**kwargs):
    return Decoder(BasicBlock, [2, 2, 2, 2], **kwargs)


def decoder50(**kwargs):
    return Decoder(Bottleneck, [3, 4, 6, 3], **kwargs)


def decoder50w2(**kwargs):
    return Decoder(Bottleneck, [3, 4, 6, 3], widen=2, **kwargs)


def decoder50w4(**kwargs):
    return Decoder(Bottleneck, [3, 4, 6, 3], widen=4, **kwargs)
