import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


class Interpolate(nn.Module):

    def __init__(self, upscale: str = 'scale', size: Optional[int] = None):
        super().__init__()
        self.upscale = upscale
        self.size = size

        if self.upscale == 'size':
            assert self.size is not None

    def forward(self, x):
        if self.upscale == 'scale':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        elif self.upscale == 'size':
            return F.interpolate(x, size=(self.size, self.size), mode='nearest')


def conv3x3(in_planes: int, out_planes: int, groups: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, padding=1, groups=groups, bias=True
    )


def conv1x1(in_planes: int, out_planes: int) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        upscale: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.conv1 = conv3x3(inplanes, planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.upsample = upsample
        self.upscale = upscale

    def forward(self, x) -> Tensor:
        identity = x

        # if upscale is not None it will also be added to upsample
        out = x
        if self.upscale is not None:
            out = self.upscale(out)

        out = self.conv1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        upscale: Optional[nn.Module] = None,
    ) -> None:
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.conv2 = conv3x3(width, width, groups)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.upscale = upscale

    def forward(self, x) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        # if upscale is not None it will also be added to upsample
        if self.upscale is not None:
            out = self.upscale(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class Decoder(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        input_height: int = 32,
        latent_dim: int = 128,
        h_dim: int = 2048,
        groups: int = 1,
        widen: int = 1,
        width_per_group: int = 512,
        first_conv3x3: bool = False,
        remove_first_maxpool: bool = False,
    ) -> None:

        super(Decoder, self).__init__()

        self.first_conv3x3 = first_conv3x3
        self.remove_first_maxpool = remove_first_maxpool
        self.upscale_factor = 8

        if not first_conv3x3:
            self.upscale_factor *= 2

        if not remove_first_maxpool:
            self.upscale_factor *= 2

        self.input_height = input_height
        self.h_dim = h_dim
        self.groups = groups
        self.inplanes = h_dim
        self.base_width = 64
        num_out_filters = width_per_group * widen

        self.linear_projection1 = nn.Linear(latent_dim, h_dim, bias=True)
        self.linear_projection2 = nn.Linear(h_dim, h_dim, bias=True)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv1x1(self.h_dim // 16, self.h_dim)

        num_out_filters /= 2
        self.layer1 = self._make_layer(
            block, int(num_out_filters), layers[0], Interpolate(
                upscale='size', size=self.input_height // self.upscale_factor
            )
        )
        num_out_filters /= 2
        self.layer2 = self._make_layer(block, int(num_out_filters), layers[1], Interpolate())
        num_out_filters /= 2
        self.layer3 = self._make_layer(block, int(num_out_filters), layers[2], Interpolate())
        num_out_filters /= 2
        self.layer4 = self._make_layer(block, int(num_out_filters), layers[3], Interpolate())

        self.conv2 = conv3x3(int(num_out_filters) * block.expansion, self.base_width)
        self.final_conv = conv3x3(self.base_width, 3)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        upscale: Optional[nn.Module] = None,
    ) -> nn.Sequential:
        upsample = None

        if self.inplanes != planes * block.expansion or upscale is not None:
            # this is passed into residual block for skip connection
            upsample = []
            if upscale is not None:
                upsample.append(upscale)
            upsample.append(conv1x1(self.inplanes, planes * block.expansion))
            upsample = nn.Sequential(*upsample)

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                upsample,
                self.groups,
                self.base_width,
                upscale,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    upscale=None,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.linear_projection1(x))
        x = self.relu(self.linear_projection2(x))

        x = x.view(x.size(0), self.h_dim // 16, 4, 4)
        x = self.conv1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.remove_first_maxpool:
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv2(x)
        x = self.relu(x)

        if not self.first_conv3x3:
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.final_conv(x)

        return x


def decoder18(**kwargs):
    # layers list is opposite the encoder (in this case [2, 2, 2, 2])
    return Decoder(BasicBlock, [2, 2, 2, 2], **kwargs)


def decoder34(**kwargs):
    # layers list is opposite the encoder (in this case [3, 6, 4, 3])
    return Decoder(BasicBlock, [3, 6, 4, 3], **kwargs)


def decoder50(**kwargs):
    # layers list is opposite the encoder
    return Decoder(Bottleneck, [3, 6, 4, 3], **kwargs)


def decoder50w2(**kwargs):
    # layers list is opposite the encoder
    return Decoder(Bottleneck, [3, 6, 4, 3], widen=2, **kwargs)


def decoder50w4(**kwargs):
    # layers list is opposite the encoder
    return Decoder(Bottleneck, [3, 6, 4, 3], widen=4, **kwargs)


if __name__ == "__main__":
    model = decoder50(input_height=96, latent_dim=128, h_dim=2048)
    print(model)
