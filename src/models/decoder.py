import torch
import torch.nn as nn

from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


class Interpolate(nn.Module):

    def __init__(self, upscale: str = 'scale', size: Optional[int] = None):
        super().__init__()
        self.upscale = upscale
        self.size = size

    def forward(self, x):
        if self.upscale == 'scale':
            return F.interpolate(x, scale_factor=2, mode='nearest')
        elif self.upscale == 'size':
            return F.interpolate(x, size=(self.size, self.size), mode='nearest')


def conv3x3(in_planes: int, out_planes: int, groups: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, padding=1, groups=groups, bias=False
    )


def conv1x1(in_planes: int, out_planes: int) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        upsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        upscale: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample
        self.interpolate = Interpolate()
        self.upscale = upscale

    def forward(self, x) -> Tensor:
        identity = x

        # add upsample before first conv in the block
        # if the skip connection has upsample
        out = x
        if self.upscale is not None:
            out = self.upscale(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

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
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        upscale: Optional[nn.Module] = None,
    ) -> None:
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.interpolate = Interpolate()
        self.upscale = upscale

    def forward(self, x) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # add upsample before second conv in the block
        # if the skip connection has upsample
        if self.upscale is not None:
            out = self.upscale(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

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
        zero_init_residual: bool = False,
        groups: int = 1,
        widen: int = 1,
        width_per_group: int = 512,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        first_conv3x3: bool = False,
        remove_first_maxpool: bool = False,
    ) -> None:

        super(Decoder, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

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

        self.linear_projection = nn.Linear(latent_dim, h_dim, bias=False)
        self.bn_linear = nn.BatchNorm1d(h_dim)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = conv1x1(self.h_dim // 16, self.h_dim)
        self.bn1 = norm_layer(self.h_dim)

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

        self.conv2 = conv3x3(num_out_filters * block.expansion, self.base_width)
        self.bn2 = norm_layer(self.base_width)
        self.final_conv = conv3x3(self.base_width, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        upscale: Optional[nn.Module] = None,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None

        if self.inplanes != planes * block.expansion:
            # this is passed into residual block for skip connection
            upsample = []
            if upscale is not None:
                upsample.append(upscale)
            upsample.append(conv1x1(self.inplanes, planes * block.expansion))
            upsample.append(norm_layer(planes * block.expansion))
            upsample = nn.Sequential(*upsample)

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                upsample,
                self.groups,
                self.base_width,
                norm_layer,
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
                    norm_layer=norm_layer,
                    upscale=None,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear_projection(x)
        x = self.bn_linear(x)
        x = self.relu(x)

        x = x.view(x.size(0), self.h_dim // 16, 4, 4)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if not self.remove_first_maxpool:
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        if not self.first_conv3x3:
            x = F.interpolate(x, scale_factor=2, mode='nearest')

        x = self.final_conv(x)

        return x


def decoder18(**kwargs):
    # layers list is opposite the encoder (in this case [2, 2, 2, 2])
    return Decoder(BasicBlock, [2, 2, 2, 2], **kwargs)


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
