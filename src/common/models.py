"""ResUNet-34 source segmentation model (three classes: background, disc rim, cup)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 3


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        if self.downsample is not None:
            residual = self.downsample(inputs)
        return self.relu(output + residual)


class ResNet34Encoder(nn.Module):
    """ResNet-34 encoder with state-dict names matching the supplied checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)
        # Kept for exact compatibility with the supplied checkpoint.
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, 1000)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                count = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2 / count))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers: list[nn.Module] = [BasicBlock(self.inplanes, channels, stride, downsample)]
        self.inplanes = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips: list[torch.Tensor] = []
        output = self.relu(self.bn1(self.conv1(inputs)))
        skips.append(output)
        output = self.layer1(self.maxpool(output))
        skips.append(output)
        output = self.layer2(output)
        skips.append(output)
        output = self.layer3(output)
        skips.append(output)
        output = self.layer4(output)
        skips.append(output)
        return output, skips


class UnetBlock(nn.Module):
    def __init__(self, up_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        half = out_channels // 2
        self.x_conv = nn.Conv2d(skip_channels, half, 1)
        self.tr_conv = nn.ConvTranspose2d(up_channels, half, 2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        inputs = self.tr_conv(inputs)
        skip = self.x_conv(skip)
        return self.bn(F.relu(torch.cat([inputs, skip], dim=1)))


class ResUnet34(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.res = ResNet34Encoder()
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)
        self.seg_head = nn.Conv2d(32, num_classes, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, skips = self.res(inputs)
        output = self.up1(F.relu(output), skips[3])
        output = self.up2(output, skips[2])
        output = self.up3(output, skips[1])
        output = self.up4(output, skips[0])
        output = self.up5(output)
        return self.seg_head(F.relu(self.bnout(output)))
