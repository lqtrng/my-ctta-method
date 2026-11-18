"""Network building blocks: ResUNet-34, Segformer-B0 and the full-image head.

Parameter names match crop_teach/networks.py, so its checkpoints load unchanged.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


# --- ResUNet-34 -------------------------------------------------------------------------


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(inputs))))) + residual)


class ResNet34Encoder(nn.Module):
    """ResNet-34 with torchvision's parameter names, returning the skip features."""

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._layer(64, 3, 1)
        self.layer2 = self._layer(128, 4, 2)
        self.layer3 = self._layer(256, 6, 2)
        self.layer4 = self._layer(512, 3, 2)

    def _layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != channels:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, channels, 1, stride, bias=False),
                                       nn.BatchNorm2d(channels))
        layers = [BasicBlock(self.inplanes, channels, stride, downsample)]
        self.inplanes = channels
        layers += [BasicBlock(channels, channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        first = self.relu(self.bn1(self.conv1(inputs)))
        layer1 = self.layer1(self.maxpool(first))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return self.layer4(layer3), [first, layer1, layer2, layer3]


class UpBlock(nn.Module):
    def __init__(self, up_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.x_conv = nn.Conv2d(skip_channels, out_channels // 2, 1)
        self.tr_conv = nn.ConvTranspose2d(up_channels, out_channels // 2, 2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.bn(F.relu(torch.cat([self.tr_conv(inputs), self.x_conv(skip)], dim=1)))


class ResUNet34Network(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.res = ResNet34Encoder()
        self.up1 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 256)
        self.up3 = UpBlock(256, 64, 256)
        self.up4 = UpBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)
        self.seg_head = nn.Conv2d(32, num_classes, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output, skips = self.res(imagenet_normalize(images))
        output = self.up1(F.relu(output), skips[3])
        output = self.up2(output, skips[2])
        output = self.up3(output, skips[1])
        output = self.up4(output, skips[0])
        return self.seg_head(F.relu(self.bnout(self.up5(output))))


# --- Segformer-B0 -----------------------------------------------------------------------


class SegformerB0Network(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        # SegformerConfig's defaults are MiT-B0's, so no download is needed.
        self.model = SegformerForSemanticSegmentation(SegformerConfig(num_labels=num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=imagenet_normalize(images)).logits  # 1/4 resolution
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


# --- The head on a frozen foundation model's features -----------------------------------


class SegmentationHead(nn.Module):
    """Small convolutional decoder on the foundation model's feature map."""

    def __init__(self, in_channels: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.GELU())
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
        )
        self.classifier = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = F.interpolate(self.reduce(features), scale_factor=2, mode="bilinear", align_corners=False)
        return self.classifier(self.refine(output))
