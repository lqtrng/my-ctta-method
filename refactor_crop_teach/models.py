"""Model interface and its two implementations.

A Model takes images in [0, 1], shape (B, 3, S, S), and returns class probabilities (B, C, S, S).
Models are only loaded from trained checkpoints here; nothing is trained from scratch.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from problem import WEIGHTS_DIR


class Model(nn.Module, ABC):
    """``model(image)`` -> class probabilities; ``model.load(checkpoint)`` reads trained weights."""

    @abstractmethod
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """(B, 3, S, S) in [0, 1] -> class probabilities (B, C, S, S)."""

    @abstractmethod
    def load(self, checkpoint: Path) -> "Model":
        """Read the trained weights from a checkpoint file. Returns self."""


def read_checkpoint(path: Path, num_classes: int) -> dict:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["num_classes"] != num_classes:
        raise ValueError(f"{path} has {checkpoint['num_classes']} classes, expected {num_classes}")
    return checkpoint


# --- ResUNet-34 (the ReDi-FM paper's network for OD/OC and prostate) -----------------------


def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    return (images - mean) / std


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


class ResUNet34(Model):
    """The ReDi-FM paper's ResUNet-34. Used as the crop model and as the target model."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.res = ResNet34Encoder()
        self.up1 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 256)
        self.up3 = UpBlock(256, 64, 256)
        self.up4 = UpBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)
        self.seg_head = nn.Conv2d(32, num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output, skips = self.res(imagenet_normalize(image))
        output = self.up1(F.relu(output), skips[3])
        output = self.up2(output, skips[2])
        output = self.up3(output, skips[1])
        output = self.up4(output, skips[0])
        return F.softmax(self.seg_head(F.relu(self.bnout(self.up5(output)))), dim=1)

    def load(self, checkpoint: Path) -> "ResUNet34":
        self.load_state_dict(read_checkpoint(checkpoint, self.num_classes)["state_dict"])
        return self


# --- SAM head: SAM ViT-H image encoder (frozen) plus a small trained head -------------------


class SegmentationHead(nn.Module):
    """Small convolutional decoder on SAM's image embedding (256 channels on a 64 x 64 grid)."""

    def __init__(self, in_channels: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.GELU())
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
        )
        self.classifier = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        output = F.interpolate(self.reduce(embedding), scale_factor=2, mode="bilinear", align_corners=False)
        return self.classifier(self.refine(output))


class SamHead(Model):
    """Resizes the S x S input to 1024 x 1024 for SAM's image encoder, runs the trained head on
    its embedding and returns S x S class probabilities. Always frozen and in eval mode.

    The encoder is passed in, so one loaded SAM serves every source domain (and a small fake one
    serves the unit test). It is kept out of the module tree: only the head is in state_dict().
    """

    SAM_WEIGHTS = WEIGHTS_DIR / "sam_vit_h_4b8939.pth"
    RESOLUTION = 1024              # SAM's fixed input size
    EMBEDDING_CHANNELS = 256       # SAM's output embedding
    HIDDEN = 256                   # head width
    PIXEL_MEAN = (123.675, 116.28, 103.53)
    PIXEL_STD = (58.395, 57.12, 57.375)

    def __init__(self, num_classes: int, encoder: nn.Module) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head = SegmentationHead(self.EMBEDDING_CHANNELS, self.HIDDEN, num_classes)
        self._encoder = [encoder.requires_grad_(False).eval()]
        self.requires_grad_(False).eval()

    @classmethod
    def load_encoder(cls, device: torch.device) -> nn.Module:
        """SAM ViT-H's image encoder, frozen, in bfloat16."""
        from segment_anything import sam_model_registry

        if not cls.SAM_WEIGHTS.is_file():
            raise FileNotFoundError(f"SAM ViT-H weights not found: {cls.SAM_WEIGHTS}")
        encoder = sam_model_registry["vit_h"](checkpoint=str(cls.SAM_WEIGHTS)).image_encoder
        return encoder.requires_grad_(False).to(device=device, dtype=torch.bfloat16).eval()

    def train(self, mode: bool = True) -> "SamHead":
        return super().train(False)  # Frozen: never in training mode.

    @torch.no_grad()
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoder = self._encoder[0]
        inputs = F.interpolate(image, size=(self.RESOLUTION, self.RESOLUTION), mode="bilinear", align_corners=False)
        mean = inputs.new_tensor(self.PIXEL_MEAN).view(1, 3, 1, 1)
        std = inputs.new_tensor(self.PIXEL_STD).view(1, 3, 1, 1)
        inputs = ((inputs * 255.0 - mean) / std).to(next(encoder.parameters()).dtype)
        # One image at a time: SAM's global-attention blocks need about 0.5 GB per image.
        embedding = torch.cat([encoder(inputs[index: index + 1]) for index in range(len(inputs))]).float()
        logits = F.interpolate(self.head(embedding), size=image.shape[-2:], mode="bilinear", align_corners=False)
        return F.softmax(logits, dim=1)

    def load(self, checkpoint: Path) -> "SamHead":
        saved = read_checkpoint(checkpoint, self.num_classes)
        if saved["backbone"] != "sam_vit_h" or saved["hidden"] != self.HIDDEN or saved["resolution"] != self.RESOLUTION:
            raise ValueError(f"{checkpoint} is not a SAM ViT-H head of width {self.HIDDEN} at {self.RESOLUTION}")
        self.head.load_state_dict(saved["state_dict"])
        return self
