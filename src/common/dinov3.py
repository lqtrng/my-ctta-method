"""Frozen DINOv3 backbone and the small decoder head trained on its patch tokens.

Shared by the head training script (``training/train_dinov3_head.py``), the
ReDi-FM + DINOv3 method and the DINOv3-only baselines.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from common.paths import dinov3_weights, require_file


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dinov3(directory: Path, arch: str, device: torch.device) -> nn.Module:
    import timm

    weights = require_file(dinov3_weights(directory, arch), "DINOv3 weights")
    model = timm.create_model(
        arch.split(".")[0],
        pretrained=True,
        dynamic_img_size=True,
        pretrained_cfg_overlay=dict(file=str(weights)),
    )
    return model.requires_grad_(False).to(device).eval()


def dinov3_tokens(
    model: nn.Module, crops: torch.Tensor, layers: tuple[int, ...]
) -> torch.Tensor:
    """Patch tokens of the requested blocks, concatenated along channels."""
    mean = crops.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = crops.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    # DINOv3 overflows in float16; bfloat16 keeps the range of float32.
    with torch.no_grad(), torch.autocast(device_type=crops.device.type, dtype=torch.bfloat16):
        outputs = model.forward_intermediates(
            (crops - mean) / std, indices=list(layers), intermediates_only=True
        )
    return torch.cat([output.float() for output in outputs], dim=1)


class SegmentationHead(nn.Module):
    """Small convolutional decoder: patch tokens -> [background, disc-rim, cup]."""

    def __init__(self, in_channels: int, hidden: int = 256, num_classes: int = 3) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        output = self.reduce(tokens)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)
        return self.classifier(self.refine(output))
