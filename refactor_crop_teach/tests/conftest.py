"""Small stand-ins shared by the unit tests: no checkpoint, no real data, runs on CPU in seconds."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from config import Problem
from method.models import Model

TOY = Problem("toy", {"A": "first", "B": "second", "C": "third"}, {"A": 3, "B": 2, "C": 4}, ("fg",), 32)


class TinyModel(Model):
    """A trainable 1x1 convolution with batch norm."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(3, num_classes, 1)
        self.bn = nn.BatchNorm2d(num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(images))


class BrightModel(Model):
    """No parameters: foreground where the first channel is bright."""

    num_classes = 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        foreground = 20.0 * (images[:, :1] - 0.5)
        return torch.cat([torch.zeros_like(foreground), foreground], dim=1)


def toy_image(size: tuple[int, int], top: int, left: int, side: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = (rng.random((*size, 3)) * 80).astype(np.uint8)
    image[top: top + side, left: left + side] = 230
    mask = np.zeros(size, dtype=np.uint8)
    mask[top: top + side, left: left + side] = 1
    return image, mask


@pytest.fixture
def toy_data(tmp_path):
    """data/toy/domain_<X>/{Images,Masks}, full-size images 40 x 48 with one bright square."""
    for domain, count in TOY.counts.items():
        for kind in ("Images", "Masks"):
            (tmp_path / "toy" / f"domain_{domain}" / kind).mkdir(parents=True)
        for index in range(count):
            image, mask = toy_image((40, 48), 8 + index, 10, 12, seed=index)
            name = f"{index:04d}_{domain}"
            Image.fromarray(image).save(tmp_path / "toy" / f"domain_{domain}" / "Images" / f"{name}.png")
            Image.fromarray(mask).save(tmp_path / "toy" / f"domain_{domain}" / "Masks" / f"{name}.png")
    return tmp_path
