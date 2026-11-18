"""Augmentation interface: the view of the test image the target model learns from.

    JitterErasingAugmentation   color jitter, then random erasing (the default)
    NoAugmentation              the image unchanged
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torchvision.transforms import ColorJitter, RandomErasing


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """(1, 3, H, W) in [0, 1] -> an augmented copy of the same shape."""


class JitterErasingAugmentation(Augmentation):
    def __init__(self, jitter: dict, erasing: dict) -> None:
        self.transform = nn.Sequential(ColorJitter(**jitter), RandomErasing(**erasing))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.transform(image[0])[None]


class NoAugmentation(Augmentation):
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image
