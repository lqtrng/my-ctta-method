"""Augmentation interface. Applied to the student's input only."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torchvision.transforms import ColorJitter, RandomErasing


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """(1, 3, S, S) in [0, 1] -> an augmented copy of the same shape."""


class JitterErase(Augmentation):
    """Color jitter, then random erasing with random values."""

    JITTER = dict(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    ERASING = dict(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random")

    def __init__(self) -> None:
        self.jitter = ColorJitter(**self.JITTER)
        self.erasing = RandomErasing(**self.ERASING)

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.erasing(self.jitter(image[0]))[None]
