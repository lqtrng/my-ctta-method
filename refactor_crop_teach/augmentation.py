from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn
from torchvision.transforms import ColorJitter, RandomErasing

try:
    from .config import ERASING, JITTER
except ImportError:
    from config import ERASING, JITTER


class Augmentation(ABC):
    @abstractmethod
    def augment(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.augment(images)


class IdentityAugmentation(Augmentation):
    def augment(self, images: torch.Tensor) -> torch.Tensor:
        return images.clone()


class ColorJitterAugmentation(Augmentation):
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.05,
    ) -> None:
        self.transform = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def augment(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[0] == 1:
            return self.transform(images[0])[None]
        return torch.stack([self.transform(img) for img in images], dim=0)


class RandomErasingAugmentation(Augmentation):
    def __init__(
        self,
        p: float = 0.5,
        scale: tuple[float, float] = (0.02, 0.2),
        ratio: tuple[float, float] = (0.3, 3.3),
        value: str | float = "random",
    ) -> None:
        self.transform = RandomErasing(p=p, scale=scale, ratio=ratio, value=value)

    def augment(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[0] == 1:
            return self.transform(images[0])[None]
        return torch.stack([self.transform(img) for img in images], dim=0)


class ColorJitterRandomErasingAugmentation(Augmentation):
    def __init__(
        self,
        jitter: dict | None = None,
        erasing: dict | None = None,
    ) -> None:
        jitter_cfg = JITTER if jitter is None else jitter
        erasing_cfg = ERASING if erasing is None else erasing
        self.transform = nn.Sequential(ColorJitter(**jitter_cfg), RandomErasing(**erasing_cfg))

    def augment(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[0] == 1:
            return self.transform(images[0])[None]
        return torch.stack([self.transform(img) for img in images], dim=0)


class GaussianNoiseAugmentation(Augmentation):
    def __init__(self, std: float = 0.05) -> None:
        self.std = std

    def augment(self, images: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(images) * self.std
        return (images + noise).clamp(0.0, 1.0)


AUGMENTATIONS: dict[str, type[Augmentation]] = {
    "color_jitter_erasing": ColorJitterRandomErasingAugmentation,
    "color_jitter": ColorJitterAugmentation,
    "random_erasing": RandomErasingAugmentation,
    "gaussian_noise": GaussianNoiseAugmentation,
    "identity": IdentityAugmentation,
}


def build_augmentation(name: str = "color_jitter_erasing") -> Augmentation:
    if name not in AUGMENTATIONS:
        raise ValueError(f"Unknown augmentation {name!r}; expected one of {sorted(AUGMENTATIONS)}")
    return AUGMENTATIONS[name]()
