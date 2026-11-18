from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch

try:
    from .proposal import CropBox
    from .structures import structure_probabilities
except ImportError:
    from proposal import CropBox
    from structures import structure_probabilities


def _pixel_confidence(probability: torch.Tensor, gamma: float = 1.3, variance: float = 0.1) -> torch.Tensor:
    gaussian = torch.exp(-((probability - 0.5) ** 2) / (2.0 * variance)) / math.sqrt(2.0 * math.pi * variance)
    return (gamma - gaussian).clamp_min(1e-6)


class RefinedRegionFusion(ABC):
    @abstractmethod
    def fuse(
        self, regions: list[tuple[CropBox, torch.Tensor]], image_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __call__(
        self, regions: list[tuple[CropBox, torch.Tensor]], image_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.fuse(regions, image_size)


class MaxForegroundFusion(RefinedRegionFusion):
    def fuse(
        self, regions: list[tuple[CropBox, torch.Tensor]], image_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not regions:
            raise ValueError("regions must not be empty")
        sample = regions[0][1]
        num_classes = sample.shape[1]
        device, dtype = sample.device, sample.dtype
        merged: torch.Tensor | None = None
        inside = torch.zeros(1, 1, image_size, image_size, device=device, dtype=dtype)
        for box, probs in regions:
            pasted = torch.zeros(1, num_classes, image_size, image_size, device=device, dtype=dtype)
            pasted[:, :, box.top : box.top + box.side, box.left : box.left + box.side] = probs
            inside[:, :, box.top : box.top + box.side, box.left : box.left + box.side] = 1.0
            if merged is None:
                merged = pasted
            else:
                better = pasted[:, 1:].sum(1, keepdim=True) > merged[:, 1:].sum(1, keepdim=True)
                merged = torch.where(better, pasted, merged)
        assert merged is not None
        structures = (structure_probabilities(merged) * inside).clamp(1e-6, 1.0 - 1e-6)
        merged_classes = merged.clone()
        merged_classes[:, 0:1] = torch.where(
            inside > 0, merged_classes[:, 0:1], torch.ones_like(merged_classes[:, 0:1])
        )
        return structures, merged_classes


class AverageFusion(RefinedRegionFusion):
    def fuse(
        self, regions: list[tuple[CropBox, torch.Tensor]], image_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not regions:
            raise ValueError("regions must not be empty")
        sample = regions[0][1]
        num_classes = sample.shape[1]
        device, dtype = sample.device, sample.dtype
        accum = torch.zeros(1, num_classes, image_size, image_size, device=device, dtype=dtype)
        counts = torch.zeros(1, 1, image_size, image_size, device=device, dtype=dtype)
        for box, probs in regions:
            accum[:, :, box.top : box.top + box.side, box.left : box.left + box.side] += probs
            counts[:, :, box.top : box.top + box.side, box.left : box.left + box.side] += 1.0
        inside = (counts > 0).to(dtype)
        merged = accum / counts.clamp_min(1.0)
        merged[:, 0:1] = torch.where(inside > 0, merged[:, 0:1], torch.ones_like(merged[:, 0:1]))
        structures = (structure_probabilities(merged) * inside).clamp(1e-6, 1.0 - 1e-6)
        return structures, merged


class ConfidenceWeightedFusion(RefinedRegionFusion):
    def fuse(
        self, regions: list[tuple[CropBox, torch.Tensor]], image_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not regions:
            raise ValueError("regions must not be empty")
        sample = regions[0][1]
        num_classes = sample.shape[1]
        device, dtype = sample.device, sample.dtype
        weighted_sum = torch.zeros(1, num_classes, image_size, image_size, device=device, dtype=dtype)
        weight_total = torch.zeros(1, 1, image_size, image_size, device=device, dtype=dtype)
        for box, probs in regions:
            fg_prob = probs[:, 1:].sum(dim=1, keepdim=True).clamp(0.0, 1.0)
            weight = _pixel_confidence(fg_prob)
            weighted_sum[:, :, box.top : box.top + box.side, box.left : box.left + box.side] += probs * weight
            weight_total[:, :, box.top : box.top + box.side, box.left : box.left + box.side] += weight
        inside = (weight_total > 0).to(dtype)
        merged = weighted_sum / weight_total.clamp_min(1e-6)
        merged[:, 0:1] = torch.where(inside > 0, merged[:, 0:1], torch.ones_like(merged[:, 0:1]))
        structures = (structure_probabilities(merged) * inside).clamp(1e-6, 1.0 - 1e-6)
        return structures, merged


FUSIONS: dict[str, type[RefinedRegionFusion]] = {
    "max_foreground": MaxForegroundFusion,
    "average": AverageFusion,
    "confidence_weighted": ConfidenceWeightedFusion,
}


def build_fusion(name: str = "max_foreground") -> RefinedRegionFusion:
    if name not in FUSIONS:
        raise ValueError(f"Unknown fusion strategy {name!r}; expected one of {sorted(FUSIONS)}")
    return FUSIONS[name]()
