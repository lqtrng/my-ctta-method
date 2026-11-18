"""Refined Region Fusion interface: the refiner's crop predictions -> one whole-image prediction.

    MostForegroundFusion   where crops overlap, the crop with the most foreground wins (the default)
    MeanFusion             where crops overlap, their class probabilities are averaged

Outside every crop all class probabilities are 0; ``inside`` marks the pixels some crop covers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from method.proposal import Box


@dataclass
class RefinedRegion:
    box: Box
    classes: torch.Tensor  # (1, C, side, side) class probabilities, in the resized image's pixels


@dataclass
class FusedRegions:
    classes: torch.Tensor  # (1, C, S, S); 0 outside every crop
    inside: torch.Tensor   # (1, 1, S, S); 1 inside some crop


def paste(region: RefinedRegion, size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The region's class probabilities on an empty (all-zero) image, and its coverage."""
    box, classes = region.box, region.classes
    pasted = torch.zeros(1, classes.shape[1], size, size, device=classes.device)
    covered = torch.zeros(1, 1, size, size, device=classes.device)
    pasted[:, :, box.top: box.top + box.side, box.left: box.left + box.side] = classes
    covered[:, :, box.top: box.top + box.side, box.left: box.left + box.side] = 1.0
    return pasted, covered


class RegionFusion(ABC):
    def __call__(self, regions: list[RefinedRegion], size: int) -> FusedRegions:
        if not regions:
            raise ValueError("Nothing to fuse: the proposal gives at least one region")
        pasted = [paste(region, size) for region in regions]
        inside = torch.stack([covered for _, covered in pasted]).amax(dim=0)
        return FusedRegions(self.fuse([classes for classes, _ in pasted], [covered for _, covered in pasted]), inside)

    @abstractmethod
    def fuse(self, pasted: list[torch.Tensor], covered: list[torch.Tensor]) -> torch.Tensor:
        """Per-region class maps (1, C, S, S), zero outside the region -> one class map."""


class MostForegroundFusion(RegionFusion):
    def fuse(self, pasted: list[torch.Tensor], covered: list[torch.Tensor]) -> torch.Tensor:
        merged = pasted[0]
        for classes in pasted[1:]:
            better = classes[:, 1:].sum(1, keepdim=True) > merged[:, 1:].sum(1, keepdim=True)
            merged = torch.where(better, classes, merged)
        return merged


class MeanFusion(RegionFusion):
    def fuse(self, pasted: list[torch.Tensor], covered: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(pasted).sum(dim=0) / torch.stack(covered).sum(dim=0).clamp_min(1.0)
