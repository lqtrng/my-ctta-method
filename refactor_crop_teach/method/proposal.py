"""Region Proposal interface: where to cut crops, from the foundation model's prediction.

    AllRegionsProposal      one crop per connected region of the foundation mask (the default)
    LargestRegionProposal   one crop, around the largest region only

Each crop is a square ``crop_scale`` x the region's extent (at least ``min_crop`` pixels), kept
inside the image. With no region at all, one crop is centered on the most confident pixel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Box:
    """A square crop, in pixels of the resized image."""

    top: int
    left: int
    side: int


class RegionProposal(ABC):
    def __init__(self, threshold: float = 0.5, crop_scale: float = 2.0, min_crop: int = 96) -> None:
        self.threshold = threshold
        self.crop_scale = crop_scale
        self.min_crop = min_crop

    def __call__(self, probability: np.ndarray) -> list[Box]:
        """probability: the foundation model's first-structure probability (H, W) -> crops."""
        components, count = ndimage.label(probability > self.threshold)
        regions = self.select([components == index for index in range(1, count + 1)])
        if not regions:
            regions = [np.zeros_like(probability, dtype=bool)]
        return [self.box(region, probability) for region in regions]

    @abstractmethod
    def select(self, regions: list[np.ndarray]) -> list[np.ndarray]:
        """Which connected regions (boolean masks) get a crop."""

    def box(self, region: np.ndarray, probability: np.ndarray) -> Box:
        size = region.shape[0]
        if not region.any():
            region = np.zeros_like(region, dtype=bool)
            region[np.unravel_index(np.argmax(probability), probability.shape)] = True
        rows, columns = np.nonzero(region)
        side = int(np.clip(max(np.ptp(rows) + 1, np.ptp(columns) + 1) * self.crop_scale, self.min_crop, size))
        top = int(np.clip(round((rows.min() + rows.max()) / 2 - side / 2), 0, size - side))
        left = int(np.clip(round((columns.min() + columns.max()) / 2 - side / 2), 0, size - side))
        return Box(top, left, side)


class AllRegionsProposal(RegionProposal):
    def select(self, regions: list[np.ndarray]) -> list[np.ndarray]:
        return regions


class LargestRegionProposal(RegionProposal):
    def select(self, regions: list[np.ndarray]) -> list[np.ndarray]:
        return [max(regions, key=lambda region: int(region.sum()))] if regions else []
