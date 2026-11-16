"""RegionProposal interface and its two strategies.

Both threshold the first structure (disc for OD/OC, prostate for prostate) of supervision_1, make
square boxes in the S frame, cut each box from the full-resolution image (scaled per axis) and
resize the cut to S x S. They differ only in how the thresholded mask is split into regions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from core.data_types import Region
from core.problem import structure_probabilities


class RegionProposal(ABC):
    THRESHOLD = 0.5    # on the first structure's probability
    SCALE = 2.0        # box side = 2 x the region's extent
    MIN_SIDE = 96      # smallest box side, in pixels of the S frame

    @abstractmethod
    def regions(self, mask: np.ndarray) -> list[np.ndarray]:
        """Split the thresholded mask (S x S, bool) into region masks, in proposal order."""

    @torch.no_grad()
    def propose(self, sup1: torch.Tensor, full: torch.Tensor) -> list[Region]:
        """sup1: class probabilities (1, C, S, S); full: the image at original resolution (1, 3, H, W)."""
        size = sup1.shape[-1]
        probability = structure_probabilities(sup1)[0, 0].cpu().numpy()
        regions = self.regions(probability > self.THRESHOLD)
        if not regions:  # Nothing passes the threshold: one box around the most confident pixel.
            peak = np.zeros_like(probability, dtype=bool)
            peak[np.unravel_index(np.argmax(probability), probability.shape)] = True
            regions = [peak]
        return [Region(self.cut(full, box, size), box) for box in map(self.square_box, regions)]

    def square_box(self, mask: np.ndarray) -> tuple[int, int, int, int]:
        """(top, left, side, side): a square around the region's center, kept inside the image."""
        size = mask.shape[0]
        rows, columns = np.nonzero(mask)
        side = int(np.clip(max(np.ptp(rows) + 1, np.ptp(columns) + 1) * self.SCALE, self.MIN_SIDE, size))
        top = int(np.clip(round((rows.min() + rows.max()) / 2 - side / 2), 0, size - side))
        left = int(np.clip(round((columns.min() + columns.max()) / 2 - side / 2), 0, size - side))
        return top, left, side, side

    @staticmethod
    def full_rectangle(box: tuple[int, int, int, int], size: int, height: int, width: int) -> tuple[int, int, int, int]:
        """The box of the S frame in full-resolution pixels: (top, bottom, left, right)."""
        top, left, box_height, box_width = box
        sy, sx = height / size, width / size
        return round(top * sy), round((top + box_height) * sy), round(left * sx), round((left + box_width) * sx)

    def cut(self, full: torch.Tensor, box: tuple[int, int, int, int], size: int) -> torch.Tensor:
        top, bottom, left, right = self.full_rectangle(box, size, full.shape[-2], full.shape[-1])
        return F.interpolate(full[:, :, top:bottom, left:right], size=(size, size), mode="bilinear",
                             align_corners=False, antialias=True)


class PerRegionSquare(RegionProposal):
    """One box per connected region, in ndimage.label order (raster order of each region's first pixel)."""

    def regions(self, mask: np.ndarray) -> list[np.ndarray]:
        components, count = ndimage.label(mask)
        return [components == index for index in range(1, count + 1)]


class UnionSquare(RegionProposal):
    """One box around all regions together."""

    def regions(self, mask: np.ndarray) -> list[np.ndarray]:
        return [mask] if mask.any() else []
