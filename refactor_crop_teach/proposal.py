from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage

try:
    from .config import CROP_SCALE, MIN_CROP, PSEUDO_THRESHOLD
except ImportError:
    from config import CROP_SCALE, MIN_CROP, PSEUDO_THRESHOLD


@dataclass(frozen=True)
class CropBox:
    top: int
    left: int
    side: int


def box_from_mask(
    mask: np.ndarray,
    probability: np.ndarray,
    crop_scale: float = CROP_SCALE,
    min_crop: int = MIN_CROP,
) -> CropBox:
    size = mask.shape[0]
    if not mask.any():
        mask = np.zeros_like(mask, dtype=bool)
        mask[np.unravel_index(np.argmax(probability), probability.shape)] = True
    rows, columns = np.nonzero(mask)
    side = int(np.clip(max(np.ptp(rows) + 1, np.ptp(columns) + 1) * crop_scale, min_crop, size))
    top = int(np.clip(round((rows.min() + rows.max()) / 2 - side / 2), 0, size - side))
    left = int(np.clip(round((columns.min() + columns.max()) / 2 - side / 2), 0, size - side))
    return CropBox(top=top, left=left, side=side)


class RegionProposal(ABC):
    @abstractmethod
    def propose(self, structure_probs: torch.Tensor) -> list[CropBox]:
        raise NotImplementedError

    def __call__(self, structure_probs: torch.Tensor) -> list[CropBox]:
        return self.propose(structure_probs)


class AllRegionsProposal(RegionProposal):
    def __init__(
        self,
        threshold: float = PSEUDO_THRESHOLD,
        crop_scale: float = CROP_SCALE,
        min_crop: int = MIN_CROP,
    ) -> None:
        self.threshold = threshold
        self.crop_scale = crop_scale
        self.min_crop = min_crop

    def propose(self, structure_probs: torch.Tensor) -> list[CropBox]:
        probability = structure_probs[0, 0].detach().cpu().numpy()
        components, count = ndimage.label(probability > self.threshold)
        boxes = [
            box_from_mask(components == index, probability, self.crop_scale, self.min_crop)
            for index in range(1, count + 1)
        ]
        if not boxes:
            boxes = [
                box_from_mask(
                    np.zeros_like(probability, dtype=bool),
                    probability,
                    self.crop_scale,
                    self.min_crop,
                )
            ]
        return boxes


class LargestRegionProposal(RegionProposal):
    def __init__(
        self,
        threshold: float = PSEUDO_THRESHOLD,
        crop_scale: float = CROP_SCALE,
        min_crop: int = MIN_CROP,
    ) -> None:
        self.threshold = threshold
        self.crop_scale = crop_scale
        self.min_crop = min_crop

    def propose(self, structure_probs: torch.Tensor) -> list[CropBox]:
        probability = structure_probs[0, 0].detach().cpu().numpy()
        binary = probability > self.threshold
        components, count = ndimage.label(binary)
        if count == 0:
            component = np.zeros_like(binary, dtype=bool)
        else:
            sizes = ndimage.sum(binary, components, range(1, count + 1))
            component = components == int(np.argmax(sizes) + 1)
        return [box_from_mask(component, probability, self.crop_scale, self.min_crop)]


class MostConfidentRegionProposal(RegionProposal):
    def __init__(
        self,
        threshold: float = PSEUDO_THRESHOLD,
        crop_scale: float = CROP_SCALE,
        min_crop: int = MIN_CROP,
    ) -> None:
        self.threshold = threshold
        self.crop_scale = crop_scale
        self.min_crop = min_crop

    def propose(self, structure_probs: torch.Tensor) -> list[CropBox]:
        probability = structure_probs[0, 0].detach().cpu().numpy()
        binary = probability > self.threshold
        components, count = ndimage.label(binary)
        if count == 0:
            component = np.zeros_like(binary, dtype=bool)
        else:
            means = ndimage.mean(probability, components, range(1, count + 1))
            component = components == int(np.argmax(means) + 1)
        return [box_from_mask(component, probability, self.crop_scale, self.min_crop)]


class WholeImageProposal(RegionProposal):
    def propose(self, structure_probs: torch.Tensor) -> list[CropBox]:
        size = int(structure_probs.shape[-1])
        return [CropBox(top=0, left=0, side=size)]


PROPOSALS: dict[str, type[RegionProposal]] = {
    "all": AllRegionsProposal,
    "largest": LargestRegionProposal,
    "confident": MostConfidentRegionProposal,
    "whole": WholeImageProposal,
}


def build_proposal(name: str = "all", **kwargs) -> RegionProposal:
    if name not in PROPOSALS:
        raise ValueError(f"Unknown proposal strategy {name!r}; expected one of {sorted(PROPOSALS)}")
    cls = PROPOSALS[name]
    return cls() if name == "whole" else cls(**kwargs)
