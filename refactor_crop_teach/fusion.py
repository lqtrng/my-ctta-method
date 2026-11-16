"""Fusion interface: crop predictions -> supervision_2 in the S frame."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from data_types import Region
from problem import structure_probabilities


class Fusion(ABC):
    @abstractmethod
    def fuse(self, predictions: list[torch.Tensor], regions: list[Region], size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """predictions[i]: the crop model's class probabilities (1, C, S, S) for regions[i], in the
        crop's own frame. Returns (supervision_2: nested structure probabilities (1, K, S, S),
        merged: class probabilities (1, C, S, S), for scoring the crop teacher on its own)."""


class MostForeground(Fusion):
    """Each crop prediction is resized to its box and pasted there. Where boxes overlap, the crop
    with the most foreground wins, per pixel; ties go to the earlier region. Outside every box,
    supervision_2 is 1e-6: confident background, not ignored by the loss."""

    @torch.no_grad()
    def fuse(self, predictions: list[torch.Tensor], regions: list[Region], size: int) -> tuple[torch.Tensor, torch.Tensor]:
        merged = None
        inside = torch.zeros(1, 1, size, size, device=predictions[0].device)
        for probs, region in zip(predictions, regions, strict=True):
            top, left, height, width = region.box
            pasted = torch.zeros(1, probs.shape[1], size, size, device=probs.device)  # All classes 0.
            pasted[:, :, top: top + height, left: left + width] = F.interpolate(
                probs, size=(height, width), mode="bilinear", align_corners=False)
            inside[:, :, top: top + height, left: left + width] = 1.0
            if merged is None:
                merged = pasted
            else:
                better = pasted[:, 1:].sum(1, keepdim=True) > merged[:, 1:].sum(1, keepdim=True)
                merged = torch.where(better, pasted, merged)  # The whole class vector is swapped.
        supervision_2 = (structure_probabilities(merged) * inside).clamp(1e-6, 1.0 - 1e-6)
        return supervision_2, merged
