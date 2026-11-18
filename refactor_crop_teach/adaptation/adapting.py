"""Adapting: one Crop-Teach step per image (steps [2] to [7] of ARCHITECTURE.md)."""

from __future__ import annotations

import random

import numpy as np
import torch

from adaptation.fusion import Fusion, MostForeground
from adaptation.learning import CropTeachLearning, Learning
from modeling.models import Model
from adaptation.region_proposal import PerRegionSquare, RegionProposal


class Adapting:
    """Holds the target model, adapts it on each image and returns its prediction.

    The three models, the strategies and the optimizer state are kept across the whole stream;
    nothing is ever reset. It sees ``image`` and ``full`` only, never the label.
    ``seed`` is set once, here at the start, and never again.
    """

    def __init__(self, full_model: Model, crop_model: Model, target_model: Model,
                 region_proposal: RegionProposal | None = None, fusion: Fusion | None = None,
                 learning: Learning | None = None, seed: int = 0) -> None:
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.full_model = full_model.requires_grad_(False).eval()
        self.crop_model = crop_model.requires_grad_(False).eval()
        self.target_model = target_model
        self.region_proposal = region_proposal or PerRegionSquare()
        self.fusion = fusion or MostForeground()
        self.learning = learning or CropTeachLearning()
        # What the last adaptation step did: loss, consensus and number of crops.
        self.last: dict = {}

    def __call__(self, image: torch.Tensor, full: torch.Tensor) -> torch.Tensor:
        """image: (1, 3, S, S); full: (1, 3, H, W). Returns class probabilities (C, S, S)."""
        size = image.shape[-1]
        with torch.no_grad():
            supervision_1 = self.full_model(image)                                   # [2]
            regions = self.region_proposal.propose(supervision_1, full)              # [3]
            predictions = [self.crop_model(region.crop) for region in regions]       # [4]
            supervision_2, _ = self.fusion.fuse(predictions, regions, size)          # [5]
        outcome = self.learning.step(self.target_model, image, supervision_1, supervision_2)  # [6]
        self.target_model.eval()
        with torch.no_grad():
            prediction = self.target_model(image)[0]                                 # [7]
        self.last = {**outcome, "crops": len(regions)}
        return prediction
