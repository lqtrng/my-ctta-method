"""Adapting: holds the target model, adapts it on each test image and returns its prediction.

One Crop-Teach step on a test image:
  1. foundation:  the frozen foundation model predicts the whole image;
  2. proposal:    the RegionProposal turns its prediction into square crops;
  3. refinement:  the frozen refiner (crop model) predicts each crop, cut from the full-size image;
  4. fusion:      the RegionFusion merges the refined crops into one whole-image prediction;
                  steps 1 and 4 give the two supervisions;
  5. learning:    the Learning strategy updates the target model on an augmented copy of the image;
  6. prediction:  the updated target model predicts the clean image (the scored prediction).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from method.augmentation import Augmentation
from method.fusion import RefinedRegion, RegionFusion
from method.learning import Learning, Supervision
from method.models import Model
from method.proposal import Box, RegionProposal
from method.structures import structure_probabilities


class Adapting:
    def __init__(self, target: Model, foundation: Model, refiner: Model, proposal: RegionProposal,
                 fusion: RegionFusion, augmentation: Augmentation, learning: Learning) -> None:
        self.target = target
        self.foundation = foundation.requires_grad_(False)
        self.refiner = refiner.requires_grad_(False)
        self.proposal = proposal
        self.fusion = fusion
        self.augmentation = augmentation
        self.learning = learning
        self.learning.attach(target)

    def __call__(self, image: torch.Tensor, full: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """image: (1, 3, S, S), full: (1, 3, H, W) -> target class probabilities (C, S, S) after the
        update, and statistics (incl. the two teachers' own predictions, for scoring)."""
        supervision, boxes = self.supervise(image, full)
        statistics = self.learning.learn(self.target, self.augmentation(image), supervision)
        teachers = {"head_alone": supervision.foundation[0], "crop_teacher": supervision.refined[0]}
        return self.target.predict(image)[0], {**statistics, "crops": len(boxes), "teachers": teachers}

    @torch.no_grad()
    def supervise(self, image: torch.Tensor, full: torch.Tensor) -> tuple[Supervision, list[Box]]:
        foundation = self.foundation.predict(image)
        probability = structure_probabilities(foundation)[0, 0].cpu().numpy()
        boxes = self.proposal(probability)
        size = image.shape[-1]
        fused = self.fusion([self.refine(full, box, size) for box in boxes], size)
        return Supervision(foundation, fused.classes, fused.inside), boxes

    def refine(self, full: torch.Tensor, box: Box, size: int) -> RefinedRegion:
        """The refiner's prediction of the box, cut from the full-size image at the input size."""
        sy, sx = full.shape[-2] / size, full.shape[-1] / size
        crop = full[:, :, round(box.top * sy): round((box.top + box.side) * sy),
                    round(box.left * sx): round((box.left + box.side) * sx)]
        crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
        classes = self.refiner.predict(crop)
        classes = F.interpolate(classes, size=(box.side, box.side), mode="bilinear", align_corners=False)
        return RefinedRegion(box, classes)
