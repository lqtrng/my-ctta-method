"""Candidate losses for the target model, on the frozen teachers of each image."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from common.redifm import (  # noqa: E402
    confidence_weight,
    hard_class_balanced_soft_loss,
    nested_foundation_classes,
    weighted_binary_segmentation_loss,
)


@dataclass
class Context:
    target_classes: torch.Tensor  # Target model on the augmented image: class probabilities.
    target: torch.Tensor  # The same as structures (OD, OC) or (prostate).
    source_classes: torch.Tensor  # DINOv3 + full-image head.
    source: torch.Tensor
    source_masks: torch.Tensor
    foundation_classes: torch.Tensor  # Crop CNN, background outside the crop.
    foundation: torch.Tensor
    foundation_masks: torch.Tensor
    consensus: torch.Tensor  # Dice of the two masks, one number for the image.
    box: tuple[int, int, int]
    entropy_threshold: float
    model: nn.Module
    initial: dict[str, torch.Tensor]


@dataclass(frozen=True)
class Candidate:
    loss: Callable[[Context], torch.Tensor]
    learning_rate: float | None = None
    freeze_batch_norm: bool = False  # Keep the source statistics instead of updating them per image.
    scale_lr_by_agreement: bool = False  # Learning rate of each image = base rate x consensus.


def foundation_hard(c: Context, weight: torch.Tensor | None = None) -> torch.Tensor:
    weight = confidence_weight(c.foundation) if weight is None else weight
    return weighted_binary_segmentation_loss(c.target, c.foundation_masks, weight)


def source_hard(c: Context, weight: torch.Tensor | None = None) -> torch.Tensor:
    weight = confidence_weight(c.source) if weight is None else weight
    return weighted_binary_segmentation_loss(c.target, c.source_masks, weight)


def paper_soft(c: Context) -> torch.Tensor:
    return hard_class_balanced_soft_loss(
        c.target_classes, c.source_classes, c.foundation_classes, c.consensus, c.entropy_threshold
    )


def distill(c: Context, teacher: torch.Tensor) -> torch.Tensor:
    """Cross-entropy from a teacher's probabilities to the target's."""
    return -(teacher * c.target_classes.clamp_min(1e-6).log()).sum(dim=1).mean()


def agreement(c: Context, outside: float) -> torch.Tensor:
    """1 where the two teachers' masks agree, ``outside`` where they do not."""
    return torch.where(c.source_masks == c.foundation_masks, torch.ones_like(c.foundation), torch.full_like(c.foundation, outside))


def baseline(c: Context) -> torch.Tensor:
    return foundation_hard(c) + c.consensus * source_hard(c) + paper_soft(c)


def crop_hard(c: Context) -> torch.Tensor:
    return foundation_hard(c)


def crop_hard_plain(c: Context) -> torch.Tensor:
    return foundation_hard(c, torch.ones_like(c.foundation))


def crop_soft(c: Context) -> torch.Tensor:
    return foundation_hard(c) + paper_soft(c)


def crop_distill(c: Context) -> torch.Tensor:
    return foundation_hard(c) + distill(c, c.foundation_classes)


def image_gate(c: Context) -> torch.Tensor:
    return c.consensus * foundation_hard(c)


def pixel_agree(c: Context) -> torch.Tensor:
    return foundation_hard(c, confidence_weight(c.foundation) * agreement(c, 0.1))


def pixel_agree_both(c: Context) -> torch.Tensor:
    agree = agreement(c, 0.1)
    return foundation_hard(c, confidence_weight(c.foundation) * agree) + source_hard(c, confidence_weight(c.source) * agree)


def fused_probability(c: Context, gated: bool = True) -> torch.Tensor:
    """Teachers fused per pixel by confidence, the source scaled by the consensus if gated."""
    weight_f = confidence_weight(c.foundation)
    weight_s = confidence_weight(c.source) * (c.consensus if gated else 1.0)
    return ((weight_f * c.foundation + weight_s * c.source) / (weight_f + weight_s).clamp_min(1e-6)).clamp(1e-6, 1 - 1e-6)


def fused_hard(c: Context, fused: torch.Tensor, extra: torch.Tensor | float = 1.0) -> torch.Tensor:
    return weighted_binary_segmentation_loss(c.target, fused > 0.5, confidence_weight(fused) * extra)


def pixel_fused(c: Context) -> torch.Tensor:
    return fused_hard(c, fused_probability(c))


def fused_nogate(c: Context) -> torch.Tensor:
    return fused_hard(c, fused_probability(c, gated=False))


def fused_distill(c: Context) -> torch.Tensor:
    fused = fused_probability(c)
    return fused_hard(c, fused) + distill(c, nested_foundation_classes(fused))


def fused_soft(c: Context) -> torch.Tensor:
    return pixel_fused(c) + paper_soft(c)


def fused_agree(c: Context) -> torch.Tensor:
    return fused_hard(c, fused_probability(c), agreement(c, 0.1))


def agree_both_soft(c: Context) -> torch.Tensor:
    return pixel_agree_both(c) + paper_soft(c)


def agree_both_gate(c: Context) -> torch.Tensor:
    agree = agreement(c, 0.1)
    return foundation_hard(c, confidence_weight(c.foundation) * agree) + c.consensus * source_hard(
        c, confidence_weight(c.source) * agree
    )


def agree_both_half(c: Context) -> torch.Tensor:
    agree = agreement(c, 0.5)
    return foundation_hard(c, confidence_weight(c.foundation) * agree) + source_hard(c, confidence_weight(c.source) * agree)


def agree_both_zero(c: Context) -> torch.Tensor:
    """Learn only where the two teachers agree."""
    agree = agreement(c, 0.0).clamp_min(1e-6)
    return foundation_hard(c, confidence_weight(c.foundation) * agree) + source_hard(c, confidence_weight(c.source) * agree)


def per_structure_consensus(c: Context) -> torch.Tensor:
    source = c.source_masks.float()
    foundation = c.foundation_masks.float()
    return ((2.0 * (source * foundation).sum((0, 2, 3)) + 1e-6) / (source.sum((0, 2, 3)) + foundation.sum((0, 2, 3)) + 1e-6))


def per_gate_source_hard(c: Context) -> torch.Tensor:
    """The DINOv3 term per structure, each scaled by that structure's own agreement."""
    weight = confidence_weight(c.source)
    return sum(
        score * weighted_binary_segmentation_loss(
            c.target[:, index : index + 1], c.source_masks[:, index : index + 1], weight[:, index : index + 1]
        )
        for index, score in enumerate(per_structure_consensus(c))
    )


def per_gate_soft(c: Context) -> torch.Tensor:
    return foundation_hard(c) + per_gate_source_hard(c) + paper_soft(c)


def per_gate(c: Context) -> torch.Tensor:
    return foundation_hard(c) + per_gate_source_hard(c)


def per_gate_half_soft(c: Context) -> torch.Tensor:
    return per_gate(c) + 0.5 * paper_soft(c)


def per_gate_distill(c: Context) -> torch.Tensor:
    """The paper's soft term replaced by cross-entropy from the fused teachers."""
    return per_gate(c) + distill(c, nested_foundation_classes(fused_probability(c)))


def per_gate_soft_agree(c: Context) -> torch.Tensor:
    """Hard terms count less where the teachers disagree."""
    agree = agreement(c, 0.5)
    weight = confidence_weight(c.source) * agree
    source = sum(
        score * weighted_binary_segmentation_loss(
            c.target[:, index : index + 1], c.source_masks[:, index : index + 1], weight[:, index : index + 1]
        )
        for index, score in enumerate(per_structure_consensus(c))
    )
    return foundation_hard(c, confidence_weight(c.foundation) * agree) + source + paper_soft(c)


def fused_classes(c: Context) -> torch.Tensor:
    return nested_foundation_classes(fused_probability(c))


def pg_half_distill(c: Context) -> torch.Tensor:
    return per_gate(c) + 0.5 * distill(c, fused_classes(c))


def pg_double_distill(c: Context) -> torch.Tensor:
    return per_gate(c) + 2.0 * distill(c, fused_classes(c))


def distill_only(c: Context) -> torch.Tensor:
    return distill(c, fused_classes(c))


def pg_distill_confident(c: Context) -> torch.Tensor:
    """Distill less from pixels where the fused teachers are unsure."""
    fused = fused_probability(c)
    classes = nested_foundation_classes(fused)
    weight = confidence_weight(fused).mean(dim=1)
    per_pixel = -(classes * c.target_classes.clamp_min(1e-6).log()).sum(dim=1)
    return per_gate(c) + (weight * per_pixel).mean()


LOSSES: dict[str, Candidate] = {
    "agree_lr2e5": Candidate(per_gate_distill, learning_rate=2e-5, freeze_batch_norm=True, scale_lr_by_agreement=True),
    "agree_lr4e5": Candidate(per_gate_distill, learning_rate=4e-5, freeze_batch_norm=True, scale_lr_by_agreement=True),
    "pgd_half_bn_lr2e5":Candidate(pg_half_distill, learning_rate=2e-5, freeze_batch_norm=True),
    "pgd_double_bn_lr2e5": Candidate(pg_double_distill, learning_rate=2e-5, freeze_batch_norm=True),
    "distill_only_bn_lr2e5": Candidate(distill_only, learning_rate=2e-5, freeze_batch_norm=True),
    "pgd_confident_bn_lr2e5": Candidate(pg_distill_confident, learning_rate=2e-5, freeze_batch_norm=True),
    "pgs_bn_lr1e5":Candidate(per_gate_soft, learning_rate=1e-5, freeze_batch_norm=True),
    "pgs_bn_lr5e6": Candidate(per_gate_soft, learning_rate=5e-6, freeze_batch_norm=True),
    "pg_bn_lr2e5": Candidate(per_gate, learning_rate=2e-5, freeze_batch_norm=True),
    "pg_half_bn_lr2e5": Candidate(per_gate_half_soft, learning_rate=2e-5, freeze_batch_norm=True),
    "pg_distill_bn_lr2e5": Candidate(per_gate_distill, learning_rate=2e-5, freeze_batch_norm=True),
    "pgs_agree_bn_lr2e5": Candidate(per_gate_soft_agree, learning_rate=2e-5, freeze_batch_norm=True),
    "baseline": Candidate(baseline),
    "baseline_frozen_bn": Candidate(baseline, freeze_batch_norm=True),
    "per_gate_soft": Candidate(per_gate_soft),
    "per_gate": Candidate(per_gate),
    "pgs_bn": Candidate(per_gate_soft, freeze_batch_norm=True),
    "pgs_bn_lr5e5": Candidate(per_gate_soft, learning_rate=5e-5, freeze_batch_norm=True),
    "pgs_bn_lr2e5": Candidate(per_gate_soft, learning_rate=2e-5, freeze_batch_norm=True),
    "fused_bn_lr5e5": Candidate(pixel_fused, learning_rate=5e-5, freeze_batch_norm=True),
    "fused_bn_lr2e5": Candidate(pixel_fused, learning_rate=2e-5, freeze_batch_norm=True),
    "crop_hard": Candidate(crop_hard),
    "crop_hard_plain": Candidate(crop_hard_plain),
    "crop_soft": Candidate(crop_soft),
    "crop_distill": Candidate(crop_distill),
    "image_gate": Candidate(image_gate),
    "pixel_agree": Candidate(pixel_agree),
    "pixel_agree_both": Candidate(pixel_agree_both),
    "pixel_fused": Candidate(pixel_fused),
    "fused_nogate": Candidate(fused_nogate),
    "fused_distill": Candidate(fused_distill),
    "fused_soft": Candidate(fused_soft),
    "fused_agree": Candidate(fused_agree),
    "agree_both_soft": Candidate(agree_both_soft),
    "agree_both_gate": Candidate(agree_both_gate),
    "agree_both_half": Candidate(agree_both_half),
    "agree_both_zero": Candidate(agree_both_zero),
    "fused_lr3e4": Candidate(pixel_fused, learning_rate=3e-4),
    "fused_lr5e5": Candidate(pixel_fused, learning_rate=5e-5),
}
