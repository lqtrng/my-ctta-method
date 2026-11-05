"""The ReDi-FM paper's loss (Eqs. 6-16), with the crop model in SAM's place.

    L = Lhard(crop mask, Phi(crop)) + r * Lhard(DINOv3 mask, Phi(DINOv3)) + Lsoft

* Phi (Eq. 6): confidence weight, gamma = 1.3, mu = 0.5, sigma^2 = 0.1 (the paper, after Yang et al. 2024).
* Lhard (Eqs. 7-8): confidence-weighted BCE + Dice against a teacher's binary mask.
* r (Eq. 13): Dice of the two teachers' masks, gating the DINOv3 (source) term (Eq. 14).
* Lsoft (Eqs. 10-12, 15): hard class-balanced soft distillation, annealed by r.

Structures are nested: OD/OC gives (disc = rim + cup, cup); prostate and polyp one foreground.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def structure_probabilities(classes: torch.Tensor) -> torch.Tensor:
    """Class probabilities -> nested structure probabilities (B, K, H, W)."""
    if classes.shape[1] == 2:
        return classes[:, 1:2].clamp(1e-6, 1.0 - 1e-6)
    disc = classes[:, 1] + classes[:, 2]
    return torch.stack((disc, classes[:, 2]), dim=1).clamp(1e-6, 1.0 - 1e-6)


def nested_classes(structures: torch.Tensor) -> torch.Tensor:
    """Nested structure probabilities -> class probabilities (the inverse of the above)."""
    if structures.shape[1] == 1:
        foreground = structures[:, 0].clamp(1e-6, 1.0 - 1e-6)
        return torch.stack((1.0 - foreground, foreground), dim=1)
    disc = structures[:, 0].clamp(1e-6, 1.0 - 1e-6)
    cup = (structures[:, 1] * disc).clamp(1e-6, 1.0 - 1e-6)
    classes = torch.stack((1.0 - disc, (disc - cup).clamp_min(1e-6), cup), dim=1)
    return classes / classes.sum(dim=1, keepdim=True).clamp_min(1e-6)


def confidence_weight(probability: torch.Tensor, gamma: float = 1.3, mean: float = 0.5,
                      variance: float = 0.1) -> torch.Tensor:
    """Eq. 6: about 0.04 at p = 0.5, about 0.94 at p = 0 or 1."""
    gaussian = torch.exp(-((probability - mean) ** 2) / (2.0 * variance)) / math.sqrt(2.0 * math.pi * variance)
    return (gamma - gaussian).clamp_min(1e-6)


def hard_loss(prediction: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Eqs. 7-8: weighted BCE + weighted Dice."""
    prediction = prediction.clamp(1e-6, 1.0 - 1e-6)
    mask = mask.float()
    bce = (F.binary_cross_entropy(prediction, mask, reduction="none") * weight).mean()
    dimensions = (0, 2, 3)
    intersection = (weight * prediction * mask).sum(dimensions)
    denominator = (weight * prediction).sum(dimensions) + (weight * mask).sum(dimensions)
    return bce + 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def mask_dice(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Eq. 13: Dice of two masks, averaged over structures."""
    dimensions = (0, 2, 3)
    first, second = first.float(), second.float()
    intersection = (first * second).sum(dimensions)
    return ((2.0 * intersection + 1e-6) / (first.sum(dimensions) + second.sum(dimensions) + 1e-6)).mean()


def soft_loss(target_classes: torch.Tensor, source_classes: torch.Tensor, foundation_classes: torch.Tensor,
              consensus: torch.Tensor, entropy_threshold: float) -> torch.Tensor:
    """Eqs. 10-12 with the annealing of Eq. 15, in the paper's written direction."""
    batch, classes, height, width = target_classes.shape
    source_labels = source_classes.argmax(dim=1)
    target_labels = target_classes.detach().argmax(dim=1)
    foundation_labels = foundation_classes.argmax(dim=1)
    entropy = -(source_classes.clamp_min(1e-6) * source_classes.clamp_min(1e-6).log()).sum(dim=1)
    uncertain = entropy / math.log(classes) > entropy_threshold
    disagreement = (target_labels != foundation_labels).sum()
    weights = []
    for index in range(classes):
        eta = ((foundation_labels == index).sum() + disagreement).float() / (batch * 2.0 * height * width)
        difficult = (source_labels == index) | uncertain
        alpha = torch.where(difficult, eta, 1.0 - eta)
        weights.append(consensus * alpha + (1.0 - consensus))
    alpha = torch.stack(weights, dim=1)
    return -(alpha * target_classes * foundation_classes.clamp_min(1e-6).log()).sum(dim=1).mean()


def crop_teach_loss(target_classes: torch.Tensor, source_classes: torch.Tensor, crop: torch.Tensor,
                    threshold: float, entropy_threshold: float) -> tuple[torch.Tensor, float]:
    """The full loss. source_classes: DINOv3 head's class probabilities; crop: the crop model's
    nested structure probabilities (background outside the crops). Returns (loss, consensus r)."""
    source = structure_probabilities(source_classes)
    source_mask, crop_mask = source > threshold, crop > 0.5
    consensus = mask_dice(source_mask, crop_mask).detach()
    target = structure_probabilities(target_classes)
    loss = hard_loss(target, crop_mask, confidence_weight(crop))
    loss = loss + consensus * hard_loss(target, source_mask, confidence_weight(source))
    loss = loss + soft_loss(target_classes, source_classes, nested_classes(crop), consensus, entropy_threshold)
    return loss, float(consensus)
