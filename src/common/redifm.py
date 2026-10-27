"""ReDi-FM components independent of the foundation model (paper Section 3).

Structure-aware prompts (Equations 2-4), confidence weights (6), weighted hard
distillation (7-8), hard class-balanced soft distillation (10-12) and the
consensus gate (13-15).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage


STRUCTURES = ("disc", "cup")


@dataclass(frozen=True)
class StepDiagnostics:
    source_hard_loss: float
    foundation_hard_loss: float
    soft_loss: float
    total_loss: float
    consensus: float


def structure_probabilities(class_probabilities: torch.Tensor) -> torch.Tensor:
    """Convert [background, disc-rim, cup] into nested [OD, OC] probabilities."""
    disc = class_probabilities[:, 1] + class_probabilities[:, 2]
    cup = class_probabilities[:, 2]
    # Softmax channels can sum to 1 + a few ULPs, while CUDA BCE requires its
    # input to be strictly inside the probability interval.
    return torch.stack((disc, cup), dim=1).clamp(1e-6, 1.0 - 1e-6)


def largest_component(mask: np.ndarray) -> np.ndarray:
    components, count = ndimage.label(mask.astype(bool))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, components, range(1, count + 1))
    return components == int(np.argmax(sizes) + 1)


def structure_aware_points(
    image: Image.Image,
    pseudo_mask: np.ndarray,
    soft_probability: np.ndarray,
) -> np.ndarray:
    """Equations (2)-(4): component centroid plus maximum intensity deviation."""
    component = largest_component(pseudo_mask)
    if not component.any():
        # A foreground anatomy is known to exist in OD/OC images. If thresholding
        # removes it, use the source probability maximum as a one-pixel anchor.
        row, column = np.unravel_index(np.argmax(soft_probability), soft_probability.shape)
        component[row, column] = True

    rows, columns = np.nonzero(component)
    center_x = int(np.clip(np.rint(columns.mean()), 0, component.shape[1] - 1))
    center_y = int(np.clip(np.rint(rows.mean()), 0, component.shape[0] - 1))

    grayscale = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    prototype = float(grayscale[component].mean())
    deviations = np.abs(grayscale[rows, columns] - prototype)
    hard_index = int(np.argmax(deviations))
    hard_x = int(columns[hard_index])
    hard_y = int(rows[hard_index])
    return np.asarray([[center_x, center_y], [hard_x, hard_y]], dtype=np.float32)


def nested_foundation_classes(structure_probabilities_: torch.Tensor) -> torch.Tensor:
    """Map independent SAM OD/OC predictions to background/disc-rim/cup.

    The cup probability is constrained to lie inside the disc probability,
    matching the benchmark's nested label convention.
    """
    disc = structure_probabilities_[:, 0].clamp(1e-6, 1.0 - 1e-6)
    cup = (structure_probabilities_[:, 1] * disc).clamp(1e-6, 1.0 - 1e-6)
    background = 1.0 - disc
    disc_rim = (disc - cup).clamp_min(1e-6)
    classes = torch.stack((background, disc_rim, cup), dim=1)
    return classes / classes.sum(dim=1, keepdim=True).clamp_min(1e-6)


def confidence_weight(
    probability: torch.Tensor,
    gamma: float = 1.3,
    mean: float = 0.5,
    variance: float = 0.1,
) -> torch.Tensor:
    """Equation (6), with sigma^2=variance."""
    sigma = math.sqrt(variance)
    gaussian = torch.exp(-((probability - mean) ** 2) / (2.0 * variance))
    gaussian = gaussian / (math.sqrt(2.0 * math.pi) * sigma)
    return (gamma - gaussian).clamp_min(1e-6)


def weighted_binary_segmentation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted BCE plus weighted Dice used by Equations (7)-(8)."""
    prediction = prediction.clamp(1e-6, 1.0 - 1e-6)
    target = target.float()
    bce = (F.binary_cross_entropy(prediction, target, reduction="none") * weight).mean()
    dimensions = (0, 2, 3)
    intersection = (weight * prediction * target).sum(dimensions)
    denominator = (weight * prediction).sum(dimensions) + (weight * target).sum(dimensions)
    dice_loss = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()
    return bce + dice_loss


def consensus_score(source_mask: torch.Tensor, foundation_mask: torch.Tensor) -> torch.Tensor:
    dimensions = (0, 2, 3)
    source = source_mask.float()
    foundation = foundation_mask.float()
    intersection = (source * foundation).sum(dimensions)
    denominator = source.sum(dimensions) + foundation.sum(dimensions)
    return ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def hard_class_balanced_soft_loss(
    target_probabilities: torch.Tensor,
    source_probabilities: torch.Tensor,
    foundation_probabilities: torch.Tensor,
    consensus: torch.Tensor,
    entropy_threshold: float,
) -> torch.Tensor:
    """Equations (10)-(12) and consensus annealing from Equation (15)."""
    batch, classes, height, width = target_probabilities.shape
    source_labels = source_probabilities.argmax(dim=1)
    target_labels = target_probabilities.detach().argmax(dim=1)
    foundation_labels = foundation_probabilities.argmax(dim=1)

    normalized_entropy = -(
        source_probabilities.clamp_min(1e-6)
        * source_probabilities.clamp_min(1e-6).log()
    ).sum(dim=1) / math.log(classes)
    high_uncertainty = normalized_entropy > entropy_threshold

    weights: list[torch.Tensor] = []
    pixel_count = float(2 * height * width)
    for class_index in range(classes):
        foundation_class = foundation_labels == class_index
        disagreement = target_labels != foundation_labels
        eta = (foundation_class.sum() + disagreement.sum()).float() / (batch * pixel_count)
        difficult = (source_labels == class_index) | high_uncertainty
        alpha = torch.where(difficult, eta, 1.0 - eta)
        alpha = consensus * alpha + (1.0 - consensus)
        weights.append(alpha)
    adaptive_weights = torch.stack(weights, dim=1)

    # Equation (12) is intentionally implemented in the paper's written
    # direction: -alpha * p_target * log(p_foundation).
    return -(
        adaptive_weights
        * target_probabilities
        * foundation_probabilities.clamp_min(1e-6).log()
    ).sum(dim=1).mean()
