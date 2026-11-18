"""Learning interface: everything after the two supervisions exist. It updates the target model
on the (augmented) test image.

    ReDiFMLearning   one Adam step on the ReDi-FM paper's loss (Eqs. 6-16), with the refined crop
                     prediction in place of SAM's prompted mask:

        L = Lhard(refined mask, Phi(refined)) + r * Lhard(foundation mask, Phi(foundation)) + Lsoft

    Phi (Eq. 6): confidence weight, gamma = 1.3, mu = 0.5, sigma^2 = 0.1.
    Lhard (Eqs. 7-8): confidence-weighted BCE + Dice against a teacher's binary mask.
    r (Eq. 13): Dice of the two teachers' masks; it gates the foundation term (Eq. 14).
    Lsoft (Eqs. 10-12, 15): hard class-balanced soft distillation, annealed by r.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from method.structures import EPS, nested_classes, structure_probabilities


@dataclass
class Supervision:
    """The two teachers' predictions for one test image."""

    foundation: torch.Tensor  # (1, C, H, W) class probabilities of the foundation model
    refined: torch.Tensor     # (1, C, H, W) fused class probabilities of the refined crops, 0 outside
    inside: torch.Tensor      # (1, 1, H, W) 1 where some crop covers the pixel

    @property
    def refined_structures(self) -> torch.Tensor:
        """Nested structure probabilities of the refined prediction, background outside the crops."""
        return (structure_probabilities(self.refined) * self.inside).clamp(EPS, 1 - EPS)


class Learning(ABC):
    @abstractmethod
    def attach(self, model: nn.Module) -> None:
        """Called once with the target model (e.g. to build the optimizer)."""

    @abstractmethod
    def learn(self, model: nn.Module, image: torch.Tensor, supervision: Supervision) -> dict[str, float]:
        """Update ``model`` on ``image`` with ``supervision``; returns statistics to log."""


def freeze_batch_norm(model: nn.Module) -> None:
    """Train mode, but batch-norm statistics fixed: batches of one image would make them noisy."""
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


class ReDiFMLearning(Learning):
    def __init__(self, learning_rate: float, pseudo_threshold: float, entropy_threshold: float) -> None:
        self.learning_rate = learning_rate
        self.pseudo_threshold = pseudo_threshold
        self.entropy_threshold = entropy_threshold
        self.optimizer: torch.optim.Optimizer | None = None

    def attach(self, model: nn.Module) -> None:
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

    def learn(self, model: nn.Module, image: torch.Tensor, supervision: Supervision) -> dict[str, float]:
        freeze_batch_norm(model)
        target = F.softmax(model(image), dim=1)
        loss, consensus = redifm_loss(target, supervision.foundation, supervision.refined_structures,
                                      self.pseudo_threshold, self.entropy_threshold)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"consensus": consensus, "loss": loss.item()}


# --- The loss --------------------------------------------------------------------------


def confidence_weight(probability: torch.Tensor, gamma: float = 1.3, mean: float = 0.5,
                      variance: float = 0.1) -> torch.Tensor:
    """Eq. 6: about 0.04 at p = 0.5, about 0.94 at p = 0 or 1."""
    gaussian = torch.exp(-((probability - mean) ** 2) / (2.0 * variance)) / math.sqrt(2.0 * math.pi * variance)
    return (gamma - gaussian).clamp_min(EPS)


def hard_loss(prediction: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Eqs. 7-8: weighted BCE + weighted Dice."""
    prediction = prediction.clamp(EPS, 1.0 - EPS)
    mask = mask.float()
    bce = (F.binary_cross_entropy(prediction, mask, reduction="none") * weight).mean()
    dimensions = (0, 2, 3)
    intersection = (weight * prediction * mask).sum(dimensions)
    denominator = (weight * prediction).sum(dimensions) + (weight * mask).sum(dimensions)
    return bce + 1.0 - ((2.0 * intersection + EPS) / (denominator + EPS)).mean()


def mask_dice(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Eq. 13: Dice of two masks, averaged over structures."""
    dimensions = (0, 2, 3)
    first, second = first.float(), second.float()
    intersection = (first * second).sum(dimensions)
    return ((2.0 * intersection + EPS) / (first.sum(dimensions) + second.sum(dimensions) + EPS)).mean()


def soft_loss(target_classes: torch.Tensor, source_classes: torch.Tensor, teacher_classes: torch.Tensor,
              consensus: torch.Tensor, entropy_threshold: float) -> torch.Tensor:
    """Eqs. 10-12 with the annealing of Eq. 15, in the paper's written direction."""
    batch, classes, height, width = target_classes.shape
    source_labels = source_classes.argmax(dim=1)
    target_labels = target_classes.detach().argmax(dim=1)
    teacher_labels = teacher_classes.argmax(dim=1)
    entropy = -(source_classes.clamp_min(EPS) * source_classes.clamp_min(EPS).log()).sum(dim=1)
    uncertain = entropy / math.log(classes) > entropy_threshold
    disagreement = (target_labels != teacher_labels).sum()
    weights = []
    for index in range(classes):
        eta = ((teacher_labels == index).sum() + disagreement).float() / (batch * 2.0 * height * width)
        difficult = (source_labels == index) | uncertain
        alpha = torch.where(difficult, eta, 1.0 - eta)
        weights.append(consensus * alpha + (1.0 - consensus))
    alpha = torch.stack(weights, dim=1)
    return -(alpha * target_classes * teacher_classes.clamp_min(EPS).log()).sum(dim=1).mean()


def redifm_loss(target_classes: torch.Tensor, foundation_classes: torch.Tensor, refined: torch.Tensor,
                threshold: float, entropy_threshold: float) -> tuple[torch.Tensor, float]:
    """target_classes, foundation_classes: class probabilities; refined: nested structure
    probabilities of the refined crops. Returns (loss, consensus r)."""
    foundation = structure_probabilities(foundation_classes)
    foundation_mask, refined_mask = foundation > threshold, refined > 0.5
    consensus = mask_dice(foundation_mask, refined_mask).detach()
    target = structure_probabilities(target_classes)
    loss = hard_loss(target, refined_mask, confidence_weight(refined))
    loss = loss + consensus * hard_loss(target, foundation_mask, confidence_weight(foundation))
    loss = loss + soft_loss(target_classes, foundation_classes, nested_classes(refined), consensus, entropy_threshold)
    return loss, float(consensus)
