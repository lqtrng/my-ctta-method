"""Learning interface: everything after the two supervisions exist.

A Learning owns the augmentation, the student forward pass, the loss, the optimizer and the
batch-norm freezing. The model it updates is owned by Adapting and passed to ``step``.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

from augmentation import Augmentation, JitterErase
from models import Model
from problem import nested_classes, structure_probabilities


class Learning(ABC):
    @abstractmethod
    def step(self, model: Model, image: torch.Tensor, sup1: torch.Tensor, sup2: torch.Tensor) -> dict:
        """One update of ``model`` on ``image`` (1, 3, S, S).
        sup1: the full-image model's class probabilities (1, C, S, S);
        sup2: the fused crop predictions, nested structure probabilities (1, K, S, S).
        Returns {"loss": float, "consensus": float}."""


class CropTeachLearning(Learning):
    """The ReDi-FM paper's loss (Eqs. 6-16), with supervision_1 from the SAM head and
    supervision_2 from the crop model:

        L = Lhard(crop mask, Phi(crop)) + r * Lhard(SAM-head mask, Phi(SAM head)) + Lsoft

    One Adam step per image over all the model's parameters, batch-norm statistics frozen.
    """

    LEARNING_RATE = 2e-5
    PSEUDO_THRESHOLD = 0.5      # tau: the SAM head's binary mask
    CROP_THRESHOLD = 0.5        # the crop model's binary mask
    ENTROPY_THRESHOLD = 0.5     # "high uncertainty" in the soft loss
    GAMMA, MU, SIGMA2 = 1.3, 0.5, 0.1   # the confidence weight Phi

    def __init__(self, augmentation: Augmentation | None = None) -> None:
        self.augmentation = augmentation or JitterErase()
        self.optimizer: torch.optim.Adam | None = None
        self._model: Model | None = None

    def step(self, model: Model, image: torch.Tensor, sup1: torch.Tensor, sup2: torch.Tensor) -> dict:
        if self.optimizer is None:  # Created once: its state is kept across the whole stream.
            self._model = model
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.LEARNING_RATE)
        if model is not self._model:
            raise ValueError("A CropTeachLearning holds the optimizer state of one model only")
        model.train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()  # Batches of one image would make these statistics noisy.
        student = model(self.augmentation(image))
        loss, consensus = self.loss(student, sup1, sup2)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach()), "consensus": float(consensus)}

    def loss(self, student: torch.Tensor, sup1: torch.Tensor, sup2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(loss, consensus r). student and sup1: class probabilities; sup2: structure probabilities."""
        head = structure_probabilities(sup1)
        head_mask, crop_mask = head > self.PSEUDO_THRESHOLD, sup2 > self.CROP_THRESHOLD
        consensus = self.mask_dice(head_mask, crop_mask).detach()
        prediction = structure_probabilities(student)
        loss = self.hard_loss(prediction, crop_mask, self.confidence_weight(sup2))
        loss = loss + consensus * self.hard_loss(prediction, head_mask, self.confidence_weight(head))
        loss = loss + self.soft_loss(student, sup1, nested_classes(sup2), consensus)
        return loss, consensus

    def confidence_weight(self, probability: torch.Tensor) -> torch.Tensor:
        """Eq. 6: about 0.04 at p = 0.5, about 0.94 at p = 0 or 1."""
        gaussian = torch.exp(-((probability - self.MU) ** 2) / (2.0 * self.SIGMA2)) / math.sqrt(2.0 * math.pi * self.SIGMA2)
        return (self.GAMMA - gaussian).clamp_min(1e-6)

    @staticmethod
    def hard_loss(prediction: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Eqs. 7-8: confidence-weighted BCE + Dice against a teacher's binary mask."""
        prediction = prediction.clamp(1e-6, 1.0 - 1e-6)
        mask = mask.float()
        bce = (F.binary_cross_entropy(prediction, mask, reduction="none") * weight).mean()
        dimensions = (0, 2, 3)
        intersection = (weight * prediction * mask).sum(dimensions)
        denominator = (weight * prediction).sum(dimensions) + (weight * mask).sum(dimensions)
        return bce + 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()

    @staticmethod
    def mask_dice(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        """Eq. 13: Dice of the two teachers' masks, averaged over structures."""
        dimensions = (0, 2, 3)
        first, second = first.float(), second.float()
        intersection = (first * second).sum(dimensions)
        return ((2.0 * intersection + 1e-6) / (first.sum(dimensions) + second.sum(dimensions) + 1e-6)).mean()

    def soft_loss(self, student: torch.Tensor, head: torch.Tensor, crop: torch.Tensor,
                  consensus: torch.Tensor) -> torch.Tensor:
        """Eqs. 10-12 with the annealing of Eq. 15: class-balanced soft distillation from the crop
        model. All three arguments are class probabilities."""
        batch, classes, height, width = student.shape
        head_labels = head.argmax(dim=1)
        student_labels = student.detach().argmax(dim=1)
        crop_labels = crop.argmax(dim=1)
        entropy = -(head.clamp_min(1e-6) * head.clamp_min(1e-6).log()).sum(dim=1)
        uncertain = entropy / math.log(classes) > self.ENTROPY_THRESHOLD
        disagreement = (student_labels != crop_labels).sum()
        weights = []
        for index in range(classes):
            eta = ((crop_labels == index).sum() + disagreement).float() / (batch * 2.0 * height * width)
            difficult = (head_labels == index) | uncertain
            alpha = torch.where(difficult, eta, 1.0 - eta)
            weights.append(consensus * alpha + (1.0 - consensus))
        alpha = torch.stack(weights, dim=1)
        return -(alpha * student * crop.clamp_min(1e-6).log()).sum(dim=1).mean()
