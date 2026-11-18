from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

try:
    from .augmentation import Augmentation, ColorJitterRandomErasingAugmentation
    from .config import ENTROPY_THRESHOLD, LEARNING_RATE, PSEUDO_THRESHOLD
    from .models import Model
    from .structures import nested_classes, structure_probabilities
except ImportError:
    from augmentation import Augmentation, ColorJitterRandomErasingAugmentation
    from config import ENTROPY_THRESHOLD, LEARNING_RATE, PSEUDO_THRESHOLD
    from models import Model
    from structures import nested_classes, structure_probabilities


def confidence_weight(
    probability: torch.Tensor,
    gamma: float = 1.3,
    mean: float = 0.5,
    variance: float = 0.1,
) -> torch.Tensor:
    gaussian = torch.exp(-((probability - mean) ** 2) / (2.0 * variance)) / math.sqrt(
        2.0 * math.pi * variance
    )
    return (gamma - gaussian).clamp_min(1e-6)


def hard_loss(prediction: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    prediction = prediction.clamp(1e-6, 1.0 - 1e-6)
    mask = mask.float()
    bce = (F.binary_cross_entropy(prediction, mask, reduction="none") * weight).mean()
    dimensions = (0, 2, 3)
    intersection = (weight * prediction * mask).sum(dimensions)
    denominator = (weight * prediction).sum(dimensions) + (weight * mask).sum(dimensions)
    return bce + 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def mask_dice(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    dimensions = (0, 2, 3)
    first, second = first.float(), second.float()
    intersection = (first * second).sum(dimensions)
    return ((2.0 * intersection + 1e-6) / (first.sum(dimensions) + second.sum(dimensions) + 1e-6)).mean()


def soft_loss(
    target_classes: torch.Tensor,
    source_classes: torch.Tensor,
    foundation_classes: torch.Tensor,
    consensus: torch.Tensor,
    entropy_threshold: float,
) -> torch.Tensor:
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


class Learning(ABC):
    def __init__(
        self,
        augmentation: Augmentation | None = None,
        learning_rate: float = LEARNING_RATE,
        freeze_batch_norm: bool = True,
    ) -> None:
        self.augmentation = ColorJitterRandomErasingAugmentation() if augmentation is None else augmentation
        self.learning_rate = learning_rate
        self.freeze_batch_norm = freeze_batch_norm
        self.optimizer: torch.optim.Optimizer | None = None
        self._attached_model: Model | None = None

    def attach(self, target_model: Model) -> None:
        trainable = [param for param in target_model.parameters() if param.requires_grad]
        if not trainable:
            raise ValueError("target_model has no trainable parameters")
        self.optimizer = torch.optim.Adam(trainable, lr=self.learning_rate)
        self._attached_model = target_model

    @abstractmethod
    def compute_loss(
        self,
        target_classes: torch.Tensor,
        coarse_classes: torch.Tensor,
        crop_structures: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def step(
        self,
        target_model: Model,
        image: torch.Tensor,
        coarse_classes: torch.Tensor,
        crop_structures: torch.Tensor,
    ) -> dict[str, float]:
        if self._attached_model is not target_model or self.optimizer is None:
            self.attach(target_model)
        assert self.optimizer is not None
        target_model.set_adaptation_train_mode(freeze_batch_norm=self.freeze_batch_norm)
        augmented = self.augmentation(image)
        target_classes = target_model.forward_probabilities(augmented)
        loss, metrics = self.compute_loss(target_classes, coarse_classes, crop_structures)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.detach()), **metrics}


class ReDiFMLearning(Learning):
    def __init__(
        self,
        augmentation: Augmentation | None = None,
        learning_rate: float = LEARNING_RATE,
        pseudo_threshold: float = PSEUDO_THRESHOLD,
        entropy_threshold: float = ENTROPY_THRESHOLD,
        freeze_batch_norm: bool = True,
    ) -> None:
        super().__init__(
            augmentation=augmentation,
            learning_rate=learning_rate,
            freeze_batch_norm=freeze_batch_norm,
        )
        self.pseudo_threshold = pseudo_threshold
        self.entropy_threshold = entropy_threshold

    def compute_loss(
        self,
        target_classes: torch.Tensor,
        coarse_classes: torch.Tensor,
        crop_structures: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        source = structure_probabilities(coarse_classes)
        source_mask = source > self.pseudo_threshold
        crop_mask = crop_structures > 0.5
        consensus = mask_dice(source_mask, crop_mask).detach()
        target = structure_probabilities(target_classes)
        loss = hard_loss(target, crop_mask, confidence_weight(crop_structures))
        loss = loss + consensus * hard_loss(target, source_mask, confidence_weight(source))
        loss = loss + soft_loss(
            target_classes,
            coarse_classes,
            nested_classes(crop_structures),
            consensus,
            self.entropy_threshold,
        )
        return loss, {"consensus": float(consensus)}


class CropOnlyLearning(Learning):
    def __init__(
        self,
        augmentation: Augmentation | None = None,
        learning_rate: float = LEARNING_RATE,
        pseudo_threshold: float = PSEUDO_THRESHOLD,
        freeze_batch_norm: bool = True,
    ) -> None:
        super().__init__(
            augmentation=augmentation,
            learning_rate=learning_rate,
            freeze_batch_norm=freeze_batch_norm,
        )
        self.pseudo_threshold = pseudo_threshold

    def compute_loss(
        self,
        target_classes: torch.Tensor,
        coarse_classes: torch.Tensor,
        crop_structures: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        source = structure_probabilities(coarse_classes)
        consensus = mask_dice(source > self.pseudo_threshold, crop_structures > 0.5).detach()
        target = structure_probabilities(target_classes)
        loss = hard_loss(target, crop_structures > 0.5, confidence_weight(crop_structures))
        return loss, {"consensus": float(consensus)}


class FusedDistillLearning(Learning):
    def __init__(
        self,
        augmentation: Augmentation | None = None,
        learning_rate: float = LEARNING_RATE,
        pseudo_threshold: float = PSEUDO_THRESHOLD,
        freeze_batch_norm: bool = True,
    ) -> None:
        super().__init__(
            augmentation=augmentation,
            learning_rate=learning_rate,
            freeze_batch_norm=freeze_batch_norm,
        )
        self.pseudo_threshold = pseudo_threshold

    def compute_loss(
        self,
        target_classes: torch.Tensor,
        coarse_classes: torch.Tensor,
        crop_structures: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        source = structure_probabilities(coarse_classes)
        source_mask = source > self.pseudo_threshold
        crop_mask = crop_structures > 0.5
        consensus = mask_dice(source_mask, crop_mask).detach()
        target = structure_probabilities(target_classes)
        source_weight = confidence_weight(source)
        crop_weight = confidence_weight(crop_structures)
        loss = hard_loss(target, crop_mask, crop_weight)
        for index in range(target.shape[1]):
            keep = slice(index, index + 1)
            agreement = mask_dice(source_mask[:, keep], crop_mask[:, keep]).detach()
            loss = loss + agreement * hard_loss(
                target[:, keep], source_mask[:, keep], source_weight[:, keep]
            )
        scaled_source_weight = consensus * source_weight
        fused = (crop_weight * crop_structures + scaled_source_weight * source) / (
            crop_weight + scaled_source_weight
        ).clamp_min(1e-6)
        fused_classes = nested_classes(fused.clamp(1e-6, 1.0 - 1e-6))
        distill = -(fused_classes * target_classes.clamp_min(1e-6).log()).sum(dim=1).mean()
        return loss + distill, {"consensus": float(consensus)}


LEARNING_STRATEGIES: dict[str, type[Learning]] = {
    "redifm": ReDiFMLearning,
    "crop_only": CropOnlyLearning,
    "fused_distill": FusedDistillLearning,
}


def build_learning(name: str = "redifm", **kwargs) -> Learning:
    if name not in LEARNING_STRATEGIES:
        raise ValueError(f"Unknown learning strategy {name!r}; expected one of {sorted(LEARNING_STRATEGIES)}")
    return LEARNING_STRATEGIES[name](**kwargs)
