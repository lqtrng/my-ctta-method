from __future__ import annotations

import pytest
import torch
from torch import nn

from refactor_crop_teach.augmentation import IdentityAugmentation
from refactor_crop_teach.learning import (
    CropOnlyLearning,
    FusedDistillLearning,
    Learning,
    ReDiFMLearning,
    build_learning,
)
from refactor_crop_teach.models import Model


class TinySegModel(Model):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self._num_classes = num_classes
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(8)
        self.head = nn.Conv2d(8, num_classes, kernel_size=1)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.bn(self.conv(images))))


def _supervisions(num_classes: int = 2, size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    coarse = torch.softmax(torch.randn(1, num_classes, size, size), dim=1)
    num_structures = num_classes - 1
    crop_structures = torch.rand(1, num_structures, size, size).clamp(1e-6, 1.0 - 1e-6)
    return coarse, crop_structures


def test_learning_interface_is_abstract() -> None:
    with pytest.raises(TypeError):
        Learning()


@pytest.mark.parametrize(
    "strategy_cls",
    [ReDiFMLearning, CropOnlyLearning, FusedDistillLearning],
)
def test_learning_strategies_update_target_and_freeze_bn(strategy_cls: type[Learning]) -> None:
    torch.manual_seed(0)
    model = TinySegModel(num_classes=3)
    learner = strategy_cls(
        augmentation=IdentityAugmentation(),
        learning_rate=1e-2,
        freeze_batch_norm=True,
    )
    assert isinstance(learner, Learning)
    before_weight = model.head.weight.detach().clone()
    before_bn_mean = model.bn.running_mean.detach().clone()

    image = torch.rand(1, 3, 32, 32)
    coarse, crop = _supervisions(num_classes=3, size=32)
    metrics = learner.step(model, image, coarse, crop)

    assert "loss" in metrics and "consensus" in metrics
    assert metrics["loss"] >= 0.0
    assert 0.0 <= metrics["consensus"] <= 1.0
    assert not torch.equal(model.head.weight, before_weight)
    assert torch.equal(model.bn.running_mean, before_bn_mean)


def test_redifm_learning_formula_properties() -> None:
    torch.manual_seed(123)
    target_classes = torch.softmax(torch.randn(1, 3, 24, 24), dim=1)
    coarse_classes = torch.softmax(torch.randn(1, 3, 24, 24), dim=1)
    crop_structures = torch.rand(1, 2, 24, 24).clamp(1e-6, 1.0 - 1e-6)

    learner = ReDiFMLearning(pseudo_threshold=0.5, entropy_threshold=0.5)
    loss, metrics = learner.compute_loss(target_classes, coarse_classes, crop_structures)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["consensus"] <= 1.0

    identical_structures = torch.stack(
        (coarse_classes[:, 1] + coarse_classes[:, 2], coarse_classes[:, 2]), dim=1
    ).clamp(1e-6, 1.0 - 1e-6)
    _, agree_metrics = learner.compute_loss(target_classes, coarse_classes, identical_structures)
    assert agree_metrics["consensus"] == pytest.approx(1.0, abs=1e-5)


def test_build_learning_registry() -> None:
    for name in ("redifm", "crop_only", "fused_distill"):
        assert isinstance(build_learning(name), Learning)
    with pytest.raises(ValueError):
        build_learning("invalid")
