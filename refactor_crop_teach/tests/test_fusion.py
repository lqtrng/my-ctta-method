from __future__ import annotations

import pytest
import torch

from refactor_crop_teach.fusion import (
    AverageFusion,
    ConfidenceWeightedFusion,
    MaxForegroundFusion,
    RefinedRegionFusion,
    build_fusion,
)
from refactor_crop_teach.proposal import CropBox


def _overlapping_regions() -> list[tuple[CropBox, torch.Tensor]]:
    box_a = CropBox(top=4, left=4, side=8)
    probs_a = torch.zeros(1, 2, 8, 8)
    probs_a[:, 0] = 0.4
    probs_a[:, 1] = 0.6

    box_b = CropBox(top=8, left=8, side=8)
    probs_b = torch.zeros(1, 2, 8, 8)
    probs_b[:, 0] = 0.1
    probs_b[:, 1] = 0.9
    return [(box_a, probs_a), (box_b, probs_b)]


def test_fusion_interface_is_abstract() -> None:
    with pytest.raises(TypeError):
        RefinedRegionFusion()


def test_max_foreground_fusion() -> None:
    strategy = MaxForegroundFusion()
    assert isinstance(strategy, RefinedRegionFusion)
    structures, classes = strategy(_overlapping_regions(), image_size=20)
    assert structures.shape == (1, 1, 20, 20)
    assert classes.shape == (1, 2, 20, 20)
    assert torch.isclose(structures[0, 0, 0, 0], torch.tensor(1e-6))
    assert classes[0, :, 0, 0].argmax().item() == 0
    assert torch.isclose(structures[0, 0, 5, 5], torch.tensor(0.6), atol=1e-5)
    assert torch.isclose(structures[0, 0, 10, 10], torch.tensor(0.9), atol=1e-5)


def test_average_fusion() -> None:
    strategy = AverageFusion()
    assert isinstance(strategy, RefinedRegionFusion)
    structures, classes = strategy(_overlapping_regions(), image_size=20)
    assert structures.shape == (1, 1, 20, 20)
    assert classes.shape == (1, 2, 20, 20)
    assert torch.isclose(classes[0, :, 0, 0].sum(), torch.tensor(1.0))
    assert classes[0, :, 0, 0].argmax().item() == 0
    assert torch.isclose(structures[0, 0, 5, 5], torch.tensor(0.6), atol=1e-5)
    assert torch.isclose(structures[0, 0, 10, 10], torch.tensor(0.75), atol=1e-5)


def test_confidence_weighted_fusion() -> None:
    strategy = ConfidenceWeightedFusion()
    assert isinstance(strategy, RefinedRegionFusion)
    structures, classes = strategy(_overlapping_regions(), image_size=20)
    assert structures.shape == (1, 1, 20, 20)
    assert classes.shape == (1, 2, 20, 20)
    overlap_fg = float(structures[0, 0, 10, 10])
    assert 0.75 < overlap_fg < 0.90
    assert classes[0, :, 0, 0].argmax().item() == 0


def test_fusion_three_class_nested_structures() -> None:
    box = CropBox(top=2, left=2, side=6)
    probs = torch.zeros(1, 3, 6, 6)
    probs[:, 0] = 0.2
    probs[:, 1] = 0.3
    probs[:, 2] = 0.5
    for strategy in (MaxForegroundFusion(), AverageFusion(), ConfidenceWeightedFusion()):
        structures, classes = strategy([(box, probs)], image_size=12)
        assert structures.shape == (1, 2, 12, 12)
        assert classes.shape == (1, 3, 12, 12)
        assert torch.isclose(structures[0, 0, 4, 4], torch.tensor(0.8), atol=1e-5)
        assert torch.isclose(structures[0, 1, 4, 4], torch.tensor(0.5), atol=1e-5)
        with pytest.raises(ValueError):
            strategy([], image_size=12)


def test_build_fusion_registry() -> None:
    for name in ("max_foreground", "average", "confidence_weighted"):
        assert isinstance(build_fusion(name), RefinedRegionFusion)
    with pytest.raises(ValueError):
        build_fusion("invalid")
