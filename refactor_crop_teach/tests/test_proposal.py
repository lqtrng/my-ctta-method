from __future__ import annotations

import pytest
import torch

from refactor_crop_teach.proposal import (
    AllRegionsProposal,
    CropBox,
    LargestRegionProposal,
    MostConfidentRegionProposal,
    RegionProposal,
    WholeImageProposal,
    build_proposal,
)


def _two_region_map(size: int = 128) -> torch.Tensor:
    probs = torch.full((1, 1, size, size), 0.05)
    probs[0, 0, 10:30, 10:30] = 0.70
    probs[0, 0, 80:90, 80:90] = 0.95
    return probs


def test_region_proposal_interface_is_abstract() -> None:
    with pytest.raises(TypeError):
        RegionProposal()


def test_all_regions_proposal() -> None:
    strategy = AllRegionsProposal(threshold=0.5, crop_scale=2.0, min_crop=16)
    assert isinstance(strategy, RegionProposal)
    boxes = strategy(_two_region_map(128))
    assert len(boxes) == 2
    assert all(isinstance(b, CropBox) for b in boxes)
    sides = sorted(b.side for b in boxes)
    assert sides == [20, 40]


def test_all_regions_proposal_empty_mask_fallback() -> None:
    strategy = AllRegionsProposal(threshold=0.5, crop_scale=2.0, min_crop=24)
    probs = torch.full((1, 1, 64, 64), 0.1)
    probs[0, 0, 32, 40] = 0.4
    boxes = strategy(probs)
    assert len(boxes) == 1
    assert boxes[0].side == 24
    assert boxes[0].top <= 32 < boxes[0].top + boxes[0].side
    assert boxes[0].left <= 40 < boxes[0].left + boxes[0].side


def test_largest_region_proposal() -> None:
    strategy = LargestRegionProposal(threshold=0.5, crop_scale=2.0, min_crop=16)
    assert isinstance(strategy, RegionProposal)
    boxes = strategy(_two_region_map(128))
    assert len(boxes) == 1
    assert boxes[0].side == 40
    assert boxes[0].top == 0
    assert boxes[0].left == 0


def test_most_confident_region_proposal() -> None:
    strategy = MostConfidentRegionProposal(threshold=0.5, crop_scale=2.0, min_crop=16)
    assert isinstance(strategy, RegionProposal)
    boxes = strategy(_two_region_map(128))
    assert len(boxes) == 1
    assert boxes[0].side == 20
    assert boxes[0].top == 74
    assert boxes[0].left == 74


def test_whole_image_proposal() -> None:
    strategy = WholeImageProposal()
    assert isinstance(strategy, RegionProposal)
    boxes = strategy(_two_region_map(96))
    assert boxes == [CropBox(top=0, left=0, side=96)]


def test_build_proposal_registry() -> None:
    for name in ("all", "largest", "confident", "whole"):
        assert isinstance(build_proposal(name), RegionProposal)
    with pytest.raises(ValueError):
        build_proposal("invalid")
