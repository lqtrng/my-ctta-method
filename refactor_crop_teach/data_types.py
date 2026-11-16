"""The two data classes passed between components."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TestImage:
    """One image of the test stream. ``label`` is for the evaluators only: it is called label,
    not target, because "target" already means the target model and the target domain."""

    __test__ = False  # Not a pytest test class, despite the name.

    domain: str
    name: str
    round: int
    image: torch.Tensor   # 1 x 3 x S x S, stretched, in [0, 1]            -> the models
    full: torch.Tensor    # 1 x 3 x H x W, original resolution, in [0, 1]  -> RegionProposal
    label: torch.Tensor   # S x S class indices (int64), nearest-neighbor  -> evaluators only


@dataclass(frozen=True)
class Region:
    crop: torch.Tensor                 # 1 x 3 x S x S, cut from full resolution and resized
    box: tuple[int, int, int, int]     # (top, left, height, width) in the S frame
