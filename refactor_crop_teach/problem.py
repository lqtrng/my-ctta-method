"""The two problems, and where their data and checkpoints are.

A problem's ``image_size`` is the S of ARCHITECTURE.md: every model works in the S x S frame.
Structures are nested: structure k is "class >= k + 1" (OD/OC: disc = rim + cup, then cup).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
CHECKPOINT_DIR = HERE / "checkpoints"
WEIGHTS_DIR = HERE / "weights"
RESULTS_DIR = HERE / "results"


@dataclass(frozen=True)
class Problem:
    name: str
    counts: dict[str, int]           # domain letter -> number of images, checked when loading
    structures: tuple[str, ...]      # nested structure names, outermost first
    image_size: int                  # S

    @property
    def num_classes(self) -> int:
        return len(self.structures) + 1

    @property
    def domains(self) -> list[str]:
        return sorted(self.counts)

    def domain_dir(self, domain: str) -> Path:
        return DATA_DIR / self.name / f"domain_{domain}"

    def checkpoint(self, source: str, kind: str) -> Path:
        """kind: head (SAM head), crop (crop model) or target (whole-image source model)."""
        return CHECKPOINT_DIR / self.name / f"source_{source}_{kind}.pth"


ODOC = Problem("odoc", {"A": 159, "B": 400, "C": 650, "D": 800, "E": 101}, ("disc", "cup"), 512)
PROSTATE = Problem("prostate", {"A": 261, "B": 384, "C": 158, "D": 468, "E": 421, "F": 175}, ("prostate",), 384)
PROBLEMS = {problem.name: problem for problem in (ODOC, PROSTATE)}


def structure_probabilities(classes: torch.Tensor) -> torch.Tensor:
    """Class probabilities (B, C, H, W) -> nested structure probabilities (B, C - 1, H, W)."""
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
