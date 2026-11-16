"""Small stand-ins shared by the tests. Everything runs on the CPU in well under a second."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from models import Model
from problem import Problem


class TinyModel(Model):
    """A trainable Model with a batch-norm layer: one convolution deep."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.out = nn.Conv2d(8, num_classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.out(F.relu(self.bn(self.conv(image)))), dim=1)

    def load(self, checkpoint: Path) -> "TinyModel":
        self.load_state_dict(torch.load(checkpoint)["state_dict"])
        return self


class FixedModel(Model):
    """Always returns the same class probabilities, and records the images it was given."""

    def __init__(self, probabilities: torch.Tensor) -> None:
        super().__init__()
        self.probabilities = probabilities
        self.seen: list[torch.Tensor] = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.seen.append(image)
        return self.probabilities

    def load(self, checkpoint: Path) -> "FixedModel":
        return self


def blob_probabilities(size: int, boxes: list[tuple[int, int, int, int]], num_classes: int = 2,
                       value: float = 0.9) -> torch.Tensor:
    """Class probabilities (1, C, size, size): the last class has ``value`` inside each
    (top, left, height, width) box, background has it elsewhere."""
    classes = torch.zeros(1, num_classes, size, size)
    classes[:, 0] = value
    classes[:, 1:] = (1.0 - value) / (num_classes - 1)
    for top, left, height, width in boxes:
        classes[:, :, top: top + height, left: left + width] = (1.0 - value) / (num_classes - 1)
        classes[:, -1, top: top + height, left: left + width] = value
    return classes


def tiny_problem(root: Path, counts: dict[str, int], image_size: int = 32, shape: tuple[int, int] = (40, 60),
                 structures: tuple[str, ...] = ("disc", "cup")) -> Problem:
    """Writes a small random dataset under root/tiny (non-square images) and returns its Problem.
    The test must also point problem.DATA_DIR at ``root``."""
    generator = np.random.default_rng(0)
    for domain, count in counts.items():
        folder = root / "tiny" / f"domain_{domain}"
        (folder / "Images").mkdir(parents=True)
        (folder / "Masks").mkdir()
        for index in range(count):
            name = f"{index:04d}_{domain.lower()}"
            Image.fromarray(generator.integers(0, 256, (*shape, 3), dtype=np.uint8)).save(folder / "Images" / f"{name}.png")
            mask = generator.integers(0, len(structures) + 1, shape, dtype=np.uint8)
            Image.fromarray(mask).save(folder / "Masks" / f"{name}.png")
    return Problem("tiny", counts, structures, image_size)
