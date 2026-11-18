"""Test images, and the DataLoader interface: a subclass decides the order of the test stream.

    DomainSequenceLoader   domain after domain, in the given order (the paper's protocol)
    ShuffledLoader         every image of every domain, in one seeded random order

Data layout (fetch_assets.py): DATA_DIR/<problem>/domain_<letter>/{Images,Masks}/<name>.*;
masks hold class indices (0 = background). Each image is resized to the problem's square size
(bilinear; masks nearest) and min-max normalized to [0, 1]; the full-size image is kept for
native-resolution crops.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import Problem


@dataclass(frozen=True)
class Sample:
    domain: str
    name: str
    image: Path
    mask: Path


@dataclass
class TestImage:
    domain: str
    name: str
    image: torch.Tensor   # (1, 3, S, S) in [0, 1], resized to the problem's size
    full: torch.Tensor    # (1, 3, H, W) in [0, 1], full size
    mask: torch.Tensor    # (S, S) class indices


def read(sample: Sample, attempts: int = 3) -> tuple[Image.Image, np.ndarray]:
    """Full-size RGB image and its class-index mask (retried: one-off disk errors happen)."""
    for attempt in range(attempts):
        try:
            with Image.open(sample.image) as opened:
                image = opened.convert("RGB")
            with Image.open(sample.mask) as opened:
                mask = np.asarray(opened.convert("L"), dtype=np.uint8).copy()
            return image, mask
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(1)


def to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    """(1, 3, H, W), min-max normalized per image to [0, 1]."""
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    low, high = float(array.min()), float(array.max())
    array = (array - low) / max(high - low, 1e-7)
    return torch.from_numpy(np.ascontiguousarray(array))[None].to(device)


class DataLoader(ABC):
    """Iterates over TestImages of the given domains, in the order the subclass decides."""

    def __init__(self, problem: Problem, data_dir: Path, domains: list[str], max_images: int | None = None,
                 device: torch.device = torch.device("cpu")) -> None:
        self.problem = problem
        self.data_dir = Path(data_dir)
        self.domains = list(domains)
        self.max_images = max_images
        self.device = device

    @abstractmethod
    def order(self) -> list[Sample]:
        """Every sample of the stream, in the order they are given to the method."""

    def domain_samples(self, domain: str) -> list[Sample]:
        """A domain's samples, sorted by file name (the names keep each dataset's order)."""
        folder = self.data_dir / self.problem.name / f"domain_{domain}"
        masks = {path.stem: path for path in (folder / "Masks").iterdir()}
        samples = [Sample(domain, path.stem, path, masks[path.stem]) for path in sorted((folder / "Images").iterdir())]
        expected = self.problem.counts[domain]
        if len(samples) != expected:
            raise RuntimeError(f"{self.problem.name} domain {domain}: expected {expected} images, found {len(samples)}")
        return samples[: self.max_images] if self.max_images else samples

    def load(self, sample: Sample) -> TestImage:
        full, mask = read(sample)
        size = self.problem.image_size
        image = full.resize((size, size), Image.Resampling.BILINEAR)
        mask = np.asarray(Image.fromarray(mask).resize((size, size), Image.Resampling.NEAREST), dtype=np.int64)
        return TestImage(sample.domain, sample.name, to_tensor(image, self.device), to_tensor(full, self.device),
                         torch.from_numpy(mask))

    def __iter__(self) -> Iterator[TestImage]:
        for sample in self.order():
            yield self.load(sample)

    def __len__(self) -> int:
        return len(self.order())


class DomainSequenceLoader(DataLoader):
    """Domain after domain, in the order of ``domains``; each domain in file-name order."""

    def order(self) -> list[Sample]:
        return [sample for domain in self.domains for sample in self.domain_samples(domain)]


class ShuffledLoader(DataLoader):
    """All images of all domains mixed, in a random order fixed by ``seed``."""

    def __init__(self, *args, seed: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed

    def order(self) -> list[Sample]:
        samples = [sample for domain in self.domains for sample in self.domain_samples(domain)]
        random.Random(self.seed).shuffle(samples)
        return samples
