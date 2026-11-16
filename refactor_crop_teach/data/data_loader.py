"""DataLoader: the order of the test images.

Layout, for each problem and domain letter:
    data/<problem>/domain_<letter>/Images/<name>.<jpg|png|...>   full-size image
    data/<problem>/domain_<letter>/Masks/<name>.png              class indices (0 = background)

The base class reads and prepares each image and repeats the stream ``rounds`` times. A subclass
defines only its single pass (``domains`` and ``one_pass``) and its ``rounds``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from core.data_types import TestImage
from core.problem import Problem


@dataclass(frozen=True)
class Sample:
    domain: str
    name: str
    image: Path
    mask: Path


class DataLoader(ABC):
    rounds = 1
    max_images: int | None = None   # per domain; only lowered for smoke tests

    def __init__(self, problem: Problem, source: str, device: torch.device | str = "cuda") -> None:
        if source not in problem.counts:
            raise ValueError(f"{problem.name} has no domain {source}")
        self.problem = problem
        self.source = source
        self.device = torch.device(device)

    @abstractmethod
    def domains(self) -> list[str]:
        """The domains of one pass, in stream order."""

    @abstractmethod
    def one_pass(self) -> Iterator[Sample]:
        """The samples of one pass over the stream, in order."""

    def __iter__(self) -> Iterator[TestImage]:
        for round_index in range(1, self.rounds + 1):
            for sample in self.one_pass():
                yield self.prepare(sample, round_index)

    def __len__(self) -> int:
        return self.rounds * sum(len(self.samples(domain)) for domain in self.domains())

    def samples(self, domain: str) -> list[Sample]:
        """A domain's samples, sorted by file name."""
        folder = self.problem.domain_dir(domain)
        masks = {path.stem: path for path in (folder / "Masks").iterdir()}
        samples = [Sample(domain, path.stem, path, masks[path.stem])
                   for path in sorted((folder / "Images").iterdir())]
        expected = self.problem.counts[domain]
        if len(samples) != expected:
            raise RuntimeError(f"{self.problem.name} domain {domain}: expected {expected} images, found {len(samples)}")
        return samples[:self.max_images] if self.max_images else samples

    def prepare(self, sample: Sample, round_index: int) -> TestImage:
        """Image and label are stretched to S x S (bilinear / nearest), never padded; the image
        and the full-resolution image are each min-max normalized to [0, 1]."""
        size = self.problem.image_size
        full, mask = self.read(sample)
        image = full.resize((size, size), Image.Resampling.BILINEAR)
        label = np.asarray(Image.fromarray(mask).resize((size, size), Image.Resampling.NEAREST), dtype=np.int64)
        return TestImage(sample.domain, sample.name, round_index, self.to_tensor(image), self.to_tensor(full),
                         torch.from_numpy(label))

    @staticmethod
    def read(sample: Sample, attempts: int = 3) -> tuple[Image.Image, np.ndarray]:
        """Full-size RGB image and its class-index mask. Retried: a one-off read error must not
        end a run of many hours."""
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

    def to_tensor(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        low, high = float(array.min()), float(array.max())
        array = (array - low) / max(high - low, 1e-7)
        return torch.from_numpy(np.ascontiguousarray(array))[None].to(self.device)


class AlphabeticalStream(DataLoader):
    """The paper's protocol: every domain but the source, in alphabetical order, one pass."""

    rounds = 1

    def domains(self) -> list[str]:
        return [domain for domain in self.problem.domains if domain != self.source]

    def one_pass(self) -> Iterator[Sample]:
        for domain in self.domains():
            yield from self.samples(domain)


class LongTermStream(DataLoader):
    """The same order as the paper's protocol, repeated 3 times."""

    rounds = 3

    def domains(self) -> list[str]:
        return [domain for domain in self.problem.domains if domain != self.source]

    def one_pass(self) -> Iterator[Sample]:
        for domain in self.domains():
            yield from self.samples(domain)
