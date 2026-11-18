from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from .config import DATA_DIR, PROBLEMS, Problem
except ImportError:
    from config import DATA_DIR, PROBLEMS, Problem


@dataclass(frozen=True)
class Sample:
    domain: str
    name: str
    image: Path
    mask: Path


@dataclass(frozen=True)
class TestItem:
    domain: str
    name: str
    image: torch.Tensor
    full_image: torch.Tensor
    target: torch.Tensor


def resolve_problem(problem: str | Problem) -> Problem:
    if isinstance(problem, Problem):
        return problem
    if problem not in PROBLEMS:
        raise ValueError(f"Unknown problem {problem!r}; expected one of {sorted(PROBLEMS)}")
    return PROBLEMS[problem]


def domain_samples(
    problem: str | Problem,
    domain: str,
    max_images: int | None = None,
    data_dir: Path = DATA_DIR,
    verify_count: bool = True,
) -> list[Sample]:
    spec = resolve_problem(problem)
    folder = data_dir / spec.name / f"domain_{domain}"
    masks = {path.stem: path for path in (folder / "Masks").iterdir() if path.is_file()}
    samples = [
        Sample(domain, path.stem, path, masks[path.stem])
        for path in sorted((folder / "Images").iterdir())
        if path.is_file()
    ]
    if verify_count and domain in spec.counts:
        expected = spec.counts[domain]
        if len(samples) != expected:
            raise RuntimeError(
                f"{spec.name} domain {domain}: expected {expected} images, found {len(samples)}"
            )
    return samples[:max_images] if max_images else samples


def read_sample(sample: Sample, attempts: int = 3) -> tuple[Image.Image, np.ndarray]:
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
    raise RuntimeError(f"Failed to read {sample.image}")


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask).resize((size, size), Image.Resampling.NEAREST),
        dtype=np.int64,
    )


def normalized(image: Image.Image) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    low, high = float(array.min()), float(array.max())
    return (array - low) / max(high - low, 1e-7)


def to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(normalized(image)))[None].to(device)


class DataLoader(ABC):
    def __init__(
        self,
        problem: str | Problem,
        source: str,
        device: torch.device | str = "cpu",
        max_images: int | None = None,
        data_dir: Path = DATA_DIR,
        verify_count: bool = True,
    ) -> None:
        self.problem = resolve_problem(problem)
        if source not in self.problem.domains:
            raise ValueError(f"{self.problem.name} has no source domain {source!r}")
        self.source = source
        self.device = torch.device(device)
        self.max_images = max_images
        self.data_dir = data_dir
        self.verify_count = verify_count
        self._domain_samples = {
            domain: domain_samples(
                self.problem,
                domain,
                max_images=max_images,
                data_dir=data_dir,
                verify_count=verify_count,
            )
            for domain in self.problem.stream(source)
        }
        self._ordered_samples = self.order_samples(self._domain_samples)

    @abstractmethod
    def order_samples(self, per_domain: dict[str, list[Sample]]) -> list[Sample]:
        raise NotImplementedError

    @property
    def samples(self) -> list[Sample]:
        return list(self._ordered_samples)

    @property
    def domains(self) -> list[str]:
        seen: list[str] = []
        for sample in self._ordered_samples:
            if sample.domain not in seen:
                seen.append(sample.domain)
        return seen

    def __len__(self) -> int:
        return len(self._ordered_samples)

    def __iter__(self) -> Iterator[TestItem]:
        size = self.problem.image_size
        for sample in self._ordered_samples:
            full, mask = read_sample(sample)
            resized = full.resize((size, size), Image.Resampling.BILINEAR)
            target = torch.from_numpy(resize_mask(mask, size))
            yield TestItem(
                domain=sample.domain,
                name=sample.name,
                image=to_tensor(resized, self.device),
                full_image=to_tensor(full, self.device),
                target=target,
            )


class AlphabeticalDomainDataLoader(DataLoader):
    def order_samples(self, per_domain: dict[str, list[Sample]]) -> list[Sample]:
        ordered: list[Sample] = []
        for domain in sorted(per_domain):
            ordered.extend(per_domain[domain])
        return ordered


class CustomDomainOrderDataLoader(DataLoader):
    def __init__(
        self,
        problem: str | Problem,
        source: str,
        domain_order: Sequence[str],
        device: torch.device | str = "cpu",
        max_images: int | None = None,
        data_dir: Path = DATA_DIR,
        verify_count: bool = True,
    ) -> None:
        self.domain_order = list(domain_order)
        super().__init__(
            problem=problem,
            source=source,
            device=device,
            max_images=max_images,
            data_dir=data_dir,
            verify_count=verify_count,
        )

    def order_samples(self, per_domain: dict[str, list[Sample]]) -> list[Sample]:
        if not self.domain_order:
            raise ValueError("domain_order must not be empty")
        unknown = set(self.domain_order) - set(per_domain)
        if unknown:
            raise ValueError(
                f"Invalid target domain(s) {sorted(unknown)}; available: {sorted(per_domain)}"
            )
        ordered: list[Sample] = []
        for domain in self.domain_order:
            ordered.extend(per_domain[domain])
        return ordered


class ReversedStreamDataLoader(DataLoader):
    def order_samples(self, per_domain: dict[str, list[Sample]]) -> list[Sample]:
        ordered: list[Sample] = []
        for domain in reversed(sorted(per_domain)):
            ordered.extend(reversed(per_domain[domain]))
        return ordered


class ShuffledStreamDataLoader(DataLoader):
    def __init__(
        self,
        problem: str | Problem,
        source: str,
        shuffle_seed: int = 0,
        device: torch.device | str = "cpu",
        max_images: int | None = None,
        data_dir: Path = DATA_DIR,
        verify_count: bool = True,
    ) -> None:
        self.shuffle_seed = shuffle_seed
        super().__init__(
            problem=problem,
            source=source,
            device=device,
            max_images=max_images,
            data_dir=data_dir,
            verify_count=verify_count,
        )

    def order_samples(self, per_domain: dict[str, list[Sample]]) -> list[Sample]:
        ordered = [sample for domain in sorted(per_domain) for sample in per_domain[domain]]
        rng = random.Random(self.shuffle_seed)
        rng.shuffle(ordered)
        return ordered


DATALOADERS: dict[str, type[DataLoader]] = {
    "alphabetical": AlphabeticalDomainDataLoader,
    "custom": CustomDomainOrderDataLoader,
    "reversed": ReversedStreamDataLoader,
    "shuffled": ShuffledStreamDataLoader,
}
