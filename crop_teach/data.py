"""Reading the prepared data.

Layout (written by prepare_data.py), for each problem and domain letter:
    DATA_DIR/<problem>/domain_<letter>/Images/<name>.<jpg|png|...>   full-size image
    DATA_DIR/<problem>/domain_<letter>/Masks/<name>.png              class indices (0 = background)

Every image is resized to the problem's square size (bilinear; masks nearest) and min-max
normalized per image to [0, 1]. The full-size image is kept for native-res crops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import DATA_DIR, PROBLEMS


@dataclass(frozen=True)
class Sample:
    domain: str
    name: str
    image: Path
    mask: Path


def domain_samples(problem: str, domain: str, max_images: int | None = None) -> list[Sample]:
    folder = DATA_DIR / problem / f"domain_{domain}"
    masks = {path.stem: path for path in (folder / "Masks").iterdir()}
    samples = [Sample(domain, path.stem, path, masks[path.stem])
               for path in sorted((folder / "Images").iterdir())]
    expected = PROBLEMS[problem].counts[domain]
    if len(samples) != expected:
        raise RuntimeError(f"{problem} domain {domain}: expected {expected} images, found {len(samples)}")
    return samples[:max_images] if max_images else samples


def read(sample: Sample, attempts: int = 3) -> tuple[Image.Image, np.ndarray]:
    """Full-size RGB image and its class-index mask. Retried: a one-off read error must not end a
    run of many hours (one happened once on the local disk)."""
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


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(mask).resize((size, size), Image.Resampling.NEAREST), dtype=np.int64)


def normalized(image: Image.Image) -> np.ndarray:
    """Channels-first float array, min-max normalized per image to [0, 1]."""
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    low, high = float(array.min()), float(array.max())
    return (array - low) / max(high - low, 1e-7)


def to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(normalized(image)))[None].to(device)


def test_sample(sample: Sample, size: int) -> tuple[Image.Image, Image.Image, torch.Tensor]:
    """(resized image, full-size image, resized class-index mask) for the evaluation stream."""
    full, mask = read(sample)
    image = full.resize((size, size), Image.Resampling.BILINEAR)
    return image, full, torch.from_numpy(resize_mask(mask, size))


class SourceDataset(Dataset):
    """Labeled source-domain images: (image, mask[1, H, W]) or, with full=True, also the
    full-size image (for native-res crops)."""

    def __init__(self, problem: str, source: str, full: bool = False, max_images: int | None = None) -> None:
        self.samples = domain_samples(problem, source, max_images)
        self.size = PROBLEMS[problem].image_size
        self.full = full

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        full, mask = read(self.samples[index])
        image = normalized(full.resize((self.size, self.size), Image.Resampling.BILINEAR))
        target = resize_mask(mask, self.size)[None].astype(np.int16)
        if self.full:
            return image, target, normalized(full)
        return image, target
