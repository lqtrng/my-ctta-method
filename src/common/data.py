"""OD/OC benchmark data: the source domain A and the B/C/D/E target stream."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from common.paths import DATA_DIR, DRISHTI_ARCHIVE


EXPECTED_COUNTS = {"B": 400, "C": 650, "D": 800, "E": 101}


@dataclass(frozen=True)
class Sample:
    domain: str
    dataset: str
    identifier: str
    image_path: Path | None = None
    mask_path: Path | None = None
    image_member: str | None = None
    mask_member: str | None = None

    @property
    def key(self) -> str:
        return f"{self.domain}/{self.identifier}"


def pair_directories(
    domain: str, dataset: str, image_dir: Path, mask_dir: Path
) -> list[Sample]:
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing image or mask directory: {image_dir}, {mask_dir}")
    masks = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}
    samples: list[Sample] = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        mask_path = masks.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"No mask with stem {image_path.stem!r}")
        samples.append(
            Sample(domain, dataset, image_path.stem, image_path=image_path, mask_path=mask_path)
        )
    return samples


def drishti_samples() -> list[Sample]:
    if not DRISHTI_ARCHIVE.is_file():
        raise FileNotFoundError(f"Drishti-GS archive not found: {DRISHTI_ARCHIVE}")
    with ZipFile(DRISHTI_ARCHIVE) as source:
        members = set(source.namelist())
    samples: list[Sample] = []
    for split in ("training", "testing"):
        prefix = f"Drishti-GS/{split}/images/"
        for image_member in sorted(
            name for name in members if name.startswith(prefix) and not name.endswith("/")
        ):
            filename = Path(image_member).name
            mask_member = f"Drishti-GS/{split}/masks/{filename}"
            if mask_member not in members:
                raise FileNotFoundError(f"No mask for archive member {image_member}")
            samples.append(
                Sample(
                    "E",
                    "Drishti-GS",
                    f"{split}/{Path(filename).stem}",
                    image_member=image_member,
                    mask_member=mask_member,
                )
            )
    return samples


def build_domains(selected: list[str], max_images: int | None) -> dict[str, list[Sample]]:
    refuge_train = DATA_DIR / "domain_B_REFUGE_train/train"
    origa = DATA_DIR / "domain_C_ORIGA"
    refuge_target = DATA_DIR / "domain_D_REFUGE_val_test"
    all_domains = {
        "B": pair_directories(
            "B", "REFUGE", refuge_train / "Images", refuge_train / "Masks"
        ),
        "C": pair_directories("C", "ORIGA", origa / "Images", origa / "Masks"),
        "D": pair_directories(
            "D",
            "REFUGE-Validation/Test",
            refuge_target / "val/Images",
            refuge_target / "val/Masks",
        )
        + pair_directories(
            "D",
            "REFUGE-Validation/Test",
            refuge_target / "test/Images",
            refuge_target / "test/Masks",
        ),
        "E": drishti_samples(),
    }
    for domain, samples in all_domains.items():
        if len(samples) != EXPECTED_COUNTS[domain]:
            raise RuntimeError(
                f"Domain {domain}: expected {EXPECTED_COUNTS[domain]} samples, "
                f"found {len(samples)}"
            )
    return {
        domain: samples[:max_images] if max_images is not None else samples
        for domain, samples in all_domains.items()
        if domain in selected
    }


def _read_archive_image(member: str) -> Image.Image:
    with ZipFile(DRISHTI_ARCHIVE) as archive, archive.open(member) as stream:
        image = Image.open(stream)
        image.load()
    return image


def load_sample(sample: Sample) -> tuple[Image.Image, np.ndarray]:
    if sample.image_path is not None and sample.mask_path is not None:
        with Image.open(sample.image_path) as opened:
            image = opened.convert("RGB")
        with Image.open(sample.mask_path) as opened:
            raw_mask = np.asarray(opened.convert("L"), dtype=np.uint8)
    elif sample.image_member is not None and sample.mask_member is not None:
        image = _read_archive_image(sample.image_member).convert("RGB")
        raw_mask = np.asarray(_read_archive_image(sample.mask_member).convert("L"))
    else:
        raise RuntimeError(f"Incomplete sample: {sample}")

    values = set(np.unique(raw_mask).tolist())
    if values <= {0, 1, 2}:
        target = raw_mask.astype(np.uint8, copy=True)
    else:
        # Benchmark convention: 255=background, 128=disc rim, 0=cup.
        target = np.zeros(raw_mask.shape, dtype=np.uint8)
        target[raw_mask < 255] = 1
        target[raw_mask == 0] = 2
    return image, target


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class DomainADataset(Dataset):
    """RIM-ONE-r3 images paired by filename stem with REFUGE-style masks."""

    def __init__(self, root: Path, image_size: int) -> None:
        self.image_size = (image_size, image_size)
        pairs: list[tuple[Path, Path]] = []

        for split in ("training", "testing"):
            image_dir = root / split / "images"
            mask_dir = root / split / "masks"
            if not image_dir.is_dir() or not mask_dir.is_dir():
                raise FileNotFoundError(f"Missing Domain-A split: {image_dir} or {mask_dir}")

            masks = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}
            for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
                try:
                    mask_path = masks[image_path.stem]
                except KeyError as error:
                    raise FileNotFoundError(f"No mask for {image_path}") from error
                pairs.append((image_path, mask_path))

        if len(pairs) != 159:
            raise RuntimeError(f"Expected 159 Domain-A pairs, found {len(pairs)}")
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize(self.image_size, Image.Resampling.BILINEAR)
        with Image.open(mask_path) as source:
            raw_mask = source.convert("L").resize(self.image_size, Image.Resampling.NEAREST)

        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        minimum = float(image_array.min())
        maximum = float(image_array.max())
        image_array = (image_array - minimum) / max(maximum - minimum, 1e-7)

        # Raw masks: 255=background, 128=optic disc, 0=optic cup.
        raw = np.asarray(raw_mask, dtype=np.uint8)
        target = np.zeros(raw.shape, dtype=np.int16)
        target[raw < 255] = 1
        target[raw == 0] = 2
        return image_array, target[np.newaxis]
