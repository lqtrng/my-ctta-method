"""Benchmark data: OD/OC (domains A-E; source A, target stream B-E by default), prostate MRI (domains A-F) and polyp (domains A-D)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from common.paths import DATA_DIR, DOMAIN_A_DIR, DRISHTI_ARCHIVE, POLYP_DIR, PROSTATE_DIR


EXPECTED_COUNTS = {"A": 159, "B": 400, "C": 650, "D": 800, "E": 101}
# Six-site prostate MRI (0140.pdf, Section 4.1): slices that contain prostate,
# 384 x 384, laid out by download/prepare_prostate.py.
PROSTATE_SITES = {"A": "BIDMC", "B": "BMC", "C": "HK", "D": "I2CVB", "E": "RUNMC", "F": "UCL"}
PROSTATE_EXPECTED_COUNTS = {"A": 261, "B": 384, "C": 158, "D": 468, "E": 421, "F": 175}
PROSTATE_IMAGE_SIZE = 384
# Four polyp datasets (0140.pdf, Section 4.1), laid out by download/prepare_polyp.py.
POLYP_DATASETS = {"A": "BKAI_IGH_NeoPolyp", "B": "CVC_ClinicDB", "C": "ETIS_LaribPolypDB", "D": "Kvasir_SEG"}
POLYP_EXPECTED_COUNTS = {"A": 1000, "B": 612, "C": 196, "D": 1000}


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
        # RIM-ONE-r3, both splits: a target when another domain is the source.
        "A": [
            sample
            for split in ("training", "testing")
            for sample in pair_directories(
                "A", "RIM-ONE-r3", DOMAIN_A_DIR / split / "images", DOMAIN_A_DIR / split / "masks"
            )
        ],
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
    # In the order given: the target stream follows ``selected``.
    return {
        domain: all_domains[domain][:max_images] if max_images is not None else all_domains[domain]
        for domain in selected
    }


def build_prostate_domains(selected: list[str], max_images: int | None) -> dict[str, list[Sample]]:
    """Prostate domains in stream order; slices are ordered by case, then slice index.

    Masks are class-index PNGs (0 background, 1 prostate), so ``load_sample``
    returns them unchanged.
    """
    domains: dict[str, list[Sample]] = {}
    for domain in selected:  # The target stream follows the order given.
        site = PROSTATE_SITES[domain]
        folder = PROSTATE_DIR / f"domain_{domain}_{site}"
        samples = pair_directories(domain, site, folder / "Images", folder / "Masks")
        if len(samples) != PROSTATE_EXPECTED_COUNTS[domain]:
            raise RuntimeError(
                f"Prostate domain {domain}: expected {PROSTATE_EXPECTED_COUNTS[domain]} "
                f"slices, found {len(samples)}"
            )
        domains[domain] = samples[:max_images] if max_images is not None else samples
    return domains


def build_polyp_domains(selected: list[str], max_images: int | None) -> dict[str, list[Sample]]:
    """Polyp domains in stream order; masks are class-index PNGs (0 background, 1 polyp)."""
    domains: dict[str, list[Sample]] = {}
    for domain in selected:  # The target stream follows the order given.
        dataset = POLYP_DATASETS[domain]
        folder = POLYP_DIR / f"domain_{domain}_{dataset}"
        samples = pair_directories(domain, dataset, folder / "Images", folder / "Masks")
        if len(samples) != POLYP_EXPECTED_COUNTS[domain]:
            raise RuntimeError(
                f"Polyp domain {domain}: expected {POLYP_EXPECTED_COUNTS[domain]} "
                f"images, found {len(samples)}"
            )
        domains[domain] = samples[:max_images] if max_images is not None else samples
    return domains


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

    def __init__(self, root: Path, image_size: int, original: bool = False) -> None:
        self.image_size = (image_size, image_size)
        self.original = original
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
            full = source.convert("RGB")
        image = full.resize(self.image_size, Image.Resampling.BILINEAR)
        with Image.open(mask_path) as source:
            raw_mask = source.convert("L").resize(self.image_size, Image.Resampling.NEAREST)
        image_array = normalized_array(image)

        # Raw masks: 255=background, 128=optic disc, 0=optic cup.
        raw = np.asarray(raw_mask, dtype=np.uint8)
        target = np.zeros(raw.shape, dtype=np.int16)
        target[raw < 255] = 1
        target[raw == 0] = 2
        if self.original:
            return image_array, target[np.newaxis], normalized_array(full)
        return image_array, target[np.newaxis]


def normalized_array(image: Image.Image) -> np.ndarray:
    """Channels-first float array, min-max normalized per image to [0, 1]."""
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    minimum = float(array.min())
    maximum = float(array.max())
    return (array - minimum) / max(maximum - minimum, 1e-7)


class SampleDataset(Dataset):
    """Labeled samples in the ``DomainADataset`` format, for source training.

    Images are resized (bilinear; masks nearest) and min-max normalized per
    image to [0, 1]; targets are class indices with a leading channel axis.
    """

    def __init__(self, samples: list[Sample], image_size: int, original: bool = False) -> None:
        self.samples = samples
        self.image_size = (image_size, image_size)
        self.original = original

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        full, target = load_sample(self.samples[index])
        image = full.resize(self.image_size, Image.Resampling.BILINEAR)
        target = np.asarray(
            Image.fromarray(target).resize(self.image_size, Image.Resampling.NEAREST), dtype=np.int16
        )
        if self.original:
            return normalized_array(image), target[np.newaxis], normalized_array(full)
        return normalized_array(image), target[np.newaxis]
