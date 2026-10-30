#!/usr/bin/env python3
"""Evaluate a source-only ResUNet-34 on the ReDi-FM OD/OC benchmark.

This is the paper's "Source Only" / no-adaptation baseline: the checkpoint is
loaded once, placed in evaluation mode, and never updated while target images
are processed.  With the default Domain-A checkpoint, the target domains are:

* Domain B: REFUGE training set (400 images)
* Domain C: ORIGA (650 images)
* Domain D: REFUGE validation and test sets (800 images)
* Domain E: Drishti-GS training and test sets (101 images)

The paper reports one DSC per source domain ("Do. A", for example), obtained by
averaging over all remaining target domains.  This script additionally reports
optic-disc and optic-cup DSC for every target domain so the aggregate is fully
auditable.

``--task prostate`` evaluates a prostate domain-A model on target domains B-F
of the six-site prostate benchmark, through the shared protocol in
``common/`` (384 x 384, per-image min-max normalization, per-image DSC).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Iterable
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.models import ARCHITECTURES  # noqa: E402
from common.paths import (  # noqa: E402
    DATA_DIR,
    DEFAULT_SOURCE_CHECKPOINT as DEFAULT_CHECKPOINT,
    DRISHTI_ARCHIVE,
    PROSTATE_NNUNET_SOURCE_CHECKPOINT,
    PROSTATE_RESUNET_SOURCE_CHECKPOINT,
    RESULTS_DIR,
)
from common.tasks import TASKS  # noqa: E402


DEFAULT_OUTPUT = RESULTS_DIR / "source_only/baseline_no_adaptation_domain_A.json"
EXPECTED_COUNTS = {"B": 400, "C": 650, "D": 800, "E": 101}


@dataclass(frozen=True)
class Sample:
    identifier: str
    image_path: Path | None = None
    mask_path: Path | None = None
    image_member: str | None = None
    mask_member: str | None = None


@dataclass(frozen=True)
class DomainResult:
    domain: str
    dataset: str
    images: int
    disc_dsc: float
    cup_dsc: float
    mean_dsc: float


def pair_directories(image_dir: Path, mask_dir: Path) -> list[Sample]:
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(f"Missing image or mask directory: {image_dir}, {mask_dir}")
    masks = {path.stem: path for path in mask_dir.iterdir() if path.is_file()}
    samples: list[Sample] = []
    for image_path in sorted(path for path in image_dir.iterdir() if path.is_file()):
        mask_path = masks.get(image_path.stem)
        if mask_path is None:
            raise FileNotFoundError(f"No mask with stem {image_path.stem!r}")
        samples.append(
            Sample(
                identifier=image_path.stem,
                image_path=image_path,
                mask_path=mask_path,
            )
        )
    return samples


def drishti_samples(archive: Path) -> list[Sample]:
    if not archive.is_file():
        raise FileNotFoundError(f"Drishti-GS archive not found: {archive}")
    samples: list[Sample] = []
    with ZipFile(archive) as source:
        members = set(source.namelist())
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
                    identifier=f"{split}/{Path(filename).stem}",
                    image_member=image_member,
                    mask_member=mask_member,
                )
            )
    return samples


def build_domains() -> dict[str, tuple[str, list[Sample]]]:
    refuge_train = DATA_DIR / "domain_B_REFUGE_train/train"
    origa = DATA_DIR / "domain_C_ORIGA"
    refuge_target = DATA_DIR / "domain_D_REFUGE_val_test"
    domains = {
        "B": (
            "REFUGE",
            pair_directories(
                refuge_train / "Images", refuge_train / "Masks"
            ),
        ),
        "C": (
            "ORIGA",
            pair_directories(origa / "Images", origa / "Masks"),
        ),
        "D": (
            "REFUGE-Validation/Test",
            pair_directories(
                refuge_target / "val/Images",
                refuge_target / "val/Masks",
            )
            + pair_directories(
                refuge_target / "test/Images",
                refuge_target / "test/Masks",
            ),
        ),
        "E": ("Drishti-GS", drishti_samples(DRISHTI_ARCHIVE)),
    }
    for domain, (_, samples) in domains.items():
        expected = EXPECTED_COUNTS[domain]
        if len(samples) != expected:
            raise RuntimeError(
                f"Domain {domain}: expected {expected} image/mask pairs, found {len(samples)}"
            )
    return domains


def decode_mask(mask: Image.Image, image_size: int) -> np.ndarray:
    raw = np.asarray(
        mask.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST),
        dtype=np.uint8,
    )
    values = set(np.unique(raw).tolist())
    if values <= {0, 1, 2}:
        return raw.copy()

    # Original benchmark masks use 255=background, 128=disc rim, 0=cup.
    target = np.zeros(raw.shape, dtype=np.uint8)
    target[raw < 255] = 1
    target[raw == 0] = 2
    return target


class TargetDataset(Dataset):
    def __init__(self, samples: Iterable[Sample], image_size: int) -> None:
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def _open_archive_image(self, member: str) -> Image.Image:
        # PIL fully loads the image before the ZipFile and member streams close.
        with ZipFile(DRISHTI_ARCHIVE) as archive:
            with archive.open(member) as stream:
                image = Image.open(stream)
                image.load()
        return image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        sample = self.samples[index]
        if sample.image_path is not None and sample.mask_path is not None:
            image_source: str | Path | BinaryIO = sample.image_path
            mask_source: str | Path | BinaryIO = sample.mask_path
            with Image.open(image_source) as opened:
                image = opened.convert("RGB")
            with Image.open(mask_source) as opened:
                target = decode_mask(opened, self.image_size)
        elif sample.image_member is not None and sample.mask_member is not None:
            image = self._open_archive_image(sample.image_member).convert("RGB")
            target = decode_mask(
                self._open_archive_image(sample.mask_member), self.image_size
            )
        else:
            raise RuntimeError(f"Incomplete sample: {sample}")

        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        minimum = float(array.min())
        maximum = float(array.max())
        array = (array - minimum) / max(maximum - minimum, 1e-7)
        return (
            torch.from_numpy(np.ascontiguousarray(array)),
            torch.from_numpy(np.ascontiguousarray(target)).long(),
            sample.identifier,
        )


def image_dice(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.flatten(1).double()
    target = target.flatten(1).double()
    intersection = (prediction * target).sum(dim=1)
    return (2.0 * intersection + 1e-6) / (
        prediction.sum(dim=1) + target.sum(dim=1) + 1e-6
    )


def evaluate_domain(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    domain: str,
    dataset_name: str,
) -> tuple[DomainResult, list[dict[str, str | float]]]:
    disc_scores: list[float] = []
    cup_scores: list[float] = []
    per_image: list[dict[str, str | float]] = []

    with torch.inference_mode():
        for images, targets, identifiers in loader:
            predictions = model(images.to(device, non_blocking=True)).argmax(dim=1).cpu()
            predicted_disc = predictions >= 1
            predicted_cup = predictions == 2
            target_disc = targets >= 1
            target_cup = targets == 2
            batch_disc = image_dice(predicted_disc, target_disc).tolist()
            batch_cup = image_dice(predicted_cup, target_cup).tolist()
            disc_scores.extend(batch_disc)
            cup_scores.extend(batch_cup)
            per_image.extend(
                {
                    "domain": domain,
                    "image": identifier,
                    "disc_dsc": disc,
                    "cup_dsc": cup,
                    "mean_dsc": (disc + cup) / 2.0,
                }
                for identifier, disc, cup in zip(identifiers, batch_disc, batch_cup)
            )

    disc_mean = float(np.mean(disc_scores))
    cup_mean = float(np.mean(cup_scores))
    result = DomainResult(
        domain=domain,
        dataset=dataset_name,
        images=len(disc_scores),
        disc_dsc=100.0 * disc_mean,
        cup_dsc=100.0 * cup_mean,
        mean_dsc=100.0 * (disc_mean + cup_mean) / 2.0,
    )
    return result, per_image


PROSTATE_CHECKPOINTS = {
    "resunet34": PROSTATE_RESUNET_SOURCE_CHECKPOINT,
    "nnunet": PROSTATE_NNUNET_SOURCE_CHECKPOINT,
}


def evaluate_task_stream(
    model: torch.nn.Module, task_name: str, device: torch.device
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Source-only scoring of a non-OD/OC task through the shared protocol."""
    from common.protocol import image_to_tensor, prepare_sample

    task = TASKS[task_name]
    results: list[dict[str, object]] = []
    per_image: list[dict[str, object]] = []
    with torch.inference_mode():
        for domain, samples in task.build_domains(list(task.target_domains), None).items():
            rows = []
            for sample in samples:
                image, target = prepare_sample(sample, task.image_size)
                prediction = model(image_to_tensor(image, device)).argmax(dim=1)[0].cpu()
                scores = task.score(prediction, target)
                rows.append(scores)
                per_image.append(
                    {
                        "domain": domain,
                        "image": sample.identifier,
                        **scores,
                        "mean_dsc": float(np.mean(list(scores.values()))),
                    }
                )
            results.append(task.domain_summary(domain, samples[0].dataset, rows))
            print(task.describe(results[-1]))
    return results, per_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), default="odoc")
    parser.add_argument("--arch", choices=list(ARCHITECTURES), default="resunet34")
    parser.add_argument("--checkpoint", type=Path,
                        help="Default: the domain-A source checkpoint of --task and --arch")
    parser.add_argument("--output", type=Path,
                        help="Default: results/source_only/[prostate_]baseline_no_adaptation_domain_A.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda when available, otherwise cpu)",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main(args: argparse.Namespace) -> None:
    if args.checkpoint is None:
        args.checkpoint = DEFAULT_CHECKPOINT if args.task == "odoc" else PROSTATE_CHECKPOINTS[args.arch]
    if args.output is None:
        prefix = "" if args.task == "odoc" else f"{args.task}_"
        args.output = DEFAULT_OUTPUT.with_name(f"{prefix}{DEFAULT_OUTPUT.name}")
    if args.batch_size < 1 or args.workers < 0 or args.image_size < 1:
        raise ValueError("batch-size/image-size must be positive; workers cannot be negative")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device(args.device)
    task = TASKS[args.task]
    if args.arch == "nnunet":
        model = ARCHITECTURES[args.arch](task.num_classes, task.image_size)
    else:
        model = ARCHITECTURES[args.arch](num_classes=task.num_classes)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    print(f"Checkpoint: {checkpoint} ({args.arch})")
    print(f"Device: {device}")
    print("Protocol: Source Only (no parameter or BN-statistic updates)")
    if args.task != "odoc":
        print(f"Task: {args.task}; metric: mean per-image DSC")
        task_results, task_per_image = evaluate_task_stream(model, args.task, device)
        do_a = float(np.mean([result["mean_dsc"] for result in task_results]))
        print(f"Do. A DSC (unweighted {'/'.join(task.target_domains)} domain mean): {do_a:.2f}")
        payload = {
            "protocol": "source_only",
            "task": args.task,
            "source_domain": "A",
            "arch": args.arch,
            "checkpoint": str(checkpoint),
            "image_size": task.image_size,
            "target_domains": task_results,
            "do_a_dsc": do_a,
            # 0140.pdf Table 1, Source Only, source domain A.
            "paper_source_only_do_a_reference": 51.02,
            "per_image": task_per_image,
        }
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {output}")
        return

    print("Metric: mean per-image DSC; OD includes the OC region")

    results: list[DomainResult] = []
    per_image: list[dict[str, str | float]] = []
    for domain, (dataset_name, samples) in build_domains().items():
        loader = DataLoader(
            TargetDataset(samples, args.image_size),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
            persistent_workers=args.workers > 0,
        )
        result, image_results = evaluate_domain(
            model, loader, device, domain, dataset_name
        )
        results.append(result)
        per_image.extend(image_results)
        print(
            f"Domain {domain} ({dataset_name}, n={result.images}): "
            f"OD={result.disc_dsc:.2f}, OC={result.cup_dsc:.2f}, "
            f"Mean={result.mean_dsc:.2f}"
        )

    do_a = float(np.mean([result.mean_dsc for result in results]))
    pooled_image_mean = float(np.mean([row["mean_dsc"] for row in per_image])) * 100.0
    print(f"Do. A DSC (unweighted B/C/D/E domain mean): {do_a:.2f}")
    print(f"Pooled target-image DSC (diagnostic only): {pooled_image_mean:.2f}")

    payload = {
        "protocol": "source_only",
        "source_domain": "A",
        "arch": args.arch,
        "checkpoint": str(checkpoint),
        "image_size": args.image_size,
        "target_domains": [asdict(result) for result in results],
        "do_a_dsc": do_a,
        "pooled_target_image_dsc": pooled_image_mean,
        "paper_source_only_do_a_reference": 53.59,
        "per_image": per_image,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main(parse_args())
