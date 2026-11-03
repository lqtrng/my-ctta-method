#!/usr/bin/env python3
"""Train nnU-Net's 2D network on domain A (prostate or polyp) with nnU-Net v2's own default recipe.

    python src/training/train_nnunet.py [--task polyp] [--epochs 100] [--batch-size 12] [--resume]
    python src/evaluate.py --method source_only --task prostate --arch nnunet_official

``--crop`` trains on crops around the prostate enlarged to 384 x 384, with the
jittered scale and center of ``train_dinov3_head.py``, for
``methods/redifm_dinov3_nnunet_crop.py``.  ``--crop-from original`` cuts those crops from
the full-size image instead of the resized one (polyp: domain A is about 1280 x 1000).

Unlike ``train_source.py``, which applies one shared recipe to every network,
this follows ``nnUNetTrainer`` and ``default_normalization_schemes`` of
https://github.com/MIC-DKFZ/nnUNet:

* per-image z-score normalization, part of the model (``common.models``);
* SGD, lr=0.01, momentum=0.99, Nesterov, weight decay=3e-5, gradient norm 12;
* polynomial decay with power 0.9, epochs of 250 iterations, mixed precision;
* soft Dice (foreground, over the batch) + cross-entropy with deep supervision;
* augmentation: rotation up to 180 degrees and scaling 0.7-1.4 (p=0.2 each), noise,
  blur, brightness, contrast, low resolution, gamma and mirroring.

What differs from nnU-Net: 100 epochs instead of 1000 (about 1.5 hours on one
GPU); the three channels repeat one grayscale slice, so
intensity transforms treat them alike; the patch is the whole 384 x 384 slice,
so there is no foreground oversampling; and every domain-A slice is trained on,
as for every source model in this project, instead of cross-validation folds.

The training state is saved every 50 epochs next to the output, for ``--resume``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform_2
from torch.utils.data import DataLoader, RandomSampler

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.models import build_network, zscore  # noqa: E402
from common.paths import (  # noqa: E402
    POLYP_NNUNET_OFFICIAL_SOURCE_CHECKPOINT,
    POLYP_NNUNET_SHARP_CROP_CHECKPOINT,
    PROSTATE_NNUNET_CROP_CHECKPOINT,
    PROSTATE_NNUNET_OFFICIAL_SOURCE_CHECKPOINT,
)
from common.protocol import seed_everything  # noqa: E402
from common.tasks import TASKS  # noqa: E402
from common.tracking import log_metrics, tracked_run  # noqa: E402
from training.train_dinov3_head import enlarged_crop, enlarged_original_crop  # noqa: E402
from training.train_source import nnunet_intensity_transforms  # noqa: E402


ITERATIONS_PER_EPOCH = 250
# Default output of each (task, --crop, --crop-from).
OUTPUTS = {
    ("prostate", False, "resized"): PROSTATE_NNUNET_OFFICIAL_SOURCE_CHECKPOINT,
    ("prostate", True, "resized"): PROSTATE_NNUNET_CROP_CHECKPOINT,
    ("polyp", False, "resized"): POLYP_NNUNET_OFFICIAL_SOURCE_CHECKPOINT,
    ("polyp", True, "original"): POLYP_NNUNET_SHARP_CROP_CHECKPOINT,
}
SAVE_EVERY = 50


def build_augmentation(image_size: int) -> Compose:
    return Compose(
        [
            SpatialTransform_2(
                (image_size, image_size),
                [image_size // 2, image_size // 2],
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=(-np.pi, np.pi),
                angle_y=(0, 0),
                angle_z=(0, 0),
                do_scale=True,
                scale=(0.7, 1.4),
                border_mode_data="constant",
                border_cval_data=0,
                border_mode_seg="constant",
                border_cval_seg=0,
                order_seg=0,
                order_data=3,
                random_crop=False,
                p_rot_per_sample=0.2,
                p_scale_per_sample=0.2,
            ),
            GaussianNoiseTransform(noise_variance=(0, 0.1), p_per_sample=0.1),
            *nnunet_intensity_transforms(),
            MirrorTransform(axes=(0, 1)),
        ]
    )


class AugmentCollate:
    def __init__(self, image_size: int, crop: bool, crop_from: str = "resized") -> None:
        self.image_size = image_size
        self.crop = crop
        self.crop_from = crop_from
        self.transform = build_augmentation(image_size)

    def __call__(self, batch: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.crop and self.crop_from == "original":
            batch = [enlarged_original_crop(image, mask, full, self.image_size) for image, mask, full in batch]
        elif self.crop:
            batch = [enlarged_crop(image, mask, self.image_size) for image, mask in batch]
        images, masks = zip(*batch)
        # The augmentation works on z-scored images, as in nnU-Net.
        normalized = zscore(torch.from_numpy(np.stack(images))).numpy()
        transformed = self.transform(data=normalized, seg=np.stack(masks).astype(np.float32))
        return (
            torch.from_numpy(np.ascontiguousarray(transformed["data"])).float(),
            torch.from_numpy(np.ascontiguousarray(transformed["seg"][:, 0])).long(),
        )


def dice_cross_entropy(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """nnU-Net's loss: cross-entropy minus soft Dice of the foreground classes over the batch."""
    probabilities = logits.softmax(dim=1)[:, 1:]
    labels = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2)[:, 1:].float()
    intersection = (probabilities * labels).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + labels.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + smooth) / (denominator + smooth).clamp_min(1e-8)
    return F.cross_entropy(logits, target) - dice.mean()


def deep_supervision_loss(outputs: list[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
    """Each lower resolution counts half; the lowest is left out, as in nnU-Net."""
    weights = np.array([0.5 ** level for level in range(len(outputs))])
    weights[-1] = 0.0
    weights /= weights.sum()
    total = 0.0
    for weight, logits in zip(weights, outputs):
        if weight == 0.0:
            continue
        label = F.interpolate(target[:, None].float(), size=logits.shape[-2:], mode="nearest")[:, 0].long()
        total = total + float(weight) * dice_cross_entropy(logits.float(), label)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=("prostate", "polyp"), default="prostate")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--crop", action="store_true", help="Train on enlarged crops around the structure")
    parser.add_argument("--crop-from", choices=("resized", "original"), default="resized",
                        help="With --crop: cut the crops from the resized image or the full-size one")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue from the state saved next to --output")
    args = parser.parse_args()
    if args.output is None:
        args.output = OUTPUTS[(args.task, args.crop, args.crop_from if args.crop else "resized")]
    return args


def main(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {args.output}")
    state_path = args.output.with_suffix(".state.pth")
    if state_path.exists() and not args.resume:
        raise FileExistsError(f"Earlier run found; continue it with --resume: {state_path}")
    seed_everything(args.seed)
    device = torch.device("cuda")
    task = TASKS[args.task]

    original = args.crop and args.crop_from == "original"
    dataset = task.source_dataset(original=original)
    # Full-size images are too large to hold in memory; those are read per batch.
    samples = dataset if original else [dataset[index] for index in range(len(dataset))]
    loader = DataLoader(
        samples,
        batch_size=args.batch_size,
        sampler=RandomSampler(samples, replacement=True, num_samples=args.batch_size * ITERATIONS_PER_EPOCH),
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=AugmentCollate(task.image_size, args.crop, args.crop_from),
        persistent_workers=args.workers > 0,
        drop_last=True,
        # Forked workers can inherit a held lock and hang forever, so start clean
        # ones, and fail instead of waiting silently if a worker ever stalls.
        multiprocessing_context="spawn" if args.workers > 0 else None,
        timeout=600 if args.workers > 0 else 0,
    )

    model = build_network("nnunet_official", task.num_classes, task.image_size).to(device)
    network = model.network  # Skips the model's z-score: the batches are already normalized.
    network.decoder.deep_supervision = True
    optimizer = torch.optim.SGD(network.parameters(), lr=0.01, momentum=0.99, weight_decay=3e-5, nesterov=True)
    scaler = torch.amp.GradScaler("cuda")
    first_epoch = 0
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        first_epoch = state["epoch"] + 1
        print(f"Resuming after epoch {first_epoch}")

    print(f"Task {task.name}; crops from the {args.crop_from} image" if args.crop else f"Task {task.name}")
    print(f"Domain A: {len(samples)} images; batch size {args.batch_size}; {args.epochs} epochs "
          f"of {ITERATIONS_PER_EPOCH} iterations; output {args.output}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tracked_run("train_nnunet_crop" if args.crop else "train_nnunet_official", args, task=task.name):
        network.train()
        for epoch in range(first_epoch, args.epochs):
            start = time.perf_counter()
            learning_rate = 0.01 * (1 - epoch / args.epochs) ** 0.9
            optimizer.param_groups[0]["lr"] = learning_rate
            losses = []
            for images, targets in loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda"):
                    outputs = network(images)
                loss = deep_supervision_loss(outputs, targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
            print(f"Epoch {epoch + 1:04d}/{args.epochs}: loss={np.mean(losses):.4f}, lr={learning_rate:.6f}, "
                  f"{time.perf_counter() - start:.0f} s", flush=True)
            log_metrics({"train_loss": float(np.mean(losses)), "lr": learning_rate}, step=epoch + 1)
            if (epoch + 1) % SAVE_EVERY == 0:
                temporary = state_path.with_suffix(".tmp")
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch},
                    temporary,
                )
                temporary.replace(state_path)

        temporary = args.output.with_suffix(".tmp")
        torch.save(model.state_dict(), temporary)
        temporary.replace(args.output)
        state_path.unlink(missing_ok=True)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main(parse_args())
