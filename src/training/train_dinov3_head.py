#!/usr/bin/env python3
"""Train a lightweight OD/OC decoder on frozen DINOv3 features (source domain A only).

SAM ships with a mask decoder; DINOv3 is a pure feature backbone.  To use it as
the frozen foundation model in ReDi-FM it needs a small decoder, which is
trained here once, before deployment, on the same labeled source data as the
source model (RIM-ONE-r3, domain A).  DINOv3 itself is never updated, and the
decoder is frozen during test-time adaptation.  No target-domain data is used.

The decoder operates on disc-centered crops, exactly as at test time, where the
crop is derived from the source model's pseudo-label (the "prompt").  Crop
scale and center are jittered during training so the decoder tolerates the
imperfect, shifted crops produced by the source model under domain shift.
With ``--full-image`` the decoder sees the whole fundus image instead, which
lets DINOv3 segment OD/OC without any source-model prompt.

``--task prostate`` trains the same decoder on prostate domain A (BIDMC) crops
centered on the prostate, with two classes {background, prostate}.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import ColorJitter
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.dinov3 import SegmentationHead, dinov3_tokens, load_dinov3  # noqa: E402
from common.paths import (  # noqa: E402
    DINOV3_ARCH,
    DINOV3_FULL_IMAGE_HEAD_CHECKPOINT,
    DINOV3_HEAD_CHECKPOINT,
    MODEL_DIR,
    POLYP_DINOV3_HEAD_CHECKPOINT,
    PROSTATE_DINOV3_FULL_IMAGE_HEAD_CHECKPOINT,
    PROSTATE_DINOV3_HEAD_CHECKPOINT,
)
from common.protocol import seed_everything  # noqa: E402
from common.redifm import structure_probabilities  # noqa: E402
from common.tasks import TASKS, Task  # noqa: E402
from common.tracking import log_metrics, tracked_run  # noqa: E402


DEFAULT_DINOV3_DIR = MODEL_DIR
DEFAULT_DINOV3_ARCH = DINOV3_ARCH
DEFAULT_OUTPUT = DINOV3_HEAD_CHECKPOINT
DEFAULT_FULL_IMAGE_OUTPUT = DINOV3_FULL_IMAGE_HEAD_CHECKPOINT
# Crop jitter: side relative to the structure, center shift, smallest side.
CROP_SCALE = (1.5, 3.0)
CROP_SHIFT = 0.25
MIN_CROP = 96


def jittered_box(
    target: np.ndarray, scale: tuple[float, float], shift: float, minimum: int
) -> tuple[int, int, int]:
    """Disc-centered square crop with random scale and center offset."""
    size = target.shape[0]
    rows, columns = np.nonzero(target >= 1)
    if len(rows) == 0:
        side = size // 2
        return (size - side) // 2, (size - side) // 2, side
    extent = max(np.ptp(rows) + 1, np.ptp(columns) + 1)
    center_y = (rows.min() + rows.max()) / 2.0 + random.uniform(-shift, shift) * extent
    center_x = (columns.min() + columns.max()) / 2.0 + random.uniform(-shift, shift) * extent
    side = int(np.clip(extent * random.uniform(*scale), minimum, size))
    top = int(np.clip(round(center_y - side / 2.0), 0, size - side))
    left = int(np.clip(round(center_x - side / 2.0), 0, size - side))
    return top, left, side


def enlarged_crop(image: np.ndarray, mask: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    """A jittered crop around the structure, resized to a network's input size."""
    top, left, side = jittered_box(mask[0], CROP_SCALE, CROP_SHIFT, MIN_CROP)
    size = (image_size, image_size)
    crop = torch.from_numpy(image[:, top : top + side, left : left + side])[None]
    label = torch.from_numpy(mask[:, top : top + side, left : left + side].astype(np.float32))[None]
    return (
        F.interpolate(crop, size=size, mode="bilinear", align_corners=False)[0].numpy(),
        F.interpolate(label, size=size, mode="nearest")[0].numpy(),
    )


def enlarged_original_crop(
    image: np.ndarray, mask: np.ndarray, original: np.ndarray, image_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """``enlarged_crop``, but the pixels come from the full-size image: a sharper crop."""
    top, left, side = jittered_box(mask[0], CROP_SCALE, CROP_SHIFT, MIN_CROP)
    scale_y, scale_x = original.shape[1] / mask.shape[1], original.shape[2] / mask.shape[2]
    size = (image_size, image_size)
    crop = torch.from_numpy(original[
        :, round(top * scale_y) : round((top + side) * scale_y), round(left * scale_x) : round((left + side) * scale_x)
    ])[None]
    label = torch.from_numpy(mask[:, top : top + side, left : left + side].astype(np.float32))[None]
    return (
        F.interpolate(crop, size=size, mode="bilinear", align_corners=False, antialias=True)[0].numpy(),
        F.interpolate(label, size=size, mode="nearest")[0].numpy(),
    )


def dice_cross_entropy(logits: torch.Tensor, target: torch.Tensor, task: Task) -> torch.Tensor:
    """Cross-entropy plus a soft Dice term per structure (OD and OC, or prostate)."""
    structures = structure_probabilities(logits.softmax(dim=1)).unbind(dim=1)
    losses = [F.cross_entropy(logits, target)]
    for prediction, label in zip(structures, task.structure_masks(target)):
        label = label.float()
        intersection = (prediction * label).sum()
        losses.append(1.0 - (2.0 * intersection + 1.0) / (prediction.sum() + label.sum() + 1.0))
    return sum(losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), default="odoc")
    parser.add_argument("--dinov3-dir", type=Path, default=DEFAULT_DINOV3_DIR)
    parser.add_argument("--dinov3-arch", default=DEFAULT_DINOV3_ARCH)
    parser.add_argument("--layers", type=int, nargs="+", default=[23])
    parser.add_argument("--crop-resolution", type=int, default=512)
    parser.add_argument("--crop-scale", type=float, nargs=2, default=list(CROP_SCALE))
    parser.add_argument("--crop-shift", type=float, default=CROP_SHIFT)
    parser.add_argument("--min-crop", type=int, default=MIN_CROP)
    parser.add_argument("--crop-from", choices=("resized", "original"), default="resized",
                        help="Cut the training crops from the resized image or the full-size one")
    parser.add_argument("--full-image", action="store_true",
                        help="Train on the whole image instead of disc-centered crops")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output is None:
        if args.task == "prostate":
            args.output = PROSTATE_DINOV3_FULL_IMAGE_HEAD_CHECKPOINT if args.full_image else PROSTATE_DINOV3_HEAD_CHECKPOINT
        elif args.task == "polyp":
            args.output = POLYP_DINOV3_HEAD_CHECKPOINT
        else:
            args.output = DEFAULT_FULL_IMAGE_OUTPUT if args.full_image else DEFAULT_OUTPUT
    return args


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    task = TASKS[args.task]
    original = args.crop_from == "original" and not args.full_image
    dataset = task.source_dataset(original=original)
    # Full-size images are too large to hold in memory; those are read when used.
    samples = dataset if original else [dataset[index] for index in range(len(dataset))]
    model = load_dinov3(args.dinov3_dir, args.dinov3_arch, device)
    channels = model.embed_dim * len(args.layers)
    head = SegmentationHead(channels, args.hidden, task.num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(samples) / args.batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, total_steps=steps)
    color = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08)
    resolution = args.crop_resolution

    method = "train_dinov3_full_image_head" if args.full_image else "train_dinov3_head"
    with tracked_run(method, args):
        for epoch in range(args.epochs):
            order = np.random.permutation(len(samples))
            running = []
            for start in tqdm(range(0, len(order), args.batch_size), desc=f"epoch {epoch}", leave=False):
                crops, labels = [], []
                for index in order[start : start + args.batch_size]:
                    image, target, *full = samples[index]
                    image = torch.from_numpy(image).to(device)
                    target = torch.from_numpy(target[0].astype(np.int64)).to(device)
                    if args.full_image:
                        top, left, side = 0, 0, target.shape[0]
                    else:
                        top, left, side = jittered_box(
                            target.cpu().numpy(), tuple(args.crop_scale), args.crop_shift, args.min_crop
                        )
                    if full:  # The same box in the full-size image's pixels.
                        source = torch.from_numpy(full[0]).to(device)
                        scale_y, scale_x = source.shape[1] / image.shape[1], source.shape[2] / image.shape[2]
                        crop = source[:, round(top * scale_y) : round((top + side) * scale_y),
                                      round(left * scale_x) : round((left + side) * scale_x)]
                    else:
                        crop = image[:, top : top + side, left : left + side]
                    label = target[top : top + side, left : left + side]
                    if random.random() < 0.5:
                        crop, label = crop.flip(-1), label.flip(-1)
                    crop = color(crop)
                    crops.append(
                        F.interpolate(crop[None], size=(resolution, resolution), mode="bilinear",
                                      align_corners=False, antialias=bool(full))[0]
                    )
                    labels.append(
                        F.interpolate(label[None, None].float(), size=(resolution, resolution),
                                      mode="nearest")[0, 0].long()
                    )
                tokens = dinov3_tokens(model, torch.stack(crops), tuple(args.layers))
                logits = F.interpolate(
                    head(tokens), size=(resolution, resolution), mode="bilinear", align_corners=False
                )
                loss = dice_cross_entropy(logits, torch.stack(labels), task)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                running.append(float(loss))
            print(f"epoch {epoch}: loss={np.mean(running):.4f}", flush=True)
            log_metrics({"train_loss": float(np.mean(running))}, step=epoch + 1)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": head.state_dict(),
                "dinov3_arch": args.dinov3_arch,
                "layers": args.layers,
                "hidden": args.hidden,
                "task": args.task,
                "num_classes": task.num_classes,
                "crop_resolution": resolution,
                "full_image": args.full_image,
                "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            args.output,
        )
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main(parse_args())
