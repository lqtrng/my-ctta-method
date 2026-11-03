#!/usr/bin/env python3
"""Train a box detector on frozen DINOv3 features, on labeled domain A only.

    python src/training/train_dinov3_detector.py --task prostate

The detector (``common.dinov3.DetectionHead``, CenterNet-style) predicts a heatmap of
object centers and each object's box size on the whole image.  Its targets come from
the domain-A masks: one box per connected region.  DINOv3 stays frozen; only the head
is trained.  ``methods/detect_crop_tta.py`` uses its boxes to place crops.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torchvision.transforms import ColorJitter
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.dinov3 import DetectionHead, dinov3_tokens, load_dinov3  # noqa: E402
from common.paths import DINOV3_ARCH, MODEL_DIR, PROSTATE_DINOV3_DETECTOR_CHECKPOINT  # noqa: E402
from common.protocol import seed_everything  # noqa: E402
from common.tasks import TASKS  # noqa: E402
from common.tracking import log_metrics, tracked_run  # noqa: E402

MIN_REGION = 20  # Pixels: smaller labeled regions are not boxes to detect.


def boxes_of(mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """(center y, center x, height, width) of each connected region, in pixels."""
    components, count = ndimage.label(mask)
    boxes = []
    for region in ndimage.find_objects(components):
        rows, columns = region
        height, width = rows.stop - rows.start, columns.stop - columns.start
        if (components[region] > 0).sum() >= MIN_REGION:
            boxes.append(((rows.start + rows.stop) / 2, (columns.start + columns.stop) / 2, height, width))
    return boxes


def targets_for(boxes: list, grid: int, cell: float) -> tuple[torch.Tensor, ...]:
    """Gaussian center heatmap, and size and offset targets at each center cell."""
    heatmap = np.zeros((grid, grid), dtype=np.float32)
    size = np.zeros((2, grid, grid), dtype=np.float32)
    offset = np.zeros((2, grid, grid), dtype=np.float32)
    centers = np.zeros((grid, grid), dtype=np.float32)
    rows, columns = np.mgrid[0:grid, 0:grid]
    for center_y, center_x, height, width in boxes:
        y, x = center_y / cell, center_x / cell
        row, column = min(int(y), grid - 1), min(int(x), grid - 1)
        sigma = max(min(height, width) / cell / 6.0, 1.0)
        heatmap = np.maximum(heatmap, np.exp(-((rows - row) ** 2 + (columns - column) ** 2) / (2 * sigma**2)))
        size[:, row, column] = (height / cell, width / cell)
        offset[:, row, column] = (y - row, x - column)
        centers[row, column] = 1.0
    return tuple(torch.from_numpy(array) for array in (heatmap, size, offset, centers))


def detection_loss(outputs: tuple, targets: tuple) -> torch.Tensor:
    """CenterNet: focal loss on the heatmap, L1 on size and offset at the centers."""
    heatmap, size, offset = outputs
    target_heatmap, target_size, target_offset, centers = targets
    probability = heatmap.sigmoid()[:, 0].clamp(1e-4, 1 - 1e-4)
    positive = target_heatmap.eq(1).float()
    positive_loss = -(((1 - probability) ** 2) * probability.log() * positive).sum()
    negative_loss = -(((1 - target_heatmap) ** 4) * (probability**2) * (1 - probability).log() * (1 - positive)).sum()
    count = positive.sum().clamp_min(1.0)
    mask = centers[:, None]
    size_loss = (F.l1_loss(size, target_size, reduction="none") * mask).sum() / count
    offset_loss = (F.l1_loss(offset, target_offset, reduction="none") * mask).sum() / count
    return (positive_loss + negative_loss) / count + 0.1 * size_loss + offset_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=["prostate"], default="prostate")
    parser.add_argument("--dinov3-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--dinov3-arch", default=DINOV3_ARCH)
    parser.add_argument("--layers", type=int, nargs="+", default=[23])
    parser.add_argument("--resolution", type=int, default=768, help="DINOv3 input size for the whole image")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROSTATE_DINOV3_DETECTOR_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    task = TASKS[args.task]
    dataset = task.source_dataset()
    samples = [dataset[index] for index in range(len(dataset))]
    model = load_dinov3(args.dinov3_dir, args.dinov3_arch, device)
    head = DetectionHead(model.embed_dim * len(args.layers), args.hidden).to(device)
    grid = 2 * args.resolution // model.patch_embed.patch_size[0]
    cell = task.image_size / grid
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(samples) / args.batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, total_steps=steps)
    color = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08)

    with tracked_run("train_dinov3_detector", args):
        for epoch in range(args.epochs):
            order = np.random.permutation(len(samples))
            losses = []
            for start in tqdm(range(0, len(order), args.batch_size), desc=f"epoch {epoch}", leave=False):
                images, targets = [], []
                for index in order[start : start + args.batch_size]:
                    image, mask = samples[index]
                    image, mask = torch.from_numpy(image), mask[0] >= 1
                    if np.random.rand() < 0.5:
                        image, mask = image.flip(-1), mask[:, ::-1].copy()
                    images.append(color(image.to(device)))
                    targets.append(targets_for(boxes_of(mask), grid, cell))
                inputs = F.interpolate(torch.stack(images), size=(args.resolution, args.resolution),
                                       mode="bilinear", align_corners=False)
                outputs = head(dinov3_tokens(model, inputs, tuple(args.layers)))
                batch = tuple(torch.stack([t[k] for t in targets]).to(device) for k in range(4))
                loss = detection_loss(outputs, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach()))
            print(f"epoch {epoch}: loss={np.mean(losses):.4f}", flush=True)
            log_metrics({"train_loss": float(np.mean(losses))}, step=epoch + 1)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": head.state_dict(), "kind": "detector", "dinov3_arch": args.dinov3_arch,
            "layers": args.layers, "hidden": args.hidden, "task": args.task, "resolution": args.resolution,
        }, args.output)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main(parse_args())
