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
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import ColorJitter
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.data import DomainADataset  # noqa: E402
from common.paths import (  # noqa: E402
    CHECKPOINT_DIR,
    DINOV3_ARCH,
    DOMAIN_A_DIR,
    MODEL_DIR,
    dinov3_weights,
    require_file,
)
from common.protocol import seed_everything  # noqa: E402


DEFAULT_DINOV3_DIR = MODEL_DIR
DEFAULT_DINOV3_ARCH = DINOV3_ARCH
DEFAULT_OUTPUT = CHECKPOINT_DIR / "dinov3_vitl16_odoc_head_domain_A.pth"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_dinov3(directory: Path, arch: str, device: torch.device) -> nn.Module:
    import timm

    weights = require_file(dinov3_weights(directory, arch), "DINOv3 weights")
    model = timm.create_model(
        arch.split(".")[0],
        pretrained=True,
        dynamic_img_size=True,
        pretrained_cfg_overlay=dict(file=str(weights)),
    )
    return model.requires_grad_(False).to(device).eval()


def dinov3_tokens(
    model: nn.Module, crops: torch.Tensor, layers: tuple[int, ...]
) -> torch.Tensor:
    """Patch tokens of the requested blocks, concatenated along channels."""
    mean = crops.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = crops.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    # DINOv3 overflows in float16; bfloat16 keeps the range of float32.
    with torch.no_grad(), torch.autocast(device_type=crops.device.type, dtype=torch.bfloat16):
        outputs = model.forward_intermediates(
            (crops - mean) / std, indices=list(layers), intermediates_only=True
        )
    return torch.cat([output.float() for output in outputs], dim=1)


class SegmentationHead(nn.Module):
    """Small convolutional decoder: patch tokens -> [background, disc-rim, cup]."""

    def __init__(self, in_channels: int, hidden: int = 256, num_classes: int = 3) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        output = self.reduce(tokens)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=False)
        return self.classifier(self.refine(output))


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


def dice_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    disc = probabilities[:, 1] + probabilities[:, 2]
    cup = probabilities[:, 2]
    losses = [F.cross_entropy(logits, target)]
    for prediction, label in ((disc, target >= 1), (cup, target == 2)):
        label = label.float()
        intersection = (prediction * label).sum()
        losses.append(1.0 - (2.0 * intersection + 1.0) / (prediction.sum() + label.sum() + 1.0))
    return sum(losses)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-dir", type=Path, default=DEFAULT_DINOV3_DIR)
    parser.add_argument("--dinov3-arch", default=DEFAULT_DINOV3_ARCH)
    parser.add_argument("--layers", type=int, nargs="+", default=[23])
    parser.add_argument("--crop-resolution", type=int, default=512)
    parser.add_argument("--crop-scale", type=float, nargs=2, default=[1.5, 3.0])
    parser.add_argument("--crop-shift", type=float, default=0.25)
    parser.add_argument("--min-crop", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    dataset = DomainADataset(DOMAIN_A_DIR, 512)
    samples = [dataset[index] for index in range(len(dataset))]
    model = load_dinov3(args.dinov3_dir, args.dinov3_arch, device)
    channels = model.embed_dim * len(args.layers)
    head = SegmentationHead(channels, args.hidden).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    steps = args.epochs * int(np.ceil(len(samples) / args.batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.learning_rate, total_steps=steps)
    color = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08)
    resolution = args.crop_resolution

    for epoch in range(args.epochs):
        order = np.random.permutation(len(samples))
        running = []
        for start in tqdm(range(0, len(order), args.batch_size), desc=f"epoch {epoch}", leave=False):
            crops, labels = [], []
            for index in order[start : start + args.batch_size]:
                image, target = samples[index]
                image = torch.from_numpy(image).to(device)
                target = torch.from_numpy(target[0].astype(np.int64)).to(device)
                top, left, side = jittered_box(
                    target.cpu().numpy(), tuple(args.crop_scale), args.crop_shift, args.min_crop
                )
                crop = image[:, top : top + side, left : left + side]
                label = target[top : top + side, left : left + side]
                if random.random() < 0.5:
                    crop, label = crop.flip(-1), label.flip(-1)
                crop = color(crop)
                crops.append(
                    F.interpolate(crop[None], size=(resolution, resolution), mode="bilinear",
                                  align_corners=False)[0]
                )
                labels.append(
                    F.interpolate(label[None, None].float(), size=(resolution, resolution),
                                  mode="nearest")[0, 0].long()
                )
            tokens = dinov3_tokens(model, torch.stack(crops), tuple(args.layers))
            logits = F.interpolate(
                head(tokens), size=(resolution, resolution), mode="bilinear", align_corners=False
            )
            loss = dice_cross_entropy(logits, torch.stack(labels))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running.append(float(loss))
        print(f"epoch {epoch}: loss={np.mean(running):.4f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": head.state_dict(),
            "dinov3_arch": args.dinov3_arch,
            "layers": args.layers,
            "hidden": args.hidden,
            "crop_resolution": resolution,
            "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        },
        args.output,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main(parse_args())
