#!/usr/bin/env python3
"""Fine-tune the OD/OC ResUNet-34 checkpoint on paper Domain A.

The paths and default hyperparameters are intentionally local to this project.
Only one checkpoint is written, after the final epoch.  Its timestamped name
prevents the downloaded source checkpoint (or an earlier run) being replaced.

Default source-training configuration follows the official ResUNet-34 OD/OC
benchmark recipe used by the paper's cited VPTTA implementation:
  * RIM-ONE-r3 Domain A, all 159 labeled images
  * 512 x 512 input, batch size 8, 200 epochs
  * SGD, lr=1e-3, momentum=0.99, weight decay=5e-4, Nesterov
  * polynomial epoch decay with power 0.9
  * geometric/brightness/noise augmentation

The downloaded checkpoint has a three-class head, so training uses the
compatible background/disc/cup encoding and Dice + cross-entropy loss.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform_2
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.data import DomainADataset  # noqa: E402
from common.models import ResUnet34  # noqa: E402
from common.paths import CHECKPOINT_DIR, DOMAIN_A_DIR  # noqa: E402
from common.protocol import seed_everything  # noqa: E402


INITIAL_CHECKPOINT = CHECKPOINT_DIR / "pretrain-ResUnet.pth"


@dataclass(frozen=True)
class TrainConfig:
    image_size: int = 512
    batch_size: int = 8
    epochs: int = 200
    learning_rate: float = 1e-3
    momentum: float = 0.99
    weight_decay: float = 5e-4
    polynomial_power: float = 0.9
    workers: int = 8
    seed: int = 42
    num_classes: int = 3


def build_augmentation(image_size: int) -> Compose:
    """Official OD/OC source-training augmentation recipe."""
    patch_size = (image_size, image_size)
    return Compose(
        [
            SpatialTransform_2(
                patch_size,
                [size // 2 for size in patch_size],
                do_elastic_deform=True,
                deformation_scale=(0, 0.25),
                do_rotation=True,
                angle_x=(-15 / 360 * 2 * np.pi, 15 / 360 * 2 * np.pi),
                angle_y=(-15 / 360 * 2 * np.pi, 15 / 360 * 2 * np.pi),
                do_scale=True,
                scale=(0.75, 1.25),
                border_mode_data="constant",
                border_cval_data=0,
                border_mode_seg="constant",
                border_cval_seg=0,
                order_seg=0,
                order_data=3,
                random_crop=True,
                p_el_per_sample=0.1,
                p_rot_per_sample=0.1,
                p_scale_per_sample=0.1,
            ),
            MirrorTransform(axes=(0, 1)),
            BrightnessMultiplicativeTransform(
                (0.7, 1.5), per_channel=True, p_per_sample=0.15
            ),
            GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15),
        ]
    )


class AugmentCollate:
    def __init__(self, image_size: int) -> None:
        self.transform = build_augmentation(image_size)

    def __call__(
        self, batch: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, masks = zip(*batch)
        transformed = self.transform(
            data=np.stack(images, axis=0), seg=np.stack(masks, axis=0)
        )
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(transformed["data"])
        ).float()
        mask_tensor = torch.from_numpy(
            np.ascontiguousarray(transformed["seg"][:, 0])
        ).long()
        return image_tensor, mask_tensor


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes: int, smooth: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, targets)
        probabilities = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dimensions = (0, 2, 3)
        intersection = (probabilities * one_hot).sum(dimensions)
        denominator = probabilities.sum(dimensions) + one_hot.sum(dimensions)
        dice_loss = 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return cross_entropy + dice_loss


def default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return CHECKPOINT_DIR / f"domain_A_ResUNet34_last_{stamp}.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--workers", type=int, default=TrainConfig.workers)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one forward/backward batch without saving a checkpoint.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> None:
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
    )
    if config.epochs < 1 or config.batch_size < 1 or config.workers < 0:
        raise ValueError("epochs/batch-size must be positive and workers cannot be negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the paper configuration, but CUDA is unavailable")
    if not INITIAL_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Initialization checkpoint not found: {INITIAL_CHECKPOINT}")

    output_path = args.output.resolve() if args.output else default_output_path()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {output_path}")
    if output_path.resolve() == INITIAL_CHECKPOINT.resolve():
        raise ValueError("Output may not be the downloaded initialization checkpoint")

    seed_everything(config.seed)
    device = torch.device("cuda:0")
    dataset = DomainADataset(DOMAIN_A_DIR, config.image_size)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=AugmentCollate(config.image_size),
        generator=generator,
        persistent_workers=config.workers > 0,
    )

    model = ResUnet34(config.num_classes)
    state_dict = torch.load(INITIAL_CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    criterion = DiceCrossEntropyLoss(config.num_classes)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: max(0.0, 1 - epoch / config.epochs)
        ** config.polynomial_power,
    )

    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"Domain A: {DOMAIN_A_DIR} ({len(dataset)} image/mask pairs)")
    print(f"Initialization: {INITIAL_CHECKPOINT}")
    print(f"Final checkpoint: {output_path}")
    print(f"Configuration: {asdict(config)}")

    model.train()
    for epoch in range(config.epochs):
        print(
            f"Epoch {epoch + 1:03d}/{config.epochs}: "
            f"loss={total_loss / seen:.6f}, lr={optimizer.param_groups[0]['lr']:.8g}"
        )

        total_loss = 0.0
        seen = 0
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.shape[0]
            seen += images.shape[0]

            if args.smoke_test:
                print(f"Smoke test passed: loss={loss.item():.6f}, batch={tuple(images.shape)}")
                return

        scheduler.step()
        print(
            f"Epoch {epoch + 1:03d}/{config.epochs}: "
            f"loss={total_loss / seen:.6f}, lr={optimizer.param_groups[0]['lr']:.8g}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        raise FileExistsError(f"Refusing to overwrite temporary file: {temporary_path}")
    torch.save(model.state_dict(), temporary_path)
    os.rename(temporary_path, output_path)
    print(f"Saved final checkpoint only: {output_path}")


if __name__ == "__main__":
    train(parse_args())
