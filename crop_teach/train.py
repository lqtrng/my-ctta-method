#!/usr/bin/env python3
"""Train one model on one labeled source domain.

    python train.py --problem odoc --source A --kind target   # whole-image CNN: the source / target model
    python train.py --problem odoc --source A --kind crop     # CNN on native-res crops: the crop model
    python train.py --problem odoc --source A --kind head     # DINOv3 full-image head: places the crops

Writes runs/checkpoints/<problem>/source_<source>_<kind>.pth (skipped if it exists, unless --force).

Recipes:
* target / crop: the problem's network (ResUNet-34 or Segformer-B0) from ImageNet weights, batch 8,
  200 epochs, polynomial decay (power 0.9), Dice + cross-entropy, batchgenerators augmentation
  (rotation, scaling, elastic, mirroring, brightness, noise). ResUNet-34: SGD lr 1e-3, momentum 0.99,
  Nesterov, weight decay 5e-4. Segformer-B0: AdamW lr 6e-5, weight decay 0.01. Prostate adds
  nnU-Net's intensity augmentation (MRI sites differ mostly in intensity).
* crop: each image is a jittered square crop around the structure (1.5-3x its extent, center
  shifted by up to 0.25x), cut from the full-size image and resized to the input size.
* head: frozen DINOv3 ViT-L/16, block 23 tokens, input 2x the image size, head of width
  config.HEAD_HIDDEN (1024 = 20M parameters, the default; CT_HEAD_HIDDEN=256 gives the original 1.4M), batch 8, 40 epochs, AdamW lr 1e-3 with one-cycle schedule, horizontal flip and
  color jitter (0.4, 0.4, 0.4, 0.08).
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import tracking
from config import BACKBONE, HEAD_HIDDEN, PROBLEMS, TRAIN_CROP_SCALE, TRAIN_CROP_SHIFT, checkpoint_path, head_resolution
from data import SourceDataset
from losses import structure_probabilities
from networks import SegmentationHead, backbone_tokens, build_network, load_backbone


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --- Crops around the structure ------------------------------------------------------------


def jittered_box(mask: np.ndarray, rng: random.Random) -> tuple[int, int, int]:
    """(top, left, side) of a square around the structure, in the resized image's pixels."""
    size = mask.shape[0]
    rows, columns = np.nonzero(mask >= 1)
    if len(rows) == 0:
        side = size // 2
        return (size - side) // 2, (size - side) // 2, side
    extent = max(np.ptp(rows) + 1, np.ptp(columns) + 1)
    center_y = (rows.min() + rows.max()) / 2 + rng.uniform(-TRAIN_CROP_SHIFT, TRAIN_CROP_SHIFT) * extent
    center_x = (columns.min() + columns.max()) / 2 + rng.uniform(-TRAIN_CROP_SHIFT, TRAIN_CROP_SHIFT) * extent
    side = int(np.clip(extent * rng.uniform(*TRAIN_CROP_SCALE), 96, size))
    top = int(np.clip(round(center_y - side / 2), 0, size - side))
    left = int(np.clip(round(center_x - side / 2), 0, size - side))
    return top, left, side


def native_crop(image: np.ndarray, mask: np.ndarray, full: np.ndarray, rng: random.Random):
    """The box's pixels from the full-size image and its mask, both at the input size."""
    size = mask.shape[-1]
    top, left, side = jittered_box(mask[0], rng)
    sy, sx = full.shape[1] / size, full.shape[2] / size
    crop = torch.from_numpy(full[:, round(top * sy): round((top + side) * sy), round(left * sx): round((left + side) * sx)])
    label = torch.from_numpy(mask[:, top: top + side, left: left + side].astype(np.float32))
    crop = F.interpolate(crop[None], size=(size, size), mode="bilinear", align_corners=False, antialias=True)[0]
    label = F.interpolate(label[None], size=(size, size), mode="nearest")[0]
    return crop.numpy(), label.numpy().astype(np.int16)


# --- CNN training (target and crop) --------------------------------------------------------


def augmentation(size: int, mri: bool):
    from batchgenerators.transforms.abstract_transforms import Compose
    from batchgenerators.transforms.color_transforms import (BrightnessMultiplicativeTransform,
                                                             ContrastAugmentationTransform, GammaTransform)
    from batchgenerators.transforms.noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
    from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
    from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform_2

    angle = (-15 / 360 * 2 * np.pi, 15 / 360 * 2 * np.pi)
    transforms = [
        SpatialTransform_2((size, size), [size // 2, size // 2], do_elastic_deform=True, deformation_scale=(0, 0.25),
                           do_rotation=True, angle_x=angle, angle_y=angle, do_scale=True, scale=(0.75, 1.25),
                           border_mode_data="constant", border_cval_data=0, border_mode_seg="constant",
                           border_cval_seg=0, order_seg=0, order_data=3, random_crop=True,
                           p_el_per_sample=0.1, p_rot_per_sample=0.1, p_scale_per_sample=0.1),
        MirrorTransform(axes=(0, 1)),
        BrightnessMultiplicativeTransform((0.7, 1.5), per_channel=True, p_per_sample=0.15),
        GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15),
    ]
    if mri:  # nnU-Net's 2D intensity augmentation, the same for the three repeated channels.
        transforms += [
            GaussianBlurTransform((0.5, 1.0), different_sigma_per_channel=False, p_per_sample=0.2, p_per_channel=1.0),
            BrightnessMultiplicativeTransform((0.75, 1.25), per_channel=False, p_per_sample=0.15),
            ContrastAugmentationTransform(per_channel=False, p_per_sample=0.15),
            SimulateLowResolutionTransform(zoom_range=(0.5, 1.0), per_channel=False, p_per_channel=1.0,
                                           order_downsample=0, order_upsample=3, p_per_sample=0.25),
            GammaTransform((0.7, 1.5), True, False, retain_stats=True, p_per_sample=0.1),
            GammaTransform((0.7, 1.5), False, False, retain_stats=True, p_per_sample=0.3),
        ]
    return Compose(transforms)


class Collate:
    def __init__(self, size: int, crop: bool, mri: bool, seed: int) -> None:
        self.size, self.crop, self.mri = size, crop, mri
        self.rng = None
        self.transform = None

    def __call__(self, batch):
        if self.transform is None:  # Built in the worker process, seeded per worker by PyTorch.
            self.transform = augmentation(self.size, self.mri)
            self.rng = random.Random(torch.initial_seed())
        if self.crop:
            batch = [native_crop(image, mask, full, self.rng) for image, mask, full in batch]
        images, masks = zip(*batch)
        out = self.transform(data=np.stack(images), seg=np.stack(masks))
        return (torch.from_numpy(np.ascontiguousarray(out["data"])).float(),
                torch.from_numpy(np.ascontiguousarray(out["seg"][:, 0])).long())


def dice_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = F.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, logits.shape[1]).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    dice = (2 * (probabilities * one_hot).sum(dims) + 1e-6) / (probabilities.sum(dims) + one_hot.sum(dims) + 1e-6)
    return F.cross_entropy(logits, targets) + 1 - dice.mean()


def train_cnn(args: argparse.Namespace, output: Path, device: torch.device) -> None:
    problem = PROBLEMS[args.problem]
    crop = args.kind == "crop"
    dataset = SourceDataset(args.problem, args.source, full=crop, max_images=args.max_images)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=args.workers, pin_memory=True,
                        collate_fn=Collate(problem.image_size, crop, args.problem == "prostate", args.seed),
                        generator=torch.Generator().manual_seed(args.seed), persistent_workers=args.workers > 0,
                        multiprocessing_context="spawn" if args.workers > 0 else None)
    model = build_network(problem.network, problem.num_classes, imagenet=True).to(device)
    if problem.network == "segformer_b0":
        optimizer = torch.optim.AdamW(model.parameters(), lr=6e-5, weight_decay=0.01)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=5e-4, nesterov=True)
    epochs = args.epochs or 200
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: max(0.0, 1 - epoch / epochs) ** 0.9)
    model.train()
    for epoch in range(epochs):
        losses = []
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            loss = dice_cross_entropy(model(images), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        print(f"epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.4f}", flush=True)
        tracking.log_metrics({"train_loss": float(np.mean(losses))}, step=epoch + 1)
    save(output, {"state_dict": model.state_dict(), "network": problem.network, "num_classes": problem.num_classes,
                  "kind": args.kind, "source": args.source, "problem": args.problem})


# --- DINOv3 full-image head ----------------------------------------------------------------


def train_head(args: argparse.Namespace, output: Path, device: torch.device) -> None:
    from torchvision.transforms import ColorJitter

    problem = PROBLEMS[args.problem]
    dataset = SourceDataset(args.problem, args.source, max_images=args.max_images)
    samples = [dataset[index] for index in range(len(dataset))]
    backbone = load_backbone(device)
    layers, hidden, resolution = (23,), HEAD_HIDDEN, head_resolution(problem)
    head = SegmentationHead(backbone.embed_dim * len(layers), hidden, problem.num_classes).to(device)
    epochs, batch = args.epochs or 40, 8
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, total_steps=epochs * int(np.ceil(len(samples) / batch)))
    color = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.08)
    for epoch in range(epochs):
        losses = []
        order = np.random.permutation(len(samples))
        for start in range(0, len(order), batch):
            images, labels = [], []
            for index in order[start: start + batch]:
                image, target = samples[index]
                image = torch.from_numpy(image).to(device)
                label = torch.from_numpy(target[0].astype(np.int64)).to(device)
                if random.random() < 0.5:
                    image, label = image.flip(-1), label.flip(-1)
                images.append(F.interpolate(color(image)[None], size=(resolution, resolution), mode="bilinear",
                                            align_corners=False)[0])
                labels.append(F.interpolate(label[None, None].float(), size=(resolution, resolution),
                                            mode="nearest")[0, 0].long())
            logits = F.interpolate(head(backbone_tokens(backbone, torch.stack(images), layers)),
                                   size=(resolution, resolution), mode="bilinear", align_corners=False)
            labels = torch.stack(labels)
            loss = F.cross_entropy(logits, labels)
            for k, prediction in enumerate(structure_probabilities(logits.softmax(dim=1)).unbind(dim=1)):
                truth = (labels >= k + 1).float()
                loss = loss + 1 - (2 * (prediction * truth).sum() + 1) / (prediction.sum() + truth.sum() + 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        print(f"epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.4f}", flush=True)
        tracking.log_metrics({"train_loss": float(np.mean(losses))}, step=epoch + 1)
    save(output, {"state_dict": head.state_dict(), "backbone": BACKBONE, "layers": list(layers), "hidden": hidden,
                  "num_classes": problem.num_classes, "resolution": resolution, "kind": "head",
                  "source": args.source, "problem": args.problem})


def save(output: Path, payload: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.rename(output)  # Only a finished model ever has the final name.
    print(f"Saved: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=list(PROBLEMS), required=True)
    parser.add_argument("--source", required=True, help="Source domain letter")
    parser.add_argument("--kind", choices=("target", "crop", "head"), required=True)
    parser.add_argument("--epochs", type=int, help="Default: 200 (target, crop), 40 (head); lower for a smoke test")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-images", type=int, help="Use only the first N source images (smoke test)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Default: runs/checkpoints/<problem>/source_<source>_<kind>.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="Retrain even if the checkpoint exists")
    args = parser.parse_args()
    if args.source not in PROBLEMS[args.problem].domains:
        raise SystemExit(f"{args.problem} has no domain {args.source}")
    output = args.output or checkpoint_path(args.problem, args.source, args.kind)
    if output.exists() and not args.force:
        print(f"Exists, skipped: {output}")
        return
    seed_everything(args.seed)
    device = torch.device(args.device)
    start = time.time()
    with tracking.run(f"train_{args.kind}", vars(args)):
        (train_head if args.kind == "head" else train_cnn)(args, output, device)
        tracking.log_metrics({"minutes": (time.time() - start) / 60})


if __name__ == "__main__":
    main()
