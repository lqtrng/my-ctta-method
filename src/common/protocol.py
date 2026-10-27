"""Online CTTA evaluation protocol shared by all adaptation methods.

Preprocessing, test-time augmentation, after-update scoring and per-image DSC
are defined once here so every method is evaluated identically.
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ColorJitter, RandomErasing

from common.data import Sample, load_sample


@dataclass(frozen=True)
class DomainResult:
    domain: str
    dataset: str
    images: int
    disc_dsc: float
    cup_dsc: float
    mean_dsc: float


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_sample(sample: Sample, image_size: int) -> tuple[Image.Image, torch.Tensor]:
    image, target = load_sample(sample)
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    target_image = Image.fromarray(target).resize(
        (image_size, image_size), Image.Resampling.NEAREST
    )
    return image, torch.from_numpy(np.asarray(target_image, dtype=np.uint8).copy()).long()


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    minimum = float(array.min())
    maximum = float(array.max())
    array = (array - minimum) / max(maximum - minimum, 1e-7)
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0).to(device)


def binary_dice_numpy(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    return float(
        (2.0 * np.logical_and(prediction, target).sum() + 1e-6)
        / (prediction.sum() + target.sum() + 1e-6)
    )


def make_augmentation(args: argparse.Namespace) -> nn.Module:
    return nn.Sequential(
        ColorJitter(
            brightness=args.jitter_brightness,
            contrast=args.jitter_contrast,
            saturation=args.jitter_saturation,
            hue=args.jitter_hue,
        ),
        RandomErasing(
            p=args.erasing_probability,
            scale=(args.erasing_min_scale, args.erasing_max_scale),
            ratio=(0.3, 3.3),
            value="random",
        ),
    )


@torch.inference_mode()
def predict_after_update(target_model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    # Score without a second, unintended BN-statistics update. The one training
    # forward pass in adaptation_step remains the only online adaptation pass.
    was_training = target_model.training
    target_model.eval()
    prediction = target_model(image_tensor).argmax(dim=1)
    target_model.train(was_training)
    return prediction


def image_dice(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.flatten().double()
    target = target.flatten().double()
    intersection = (prediction * target).sum()
    return float(
        (2.0 * intersection + 1e-6)
        / (prediction.sum() + target.sum() + 1e-6)
    )
