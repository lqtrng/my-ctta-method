#!/usr/bin/env python3
"""Continual test-time adaptation, taught by a full-image model and a crop model.

A single, independent file: it imports nothing from this repository and does no
offline training.  Bring three trained models and a folder of images:

    python crop_teach_tta.py \\
        --data data/polyp-4 --domains B C D --image-size 352 \\
        --source-model checkpoints/polyp_dinov3_head_full_image.pth \\
        --dinov3-weights checkpoints/timm-vit_large_patch16_dinov3.lvd1689m \\
        --crop-model checkpoints/polyp_ResUNet34_crop.pth \\
        --target-model checkpoints/polyp_ResUNet34.pth \\
        --output results/polyp

For each image, in stream order and without any reset:

1. The frozen source model predicts a mask on the whole image.  Its largest region of
   the first structure places a square crop.
2. The frozen crop model predicts on the crop, enlarged to --image-size and pasted back.
   Outside the crop is background.  This is the main pseudo-label.
3. The target model takes one Adam update on an augmented copy of the image:

       L = H(target, crop mask)
         + sum_k agreement_k * H(target_k, source mask_k)
         + cross-entropy(teachers fused per pixel, target)

   H is confidence-weighted BCE + Dice, and agreement_k the Dice of the two teachers'
   masks of structure k.
4. The updated target model predicts the original image: the method's output.

Models (auto-detected from the file):
  --source-model   a DINOv3 head checkpoint (with --dinov3-weights), a CNN, or TorchScript
  --crop-model     a CNN state dict (ResUNet-34, nnU-Net 2D) or TorchScript
  --target-model   the same; must be trainable and take the whole image
A TorchScript model maps images in [0, 1], shape (B, 3, H, W), to logits (B, C, H, W).

Classes are nested structures: with C classes, structure k >= 1 is "class >= k"
(C=2: foreground; C=3: disc = rim + cup, and cup).

Data:
  --data FOLDER    subfolders are domains (domain_B_Name or any name); inside each, any
                   folder holding images directly, usually Images/ with Masks/ beside it
                   (same file stems).  Without masks, predictions are saved but not scored.

Output (--output): masks/<domain>/<image>.png with class indices at the original image
size, per_image.csv, and scores.json (mean DSC per domain, and their unweighted mean).

Requirements: torch, torchvision, numpy, pillow, scipy; timm for DINOv3;
dynamic-network-architectures for nnU-Net.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch import nn
from torchvision.transforms import ColorJitter, RandomErasing


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --- CNN architectures ---------------------------------------------------------------


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        output = self.relu(self.bn1(self.conv1(inputs)))
        return self.relu(self.bn2(self.conv2(output)) + residual)


class ResNet34Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.in1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)
        # Unused, but part of the checkpoints' state dict.
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, 1000)

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(channels),
            )
        layers: list[nn.Module] = [BasicBlock(self.inplanes, channels, stride, downsample)]
        self.inplanes = channels
        layers.extend(BasicBlock(channels, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output = self.relu(self.bn1(self.conv1(inputs)))
        skips = [output]
        output = self.layer1(self.maxpool(output))
        skips.append(output)
        for layer in (self.layer2, self.layer3, self.layer4):
            output = layer(output)
            skips.append(output)
        return output, skips


class UnetBlock(nn.Module):
    def __init__(self, up_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.x_conv = nn.Conv2d(skip_channels, out_channels // 2, 1)
        self.tr_conv = nn.ConvTranspose2d(up_channels, out_channels // 2, 2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.bn(F.relu(torch.cat([self.tr_conv(inputs), self.x_conv(skip)], dim=1)))


class ResUnet34(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.res = ResNet34Encoder()
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)
        self.seg_head = nn.Conv2d(32, num_classes, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, skips = self.res(inputs)
        output = self.up1(F.relu(output), skips[3])
        output = self.up2(output, skips[2])
        output = self.up3(output, skips[1])
        output = self.up4(output, skips[0])
        return self.seg_head(F.relu(self.bnout(self.up5(output))))


def nnunet_2d(num_classes: int, stages: int) -> nn.Module:
    """nnU-Net v2's default 2D network; features double per stage from 32 up to 512."""
    from dynamic_network_architectures.architectures.unet import PlainConvUNet

    return PlainConvUNet(
        input_channels=3,
        n_stages=stages,
        features_per_stage=[min(32 * 2 ** stage, 512) for stage in range(stages)],
        conv_op=nn.Conv2d,
        kernel_sizes=3,
        strides=[1] + [2] * (stages - 1),
        n_conv_per_stage=2,
        num_classes=num_classes,
        n_conv_per_stage_decoder=2,
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=False,
    )


def zscore(images: torch.Tensor) -> torch.Tensor:
    mean = images.mean(dim=(2, 3), keepdim=True)
    std = images.std(dim=(2, 3), keepdim=True, unbiased=False)
    return (images - mean) / std.clamp_min(1e-8)


class ZScoreNetwork(nn.Module):
    """A network that z-scores its own input (an nnU-Net trained with nnU-Net's recipe)."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(zscore(images))


# --- DINOv3 + head --------------------------------------------------------------------


class SegmentationHead(nn.Module):
    """Small convolutional decoder from DINOv3 patch tokens to class logits."""

    def __init__(self, in_channels: int, hidden: int, num_classes: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(in_channels, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.GELU())
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False), nn.BatchNorm2d(hidden), nn.GELU(),
        )
        self.classifier = nn.Conv2d(hidden, num_classes, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        output = F.interpolate(self.reduce(tokens), scale_factor=2, mode="bilinear", align_corners=False)
        return self.classifier(self.refine(output))


class DinoV3FullImage(nn.Module):
    """Frozen DINOv3 and head: whole image -> class logits at the image's size."""

    def __init__(self, backbone: nn.Module, head: nn.Module, layers: list[int], resolution: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.layers = layers
        self.resolution = resolution

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(images, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        mean = inputs.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = inputs.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        # DINOv3 overflows in float16; bfloat16 keeps the range of float32.
        with torch.no_grad(), torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16):
            outputs = self.backbone.forward_intermediates(
                (inputs - mean) / std, indices=self.layers, intermediates_only=True
            )
        logits = self.head(torch.cat([output.float() for output in outputs], dim=1))
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


def load_dinov3(weights: Path, arch: str) -> nn.Module:
    import timm

    weights = weights / "model.safetensors" if weights.is_dir() else weights
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
    return timm.create_model(
        arch.split(".")[0], pretrained=True, dynamic_img_size=True, pretrained_cfg_overlay=dict(file=str(weights))
    )


# --- Loading any of the three models from a path ----------------------------------------


def state_num_classes(state: dict[str, torch.Tensor], key: str) -> int:
    return state[key].shape[0]


def load_cnn_state(state: dict[str, torch.Tensor]) -> nn.Module:
    """Rebuild the network a state dict belongs to, from its parameter names."""
    if any(key.startswith("res.") for key in state):
        model = ResUnet34(state_num_classes(state, "seg_head.weight"))
    else:
        prefix = "network." if any(key.startswith("network.") for key in state) else ""
        seg_layers = {
            int(match.group(1)): key
            for key in state
            for match in [re.fullmatch(rf"{prefix}decoder\.seg_layers\.(\d+)\.weight", key)]
            if match
        }
        if not seg_layers:
            raise ValueError("Unknown state dict: expected ResUNet-34 or nnU-Net parameter names")
        network = nnunet_2d(state_num_classes(state, seg_layers[max(seg_layers)]), len(seg_layers) + 1)
        model = ZScoreNetwork(network) if prefix else network
    model.load_state_dict(state, strict=True)
    return model


def load_model(path: Path, dinov3_weights: Path | None, role: str) -> nn.Module:
    """A model from a path: TorchScript, a DINOv3 head checkpoint or a CNN state dict."""
    if not path.is_file():
        raise FileNotFoundError(f"{role} model not found: {path}")
    try:
        return torch.jit.load(str(path), map_location="cpu")
    except RuntimeError:
        pass
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        return checkpoint
    if "dinov3_arch" not in checkpoint:
        return load_cnn_state(checkpoint)
    if dinov3_weights is None:
        raise SystemExit(f"{path} is a DINOv3 head: pass --dinov3-weights")
    if not checkpoint.get("full_image", False):
        raise SystemExit(f"{path} was trained on crops; the {role} model takes whole images")
    backbone = load_dinov3(dinov3_weights, checkpoint["dinov3_arch"])
    layers = list(checkpoint["layers"])
    head = SegmentationHead(backbone.embed_dim * len(layers), checkpoint["hidden"], checkpoint.get("num_classes", 3))
    head.load_state_dict(checkpoint["state_dict"])
    return DinoV3FullImage(backbone, head, layers, checkpoint["crop_resolution"])


# --- Data -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    domain: str
    identifier: str
    image_path: Path
    mask_path: Path | None


def images_in(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def domain_samples(domain: str, folder: Path) -> list[Sample]:
    """Every folder of images below ``folder``, paired with a Masks folder beside it.

    Folders named "images" are used if there are any; otherwise every folder of image
    files whose name does not mention masks or labels.
    """
    folders = {path.parent for path in folder.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES}
    named = {candidate for candidate in folders if candidate.name.lower() == "images"}
    candidates = named or {
        candidate for candidate in folders if not re.search(r"mask|label", candidate.name.lower())
    }
    samples = []
    for candidate in sorted(candidates):
        masks = {}
        for name in ("Masks", "masks", "mask", "labels"):
            if (candidate.parent / name).is_dir():
                masks = {path.stem: path for path in images_in(candidate.parent / name)}
                break
        samples += [
            Sample(domain, path.stem, path, masks.get(path.stem)) for path in images_in(candidate)
        ]
    return samples


def build_stream(data: Path, selected: list[str] | None) -> dict[str, list[Sample]]:
    folders = {}
    for folder in sorted(path for path in data.iterdir() if path.is_dir()):
        match = re.match(r"domain_([^_]+)", folder.name)
        folders[match.group(1) if match else folder.name] = folder
    if selected is None:
        selected = [name for name in folders if name != "raw"]
    missing = set(selected) - set(folders)
    if missing:
        raise SystemExit(f"No domain(s) {sorted(missing)} in {data}; found {sorted(folders)}")
    stream = {name: domain_samples(name, folders[name]) for name in selected}
    for name, samples in stream.items():
        if not samples:
            raise SystemExit(f"No images found in domain {name}: {folders[name]}")
    return stream


def read_mask(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as opened:
        raw = np.asarray(opened.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST), dtype=np.uint8)
    if set(np.unique(raw).tolist()) <= {0, 1, 2}:
        return torch.from_numpy(raw.copy()).long()
    # Otherwise the fundus convention: 255 background, 128 disc rim, 0 cup.
    target = np.zeros(raw.shape, dtype=np.int64)
    target[raw < 255] = 1
    target[raw == 0] = 2
    return torch.from_numpy(target)


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    array = (array - float(array.min())) / max(float(array.max() - array.min()), 1e-7)
    return torch.from_numpy(array.copy()).unsqueeze(0).to(device)


# --- The method --------------------------------------------------------------------------


def structure_probabilities(classes: torch.Tensor) -> torch.Tensor:
    """Class probabilities -> nested structures: structure k is "class >= k"."""
    structures = classes.flip(1).cumsum(1).flip(1)[:, 1:]
    return structures.clamp(1e-6, 1.0 - 1e-6)


def nested_classes(structures: torch.Tensor) -> torch.Tensor:
    """Independent structure probabilities -> classes, each structure inside the previous one."""
    cumulative = structures[:, 0].clamp(1e-6, 1.0 - 1e-6)
    inside = [cumulative]
    for index in range(1, structures.shape[1]):
        cumulative = (structures[:, index] * cumulative).clamp(1e-6, 1.0 - 1e-6)
        inside.append(cumulative)
    classes = [1.0 - inside[0]]
    for outer, inner in zip(inside, inside[1:]):
        classes.append((outer - inner).clamp_min(1e-6))
    classes.append(inside[-1])
    stacked = torch.stack(classes, dim=1)
    return stacked / stacked.sum(dim=1, keepdim=True).clamp_min(1e-6)


def largest_component(mask: np.ndarray) -> np.ndarray:
    components, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndimage.sum(mask, components, range(1, count + 1))
    return components == int(np.argmax(sizes) + 1)


def crop_box(mask: np.ndarray, probability: np.ndarray, scale: float, minimum: int) -> tuple[int, int, int]:
    """(top, left, side) of a square around the largest region of the mask."""
    size = mask.shape[0]
    component = largest_component(mask)
    if not component.any():
        row, column = np.unravel_index(np.argmax(probability), probability.shape)
        component[row, column] = True
    rows, columns = np.nonzero(component)
    center_y = (rows.min() + rows.max()) / 2.0
    center_x = (columns.min() + columns.max()) / 2.0
    side = int(np.clip(max(np.ptp(rows) + 1, np.ptp(columns) + 1) * scale, minimum, size))
    top = int(np.clip(round(center_y - side / 2.0), 0, size - side))
    left = int(np.clip(round(center_x - side / 2.0), 0, size - side))
    return top, left, side


def predict_crop(model: nn.Module, image: torch.Tensor, box: tuple[int, int, int]) -> torch.Tensor:
    """The crop model on the enlarged crop: nested structures, background outside the crop."""
    top, left, side = box
    crop = image[:, :, top : top + side, left : left + side]
    enlarged = F.interpolate(crop, size=image.shape[-2:], mode="bilinear", align_corners=False)
    probabilities = F.softmax(model(enlarged), dim=1)
    pasted = torch.zeros_like(probabilities)
    pasted[:, :, top : top + side, left : left + side] = F.interpolate(
        probabilities, size=(side, side), mode="bilinear", align_corners=False
    )
    return structure_probabilities(pasted)


def confidence_weight(probability: torch.Tensor, gamma: float = 1.3, variance: float = 0.1) -> torch.Tensor:
    """About 0.04 where a teacher is unsure (p = 0.5), about 0.94 where it is sure."""
    gaussian = torch.exp(-((probability - 0.5) ** 2) / (2.0 * variance)) / math.sqrt(2.0 * math.pi * variance)
    return (gamma - gaussian).clamp_min(1e-6)


def weighted_hard_loss(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Confidence-weighted BCE + Dice against a teacher's mask."""
    prediction = prediction.clamp(1e-6, 1.0 - 1e-6)
    target = target.float()
    bce = (F.binary_cross_entropy(prediction, target, reduction="none") * weight).mean()
    dimensions = (0, 2, 3)
    intersection = (weight * prediction * target).sum(dimensions)
    denominator = (weight * prediction).sum(dimensions) + (weight * target).sum(dimensions)
    return bce + 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def mask_dice(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Dice of two masks per structure, averaged over structures."""
    dimensions = (0, 2, 3)
    first, second = first.float(), second.float()
    numerator = 2.0 * (first * second).sum(dimensions) + 1e-6
    return (numerator / (first.sum(dimensions) + second.sum(dimensions) + 1e-6)).mean()


def teacher_loss(
    target_classes: torch.Tensor,
    source: torch.Tensor,
    source_masks: torch.Tensor,
    crop: torch.Tensor,
    crop_masks: torch.Tensor,
    consensus: torch.Tensor,
) -> torch.Tensor:
    target = structure_probabilities(target_classes)
    source_weight = confidence_weight(source)
    crop_weight = confidence_weight(crop)

    # Hard terms: the crop model always counts, the source by its agreement on each structure.
    loss = weighted_hard_loss(target, crop_masks, crop_weight)
    for index in range(target.shape[1]):
        keep = slice(index, index + 1)
        agreement = mask_dice(source_masks[:, keep], crop_masks[:, keep])
        loss = loss + agreement * weighted_hard_loss(target[:, keep], source_masks[:, keep], source_weight[:, keep])

    # Soft term: cross-entropy from the teachers fused per pixel by their confidence.
    scaled = consensus * source_weight
    fused = (crop_weight * crop + scaled * source) / (crop_weight + scaled).clamp_min(1e-6)
    fused_classes = nested_classes(fused.clamp(1e-6, 1.0 - 1e-6))
    return loss - (fused_classes * target_classes.clamp_min(1e-6).log()).sum(dim=1).mean()


class Method:
    """The three models and one online step."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.source = load_model(args.source_model, args.dinov3_weights, "source").to(device).eval()
        self.crop = load_model(args.crop_model, args.dinov3_weights, "crop").to(device).eval()
        self.target = load_model(args.target_model, args.dinov3_weights, "target").to(device)
        for frozen in (self.source, self.crop):
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
        if isinstance(self.target, DinoV3FullImage):
            self.target.backbone.requires_grad_(False)  # Only its head adapts.
        trainable = [parameter for parameter in self.target.parameters() if parameter.requires_grad]
        if not trainable:
            raise SystemExit("The target model has no trainable parameters")
        self.optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
        self.augmentation = nn.Sequential(
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value="random"),
        )

    def step(self, image: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        threshold = self.args.threshold
        with torch.inference_mode():
            source = structure_probabilities(F.softmax(self.source(image), dim=1))
            source_masks = source > threshold
            soft = source[0, 0].cpu().numpy()
            box = crop_box(soft > threshold, soft, self.args.crop_scale, self.args.min_crop)
            crop = predict_crop(self.crop, image, box)
            crop_masks = crop > 0.5
        source, crop, source_masks, crop_masks = (t.clone() for t in (source, crop, source_masks, crop_masks))
        consensus = mask_dice(source_masks, crop_masks)

        self.target.train()
        if isinstance(self.target, DinoV3FullImage):
            self.target.backbone.eval()
        for module in self.target.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()  # Batches of one image would make these statistics noisy.
        self.optimizer.zero_grad(set_to_none=True)
        augmented = self.augmentation(image.squeeze(0)).unsqueeze(0)
        target_classes = F.softmax(self.target(augmented), dim=1)
        self.num_classes = target_classes.shape[1]
        teacher_loss(target_classes, source, source_masks, crop, crop_masks, consensus).backward()
        self.optimizer.step()

        self.target.eval()
        with torch.inference_mode():
            prediction = self.target(image).argmax(dim=1)[0]
        return prediction, {"consensus": float(consensus), "crop_side": float(box[2])}


# --- Running a stream ----------------------------------------------------------------------


def structure_dices(prediction: torch.Tensor, target: torch.Tensor, num_classes: int) -> list[float]:
    """DSC of each nested structure."""
    scores = []
    for structure in range(1, num_classes):
        first, second = (prediction >= structure).double(), (target >= structure).double()
        numerator = 2.0 * (first * second).sum() + 1e-6
        scores.append(float(numerator / (first.sum() + second.sum() + 1e-6)))
    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="Folder with one subfolder per domain")
    parser.add_argument("--domains", nargs="+", help="Target stream, in order (default: every subfolder)")
    parser.add_argument("--image-size", type=int, required=True, help="Square input size of the CNNs")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--crop-model", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--dinov3-weights", type=Path, help="model.safetensors, or the folder holding it")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=1, help="Passes over the stream, never reset")
    parser.add_argument("--max-images", type=int, help="Limit images per domain, for a smoke test")
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--threshold", type=float, default=0.5, help="Source probability that makes a mask")
    parser.add_argument("--crop-scale", type=float, default=2.0, help="Crop side as a multiple of the region")
    parser.add_argument("--min-crop", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    stream = build_stream(args.data, args.domains)
    method = Method(args, device)
    print(f"Target stream {' -> '.join(stream)}; {sum(len(s) for s in stream.values())} images on {device}", flush=True)

    rows: list[dict] = []
    rounds_scores: list[dict] = []
    for round_index in range(1, args.rounds + 1):
        rows = []
        domain_scores: dict[str, float] = {}
        for domain, samples in stream.items():
            samples = samples[: args.max_images] if args.max_images else samples
            scores = []
            for sample in samples:
                with Image.open(sample.image_path) as opened:
                    original = opened.convert("RGB")
                resized = original.resize((args.image_size, args.image_size), Image.Resampling.BILINEAR)
                prediction, extras = method.step(image_to_tensor(resized, device))
                row = {"round": round_index, "domain": domain, "image": sample.identifier, **extras}
                if sample.mask_path is not None:
                    target = read_mask(sample.mask_path, args.image_size)
                    dices = structure_dices(prediction.cpu(), target, method.num_classes)
                    row["dsc"] = float(np.mean(dices))
                    scores.append(row["dsc"])
                rows.append(row)
                if round_index == args.rounds:
                    folder = args.output / "masks" / domain
                    folder.mkdir(parents=True, exist_ok=True)
                    saved = Image.fromarray(prediction.cpu().numpy().astype(np.uint8), mode="L")
                    saved.resize(original.size, Image.Resampling.NEAREST).save(folder / f"{sample.identifier}.png")
            if scores:
                domain_scores[domain] = 100.0 * float(np.mean(scores))
                print(f"Round {round_index}, domain {domain} (n={len(scores)}): DSC={domain_scores[domain]:.2f}", flush=True)
            else:
                print(f"Round {round_index}, domain {domain} (n={len(samples)}): no masks, predictions only", flush=True)
        rounds_scores.append({
            "round": round_index,
            "domains": domain_scores,
            "mean_dsc": float(np.mean(list(domain_scores.values()))) if domain_scores else None,
        })
        if domain_scores:
            print(f"Round {round_index}: mean DSC over domains = {rounds_scores[-1]['mean_dsc']:.2f}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "scores.json").write_text(json.dumps({
        "rounds": rounds_scores,
        "config": {key: str(value) for key, value in vars(args).items()},
    }, indent=2))
    with (args.output / "per_image.csv").open("w", newline="") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved predictions and scores to {args.output}")


if __name__ == "__main__":
    main()
