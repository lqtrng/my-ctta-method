"""Continual test-time adaptation where a DINOv3 detector places the crops.

The same method as ``redifm_dinov3_nnunet_crop``, with DINOv3's full-image
segmentation head replaced by a detector head (``training/train_dinov3_detector.py``).
For each target image:

* Frozen DINOv3 + detector head predicts boxes.  ``--boxes all`` crops every detection
  above ``--detection-threshold``; ``--boxes best`` only the most confident one.
* The frozen crop CNN predicts on each crop, enlarged to its input size; where crops
  overlap, each pixel keeps the crop most confident of foreground.  Outside every crop
  is background.  This is the supervision.
* The target CNN, from its whole-image domain-A checkpoint, takes one online update on
  an augmented copy, then predicts the original image; that prediction is scored.

A detector gives boxes, not a mask, so DINOv3 no longer teaches directly: the loss is
the crop teacher's confidence-weighted hard mask plus cross-entropy from its
probabilities, the terms of ``redifm_dinov3_nnunet_crop.teacher_loss`` that remain
without a DINOv3 mask.

    python src/evaluate.py --method detect_crop_tta --task prostate [--boxes all]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from common.dinov3 import DetectionHead, decode_boxes, dinov3_tokens, load_dinov3
from common.models import build_network
from common.paths import PROSTATE_DINOV3_DETECTOR_CHECKPOINT, SOURCE_CHECKPOINTS, require_file
from common.protocol import image_to_tensor, make_augmentation, predict_after_update
from common.redifm import (
    confidence_weight,
    nested_foundation_classes,
    structure_probabilities,
    weighted_binary_segmentation_loss,
)
from common.tasks import Task
from methods import redifm_dinov3
from methods.redifm_dinov3_nnunet_crop import (
    FOUNDATION_ARCH,
    LEARNING_RATE,
    NnUNetCropSegmenter,
    crop_supervision,
)

DETECTOR_CHECKPOINTS = {"prostate": PROSTATE_DINOV3_DETECTOR_CHECKPOINT}


class Detector:
    """Frozen DINOv3 + detector head: image -> boxes."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        checkpoint = torch.load(require_file(args.detector_checkpoint, "DINOv3 detector"),
                                map_location="cpu", weights_only=False)
        self.backbone = load_dinov3(args.dinov3_dir, checkpoint["dinov3_arch"], device)
        self.layers = tuple(checkpoint["layers"])
        self.resolution = checkpoint["resolution"]
        self.head = DetectionHead(self.backbone.embed_dim * len(self.layers), checkpoint["hidden"])
        self.head.load_state_dict(checkpoint["state_dict"])
        self.head.requires_grad_(False).to(device).eval()

    @torch.inference_mode()
    def boxes(self, image_tensor: torch.Tensor, threshold: float, keep_all: bool) -> list[tuple[float, ...]]:
        inputs = F.interpolate(image_tensor, size=(self.resolution, self.resolution), mode="bilinear",
                               align_corners=False)
        outputs = self.head(dinov3_tokens(self.backbone, inputs, self.layers))
        return decode_boxes(*outputs, image_tensor.shape[-1], threshold, keep_all)


def crop_of(box: tuple[float, ...], size: int, scale: float, minimum: int) -> tuple[int, int, int]:
    """(top, left, side): a square around the box, ``scale`` times its longest side."""
    _, center_y, center_x, height, width = box
    side = int(np.clip(max(height, width) * scale, minimum, size))
    top = int(np.clip(round(center_y - side / 2.0), 0, size - side))
    left = int(np.clip(round(center_x - side / 2.0), 0, size - side))
    return top, left, side


def crop_loss(target_classes: torch.Tensor, crop_classes: torch.Tensor) -> torch.Tensor:
    """The crop teacher's hard mask (confidence-weighted) plus cross-entropy from its probabilities."""
    crop = structure_probabilities(crop_classes)
    hard = weighted_binary_segmentation_loss(
        structure_probabilities(target_classes), crop > 0.5, confidence_weight(crop)
    )
    soft = -(nested_foundation_classes(crop) * target_classes.clamp_min(1e-6).log()).sum(dim=1).mean()
    return hard + soft


def add_arguments(parser: argparse._ArgumentGroup) -> None:
    # DINOv3 location, crop scale, learning rate and augmentation: those of redifm_dinov3.
    redifm_dinov3.add_arguments(parser)
    parser.add_argument("--detector-checkpoint", type=Path, help="Default: the DINOv3 detector of --task")
    parser.add_argument("--foundation-checkpoint", type=Path, help="Default: the crop-trained CNN of --task")
    parser.add_argument("--target-checkpoint", type=Path,
                        help="Where the target CNN starts. Default: the whole-image domain-A CNN of --task")
    parser.add_argument("--boxes", choices=("best", "all"), default="all",
                        help="Crop every detection above --detection-threshold, or only the best one")
    parser.add_argument("--detection-threshold", type=float, default=0.3)
    parser.add_argument("--no-augmentation", action="store_true",
                        help="The target learns from the test image itself instead of an augmented copy")


def build(args: argparse.Namespace, task: Task):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.detector_checkpoint = args.detector_checkpoint or DETECTOR_CHECKPOINTS[task.name]
    args.learning_rate = args.learning_rate or LEARNING_RATE
    arch = FOUNDATION_ARCH[task.name]
    target_checkpoint = args.target_checkpoint or SOURCE_CHECKPOINTS[(task.name, arch)]
    args.arch, args.target_checkpoint = arch, target_checkpoint

    detector = Detector(args, device)
    segmenter = NnUNetCropSegmenter(args, task, device)
    args.foundation_checkpoint = segmenter.checkpoint
    target_model = build_network(arch, task.num_classes, args.image_size)
    target_model.load_state_dict(torch.load(require_file(target_checkpoint, "Target checkpoint"),
                                            map_location="cpu", weights_only=True))
    target_model.to(device).train()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=args.learning_rate)
    augmentation = None if args.no_augmentation else make_augmentation(args)
    print(f"Crop placement: DINOv3 detector ({Path(args.detector_checkpoint).name}), {args.boxes} boxes; "
          f"pseudo ground truth: {arch} on the crops ({segmenter.checkpoint.name}); "
          f"target: {arch} ({target_checkpoint.name}), one online update per image")

    def segment(image: Image.Image) -> tuple[torch.Tensor, dict]:
        image_tensor = image_to_tensor(image, device)
        boxes = detector.boxes(image_tensor, args.detection_threshold, args.boxes == "all")
        crops = [crop_of(box, args.image_size, args.crop_scale, args.min_crop) for box in boxes]
        crop_classes = crop_supervision(segmenter, image_tensor, crops)

        target_model.train()
        for module in target_model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()  # Batches of one image would make these statistics noisy.
        optimizer.zero_grad(set_to_none=True)
        augmented = image_tensor if augmentation is None else augmentation(image_tensor.squeeze(0)).unsqueeze(0)
        crop_loss(F.softmax(target_model(augmented), dim=1), crop_classes).backward()
        optimizer.step()
        prediction = predict_after_update(target_model, image_tensor)[0]
        return prediction, {"boxes": float(len(boxes)), "best_box_score": boxes[0][0]}

    return segment
