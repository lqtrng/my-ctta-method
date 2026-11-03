"""ReDi-FM with DINOv3 as both the source model and the foundation model.

No ResUNet-34 or nnU-Net is used:

* Source model: frozen DINOv3 + the head trained on whole domain-A images
  (``training/train_dinov3_head.py --full-image``).
* Target model: a copy of that head on the same frozen DINOv3.  Only the head
  adapts, one online update per image.
* Foundation model: frozen DINOv3 + the head trained on domain-A crops, on the
  crop placed by the source model's mask (``redifm_dinov3.DinoV3Segmenter``).

The online update and its objective are those of ``redifm_dinov3.adaptation_step``.

    python src/evaluate.py --method redifm_dinov3_dual [--task prostate] [--seed 0]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from common.dinov3 import SegmentationHead, dinov3_tokens
from common.paths import (
    DINOV3_FULL_IMAGE_HEAD_CHECKPOINT,
    POLYP_DINOV3_FULL_IMAGE_HEAD_CHECKPOINT,
    PROSTATE_DINOV3_FULL_IMAGE_HEAD_CHECKPOINT,
    require_file,
)
from common.protocol import image_to_tensor, make_augmentation, predict_after_update
from common.tasks import Task
from methods import redifm_dinov3
from methods.redifm_dinov3 import PAPER_DO_A  # noqa: F401  (read by evaluate.py)


FULL_IMAGE_HEAD_CHECKPOINTS = {
    "odoc": DINOV3_FULL_IMAGE_HEAD_CHECKPOINT,
    "prostate": PROSTATE_DINOV3_FULL_IMAGE_HEAD_CHECKPOINT,
    "polyp": POLYP_DINOV3_FULL_IMAGE_HEAD_CHECKPOINT,
}


class FullImageNetwork(nn.Module):
    """Whole image -> class logits, with a frozen DINOv3 and a head that can adapt."""

    def __init__(self, backbone: nn.Module, head: nn.Module, layers: tuple[int, ...], resolution: int) -> None:
        super().__init__()
        self.head = head
        self.layers = layers
        self.resolution = resolution
        # In a list so that train(), parameters() and state_dict() only see the head.
        self.backbone = [backbone]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(
            images, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        logits = self.head(dinov3_tokens(self.backbone[0], inputs, self.layers))
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


def load_full_image_network(backbone: nn.Module, path: Path, arch: str) -> FullImageNetwork:
    checkpoint = torch.load(require_file(path, "Full-image DINOv3 head"), map_location="cpu", weights_only=False)
    if checkpoint["dinov3_arch"] != arch:
        raise ValueError("Full-image head was trained on a different DINOv3 variant")
    if not checkpoint.get("full_image", False):
        raise ValueError("Head checkpoint was trained on crops, not full images")
    layers = tuple(checkpoint["layers"])
    head = SegmentationHead(backbone.embed_dim * len(layers), checkpoint["hidden"], checkpoint.get("num_classes", 3))
    head.load_state_dict(checkpoint["state_dict"])
    return FullImageNetwork(backbone, head, layers, checkpoint["crop_resolution"])


def add_arguments(parser: argparse._ArgumentGroup) -> None:
    # The foundation model's options and the adaptation's are those of redifm_dinov3.
    redifm_dinov3.add_arguments(parser)
    parser.add_argument("--full-head-checkpoint", type=Path,
                        help="Default: the full-image DINOv3 head trained for --task")


def build(args: argparse.Namespace, task: Task):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.full_head_checkpoint = args.full_head_checkpoint or FULL_IMAGE_HEAD_CHECKPOINTS[task.name]
    args.head_checkpoint = args.head_checkpoint or redifm_dinov3.HEAD_CHECKPOINTS[task.name]
    args.learning_rate = args.learning_rate or task.learning_rate
    args.candidates = "head"
    # For the saved config: no ResUNet-34 or nnU-Net checkpoint is involved.
    args.arch, args.source_checkpoint = "dinov3_full_image", args.full_head_checkpoint

    segmenter = redifm_dinov3.DinoV3Segmenter(args, device)  # Owns the shared frozen DINOv3.
    source_model = load_full_image_network(segmenter.model, args.full_head_checkpoint, args.dinov3_arch)
    target_model = copy.deepcopy(source_model.head)
    target_model = FullImageNetwork(segmenter.model, target_model, source_model.layers, source_model.resolution)
    source_model.requires_grad_(False).to(device).eval()
    target_model.to(device).train()
    optimizer = torch.optim.Adam(target_model.parameters(), lr=args.learning_rate)
    augmentation = make_augmentation(args)
    print(f"Source/target: DINOv3 {args.dinov3_arch} + full-image head at {source_model.resolution} on {device}; "
          f"foundation: DINOv3 + crop head; one online update of the head per image")

    def segment(image: Image.Image) -> tuple[torch.Tensor, dict]:
        image_tensor = image_to_tensor(image, device)
        step, foundation = redifm_dinov3.adaptation_step(
            source_model, target_model, segmenter, optimizer, augmentation, image, image_tensor, args
        )
        prediction = predict_after_update(target_model, image_tensor)[0]
        return prediction, {"consensus": step.consensus, **foundation}

    return segment
