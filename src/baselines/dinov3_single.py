"""One frozen DINOv3 segments the image, without the ResUNet-34 and without adaptation.

A frozen DINOv3 ViT-L/16 with a decoder head trained on whole domain-A images
(``training/train_dinov3_head.py --full-image``) predicts every target image once.
No source model, no second teacher and no online update are involved.

    python src/evaluate.py --method dinov3_single
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from common.dinov3 import SegmentationHead, dinov3_tokens, load_dinov3
from common.paths import DINOV3_ARCH, DINOV3_FULL_IMAGE_HEAD_CHECKPOINT, MODEL_DIR
from common.protocol import image_to_tensor
from common.tasks import Task


class FullImageSegmenter:
    """Frozen DINOv3 plus the full-image decoder head: image -> class probabilities."""

    def __init__(
        self, model: nn.Module, checkpoint_path: Path, arch: str, device: torch.device
    ) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["dinov3_arch"] != arch:
            raise ValueError("Full-image head was trained on a different DINOv3 variant")
        if not checkpoint.get("full_image", False):
            raise ValueError("Head checkpoint was trained on crops, not full images")
        self.model = model
        self.layers = tuple(checkpoint["layers"])
        self.resolution = checkpoint["crop_resolution"]
        self.head = SegmentationHead(model.embed_dim * len(self.layers), checkpoint["hidden"])
        self.head.load_state_dict(checkpoint["state_dict"])
        self.head.requires_grad_(False).to(device).eval()

    @torch.inference_mode()
    def __call__(self, image_tensor: torch.Tensor) -> torch.Tensor:
        size = image_tensor.shape[-2:]
        inputs = F.interpolate(
            image_tensor, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False
        )
        logits = self.head(dinov3_tokens(self.model, inputs, self.layers))
        logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)
        return logits.softmax(dim=1)


def add_arguments(parser: argparse._ArgumentGroup) -> None:
    parser.add_argument("--dinov3-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--dinov3-arch", default=DINOV3_ARCH)
    parser.add_argument("--full-head-checkpoint", type=Path, default=DINOV3_FULL_IMAGE_HEAD_CHECKPOINT)


def build(args: argparse.Namespace, task: Task):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_dinov3(args.dinov3_dir, args.dinov3_arch, device)
    segmenter = FullImageSegmenter(model, args.full_head_checkpoint, args.dinov3_arch, device)
    print(f"Foundation model: DINOv3 {args.dinov3_arch} on {device}, full-image head")

    def segment(image: Image.Image) -> tuple[torch.Tensor, dict]:
        return segmenter(image_to_tensor(image, device)).argmax(dim=1)[0], {}

    return segment
