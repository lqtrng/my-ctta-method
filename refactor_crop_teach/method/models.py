"""Model interface: every network the method uses.

A Model takes images in [0, 1], shape (B, 3, H, W), and returns class logits (B, C, H, W).
``predict`` gives class probabilities in eval mode, without gradients.

    ResUNet34Model     the target model and the crop (refiner) model: OD/OC, prostate
    SegformerB0Model   the target model and the crop (refiner) model: polyp
    SAMHeadModel       frozen SAM ViT-H image encoder + a trained head: the foundation model
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from method.architectures import ResUNet34Network, SegformerB0Network, SegmentationHead


class Model(nn.Module, ABC):
    num_classes: int

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Images (B, 3, H, W) in [0, 1] -> class logits (B, C, H, W)."""

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """Class probabilities (B, C, H, W), in eval mode (batch-norm statistics untouched)."""
        self.eval()
        return F.softmax(self(images), dim=1)


class CNNModel(Model):
    """A CNN trained by crop_teach/train.py; the checkpoint names its network."""

    network_class: type[nn.Module]
    name: str

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.network = self.network_class(num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    def load_state(self, state_dict: dict) -> None:
        self.network.load_state_dict(state_dict)


class ResUNet34Model(CNNModel):
    network_class = ResUNet34Network
    name = "resunet34"


class SegformerB0Model(CNNModel):
    network_class = SegformerB0Network
    name = "segformer_b0"


SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
SAM_PIXEL_STD = (58.395, 57.12, 57.375)
SAM_RESOLUTION = 1024
SAM_CHANNELS = 256


def load_sam_encoder(checkpoint: Path, device: torch.device) -> nn.Module:
    """SAM ViT-H's image encoder, frozen, in bfloat16: 256 channels on a 64 x 64 grid."""
    from segment_anything import sam_model_registry

    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f"SAM ViT-H weights not found (run fetch_assets.py): {checkpoint}")
    encoder = sam_model_registry["vit_h"](checkpoint=str(checkpoint)).image_encoder
    return encoder.requires_grad_(False).to(device=device, dtype=torch.bfloat16).eval()


class SAMHeadModel(Model):
    """Whole image -> class logits: frozen SAM encoder at 1024 x 1024, then the trained head."""

    def __init__(self, encoder: nn.Module, head: SegmentationHead, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.head = head
        self.encoder = [encoder]  # A list, so parameters() and state_dict() see only the head.

    @torch.no_grad()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        """One image at a time: SAM's global attention needs about 0.5 GB per image."""
        inputs = F.interpolate(images, size=(SAM_RESOLUTION, SAM_RESOLUTION), mode="bilinear", align_corners=False)
        mean = inputs.new_tensor(SAM_PIXEL_MEAN).view(1, 3, 1, 1)
        std = inputs.new_tensor(SAM_PIXEL_STD).view(1, 3, 1, 1)
        inputs = ((inputs * 255.0 - mean) / std).to(torch.bfloat16)
        return torch.cat([self.encoder[0](inputs[index: index + 1]) for index in range(len(inputs))]).float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.features(images))
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


CNN_MODELS = {model.name: model for model in (ResUNet34Model, SegformerB0Model)}


def load_model(path: Path, device: torch.device, sam_checkpoint: Path | None = None) -> Model:
    """A checkpoint written by crop_teach/train.py: a CNN (target or crop kind) or a SAM head."""
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing checkpoint (run fetch_assets.py): {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "network" in checkpoint:
        model = CNN_MODELS[checkpoint["network"]](checkpoint["num_classes"])
        model.load_state(checkpoint["state_dict"])
        return model.to(device)
    if checkpoint.get("backbone") != "sam_vit_h":
        raise ValueError(f"{path}: only SAM ViT-H heads are supported, got {checkpoint.get('backbone')}")
    encoder = load_sam_encoder(sam_checkpoint, device)
    head = SegmentationHead(SAM_CHANNELS * len(checkpoint["layers"]), checkpoint["hidden"], checkpoint["num_classes"])
    head.load_state_dict(checkpoint["state_dict"])
    return SAMHeadModel(encoder, head, checkpoint["num_classes"]).to(device)
