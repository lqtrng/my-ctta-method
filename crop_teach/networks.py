"""Networks: the source/target CNNs (ResUNet-34, Segformer-B0) and the frozen DINOv3 with its head.

Every network takes images in [0, 1], shape (B, 3, H, W), and returns class logits (B, C, H, W).
The CNNs start from ImageNet weights (torchvision ResNet-34, nvidia/mit-b0), whatever the source
domain, so no model has seen another benchmark domain before training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from config import BACKBONE, DINOV3_ARCH, SAM_CHECKPOINT, dinov3_weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


# --- ResUNet-34 (the ReDi-FM paper's network for OD/OC and prostate) -----------------------


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs if self.downsample is None else self.downsample(inputs)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(inputs))))) + residual)


class ResNet34Encoder(nn.Module):
    """ResNet-34 with torchvision's parameter names, returning the skip features."""

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._layer(64, 3, 1)
        self.layer2 = self._layer(128, 4, 2)
        self.layer3 = self._layer(256, 6, 2)
        self.layer4 = self._layer(512, 3, 2)

    def _layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != channels:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, channels, 1, stride, bias=False),
                                       nn.BatchNorm2d(channels))
        layers = [BasicBlock(self.inplanes, channels, stride, downsample)]
        self.inplanes = channels
        layers += [BasicBlock(channels, channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        first = self.relu(self.bn1(self.conv1(inputs)))
        layer1 = self.layer1(self.maxpool(first))
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return self.layer4(layer3), [first, layer1, layer2, layer3]


class UpBlock(nn.Module):
    def __init__(self, up_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.x_conv = nn.Conv2d(skip_channels, out_channels // 2, 1)
        self.tr_conv = nn.ConvTranspose2d(up_channels, out_channels // 2, 2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.bn(F.relu(torch.cat([self.tr_conv(inputs), self.x_conv(skip)], dim=1)))


class ResUNet34(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.res = ResNet34Encoder()
        self.up1 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 256)
        self.up3 = UpBlock(256, 64, 256)
        self.up4 = UpBlock(256, 64, 256)
        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)
        self.seg_head = nn.Conv2d(32, num_classes, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output, skips = self.res(imagenet_normalize(images))
        output = self.up1(F.relu(output), skips[3])
        output = self.up2(output, skips[2])
        output = self.up3(output, skips[1])
        output = self.up4(output, skips[0])
        return self.seg_head(F.relu(self.bnout(self.up5(output))))

    def load_imagenet(self) -> None:
        from torchvision.models import ResNet34_Weights, resnet34

        state = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1).state_dict()
        missing = self.res.load_state_dict({k: v for k, v in state.items() if not k.startswith("fc.")}, strict=True)
        assert not missing.missing_keys and not missing.unexpected_keys


# --- Segformer-B0 (the ReDi-FM paper's network for polyp) -----------------------------------


class SegformerB0(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        config = SegformerConfig.from_pretrained("nvidia/mit-b0", num_labels=num_classes)
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=imagenet_normalize(images)).logits  # 1/4 resolution
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

    def load_imagenet(self) -> None:
        from transformers import SegformerForSemanticSegmentation

        pretrained = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=self.model.config.num_labels)
        self.model.segformer.load_state_dict(pretrained.segformer.state_dict())  # encoder only


NETWORKS = {"resunet34": ResUNet34, "segformer_b0": SegformerB0}


def build_network(name: str, num_classes: int, imagenet: bool = False) -> nn.Module:
    network = NETWORKS[name](num_classes)
    if imagenet:
        network.load_imagenet()
    return network


# --- The frozen backbone (SAM ViT-H or DINOv3) and the full-image head ---------------------

SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
SAM_PIXEL_STD = (58.395, 57.12, 57.375)


def load_backbone(device: torch.device) -> nn.Module:
    """config.BACKBONE, frozen; ``embed_dim`` = channels of the features the head reads."""
    if BACKBONE == "dinov3":
        return load_dinov3(device)
    from segment_anything import sam_model_registry

    if not SAM_CHECKPOINT.is_file():
        raise FileNotFoundError(f"SAM ViT-H checkpoint not found: {SAM_CHECKPOINT}")
    encoder = sam_model_registry["vit_h"](checkpoint=str(SAM_CHECKPOINT)).image_encoder
    encoder = encoder.requires_grad_(False).to(device=device, dtype=torch.bfloat16).eval()
    encoder.embed_dim = 256  # Its output embedding: 256 channels on a 64 x 64 grid.
    encoder.is_sam = True
    return encoder


def backbone_tokens(model: nn.Module, images: torch.Tensor, layers: tuple[int, ...]) -> torch.Tensor:
    """Features for the head. SAM: its output embedding (``layers`` unused), from 1024 x 1024 input,
    one image at a time (its global-attention blocks need ~0.5 GB per image); same features."""
    if not getattr(model, "is_sam", False):
        return dinov3_tokens(model, images, layers)
    if images.shape[-2:] != (1024, 1024):
        raise ValueError("SAM's image encoder needs a 1024 x 1024 input")
    mean = images.new_tensor(SAM_PIXEL_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(SAM_PIXEL_STD).view(1, 3, 1, 1)
    inputs = ((images * 255.0 - mean) / std).to(torch.bfloat16)
    with torch.no_grad():
        return torch.cat([model(inputs[index: index + 1]) for index in range(len(inputs))]).float()



def load_dinov3(device: torch.device, arch: str = DINOV3_ARCH) -> nn.Module:
    import timm

    weights = dinov3_weights(arch)
    if not weights.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights}")
    model = timm.create_model(arch.split(".")[0], pretrained=True, dynamic_img_size=True,
                              pretrained_cfg_overlay=dict(file=str(weights)))
    return model.requires_grad_(False).to(device).eval()


def dinov3_tokens(model: nn.Module, images: torch.Tensor, layers: tuple[int, ...]) -> torch.Tensor:
    """Patch tokens of the given blocks, concatenated along channels (bfloat16: float16 overflows)."""
    with torch.no_grad(), torch.autocast(device_type=images.device.type, dtype=torch.bfloat16):
        outputs = model.forward_intermediates(imagenet_normalize(images), indices=list(layers),
                                              intermediates_only=True)
    return torch.cat([output.float() for output in outputs], dim=1)


class SegmentationHead(nn.Module):
    """Small convolutional decoder on the patch tokens."""

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


class FullImageNetwork(nn.Module):
    """Whole image -> class logits: frozen DINOv3 at ``resolution`` plus the head."""

    def __init__(self, backbone: nn.Module, head: nn.Module, layers: tuple[int, ...], resolution: int) -> None:
        super().__init__()
        self.head = head
        self.layers = layers
        self.resolution = resolution
        self.backbone = [backbone]  # A list, so parameters(), train() and state_dict() see only the head.

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(images, size=(self.resolution, self.resolution), mode="bilinear", align_corners=False)
        logits = self.head(backbone_tokens(self.backbone[0], inputs, self.layers))
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


def load_head(path, backbone: nn.Module) -> FullImageNetwork:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("backbone", "dinov3") != BACKBONE:
        raise ValueError(f"{path} was trained on {checkpoint.get('backbone')}, but CT_BACKBONE is {BACKBONE}")
    layers = tuple(checkpoint["layers"])
    head = SegmentationHead(backbone.embed_dim * len(layers), checkpoint["hidden"], checkpoint["num_classes"])
    head.load_state_dict(checkpoint["state_dict"])
    network = FullImageNetwork(backbone, head, layers, checkpoint["resolution"])
    return network.to(next(backbone.parameters()).device)
