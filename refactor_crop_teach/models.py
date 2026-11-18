from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .config import SAM_CHECKPOINT, SAM_RESOLUTION
    from .fusion import MaxForegroundFusion, RefinedRegionFusion
    from .proposal import AllRegionsProposal, CropBox, RegionProposal
    from .structures import structure_probabilities
except ImportError:
    from config import SAM_CHECKPOINT, SAM_RESOLUTION
    from fusion import MaxForegroundFusion, RefinedRegionFusion
    from proposal import AllRegionsProposal, CropBox, RegionProposal
    from structures import structure_probabilities

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SAM_PIXEL_MEAN = (123.675, 116.28, 103.53)
SAM_PIXEL_STD = (58.395, 57.12, 57.375)


def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


class Model(nn.Module, ABC):
    @property
    @abstractmethod
    def num_classes(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward_probabilities(self, images: torch.Tensor) -> torch.Tensor:
        return F.softmax(self(images), dim=1)

    @torch.no_grad()
    def predict_probabilities(
        self, images: torch.Tensor, full_images: torch.Tensor | None = None
    ) -> torch.Tensor:
        self.eval()
        return F.softmax(self(images), dim=1)

    def set_adaptation_train_mode(self, freeze_batch_norm: bool = True) -> None:
        self.train()
        if freeze_batch_norm:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()

    def freeze(self) -> Model:
        self.eval()
        self.requires_grad_(False)
        return self


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
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
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, channels, 1, stride, bias=False),
                nn.BatchNorm2d(channels),
            )
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


class ResUNet34Model(Model):
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

    @property
    def num_classes(self) -> int:
        return int(self.seg_head.out_channels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output, skips = self.res(imagenet_normalize(images))
        output = self.up1(F.relu(output), skips[3])
        output = self.up2(output, skips[2])
        output = self.up3(output, skips[1])
        output = self.up4(output, skips[0])
        return self.seg_head(F.relu(self.bnout(self.up5(output))))


class SegformerB0Model(Model):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        config = SegformerConfig(num_labels=num_classes)
        self.model = SegformerForSemanticSegmentation(config)

    @property
    def num_classes(self) -> int:
        return int(self.model.config.num_labels)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(pixel_values=imagenet_normalize(images)).logits
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int, num_classes: int) -> None:
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
        output = F.interpolate(self.reduce(tokens), scale_factor=2, mode="bilinear", align_corners=False)
        return self.classifier(self.refine(output))


def load_sam_encoder(device: torch.device, checkpoint_path: Path = SAM_CHECKPOINT) -> nn.Module:
    from segment_anything import sam_model_registry

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM ViT-H checkpoint not found: {checkpoint_path}")
    encoder = sam_model_registry["vit_h"](checkpoint=str(checkpoint_path)).image_encoder
    encoder = encoder.requires_grad_(False).to(device=device, dtype=torch.bfloat16).eval()
    encoder.embed_dim = 256
    return encoder


class SAMHeadModel(Model):
    def __init__(
        self,
        encoder: nn.Module,
        head: SegmentationHead,
        resolution: int = SAM_RESOLUTION,
    ) -> None:
        super().__init__()
        self.head = head
        self.resolution = resolution
        encoder.eval()
        encoder.requires_grad_(False)
        self.backbone = [encoder]

    @property
    def num_classes(self) -> int:
        return int(self.head.classifier.out_channels)

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        encoder = self.backbone[0]
        params = list(encoder.parameters())
        dtype = params[0].dtype if params else images.dtype
        mean = images.new_tensor(SAM_PIXEL_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(SAM_PIXEL_STD).view(1, 3, 1, 1)
        inputs = ((images * 255.0 - mean) / std).to(dtype)
        with torch.no_grad():
            return torch.cat([encoder(inputs[index : index + 1]) for index in range(len(inputs))]).float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        inputs = F.interpolate(
            images,
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        )
        logits = self.head(self._encode(inputs))
        return F.interpolate(logits, size=images.shape[-2:], mode="bilinear", align_corners=False)

    @classmethod
    def from_checkpoint(
        cls,
        head_path: Path,
        device: torch.device,
        sam_checkpoint: Path = SAM_CHECKPOINT,
        encoder: nn.Module | None = None,
    ) -> SAMHeadModel:
        if not head_path.is_file():
            raise FileNotFoundError(f"Missing head checkpoint: {head_path}")
        checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
        backbone_name = checkpoint.get("backbone", "sam_vit_h")
        if backbone_name != "sam_vit_h":
            raise ValueError(f"Unsupported head backbone {backbone_name!r}; expected 'sam_vit_h'")
        sam_encoder = load_sam_encoder(device, sam_checkpoint) if encoder is None else encoder.to(device)
        embed_dim = getattr(sam_encoder, "embed_dim", 256)
        layers = tuple(checkpoint.get("layers", (23,)))
        head = SegmentationHead(embed_dim * len(layers), checkpoint["hidden"], checkpoint["num_classes"])
        head.load_state_dict(checkpoint["state_dict"])
        model = cls(sam_encoder, head, resolution=int(checkpoint.get("resolution", SAM_RESOLUTION)))
        return model.to(device)


class CropRefineModel(Model):
    def __init__(
        self,
        coarse_model: Model,
        crop_model: Model,
        region_proposal: RegionProposal | None = None,
        region_fusion: RefinedRegionFusion | None = None,
    ) -> None:
        super().__init__()
        self.coarse_model = coarse_model.freeze()
        self.crop_model = crop_model.freeze()
        self.region_proposal = region_proposal or AllRegionsProposal()
        self.region_fusion = region_fusion or MaxForegroundFusion()

    @property
    def num_classes(self) -> int:
        return self.crop_model.num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        probs = self.predict_probabilities(images)
        return probs.clamp_min(1e-6).log()

    @torch.no_grad()
    def predict_probabilities(
        self, images: torch.Tensor, full_images: torch.Tensor | None = None
    ) -> torch.Tensor:
        full = images if full_images is None else full_images
        size = images.shape[-1]
        sy, sx = full.shape[-2] / size, full.shape[-1] / size
        outputs = []
        for idx in range(images.shape[0]):
            img = images[idx : idx + 1]
            full_img = full[idx : idx + 1]
            coarse_classes = self.coarse_model.predict_probabilities(img)
            boxes: list[CropBox] = self.region_proposal.propose(structure_probabilities(coarse_classes))
            regions: list[tuple[CropBox, torch.Tensor]] = []
            for box in boxes:
                crop = full_img[
                    :,
                    :,
                    round(box.top * sy) : round((box.top + box.side) * sy),
                    round(box.left * sx) : round((box.left + box.side) * sx),
                ]
                crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
                classes = self.crop_model.predict_probabilities(crop)
                resized = F.interpolate(classes, size=(box.side, box.side), mode="bilinear", align_corners=False)
                regions.append((box, resized))
            _, merged = self.region_fusion.fuse(regions, size)
            outputs.append(merged)
        return torch.cat(outputs, dim=0)


NETWORKS: dict[str, type[Model]] = {
    "resunet34": ResUNet34Model,
    "segformer_b0": SegformerB0Model,
}


def build_model(name: str, num_classes: int) -> Model:
    if name not in NETWORKS:
        raise ValueError(f"Unknown model {name!r}; expected one of {sorted(NETWORKS)}")
    return NETWORKS[name](num_classes)


def load_cnn_model(path: Path, device: torch.device) -> Model:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["network"], int(checkpoint["num_classes"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)
