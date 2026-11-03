"""DINOv3 places the crop and a crop-trained ResUNet-34 segments it, without adaptation.

Frozen DINOv3 + the head trained on whole domain-A images predicts a mask, which places
a crop (the rule of ``redifm_dinov3.DinoV3Segmenter.crop_box``).  The frozen ResUNet-34
trained on enlarged domain-A crops (``training/train_source.py --crop``) predicts on the
crop, enlarged to its input size.  Its prediction, pasted back with background outside
the crop, is the output.  Nothing updates between images.

``--crop-region`` chooses what gets cropped: the largest region of the mask, the most
confident one, or ``all``: every region gets its own crop, and where crops overlap each
pixel takes the prediction of the crop most confident that it is foreground.

    python src/evaluate.py --method dinov3_crop [--task polyp] [--crop-region all]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from common.dinov3 import load_dinov3
from common.models import build_network
from common.paths import (
    DINOV3_ARCH,
    MODEL_DIR,
    POLYP_RESUNET_CROP_CHECKPOINT,
    RESUNET_CROP_CHECKPOINT,
    require_file,
)
from common.protocol import image_to_tensor
from common.redifm import structure_probabilities
from common.tasks import Task
from methods.redifm_dinov3 import DinoV3Segmenter
from methods.redifm_dinov3_dual import FULL_IMAGE_HEAD_CHECKPOINTS, load_full_image_network


CROP_CHECKPOINTS = {"odoc": RESUNET_CROP_CHECKPOINT, "polyp": POLYP_RESUNET_CROP_CHECKPOINT}


class CropPlacer:
    crop_box = DinoV3Segmenter.crop_box  # Same crop rule; it only reads self.args.

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args


def add_arguments(parser: argparse._ArgumentGroup) -> None:
    parser.add_argument("--dinov3-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--dinov3-arch", default=DINOV3_ARCH)
    parser.add_argument("--full-head-checkpoint", type=Path, help="Default: the full-image head of --task")
    parser.add_argument("--crop-checkpoint", type=Path, help="Default: the crop-trained ResUNet-34 of --task")
    parser.add_argument("--crop-from", choices=("resized", "original"), default="resized",
                        help="Cut the crop's pixels from the resized image or the original full-size one")
    parser.add_argument("--pseudo-threshold", type=float, default=0.5)
    parser.add_argument("--crop-scale", type=float, default=2.0, help="Crop side as a multiple of the region")
    parser.add_argument("--min-crop", type=int, default=96)
    parser.add_argument("--crop-region", choices=("largest", "confident", "all"), default="all",
                        help="Which regions of the mask get a crop: the largest, the most confident or all")


def build(args: argparse.Namespace, task: Task):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.full_head_checkpoint = args.full_head_checkpoint or FULL_IMAGE_HEAD_CHECKPOINTS[task.name]
    args.crop_checkpoint = args.crop_checkpoint or CROP_CHECKPOINTS[task.name]

    backbone = load_dinov3(args.dinov3_dir, args.dinov3_arch, device)
    source_model = load_full_image_network(backbone, args.full_head_checkpoint, args.dinov3_arch)
    source_model.requires_grad_(False).to(device).eval()
    crop_model = build_network("resunet34", task.num_classes, args.image_size)
    crop_model.load_state_dict(torch.load(require_file(args.crop_checkpoint, "Crop ResUNet-34"),
                                          map_location="cpu", weights_only=True))
    crop_model.requires_grad_(False).to(device).eval()
    placer = CropPlacer(args)
    print(f"Crop placement: DINOv3 {args.dinov3_arch} + full-image head at {source_model.resolution} on {device}; "
          f"prediction: {args.crop_checkpoint.name}; no adaptation")

    @torch.inference_mode()
    def segment(image: Image.Image) -> tuple[torch.Tensor, dict]:
        image_tensor = image_to_tensor(image, device)
        source = structure_probabilities(F.softmax(source_model(image_tensor), dim=1))
        soft = source[0, 0].cpu().numpy()
        mask = soft > args.pseudo_threshold
        if args.crop_region == "all":
            components, count = ndimage.label(mask)
            boxes = [placer.crop_box(components == index, soft) for index in range(1, count + 1)]
            boxes = boxes or [placer.crop_box(mask, soft)]
        else:
            boxes = [placer.crop_box(mask, soft)]

        full = image_to_tensor(image.info["original"], device) if args.crop_from == "original" else image_tensor
        scale_y = full.shape[-2] / image_tensor.shape[-2]
        scale_x = full.shape[-1] / image_tensor.shape[-1]
        merged = None
        for top, left, side in boxes:
            # The box is in resized coordinates; the same area of the full-size pixels.
            crop = full[
                :, :,
                round(top * scale_y) : round((top + side) * scale_y),
                round(left * scale_x) : round((left + side) * scale_x),
            ]
            enlarged = F.interpolate(
                crop, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False,
                antialias=args.crop_from == "original",
            )
            probabilities = F.softmax(crop_model(enlarged), dim=1)
            pasted = torch.zeros_like(probabilities)
            pasted[:, :, top : top + side, left : left + side] = F.interpolate(
                probabilities, size=(side, side), mode="bilinear", align_corners=False
            )
            if merged is None:
                merged = pasted
            else:
                # Each pixel keeps the crop that is most confident of foreground there.
                better = pasted[:, 1:].sum(dim=1, keepdim=True) > merged[:, 1:].sum(dim=1, keepdim=True)
                merged = torch.where(better, pasted, merged)
        # Outside every crop all classes are 0, so argmax gives background there.
        return merged.argmax(dim=1)[0], {"crops": float(len(boxes))}

    return segment
