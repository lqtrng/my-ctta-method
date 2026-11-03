"""Two frozen DINOv3 models in place of ReDi-FM's source ResUNet + foundation model.

No ResUNet-34 is used and nothing is adapted.  ReDi-FM's source model is
replaced by a frozen full-image DINOv3 (``dinov3_single``), and its foundation
model is the DINOv3 crop teacher of ``redifm_dinov3``:

1. The full-image DINOv3 predicts OD/OC probabilities ``P_s``.
2. Its disc mask places the crop (the "prompt") for the crop teacher, which
   predicts ``P_fm``.
3. The two are fused per pixel with the weighting of Equation (16): the
   foundation model always counts, the source counts as much as the two agree,

       P = (Phi_fm * P_fm + r * Phi_s * P_s) / (Phi_fm + r * Phi_s)

   with Phi the confidence weight of Equation (6) and r the consensus of
   Equation (13).  The fused prediction is scored; each model alone is kept as
   a diagnostic.

    python src/evaluate.py --method dinov3_dual
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from baselines.dinov3_single import FullImageSegmenter
from common.paths import DINOV3_FULL_IMAGE_HEAD_CHECKPOINT
from common.protocol import image_to_tensor
from common.redifm import (
    confidence_weight,
    consensus_score,
    nested_foundation_classes,
    structure_probabilities,
)
from common.tasks import Task
from methods import redifm_dinov3


def fuse(
    source_structures: torch.Tensor,
    foundation_structures: torch.Tensor,
    consensus: torch.Tensor,
) -> torch.Tensor:
    """Equation (16) as a per-pixel weighted mean of the two OD/OC probabilities."""
    source_weight = consensus * confidence_weight(source_structures)
    foundation_weight = confidence_weight(foundation_structures)
    return (foundation_weight * foundation_structures + source_weight * source_structures) / (
        foundation_weight + source_weight
    )


def add_arguments(parser: argparse._ArgumentGroup) -> None:
    # The crop teacher's options (DINOv3 location, crop scale, crop head) are
    # those of methods/redifm_dinov3.py, with its defaults.
    redifm_dinov3.add_arguments(parser)
    parser.add_argument("--full-head-checkpoint", type=Path, default=DINOV3_FULL_IMAGE_HEAD_CHECKPOINT)


def build(args: argparse.Namespace, task: Task):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # The crop teacher owns the frozen DINOv3; the full-image head shares it.
    args.candidates = "head"
    redifm_dinov3.resolve_checkpoints(args, task)
    crop_teacher = redifm_dinov3.DinoV3Segmenter(args, device)
    full_image = FullImageSegmenter(crop_teacher.model, args.full_head_checkpoint, args.dinov3_arch, device)
    print(f"Foundation model: DINOv3 {args.dinov3_arch} on {device} (full image + crop), no adaptation")

    def segment(image: Image.Image) -> tuple[torch.Tensor, dict]:
        image_tensor = image_to_tensor(image, device)
        full_classes = full_image(image_tensor)
        full_structures = structure_probabilities(full_classes)
        full_masks = full_structures > args.pseudo_threshold
        crop_masks, crop_structures, _ = crop_teacher.supervision(
            image, image_tensor, full_structures, args.pseudo_threshold
        )
        consensus = consensus_score(full_masks, crop_masks)
        fused = fuse(full_structures, crop_structures, consensus)
        return nested_foundation_classes(fused).argmax(dim=1)[0], {
            "consensus": float(consensus),
            "full_image": full_classes.argmax(dim=1)[0],
            "crop": nested_foundation_classes(crop_structures).argmax(dim=1)[0],
        }

    return segment
