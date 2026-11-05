"""Continual test-time adaptation of a CNN, taught by DINOv3 and the same CNN on crops.

The CNN is nnU-Net on prostate (``training/train_nnunet.py``) and ResUNet-34 on OD/OC
(``training/train_source.py``).  For each target image:

* Frozen DINOv3 + the head trained on whole domain-A images predicts a mask, which
  places the crop.
* The frozen CNN trained on enlarged domain-A crops (``--crop``) predicts on the crop,
  enlarged to its input size in the same way: the pseudo ground truth.  Outside the
  crop is background.
* Target model: the CNN starting from its whole-image domain-A checkpoint.  It learns
  from the pseudo ground truth on an augmented image (or the image itself, with
  --no-augmentation), one online update, and then
  predicts the original image; that prediction is what gets scored.

The loss (``--loss``):

* ``paper`` (default): ReDi-FM's own loss (Eqs. 6-16): the crop CNN's mask, DINOv3's mask
  scaled by the consensus, and the hard class-balanced soft loss.
* ``searched``: ``teacher_loss``, chosen in ``extras/loss_search/`` on labeled target images:
  the crop CNN's mask, DINOv3's mask scaled by the agreement on each structure, and
  distillation from the two teachers fused per pixel.  The earlier Crop-Teach results.

Either way batch-norm statistics stay frozen and the learning rate is 2e-5.

    python src/evaluate.py --method redifm_dinov3_nnunet_crop [--task prostate] [--seed 0]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torch import nn

from common.dinov3 import SegmentationHead, dinov3_tokens, load_dinov3
from common.models import build_network
from common.paths import (
    POLYP_NNUNET_SHARP_CROP_CHECKPOINT,
    POLYP_RESUNET_CROP_CHECKPOINT,
    PROSTATE_NNUNET_CROP_CHECKPOINT,
    RESUNET_CROP_CHECKPOINT,
    RESUNET_SHARP_CROP_CHECKPOINT,
    SOURCE_CHECKPOINTS,
    require_file,
)
from common.protocol import binary_dice_numpy, image_to_tensor, make_augmentation, predict_after_update
from common.redifm import (
    confidence_weight,
    consensus_score,
    hard_class_balanced_soft_loss,
    nested_foundation_classes,
    structure_probabilities,
    weighted_binary_segmentation_loss,
)
from common.tasks import Task
from methods import redifm_dinov3
from methods.redifm_dinov3 import PAPER_DO_A, DinoV3Segmenter  # noqa: F401  (PAPER_DO_A: read by evaluate.py)
from methods.redifm_dinov3_dual import FULL_IMAGE_HEAD_CHECKPOINTS, FullImageNetwork, load_full_image_network


# The CNN of each task (polyp: nnU, the only polyp CNN with a native-res crop model).
FOUNDATION_ARCH = {"odoc": "resunet34", "prostate": "nnunet_official", "polyp": "nnunet_official"}
# Crop-trained checkpoints by (task, CNN, --crop-from). "original" = native-res crops (the default);
# prostate slices are already at native size, so its one crop model serves both.
FOUNDATION_CHECKPOINTS = {
    ("odoc", "resunet34", "original"): RESUNET_SHARP_CROP_CHECKPOINT,
    ("odoc", "resunet34", "resized"): RESUNET_CROP_CHECKPOINT,
    ("prostate", "nnunet_official", "original"): PROSTATE_NNUNET_CROP_CHECKPOINT,
    ("prostate", "nnunet_official", "resized"): PROSTATE_NNUNET_CROP_CHECKPOINT,
    ("polyp", "nnunet_official", "original"): POLYP_NNUNET_SHARP_CROP_CHECKPOINT,
    ("polyp", "resunet34", "resized"): POLYP_RESUNET_CROP_CHECKPOINT,
}
LEARNING_RATE = 2e-5


class NnUNetCropSegmenter:
    """Frozen crop CNN, with the supervision interface of ``DinoV3Segmenter``."""

    crop_box = DinoV3Segmenter.crop_box  # Same crop rule; it only reads self.args.

    def __init__(self, args: argparse.Namespace, task: Task, device: torch.device) -> None:
        self.args = args
        arch = args.cnn or FOUNDATION_ARCH[task.name]
        key = (task.name, arch, getattr(args, "crop_from", "resized"))
        if args.foundation_checkpoint is None and key not in FOUNDATION_CHECKPOINTS:
            raise SystemExit(f"No crop model for {key}: train one, or pass --foundation-checkpoint")
        self.checkpoint = args.foundation_checkpoint or FOUNDATION_CHECKPOINTS[key]
        model = build_network(arch, task.num_classes, args.image_size)
        model.load_state_dict(torch.load(require_file(self.checkpoint, "Crop checkpoint"),
                                         map_location="cpu", weights_only=True))
        model.requires_grad_(False).to(device).eval()
        self.model = model

    def predict_crop(
        self, image_tensor: torch.Tensor, box: tuple[int, int, int], full: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Class probabilities for the whole image, from the pixels inside the box.

        With ``full`` (the full-size image) the box's pixels are cut from it instead.
        """
        top, left, side = box
        if full is None:
            crop = image_tensor[:, :, top : top + side, left : left + side]
        else:
            scale_y, scale_x = full.shape[-2] / image_tensor.shape[-2], full.shape[-1] / image_tensor.shape[-1]
            crop = full[:, :, round(top * scale_y) : round((top + side) * scale_y),
                        round(left * scale_x) : round((left + side) * scale_x)]
        probabilities = self.crop_probabilities(crop, image_tensor.shape[-1], antialias=full is not None)
        pasted = torch.zeros_like(probabilities)
        pasted[:, :, top : top + side, left : left + side] = F.interpolate(
            probabilities, size=(side, side), mode="bilinear", align_corners=False
        )
        return pasted

    def crop_probabilities(self, crop: torch.Tensor, size: int, antialias: bool) -> torch.Tensor:
        """Class probabilities of a crop, enlarged to the CNN's input size."""
        enlarged = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False, antialias=antialias)
        return F.softmax(self.model(enlarged), dim=1)

    @torch.inference_mode()
    def supervision(
        self,
        image: Image.Image,
        image_tensor: torch.Tensor,
        source_structures: torch.Tensor,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        source_soft = source_structures[0].detach().cpu().numpy()
        source_binary = source_soft > threshold
        if self.args.crop_region == "all":  # One crop per region, as in baselines/dinov3_crop.py.
            components, count = ndimage.label(source_binary[0])
            boxes = [self.crop_box(components == index, source_soft[0]) for index in range(1, count + 1)]
            boxes = boxes or [self.crop_box(source_binary[0], source_soft[0])]
        else:
            boxes = [self.crop_box(source_binary[0], source_soft[0])]
        full = image_to_tensor(image.info["original"], image_tensor.device) if self.args.crop_from == "original" else None
        merged = crop_supervision(self, image_tensor, boxes, full)
        self.last_classes = merged  # The crop teacher's own prediction: argmax (background outside crops).
        structures = structure_probabilities(merged)
        inside = torch.zeros_like(structures)
        for top, left, side in boxes:
            inside[:, :, top : top + side, left : left + side] = 1.0
        structures = (structures * inside).clamp(1e-6, 1.0 - 1e-6)
        masks = structures > 0.5
        agreement = np.mean([
            binary_dice_numpy(masks[0, index].cpu().numpy(), source_binary[index])
            for index in range(masks.shape[1])
        ])
        return masks, structures.float(), {"foundation_crops": float(len(boxes)),
                                           "foundation_crop_side": float(max(box[2] for box in boxes)),
                                           "foundation_agreement": float(agreement)}


class DinoV3CropSegmenter(NnUNetCropSegmenter):
    """Frozen DINOv3 + crop head in the crop CNN's place (``--crop-placer source``)."""

    def __init__(self, args: argparse.Namespace, task: Task, device: torch.device) -> None:
        self.args = args
        self.checkpoint = args.head_checkpoint
        checkpoint = torch.load(require_file(self.checkpoint, "DINOv3 crop head"), map_location="cpu",
                                weights_only=False)
        self.backbone = load_dinov3(args.dinov3_dir, checkpoint["dinov3_arch"], device)
        self.layers = tuple(checkpoint["layers"])
        self.resolution = checkpoint["crop_resolution"]
        self.head = SegmentationHead(self.backbone.embed_dim * len(self.layers), checkpoint["hidden"],
                                     checkpoint["num_classes"])
        self.head.load_state_dict(checkpoint["state_dict"])
        self.head.requires_grad_(False).to(device).eval()

    def crop_probabilities(self, crop: torch.Tensor, size: int, antialias: bool) -> torch.Tensor:
        resolution = (self.resolution, self.resolution)
        crop = F.interpolate(crop, size=resolution, mode="bilinear", align_corners=False, antialias=antialias)
        logits = self.head(dinov3_tokens(self.backbone, crop, self.layers))
        return F.softmax(F.interpolate(logits, size=resolution, mode="bilinear", align_corners=False), dim=1)


def crop_supervision(
    segmenter: NnUNetCropSegmenter,
    image_tensor: torch.Tensor,
    crops: list[tuple[int, int, int]],
    full: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class probabilities from every crop; each pixel keeps the crop most sure of foreground."""
    merged = None
    with torch.inference_mode():
        for crop in crops:
            pasted = segmenter.predict_crop(image_tensor, crop, full)
            if merged is None:
                merged = pasted
            else:
                better = pasted[:, 1:].sum(dim=1, keepdim=True) > merged[:, 1:].sum(dim=1, keepdim=True)
                merged = torch.where(better, pasted, merged)
    return merged.clone()


def teacher_loss(
    target_classes: torch.Tensor,
    source: torch.Tensor,
    source_masks: torch.Tensor,
    foundation: torch.Tensor,
    foundation_masks: torch.Tensor,
    consensus: torch.Tensor,
) -> torch.Tensor:
    target = structure_probabilities(target_classes)
    source_weight = confidence_weight(source)
    foundation_weight = confidence_weight(foundation)

    # Hard terms: the crop CNN always counts, DINOv3 by its agreement on each structure.
    loss = weighted_binary_segmentation_loss(target, foundation_masks, foundation_weight)
    for index in range(target.shape[1]):
        keep = slice(index, index + 1)
        agreement = consensus_score(source_masks[:, keep], foundation_masks[:, keep])
        loss = loss + agreement * weighted_binary_segmentation_loss(
            target[:, keep], source_masks[:, keep], source_weight[:, keep]
        )

    # Soft term: cross-entropy from the teachers fused per pixel by their confidence.
    fused_classes = nested_foundation_classes(fuse_teachers(source, foundation, consensus))
    return loss - (fused_classes * target_classes.clamp_min(1e-6).log()).sum(dim=1).mean()


def paper_loss(
    target_classes: torch.Tensor,
    source_classes: torch.Tensor,
    source: torch.Tensor,
    source_masks: torch.Tensor,
    foundation: torch.Tensor,
    foundation_masks: torch.Tensor,
    consensus: torch.Tensor,
    entropy_threshold: float,
) -> torch.Tensor:
    """ReDi-FM's loss (Eqs. 6-16), with the crop CNN in SAM's place (``baseline`` in the loss search)."""
    target = structure_probabilities(target_classes)
    hard = weighted_binary_segmentation_loss(target, foundation_masks, confidence_weight(foundation))
    hard = hard + consensus * weighted_binary_segmentation_loss(target, source_masks, confidence_weight(source))
    soft = hard_class_balanced_soft_loss(
        target_classes, source_classes, nested_foundation_classes(foundation), consensus, entropy_threshold
    )
    return hard + soft


def fuse_teachers(source: torch.Tensor, foundation: torch.Tensor, consensus: torch.Tensor) -> torch.Tensor:
    """Both teachers per pixel, weighted by their confidence; DINOv3 also by the consensus."""
    foundation_weight = confidence_weight(foundation)
    scaled_source_weight = consensus * confidence_weight(source)
    fused = (foundation_weight * foundation + scaled_source_weight * source) / (
        foundation_weight + scaled_source_weight
    ).clamp_min(1e-6)
    return fused.clamp(1e-6, 1.0 - 1e-6)


def adaptation_step(
    source_model: nn.Module,
    target_model: nn.Module,
    segmenter: NnUNetCropSegmenter,
    optimizer: torch.optim.Optimizer,
    augmentation: nn.Module | None,
    image: Image.Image,
    image_tensor: torch.Tensor,
    threshold: float,
    details: dict | None = None,
    loss_name: str = "paper",
    entropy_threshold: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """The frozen teachers supervise one online update of the target model.

    ``details``, if given, receives the supervision of this step, for inspection.
    """
    with torch.inference_mode():
        source_classes = F.softmax(source_model(image_tensor), dim=1)
        source = structure_probabilities(source_classes)
        source_masks = source > threshold
    foundation_masks, foundation, foundation_diagnostics = segmenter.supervision(image, image_tensor, source, threshold)
    consensus = consensus_score(source_masks, foundation_masks).detach()

    target_model.train()
    for module in target_model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()  # Batches of one image would make these statistics noisy.
    optimizer.zero_grad(set_to_none=True)
    augmented = image_tensor if augmentation is None else augmentation(image_tensor.squeeze(0)).unsqueeze(0)
    target_classes = F.softmax(target_model(augmented), dim=1)
    if loss_name == "paper":
        loss = paper_loss(target_classes, source_classes, source, source_masks, foundation, foundation_masks,
                          consensus, entropy_threshold)
    else:
        loss = teacher_loss(target_classes, source, source_masks, foundation, foundation_masks, consensus)
    loss.backward()
    optimizer.step()
    if details is not None:
        details.update(
            augmented=augmented.detach(),
            source_classes=source_classes,
            source=source,
            foundation=foundation,
            fused=fuse_teachers(source, foundation, consensus),
            agreements=[float(consensus_score(source_masks[:, k : k + 1], foundation_masks[:, k : k + 1]))
                        for k in range(source.shape[1])],
            loss=float(loss.detach()),
        )
    return float(consensus), foundation_diagnostics


def add_arguments(parser: argparse._ArgumentGroup) -> None:
    # The crop rule and the adaptation's options are those of redifm_dinov3.
    redifm_dinov3.add_arguments(parser)
    parser.add_argument("--cnn", choices=("resunet34", "nnunet_official"),
                        help="Crop teacher and target network (default: that of --task); "
                             "pass its --foundation-checkpoint when it differs")
    parser.add_argument("--full-head-checkpoint", type=Path,
                        help="Default: the full-image DINOv3 head trained for --task")
    parser.add_argument("--foundation-checkpoint", type=Path,
                        help="Default: the crop-trained CNN of --task")
    parser.add_argument("--crop-from", choices=("resized", "original"), default="original",
                        help="Cut the crop CNN's pixels from the full-size image (native-res, the default) "
                             "or the resized one (low-res); the default crop model matches it")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="The target learns from the test image itself instead of an augmented copy")
    parser.add_argument("--loss", choices=("paper", "searched"), default="paper",
                        help="paper: ReDi-FM's loss; searched: teacher_loss from extras/loss_search "
                             "(chosen on labeled target images; the earlier Crop-Teach results)")
    parser.add_argument("--target", choices=("cnn", "dinov3_head"), default="cnn",
                        help="What adapts: the whole-image CNN, or a copy of the full-image DINOv3 head "
                             "on the same frozen DINOv3")
    parser.add_argument("--target-checkpoint", type=Path,
                        help="Where the target starts. Default: the whole-image domain-A CNN of --task "
                             "(with --target dinov3_head: the --full-head-checkpoint)")


def load_models(args: argparse.Namespace, task: Task) -> tuple:
    """(device, source model, target model, segmenter, optimizer, augmentation)."""
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.full_head_checkpoint = args.full_head_checkpoint or FULL_IMAGE_HEAD_CHECKPOINTS[task.name]
    args.learning_rate = args.learning_rate or LEARNING_RATE
    arch = args.cnn or FOUNDATION_ARCH[task.name]
    if args.target == "dinov3_head":
        target_checkpoint = args.target_checkpoint or args.full_head_checkpoint
    else:
        target_checkpoint = args.target_checkpoint or SOURCE_CHECKPOINTS[(task.name, arch)]
    # For the saved config: the source is the full-image head; arch is the crop CNN.
    args.arch, args.source_checkpoint, args.target_checkpoint = arch, args.full_head_checkpoint, target_checkpoint

    backbone = load_dinov3(args.dinov3_dir, args.dinov3_arch, device)
    source_model = load_full_image_network(backbone, args.full_head_checkpoint, args.dinov3_arch)
    source_model.requires_grad_(False).to(device).eval()
    if args.target == "dinov3_head":
        # A trainable head on the same frozen DINOv3 (shared, not copied): by default a copy of
        # the crop-placing head, or another full-image head given with --target-checkpoint.
        if target_checkpoint == args.full_head_checkpoint:
            head = copy.deepcopy(source_model.head)
            target_model = FullImageNetwork(backbone, head, source_model.layers, source_model.resolution)
        else:
            target_model = load_full_image_network(backbone, target_checkpoint, args.dinov3_arch)
        target_model.head.requires_grad_(True)
        target_name = "DINOv3 full-image head"
    else:
        target_model = build_network(arch, task.num_classes, args.image_size)
        target_model.load_state_dict(torch.load(require_file(target_checkpoint, "Target checkpoint"),
                                                 map_location="cpu", weights_only=True))
        target_name = arch
    target_model.to(device).train()
    segmenter = NnUNetCropSegmenter(args, task, device)
    optimizer = torch.optim.Adam(target_model.parameters(), lr=args.learning_rate)
    augmentation = None if args.no_augmentation else make_augmentation(args)
    args.foundation_checkpoint = segmenter.checkpoint
    print(f"Crop placement: DINOv3 {args.dinov3_arch} + full-image head at {source_model.resolution} on {device}; "
          f"pseudo ground truth: {arch} on the crop ({segmenter.checkpoint.name}); "
          f"target: {target_name} ({target_checkpoint.name}), one online update per image")
    return device, source_model, target_model, segmenter, optimizer, augmentation


def build(args: argparse.Namespace, task: Task):
    device, source_model, target_model, segmenter, optimizer, augmentation = load_models(args, task)

    def segment(image: Image.Image) -> tuple[torch.Tensor, dict]:
        image_tensor = image_to_tensor(image, device)
        details: dict = {}
        consensus, foundation = adaptation_step(
            source_model, target_model, segmenter, optimizer, augmentation, image, image_tensor, args.pseudo_threshold,
            details=details, loss_name=args.loss, entropy_threshold=args.entropy_threshold,
        )
        prediction = predict_after_update(target_model, image_tensor)[0]
        # Both frozen teachers' own predictions, scored as no-adaptation variants by evaluate.py:
        # "head_alone" = dinov3_single, "crop_teacher" = dinov3_crop (same crops), at no extra cost.
        teachers = {"head_alone": details["source_classes"].argmax(dim=1)[0],
                    "crop_teacher": segmenter.last_classes.argmax(dim=1)[0]}
        return prediction, {"consensus": consensus, **foundation, **teachers}

    return segment
