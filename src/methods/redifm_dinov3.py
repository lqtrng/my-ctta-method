"""ReDi-FM continual test-time adaptation with a frozen DINOv3 foundation model.

This is a variant of ``methods/redifm_sam.py`` in which the frozen,
promptable SAM is replaced by a frozen, self-supervised DINOv3 ViT.  Everything
that defines the evaluation protocol (data stream, preprocessing, one Adam
update per image at batch size one, scoring after the update, per-image DSC)
is shared with every other method through ``src/evaluate.py``:

    python src/evaluate.py --method redifm_dinov3 [--task prostate] [--arch nnunet] [--rounds 3]

DINOv3 has no mask decoder, so Section 3.2 of the paper is re-expressed for a
feature backbone:

* Prompting.  The source pseudo-label defines a region of interest around the
  optic disc, which is cropped and encoded by DINOv3 at high resolution.
* Candidates (``--candidates``).  ``head`` (default): a small decoder trained
  once on labeled source domain A (``training/train_dinov3_head.py``) maps the frozen
  DINOv3 tokens of the crop to OD/OC probabilities -- the counterpart of SAM's
  pretrained mask decoder.  ``prototype``: training-free alternative in which
  the structure-aware center and hard points (Equations 2-4) and the confident
  source region become foreground prototypes, k-means of the confident source
  background become background prototypes, and
  ``p_fm = sigmoid((s_fg - s_bg) / T)``.  As in Equation (5), the candidate
  with the highest Dice agreement with the source pseudo-label is selected.
* Learning.  The selected DINOv3 mask/probability take the place of SAM's in
  the unchanged objective: uncertainty-weighted hard distillation (6)-(8),
  hard class-balanced soft distillation (10)-(12), and consensus gating
  (13)-(16).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from common.dinov3 import IMAGENET_MEAN, IMAGENET_STD, SegmentationHead
from common.models import ARCHITECTURES, build_network
from common.paths import (
    DINOV3_ARCH,
    DINOV3_HEAD_CHECKPOINT,
    MODEL_DIR,
    PROSTATE_DINOV3_HEAD_CHECKPOINT,
    SOURCE_CHECKPOINTS,
    dinov3_weights,
    require_file,
)
from common.protocol import (
    binary_dice_numpy,
    image_to_tensor,
    make_augmentation,
    predict_after_update,
)
from common.redifm import (
    StepDiagnostics,
    confidence_weight,
    consensus_score,
    hard_class_balanced_soft_loss,
    largest_component,
    nested_foundation_classes,
    structure_aware_points,
    structure_probabilities,
    weighted_binary_segmentation_loss,
)
from common.tasks import TASKS, Task


HEAD_CHECKPOINTS = {"odoc": DINOV3_HEAD_CHECKPOINT, "prostate": PROSTATE_DINOV3_HEAD_CHECKPOINT}
# 0140.pdf Table 1, ReDi-FM with SAM, source domain A.
PAPER_DO_A = {"odoc": 73.11, "prostate": 66.10}


def resolve_checkpoints(args: argparse.Namespace, task: Task) -> None:
    """Fill checkpoints left unset with the ones of ``--task`` and ``--arch``."""
    args.source_checkpoint = args.source_checkpoint or SOURCE_CHECKPOINTS[(task.name, args.arch)]
    args.head_checkpoint = args.head_checkpoint or HEAD_CHECKPOINTS[task.name]


class DinoV3Segmenter:
    """Frozen DINOv3 turned into a prompt-conditioned binary segmenter."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        import timm

        weights = require_file(dinov3_weights(args.dinov3_dir, args.dinov3_arch), "DINOv3 weights")
        self.model = timm.create_model(
            args.dinov3_arch.split(".")[0],
            pretrained=True,
            dynamic_img_size=True,
            pretrained_cfg_overlay=dict(file=str(weights)),
        )
        self.model.requires_grad_(False).to(device).eval()
        self.patch = self.model.patch_embed.patch_size[0]
        self.device = device
        self.args = args
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        self.head: nn.Module | None = None
        self.layers: list[int] = [len(self.model.blocks) - 1]
        self.structures = TASKS[args.task].structures
        if args.candidates != "prototype":
            checkpoint = torch.load(args.head_checkpoint, map_location="cpu", weights_only=False)
            if checkpoint["dinov3_arch"] != args.dinov3_arch:
                raise ValueError("Decoder head was trained on a different DINOv3 variant")
            if checkpoint.get("task", "odoc") != args.task:
                raise ValueError(f"Decoder head was trained for task {checkpoint.get('task', 'odoc')!r}")
            self.layers = list(checkpoint["layers"])
            self.head = SegmentationHead(
                self.model.embed_dim * len(self.layers),
                checkpoint["hidden"],
                checkpoint.get("num_classes", 3),
            )
            self.head.load_state_dict(checkpoint["state_dict"])
            self.head.requires_grad_(False).to(device).eval()

    def crop_box(self, disc_mask: np.ndarray, disc_probability: np.ndarray) -> tuple[int, int, int]:
        """Square region of interest centered on the source optic disc."""
        size = disc_mask.shape[0]
        component = largest_component(disc_mask)
        if not component.any():
            row, column = np.unravel_index(np.argmax(disc_probability), disc_probability.shape)
            component[row, column] = True
        rows, columns = np.nonzero(component)
        center_y = (rows.min() + rows.max()) / 2.0
        center_x = (columns.min() + columns.max()) / 2.0
        side = max(np.ptp(rows) + 1, np.ptp(columns) + 1) * self.args.crop_scale
        side = int(np.clip(side, self.args.min_crop, size))
        top = int(np.clip(round(center_y - side / 2.0), 0, size - side))
        left = int(np.clip(round(center_x - side / 2.0), 0, size - side))
        return top, left, side

    @torch.inference_mode()
    def features(
        self, image_tensor: torch.Tensor, box: tuple[int, int, int]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Normalized patch features and, with a decoder, its OD/OC logits."""
        top, left, side = box
        crop = image_tensor[:, :, top : top + side, left : left + side]
        resolution = self.args.crop_resolution
        crop = F.interpolate(crop, size=(resolution, resolution), mode="bilinear", align_corners=False)
        # DINOv3 overflows in float16; bfloat16 keeps the range of float32.
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            tokens, intermediates = self.model.forward_intermediates(
                (crop - self.mean) / self.std, indices=self.layers
            )
        tokens = tokens[:, self.model.num_prefix_tokens :].float()
        grid = resolution // self.patch
        features = F.normalize(tokens, dim=-1).reshape(grid, grid, -1)
        if self.head is None:
            return features, None
        classes = self.head(torch.cat([layer.float() for layer in intermediates], dim=1))
        structures = structure_probabilities(classes.softmax(dim=1))[0]
        return features, torch.logit(structures, eps=1e-6)

    def background_prototypes(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Fundus background is heterogeneous (vessels, retina): use k-means."""
        samples = features[mask]
        count = min(self.args.background_prototypes, samples.shape[0])
        generator = torch.Generator(device=features.device).manual_seed(0)
        index = torch.randperm(samples.shape[0], generator=generator, device=features.device)
        centers = samples[index[:count]]
        for _ in range(10):
            assignment = (samples @ centers.T).argmax(dim=1)
            centers = torch.stack(
                [
                    samples[assignment == k].mean(dim=0) if (assignment == k).any() else centers[k]
                    for k in range(count)
                ]
            )
            centers = F.normalize(centers, dim=-1)
        return centers

    def candidate_logits(
        self,
        features: torch.Tensor,
        source_probability: torch.Tensor,
        points: np.ndarray,
        box: tuple[int, int, int],
        region: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """One logit map per prompt set on the DINOv3 patch grid.

        ``region`` restricts the background prototypes (the optic cup is
        contrasted with the disc rim rather than with the whole retina).
        """
        grid = features.shape[0]
        flat = features.reshape(-1, features.shape[-1])
        probability = source_probability.reshape(-1)
        confident = self.args.prototype_confidence
        foreground = probability > confident
        background = probability < 1.0 - confident
        if region is not None and (background & region.reshape(-1)).any():
            background = background & region.reshape(-1)
        margin = self.args.background_margin
        if margin > 0:
            # The source under-segments under shift, so pixels just outside its
            # mask are unreliable background; keep prototypes away from them.
            near = F.max_pool2d(
                (source_probability > 0.5).float()[None, None], 2 * margin + 1, 1, margin
            )[0, 0].reshape(-1).bool()
            if (background & ~near).any():
                background = background & ~near
        if not foreground.any():
            foreground = probability >= probability.max()
        if not background.any():
            background = probability <= probability.min()
        background_centers = self.background_prototypes(flat, background)

        top, left, side = box
        point_features = []
        for x, y in points:
            column = int(np.clip((x - left) / side * grid, 0, grid - 1))
            row = int(np.clip((y - top) / side * grid, 0, grid - 1))
            point_features.append(features[row, column])
        center_feature, hard_feature = point_features
        region_feature = F.normalize(flat[foreground].mean(dim=0), dim=0)

        prompt_sets = (
            torch.stack((center_feature, hard_feature)),
            region_feature[None],
            torch.stack((region_feature, hard_feature)),
        )
        background_score = (flat @ background_centers.T).max(dim=1).values
        logits = []
        for prototypes in prompt_sets:
            foreground_score = (flat @ prototypes.T).max(dim=1).values
            margin_score = foreground_score - background_score
            for _ in range(self.args.refine_iterations):
                # EM-style re-estimation: prototypes from DINOv3's own partition,
                # which lets the mask move beyond the source's size bias.
                inside = margin_score > 0
                outside = ~inside if region is None else (~inside & region.reshape(-1))
                if not inside.any() or outside.sum() < self.args.background_prototypes:
                    break
                foreground_prototype = F.normalize(flat[inside].mean(dim=0), dim=0)
                refined_background = self.background_prototypes(flat, outside)
                margin_score = flat @ foreground_prototype - (
                    flat @ refined_background.T
                ).max(dim=1).values
            logits.append((margin_score / self.args.temperature).reshape(grid, grid))
        return logits

    def paste(self, logits: torch.Tensor, box: tuple[int, int, int], size: int) -> torch.Tensor:
        """Upsample patch logits into the crop; outside the crop is background."""
        top, left, side = box
        canvas = torch.full((size, size), -30.0, device=logits.device)
        canvas[top : top + side, left : left + side] = F.interpolate(
            logits[None, None], size=(side, side), mode="bilinear", align_corners=False
        )[0, 0]
        return torch.sigmoid(canvas.clamp(-30.0, 30.0))

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
        size = source_soft.shape[-1]
        box = self.crop_box(source_binary[0], source_soft[0])
        features, head_logits = self.features(image_tensor.to(self.device), box)
        grid = features.shape[0]
        top, left, side = box
        source_grid = F.adaptive_avg_pool2d(
            source_structures[:, :, top : top + side, left : left + side].to(self.device), grid
        )[0]

        masks: list[torch.Tensor] = []
        probabilities: list[torch.Tensor] = []
        diagnostics: dict[str, float] = {}
        disc_region = None
        for index, structure in enumerate(self.structures):
            points = structure_aware_points(image, source_binary[index], source_soft[index])
            region = disc_region if structure == "cup" and self.args.cup_background == "rim" else None
            candidate_logits = []
            if head_logits is not None:
                candidate_logits.append(head_logits[index])
            if self.args.candidates != "head":
                candidate_logits += self.candidate_logits(
                    features, source_grid[index], points, box, region
                )
            candidates = [self.paste(logits, box, size) for logits in candidate_logits]
            agreements = [
                binary_dice_numpy((candidate > 0.5).cpu().numpy(), source_binary[index])
                for candidate in candidates
            ]
            selected = int(np.argmax(agreements))
            if structure == "disc":
                # Disc region on the patch grid: the selected DINOv3 disc, dilated by
                # one patch so that the rim boundary is represented.
                disc_region = F.max_pool2d(
                    (candidate_logits[selected] > 0).float()[None, None], 3, 1, 1
                )[0, 0].bool()
            probabilities.append(candidates[selected])
            masks.append(candidates[selected] > 0.5)
            diagnostics[f"dinov3_{structure}_candidate"] = float(selected)
            diagnostics[f"dinov3_{structure}_agreement"] = float(agreements[selected])
        device = source_structures.device
        mask_tensor = torch.stack(masks)[None].to(device)
        probability_tensor = torch.stack(probabilities)[None].to(device).clamp(1e-6, 1.0 - 1e-6)
        return mask_tensor, probability_tensor.float(), diagnostics


def adaptation_step(
    source_model: nn.Module,
    target_model: nn.Module,
    segmenter: DinoV3Segmenter,
    optimizer: torch.optim.Optimizer,
    augmentation: nn.Module,
    image: Image.Image,
    image_tensor: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[StepDiagnostics, dict[str, float]]:
    """The paper's one online update, with DINOv3 as the frozen teacher."""
    source_model.eval()
    with torch.inference_mode():
        source_classes = F.softmax(source_model(image_tensor), dim=1)
        source_structures = structure_probabilities(source_classes)
        source_masks = source_structures > args.pseudo_threshold

    foundation_masks, foundation_structures, foundation_diagnostics = segmenter.supervision(
        image, image_tensor, source_structures, args.pseudo_threshold
    )
    foundation_classes = nested_foundation_classes(foundation_structures)
    consensus = consensus_score(source_masks, foundation_masks).detach()

    target_model.train()
    optimizer.zero_grad(set_to_none=True)
    augmented = augmentation(image_tensor.squeeze(0)).unsqueeze(0)
    target_classes = F.softmax(target_model(augmented), dim=1)
    target_structures = structure_probabilities(target_classes)

    source_weights = confidence_weight(source_structures)
    foundation_weights = confidence_weight(foundation_structures)
    source_hard = weighted_binary_segmentation_loss(
        target_structures, source_masks, source_weights
    )
    foundation_hard = weighted_binary_segmentation_loss(
        target_structures, foundation_masks, foundation_weights
    )
    hard_loss = foundation_hard + consensus * source_hard
    soft_loss = hard_class_balanced_soft_loss(
        target_classes,
        source_classes,
        foundation_classes,
        consensus,
        args.entropy_threshold,
    )
    total_loss = hard_loss + soft_loss
    total_loss.backward()
    optimizer.step()

    diagnostics = StepDiagnostics(
        source_hard_loss=float(source_hard.detach()),
        foundation_hard_loss=float(foundation_hard.detach()),
        soft_loss=float(soft_loss.detach()),
        total_loss=float(total_loss.detach()),
        consensus=float(consensus),
    )
    return diagnostics, foundation_diagnostics


def add_arguments(parser: argparse._ArgumentGroup) -> None:
    parser.add_argument("--arch", choices=list(ARCHITECTURES), default="resunet34",
                        help="Source/target segmentation network")
    parser.add_argument("--source-checkpoint", type=Path,
                        help="Default: the domain-A checkpoint of --task and --arch")
    parser.add_argument("--dinov3-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--dinov3-arch", default=DINOV3_ARCH)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pseudo-threshold", type=float, default=0.5)
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.5,
        help="Normalized source-entropy threshold; not specified by the paper",
    )
    parser.add_argument("--crop-scale", type=float, default=2.0,
                        help="Crop side as a multiple of the source disc extent")
    parser.add_argument("--min-crop", type=int, default=96)
    parser.add_argument("--crop-resolution", type=int, default=512,
                        help="DINOv3 input resolution for the crop (multiple of 16)")
    parser.add_argument("--temperature", type=float, default=0.05,
                        help="Temperature of the prototype-similarity logits")
    parser.add_argument("--prototype-confidence", type=float, default=0.9,
                        help="Source probability defining confident prototype pixels")
    parser.add_argument("--background-prototypes", type=int, default=4)
    parser.add_argument("--background-margin", type=int, default=0,
                        help="Exclude background prototypes within this many patches of the source mask")
    parser.add_argument("--cup-background", choices=("all", "rim"), default="all",
                        help="Contrast the cup with the whole crop or only with the disc rim")
    parser.add_argument("--candidates", choices=("prototype", "head", "all"), default="head",
                        help="DINOv3 candidate masks: prompt prototypes, the source-trained decoder, or both")
    parser.add_argument("--head-checkpoint", type=Path,
                        help="Default: the DINOv3 decoder head trained for --task")
    parser.add_argument("--refine-iterations", type=int, default=0,
                        help="EM re-estimations of the prototypes from DINOv3's own partition")
    parser.add_argument("--jitter-brightness", type=float, default=0.2)
    parser.add_argument("--jitter-contrast", type=float, default=0.2)
    parser.add_argument("--jitter-saturation", type=float, default=0.2)
    parser.add_argument("--jitter-hue", type=float, default=0.05)
    parser.add_argument("--erasing-probability", type=float, default=0.5)
    parser.add_argument("--erasing-min-scale", type=float, default=0.02)
    parser.add_argument("--erasing-max-scale", type=float, default=0.2)


def build(args: argparse.Namespace, task: Task):
    resolve_checkpoints(args, task)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    state_dict = torch.load(args.source_checkpoint, map_location="cpu", weights_only=True)
    source_model = build_network(args.arch, task.num_classes, args.image_size)
    source_model.load_state_dict(state_dict, strict=True)
    source_model.requires_grad_(False).to(device).eval()
    target_model = build_network(args.arch, task.num_classes, args.image_size)
    target_model.load_state_dict(state_dict, strict=True)
    target_model.to(device).train()
    segmenter = DinoV3Segmenter(args, device)
    optimizer = torch.optim.Adam(target_model.parameters(), lr=args.learning_rate)
    augmentation = make_augmentation(args)
    print(f"Checkpoint: {Path(args.source_checkpoint).resolve()} ({args.arch})")
    print(f"Foundation model: DINOv3 {args.dinov3_arch} on {device}; one online update per image")

    def segment(image: Image.Image) -> tuple[torch.Tensor, dict]:
        image_tensor = image_to_tensor(image, device)
        step, foundation = adaptation_step(
            source_model, target_model, segmenter, optimizer, augmentation, image, image_tensor, args
        )
        prediction = predict_after_update(target_model, image_tensor)[0]
        return prediction, {"consensus": step.consensus, **foundation}

    return segment
