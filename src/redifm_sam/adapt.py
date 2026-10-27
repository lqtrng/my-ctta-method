#!/usr/bin/env python3
"""ReDi-FM continual test-time adaptation for OD/OC segmentation.

This implements Sections 3.1-3.3 of the included ReDi-FM paper for the
Domain-A ResUNet-34 checkpoint and the B -> C -> D -> E target stream:

* a frozen source model creates OD/OC pseudo-labels;
* center and hard points prompt a frozen SAM ViT-H model;
* SAM candidates are selected by Dice agreement with the source anchor;
* the online target model receives uncertainty-weighted hard distillation,
  hard class-balanced soft distillation, and consensus-aware gating;
* exactly one Adam update is made for each target image (batch size one).

The paper does not publish the threshold used for "high" source entropy or
the precise color-jitter/random-erasing magnitudes.  They are therefore
explicit command-line settings below and are saved in the output JSON.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.data import Sample, atomic_json_write, build_domains  # noqa: E402
from common.models import NUM_CLASSES, ResUnet34  # noqa: E402
from common.paths import DEFAULT_SOURCE_CHECKPOINT, RESULTS_DIR, SAM_CHECKPOINT, require_file  # noqa: E402
from common.protocol import (  # noqa: E402
    DomainResult,
    binary_dice_numpy,
    image_dice,
    image_to_tensor,
    make_augmentation,
    predict_after_update,
    prepare_sample,
    seed_everything,
)
from common.redifm import (  # noqa: E402
    STRUCTURES,
    StepDiagnostics,
    confidence_weight,
    consensus_score,
    hard_class_balanced_soft_loss,
    nested_foundation_classes,
    structure_aware_points,
    structure_probabilities,
    weighted_binary_segmentation_loss,
)


DEFAULT_SAM_CHECKPOINT = SAM_CHECKPOINT
DEFAULT_OUTPUT = RESULTS_DIR / "redifm_sam/proposed_redifm_domain_A.json"


def require_sam() -> tuple[Any, Any]:
    try:
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as error:
        raise SystemExit(
            "Missing Segment Anything. Install it with:\n"
            "  uv pip install --python .venv/bin/python "
            "git+https://github.com/facebookresearch/segment-anything.git"
        ) from error
    return SamPredictor, sam_model_registry


def select_sam_candidate(
    predictor: Any,
    points: np.ndarray,
    source_anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Equation (5): choose the SAM candidate closest to the source anchor."""
    logits, quality, _ = predictor.predict(
        point_coords=points,
        point_labels=np.ones(len(points), dtype=np.int32),
        multimask_output=True,
        return_logits=True,
    )
    candidate_masks = logits > 0.0
    agreements = np.asarray(
        [binary_dice_numpy(mask, source_anchor) for mask in candidate_masks]
    )
    selected = int(np.argmax(agreements))
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits[selected], -30.0, 30.0)))
    return candidate_masks[selected], probability.astype(np.float32), float(quality[selected])


@torch.inference_mode()
def foundation_supervision(
    predictor: Any,
    image: Image.Image,
    source_structure_probabilities: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Generate binary masks and probabilities from frozen SAM for OD and OC."""
    source_soft = source_structure_probabilities[0].detach().cpu().numpy()
    source_binary = source_soft > threshold
    sam_device = predictor.device
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if sam_device.type == "cuda"
        else nullcontext()
    )

    masks: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    qualities: dict[str, float] = {}
    try:
        # ViT-H attention is memory-heavy at SAM's fixed 1024-pixel input.
        # Mixed precision is confined to frozen SAM inference.
        with amp_context:
            predictor.set_image(np.asarray(image, dtype=np.uint8))
            for index, structure in enumerate(STRUCTURES):
                points = structure_aware_points(
                    image, source_binary[index], source_soft[index]
                )
                mask, probability, quality = select_sam_candidate(
                    predictor, points, source_binary[index]
                )
                masks.append(mask)
                probabilities.append(probability)
                qualities[f"sam_{structure}_quality"] = quality
    finally:
        predictor.reset_image()
    device = source_structure_probabilities.device
    mask_tensor = torch.from_numpy(np.stack(masks)).unsqueeze(0).to(device=device)
    probability_tensor = torch.from_numpy(np.stack(probabilities)).unsqueeze(0).to(device)
    return mask_tensor.bool(), probability_tensor.float(), qualities


def adaptation_step(
    source_model: nn.Module,
    target_model: nn.Module,
    predictor: Any,
    optimizer: torch.optim.Optimizer,
    augmentation: nn.Module,
    image: Image.Image,
    image_tensor: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[StepDiagnostics, dict[str, float]]:
    """Perform the paper's one online update for a single target image."""
    source_model.eval()
    with torch.inference_mode():
        source_device = next(source_model.parameters()).device
        source_input = image_tensor.to(source_device)
        source_classes = F.softmax(source_model(source_input), dim=1).to(
            image_tensor.device
        )
        source_structures = structure_probabilities(source_classes)
        source_masks = source_structures > args.pseudo_threshold

    if image_tensor.is_cuda:
        torch.cuda.empty_cache()
    foundation_masks, foundation_structures, sam_diagnostics = foundation_supervision(
        predictor,
        image,
        source_structures,
        args.pseudo_threshold,
    )
    # SAM and the target model run sequentially. Once reset_image() has dropped
    # SAM's image embedding, release its cached CUDA workspace before retaining
    # target-model activations for backpropagation.
    if image_tensor.is_cuda:
        torch.cuda.empty_cache()
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
    return diagnostics, sam_diagnostics


def evaluate_stream(
    source_model: nn.Module,
    target_model: nn.Module,
    predictor: Any,
    optimizer: torch.optim.Optimizer,
    domains: dict[str, list[Sample]],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[DomainResult], list[dict[str, str | float]]]:
    augmentation = make_augmentation(args)
    results: list[DomainResult] = []
    per_image: list[dict[str, str | float]] = []

    for domain, samples in domains.items():
        disc_scores: list[float] = []
        cup_scores: list[float] = []
        for sample in tqdm(samples, desc=f"ReDi-FM domain {domain}"):
            image, target = prepare_sample(sample, args.image_size)
            image_tensor = image_to_tensor(image, device)
            target = target.to(device)

            adaptation_step(
                source_model,
                target_model,
                predictor,
                optimizer,
                augmentation,
                image,
                image_tensor,
                args,
            )
            prediction = predict_after_update(target_model, image_tensor)[0]
            disc_score = image_dice(prediction >= 1, target >= 1)
            cup_score = image_dice(prediction == 2, target == 2)
            disc_scores.append(disc_score)
            cup_scores.append(cup_score)
            per_image.append(
                {
                    "domain": domain,
                    "image": sample.identifier,
                    "disc_dsc": disc_score,
                    "cup_dsc": cup_score,
                    "mean_dsc": (disc_score + cup_score) / 2.0,
                }
            )

        disc_mean = float(np.mean(disc_scores))
        cup_mean = float(np.mean(cup_scores))
        result = DomainResult(
            domain=domain,
            dataset=samples[0].dataset,
            images=len(samples),
            disc_dsc=100.0 * disc_mean,
            cup_dsc=100.0 * cup_mean,
            mean_dsc=100.0 * (disc_mean + cup_mean) / 2.0,
        )
        results.append(result)
        print(
            f"Domain {domain} ({result.dataset}, n={result.images}): "
            f"OD={result.disc_dsc:.2f}, OC={result.cup_dsc:.2f}, "
            f"Mean={result.mean_dsc:.2f}"
        )
    return results, per_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-adapted-checkpoint", type=Path)
    parser.add_argument("--domains", nargs="+", choices=list("BCDE"), default=list("BCDE"))
    parser.add_argument("--max-images", type=int, help="Limit images per domain for a smoke test")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pseudo-threshold", type=float, default=0.5)
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.5,
        help="Normalized source-entropy threshold; not specified by the paper",
    )
    parser.add_argument("--jitter-brightness", type=float, default=0.2)
    parser.add_argument("--jitter-contrast", type=float, default=0.2)
    parser.add_argument("--jitter-saturation", type=float, default=0.2)
    parser.add_argument("--jitter-hue", type=float, default=0.05)
    parser.add_argument("--erasing-probability", type=float, default=0.5)
    parser.add_argument("--erasing-min-scale", type=float, default=0.02)
    parser.add_argument("--erasing-max-scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Trainable target-model device (default: cpu/system RAM)",
    )
    parser.add_argument(
        "--source-device",
        default="cpu",
        help="Frozen source-model device (default: cpu/system RAM)",
    )
    parser.add_argument(
        "--sam-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="SAM device (default: cuda when available)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    require_file(args.source_checkpoint, "Source checkpoint")
    require_file(args.sam_checkpoint, "SAM checkpoint")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be positive")
    if args.image_size < 1 or args.learning_rate <= 0:
        raise ValueError("image size and learning rate must be positive")
    if not 0.0 <= args.pseudo_threshold <= 1.0:
        raise ValueError("--pseudo-threshold must be in [0, 1]")
    if not 0.0 <= args.entropy_threshold <= 1.0:
        raise ValueError("--entropy-threshold must be in [0, 1]")
    if not 0.0 <= args.erasing_probability <= 1.0:
        raise ValueError("--erasing-probability must be in [0, 1]")
    if not 0.0 < args.erasing_min_scale <= args.erasing_max_scale <= 1.0:
        raise ValueError("invalid random-erasing scale interval")


def load_models(
    source_checkpoint: Path,
    sam_checkpoint: Path,
    target_device: torch.device,
    source_device: torch.device,
    sam_device: torch.device,
) -> tuple[nn.Module, nn.Module, Any]:
    state_dict = torch.load(
        source_checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    source_model = ResUnet34(num_classes=NUM_CLASSES)
    source_model.load_state_dict(state_dict, strict=True)
    source_model.requires_grad_(False).to(source_device).eval()

    target_model = ResUnet34(num_classes=NUM_CLASSES)
    target_model.load_state_dict(state_dict, strict=True)
    target_model.to(target_device).train()

    SamPredictor, sam_model_registry = require_sam()
    sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
    sam.requires_grad_(False).to(sam_device).eval()
    return source_model, target_model, SamPredictor(sam)


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    seed_everything(args.seed)
    device = torch.device(args.device)
    source_device = torch.device(args.source_device)
    sam_device = torch.device(args.sam_device)
    domains = build_domains(args.domains, args.max_images)
    source_model, target_model, predictor = load_models(
        args.source_checkpoint.resolve(),
        args.sam_checkpoint.resolve(),
        device,
        source_device,
        sam_device,
    )
    optimizer = torch.optim.Adam(target_model.parameters(), lr=args.learning_rate)

    print(f"Checkpoint: {args.source_checkpoint.resolve()}")
    print(f"Target device: {device}")
    print(f"Source device: {source_device}")
    print(f"SAM device: {sam_device}")
    print("Protocol: ReDi-FM continual test-time adaptation (one online update per image)")
    print("Metric: mean per-image DSC; OD includes the OC region")

    results, per_image = evaluate_stream(
        source_model,
        target_model,
        predictor,
        optimizer,
        domains,
        device,
        args,
    )
    do_a = float(np.mean([result.mean_dsc for result in results]))
    pooled_image_mean = float(np.mean([row["mean_dsc"] for row in per_image])) * 100.0
    print(f"Do. A DSC (unweighted B/C/D/E domain mean): {do_a:.2f}")
    print(f"Pooled target-image DSC (diagnostic only): {pooled_image_mean:.2f}")

    payload = {
        "protocol": "redifm_continual_test_time_adaptation",
        "source_domain": "A",
        "checkpoint": str(args.source_checkpoint.resolve()),
        "target_device": str(device),
        "source_device": str(source_device),
        "sam_device": str(sam_device),
        "image_size": args.image_size,
        "target_domains": [asdict(result) for result in results],
        "do_a_dsc": do_a,
        "pooled_target_image_dsc": pooled_image_mean,
        "paper_redifm_do_a_reference": 73.11,
        "per_image": per_image,
    }
    atomic_json_write(args.output, payload)
    print(f"Saved: {args.output.resolve()}")

    if args.save_adapted_checkpoint is not None:
        destination = args.save_adapted_checkpoint.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(target_model.state_dict(), destination)
        print(f"Saved adapted target model: {destination}")


if __name__ == "__main__":
    main(parse_args())
