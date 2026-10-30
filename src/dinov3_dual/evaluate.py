#!/usr/bin/env python3
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
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.data import atomic_json_write, build_domains  # noqa: E402
from common.paths import RESULTS_DIR  # noqa: E402
from common.protocol import (  # noqa: E402
    DomainResult,
    image_dice,
    image_to_tensor,
    prepare_sample,
    seed_everything,
)
from common.redifm import (  # noqa: E402
    confidence_weight,
    consensus_score,
    nested_foundation_classes,
    structure_probabilities,
)
from dinov3_single.evaluate import FullImageSegmenter, domain_result  # noqa: E402
from redifm_dinov3.adapt import DinoV3Segmenter, build_parser as build_crop_parser  # noqa: E402
from redifm_dinov3.train_head import DEFAULT_FULL_IMAGE_OUTPUT  # noqa: E402


DEFAULT_OUTPUT = RESULTS_DIR / "dinov3_dual/dinov3_dual_domain_A.json"
VARIANTS = ("full_image", "crop", "fused")
SCORED = "fused"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-head-checkpoint", type=Path, default=DEFAULT_FULL_IMAGE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--domains", nargs="+", choices=list("BCDE"), default=list("BCDE"))
    parser.add_argument("--max-images", type=int, help="Limit images per domain for a smoke test")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--pseudo-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(args: argparse.Namespace, crop_args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    domains = build_domains(args.domains, args.max_images)

    # The crop teacher owns the frozen DINOv3; the full-image head shares it.
    crop_args.candidates = "head"
    crop_teacher = DinoV3Segmenter(crop_args, device)
    full_image = FullImageSegmenter(
        crop_teacher.model, args.full_head_checkpoint, crop_args.dinov3_arch, device
    )

    print(f"Foundation model: DINOv3 {crop_args.dinov3_arch} on {device} (full image + crop)")
    print("Protocol: no adaptation; two frozen DINOv3 predictions fused per target image")
    print("Metric: mean per-image DSC; OD includes the OC region")

    results: dict[str, list[DomainResult]] = {variant: [] for variant in VARIANTS}
    per_image: list[dict[str, Any]] = []
    for domain, samples in domains.items():
        scores = {variant: ([], []) for variant in VARIANTS}
        for sample in tqdm(samples, desc=f"DINOv3 dual domain {domain}"):
            image, target = prepare_sample(sample, args.image_size)
            image_tensor = image_to_tensor(image, device)
            target = target.to(device)

            full_classes = full_image(image_tensor)
            full_structures = structure_probabilities(full_classes)
            full_masks = full_structures > args.pseudo_threshold
            crop_masks, crop_structures, _ = crop_teacher.supervision(
                image, image_tensor, full_structures, args.pseudo_threshold
            )
            consensus = consensus_score(full_masks, crop_masks)
            fused = fuse(full_structures, crop_structures, consensus)
            predictions = {
                "full_image": full_classes.argmax(dim=1)[0],
                "crop": nested_foundation_classes(crop_structures).argmax(dim=1)[0],
                "fused": nested_foundation_classes(fused).argmax(dim=1)[0],
            }

            row: dict[str, Any] = {
                "domain": domain,
                "image": sample.identifier,
                "consensus": float(consensus),
            }
            for variant, prediction in predictions.items():
                disc_score = image_dice(prediction >= 1, target >= 1)
                cup_score = image_dice(prediction == 2, target == 2)
                scores[variant][0].append(disc_score)
                scores[variant][1].append(cup_score)
                prefix = "" if variant == SCORED else f"{variant}_"
                row[f"{prefix}disc_dsc"] = disc_score
                row[f"{prefix}cup_dsc"] = cup_score
                row[f"{prefix}mean_dsc"] = (disc_score + cup_score) / 2.0
            per_image.append(row)

        for variant, (disc_scores, cup_scores) in scores.items():
            result = domain_result(domain, samples[0].dataset, disc_scores, cup_scores)
            results[variant].append(result)
            print(
                f"Domain {domain} ({result.dataset}, n={result.images}) [{variant}]: "
                f"OD={result.disc_dsc:.2f}, OC={result.cup_dsc:.2f}, Mean={result.mean_dsc:.2f}"
            )

    do_a = {variant: float(np.mean([r.mean_dsc for r in rows])) for variant, rows in results.items()}
    for variant, value in do_a.items():
        print(f"Do. A DSC [{variant}] (unweighted {'/'.join(args.domains)} domain mean): {value:.2f}")

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in {**vars(crop_args), **vars(args)}.items()
    }
    payload = {
        "protocol": "dinov3_dual_no_adaptation",
        "source_domain": "A",
        "config": config,
        "scored_variant": SCORED,
        "target_domains": [asdict(result) for result in results[SCORED]],
        "do_a_dsc": do_a[SCORED],
        "diagnostic_variants": {
            variant: {"target_domains": [asdict(r) for r in rows], "do_a_dsc": do_a[variant]}
            for variant, rows in results.items()
            if variant != SCORED
        },
        "per_image": per_image,
    }
    atomic_json_write(args.output, payload)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    # Options of the crop teacher (DINOv3 location, crop scale, crop head) are
    # shared with redifm_dinov3/adapt.py and keep its defaults.
    own_args, remaining = build_parser().parse_known_args()
    main(own_args, build_crop_parser().parse_args(remaining))
