#!/usr/bin/env python3
"""One frozen DINOv3 segments OD/OC, without the ResUNet-34 and without adaptation.

A frozen DINOv3 ViT-L/16 with a decoder head trained on whole domain-A images
(``redifm_dinov3/train_head.py --full-image``) predicts every target image once.
No source model, no second teacher and no online update are involved.  The
target stream, preprocessing and per-image DSC are the ones shared by every
method in ``src/``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.data import atomic_json_write, build_domains  # noqa: E402
from common.paths import DINOV3_ARCH, MODEL_DIR, RESULTS_DIR  # noqa: E402
from common.protocol import (  # noqa: E402
    DomainResult,
    image_dice,
    image_to_tensor,
    prepare_sample,
    seed_everything,
)
from redifm_dinov3.train_head import (  # noqa: E402
    DEFAULT_FULL_IMAGE_OUTPUT,
    SegmentationHead,
    dinov3_tokens,
    load_dinov3,
)


DEFAULT_OUTPUT = RESULTS_DIR / "dinov3_single/dinov3_single_domain_A.json"


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


def domain_result(domain: str, dataset: str, disc_scores: list[float], cup_scores: list[float]) -> DomainResult:
    disc_mean = float(np.mean(disc_scores))
    cup_mean = float(np.mean(cup_scores))
    return DomainResult(
        domain=domain,
        dataset=dataset,
        images=len(disc_scores),
        disc_dsc=100.0 * disc_mean,
        cup_dsc=100.0 * cup_mean,
        mean_dsc=100.0 * (disc_mean + cup_mean) / 2.0,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dinov3-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--dinov3-arch", default=DINOV3_ARCH)
    parser.add_argument("--full-head-checkpoint", type=Path, default=DEFAULT_FULL_IMAGE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--domains", nargs="+", choices=list("BCDE"), default=list("BCDE"))
    parser.add_argument("--max-images", type=int, help="Limit images per domain for a smoke test")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device)
    domains = build_domains(args.domains, args.max_images)
    model = load_dinov3(args.dinov3_dir, args.dinov3_arch, device)
    segmenter = FullImageSegmenter(model, args.full_head_checkpoint, args.dinov3_arch, device)

    print(f"Foundation model: DINOv3 {args.dinov3_arch} on {device}")
    print("Protocol: no adaptation; one frozen DINOv3 predicts every target image once")
    print("Metric: mean per-image DSC; OD includes the OC region")

    results: list[DomainResult] = []
    per_image: list[dict[str, Any]] = []
    for domain, samples in domains.items():
        disc_scores: list[float] = []
        cup_scores: list[float] = []
        for sample in tqdm(samples, desc=f"DINOv3 single domain {domain}"):
            image, target = prepare_sample(sample, args.image_size)
            target = target.to(device)
            prediction = segmenter(image_to_tensor(image, device)).argmax(dim=1)[0]
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
        result = domain_result(domain, samples[0].dataset, disc_scores, cup_scores)
        results.append(result)
        print(
            f"Domain {domain} ({result.dataset}, n={result.images}): "
            f"OD={result.disc_dsc:.2f}, OC={result.cup_dsc:.2f}, Mean={result.mean_dsc:.2f}"
        )

    do_a = float(np.mean([result.mean_dsc for result in results]))
    print(f"Do. A DSC (unweighted {'/'.join(args.domains)} domain mean): {do_a:.2f}")

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    payload = {
        "protocol": "dinov3_single_no_adaptation",
        "source_domain": "A",
        "config": config,
        "target_domains": [asdict(result) for result in results],
        "do_a_dsc": do_a,
        "per_image": per_image,
    }
    atomic_json_write(args.output, payload)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main(build_parser().parse_args())
