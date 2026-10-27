#!/usr/bin/env python3
"""Evaluate frozen MedCLIP-SAMv2 on the ReDi-FM OD/OC benchmark.

MedCLIP-SAMv2 is a pipeline rather than a single neural network:

1. Fine-tuned BiomedCLIP and M2IB produce one saliency map per text prompt.
2. K-means converts each saliency map to a coarse foreground region.
3. The region's bounding box prompts a frozen SAM ViT-H model.

The two foundation-model checkpoints are never updated.  M2IB optimizes only a
temporary information-bottleneck tensor for the current image and discards it
after producing the saliency map.  That is inference-time prompt generation,
not model fine-tuning.

The public MedCLIP-SAMv2 release does not include the optic-disc/optic-cup text
prompts used by the ReDi-FM evaluation.  Consequently, the prompts are exposed
as command-line arguments and the defaults below are deliberately explicit.

Example (first image from each target domain):

    python src/medclip_samv2/zero_shot.py --max-images 1

Full Domain-A protocol (target domains B/C/D/E):

    python src/medclip_samv2/zero_shot.py
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from sklearn.cluster import KMeans
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.data import Sample, atomic_json_write, build_domains, load_sample  # noqa: E402
from common.paths import (  # noqa: E402
    BIOMEDCLIP_CHECKPOINT,
    BIOMEDCLIP_DIR,
    RESULTS_DIR,
    SAM_CHECKPOINT,
    require_file,
)
from common.protocol import DomainResult  # noqa: E402


DEFAULT_BIOMEDCLIP_CHECKPOINT = BIOMEDCLIP_CHECKPOINT
DEFAULT_SAM_CHECKPOINT = SAM_CHECKPOINT
DEFAULT_OUTPUT = RESULTS_DIR / "medclip_samv2/zero_shot_medclip_samv2_domain_A.json"
DEFAULT_BOX_CACHE = RESULTS_DIR / "medclip_samv2/medclip_samv2_boxes_domain_A.json"


def require_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    """Import optional heavyweight dependencies with an actionable error."""
    try:
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError as error:
        raise SystemExit(
            "Missing MedCLIP-SAMv2 dependencies. Install them in this environment with:\n"
            "  uv pip install --python .venv/bin/python transformers opencv-python-headless "
            "git+https://github.com/facebookresearch/segment-anything.git"
        ) from error
    return AutoConfig, AutoModel, AutoProcessor, AutoTokenizer, (
        SamPredictor,
        sam_model_registry,
    )


def normalize(values: torch.Tensor) -> torch.Tensor:
    minimum = values.amin()
    scale = values.amax() - minimum
    if not torch.isfinite(scale) or scale <= 1e-12:
        return torch.zeros_like(values)
    return (values - minimum) / scale


class Bottleneck(torch.nn.Module):
    """M2IB bottleneck; ``alpha`` is the only optimized tensor."""

    def __init__(self, feature_shape: tuple[int, ...], variance: float, device: torch.device):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.full((1, *feature_shape), 5.0, device=device))
        self.variance = variance
        self.capacity: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor, *_: Any) -> tuple[torch.Tensor]:
        gate = self.alpha.sigmoid().expand_as(hidden_states)
        mean = hidden_states * gate
        variance = (1.0 - gate).square().clamp_min(1e-8)
        self.capacity = -0.5 * (1.0 + variance.log() - mean.square() - variance)
        sampled = mean + variance.sqrt() * torch.randn_like(mean)
        return (sampled,)


class LayerThenBottleneck(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, bottleneck: Bottleneck):
        super().__init__()
        self.layer = layer
        self.bottleneck = bottleneck

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor]:
        layer_output = self.layer(*args, **kwargs)
        hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        return self.bottleneck(hidden_states)


@contextmanager
def replace_vision_layer(model: torch.nn.Module, index: int, replacement: torch.nn.Module) -> Iterator[None]:
    layers = model.vision_model.encoder.layers
    original = layers[index]
    layers[index] = replacement
    try:
        yield
    finally:
        layers[index] = original


def m2ib_saliency(
    model: torch.nn.Module,
    processor: Any,
    tokenizer: Any,
    image: Image.Image,
    prompts: list[str],
    device: torch.device,
    layer_index: int,
    beta: float,
    variance: float,
    steps: int,
    learning_rate: float,
    noise_batch_size: int,
) -> np.ndarray:
    pixels = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        vision_output = model.vision_model(pixels, output_hidden_states=True)
        features = vision_output.hidden_states[layer_index + 1]
        text_features = model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_features = text_features.mean(dim=0, keepdim=True)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

    bottleneck = Bottleneck(tuple(features.shape[1:]), variance, device)
    wrapped = LayerThenBottleneck(model.vision_model.encoder.layers[layer_index], bottleneck)
    optimizer = torch.optim.Adam([bottleneck.alpha], lr=learning_rate)
    repeated_pixels = pixels.expand(noise_batch_size, -1, -1, -1)
    target_text = text_features.expand(noise_batch_size, -1)

    with replace_vision_layer(model, layer_index, wrapped):
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            image_features = model.get_image_features(pixel_values=repeated_pixels)
            fit = torch.nn.functional.cosine_similarity(target_text, image_features).mean()
            if bottleneck.capacity is None:
                raise RuntimeError("M2IB bottleneck was not reached during the forward pass")
            loss = beta * bottleneck.capacity.mean() - fit
            loss.backward()
            optimizer.step()

    if bottleneck.capacity is None:
        raise RuntimeError("M2IB did not produce a capacity map")
    token_saliency = bottleneck.capacity.mean(dim=0).sum(dim=-1)[1:]
    side = int(round(token_saliency.numel() ** 0.5))
    if side * side != token_saliency.numel():
        raise RuntimeError(f"Cannot reshape {token_saliency.numel()} vision tokens into a square")
    heatmap = token_saliency.reshape(1, 1, side, side)
    heatmap = torch.nn.functional.interpolate(
        heatmap, size=(224, 224), mode="bilinear", align_corners=False
    ).squeeze()
    return normalize(heatmap).detach().cpu().numpy().astype(np.float32)


def saliency_to_box(saliency: np.ndarray, image_size: tuple[int, int]) -> list[int]:
    clustering_input = np.asarray(
        Image.fromarray(saliency).resize((256, 256), Image.Resampling.NEAREST),
        dtype=np.float32,
    )
    kmeans = KMeans(n_clusters=2, random_state=10, n_init=10)
    labels = kmeans.fit_predict(clustering_input.reshape(-1, 1)).reshape(256, 256)
    foreground_cluster = int(np.argmax(kmeans.cluster_centers_.ravel()))
    foreground = labels == foreground_cluster
    foreground = np.asarray(
        Image.fromarray(foreground).resize(image_size, Image.Resampling.NEAREST),
        dtype=bool,
    )

    components, count = ndimage.label(foreground)
    if count:
        sizes = ndimage.sum(foreground, components, range(1, count + 1))
        foreground = components == int(np.argmax(sizes) + 1)
    ys, xs = np.nonzero(foreground)
    width, height = image_size
    if len(xs) == 0:
        return [0, 0, width - 1, height - 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def load_biomedclip(
    checkpoint: Path, config_dir: Path, device: torch.device
) -> tuple[torch.nn.Module, Any, Any]:
    """Config, remote code and tokenizer come from the local ``config_dir``."""
    AutoConfig, AutoModel, AutoProcessor, AutoTokenizer, _ = require_runtime_dependencies()
    config = AutoConfig.from_pretrained(config_dir, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_config(config, trust_remote_code=True)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state_dict, strict=True)
    model.requires_grad_(False).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        config_dir, trust_remote_code=True, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config_dir, trust_remote_code=True, local_files_only=True
    )
    return model, processor, tokenizer


def generate_boxes(
    samples: list[Sample], args: argparse.Namespace, device: torch.device
) -> dict[str, dict[str, list[int]]]:
    cache_settings = {
        "disc_prompts": args.disc_prompt,
        "cup_prompts": args.cup_prompt,
        "layer": args.m2ib_layer,
        "beta": args.m2ib_beta,
        "variance": args.m2ib_variance,
        "steps": args.m2ib_steps,
        "learning_rate": args.m2ib_learning_rate,
        "noise_batch_size": args.m2ib_batch_size,
    }
    cache: dict[str, dict[str, list[int]]] = {}
    if args.box_cache.is_file():
        cache_payload = json.loads(args.box_cache.read_text(encoding="utf-8"))
        if cache_payload.get("settings") == cache_settings:
            cache = cache_payload.get("boxes", {})
        else:
            print(f"Ignoring incompatible M2IB box cache: {args.box_cache}")
    missing = [sample for sample in samples if sample.key not in cache]
    if not missing:
        print(f"Reusing all {len(samples)} cached M2IB boxes: {args.box_cache}")
        return cache

    print(f"Generating M2IB boxes for {len(missing)} images...")
    model, processor, tokenizer = load_biomedclip(args.biomedclip_checkpoint, args.biomedclip_dir, device)
    for index, sample in enumerate(tqdm(missing, desc="BiomedCLIP/M2IB"), start=1):
        image, _ = load_sample(sample)
        boxes: dict[str, list[int]] = {}
        for structure, prompts in (
            ("disc", args.disc_prompt),
            ("cup", args.cup_prompt),
        ):
            saliency = m2ib_saliency(
                model,
                processor,
                tokenizer,
                image,
                prompts,
                device,
                args.m2ib_layer,
                args.m2ib_beta,
                args.m2ib_variance,
                args.m2ib_steps,
                args.m2ib_learning_rate,
                args.m2ib_batch_size,
            )
            boxes[structure] = saliency_to_box(saliency, image.size)
        cache[sample.key] = boxes
        if index % args.cache_interval == 0 or index == len(missing):
            atomic_json_write(
                args.box_cache,
                {
                    "method": "MedCLIP-SAMv2 M2IB + K-means",
                    "settings": cache_settings,
                    "boxes": cache,
                },
            )

    del model, processor, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return cache


def dice(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    return float(
        (2.0 * np.logical_and(prediction, target).sum() + 1e-6)
        / (prediction.sum() + target.sum() + 1e-6)
    )


def evaluate_with_sam(
    domains: dict[str, list[Sample]],
    boxes: dict[str, dict[str, list[int]]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[DomainResult], list[dict[str, str | float]]]:
    *_, sam_objects = require_runtime_dependencies()
    SamPredictor, sam_model_registry = sam_objects
    print("Loading frozen SAM ViT-H...")
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_checkpoint))
    sam.requires_grad_(False).to(device).eval()
    predictor = SamPredictor(sam)

    domain_results: list[DomainResult] = []
    per_image: list[dict[str, str | float]] = []
    for domain, samples in domains.items():
        disc_scores: list[float] = []
        cup_scores: list[float] = []
        for sample in tqdm(samples, desc=f"SAM domain {domain}"):
            image, target = load_sample(sample)
            image_array = np.asarray(image, dtype=np.uint8)
            predictor.set_image(image_array)
            predictions: dict[str, np.ndarray] = {}
            for structure in ("disc", "cup"):
                prompt_box = np.asarray(boxes[sample.key][structure], dtype=np.float32)
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=prompt_box,
                    multimask_output=False,
                )
                predictions[structure] = masks[0]
            disc_score = dice(predictions["disc"], target >= 1)
            cup_score = dice(predictions["cup"], target == 2)
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
        domain_results.append(result)
        print(
            f"Domain {domain} ({result.dataset}, n={result.images}): "
            f"OD={result.disc_dsc:.2f}, OC={result.cup_dsc:.2f}, "
            f"Mean={result.mean_dsc:.2f}"
        )
    return domain_results, per_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biomedclip-checkpoint", type=Path, default=DEFAULT_BIOMEDCLIP_CHECKPOINT)
    parser.add_argument("--biomedclip-dir", type=Path, default=BIOMEDCLIP_DIR,
                        help="BiomedCLIP config, remote code and tokenizer")
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--box-cache", type=Path, default=DEFAULT_BOX_CACHE)
    parser.add_argument("--domains", nargs="+", choices=list("BCDE"), default=list("BCDE"))
    parser.add_argument("--max-images", type=int, help="Limit images per domain for a smoke test")
    parser.add_argument(
        "--disc-prompt",
        action="append",
        default=None,
        help="OD prompt; repeat for an ensemble",
    )
    parser.add_argument(
        "--cup-prompt",
        action="append",
        default=None,
        help="OC prompt; repeat for an ensemble",
    )
    parser.add_argument("--m2ib-layer", type=int, default=8)
    parser.add_argument("--m2ib-beta", type=float, default=0.1)
    parser.add_argument("--m2ib-variance", type=float, default=1.0)
    parser.add_argument("--m2ib-steps", type=int, default=10)
    parser.add_argument("--m2ib-learning-rate", type=float, default=1.0)
    parser.add_argument("--m2ib-batch-size", type=int, default=4)
    parser.add_argument("--cache-interval", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda when available)",
    )
    args = parser.parse_args()
    if args.disc_prompt is None:
        args.disc_prompt = ["an optic disc in a retinal fundus image"]
    if args.cup_prompt is None:
        args.cup_prompt = ["an optic cup in a retinal fundus image"]
    return args


def validate_args(args: argparse.Namespace) -> None:
    require_file(args.biomedclip_checkpoint, "Fine-tuned BiomedCLIP checkpoint")
    require_file(args.biomedclip_dir / "config.json", "BiomedCLIP config")
    require_file(args.sam_checkpoint, "SAM checkpoint")
    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be positive")
    if not 0 <= args.m2ib_layer < 12:
        raise ValueError("--m2ib-layer must be between 0 and 11")
    if args.m2ib_steps < 1 or args.m2ib_batch_size < 1 or args.cache_interval < 1:
        raise ValueError("M2IB steps/batch size and cache interval must be positive")


def main(args: argparse.Namespace) -> None:
    validate_args(args)
    device = torch.device(args.device)
    domains = build_domains(args.domains, args.max_images)
    samples = [sample for domain_samples in domains.values() for sample in domain_samples]

    print(f"Device: {device}")
    print(f"BiomedCLIP checkpoint: {args.biomedclip_checkpoint.resolve()}")
    print(f"SAM checkpoint: {args.sam_checkpoint.resolve()}")
    print("Protocol: frozen zero-shot MedCLIP-SAMv2 (no checkpoint updates)")
    print("Metric: mean per-image DSC; OD includes the OC region")
    print(f"OD prompts: {args.disc_prompt}")
    print(f"OC prompts: {args.cup_prompt}")

    boxes = generate_boxes(samples, args, device)
    results, per_image = evaluate_with_sam(domains, boxes, args, device)
    do_a = float(np.mean([result.mean_dsc for result in results]))
    pooled_image_mean = float(np.mean([row["mean_dsc"] for row in per_image])) * 100.0
    print(f"Do. A DSC (unweighted B/C/D/E domain mean): {do_a:.2f}")
    print(f"Pooled target-image DSC (diagnostic only): {pooled_image_mean:.2f}")

    payload = {
        "protocol": "zero_shot_medclip_samv2",
        "source_domain": "A",
        "checkpoint": {
            "biomedclip": str(args.biomedclip_checkpoint.resolve()),
            "sam": str(args.sam_checkpoint.resolve()),
        },
        "image_size": "original",
        "biomedclip_checkpoint": str(args.biomedclip_checkpoint.resolve()),
        "sam_checkpoint": str(args.sam_checkpoint.resolve()),
        "foundation_model_parameters": 836_993_585,
        "trainable_checkpoint_parameters": 0,
        "disc_prompts": args.disc_prompt,
        "cup_prompts": args.cup_prompt,
        "m2ib": {
            "layer": args.m2ib_layer,
            "beta": args.m2ib_beta,
            "variance": args.m2ib_variance,
            "steps": args.m2ib_steps,
            "learning_rate": args.m2ib_learning_rate,
            "noise_batch_size": args.m2ib_batch_size,
        },
        "target_domains": [asdict(result) for result in results],
        "do_a_dsc": do_a,
        "pooled_target_image_dsc": pooled_image_mean,
        "paper_medclip_samv2_do_a_reference": 62.83,
        "per_image": per_image,
    }
    atomic_json_write(args.output, payload)
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main(parse_args())
