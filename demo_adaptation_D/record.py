#!/usr/bin/env python3
"""Record what the adapting target sees and predicts, image by image, for index.html.

    python demo_adaptation_D/record.py [--task odoc] [--domain D] [--seed 0]

Runs ``redifm_dinov3_nnunet_crop`` on one target domain, starting from the domain-A
model, with the method's own models, settings and update.  For every step it saves
the image, the augmented copy the target learned from, the ground truth, both
teachers' masks, the fused supervision, and the target's prediction just before and
just after its update on that image.  Then open demo_adaptation_D/index.html.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.protocol import image_to_tensor, predict_after_update, prepare_sample, seed_everything  # noqa: E402
from common.redifm import structure_probabilities  # noqa: E402
from common.tasks import TASKS  # noqa: E402
from methods.redifm_dinov3_nnunet_crop import adaptation_step, add_arguments, load_models  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"


def structures_to_classes(structures: torch.Tensor) -> np.ndarray:
    """Nested structure probabilities -> class map: each structure inside the previous one."""
    masks = (structures[0] > 0.5).cpu().numpy()
    classes = np.zeros(masks.shape[1:], dtype=np.uint8)
    inside = np.ones(masks.shape[1:], dtype=bool)
    for index, mask in enumerate(masks):
        inside &= mask
        classes[inside] = index + 1
    return classes


# Overlay colors (RGBA): structure 1 (disc rim / foreground), structure 2 (cup).
CLASS_COLORS = {1: (42, 120, 214, 120), 2: (237, 161, 0, 150)}


def save_mask(classes: np.ndarray, path: Path) -> None:
    """A transparent overlay, colored per class, so the page only has to stack images."""
    overlay = np.zeros((*classes.shape, 4), dtype=np.uint8)
    for value, color in CLASS_COLORS.items():
        overlay[classes == value] = color
    Image.fromarray(overlay, mode="RGBA").save(path, optimize=True)


def save_outline(classes: np.ndarray, path: Path) -> None:
    """The ground truth as white outlines of each structure."""
    overlay = np.zeros((*classes.shape, 4), dtype=np.uint8)
    for value in range(1, int(classes.max()) + 1):
        region = classes >= value
        overlay[region & ~ndimage.binary_erosion(region, iterations=2)] = (255, 255, 255, 255)
    Image.fromarray(overlay, mode="RGBA").save(path, optimize=True)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    array = (tensor[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path, quality=85)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="odoc")
    parser.add_argument("--domain", default="D")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--device")
    add_arguments(parser)
    args = parser.parse_args()
    task = TASKS[args.task]
    args.image_size = task.image_size

    seed_everything(args.seed)
    samples = task.build_domains([args.domain], args.max_images)[args.domain]
    device, source_model, target_model, segmenter, optimizer, augmentation = load_models(args, task)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for old in DATA_DIR.iterdir():
        old.unlink()
    steps = []
    for step, sample in enumerate(tqdm(samples, desc=f"{task.name} domain {args.domain}")):
        image, target = prepare_sample(sample, task.image_size)
        image_tensor = image_to_tensor(image, device)
        before = predict_after_update(target_model, image_tensor)[0].cpu()
        details: dict = {}
        consensus, _ = adaptation_step(
            source_model, target_model, segmenter, optimizer, augmentation, image, image_tensor,
            args.pseudo_threshold, details,
        )
        after = predict_after_update(target_model, image_tensor)[0].cpu()

        with torch.inference_mode():
            soft = details["source"][0, 0].cpu().numpy()
            top, left, side = segmenter.crop_box(soft > args.pseudo_threshold, soft)
        masks = {
            "truth": target.numpy().astype(np.uint8),
            "dinov3": structures_to_classes(details["source"]),
            "crop": structures_to_classes(details["foundation"]),
            "fused": structures_to_classes(details["fused"]),
            "before": before.numpy().astype(np.uint8),
            "after": after.numpy().astype(np.uint8),
        }
        name = f"{step:04d}"
        image.save(DATA_DIR / f"{name}_image.jpg", quality=85)
        save_image(details["augmented"], DATA_DIR / f"{name}_augmented.jpg")
        for key, classes in masks.items():
            save_mask(classes, DATA_DIR / f"{name}_{key}.png")
        save_outline(masks["truth"], DATA_DIR / f"{name}_outline.png")
        scores = {
            key: task.score(torch.from_numpy(classes).long(), target)
            for key, classes in masks.items() if key != "truth"
        }
        steps.append({
            "step": step + 1,
            "name": name,
            "image": sample.identifier,
            "crop": [top, left, side],
            "consensus": consensus,
            "agreements": details["agreements"],
            "loss": details["loss"],
            "scores": {key: {k.removesuffix("_dsc"): v for k, v in value.items()} for key, value in scores.items()},
        })

    meta = {
        "task": task.name,
        "domain": args.domain,
        "dataset": samples[0].dataset,
        "structures": list(task.structures),
        "size": task.image_size,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "augmentation": not args.no_augmentation,
        "source": Path(args.full_head_checkpoint).name,
        "crop_model": Path(segmenter.checkpoint).name,
        "target": Path(args.target_checkpoint).name,
    }
    (DATA_DIR / "steps.js").write_text("window.DEMO = " + json.dumps({"meta": meta, "steps": steps}) + ";\n")
    after = np.mean([np.mean(list(s["scores"]["after"].values())) for s in steps])
    print(f"Recorded {len(steps)} steps to {DATA_DIR}; mean DSC after update {100 * after:.2f}")


if __name__ == "__main__":
    main()
