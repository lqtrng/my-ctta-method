#!/usr/bin/env python3
"""Evaluate on the target stream of one source domain: source-only or Crop-Teach.

    python adapt.py --problem odoc --source A --method source_only
    python adapt.py --problem odoc --source A --method crop_teach --seed 0
    python adapt.py --problem odoc --source A --method crop_teach --seed 0 --rounds 3   # long-term

Protocol (the ReDi-FM paper, Section 4): the target domains are all domains but the source, streamed
in alphabetical order, one image at a time, never reset; with --rounds the whole stream repeats.
A Crop-Teach step on each image:
  1. the frozen DINOv3 full-image head (trained on the source) predicts the whole image; every
     connected region of its first-structure mask gets a square crop, 2x the region's extent;
  2. the frozen crop model (trained on native-res source crops) predicts each crop cut from the
     full-size image; the predictions are pasted back, background outside the crops;
  3. the target model (the whole-image source CNN) takes one Adam update (lr 2e-5, batch-norm
     statistics frozen) on a color-jittered, randomly erased copy, with the paper's loss
     (losses.crop_teach_loss), then predicts the original image: the scored prediction.

Writes runs/results/<problem>/<method>_source<S>[_seed<k>][_rounds<R>].json (skipped if it exists).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn
from torchvision.transforms import ColorJitter, RandomErasing

import tracking
from config import (CROP_SCALE, ENTROPY_THRESHOLD, ERASING, JITTER, LEARNING_RATE, MIN_CROP, PROBLEMS,
                    PSEUDO_THRESHOLD, RESULTS_DIR, checkpoint_path)
from data import domain_samples, test_sample, to_tensor
from losses import crop_teach_loss, structure_probabilities
from metrics import score
from networks import build_network, load_backbone, load_head


def load_cnn(path: Path, device: torch.device) -> nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint (train it first with train.py): {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_network(checkpoint["network"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device)


def crop_box(mask: np.ndarray, probability: np.ndarray) -> tuple[int, int, int]:
    """(top, left, side): a square 2x the region's extent (at least MIN_CROP), inside the image."""
    size = mask.shape[0]
    if not mask.any():  # No region: center on the most confident pixel.
        mask = np.zeros_like(mask, dtype=bool)
        mask[np.unravel_index(np.argmax(probability), probability.shape)] = True
    rows, columns = np.nonzero(mask)
    side = int(np.clip(max(np.ptp(rows) + 1, np.ptp(columns) + 1) * CROP_SCALE, MIN_CROP, size))
    top = int(np.clip(round((rows.min() + rows.max()) / 2 - side / 2), 0, size - side))
    left = int(np.clip(round((columns.min() + columns.max()) / 2 - side / 2), 0, size - side))
    return top, left, side



class CropTeach:
    def __init__(self, problem: str, source: str, device: torch.device) -> None:
        self.device = device
        backbone = load_backbone(device)
        self.head = load_head(checkpoint_path(problem, source, "head"), backbone).eval()
        self.crop_model = load_cnn(checkpoint_path(problem, source, "crop"), device).eval().requires_grad_(False)
        self.target = load_cnn(checkpoint_path(problem, source, "target"), device)
        self.optimizer = torch.optim.Adam(self.target.parameters(), lr=LEARNING_RATE)
        self.augment = nn.Sequential(ColorJitter(**JITTER), RandomErasing(**ERASING))

    @torch.no_grad()
    def crop_teacher(self, image: torch.Tensor, full: torch.Tensor, source: torch.Tensor):
        """(nested structure probabilities, merged class probabilities, number of crops) from every crop;
        each pixel keeps the most foreground crop; outside the crops all classes are 0 (background)."""
        size = image.shape[-1]
        probability = source[0, 0].cpu().numpy()
        components, count = ndimage.label(probability > PSEUDO_THRESHOLD)
        boxes = [crop_box(components == index, probability) for index in range(1, count + 1)]
        boxes = boxes or [crop_box(np.zeros_like(probability, dtype=bool), probability)]
        sy, sx = full.shape[-2] / size, full.shape[-1] / size
        merged, inside = None, torch.zeros_like(image[:, :1])
        for top, left, side in boxes:
            crop = full[:, :, round(top * sy): round((top + side) * sy), round(left * sx): round((left + side) * sx)]
            crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
            classes = F.softmax(self.crop_model(crop), dim=1)
            pasted = torch.zeros(1, classes.shape[1], size, size, device=self.device)
            pasted[:, :, top: top + side, left: left + side] = F.interpolate(
                classes, size=(side, side), mode="bilinear", align_corners=False)
            inside[:, :, top: top + side, left: left + side] = 1.0
            if merged is None:
                merged = pasted
            else:
                better = pasted[:, 1:].sum(1, keepdim=True) > merged[:, 1:].sum(1, keepdim=True)
                merged = torch.where(better, pasted, merged)
        return (structure_probabilities(merged) * inside).clamp(1e-6, 1 - 1e-6), merged, len(boxes)

    def __call__(self, image: torch.Tensor, full: torch.Tensor) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            source_classes = F.softmax(self.head(image), dim=1)
        crop, merged, boxes = self.crop_teacher(image, full, structure_probabilities(source_classes))
        self.target.train()
        for module in self.target.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()  # Batches of one image would make these statistics noisy.
        augmented = self.augment(image[0])[None]
        loss, consensus = crop_teach_loss(F.softmax(self.target(augmented), dim=1), source_classes, crop,
                                          PSEUDO_THRESHOLD, ENTROPY_THRESHOLD)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        # The two frozen teachers' own predictions, scored as no-adaptation results at no extra cost.
        teachers = {"head_alone": source_classes[0], "crop_teacher": merged[0]}
        return predict(self.target, image), {"consensus": consensus, "crops": boxes, "loss": float(loss),
                                             "teachers": teachers}


class SourceOnly:
    def __init__(self, problem: str, source: str, device: torch.device) -> None:
        self.target = load_cnn(checkpoint_path(problem, source, "target"), device)

    def __call__(self, image: torch.Tensor, full: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return predict(self.target, image), {}


@torch.no_grad()
def predict(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Class probabilities of the clean image, in eval mode (no batch-norm update)."""
    model.eval()
    return F.softmax(model(image), dim=1)[0]


METHODS = {"source_only": SourceOnly, "crop_teach": CropTeach}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=list(PROBLEMS), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--method", choices=list(METHODS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1, help="Passes over the stream, never reset")
    parser.add_argument("--max-images", type=int, help="Per domain, for a smoke test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    problem = PROBLEMS[args.problem]
    seed_tag = "" if args.method == "source_only" else f"_seed{args.seed}"
    round_tag = f"_rounds{args.rounds}" if args.rounds > 1 else ""
    output = args.output or RESULTS_DIR / args.problem / f"{args.method}_source{args.source}{seed_tag}{round_tag}.json"
    if output.exists() and not args.force:
        print(f"Exists, skipped: {output}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    method = METHODS[args.method](args.problem, args.source, device)
    stream = {domain: domain_samples(args.problem, domain, args.max_images) for domain in problem.stream(args.source)}
    polyp = args.problem == "polyp"
    print(f"{args.problem}: {args.method}, source {args.source}, stream {' -> '.join(stream)}, "
          f"seed {args.seed}, {args.rounds} round(s)", flush=True)

    rounds, rows = [], []
    start, images = time.time(), 0
    with tracking.run(f"{args.method}_{args.problem}_source{args.source}", vars(args)):
        for round_index in range(1, args.rounds + 1):
            domains = {}
            for domain, samples in stream.items():
                scores = []
                for sample in samples:
                    image, full, target = test_sample(sample, problem.image_size)
                    image, full = to_tensor(image, device), to_tensor(full, device)
                    classes, extras = method(image, full)
                    s = score(classes.argmax(0).cpu(), 1 - classes[0], target, problem.structures, polyp)
                    s["mean_dsc"] = float(np.mean([s[f"{name}_dsc"] for name in problem.structures]))
                    for name, teacher in extras.pop("teachers", {}).items():
                        t = score(teacher.argmax(0).cpu(), 1 - teacher[0], target, problem.structures, False)
                        s[f"{name}_mean_dsc"] = float(np.mean([t[f"{k}_dsc"] for k in problem.structures]))
                    scores.append(s)
                    rows.append({"round": round_index, "domain": domain, "image": sample.name, **s, **extras})
                    images += 1
                domains[domain] = {key: 100 * float(np.mean([s[key] for s in scores])) for key in scores[0]}
                domains[domain]["images"] = len(scores)
                print(f"round {round_index} domain {domain}: " + ", ".join(
                    f"{k}={v:.2f}" for k, v in domains[domain].items() if k != "images"), flush=True)
            summary = {key: float(np.mean([d[key] for d in domains.values()])) for key in next(iter(domains.values()))
                       if key != "images"}
            rounds.append({"round": round_index, "domains": domains, "mean": summary})
            print(f"round {round_index} Do. {args.source}: DSC={summary['mean_dsc']:.2f}", flush=True)
            tracking.log_metrics({f"do_{key}": value for key, value in summary.items()}, step=round_index)

        seconds = time.time() - start
        result = {"problem": args.problem, "source": args.source, "method": args.method, "seed": args.seed,
                  "stream": list(stream), "rounds": rounds, "do_dsc": rounds[0]["mean"]["mean_dsc"],
                  "ms_per_image": 1000 * seconds / max(images, 1), "minutes": seconds / 60,
                  "config": {"learning_rate": 2e-5, "pseudo_threshold": PSEUDO_THRESHOLD,
                             "entropy_threshold": ENTROPY_THRESHOLD, "crop_scale": CROP_SCALE,
                             "min_crop": MIN_CROP, "max_images": args.max_images}}
        if args.rounds > 1:
            scores = [r["mean"]["mean_dsc"] for r in rounds]
            result.update(average_dsc=float(np.mean(scores)), degradation=scores[0] - float(np.mean(scores)))
        tracking.log_metrics({"minutes": seconds / 60, "ms_per_image": result["ms_per_image"]})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    with open(output.with_suffix(".csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
