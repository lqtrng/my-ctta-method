#!/usr/bin/env python3
"""Evaluate a source-only model on a ReDi-FM benchmark (no adaptation).

This is the paper's "Source Only" baseline: the domain-A checkpoint is loaded
once, placed in evaluation mode, and never updated while target images are
processed.  For OD/OC (``--task odoc``) the target domains are:

* Domain B: REFUGE training set (400 images)
* Domain C: ORIGA (650 images)
* Domain D: REFUGE validation and test sets (800 images)
* Domain E: Drishti-GS training and test sets (101 images)

``--task prostate`` uses target domains B-F of the six-site prostate benchmark.
The data stream, preprocessing and per-image DSC are the ones shared by every
method in ``common/``.  The paper reports one DSC per source domain ("Do. A"),
the mean over the target domains; per-structure DSC is reported for every
target domain so the aggregate is fully auditable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.data import atomic_json_write  # noqa: E402
from common.models import ARCHITECTURES, build_network  # noqa: E402
from common.paths import RESULTS_DIR, SOURCE_CHECKPOINTS, require_file  # noqa: E402
from common.protocol import image_to_tensor, prepare_sample  # noqa: E402
from common.tasks import TASKS  # noqa: E402
from common.tracking import log_result, tracked_run  # noqa: E402


DEFAULT_OUTPUT = RESULTS_DIR / "source_only/baseline_no_adaptation_domain_A.json"
# 0140.pdf Table 1, Source Only, source domain A.
PAPER_DO_A = {"odoc": 53.59, "prostate": 51.02}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASKS), default="odoc")
    parser.add_argument("--arch", choices=list(ARCHITECTURES), default="resunet34")
    parser.add_argument("--checkpoint", type=Path,
                        help="Default: the domain-A source checkpoint of --task and --arch")
    parser.add_argument("--output", type=Path,
                        help="Default: results/source_only/[prostate_]baseline_no_adaptation_domain_A.json")
    parser.add_argument("--max-images", type=int, help="Limit images per domain for a smoke test")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda when available, otherwise cpu)",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main(args: argparse.Namespace, method: str = "source_only") -> None:
    task = TASKS[args.task]
    if args.checkpoint is None:
        args.checkpoint = SOURCE_CHECKPOINTS[(args.task, args.arch)]
    if args.output is None:
        prefix = "" if args.task == "odoc" else f"{args.task}_"
        args.output = DEFAULT_OUTPUT.with_name(f"{prefix}{DEFAULT_OUTPUT.name}")
    checkpoint = require_file(args.checkpoint.resolve(), "Source checkpoint")

    device = torch.device(args.device)
    model = build_network(args.arch, task.num_classes, task.image_size)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(device).eval()

    print(f"Checkpoint: {checkpoint} ({args.arch})")
    print(f"Device: {device}")
    print("Protocol: Source Only (no parameter or BN-statistic updates)")
    print("Metric: mean per-image DSC" + ("; OD includes the OC region" if task.name == "odoc" else ""))

    with tracked_run(method, args):
        results: list[dict[str, object]] = []
        per_image: list[dict[str, object]] = []
        domains = task.build_domains(list(task.target_domains), args.max_images)
        with torch.inference_mode():
            for domain, samples in domains.items():
                rows = []
                for sample in tqdm(samples, desc=f"Source only domain {domain}"):
                    image, target = prepare_sample(sample, task.image_size)
                    prediction = model(image_to_tensor(image, device)).argmax(dim=1)[0].cpu()
                    scores = task.score(prediction, target)
                    rows.append(scores)
                    per_image.append(
                        {
                            "domain": domain,
                            "image": sample.identifier,
                            **scores,
                            "mean_dsc": float(np.mean(list(scores.values()))),
                        }
                    )
                results.append(task.domain_summary(domain, samples[0].dataset, rows))
                print(task.describe(results[-1]))

        do_a = float(np.mean([result["mean_dsc"] for result in results]))
        pooled_image_mean = float(np.mean([row["mean_dsc"] for row in per_image])) * 100.0
        print(f"Do. A DSC (unweighted {'/'.join(domains)} domain mean): {do_a:.2f}")
        print(f"Pooled target-image DSC (diagnostic only): {pooled_image_mean:.2f}")

        payload = {
            "protocol": "source_only",
            "task": task.name,
            "source_domain": "A",
            "arch": args.arch,
            "checkpoint": str(checkpoint),
            "image_size": task.image_size,
            "target_domains": results,
            "do_a_dsc": do_a,
            "pooled_target_image_dsc": pooled_image_mean,
            "paper_source_only_do_a_reference": PAPER_DO_A[task.name],
            "per_image": per_image,
        }
        atomic_json_write(args.output, payload)
        log_result(payload, args.output)
        print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main(parse_args())
