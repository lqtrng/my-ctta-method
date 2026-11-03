#!/usr/bin/env python3
"""Evaluate any method on any benchmark with one protocol.

    python src/evaluate.py --method redifm_dinov3 --task odoc --seed 0
    python src/evaluate.py --method source_only --task prostate --arch nnunet
    python src/evaluate.py --method redifm_dinov3 --rounds 3        # long-term CTTA (Table 3)
    python src/evaluate.py --method redifm_sam --help                # the method's own options

A method is a black box: ``build(args, task)`` returns a segmenter, and
``segmenter(image)`` maps one target image (PIL, already at the task's size) to
a class-index map plus a dict of extras.  A continual test-time adaptation
method updates itself inside that call, so the prediction it returns is the one
made after its online update.  The evaluator never resets it.

The protocol is identical for every method (0140.pdf, Section 4):

* the target domains are streamed in order (B -> C -> ...), one image at a
  time, for ``--rounds`` passes without any reset;
* each prediction is scored per structure with DSC by the task (``common/tasks.py``);
* a domain's score is the mean over its images, and Do. A the unweighted mean
  over target domains.

Extras with a number are logged per image (e.g. ReDi-FM's consensus); extras
with a class map are scored as diagnostic variants of the prediction.  Results
go to ``results/<method>/`` as JSON and to MLflow.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from common.data import atomic_json_write
from common.paths import RESULTS_DIR
from common.protocol import prepare_sample, seed_everything
from common.tasks import TASKS, Task
from common.tracking import log_result, tracked_run


# --method -> module with add_arguments(parser) and build(args, task).
METHODS = {
    "source_only": "baselines.source_only",
    "dinov3_single": "baselines.dinov3_single",
    "dinov3_dual": "baselines.dinov3_dual",
    "dinov3_crop": "baselines.dinov3_crop",
    "redifm_sam": "methods.redifm_sam",
    "redifm_dinov3": "methods.redifm_dinov3",
    "redifm_dinov3_dual": "methods.redifm_dinov3_dual",
    "redifm_dinov3_nnunet_crop": "methods.redifm_dinov3_nnunet_crop",
    "detect_crop_tta": "methods.detect_crop_tta",
}


def method_name(args: argparse.Namespace) -> str:
    """The method's results folder: ``--method``, suffixed by a non-default ``--arch``."""
    arch = getattr(args, "arch", "resunet34")
    return args.method if arch == "resunet34" else f"{args.method}_{arch}"


def default_output(args: argparse.Namespace, name: str) -> Path:
    prefix = "" if args.task == "odoc" else f"{args.task}_"
    suffix = "_long_term" if args.rounds > 1 else ""
    return RESULTS_DIR / name / f"{prefix}{name}{suffix}_domain_A.json"


def run_stream(segmenter: Any, task: Task, domains: dict, args: argparse.Namespace, round_index: int) -> tuple[list, list, dict]:
    """One pass over the target stream: per-domain summaries, per-image rows, variants."""
    summaries: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    variants: dict[str, list[dict[str, Any]]] = {}
    for domain, samples in domains.items():
        rows: list[dict[str, float]] = []
        variant_rows: dict[str, list[dict[str, float]]] = {}
        for sample in tqdm(samples, desc=f"{args.method} domain {domain}"):
            image, target = prepare_sample(sample, args.image_size)
            prediction, extras = segmenter(image)
            scores = task.score(prediction.cpu(), target)
            rows.append(scores)
            row = {"domain": domain, "image": sample.identifier, **scores,
                   "mean_dsc": float(np.mean(list(scores.values())))}
            if args.rounds > 1:
                row["round"] = round_index
            for key, value in extras.items():
                if isinstance(value, torch.Tensor):
                    variant_scores = task.score(value.cpu(), target)
                    variant_rows.setdefault(key, []).append(variant_scores)
                    row.update({f"{key}_{name}": score for name, score in variant_scores.items()})
                    row[f"{key}_mean_dsc"] = float(np.mean(list(variant_scores.values())))
                else:
                    row[key] = value
            per_image.append(row)
        summaries.append(task.domain_summary(domain, samples[0].dataset, rows))
        print(task.describe(summaries[-1]))
        for key, values in variant_rows.items():
            variants.setdefault(key, []).append(task.domain_summary(domain, samples[0].dataset, values))
            print(f"  [{key}] {task.describe(variants[key][-1])}")
    return summaries, per_image, variants


def do_a(summaries: list[dict[str, Any]]) -> float:
    return float(np.mean([summary["mean_dsc"] for summary in summaries]))


def source_only_do_a(args: argparse.Namespace, domains: list[str]) -> float | None:
    """Do. A of the source model on the same domains, for the long-term Gain (Table 3)."""
    arch = getattr(args, "arch", "resunet34")
    name = "source_only" if arch == "resunet34" else f"source_only_{arch}"
    path = default_output(argparse.Namespace(task=args.task, rounds=1), name)
    if not path.is_file():
        print(f"No source-only result at {path}; Gain is not reported")
        return None
    rows = [row for row in json.loads(path.read_text())["target_domains"] if row["domain"] in domains]
    return do_a(rows) if len(rows) == len(domains) else None


def build_parser(method: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=list(METHODS), required=True)
    parser.add_argument("--task", choices=list(TASKS), default="odoc")
    parser.add_argument("--domains", nargs="+", help="Target domains (default: every target of --task)")
    parser.add_argument("--rounds", type=int, default=1, help="Passes over the target stream, never reset")
    parser.add_argument("--max-images", type=int, help="Limit images per domain for a smoke test")
    parser.add_argument("--image-size", type=int, help="Default: the task's (512 OD/OC, 384 prostate, 352 polyp)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="PyTorch device (default: the method's, usually cuda when available)")
    parser.add_argument("--output", type=Path,
                        help="Default: results/<method>/[<task>_]<method>[_long_term]_domain_A.json")
    if method in METHODS:
        importlib.import_module(METHODS[method]).add_arguments(parser.add_argument_group(f"{method} options"))
    return parser


def parse_args() -> argparse.Namespace:
    # Parse --method first, then again with that method's options added.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--method", choices=list(METHODS))
    method = pre.parse_known_args()[0].method
    args = build_parser(method).parse_args()
    task = TASKS[args.task]
    args.domains = args.domains or list(task.target_domains)
    unknown = set(args.domains) - set(task.target_domains)
    if unknown:
        raise SystemExit(f"Task {task.name} has no target domain(s) {sorted(unknown)}")
    if args.rounds < 1 or (args.max_images is not None and args.max_images < 1):
        raise SystemExit("--rounds and --max-images must be positive")
    args.image_size = args.image_size or task.image_size
    return args


def main(args: argparse.Namespace) -> None:
    task = TASKS[args.task]
    name = method_name(args)
    args.output = args.output or default_output(args, name)
    seed_everything(args.seed)
    domains = task.build_domains(args.domains, args.max_images)
    module = importlib.import_module(METHODS[args.method])
    segmenter = module.build(args, task)
    print(f"Method: {name}; task {task.name}; target stream {' -> '.join(args.domains)}"
          + (f", {args.rounds} rounds without reset" if args.rounds > 1 else ""))
    print("Metric: mean per-image DSC" + ("; OD includes the OC region" if task.name == "odoc" else ""))

    run_name = name + ("_long_term" if args.rounds > 1 else "")
    with tracked_run(run_name, args):
        rounds = []
        for index in range(1, args.rounds + 1):
            if args.rounds > 1:
                print(f"=== Round {index} ===")
            summaries, per_image, variants = run_stream(segmenter, task, domains, args, index)
            rounds.append((summaries, per_image, variants))
            print(f"Do. A DSC (unweighted {'/'.join(args.domains)} domain mean): {do_a(summaries):.2f}")

        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        payload: dict[str, Any] = {"method": name, "task": task.name, "source_domain": "A", "config": config}
        summaries, per_image, variants = rounds[-1]
        if args.rounds == 1:
            payload["target_domains"] = summaries
            payload["do_a_dsc"] = do_a(summaries)
            payload["pooled_target_image_dsc"] = 100.0 * float(np.mean([row["mean_dsc"] for row in per_image]))
            if hasattr(module, "PAPER_DO_A"):
                payload["paper_do_a_reference"] = module.PAPER_DO_A.get(task.name)
            if variants:
                payload["diagnostic_variants"] = {
                    key: {"target_domains": rows, "do_a_dsc": do_a(rows)} for key, rows in variants.items()}
        else:
            scores = [do_a(summaries) for summaries, _, _ in rounds]
            average = float(np.mean(scores))
            source_only = source_only_do_a(args, args.domains)
            payload["rounds"] = [{"round": index, "do_a_dsc": score, "target_domains": summaries}
                                 for index, (score, (summaries, _, _)) in enumerate(zip(scores, rounds), 1)]
            payload.update(average_dsc=average, performance_degradation=scores[0] - average)
            if source_only is not None:
                payload.update(source_only_dsc=source_only, gain_dsc=average - source_only)
            print("Rounds: " + " | ".join(f"{score:.2f}" for score in scores) + f" | Average {average:.2f}"
                  + (f" | Gain {average - source_only:+.2f}" if source_only is not None else "")
                  + f" | Perform. Degra. {scores[0] - average:.2f}")
        payload["per_image"] = [row for _, rows, _ in rounds for row in rows]
        atomic_json_write(args.output, payload)
        log_result(payload, args.output)
        print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main(parse_args())
