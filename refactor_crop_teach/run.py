#!/usr/bin/env python3
"""Run one experiment on one source domain's target stream.

    python run.py --problem prostate --source A --method source_only
    python run.py --problem prostate --source A --method foundation_only
    python run.py --problem prostate --source A --method crop_teach --seed 0
    python run.py --problem prostate --source A --method crop_teach --seed 0 --rounds 3       # long-term
    python run.py --problem prostate --source A --method crop_teach --proposal largest_region
    python run.py --problem prostate --source A --method crop_teach --domains D,B,C,E        # another order

Methods: source_only and foundation_only are scored with EvaluateNonAdapting (the target model or
the foundation model, frozen); crop_teach with EvaluateAdapting. The stream is the paper's (all
other domains, alphabetical) unless --domains / --order say otherwise.

Writes runs/results/<problem>/<method>_source<S>[_seed<k>][_rounds<R>][_<strategy>...].json and a
per-image .csv next to it (skipped if it exists, unless --force), and logs the run to MLflow.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

import config
from method import tracking
from method.adapting import Adapting
from method.augmentation import JitterErasingAugmentation, NoAugmentation
from method.data import DomainSequenceLoader, ShuffledLoader
from method.evaluation import EvaluateAdapting, EvaluateNonAdapting
from method.fusion import MeanFusion, MostForegroundFusion
from method.learning import ReDiFMLearning
from method.models import load_model
from method.proposal import AllRegionsProposal, LargestRegionProposal

# Every strategy, by command-line name; the first of each is the default (the method as published).
PROPOSALS = {
    "all_regions": lambda: AllRegionsProposal(config.PSEUDO_THRESHOLD, config.CROP_SCALE, config.MIN_CROP),
    "largest_region": lambda: LargestRegionProposal(config.PSEUDO_THRESHOLD, config.CROP_SCALE, config.MIN_CROP),
}
FUSIONS = {"most_foreground": MostForegroundFusion, "mean": MeanFusion}
AUGMENTATIONS = {
    "jitter_erasing": lambda: JitterErasingAugmentation(config.JITTER, config.ERASING),
    "none": NoAugmentation,
}
LEARNINGS = {
    "redifm": lambda: ReDiFMLearning(config.LEARNING_RATE, config.PSEUDO_THRESHOLD, config.ENTROPY_THRESHOLD),
}
ORDERS = {"domain_sequence": DomainSequenceLoader, "shuffled": ShuffledLoader}
STRATEGIES = {"proposal": PROPOSALS, "fusion": FUSIONS, "augmentation": AUGMENTATIONS, "learning": LEARNINGS,
              "order": ORDERS}
METHODS = ("source_only", "foundation_only", "crop_teach")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", choices=list(config.PROBLEMS), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1, help="Passes over the stream, never reset")
    parser.add_argument("--domains", help="Comma-separated target domains, in order (default: the paper's)")
    for name, choices in STRATEGIES.items():
        parser.add_argument(f"--{name}", choices=list(choices), default=next(iter(choices)))
    parser.add_argument("--max-images", type=int, help="Per domain, for a quick check")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return args.output
    name = f"{args.method}_source{args.source}"
    if args.method == "crop_teach":
        name += f"_seed{args.seed}"
    if args.rounds > 1:
        name += f"_rounds{args.rounds}"
    for strategy, choices in STRATEGIES.items():
        if getattr(args, strategy) != next(iter(choices)):
            name += f"_{getattr(args, strategy)}"
    if args.domains:
        name += "_" + args.domains.replace(",", "")
    return config.RESULTS_DIR / args.problem / f"{name}.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_evaluator(args: argparse.Namespace, problem: config.Problem, device: torch.device):
    domains = args.domains.split(",") if args.domains else problem.stream(args.source)
    loader_class = ORDERS[args.order]
    extra = {"seed": args.seed} if loader_class is ShuffledLoader else {}
    loader = loader_class(problem, config.DATA_DIR, domains, args.max_images, device, **extra)

    def checkpoint(kind: str) -> Path:
        return config.checkpoint_path(args.problem, args.source, kind)

    log = tracking.log_metrics
    if args.method == "source_only":
        return EvaluateNonAdapting(load_model(checkpoint("target"), device), loader, problem, args.rounds, log)
    foundation = load_model(checkpoint("head"), device, config.SAM_CHECKPOINT)
    if args.method == "foundation_only":
        return EvaluateNonAdapting(foundation, loader, problem, args.rounds, log)
    refiner = load_model(checkpoint("crop"), device)
    target = load_model(checkpoint("target"), device)
    adapting = Adapting(target, foundation, refiner, PROPOSALS[args.proposal](), FUSIONS[args.fusion](),
                        AUGMENTATIONS[args.augmentation](), LEARNINGS[args.learning]())
    return EvaluateAdapting(adapting, loader, problem, args.rounds, log)


def main() -> None:
    args = parse_args()
    problem = config.PROBLEMS[args.problem]
    output = output_path(args)
    if output.exists() and not args.force:
        print(f"Exists, skipped: {output}")
        return

    seed_everything(args.seed)  # Before the models are built: their construction draws random numbers.
    device = torch.device(args.device)
    evaluator = build_evaluator(args, problem, device)
    stream = evaluator.loader.domains
    print(f"{args.problem}: {args.method}, source {args.source}, stream {' -> '.join(stream)}, "
          f"seed {args.seed}, {args.rounds} round(s)", flush=True)

    with tracking.run(f"{args.method}_{args.problem}_source{args.source}", vars(args), config.MLFLOW_URI,
                      config.RUNS_DIR):
        result = evaluator.run()
        summary = {"problem": args.problem, "source": args.source, "method": args.method, "seed": args.seed,
                   "stream": stream, "rounds": result.rounds, "do_dsc": result.rounds[0]["mean"]["mean_dsc"],
                   "ms_per_image": result.ms_per_image, "minutes": result.seconds / 60,
                   "strategies": {name: getattr(args, name) for name in STRATEGIES},
                   "config": {"learning_rate": config.LEARNING_RATE, "pseudo_threshold": config.PSEUDO_THRESHOLD,
                              "entropy_threshold": config.ENTROPY_THRESHOLD, "crop_scale": config.CROP_SCALE,
                              "min_crop": config.MIN_CROP, "max_images": args.max_images}}
        if args.rounds > 1:
            scores = [r["mean"]["mean_dsc"] for r in result.rounds]
            summary.update(average_dsc=float(np.mean(scores)), degradation=scores[0] - float(np.mean(scores)))
        tracking.log_metrics({"minutes": result.seconds / 60, "ms_per_image": result.ms_per_image})
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2))
        with open(output.with_suffix(".csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in result.rows for key in row}))
            writer.writeheader()
            writer.writerows(result.rows)
        tracking.log_artifact(output)
        tracking.log_artifact(output.with_suffix(".csv"))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
