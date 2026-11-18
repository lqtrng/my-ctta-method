from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "refactor_crop_teach"

import numpy as np
import torch

from . import tracking
from .adapting import Adapting
from .augmentation import AUGMENTATIONS, build_augmentation
from .config import (
    CROP_SCALE,
    ENTROPY_THRESHOLD,
    LEARNING_RATE,
    MIN_CROP,
    PROBLEMS,
    PSEUDO_THRESHOLD,
    RESULTS_DIR,
    SAM_CHECKPOINT,
    checkpoint_path,
)
from .dataloader import (
    AlphabeticalDomainDataLoader,
    CustomDomainOrderDataLoader,
    DataLoader,
    ReversedStreamDataLoader,
    ShuffledStreamDataLoader,
)
from .fusion import FUSIONS, build_fusion
from .learning import LEARNING_STRATEGIES, build_learning
from .metrics import score
from .models import CropRefineModel, Model, SAMHeadModel, load_cnn_model
from .proposal import PROPOSALS, build_proposal


def _aggregate_round(
    round_index: int,
    domains_order: list[str],
    per_domain_scores: dict[str, list[dict[str, float]]],
) -> dict:
    domains: dict[str, dict[str, float | int]] = {}
    for domain in domains_order:
        items = per_domain_scores[domain]
        domains[domain] = {
            key: 100.0 * float(np.mean([entry[key] for entry in items])) for key in items[0]
        }
        domains[domain]["images"] = len(items)
        metrics_str = ", ".join(
            f"{key}={val:.2f}" for key, val in domains[domain].items() if key != "images"
        )
        print(f"round {round_index} domain {domain}: {metrics_str}", flush=True)
    first_domain = next(iter(domains.values()))
    summary = {
        key: float(np.mean([d[key] for d in domains.values()]))
        for key in first_domain
        if key != "images"
    }
    return {"round": round_index, "domains": domains, "mean": summary}


def _save_outputs(output: Path, result: dict, rows: list[dict]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    with open(output.with_suffix(".csv"), "w", newline="") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output}")


class EvaluateNonAdapting:
    def __init__(
        self,
        dataloader: DataLoader,
        model: Model,
        method_name: str = "source_only",
        seed: int = 0,
    ) -> None:
        self.dataloader = dataloader
        self.model = model.freeze()
        self.method_name = method_name
        self.seed = seed

    def evaluate(self, output: Path | None = None) -> dict:
        problem = self.dataloader.problem
        polyp = problem.name == "polyp"
        domains_order = self.dataloader.domains
        per_domain_scores: dict[str, list[dict[str, float]]] = {d: [] for d in domains_order}
        rows: list[dict] = []
        start = time.time()
        images = 0

        params = {
            "problem": problem.name,
            "source": self.dataloader.source,
            "method": self.method_name,
            "seed": self.seed,
            "max_images": self.dataloader.max_images,
        }
        with tracking.run(f"{self.method_name}_{problem.name}_source{self.dataloader.source}", params):
            for item in self.dataloader:
                classes = self.model.predict_probabilities(item.image, item.full_image)[0]
                s = score(
                    classes.argmax(0).cpu(),
                    1.0 - classes[0],
                    item.target,
                    problem.structures,
                    polyp,
                )
                s["mean_dsc"] = float(np.mean([s[f"{name}_dsc"] for name in problem.structures]))
                per_domain_scores[item.domain].append(s)
                rows.append({"round": 1, "domain": item.domain, "image": item.name, **s})
                images += 1

            round_summary = _aggregate_round(1, domains_order, per_domain_scores)
            print(
                f"round 1 Do. {self.dataloader.source}: DSC={round_summary['mean']['mean_dsc']:.2f}",
                flush=True,
            )
            tracking.log_metrics(
                {f"do_{key}": val for key, val in round_summary["mean"].items()}, step=1
            )

            seconds = time.time() - start
            result = {
                "problem": problem.name,
                "source": self.dataloader.source,
                "method": self.method_name,
                "seed": self.seed,
                "stream": domains_order,
                "rounds": [round_summary],
                "do_dsc": round_summary["mean"]["mean_dsc"],
                "ms_per_image": 1000.0 * seconds / max(images, 1),
                "minutes": seconds / 60.0,
                "config": {
                    "learning_rate": LEARNING_RATE,
                    "pseudo_threshold": PSEUDO_THRESHOLD,
                    "entropy_threshold": ENTROPY_THRESHOLD,
                    "crop_scale": CROP_SCALE,
                    "min_crop": MIN_CROP,
                    "max_images": self.dataloader.max_images,
                },
            }
            tracking.log_metrics(
                {"minutes": result["minutes"], "ms_per_image": result["ms_per_image"]}
            )

        if output is not None:
            _save_outputs(output, result, rows)
        return result

    def __call__(self, output: Path | None = None) -> dict:
        return self.evaluate(output=output)


class EvaluateAdapting:
    def __init__(
        self,
        dataloader: DataLoader,
        adapting: Adapting,
        rounds: int = 1,
        method_name: str = "crop_teach",
        seed: int = 0,
    ) -> None:
        if rounds < 1:
            raise ValueError("rounds must be >= 1")
        self.dataloader = dataloader
        self.adapting = adapting
        self.rounds = rounds
        self.method_name = method_name
        self.seed = seed

    def evaluate(self, output: Path | None = None) -> dict:
        problem = self.dataloader.problem
        polyp = problem.name == "polyp"
        domains_order = self.dataloader.domains
        rounds_list: list[dict] = []
        rows: list[dict] = []
        start = time.time()
        images = 0

        params = {
            "problem": problem.name,
            "source": self.dataloader.source,
            "method": self.method_name,
            "seed": self.seed,
            "rounds": self.rounds,
            "max_images": self.dataloader.max_images,
        }
        with tracking.run(f"{self.method_name}_{problem.name}_source{self.dataloader.source}", params):
            for round_index in range(1, self.rounds + 1):
                per_domain_scores: dict[str, list[dict[str, float]]] = {d: [] for d in domains_order}
                for item in self.dataloader:
                    classes, extras = self.adapting.adapt_and_predict(item.image, item.full_image)
                    s = score(
                        classes.argmax(0).cpu(),
                        1.0 - classes[0],
                        item.target,
                        problem.structures,
                        polyp,
                    )
                    s["mean_dsc"] = float(
                        np.mean([s[f"{name}_dsc"] for name in problem.structures])
                    )
                    for teacher_name, teacher_probs in extras.pop("teachers", {}).items():
                        t = score(
                            teacher_probs.argmax(0).cpu(),
                            1.0 - teacher_probs[0],
                            item.target,
                            problem.structures,
                            False,
                        )
                        s[f"{teacher_name}_mean_dsc"] = float(
                            np.mean([t[f"{k}_dsc"] for k in problem.structures])
                        )
                    per_domain_scores[item.domain].append(s)
                    rows.append(
                        {
                            "round": round_index,
                            "domain": item.domain,
                            "image": item.name,
                            **s,
                            **extras,
                        }
                    )
                    images += 1

                round_summary = _aggregate_round(round_index, domains_order, per_domain_scores)
                rounds_list.append(round_summary)
                print(
                    f"round {round_index} Do. {self.dataloader.source}: "
                    f"DSC={round_summary['mean']['mean_dsc']:.2f}",
                    flush=True,
                )
                tracking.log_metrics(
                    {f"do_{key}": val for key, val in round_summary["mean"].items()},
                    step=round_index,
                )

            seconds = time.time() - start
            result: dict = {
                "problem": problem.name,
                "source": self.dataloader.source,
                "method": self.method_name,
                "seed": self.seed,
                "stream": domains_order,
                "rounds": rounds_list,
                "do_dsc": rounds_list[0]["mean"]["mean_dsc"],
                "ms_per_image": 1000.0 * seconds / max(images, 1),
                "minutes": seconds / 60.0,
                "config": {
                    "learning_rate": self.adapting.learning.learning_rate,
                    "pseudo_threshold": getattr(
                        self.adapting.learning, "pseudo_threshold", PSEUDO_THRESHOLD
                    ),
                    "entropy_threshold": getattr(
                        self.adapting.learning, "entropy_threshold", ENTROPY_THRESHOLD
                    ),
                    "crop_scale": getattr(self.adapting.region_proposal, "crop_scale", CROP_SCALE),
                    "min_crop": getattr(self.adapting.region_proposal, "min_crop", MIN_CROP),
                    "max_images": self.dataloader.max_images,
                },
            }
            if self.rounds > 1:
                round_scores = [r["mean"]["mean_dsc"] for r in rounds_list]
                avg_dsc = float(np.mean(round_scores))
                result.update(average_dsc=avg_dsc, degradation=round_scores[0] - avg_dsc)
            tracking.log_metrics(
                {"minutes": result["minutes"], "ms_per_image": result["ms_per_image"]}
            )

        if output is not None:
            _save_outputs(output, result, rows)
        return result

    def __call__(self, output: Path | None = None) -> dict:
        return self.evaluate(output=output)


evaluate_non_adapting = EvaluateNonAdapting
evaluate_adapting = EvaluateAdapting


def _build_dataloader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    if args.order == "custom":
        if not args.domains:
            raise ValueError("--domains is required when --order custom is selected")
        return CustomDomainOrderDataLoader(
            args.problem, args.source, args.domains, device=device, max_images=args.max_images
        )
    if args.order == "reversed":
        return ReversedStreamDataLoader(
            args.problem, args.source, device=device, max_images=args.max_images
        )
    if args.order == "shuffled":
        return ShuffledStreamDataLoader(
            args.problem,
            args.source,
            shuffle_seed=args.seed,
            device=device,
            max_images=args.max_images,
        )
    return AlphabeticalDomainDataLoader(
        args.problem, args.source, device=device, max_images=args.max_images
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=list(PROBLEMS), required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument(
        "--method",
        choices=("source_only", "head_alone", "crop_infer", "crop_teach"),
        required=True,
    )
    parser.add_argument("--proposal", choices=list(PROPOSALS), default="all")
    parser.add_argument("--fusion", choices=list(FUSIONS), default="max_foreground")
    parser.add_argument("--augmentation", choices=list(AUGMENTATIONS), default="color_jitter_erasing")
    parser.add_argument("--learning", choices=list(LEARNING_STRATEGIES), default="redifm")
    parser.add_argument(
        "--order",
        choices=("alphabetical", "custom", "reversed", "shuffled"),
        default="alphabetical",
    )
    parser.add_argument("--domains", nargs="+")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--sam-checkpoint", type=Path, default=SAM_CHECKPOINT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    non_adapting = args.method in ("source_only", "head_alone", "crop_infer")
    seed_tag = "" if non_adapting else f"_seed{args.seed}"
    round_tag = f"_rounds{args.rounds}" if args.rounds > 1 and not non_adapting else ""
    output = (
        args.output
        or RESULTS_DIR / args.problem / f"{args.method}_source{args.source}{seed_tag}{round_tag}.json"
    )
    if output.exists() and not args.force:
        print(f"Exists, skipped: {output}")
        return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dataloader = _build_dataloader(args, device)

    print(
        f"{args.problem}: {args.method}, source {args.source}, "
        f"stream {' -> '.join(dataloader.domains)}, seed {args.seed}, {args.rounds} round(s)",
        flush=True,
    )

    if args.method == "source_only":
        target_model = load_cnn_model(checkpoint_path(args.problem, args.source, "target"), device)
        evaluator = EvaluateNonAdapting(
            dataloader=dataloader, model=target_model, method_name=args.method, seed=args.seed
        )
        evaluator.evaluate(output=output)
        return

    if args.method == "head_alone":
        head_model = SAMHeadModel.from_checkpoint(
            checkpoint_path(args.problem, args.source, "head"),
            device=device,
            sam_checkpoint=args.sam_checkpoint,
        )
        evaluator = EvaluateNonAdapting(
            dataloader=dataloader, model=head_model, method_name=args.method, seed=args.seed
        )
        evaluator.evaluate(output=output)
        return

    proposal = build_proposal(args.proposal)
    fusion = build_fusion(args.fusion)

    if args.method == "crop_infer":
        coarse_model = SAMHeadModel.from_checkpoint(
            checkpoint_path(args.problem, args.source, "head"),
            device=device,
            sam_checkpoint=args.sam_checkpoint,
        )
        crop_model = load_cnn_model(checkpoint_path(args.problem, args.source, "crop"), device)
        refine_model = CropRefineModel(
            coarse_model=coarse_model,
            crop_model=crop_model,
            region_proposal=proposal,
            region_fusion=fusion,
        )
        evaluator = EvaluateNonAdapting(
            dataloader=dataloader, model=refine_model, method_name=args.method, seed=args.seed
        )
        evaluator.evaluate(output=output)
        return

    augmentation = build_augmentation(args.augmentation)
    learning = build_learning(
        args.learning, augmentation=augmentation, learning_rate=args.learning_rate
    )
    adapting = Adapting.from_checkpoints(
        problem=args.problem,
        source=args.source,
        device=device,
        sam_checkpoint=args.sam_checkpoint,
        region_proposal=proposal,
        region_fusion=fusion,
        learning=learning,
    )
    adapting_evaluator = EvaluateAdapting(
        dataloader=dataloader,
        adapting=adapting,
        rounds=args.rounds,
        method_name=args.method,
        seed=args.seed,
    )
    adapting_evaluator.evaluate(output=output)


if __name__ == "__main__":
    main()
