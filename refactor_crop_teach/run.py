#!/usr/bin/env python3
"""Run every evaluation: for each problem and source domain, source only, the SAM head alone and
Crop-Teach, on the stream below. No arguments; change the constants or swap a component here.

    python run.py

Writes results/<problem>/<method>_source<S>.{json,csv}, where <method> is source_only, sam_head or
crop_teach_seed<k>; an evaluation whose JSON exists is skipped.
"""

from __future__ import annotations

import torch

from adaptation.adapting import Adapting
from data.data_loader import AlphabeticalStream
from evaluation.evaluate import EvaluateAdapting, EvaluateNonAdapting, Result
from adaptation.fusion import MostForeground
from adaptation.learning import CropTeachLearning
from modeling.models import ResUNet34, SamHead
from core.problem import ODOC, PROSTATE, RESULTS_DIR, Problem
from adaptation.region_proposal import PerRegionSquare

DEVICE = torch.device("cuda")
PROBLEMS = (ODOC, PROSTATE)
SEEDS = (0,)                         # one Crop-Teach run per seed
STREAM = AlphabeticalStream          # or LongTermStream; its name goes into the result file's name
TAG = "" if STREAM is AlphabeticalStream else f"_{STREAM.__name__}"


def evaluations(problem: Problem, source: str, encoder: torch.nn.Module) -> dict:
    """result name -> a function building its evaluator (built only if the result is missing)."""
    classes = problem.num_classes
    loader = STREAM(problem, source, DEVICE)

    def resunet(kind: str) -> ResUNet34:
        return ResUNet34(classes).load(problem.checkpoint(source, kind)).to(DEVICE)

    def sam_head() -> SamHead:
        return SamHead(classes, encoder).load(problem.checkpoint(source, "head")).to(DEVICE)

    def crop_teach(seed: int) -> EvaluateAdapting:
        return EvaluateAdapting(loader, Adapting(
            full_model=sam_head(), crop_model=resunet("crop"), target_model=resunet("target"),
            region_proposal=PerRegionSquare(), fusion=MostForeground(), learning=CropTeachLearning(), seed=seed))

    return {
        "source_only": lambda: EvaluateNonAdapting(loader, resunet("target"), "source_only"),
        "sam_head": lambda: EvaluateNonAdapting(loader, sam_head(), "sam_head"),
        **{f"crop_teach_seed{seed}": lambda seed=seed: crop_teach(seed) for seed in SEEDS},
    }


def report(result: Result) -> None:
    for entry in result.rounds:
        domains = ", ".join(f"{domain} {scores['mean_dsc']:.2f}" for domain, scores in entry["domains"].items())
        print(f"  round {entry['round']}: DSC {entry['mean']['mean_dsc']:.2f} ({domains})", flush=True)


def main() -> None:
    encoder = SamHead.load_encoder(DEVICE)
    for problem in PROBLEMS:
        for source in problem.domains:
            for method, build in evaluations(problem, source, encoder).items():
                path = RESULTS_DIR / problem.name / f"{method}_source{source}{TAG}"
                if path.with_suffix(".json").exists():
                    print(f"Exists, skipped: {path}.json")
                    continue
                print(f"{problem.name}, source {source}: {method}", flush=True)
                result = build().run()
                report(result)
                result.save(path)


if __name__ == "__main__":
    main()
