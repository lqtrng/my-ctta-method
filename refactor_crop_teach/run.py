#!/usr/bin/env python3
"""Run every evaluation: for each problem and source domain, source only, the SAM head alone and
Crop-Teach, on the stream below. No arguments; change the constants or swap a component here.

    python run.py

Writes results/<problem>/<method>_source<S>.{json,csv}, where <method> is source_only, sam_head or
crop_teach_seed<k>; an evaluation whose JSON exists is skipped.
"""

from __future__ import annotations

import mlflow
import torch

from adaptation.adapting import Adapting
from data.data_loader import AlphabeticalStream
from evaluation.evaluate import EvaluateAdapting, EvaluateNonAdapting
from adaptation.fusion import MostForeground
from adaptation.learning import CropTeachLearning
from modeling.models import ResUNet34, SamHead
from core.problem import HERE, ODOC, PROSTATE, RESULTS_DIR, Problem
from adaptation.region_proposal import PerRegionSquare

DEVICE = torch.device("cuda")
PROBLEMS = (ODOC, PROSTATE)
SEEDS = (0,)                         # one Crop-Teach run per seed
STREAM = AlphabeticalStream          # or LongTermStream; its name goes into the result file's name
TAG = "" if STREAM is AlphabeticalStream else f"_{STREAM.__name__}"
MLRUNS = HERE / "mlruns"             # MLflow store: mlflow ui --backend-store-uri sqlite:///<MLRUNS>/mlflow.db


def evaluations(problem: Problem, source: str, encoder: torch.nn.Module) -> dict:
    """result name -> a function building (MLflow experiment, evaluator, hyperparameters to log);
    built only if the result is missing."""
    classes = problem.num_classes
    loader = STREAM(problem, source, DEVICE)

    def resunet(kind: str) -> ResUNet34:
        return ResUNet34(classes).load(problem.checkpoint(source, kind)).to(DEVICE)

    def sam_head() -> SamHead:
        return SamHead(classes, encoder).load(problem.checkpoint(source, "head")).to(DEVICE)

    def non_adapting(kind: str, model: torch.nn.Module, method: str) -> tuple:
        params = {"loader": STREAM.__name__, "model": type(model).__name__,
                  "checkpoint": problem.checkpoint(source, kind)}
        return "non_adapting", EvaluateNonAdapting(loader, model, method), params

    def crop_teach(seed: int) -> tuple:
        region_proposal, fusion, learning = PerRegionSquare(), MostForeground(), CropTeachLearning()
        adapting = Adapting(full_model=sam_head(), crop_model=resunet("crop"), target_model=resunet("target"),
                            region_proposal=region_proposal, fusion=fusion, learning=learning, seed=seed)
        params = {
            "loader": STREAM.__name__,
            "checkpoint.full_model": problem.checkpoint(source, "head"),
            "checkpoint.crop_model": problem.checkpoint(source, "crop"),
            "checkpoint.target_model": problem.checkpoint(source, "target"),
            "region_proposal": type(region_proposal).__name__,
            "region_proposal.threshold": region_proposal.THRESHOLD,
            "region_proposal.scale": region_proposal.SCALE,
            "region_proposal.min_side": region_proposal.MIN_SIDE,
            "fusion": type(fusion).__name__,
            "learning": type(learning).__name__,
            "learning.learning_rate": learning.LEARNING_RATE,
            "learning.pseudo_threshold": learning.PSEUDO_THRESHOLD,
            "learning.crop_threshold": learning.CROP_THRESHOLD,
            "learning.entropy_threshold": learning.ENTROPY_THRESHOLD,
            "learning.gamma": learning.GAMMA,
            "learning.mu": learning.MU,
            "learning.sigma2": learning.SIGMA2,
            "augmentation": type(learning.augmentation).__name__,
            "augmentation.jitter": learning.augmentation.JITTER,
            "augmentation.erasing": learning.augmentation.ERASING,
        }
        return "adapting", EvaluateAdapting(loader, adapting), params

    return {
        "source_only": lambda: non_adapting("target", resunet("target"), "source_only"),
        "sam_head": lambda: non_adapting("head", sam_head(), "sam_head"),
        **{f"crop_teach_seed{seed}": lambda seed=seed: crop_teach(seed) for seed in SEEDS},
    }


def main() -> None:
    """Progress and scores go to MLflow, nothing is printed: one run per evaluation, in the
    experiment adapting or non_adapting; the evaluator logs its scores into it."""
    MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS / 'mlflow.db'}")
    encoder = SamHead.load_encoder(DEVICE)
    for problem in PROBLEMS:
        for source in problem.domains:
            for method, build in evaluations(problem, source, encoder).items():
                path = RESULTS_DIR / problem.name / f"{method}_source{source}{TAG}"
                if path.with_suffix(".json").exists():
                    continue
                experiment, evaluator, params = build()
                if mlflow.get_experiment_by_name(experiment) is None:
                    mlflow.create_experiment(experiment, artifact_location=(MLRUNS / "artifacts" / experiment).as_uri())
                mlflow.set_experiment(experiment)
                with mlflow.start_run():
                    mlflow.log_params({key: str(value) for key, value in params.items()})
                    evaluator.run().save(path)


if __name__ == "__main__":
    main()
