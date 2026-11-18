"""Evaluation: run a method over a DataLoader's stream and score every prediction.

    EvaluateNonAdapting   a fixed Model predicts each image (source-only, foundation-only)
    EvaluateAdapting      an Adapting adapts on each image, then predicts it (Crop-Teach)

Each image's prediction is scored right after it is made (the paper's online protocol). With
``rounds`` > 1 the whole stream repeats, never reset (the paper's long-term setting).
Per-domain scores are means over the domain's images, in percent; a round's score is the mean
over domains (Do. X in the paper).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from config import Problem
from method.adapting import Adapting
from method.data import DataLoader, TestImage
from method.metrics import score
from method.models import Model


@dataclass
class EvaluationResult:
    rounds: list[dict] = field(default_factory=list)  # per round: per-domain and mean scores
    rows: list[dict] = field(default_factory=list)    # one per image and round
    seconds: float = 0.0

    @property
    def ms_per_image(self) -> float:
        return 1000 * self.seconds / max(len(self.rows), 1)


class Evaluator(ABC):
    def __init__(self, loader: DataLoader, problem: Problem, rounds: int = 1, log=None) -> None:
        self.loader = loader
        self.problem = problem
        self.rounds = rounds
        self.log = log or (lambda metrics, step: None)

    @abstractmethod
    def predict(self, test: TestImage) -> tuple[torch.Tensor, dict]:
        """Class probabilities (C, S, S) of the image, and per-image statistics."""

    def run(self) -> EvaluationResult:
        result = EvaluationResult()
        start = time.time()
        for round_index in range(1, self.rounds + 1):
            per_domain = defaultdict(list)
            for test in self.loader:
                classes, statistics = self.predict(test)
                scores = self.score(classes, test.mask, polyp_metrics=self.problem.name == "polyp")
                for name, teacher in statistics.pop("teachers", {}).items():
                    scores[f"{name}_mean_dsc"] = self.score(teacher, test.mask, polyp_metrics=False)["mean_dsc"]
                per_domain[test.domain].append(scores)
                result.rows.append({"round": round_index, "domain": test.domain, "image": test.name,
                                    **scores, **statistics})
            result.rounds.append(self.summarize(round_index, per_domain))
        result.seconds = time.time() - start
        return result

    def score(self, classes: torch.Tensor, mask: torch.Tensor, polyp_metrics: bool) -> dict[str, float]:
        scores = score(classes.argmax(0).cpu(), 1 - classes[0], mask, self.problem.structures, polyp_metrics)
        scores["mean_dsc"] = float(np.mean([scores[f"{name}_dsc"] for name in self.problem.structures]))
        return scores

    def summarize(self, round_index: int, per_domain: dict[str, list[dict]]) -> dict:
        domains = {}
        for domain, scores in per_domain.items():
            domains[domain] = {key: 100 * float(np.mean([s[key] for s in scores])) for key in scores[0]}
            domains[domain]["images"] = len(scores)
            print(f"round {round_index} domain {domain}: " + ", ".join(
                f"{k}={v:.2f}" for k, v in domains[domain].items() if k != "images"), flush=True)
        keys = [key for key in next(iter(domains.values())) if key != "images"]
        mean = {key: float(np.mean([d[key] for d in domains.values()])) for key in keys}
        print(f"round {round_index} mean over domains: DSC={mean['mean_dsc']:.2f}", flush=True)
        self.log({f"do_{key}": value for key, value in mean.items()}, round_index)
        return {"round": round_index, "domains": domains, "mean": mean}


class EvaluateNonAdapting(Evaluator):
    def __init__(self, model: Model, loader: DataLoader, problem: Problem, rounds: int = 1, log=None) -> None:
        super().__init__(loader, problem, rounds, log)
        self.model = model

    def predict(self, test: TestImage) -> tuple[torch.Tensor, dict]:
        return self.model.predict(test.image)[0], {}


class EvaluateAdapting(Evaluator):
    def __init__(self, adapting: Adapting, loader: DataLoader, problem: Problem, rounds: int = 1, log=None) -> None:
        super().__init__(loader, problem, rounds, log)
        self.adapting = adapting

    def predict(self, test: TestImage) -> tuple[torch.Tensor, dict]:
        return self.adapting(test.image, test.full)
