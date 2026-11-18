"""The evaluation layer: run a DataLoader through a model and score it.

score(prediction, label) -> per image -> per domain -> per round -> Result (JSON / CSV).

The evaluators log into the active MLflow run (run.py opens it):
params for what was run; each image's scores as it is scored (step = position in the stream);
domain/<key>, one point per domain as it finishes (step = the domain's position in the stream);
the per-round and per-domain summary at the end, and the summary JSON as an artifact.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import numpy as np
import torch

from adaptation.adapting import Adapting
from data.data_loader import DataLoader
from modeling.models import Model


def dice(prediction: torch.Tensor, label: torch.Tensor) -> float:
    prediction, label = prediction.flatten().double(), label.flatten().double()
    return float((2.0 * (prediction * label).sum() + 1e-6) / (prediction.sum() + label.sum() + 1e-6))


def score(prediction: torch.Tensor, label: torch.Tensor, structures: tuple[str, ...]) -> dict[str, float]:
    """prediction: class probabilities (C, S, S); label: class indices (S, S).
    Dice of every nested structure (structure k is class >= k + 1), and their mean."""
    classes = prediction.argmax(0).cpu()
    scores = {f"{name}_dsc": dice(classes >= index + 1, label >= index + 1) for index, name in enumerate(structures)}
    scores["mean_dsc"] = float(np.mean(list(scores.values())))
    return scores


@dataclass
class Result:
    info: dict                                    # what was run: problem, source, method, ...
    rows: list[dict] = field(default_factory=list)      # one per image
    rounds: list[dict] = field(default_factory=list)    # one per round: per-domain and mean scores (x100)
    seconds: float = 0.0

    @property
    def do_dsc(self) -> float:
        """Mean DSC over the domains of the first round, in percent."""
        return self.rounds[0]["mean"]["mean_dsc"]

    def summarize(self) -> None:
        """Per image -> per domain -> per round."""
        self.rounds = []
        for round_index in sorted({row["round"] for row in self.rows}):
            in_round = [row for row in self.rows if row["round"] == round_index]
            domains = {}
            for domain in dict.fromkeys(row["domain"] for row in in_round):
                scores = [row for row in in_round if row["domain"] == domain]
                keys = [key for key in scores[0] if key.endswith("_dsc")]
                domains[domain] = {key: 100 * float(np.mean([row[key] for row in scores])) for key in keys}
                domains[domain]["images"] = len(scores)
            keys = [key for key in next(iter(domains.values())) if key != "images"]
            mean = {key: float(np.mean([domain[key] for domain in domains.values()])) for key in keys}
            self.rounds.append({"round": round_index, "domains": domains, "mean": mean})

    def to_dict(self) -> dict:
        result = {**self.info, "rounds": self.rounds, "do_dsc": self.do_dsc,
                  "ms_per_image": 1000 * self.seconds / max(len(self.rows), 1), "minutes": self.seconds / 60}
        if len(self.rounds) > 1:
            scores = [entry["mean"]["mean_dsc"] for entry in self.rounds]
            result.update(average_dsc=float(np.mean(scores)), degradation=scores[0] - float(np.mean(scores)))
        return result

    def metrics(self) -> dict[str, float]:
        """The summary flattened to name -> number: do_dsc, ms_per_image, ..., and round<r>/<domain or mean>/<key>."""
        summary = self.to_dict()
        flat = {key: summary[key] for key in ("do_dsc", "ms_per_image", "minutes", "average_dsc", "degradation")
                if key in summary}
        for entry in self.rounds:
            flat.update({f"round{entry['round']}/mean/{key}": value for key, value in entry["mean"].items()})
            for domain, scores in entry["domains"].items():
                flat.update({f"round{entry['round']}/{domain}/{key}": value for key, value in scores.items()})
        return flat

    def save(self, path: Path) -> None:
        """Writes <path>.json (the summary) and <path>.csv (one row per image)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".json").write_text(json.dumps(self.to_dict(), indent=2))
        with open(path.with_suffix(".csv"), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in self.rows for key in row}))
            writer.writeheader()
            writer.writerows(self.rows)


def domain_point(rows: list[dict]) -> dict[str, float]:
    """The mean of every _dsc score over one domain's images (x100), as domain/<key>."""
    return {f"domain/{key}": 100 * float(np.mean([row[key] for row in rows])) for key in rows[0] if key.endswith("_dsc")}


class EvaluateAdapting:
    """DataLoader -> Adapting -> score. Adapting gets ``image`` and ``full`` only, never ``label``.
    Logs into the active MLflow run (run.py opens it and logs the hyperparameters)."""

    def __init__(self, loader: DataLoader, adapting: Adapting, method: str = "crop_teach") -> None:
        self.loader = loader
        self.adapting = adapting
        self.method = method

    def run(self) -> Result:
        structures = self.loader.problem.structures
        result = Result({"problem": self.loader.problem.name, "source": self.loader.source, "method": self.method,
                         "seed": getattr(self.adapting, "seed", None), "stream": self.loader.domains()})
        mlflow.log_params({key: str(value) for key, value in result.info.items()})
        domain, in_domain, points = None, [], 0
        for step, test_image in enumerate(self.loader):
            if (test_image.round, test_image.domain) != domain and in_domain:    # a domain just finished
                mlflow.log_metrics(domain_point(in_domain), step=points)
                in_domain, points = [], points + 1
            domain = (test_image.round, test_image.domain)
            start = time.time()
            prediction = self.adapting(test_image.image, test_image.full)
            result.seconds += time.time() - start
            scores = {**score(prediction, test_image.label, structures), **getattr(self.adapting, "last", {})}
            result.rows.append({"round": test_image.round, "domain": test_image.domain, "image": test_image.name,
                                **scores})
            in_domain.append(scores)
            mlflow.log_metrics({**scores, "round": test_image.round}, step=step)
        if in_domain:
            mlflow.log_metrics(domain_point(in_domain), step=points)
        result.summarize()
        mlflow.log_metrics(result.metrics())
        mlflow.log_dict(result.to_dict(), "result.json")
        return result


class EvaluateNonAdapting:
    """DataLoader -> one frozen Model -> score: source only (the target ResUNet34) or the SAM head alone.
    Logs into the active MLflow run (run.py opens it)."""

    def __init__(self, loader: DataLoader, model: Model, method: str = "source_only") -> None:
        self.loader = loader
        self.model = model.requires_grad_(False).eval()
        self.method = method

    @torch.no_grad()
    def run(self) -> Result:
        structures = self.loader.problem.structures
        result = Result({"problem": self.loader.problem.name, "source": self.loader.source, "method": self.method,
                         "stream": self.loader.domains()})
        mlflow.log_params({key: str(value) for key, value in result.info.items()})
        domain, in_domain, points = None, [], 0
        for step, test_image in enumerate(self.loader):
            if (test_image.round, test_image.domain) != domain and in_domain:    # a domain just finished
                mlflow.log_metrics(domain_point(in_domain), step=points)
                in_domain, points = [], points + 1
            domain = (test_image.round, test_image.domain)
            start = time.time()
            prediction = self.model(test_image.image)[0]
            result.seconds += time.time() - start
            scores = score(prediction, test_image.label, structures)
            result.rows.append({"round": test_image.round, "domain": test_image.domain, "image": test_image.name,
                                **scores})
            in_domain.append(scores)
            mlflow.log_metrics({**scores, "round": test_image.round}, step=step)
        if in_domain:
            mlflow.log_metrics(domain_point(in_domain), step=points)
        result.summarize()
        mlflow.log_metrics(result.metrics())
        mlflow.log_dict(result.to_dict(), "result.json")
        return result
