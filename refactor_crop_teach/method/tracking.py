"""MLflow logging into runs/mlruns.db. CT_MLFLOW=0 turns it off.

Browse: mlflow ui --backend-store-uri sqlite:///runs/mlruns.db
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

ENABLED = os.environ.get("CT_MLFLOW", "1") != "0"


@contextlib.contextmanager
def run(name: str, params: dict, uri: str, folder: Path, experiment: str = "refactor_crop_teach"):
    if not ENABLED:
        yield
        return
    import mlflow

    folder.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=name):
        mlflow.log_params({key: str(value)[:500] for key, value in params.items()})
        yield


def log_metrics(metrics: dict, step: int | None = None) -> None:
    if not ENABLED:
        return
    import mlflow

    if mlflow.active_run() is not None:
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)


def log_artifact(path: Path) -> None:
    if not ENABLED:
        return
    import mlflow

    if mlflow.active_run() is not None:
        mlflow.log_artifact(str(path))
