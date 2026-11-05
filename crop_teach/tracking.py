"""MLflow logging (runs/mlruns.db). CT_MLFLOW=0, or mlflow not installed, turns it off."""

from __future__ import annotations

import contextlib
import os

from config import MLFLOW_URI, RUNS_DIR

ENABLED = os.environ.get("CT_MLFLOW", "1") != "0"
try:
    import mlflow
except ImportError:
    ENABLED = False


@contextlib.contextmanager
def run(name: str, params: dict):
    if not ENABLED:
        yield
        return
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("crop_teach")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({key: str(value)[:500] for key, value in params.items()})
        yield


def log_metrics(metrics: dict, step: int | None = None) -> None:
    if ENABLED and mlflow.active_run() is not None:
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)
