"""MLflow tracking shared by every entry point.

Each script wraps its work in ``tracked_run`` and hands its result payload to
``log_result``; training loops call ``log_metrics`` once per epoch.  A run
records:

* params: every command-line option;
* tags: method, task, seed, git commit and whether the tree had local changes;
* metrics: Do. A, every number of every target domain (``B/disc_dsc``, ...),
  and each per-image number as a curve over the target stream (``stream/...``);
* artifacts: the result JSON the script writes anyway.

The store is ``mlruns/`` (see ``common.paths.MLFLOW_DIR``); browse it with

    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

``REDIFM_MLFLOW=0`` turns tracking off; the scripts then only write their JSON.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from common.paths import MLFLOW_DIR, PROJECT_DIR


def enabled() -> bool:
    return os.environ.get("REDIFM_MLFLOW", "1") != "0"


def _experiment(task: str) -> str:
    import mlflow

    if "MLFLOW_TRACKING_URI" not in os.environ:
        MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
    name = f"redifm-{task}"
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(MLFLOW_DIR / "artifacts" / name).as_uri())
    return name


def _git_tags() -> dict[str, str]:
    def git(*command: str) -> str:
        return subprocess.run(
            ["git", "-C", str(PROJECT_DIR), *command], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": str(bool(git("status", "--porcelain")))}
    except (OSError, subprocess.CalledProcessError):
        return {}


@contextmanager
def tracked_run(method: str, args: argparse.Namespace, task: str | None = None) -> Iterator[None]:
    """One MLflow run in the experiment ``redifm-<task>``, named ``<method>-seed<seed>``."""
    if not enabled():
        yield
        return
    import mlflow

    options = vars(args)
    task = task or options.get("task", "odoc")
    seed = options.get("seed")
    mlflow.set_experiment(_experiment(task))
    with mlflow.start_run(run_name=f"{method}-seed{seed}" if seed is not None else method):
        mlflow.set_tags(
            {
                "method": method,
                "task": task,
                "seed": str(seed),
                "smoke_test": str(bool(options.get("max_images") or options.get("smoke_test"))),
                **_git_tags(),
            }
        )
        mlflow.log_params({key: str(value)[:6000] for key, value in options.items()})
        yield


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log numbers to the active run, if there is one (e.g. a training loss per epoch)."""
    if not enabled():
        return
    import mlflow

    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics, step=step)


# Numbers in a payload that describe the run rather than measure it.
NOT_METRICS = {"image_size", "round"}


def _numbers(row: dict[str, Any], prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}{key}": float(value)
        for key, value in row.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool) and key not in NOT_METRICS
    }


def _domain_metrics(domains: list[dict[str, Any]], prefix: str = "") -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in domains:
        metrics.update(_numbers(row, f"{prefix}{row['domain']}/"))
    return metrics


def log_result(payload: dict[str, Any], output: Path | None = None) -> None:
    """Log a result payload (the dict a script saves as JSON) to the active run."""
    if not enabled():
        return
    import mlflow
    from mlflow.entities import Metric

    run = mlflow.active_run()
    if run is None:
        return
    metrics = _numbers(payload)
    metrics.update(_domain_metrics(payload.get("target_domains", [])))
    for row in payload.get("rounds", []):
        metrics.update(_numbers(row, f"round{row['round']}/"))
        metrics.update(_domain_metrics(row["target_domains"], f"round{row['round']}/"))
    for variant, result in payload.get("diagnostic_variants", {}).items():
        metrics.update(_numbers(result, f"{variant}/"))
        metrics.update(_domain_metrics(result["target_domains"], f"{variant}/"))
    mlflow.log_metrics(metrics)

    # One point per target image, in stream order, so drift and forgetting
    # along B -> C -> ... show up as curves.  Batched: thousands of points.
    now = int(time.time() * 1000)
    stream = [
        Metric(f"stream/{key}", value, now, step)
        for step, row in enumerate(payload.get("per_image", []))
        for key, value in _numbers(row).items()
    ]
    client = mlflow.MlflowClient()
    for start in range(0, len(stream), 1000):
        client.log_batch(run.info.run_id, metrics=stream[start : start + 1000])

    if output is not None and Path(output).is_file():
        mlflow.log_artifact(str(output))
