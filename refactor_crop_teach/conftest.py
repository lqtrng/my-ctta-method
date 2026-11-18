"""Lets pytest import the modules of this folder (and tests/helpers.py) by their plain names."""

import os
import sys
import tempfile
from pathlib import Path

import pytest

MLRUNS = Path(tempfile.mkdtemp(prefix="ct_mlruns_"))   # the tests log to a throwaway MLflow store

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "tests")]


@pytest.fixture(autouse=True)
def mlflow_run():
    """Each test runs inside its own MLflow run, as run.py opens one around each evaluation."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS / 'mlflow.db'}")
    if mlflow.get_experiment_by_name("tests") is None:
        mlflow.create_experiment("tests", artifact_location=(MLRUNS / "artifacts").as_uri())
    mlflow.set_experiment("tests")
    with mlflow.start_run():
        yield
