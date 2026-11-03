"""Shared helpers: repo paths, config loading, MLflow setup."""
import os
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parents[1]
ENTITY_KEY = "athlete_id"
TIMESTAMP_COL = "event_timestamp"


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_mlflow(cfg: dict) -> str:
    """Local MLflow: SQLite backend + local artifact folder. Returns the experiment id.

    MLFLOW_TRACKING_URI wins if set (e.g. when launched with `mlflow run`)."""
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    name = cfg["mlflow"]["experiment_name"]
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name, artifact_location=(ROOT / "mlartifacts").as_uri()
        )
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(name)
    return experiment_id
