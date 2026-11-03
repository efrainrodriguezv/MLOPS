"""Step 3 — one experiment = (feature version, hyperparameter set).

Features come from the Feast feature store (point-in-time join on the entity dataframe),
training runs inside a CodeCarbon tracker, and everything is logged to MLflow:
params, metrics, emissions, plots and the model itself.
"""
import inspect
import time

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from codecarbon import OfflineEmissionsTracker
from feast import FeatureStore
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.common import ENTITY_KEY, ROOT, TIMESTAMP_COL

matplotlib.use("Agg")


def get_training_frame(store: FeatureStore, service_name: str, labels: pd.DataFrame) -> pd.DataFrame:
    """Pull the features of one feature service for every labelled athlete."""
    service = store.get_feature_service(service_name)
    return store.get_historical_features(entity_df=labels, features=service).to_df()


def feature_columns(frame: pd.DataFrame, target: str) -> list[str]:
    excluded = {ENTITY_KEY, TIMESTAMP_COL, target, "split"}
    return sorted(c for c in frame.columns if c not in excluded)


def build_model(hparams: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(**hparams, random_state=seed, n_jobs=-1)),
    ])


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _plot_pred_vs_actual(y_true, y_pred, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=4, alpha=0.3)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color="tab:orange", lw=1.5, label="perfect prediction")
    ax.set(xlabel="actual total_lift (lb)", ylabel="predicted total_lift (lb)", title=title)
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_residuals(y_true, y_pred, title):
    resid = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(resid, bins=60, color="tab:blue")
    axes[0].set(title="Residual distribution", xlabel="actual − predicted (lb)")
    axes[1].scatter(y_pred, resid, s=4, alpha=0.3)
    axes[1].axhline(0, color="tab:orange", lw=1.5)
    axes[1].set(title="Residuals vs prediction", xlabel="predicted (lb)", ylabel="residual (lb)")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def _plot_importance(pipeline: Pipeline, columns: list[str], title):
    importances = pd.Series(pipeline.named_steps["model"].feature_importances_, index=columns)
    importances = importances.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(columns) + 1.5))
    importances.plot(kind="barh", ax=ax, color="tab:green")
    ax.set(title=title, xlabel="impurity-based importance")
    fig.tight_layout()
    return fig


def _log_model(model: Pipeline, example: pd.DataFrame) -> None:
    """Log the model in a way that works across MLflow versions.

    Newer MLflow saves sklearn models with skops and asks you to whitelist the
    tree types you created yourself; older MLflow has no such argument.
    """
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    kwargs = {"input_example": example}
    kwargs["name" if "name" in params else "artifact_path"] = "model"
    if "skops_trusted_types" in params:
        kwargs["skops_trusted_types"] = ["sklearn.tree._tree.Tree", "numpy.dtype"]
    mlflow.sklearn.log_model(model, **kwargs)


def run_experiment(store: FeatureStore, labels: pd.DataFrame, version: str, service_name: str,
                   hp_name: str, hparams: dict, cfg: dict) -> dict:
    target = cfg["model"]["target"]
    seed = cfg["data"]["seed"]
    run_name = f"features_{version}__{hp_name}"

    frame = get_training_frame(store, service_name, labels)
    columns = feature_columns(frame, target)
    train = frame[frame["split"] == "train"]
    test = frame[frame["split"] == "test"]

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tags({"feature_version": version, "feature_service": service_name,
                         "hp_set": hp_name, "algorithm": cfg["model"]["algorithm"]})
        mlflow.log_params({f"hp_{k}": v for k, v in hparams.items()})
        mlflow.log_params({"n_features": len(columns), "features": ",".join(columns),
                           "n_train": len(train), "n_test": len(test)})

        model = build_model(hparams, seed)
        emissions_dir = ROOT / "emissions"
        emissions_dir.mkdir(exist_ok=True)
        tracker = OfflineEmissionsTracker(
            country_iso_code=cfg["carbon"]["country_iso_code"],
            project_name=run_name,
            output_dir=str(emissions_dir),
            log_level="error",
        )
        tracker.start()
        start = time.perf_counter()
        model.fit(train[columns], train[target])
        preds = model.predict(test[columns])
        duration = time.perf_counter() - start
        emissions_kg = tracker.stop() or 0.0
        energy_kwh = float(getattr(tracker.final_emissions_data, "energy_consumed", 0.0) or 0.0)

        metrics = evaluate(test[target].to_numpy(), preds)
        metrics.update({
            "train_seconds": duration,
            "emissions_g_co2eq": emissions_kg * 1000,
            "energy_wh": energy_kwh * 1000,
        })
        mlflow.log_metrics(metrics)

        y_true = test[target].to_numpy()
        for fig, name in [
            (_plot_pred_vs_actual(y_true, preds, run_name), "pred_vs_actual.png"),
            (_plot_residuals(y_true, preds, run_name), "residuals.png"),
            (_plot_importance(model, columns, run_name), "feature_importance.png"),
        ]:
            mlflow.log_figure(fig, f"plots/{name}")
            plt.close(fig)

        # float example -> float schema, so integer features with gaps still validate
        _log_model(model, test[columns].head(5).astype("float64"))
        run_id = mlflow.active_run().info.run_id

    print(f"[train] {run_name}: rmse={metrics['rmse']:.1f} r2={metrics['r2']:.3f} "
          f"co2={metrics['emissions_g_co2eq']:.4f} g")
    return {"run_id": run_id, "run_name": run_name, "feature_version": version,
            "hp_set": hp_name, **metrics}
