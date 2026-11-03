"""Step 4 — compare the experiments: metric table, metric/emission charts, side-by-side plots.

Writes everything to reports/ (commit these) and logs it to the parent MLflow run.
"""
import tempfile

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.common import ROOT

matplotlib.use("Agg")
REPORTS = ROOT / "reports"
COLS = ["run_id", "run_name", "feature_version", "hp_set", "mae", "rmse", "r2",
        "train_seconds", "energy_wh", "emissions_g_co2eq"]


def _grouped_bars(table: pd.DataFrame, metric: str, ylabel: str, ax):
    pivot = table.pivot(index="feature_version", columns="hp_set", values=metric)
    pivot.plot(kind="bar", ax=ax, rot=0)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3g", fontsize=8)
    ax.set(title=ylabel, xlabel="feature version", ylabel=ylabel)


def run(results: list[dict]) -> pd.DataFrame:
    REPORTS.mkdir(exist_ok=True)
    table = pd.DataFrame(results)[COLS].sort_values(["feature_version", "hp_set"])
    table.round(4).to_csv(REPORTS / "results.csv", index=False)
    shown = table.drop(columns=["run_id"]).round(4)
    (REPORTS / "results.md").write_text(
        shown.to_markdown(index=False) if _has_tabulate() else shown.to_string(index=False))

    # Quantitative: model metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _grouped_bars(table, "rmse", "RMSE (lb, lower is better)", axes[0])
    _grouped_bars(table, "mae", "MAE (lb, lower is better)", axes[1])
    _grouped_bars(table, "r2", "R² (higher is better)", axes[2])
    fig.suptitle("Model metrics: 2 feature versions × 2 hyperparameter sets")
    fig.tight_layout()
    fig.savefig(REPORTS / "metrics_comparison.png", dpi=150)
    mlflow.log_figure(fig, "comparison/metrics_comparison.png")
    plt.close(fig)

    # Carbon: emissions, energy and the accuracy/emissions trade-off
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    _grouped_bars(table, "emissions_g_co2eq", "Emissions (g CO₂eq)", axes[0])
    _grouped_bars(table, "train_seconds", "Train + predict time (s)", axes[1])
    for _, row in table.iterrows():
        axes[2].scatter(row["emissions_g_co2eq"], row["rmse"], s=80)
        axes[2].annotate(row["run_name"], (row["emissions_g_co2eq"], row["rmse"]),
                         fontsize=8, xytext=(5, 5), textcoords="offset points")
    axes[2].margins(x=0.3, y=0.15)
    axes[2].set(title="Accuracy vs emissions", xlabel="g CO₂eq", ylabel="RMSE (lb)")
    fig.suptitle("Carbon emissions per experiment (CodeCarbon)")
    fig.tight_layout()
    fig.savefig(REPORTS / "emissions_comparison.png", dpi=150)
    mlflow.log_figure(fig, "comparison/emissions_comparison.png")
    plt.close(fig)

    # Qualitative: the four predicted-vs-actual plots side by side
    fig, axes = plt.subplots(1, len(table), figsize=(5 * len(table), 5))
    with tempfile.TemporaryDirectory() as tmp:
        for ax, (_, row) in zip(axes, table.iterrows(), strict=True):
            local = mlflow.artifacts.download_artifacts(
                run_id=row["run_id"], artifact_path="plots/pred_vs_actual.png", dst_path=tmp)
            ax.imshow(plt.imread(local))
            ax.axis("off")
    fig.suptitle("Predicted vs actual, per experiment")
    fig.tight_layout()
    fig.savefig(REPORTS / "pred_vs_actual_grid.png", dpi=150)
    mlflow.log_figure(fig, "comparison/pred_vs_actual_grid.png")
    plt.close(fig)

    mlflow.log_artifact(str(REPORTS / "results.csv"), "comparison")
    print(f"[compare] results written to {REPORTS}")
    print(shown.round(3).to_string(index=False))
    return table


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False
