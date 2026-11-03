"""End-to-end ML pipeline, tracked as one parent MLflow run with a nested run per experiment.

    prepare data -> register features (Feast) -> 4 experiments -> compare -> online demo

Usage:
    python -m src.pipeline                 # full pipeline
    python -m src.pipeline --skip-online   # without the online-store demo
"""
import argparse

import mlflow
import pandas as pd

from src import compare, online_demo, prepare_data, register_features, train
from src.common import ROOT, load_config, setup_mlflow


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--skip-online", action="store_true", help="skip the online-store demo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_mlflow(cfg)

    with mlflow.start_run(run_name="pipeline"):
        mlflow.log_artifact(args.config)
        summary = prepare_data.run(cfg)
        mlflow.log_params({f"data_{k}": v for k, v in summary.items()})

        store = register_features.run(cfg)
        labels = pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / "labels.parquet")

        results = []
        for version, service_name in cfg["feature_store"]["versions"].items():
            for hp_name, hparams in cfg["hyperparameters"].items():
                results.append(train.run_experiment(store, labels, version, service_name,
                                                    hp_name, hparams, cfg))

        compare.run(results)
        best = min(results, key=lambda r: r["rmse"])
        mlflow.set_tag("best_run", best["run_name"])
        print(f"[pipeline] best experiment: {best['run_name']} (RMSE {best['rmse']:.1f})")

        if not args.skip_online:
            online_demo.run(store, best, cfg)

    print("[pipeline] done — open the UI with: "
          "mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050")


if __name__ == "__main__":
    main()
