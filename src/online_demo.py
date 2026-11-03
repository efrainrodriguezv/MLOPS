"""Step 5 (optional) — serve features from the online store and score with the best model.

Materializes the offline tables into the SQLite online store, fetches feature vectors
for a few test athletes through the winning feature service, and predicts with the
best model from MLflow: the same feature definitions feed training and serving.
"""
from datetime import datetime, timedelta, timezone

import mlflow
import pandas as pd
from feast import FeatureStore

from src.common import ENTITY_KEY, ROOT


def run(store: FeatureStore, best: dict, cfg: dict, n: int = 5) -> pd.DataFrame:
    store.materialize(start_date=datetime(2000, 1, 1, tzinfo=timezone.utc),
                      end_date=datetime.now(timezone.utc) + timedelta(days=1))

    labels = pd.read_parquet(ROOT / cfg["data"]["processed_dir"] / "labels.parquet")
    sample = labels[labels["split"] == "test"].head(n)
    service_name = cfg["feature_store"]["versions"][best["feature_version"]]

    online = store.get_online_features(
        features=store.get_feature_service(service_name),
        entity_rows=[{ENTITY_KEY: int(i)} for i in sample[ENTITY_KEY]],
    ).to_df()

    model = mlflow.sklearn.load_model(f"runs:/{best['run_id']}/model")
    columns = list(model.feature_names_in_)
    online["predicted_total_lift"] = model.predict(online[columns])
    online = online.merge(sample[[ENTITY_KEY, "total_lift"]], on=ENTITY_KEY)
    print("[online] predictions from online features:")
    print(online[[ENTITY_KEY, "total_lift", "predicted_total_lift"]].round(1).to_string(index=False))
    return online
