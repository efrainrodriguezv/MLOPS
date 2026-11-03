"""Step 2 — register the entity, feature views and feature services in the Feast registry
(equivalent to running `feast apply` inside feature_repo/)."""
import importlib.util

from feast import FeatureStore

from src.common import ROOT, load_config


def _load_definitions(repo_path):
    spec = importlib.util.spec_from_file_location("features", repo_path / "features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ALL_OBJECTS


def run(cfg: dict) -> FeatureStore:
    repo_path = ROOT / cfg["feature_store"]["repo_path"]
    (repo_path / "data").mkdir(exist_ok=True)
    store = FeatureStore(repo_path=str(repo_path))
    store.apply(_load_definitions(repo_path))
    services = [s.name for s in store.list_feature_services()]
    print(f"[features] registered feature services: {services}")
    return store


if __name__ == "__main__":
    run(load_config())
