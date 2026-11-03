"""Step 1 — load athletes.csv, clean it, and write the Feast offline-store tables.

Outputs (data/processed/):
  athlete_basic.parquet       -> feature view `athlete_basic`       (used by v1 and v2)
  athlete_engineered.parquet  -> feature view `athlete_engineered`  (used by v2 only)
  labels.parquet              -> entity dataframe: athlete_id, event_timestamp, total_lift, split

Both feature versions are built for the SAME athletes, labels and train/test split,
so the only thing that differs between experiments is the feature set.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.common import ENTITY_KEY, ROOT, TIMESTAMP_COL, load_config

LIFT_COLS = ["deadlift", "candj", "snatch", "backsq"]
HOWLONG_ORDER = {
    "less than 6 months": 0,
    "6-12 months": 1,
    "1-2 years": 2,
    "2-4 years": 3,
    "4+ years": 4,
}


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, compression="infer", low_memory=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Course-provided cleaning, with two changes: the `v2['deadlift']` typo is fixed,
    and `athlete_id` is kept because it is the feature store's entity key."""
    data = df.copy()

    data = data.dropna(subset=["region", "age", "weight", "height", "howlong", "gender", "eat",
                               "train", "background", "experience", "schedule",
                               "deadlift", "candj", "snatch", "backsq"])
    data = data.drop(columns=["affiliate", "team", "name", "fran", "helen", "grace",
                              "filthy50", "fgonebad", "run400", "run5k", "pullups", "train"])

    data = data[data["weight"] < 1500]
    data = data[data["gender"] != "--"]
    data = data[data["age"] >= 18]
    data = data[(data["height"] < 96) & (data["height"] > 48)]

    data = data[((data["deadlift"] > 0) & (data["deadlift"] <= 1105))
                | ((data["gender"] == "Female") & (data["deadlift"] <= 636))]
    data = data[(data["candj"] > 0) & (data["candj"] <= 395)]
    data = data[(data["snatch"] > 0) & (data["snatch"] <= 496)]
    data = data[(data["backsq"] > 0) & (data["backsq"] <= 1069)]

    data = data.replace({"Decline to answer|": np.nan})
    data = data.dropna(subset=["background", "experience", "schedule", "howlong", "eat"])

    # Entity key must be present and unique for the feature store
    data = data.dropna(subset=[ENTITY_KEY]).drop_duplicates(subset=[ENTITY_KEY])
    data[ENTITY_KEY] = data[ENTITY_KEY].astype("int64")
    return data


def _count_answers(series: pd.Series) -> pd.Series:
    """Number of options ticked in a '|'-delimited multi-select survey answer."""
    return series.fillna("").str.split("|").apply(lambda parts: sum(1 for p in parts if p.strip()))


def _contains(series: pd.Series, phrase: str) -> pd.Series:
    return series.fillna("").str.lower().str.contains(phrase, regex=False).astype("int64")


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature version 1: raw body metrics only."""
    return pd.DataFrame({
        ENTITY_KEY: df[ENTITY_KEY],
        "age": df["age"].astype("float64"),
        "height": df["height"].astype("float64"),
        "weight": df["weight"].astype("float64"),
        "gender_male": (df["gender"] == "Male").astype("int64"),
    })


def build_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extra features added in version 2: derived body metric, region and survey signals."""
    howlong_first = df["howlong"].str.split("|").str[0].str.strip().str.lower()
    region_codes = {r: i for i, r in enumerate(sorted(df["region"].unique()))}
    return pd.DataFrame({
        ENTITY_KEY: df[ENTITY_KEY],
        "bmi": (703 * df["weight"] / df["height"] ** 2).astype("float64"),
        "region_code": df["region"].map(region_codes).astype("int64"),
        "howlong_ord": howlong_first.map(HOWLONG_ORDER).fillna(-1).astype("int64"),
        "eat_n": _count_answers(df["eat"]).astype("int64"),
        "background_n": _count_answers(df["background"]).astype("int64"),
        "experience_n": _count_answers(df["experience"]).astype("int64"),
        "schedule_n": _count_answers(df["schedule"]).astype("int64"),
        "exp_began_with_coach": _contains(df["experience"], "began crossfit with a coach"),
        "sched_multi_daily": _contains(df["schedule"], "multiple workouts in a day"),
        "bg_college_sports": _contains(df["background"], "college"),
    })


def run(cfg: dict) -> dict:
    data_cfg = cfg["data"]
    out_dir = ROOT / data_cfg["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp(data_cfg["feature_timestamp"], tz="UTC")

    raw = load_raw(ROOT / data_cfg["raw_path"])
    clean = clean_data(raw)
    clean["total_lift"] = clean[LIFT_COLS].sum(axis=1)

    basic = build_basic_features(clean).assign(**{TIMESTAMP_COL: ts})
    engineered = build_engineered_features(clean).assign(**{TIMESTAMP_COL: ts})

    train_ids, _ = train_test_split(clean[ENTITY_KEY], test_size=data_cfg["test_size"],
                                    random_state=data_cfg["seed"])
    labels = pd.DataFrame({
        ENTITY_KEY: clean[ENTITY_KEY],
        "total_lift": clean["total_lift"].astype("float64"),
        "split": np.where(clean[ENTITY_KEY].isin(set(train_ids)), "train", "test"),
    })
    # entity timestamp one day after the features, so point-in-time joins find them
    labels[TIMESTAMP_COL] = ts + pd.Timedelta(days=1)

    basic.to_parquet(out_dir / "athlete_basic.parquet", index=False)
    engineered.to_parquet(out_dir / "athlete_engineered.parquet", index=False)
    labels.to_parquet(out_dir / "labels.parquet", index=False)

    summary = {
        "raw_rows": len(raw),
        "clean_rows": len(clean),
        "train_rows": int((labels["split"] == "train").sum()),
        "test_rows": int((labels["split"] == "test").sum()),
    }
    print(f"[prepare] {summary}")
    return summary


if __name__ == "__main__":
    run(load_config())
