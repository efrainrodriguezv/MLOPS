"""Feast definitions: one entity, two feature views, two feature services (= feature versions).

  athlete_features_v1 = athlete_basic
  athlete_features_v2 = athlete_basic + athlete_engineered
"""
from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureService, FeatureView, Field, FileSource
from feast.types import Float64, Int64

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
TTL = timedelta(days=3650)

athlete = Entity(name="athlete", join_keys=["athlete_id"],
                 description="One CrossFit athlete from athletes.csv")

basic_source = FileSource(
    name="athlete_basic_source",
    path=str(DATA_DIR / "athlete_basic.parquet"),
    timestamp_field="event_timestamp",
)
engineered_source = FileSource(
    name="athlete_engineered_source",
    path=str(DATA_DIR / "athlete_engineered.parquet"),
    timestamp_field="event_timestamp",
)

athlete_basic = FeatureView(
    name="athlete_basic",
    entities=[athlete],
    ttl=TTL,
    schema=[
        Field(name="age", dtype=Float64),
        Field(name="height", dtype=Float64),
        Field(name="weight", dtype=Float64),
        Field(name="gender_male", dtype=Int64),
    ],
    source=basic_source,
    tags={"version": "1"},
)

athlete_engineered = FeatureView(
    name="athlete_engineered",
    entities=[athlete],
    ttl=TTL,
    schema=[
        Field(name="bmi", dtype=Float64),
        Field(name="region_code", dtype=Int64),
        Field(name="howlong_ord", dtype=Int64),
        Field(name="eat_n", dtype=Int64),
        Field(name="background_n", dtype=Int64),
        Field(name="experience_n", dtype=Int64),
        Field(name="schedule_n", dtype=Int64),
        Field(name="exp_began_with_coach", dtype=Int64),
        Field(name="sched_multi_daily", dtype=Int64),
        Field(name="bg_college_sports", dtype=Int64),
    ],
    source=engineered_source,
    tags={"version": "2"},
)

athlete_features_v1 = FeatureService(
    name="athlete_features_v1",
    features=[athlete_basic],
    description="Feature version 1: body metrics only",
)
athlete_features_v2 = FeatureService(
    name="athlete_features_v2",
    features=[athlete_basic, athlete_engineered],
    description="Feature version 2: body metrics + engineered and survey features",
)

ALL_OBJECTS = [athlete, athlete_basic, athlete_engineered, athlete_features_v1, athlete_features_v2]
