"""Create a small synthetic athletes.csv with the real file's columns, for smoke-testing
the pipeline without the course data:

    python tests/make_synthetic_athletes.py data/raw/athletes_synthetic.csv
    # then point config.yaml -> data.raw_path at that file
"""
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
N = 6000
HOWLONG = ["Less than 6 months|", "6-12 months|", "1-2 years|", "2-4 years|", "4+ years|",
           "Decline to answer|"]
EAT = ["I eat whatever is convenient|", "I eat quality foods but don't measure the amount|",
       "I eat 1-3 full cheat meals per week|", "I weigh and measure my food|"]
BACKGROUND = ["I played youth or high school level sports|", "I regularly play recreational sports|",
              "I played college sports|", "I have no athletic background besides CrossFit|"]
EXPERIENCE = ["I began CrossFit with a coach (e.g. at an affiliate)|",
              "I began CrossFit by trying it alone (e.g. from the website)|",
              "I have attended one or more specialty courses|",
              "I have completed the CrossFit Level 1 certificate course|"]
SCHEDULE = ["I do multiple workouts in a day 1x a week|", "I usually only do 1 workout a day|",
            "I typically rest 4 or more days per month|", "I strictly schedule my rest days|"]


def multi(options):
    k = rng.integers(1, 3)
    return "".join(rng.choice(options, size=k, replace=False))


male = rng.random(N) < 0.65
height = np.where(male, rng.normal(70, 3, N), rng.normal(65, 3, N))
weight = np.where(male, rng.normal(190, 25, N), rng.normal(145, 20, N))
howlong_idx = rng.integers(0, 5, N)
strength = (weight * 1.2 + howlong_idx * 25 + male * 120) * rng.normal(1, 0.12, N)

df = pd.DataFrame({
    "athlete_id": np.arange(100000, 100000 + N),
    "name": "x", "region": rng.choice(["North East", "South West", "Europe", "Canada West"], N),
    "team": None, "affiliate": None,
    "gender": np.where(male, "Male", "Female"),
    "age": rng.integers(18, 60, N).astype(float), "height": height.round(), "weight": weight.round(),
    "fran": np.nan, "helen": np.nan, "grace": np.nan, "filthy50": np.nan, "fgonebad": np.nan,
    "run400": np.nan, "run5k": np.nan, "pullups": np.nan,
    "deadlift": (strength * 1.0).round(), "candj": (strength * 0.45).clip(1, 390).round(),
    "snatch": (strength * 0.35).clip(1, 490).round(), "backsq": (strength * 0.85).clip(1, 1060).round(),
    "eat": [multi(EAT) for _ in range(N)], "train": "I workout mostly at a CrossFit Affiliate|",
    "background": [multi(BACKGROUND) for _ in range(N)],
    "experience": [multi(EXPERIENCE) for _ in range(N)],
    "schedule": [multi(SCHEDULE) for _ in range(N)],
    "howlong": [HOWLONG[i] for i in howlong_idx],
})
df.to_csv(sys.argv[1] if len(sys.argv) > 1 else "athletes_synthetic.csv", index=False)
print(f"wrote {len(df)} synthetic rows")
