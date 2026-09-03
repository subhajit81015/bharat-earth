from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "environmental_stress.csv"
)


def test_environmental_stress_dataset_exists():
    assert OUTPUT_FILE.exists()


def test_stress_score_range():
    df = pd.read_csv(OUTPUT_FILE)

    assert df["environmental_stress_score"].between(
        0,
        100,
    ).all()


def test_stress_levels_are_valid():
    df = pd.read_csv(OUTPUT_FILE)

    valid_levels = {
        "LOW",
        "MODERATE",
        "ELEVATED",
        "HIGH",
        "SEVERE",
    }

    observed_levels = set(
        df["stress_level"].dropna().unique()
    )

    assert observed_levels.issubset(valid_levels)