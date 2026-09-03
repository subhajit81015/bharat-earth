from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "environmental_risk.csv"
)


def test_environmental_risk_dataset_exists():
    assert INPUT_FILE.exists()


def test_environmental_risk_score_is_valid():
    df = pd.read_csv(INPUT_FILE)

    assert df["environmental_risk_score"].notna().all()

    assert df["environmental_risk_score"].between(
        0,
        100,
    ).all()


def test_environmental_risk_levels_are_valid():
    df = pd.read_csv(INPUT_FILE)

    valid_levels = {
        "LOW",
        "MODERATE",
        "ELEVATED",
        "HIGH",
        "CRITICAL",
    }

    observed_levels = set(
        df["environmental_risk_level"]
        .dropna()
        .unique()
    )

    assert observed_levels.issubset(valid_levels)


def test_critical_risk_has_high_score():
    df = pd.read_csv(INPUT_FILE)

    critical = df[
        df["environmental_risk_level"] == "CRITICAL"
    ]

    if not critical.empty:
        assert (
            critical["environmental_risk_score"] >= 80
        ).all()