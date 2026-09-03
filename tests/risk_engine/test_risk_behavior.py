from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "environmental_risk.csv"
)


def test_risk_score_has_reasonable_distribution():
    df = pd.read_csv(INPUT_FILE)

    assert df["environmental_risk_score"].mean() > 0
    assert df["environmental_risk_score"].std() > 0


def test_risk_score_never_exceeds_100():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["environmental_risk_score"] <= 100
    ).all()


def test_persistent_drought_has_positive_risk():
    df = pd.read_csv(INPUT_FILE)

    persistent = df[
        df["persistent_drought_signal"] == 1
    ]

    if not persistent.empty:
        assert (
            persistent["environmental_risk_score"] > 0
        ).all()


def test_risk_levels_match_score_ranges():
    df = pd.read_csv(INPUT_FILE)

    low = df[
        df["environmental_risk_level"] == "LOW"
    ]

    if not low.empty:
        assert (
            low["environmental_risk_score"] <= 20
        ).all()

    critical = df[
        df["environmental_risk_level"] == "CRITICAL"
    ]

    if not critical.empty:
        assert (
            critical["environmental_risk_score"] >= 80
        ).all()