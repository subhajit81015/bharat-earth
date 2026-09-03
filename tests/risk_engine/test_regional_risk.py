from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "regional_risk_profile.csv"
)


def test_regional_risk_dataset_exists():
    assert INPUT_FILE.exists()


def test_regional_risk_columns_exist():
    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "subdivision",
        "years_observed",
        "observations",
        "average_rainfall_mm",
        "average_risk_score",
        "maximum_risk_score",
        "average_rainfall_zscore",
        "dry_months",
        "persistent_drought_months",
        "maximum_dry_episode",
        "dry_month_pct",
        "persistent_drought_pct",
        "risk_rank",
    }

    assert required_columns.issubset(df.columns)


def test_regional_risk_has_expected_regions():
    df = pd.read_csv(INPUT_FILE)

    assert len(df) == 36
    assert df["subdivision"].nunique() == 36


def test_risk_scores_are_valid():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["average_risk_score"]
        .between(0, 100)
        .all()
    )

    assert (
        df["maximum_risk_score"]
        .between(0, 100)
        .all()
    )


def test_risk_rank_is_unique():
    df = pd.read_csv(INPUT_FILE)

    assert df["risk_rank"].is_unique

    assert set(df["risk_rank"]) == set(
        range(1, len(df) + 1)
    )


def test_dry_percentages_are_valid():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["dry_month_pct"]
        .between(0, 100)
        .all()
    )

    assert (
        df["persistent_drought_pct"]
        .between(0, 100)
        .all()
    )


def test_persistent_drought_cannot_exceed_dry_months():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["persistent_drought_months"]
        <= df["dry_months"]
    ).all()


def test_maximum_dry_episode_is_valid():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["maximum_dry_episode"] >= 0
    ).all()