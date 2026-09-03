from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "model_ready_environmental.csv"
)


EXPECTED_COLUMNS = {
    "subdivision",
    "year",
    "month",
    "rainfall_mm",
    "date",
    "rainfall_3m",
    "rainfall_6m",
    "rainfall_12m",
    "historical_monthly_mean",
    "rainfall_anomaly",
    "rainfall_anomaly_pct",
    "rainfall_deficit_mm",
    "rainfall_missing",
    "rainfall_stress",
    "rainfall_zscore",
    "rainfall_condition",
    "dry_episode_length",
    "persistent_drought_signal",
    "environmental_risk_score",
    "environmental_risk_level",
    "season",
}


def test_model_dataset_exists():
    assert INPUT_FILE.exists()


def test_model_dataset_has_expected_rows():
    df = pd.read_csv(INPUT_FILE)

    assert len(df) == 50256


def test_model_dataset_has_expected_columns():
    df = pd.read_csv(INPUT_FILE)

    assert EXPECTED_COLUMNS.issubset(df.columns)


def test_model_dataset_has_unique_time_records():
    df = pd.read_csv(INPUT_FILE)

    duplicate_count = df.duplicated(
        subset=[
            "subdivision",
            "year",
            "month",
        ]
    ).sum()

    assert duplicate_count == 0


def test_seasons_are_valid():
    df = pd.read_csv(INPUT_FILE)

    valid_seasons = {
        "WINTER",
        "PRE_MONSOON",
        "MONSOON",
        "POST_MONSOON",
    }

    assert set(df["season"].dropna().unique()).issubset(
        valid_seasons
    )


def test_risk_scores_are_valid():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["environmental_risk_score"]
        .between(0, 100)
        .all()
    )


def test_drought_signal_is_binary():
    df = pd.read_csv(INPUT_FILE)

    assert set(
        df["persistent_drought_signal"]
        .dropna()
        .unique()
    ).issubset({0, 1})