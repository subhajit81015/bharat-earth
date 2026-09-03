from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v3.csv"
)


EXPECTED_COLUMNS = {
    "subdivision",
    "year",
    "month",
    "season",
    "rainfall_mm",
    "rainfall_3m",
    "rainfall_6m",
    "rainfall_12m",
    "historical_monthly_mean",
    "rainfall_anomaly",
    "rainfall_anomaly_pct",
    "rainfall_deficit_mm",
    "rainfall_missing",
    "rainfall_zscore",
    "target_3m_severe_anomaly",
}


LEAKAGE_COLUMNS = {
    "persistent_drought_signal",
    "environmental_risk_score",
    "environmental_risk_level",
    "target_3m_stress",
}


def test_ml_dataset_v3_exists():
    assert INPUT_FILE.exists()


def test_ml_dataset_v3_has_expected_rows():
    df = pd.read_csv(INPUT_FILE)

    assert len(df) == 50133


def test_ml_dataset_v3_has_expected_columns():
    df = pd.read_csv(INPUT_FILE)

    assert set(df.columns) == EXPECTED_COLUMNS


def test_no_target_leakage_columns():
    df = pd.read_csv(INPUT_FILE)

    assert (
        LEAKAGE_COLUMNS
        & set(df.columns)
    ) == set()


def test_target_is_binary():
    df = pd.read_csv(INPUT_FILE)

    assert set(
        df["target_3m_severe_anomaly"]
        .dropna()
        .unique()
    ).issubset({0, 1})


def test_target_has_expected_positive_count():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["target_3m_severe_anomaly"]
        .sum()
        == 2241
    )


def test_target_rate_is_reasonable():
    df = pd.read_csv(INPUT_FILE)

    target_rate = (
        df["target_3m_severe_anomaly"]
        .mean()
    )

    assert 0.04 < target_rate < 0.05


def test_month_values_are_valid():
    df = pd.read_csv(INPUT_FILE)

    valid_months = {
        "JAN",
        "FEB",
        "MAR",
        "APR",
        "MAY",
        "JUN",
        "JUL",
        "AUG",
        "SEP",
        "OCT",
        "NOV",
        "DEC",
    }

    assert set(
        df["month"].dropna().unique()
    ).issubset(valid_months)


def test_season_values_are_valid():
    df = pd.read_csv(INPUT_FILE)

    valid_seasons = {
        "WINTER",
        "PRE_MONSOON",
        "MONSOON",
        "POST_MONSOON",
    }

    assert set(
        df["season"].dropna().unique()
    ).issubset(valid_seasons)