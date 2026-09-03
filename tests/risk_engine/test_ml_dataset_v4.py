from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v4.csv"
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
    "rainfall_lag_1m",
    "rainfall_lag_2m",
    "rainfall_lag_3m",
    "rainfall_prev_3m",
    "rainfall_prev_6m",
    "rainfall_prev_12m",
    "rainfall_trend_3m",
    "month_sin",
    "month_cos",
    "target_3m_severe_anomaly",
}


LEAKAGE_COLUMNS = {
    "persistent_drought_signal",
    "environmental_risk_score",
    "environmental_risk_level",
    "target_3m_stress",
}


def test_ml_dataset_v4_exists():
    assert INPUT_FILE.exists()


def test_ml_dataset_v4_has_expected_rows():
    df = pd.read_csv(INPUT_FILE)

    assert len(df) == 50133


def test_ml_dataset_v4_has_expected_columns():
    df = pd.read_csv(INPUT_FILE)

    assert set(df.columns) == EXPECTED_COLUMNS


def test_temporal_features_exist():
    df = pd.read_csv(INPUT_FILE)

    temporal_features = {
        "rainfall_lag_1m",
        "rainfall_lag_2m",
        "rainfall_lag_3m",
        "rainfall_prev_3m",
        "rainfall_prev_6m",
        "rainfall_prev_12m",
        "rainfall_trend_3m",
        "month_sin",
        "month_cos",
    }

    assert temporal_features.issubset(
        df.columns
    )


def test_no_leakage_columns():
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


def test_target_positive_count():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["target_3m_severe_anomaly"]
        .sum()
        == 2241
    )


def test_target_rate():
    df = pd.read_csv(INPUT_FILE)

    rate = (
        df["target_3m_severe_anomaly"]
        .mean()
    )

    assert 0.04 < rate < 0.05


def test_month_features_are_bounded():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["month_sin"]
        .between(-1, 1)
        .all()
    )

    assert (
        df["month_cos"]
        .between(-1, 1)
        .all()
    )