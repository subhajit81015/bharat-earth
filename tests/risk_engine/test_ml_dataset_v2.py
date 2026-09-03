from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v2.csv"
)


NEW_FEATURES = {
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


def test_ml_dataset_v2_exists():
    assert INPUT_FILE.exists()


def test_ml_dataset_v2_has_expected_rows():
    df = pd.read_csv(INPUT_FILE)

    assert len(df) == 50133


def test_temporal_features_exist():
    df = pd.read_csv(INPUT_FILE)

    assert NEW_FEATURES.issubset(
        df.columns
    )


def test_month_cyclical_features_are_complete():
    df = pd.read_csv(INPUT_FILE)

    assert df["month_sin"].notna().all()
    assert df["month_cos"].notna().all()


def test_target_is_unchanged():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["target_3m_stress"].mean()
        == 0.007320527397123651
    )


def test_temporal_features_contain_expected_nulls():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["rainfall_lag_1m"].isna().sum()
        > 0
    )

    assert (
        df["rainfall_lag_2m"].isna().sum()
        > 0
    )

    assert (
        df["rainfall_lag_3m"].isna().sum()
        > 0
    )


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