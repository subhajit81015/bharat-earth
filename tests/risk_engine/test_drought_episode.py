from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "drought_episodes.csv"
)


def test_drought_episode_dataset_exists():
    assert INPUT_FILE.exists()


def test_drought_episode_columns_exist():
    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "subdivision",
        "year",
        "month",
        "date",
        "rainfall_zscore",
        "rainfall_condition",
        "is_dry",
        "dry_episode_length",
        "persistent_drought_signal",
    }

    assert required_columns.issubset(df.columns)


def test_persistent_signal_is_binary():
    df = pd.read_csv(INPUT_FILE)

    assert set(
        df["persistent_drought_signal"].dropna().unique()
    ).issubset({0, 1})


def test_persistent_signal_requires_three_months():
    df = pd.read_csv(INPUT_FILE)

    signal = df[
        df["persistent_drought_signal"] == 1
    ]

    assert (
        signal["dry_episode_length"] >= 3
    ).all()
    