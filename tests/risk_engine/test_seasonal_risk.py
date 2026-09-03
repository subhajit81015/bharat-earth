from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "seasonal_risk_profile.csv"
)


EXPECTED_SEASONS = {
    "WINTER",
    "PRE_MONSOON",
    "MONSOON",
    "POST_MONSOON",
}


def test_seasonal_dataset_exists():
    assert INPUT_FILE.exists()


def test_seasonal_dataset_has_expected_shape():
    df = pd.read_csv(INPUT_FILE)

    assert len(df) == 144
    assert df["subdivision"].nunique() == 36
    assert df["season"].nunique() == 4


def test_expected_seasons_exist():
    df = pd.read_csv(INPUT_FILE)

    assert set(df["season"].unique()) == EXPECTED_SEASONS


def test_each_subdivision_has_four_seasons():
    df = pd.read_csv(INPUT_FILE)

    season_counts = (
        df.groupby("subdivision")["season"]
        .nunique()
    )

    assert (season_counts == 4).all()


def test_dry_percentage_is_valid():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["dry_month_pct"]
        .between(0, 100)
        .all()
    )


def test_rainfall_values_are_non_negative():
    df = pd.read_csv(INPUT_FILE)

    assert (
        df["average_rainfall_mm"] >= 0
    ).all()

    assert (
        df["rainfall_std_mm"] >= 0
    ).all()