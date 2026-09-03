from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Sub_Division_IMD_2017.csv"


def test_rainfall_dataset_exists():
    assert RAW_FILE.exists(), f"Dataset not found: {RAW_FILE}"


def test_rainfall_dataset_has_expected_columns():
    df = pd.read_csv(RAW_FILE)

    assert len(df.columns) == 19, (
        f"Expected 19 columns, found {len(df.columns)}"
    )


def test_rainfall_dataset_is_not_empty():
    df = pd.read_csv(RAW_FILE)

    assert len(df) > 0, "Rainfall dataset is empty"