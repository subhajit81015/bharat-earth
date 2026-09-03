from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Sub_Division_IMD_2017.csv"

PROCESSED_FILE = (
    PROJECT_ROOT / "data" / "processed" / "rainfall_cleaned.csv"
)


def test_processed_dataset_exists():
    assert RAW_FILE.exists()
    assert PROCESSED_FILE.exists()


def test_processed_dataset_is_readable():
    df = pd.read_csv(PROCESSED_FILE)

    assert not df.empty
    assert len(df.columns) == 19