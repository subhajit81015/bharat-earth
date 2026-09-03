from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Sub_Division_IMD_2017.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_FILE = PROCESSED_DIR / "rainfall_cleaned.csv"


RAINFALL_COLUMNS = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
    "annual",
    "jf",
    "mam",
    "jjas",
    "ond",
]


def process_rainfall_data() -> Path:
    """Clean and standardize the rainfall dataset."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE)

    # Remove completely empty rows.
    df = df.dropna(how="all")

    # Standardize column names.
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Convert rainfall measurements to numeric.
    for column in RAINFALL_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Remove exact duplicate records.
    df = df.drop_duplicates()

    # Save processed dataset.
    df.to_csv(OUTPUT_FILE, index=False)

    return OUTPUT_FILE


if __name__ == "__main__":
    output = process_rainfall_data()
    print(f"Processed dataset: {output}")