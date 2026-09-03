from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rainfall_cleaned.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "features"

OUTPUT_FILE = OUTPUT_DIR / "rainfall_long.csv"


MONTH_COLUMNS = [
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
]


def transform_rainfall() -> Path:
    """Convert monthly rainfall data from wide to long format."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Processed rainfall dataset not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "subdivision",
        "year",
        *MONTH_COLUMNS,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Convert wide format to long format
    long_df = df.melt(
        id_vars=["subdivision", "year"],
        value_vars=MONTH_COLUMNS,
        var_name="month",
        value_name="rainfall_mm",
    )

    # Convert year to numeric
    long_df["year"] = pd.to_numeric(
        long_df["year"],
        errors="coerce",
    )

    # Convert rainfall to numeric
    long_df["rainfall_mm"] = pd.to_numeric(
        long_df["rainfall_mm"],
        errors="coerce",
    )

    # Normalize month names
    long_df["month"] = long_df["month"].str.upper()

    # Correct chronological month order
    month_order = [
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
    ]

    long_df["month"] = pd.Categorical(
        long_df["month"],
        categories=month_order,
        ordered=True,
    )

    # Sort dataset
    long_df = long_df.sort_values(
        ["subdivision", "year", "month"]
    ).reset_index(drop=True)

    # Save transformed dataset
    long_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return OUTPUT_FILE


if __name__ == "__main__":
    output = transform_rainfall()

    print(
        f"Transformed dataset created: {output}"
    )