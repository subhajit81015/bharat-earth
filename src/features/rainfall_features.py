from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "rainfall_long.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "rainfall_features.csv"
)


MONTH_ORDER = [
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


def create_rainfall_features() -> Path:
    """Create time-series rainfall features."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "subdivision",
        "year",
        "month",
        "rainfall_mm",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # Normalize data types
    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["rainfall_mm"] = pd.to_numeric(
        df["rainfall_mm"],
        errors="coerce",
    )

    df["month"] = df["month"].str.upper()

    # Convert month to chronological order
    df["month"] = pd.Categorical(
        df["month"],
        categories=MONTH_ORDER,
        ordered=True,
    )

    # Create a real time index
    month_number = df["month"].cat.codes + 1

    df["date"] = pd.to_datetime(
        {
            "year": df["year"],
            "month": month_number,
            "day": 1,
        },
        errors="coerce",
    )

    # Sort chronologically
    df = df.sort_values(
        ["subdivision", "date"]
    ).reset_index(drop=True)

    group = df.groupby(
        "subdivision",
        group_keys=False,
    )

    # ---------------------------------------------------------
    # Rolling rainfall
    # ---------------------------------------------------------

    df["rainfall_3m"] = group["rainfall_mm"].transform(
        lambda x: x.rolling(
            window=3,
            min_periods=1,
        ).sum()
    )

    df["rainfall_6m"] = group["rainfall_mm"].transform(
        lambda x: x.rolling(
            window=6,
            min_periods=1,
        ).sum()
    )

    df["rainfall_12m"] = group["rainfall_mm"].transform(
        lambda x: x.rolling(
            window=12,
            min_periods=1,
        ).sum()
    )

    # ---------------------------------------------------------
    # Historical monthly baseline
    # ---------------------------------------------------------

    monthly_mean = df.groupby(
        ["subdivision", "month"],
        observed=True,
    )["rainfall_mm"].transform("mean")

    df["historical_monthly_mean"] = monthly_mean

    # ---------------------------------------------------------
    # Rainfall anomaly
    # ---------------------------------------------------------

    df["rainfall_anomaly"] = (
        df["rainfall_mm"]
        - df["historical_monthly_mean"]
    )

    df["rainfall_anomaly_pct"] = (
        df["rainfall_anomaly"]
        / df["historical_monthly_mean"].replace(0, pd.NA)
        * 100
    )

    # ---------------------------------------------------------
    # Rainfall deficit
    # ---------------------------------------------------------

    df["rainfall_deficit_mm"] = (
        df["historical_monthly_mean"]
        - df["rainfall_mm"]
    ).clip(lower=0)

    # ---------------------------------------------------------
    # Missing-data indicator
    # ---------------------------------------------------------

    df["rainfall_missing"] = (
        df["rainfall_mm"].isna().astype(int)
    )

    # ---------------------------------------------------------
    # Simple environmental stress indicator
    # ---------------------------------------------------------

    df["rainfall_stress"] = (
        df["rainfall_anomaly_pct"] <= -25
    ).astype(int)

    # Save
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return OUTPUT_FILE


if __name__ == "__main__":
    output = create_rainfall_features()

    print(
        f"Feature dataset created: {output}"
    )