from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "forecast_target.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset.csv"
)


FEATURE_COLUMNS = [
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
    "target_3m_stress",
]


def create_ml_dataset() -> Path:
    """Create a leakage-safe dataset for forecasting."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Forecast target dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    missing_columns = (
        set(FEATURE_COLUMNS) - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    result = df[FEATURE_COLUMNS].copy()

    result = result.sort_values(
        ["subdivision", "year", "month"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return OUTPUT_FILE


if __name__ == "__main__":
    output = create_ml_dataset()

    print(
        f"ML dataset created: {output}"
    )