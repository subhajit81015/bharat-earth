from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "severe_anomaly_target.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v3.csv"
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
    "target_3m_severe_anomaly",
]


LEAKAGE_COLUMNS = {
    "persistent_drought_signal",
    "environmental_risk_score",
    "environmental_risk_level",
    "target_3m_stress",
}


def create_ml_dataset_v3() -> Path:
    """Create a leakage-safe dataset for severe anomaly forecasting."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
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

    present_leakage_columns = (
        LEAKAGE_COLUMNS
        & set(result.columns)
    )

    if present_leakage_columns:
        raise ValueError(
            "Leakage columns found: "
            f"{sorted(present_leakage_columns)}"
        )

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

    print(
        f"ML V3 dataset created: {OUTPUT_FILE}"
    )

    print(
        "SHAPE:",
        result.shape,
    )

    print(
        "COLUMNS:",
        len(result.columns),
    )

    print(
        "\nTARGET DISTRIBUTION:"
    )

    print(
        result[
            "target_3m_severe_anomaly"
        ].value_counts()
    )

    print(
        "\nTARGET RATE:",
        result[
            "target_3m_severe_anomaly"
        ].mean(),
    )

    print(
        "\nLEAKAGE COLUMNS PRESENT:"
    )

    print(
        sorted(
            LEAKAGE_COLUMNS
            & set(result.columns)
        )
    )

    return OUTPUT_FILE


if __name__ == "__main__":
    create_ml_dataset_v3()