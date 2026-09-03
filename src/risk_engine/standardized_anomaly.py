from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OBSERVATION_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "rainfall_features.csv"
)

BASELINE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "rainfall_baseline.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "standardized_rainfall.csv"
)


def create_standardized_anomaly() -> Path:
    """Calculate standardized rainfall anomalies."""

    if not OBSERVATION_FILE.exists():
        raise FileNotFoundError(
            f"Observation dataset not found: {OBSERVATION_FILE}"
        )

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"Baseline dataset not found: {BASELINE_FILE}"
        )

    observations = pd.read_csv(OBSERVATION_FILE)
    baseline = pd.read_csv(BASELINE_FILE)

    required_observation_columns = {
        "subdivision",
        "year",
        "month",
        "rainfall_mm",
    }

    required_baseline_columns = {
        "subdivision",
        "month",
        "historical_mean_mm",
        "historical_std_mm",
    }

    missing_observation = (
        required_observation_columns
        - set(observations.columns)
    )

    missing_baseline = (
        required_baseline_columns
        - set(baseline.columns)
    )

    if missing_observation:
        raise ValueError(
            "Missing observation columns: "
            f"{sorted(missing_observation)}"
        )

    if missing_baseline:
        raise ValueError(
            "Missing baseline columns: "
            f"{sorted(missing_baseline)}"
        )

    # Join each observation with its
    # subdivision-month historical baseline.
    df = observations.merge(
        baseline,
        on=["subdivision", "month"],
        how="left",
        validate="many_to_one",
    )

    # Calculate standardized anomaly.
    df["rainfall_zscore"] = (
        df["rainfall_mm"]
        - df["historical_mean_mm"]
    ) / df["historical_std_mm"].replace(0, pd.NA)

    # Classify rainfall conditions.
    df["rainfall_condition"] = pd.cut(
        df["rainfall_zscore"],
        bins=[
            float("-inf"),
            -2.0,
            -1.5,
            -1.0,
            0.0,
            float("inf"),
        ],
        labels=[
            "EXTREME_DRY",
            "SEVERE_DRY",
            "DRY",
            "BELOW_NORMAL",
            "NORMAL_OR_WET",
        ],
    )

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
    output = create_standardized_anomaly()

    print(
        f"Standardized rainfall dataset created: {output}"
    )