from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "rainfall_features.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "environmental_stress.csv"
)


def calculate_stress_score() -> Path:
    """Calculate an interpretable rainfall stress score from 0 to 100."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "subdivision",
        "year",
        "month",
        "rainfall_mm",
        "rainfall_anomaly_pct",
        "rainfall_deficit_mm",
        "rainfall_missing",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Missing observations should not automatically become
    # high environmental stress.
    valid_anomaly = df["rainfall_anomaly_pct"].fillna(0)

    # Convert negative anomaly into a 0-100 stress component.
    anomaly_stress = (
        (-valid_anomaly).clip(lower=0, upper=100)
    )

    # Missing data is tracked separately.
    data_quality_penalty = (
        df["rainfall_missing"] * 10
    )

    # Weighted interpretable score.
    df["environmental_stress_score"] = (
        anomaly_stress * 0.80
        + data_quality_penalty
    ).clip(
        lower=0,
        upper=100,
    )

    # Classification
    df["stress_level"] = pd.cut(
        df["environmental_stress_score"],
        bins=[
            -1,
            20,
            40,
            60,
            80,
            100,
        ],
        labels=[
            "LOW",
            "MODERATE",
            "ELEVATED",
            "HIGH",
            "SEVERE",
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
    output = calculate_stress_score()

    print(
        f"Environmental stress dataset created: {output}"
    )