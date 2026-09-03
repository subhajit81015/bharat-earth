from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "severe_anomaly_target.csv"
)

TEMPORAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v2.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v4.csv"
)


TARGET = "target_3m_severe_anomaly"


BASE_FEATURES = [
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
]


TEMPORAL_FEATURES = [
    "rainfall_lag_1m",
    "rainfall_lag_2m",
    "rainfall_lag_3m",
    "rainfall_prev_3m",
    "rainfall_prev_6m",
    "rainfall_prev_12m",
    "rainfall_trend_3m",
    "month_sin",
    "month_cos",
]


def create_ml_dataset_v4() -> Path:

    if not TARGET_FILE.exists():
        raise FileNotFoundError(
            f"Target dataset not found: {TARGET_FILE}"
        )

    if not TEMPORAL_FILE.exists():
        raise FileNotFoundError(
            f"Temporal dataset not found: {TEMPORAL_FILE}"
        )

    target_df = pd.read_csv(TARGET_FILE)

    temporal_df = pd.read_csv(TEMPORAL_FILE)

    required_target = set(
        BASE_FEATURES + [TARGET]
    )

    missing_target = (
        required_target
        - set(target_df.columns)
    )

    if missing_target:
        raise ValueError(
            f"Missing target columns: "
            f"{sorted(missing_target)}"
        )

    missing_temporal = (
        set(TEMPORAL_FEATURES)
        - set(temporal_df.columns)
    )

    if missing_temporal:
        raise ValueError(
            f"Missing temporal columns: "
            f"{sorted(missing_temporal)}"
        )

    keys = [
        "subdivision",
        "year",
        "month",
    ]

    temporal_subset = temporal_df[
        keys + TEMPORAL_FEATURES
    ].copy()

    result = target_df[
        BASE_FEATURES + [TARGET]
    ].merge(
        temporal_subset,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    result = result.sort_values(
        ["subdivision", "year", "month"]
    ).reset_index(drop=True)

    leakage_columns = {
        "persistent_drought_signal",
        "environmental_risk_score",
        "environmental_risk_level",
        "target_3m_stress",
    }

    present_leakage = (
        leakage_columns
        & set(result.columns)
    )

    if present_leakage:
        raise ValueError(
            "Leakage columns found: "
            f"{sorted(present_leakage)}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"ML V4 dataset created: {OUTPUT_FILE}"
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
        result[TARGET].value_counts()
    )

    print(
        "\nTARGET RATE:",
        result[TARGET].mean(),
    )

    print(
        "\nNEW TEMPORAL FEATURES:"
    )

    print(
        TEMPORAL_FEATURES
    )

    print(
        "\nLEAKAGE COLUMNS:"
    )

    print(
        sorted(
            leakage_columns
            & set(result.columns)
        )
    )

    return OUTPUT_FILE


if __name__ == "__main__":
    create_ml_dataset_v4()