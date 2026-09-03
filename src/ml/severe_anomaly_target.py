from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "model_ready_environmental.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "severe_anomaly_target.csv"
)


SEVERE_THRESHOLD = -1.5


def create_severe_anomaly_target() -> Path:
    """Create a three-month severe rainfall anomaly target."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "subdivision",
        "year",
        "month",
        "date",
        "rainfall_zscore",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    if df["date"].isna().any():
        raise ValueError(
            "Invalid dates found in dataset."
        )

    df = df.sort_values(
        ["subdivision", "date"]
    ).reset_index(drop=True)

    group = df.groupby(
        "subdivision",
        group_keys=False,
    )

    future_1 = group[
        "rainfall_zscore"
    ].shift(-1)

    future_2 = group[
        "rainfall_zscore"
    ].shift(-2)

    future_3 = group[
        "rainfall_zscore"
    ].shift(-3)

    df["target_3m_severe_anomaly"] = (
        (future_1 <= SEVERE_THRESHOLD)
        | (future_2 <= SEVERE_THRESHOLD)
        | (future_3 <= SEVERE_THRESHOLD)
    ).astype(int)

    future_date_3 = group["date"].shift(-3)

    expected_future_date = (
        df["date"]
        + pd.DateOffset(months=3)
    )

    df["has_full_3m_horizon"] = (
        future_date_3
        == expected_future_date
    )

    result = df[
        df["has_full_3m_horizon"]
    ].copy()

    result = result.drop(
        columns=["has_full_3m_horizon"]
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
        f"Severe anomaly target created: {OUTPUT_FILE}"
    )

    print(
        "SHAPE:",
        result.shape,
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

    return OUTPUT_FILE


if __name__ == "__main__":
    create_severe_anomaly_target()