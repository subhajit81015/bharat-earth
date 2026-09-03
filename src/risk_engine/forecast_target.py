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
    / "forecast_target.csv"
)


def create_forecast_target() -> Path:
    """Create a three-calendar-month-ahead stress target."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Model dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "subdivision",
        "year",
        "month",
        "date",
        "persistent_drought_signal",
    }

    missing_columns = required_columns - set(df.columns)

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

    # Future drought signals.
    future_1 = group[
        "persistent_drought_signal"
    ].shift(-1)

    future_2 = group[
        "persistent_drought_signal"
    ].shift(-2)

    future_3 = group[
        "persistent_drought_signal"
    ].shift(-3)

    # Target:
    # 1 = persistent drought occurs in any
    #     of the next three observations.
    # 0 = otherwise.
    df["target_3m_stress"] = (
        (future_1 == 1)
        | (future_2 == 1)
        | (future_3 == 1)
    ).astype(int)

    # Verify that the third future observation
    # is exactly three calendar months later.
    future_date_3 = group["date"].shift(-3)

    expected_future_date = (
        df["date"] + pd.DateOffset(months=3)
    )

    df["has_full_3m_horizon"] = (
        future_date_3 == expected_future_date
    )

    # Keep only records with a complete
    # three-calendar-month future horizon.
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

    return OUTPUT_FILE


if __name__ == "__main__":
    output = create_forecast_target()

    print(
        f"Forecast target dataset created: {output}"
    )