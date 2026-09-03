from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "features" / "standardized_rainfall.csv"

OUTPUT_FILE = PROJECT_ROOT / "data" / "features" / "drought_episodes.csv"


DRY_LEVELS = {
    "DRY",
    "SEVERE_DRY",
    "EXTREME_DRY",
}


MONTH_ORDER = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def detect_drought_episodes() -> Path:
    """Detect persistent consecutive rainfall-stress episodes."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "subdivision",
        "year",
        "month",
        "rainfall_mm",
        "rainfall_zscore",
        "rainfall_condition",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    # Normalize month names.
    df["month"] = df["month"].str.upper().str.strip()

    df["month_number"] = df["month"].map(MONTH_ORDER)

    if df["month_number"].isna().any():
        invalid_months = sorted(
            df.loc[
                df["month_number"].isna(),
                "month",
            ]
            .dropna()
            .unique()
        )

        raise ValueError(f"Invalid month values: {invalid_months}")

    # Create calendar date.
    df["date"] = pd.to_datetime(
        {
            "year": pd.to_numeric(
                df["year"],
                errors="coerce",
            ),
            "month": df["month_number"],
            "day": 1,
        },
        errors="coerce",
    )

    if df["date"].isna().any():
        raise ValueError("Invalid year/month values detected.")

    # Sort chronologically.
    df = df.sort_values(["subdivision", "date"]).reset_index(drop=True)

    # Identify dry months.
    df["is_dry"] = (df["rainfall_condition"].isin(DRY_LEVELS)).astype(int)

    # Create a monthly period index.
    df["month_index"] = df["date"].dt.year * 12 + df["date"].dt.month

    # Difference between consecutive observations.
    df["previous_month_index"] = df.groupby("subdivision")["month_index"].shift(1)

    df["is_consecutive"] = df["month_index"] == df["previous_month_index"] + 1

    # Start a new episode when:
    # 1. rainfall condition changes,
    # 2. months are not consecutive,
    # 3. or a subdivision changes.
    previous_dry = df.groupby("subdivision")["is_dry"].shift(1).fillna(0)

    new_episode = (df["is_dry"] != previous_dry) | (~df["is_consecutive"])

    df["episode_group"] = new_episode.astype(int).groupby(df["subdivision"]).cumsum()

    df["episode_id"] = df["subdivision"].astype(str) + "_" + df["episode_group"].astype(str)

    # Count observations in each episode.
    df["dry_episode_length"] = df.groupby("episode_id")["is_dry"].transform("sum")

    # Persistent drought signal:
    # at least 3 consecutive dry months.
    df["persistent_drought_signal"] = (
        (df["is_dry"] == 1) & (df["dry_episode_length"] >= 3)
    ).astype(int)

    output_columns = [
        "subdivision",
        "year",
        "month",
        "date",
        "rainfall_mm",
        "rainfall_zscore",
        "rainfall_condition",
        "is_dry",
        "dry_episode_length",
        "persistent_drought_signal",
    ]

    result = df[output_columns]

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
    output = detect_drought_episodes()

    print(f"Drought episode dataset created: {output}")
