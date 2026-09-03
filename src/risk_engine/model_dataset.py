from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAINFALL_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "rainfall_features.csv"
)

ANOMALY_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "standardized_rainfall.csv"
)

DROUGHT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "drought_episodes.csv"
)

RISK_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "environmental_risk.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "model_ready_environmental.csv"
)


SEASON_MAP = {
    "JAN": "WINTER",
    "FEB": "WINTER",
    "MAR": "PRE_MONSOON",
    "APR": "PRE_MONSOON",
    "MAY": "PRE_MONSOON",
    "JUN": "MONSOON",
    "JUL": "MONSOON",
    "AUG": "MONSOON",
    "SEP": "MONSOON",
    "OCT": "POST_MONSOON",
    "NOV": "POST_MONSOON",
    "DEC": "POST_MONSOON",
}


def create_model_dataset() -> Path:
    """Combine environmental features into one model-ready dataset."""

    for file_path in [
        RAINFALL_FILE,
        ANOMALY_FILE,
        DROUGHT_FILE,
        RISK_FILE,
    ]:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {file_path}"
            )

    rainfall = pd.read_csv(RAINFALL_FILE)

    anomaly = pd.read_csv(
        ANOMALY_FILE,
        usecols=[
            "subdivision",
            "year",
            "month",
            "rainfall_zscore",
            "rainfall_condition",
        ],
    )

    drought = pd.read_csv(
        DROUGHT_FILE,
        usecols=[
            "subdivision",
            "year",
            "month",
            "dry_episode_length",
            "persistent_drought_signal",
        ],
    )

    risk = pd.read_csv(
        RISK_FILE,
        usecols=[
            "subdivision",
            "year",
            "month",
            "environmental_risk_score",
            "environmental_risk_level",
        ],
    )

    # Join anomaly information.
    df = rainfall.merge(
        anomaly,
        on=[
            "subdivision",
            "year",
            "month",
        ],
        how="left",
        validate="one_to_one",
    )

    # Join drought information.
    df = df.merge(
        drought,
        on=[
            "subdivision",
            "year",
            "month",
        ],
        how="left",
        validate="one_to_one",
    )

    # Join environmental risk.
    df = df.merge(
        risk,
        on=[
            "subdivision",
            "year",
            "month",
        ],
        how="left",
        validate="one_to_one",
    )

    # Add season.
    df["season"] = df["month"].map(SEASON_MAP)

    if df["season"].isna().any():
        raise ValueError(
            "Some rows have invalid month values."
        )

    # Sort chronologically.
    df = df.sort_values(
        ["subdivision", "year", "month"]
    ).reset_index(drop=True)

    # Remove duplicate records if present.
    df = df.drop_duplicates(
        subset=[
            "subdivision",
            "year",
            "month",
        ]
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
    output = create_model_dataset()

    print(
        f"Model-ready dataset created: {output}"
    )