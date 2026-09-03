from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "standardized_rainfall.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "seasonal_risk_profile.csv"
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


def create_seasonal_risk_profile() -> Path:
    """Create subdivision-season rainfall risk profiles."""

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
        "rainfall_zscore",
        "rainfall_condition",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    df["month"] = (
        df["month"]
        .str.upper()
        .str.strip()
    )

    df["season"] = df["month"].map(SEASON_MAP)

    if df["season"].isna().any():
        invalid_months = sorted(
            df.loc[
                df["season"].isna(),
                "month",
            ]
            .dropna()
            .unique()
        )

        raise ValueError(
            f"Invalid month values: {invalid_months}"
        )

    df["is_dry"] = (
        df["rainfall_condition"].isin(
            [
                "DRY",
                "SEVERE_DRY",
                "EXTREME_DRY",
            ]
        )
    ).astype(int)

    seasonal = (
        df.groupby(
            ["subdivision", "season"]
        )
        .agg(
            years_observed=(
                "year",
                "nunique",
            ),
            observations=(
                "year",
                "size",
            ),
            average_rainfall_mm=(
                "rainfall_mm",
                "mean",
            ),
            rainfall_std_mm=(
                "rainfall_mm",
                "std",
            ),
            average_zscore=(
                "rainfall_zscore",
                "mean",
            ),
            minimum_zscore=(
                "rainfall_zscore",
                "min",
            ),
            dry_months=(
                "is_dry",
                "sum",
            ),
        )
        .reset_index()
    )

    seasonal["dry_month_pct"] = (
        seasonal["dry_months"]
        / seasonal["observations"]
        * 100
    )

    seasonal = seasonal.sort_values(
        [
            "subdivision",
            "season",
        ]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    seasonal.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return OUTPUT_FILE


if __name__ == "__main__":
    output = create_seasonal_risk_profile()

    print(
        f"Seasonal risk profile created: {output}"
    )