from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "environmental_risk.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "regional_risk_profile.csv"
)


def create_regional_risk_profile() -> Path:
    """Create historical environmental risk profiles by subdivision."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "subdivision",
        "year",
        "rainfall_mm",
        "rainfall_zscore",
        "is_dry",
        "dry_episode_length",
        "persistent_drought_signal",
        "environmental_risk_score",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    regional = (
        df.groupby("subdivision")
        .agg(
            years_observed=("year", "nunique"),
            observations=("year", "size"),
            average_rainfall_mm=("rainfall_mm", "mean"),
            average_risk_score=(
                "environmental_risk_score",
                "mean",
            ),
            maximum_risk_score=(
                "environmental_risk_score",
                "max",
            ),
            average_rainfall_zscore=(
                "rainfall_zscore",
                "mean",
            ),
            dry_months=("is_dry", "sum"),
            persistent_drought_months=(
                "persistent_drought_signal",
                "sum",
            ),
            maximum_dry_episode=(
                "dry_episode_length",
                "max",
            ),
        )
        .reset_index()
    )

    regional["dry_month_pct"] = (
        regional["dry_months"]
        / regional["observations"]
        * 100
    )

    regional["persistent_drought_pct"] = (
        regional["persistent_drought_months"]
        / regional["observations"]
        * 100
    )

    regional = regional.sort_values(
        "average_risk_score",
        ascending=False,
    ).reset_index(drop=True)

    regional["risk_rank"] = (
        regional.index + 1
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    regional.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return OUTPUT_FILE


if __name__ == "__main__":
    output = create_regional_risk_profile()

    print(
        f"Regional risk profile created: {output}"
    )