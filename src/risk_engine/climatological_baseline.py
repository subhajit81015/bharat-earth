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
    / "rainfall_baseline.csv"
)


def create_climatological_baseline() -> Path:
    """Create subdivision-month climatological rainfall baselines."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "subdivision",
        "month",
        "rainfall_mm",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    baseline = (
        df.groupby(
            ["subdivision", "month"],
            observed=True,
        )["rainfall_mm"]
        .agg(
            historical_mean_mm="mean",
            historical_std_mm="std",
            historical_min_mm="min",
            historical_max_mm="max",
            observations="count",
        )
        .reset_index()
    )

    # Avoid division by zero for locations/months
    # where historical rainfall has no variation.
    baseline["historical_std_mm"] = (
        baseline["historical_std_mm"].fillna(0)
    )

    baseline["coefficient_of_variation"] = (
        baseline["historical_std_mm"]
        / baseline["historical_mean_mm"].replace(0, pd.NA)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    baseline.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return OUTPUT_FILE


if __name__ == "__main__":
    output = create_climatological_baseline()

    print(
        f"Climatological baseline created: {output}"
    )