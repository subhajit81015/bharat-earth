from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset.csv"
)


def run_diagnostics() -> None:

    df = pd.read_csv(INPUT_FILE)

    print("SHAPE:", df.shape)

    print("\nTARGET COUNTS:")
    print(
        df["target_3m_stress"]
        .value_counts(dropna=False)
    )

    print("\nTARGET RATE:")
    print(
        df["target_3m_stress"].mean()
    )

    print("\nTARGET BY YEAR:")

    yearly = (
        df.groupby("year")["target_3m_stress"]
        .agg(
            observations="count",
            positives="sum",
            positive_rate="mean",
        )
    )

    print(yearly.to_string())

    print("\nTARGET BY MONTH:")

    monthly = (
        df.groupby("month")["target_3m_stress"]
        .agg(
            observations="count",
            positives="sum",
            positive_rate="mean",
        )
        .sort_values(
            "positive_rate",
            ascending=False,
        )
    )

    print(monthly.to_string())

    print("\nTARGET BY SEASON:")

    seasonal = (
        df.groupby("season")["target_3m_stress"]
        .agg(
            observations="count",
            positives="sum",
            positive_rate="mean",
        )
        .sort_values(
            "positive_rate",
            ascending=False,
        )
    )

    print(seasonal.to_string())

    print("\nTARGET BY SUBDIVISION:")

    regional = (
        df.groupby("subdivision")[
            "target_3m_stress"
        ]
        .agg(
            observations="count",
            positives="sum",
            positive_rate="mean",
        )
        .sort_values(
            "positive_rate",
            ascending=False,
        )
    )

    print(
        regional.head(20).to_string()
    )


if __name__ == "__main__":
    run_diagnostics()