from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "drought_episodes.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "environmental_risk.csv"
)


def calculate_environmental_risk() -> Path:
    """Calculate an interpretable environmental risk score."""

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
        "is_dry",
        "dry_episode_length",
        "persistent_drought_signal",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # 1. Rainfall anomaly component
    # ---------------------------------------------------------

    zscore = df["rainfall_zscore"].fillna(0)

    anomaly_risk = (
        (-zscore) * 25
    ).clip(
        lower=0,
        upper=50,
    )

    # ---------------------------------------------------------
    # 2. Persistence component
    # ---------------------------------------------------------

    persistence_risk = (
        df["dry_episode_length"]
        .clip(lower=0, upper=6)
        / 6
        * 30
    )

    # Only apply persistence risk to dry observations.
    persistence_risk = (
        persistence_risk * df["is_dry"]
    )

    # ---------------------------------------------------------
    # 3. Persistent drought component
    # ---------------------------------------------------------

    drought_risk = (
        df["persistent_drought_signal"] * 20
    )

    # ---------------------------------------------------------
    # 4. Final score
    # ---------------------------------------------------------

    df["environmental_risk_score"] = (
        anomaly_risk
        + persistence_risk
        + drought_risk
    ).clip(
        lower=0,
        upper=100,
    )

    # ---------------------------------------------------------
    # 5. Risk classification
    # ---------------------------------------------------------

    df["environmental_risk_level"] = pd.cut(
        df["environmental_risk_score"],
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
            "CRITICAL",
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
    output = calculate_environmental_risk()

    print(
        f"Environmental risk dataset created: {output}"
    )