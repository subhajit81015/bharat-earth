from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v2.csv"
)


# Legacy target columns that must never enter
# the model feature dataset.
LEAKAGE_COLUMNS = {
    "target_3m_stress",
    "target_3m_severe_anomaly",
}


def create_temporal_features() -> Path:

    print("=" * 70)
    print("TEMPORAL FEATURE ENGINEERING")
    print("=" * 70)

    # ============================================================
    # 1. LOAD INPUT
    # ============================================================

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("\nINPUT FILE:")
    print(INPUT_FILE)

    print("\nINPUT SHAPE:")
    print(df.shape)

    print("\nINPUT COLUMNS:")
    print(df.columns.tolist())

    # ============================================================
    # 2. REQUIRED COLUMNS
    # ============================================================

    required_columns = {
        "subdivision",
        "year",
        "month",
        "rainfall_mm",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ============================================================
    # 3. REMOVE LEGACY TARGET COLUMNS
    # ============================================================

    leakage_present = (
        LEAKAGE_COLUMNS
        & set(df.columns)
    )

    print("\nLEGACY / LEAKAGE COLUMNS FOUND:")
    print(sorted(leakage_present))

    if leakage_present:
        df = df.drop(
            columns=sorted(leakage_present)
        )

        print("\nREMOVED LEAKAGE COLUMNS:")
        print(sorted(leakage_present))
    else:
        print("None")

    # ============================================================
    # 4. NORMALIZE MONTH
    # ============================================================

    month_order = {
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

    original_month = (
        df["month"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    numeric_month = pd.to_numeric(
        original_month,
        errors="coerce",
    )

    text_month = original_month.map(
        month_order
    )

    month_number = numeric_month.where(
        numeric_month.notna(),
        text_month,
    )

    detected_months = sorted(
        month_number
        .dropna()
        .unique()
        .tolist()
    )

    print("\nDETECTED MONTH VALUES:")
    print(detected_months)

    # ============================================================
    # 5. HANDLE ZERO-BASED MONTHS
    # ============================================================

    if (
        len(detected_months) > 0
        and min(detected_months) == 0
        and max(detected_months) <= 11
    ):
        print(
            "\nZERO-BASED MONTH ENCODING DETECTED."
        )

        print(
            "Converting 0-11 -> 1-12."
        )

        month_number = (
            month_number + 1
        )

    # ============================================================
    # 6. VALIDATE MONTH
    # ============================================================

    invalid_month = (
        month_number.isna()
        | ~month_number.between(1, 12)
    )

    invalid_count = int(
        invalid_month.sum()
    )

    if invalid_count > 0:

        bad_values = (
            df.loc[
                invalid_month,
                "month",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Invalid month values detected: "
            f"{bad_values}"
        )

    print("\nMONTH VALIDATION:")
    print("INVALID MONTH COUNT:", invalid_count)

    # ============================================================
    # 7. CREATE DATE
    # ============================================================

    numeric_year = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["date"] = pd.to_datetime(
        {
            "year": numeric_year,
            "month": month_number,
            "day": 1,
        },
        errors="coerce",
    )

    invalid_dates = int(
        df["date"].isna().sum()
    )

    if invalid_dates > 0:
        raise ValueError(
            f"Invalid dates detected: {invalid_dates}"
        )

    # ============================================================
    # 8. SORT TEMPORALLY
    # ============================================================

    df = df.sort_values(
        [
            "subdivision",
            "date",
        ]
    ).reset_index(drop=True)

    grouped = df.groupby(
        "subdivision",
        group_keys=False,
    )

    # ============================================================
    # 9. RAINFALL LAG FEATURES
    # ============================================================

    df["rainfall_lag_1m"] = (
        grouped["rainfall_mm"]
        .shift(1)
    )

    df["rainfall_lag_2m"] = (
        grouped["rainfall_mm"]
        .shift(2)
    )

    df["rainfall_lag_3m"] = (
        grouped["rainfall_mm"]
        .shift(3)
    )

    # ============================================================
    # 10. PREVIOUS 3 MONTHS
    # ============================================================

    df["rainfall_prev_3m"] = (
        grouped["rainfall_mm"]
        .transform(
            lambda x:
                x.shift(1)
                .rolling(
                    3,
                    min_periods=1,
                )
                .sum()
        )
    )

    # ============================================================
    # 11. PREVIOUS 6 MONTHS
    # ============================================================

    df["rainfall_prev_6m"] = (
        grouped["rainfall_mm"]
        .transform(
            lambda x:
                x.shift(1)
                .rolling(
                    6,
                    min_periods=1,
                )
                .sum()
        )
    )

    # ============================================================
    # 12. PREVIOUS 12 MONTHS
    # ============================================================

    df["rainfall_prev_12m"] = (
        grouped["rainfall_mm"]
        .transform(
            lambda x:
                x.shift(1)
                .rolling(
                    12,
                    min_periods=1,
                )
                .sum()
        )
    )

    # ============================================================
    # 13. RAINFALL TREND
    # ============================================================

    df["rainfall_trend_3m"] = (
        df["rainfall_lag_1m"]
        - df["rainfall_lag_3m"]
    )

    # ============================================================
    # 14. CYCLICAL MONTH FEATURES
    # ============================================================

    df["month_sin"] = (
        np.sin(
            2
            * np.pi
            * month_number
            / 12
        )
    )

    df["month_cos"] = (
        np.cos(
            2
            * np.pi
            * month_number
            / 12
        )
    )

    # ============================================================
    # 15. FINAL LEAKAGE CHECK
    # ============================================================

    remaining_leakage = (
        LEAKAGE_COLUMNS
        & set(df.columns)
    )

    if remaining_leakage:
        raise ValueError(
            "Leakage columns remain: "
            f"{sorted(remaining_leakage)}"
        )

    # ============================================================
    # 16. REQUIRED TEMPORAL FEATURES
    # ============================================================

    new_features = [
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

    missing_features = (
        set(new_features)
        - set(df.columns)
    )

    if missing_features:
        raise ValueError(
            "Temporal features missing: "
            f"{sorted(missing_features)}"
        )

    # ============================================================
    # 17. SAVE DATASET
    # ============================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ============================================================
    # 18. FINAL REPORT
    # ============================================================

    print("\n" + "=" * 70)
    print("TEMPORAL FEATURE DATASET CREATED")
    print("=" * 70)

    print("\nOUTPUT FILE:")
    print(OUTPUT_FILE)

    print("\nOUTPUT SHAPE:")
    print(df.shape)

    print("\nOUTPUT COLUMNS:")
    print(df.columns.tolist())

    print("\nNEW FEATURES:")

    for feature in new_features:
        print(
            f"  PASS  {feature}"
        )

    print("\nLEAKAGE CHECK:")

    if remaining_leakage:
        print(
            "FAIL",
            sorted(remaining_leakage),
        )
    else:
        print("PASS")

    # ============================================================
    # 19. MONTH REPORT
    # ============================================================

    print("\nMONTH VALIDATION:")
    print(
        "SOURCE MONTH REPRESENTATION: "
        "JAN-DEC"
    )

    print(
        "MONTH NULL COUNT:",
        int(
            df["month"]
            .isna()
            .sum()
        ),
    )

    print(
        "MONTH VALUES:",
        sorted(
            df["month"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    )

    # ============================================================
    # 20. TEMPORAL NULL REPORT
    # ============================================================

    print(
        "\nTEMPORAL FEATURE NULL RATES:"
    )

    for feature in new_features:

        null_rate = (
            df[feature]
            .isna()
            .mean()
        )

        print(
            f"  {feature:<25} "
            f"{null_rate:.6f}"
        )

    print("\nFINAL STATUS: PASS")
    print("=" * 70)

    return OUTPUT_FILE


if __name__ == "__main__":
    create_temporal_features()