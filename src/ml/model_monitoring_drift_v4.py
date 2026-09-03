# ======================================================================
# 11. MODEL MONITORING & DRIFT V4
# ======================================================================

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v4.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "monitoring_v4"
)

DRIFT_FILE = OUTPUT_DIR / "feature_drift.csv"
SUMMARY_FILE = OUTPUT_DIR / "drift_summary.json"
REPORT_FILE = OUTPUT_DIR / "drift_report.md"


# ======================================================================
# CONFIGURATION
# ======================================================================

TARGET = "target_3m_severe_anomaly"

YEAR_COLUMN = "year"

TRAIN_END_YEAR = 2013
PRODUCTION_START_YEAR = 2016

PSI_SIGNIFICANT = 0.25
PSI_MODERATE = 0.10

EPSILON = 1e-6

RANDOM_STATE = 42


# ======================================================================
# MODEL FEATURES
# ======================================================================

FEATURE_COLUMNS = [
    "subdivision",
    "year",
    "month",
    "season",
    "rainfall_mm",
    "rainfall_3m",
    "rainfall_6m",
    "rainfall_12m",
    "historical_monthly_mean",
    "rainfall_anomaly",
    "rainfall_anomaly_pct",
    "rainfall_deficit_mm",
    "rainfall_missing",
    "rainfall_zscore",
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


CATEGORICAL_FEATURES = [
    "subdivision",
    "month",
    "season",
]


NUMERIC_FEATURES = [
    feature
    for feature in FEATURE_COLUMNS
    if feature not in CATEGORICAL_FEATURES
]


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)
    print()


def status_line(name: str, status: str, extra: str = "") -> None:
    if extra:
        print(f"{name}: {status} {extra}")
    else:
        print(f"{name}: {status}")


# ======================================================================
# LOAD DATA
# ======================================================================

def load_dataset() -> pd.DataFrame:

    banner("LOADING V4 DATASET")

    print("INPUT:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"V4 dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("SHAPE:")
    print(df.shape)

    print("COLUMNS:")
    print(df.columns.tolist())

    return df


# ======================================================================
# MONTH NORMALIZATION
# ======================================================================

def normalize_month(df: pd.DataFrame) -> pd.DataFrame:

    print()
    print("MONTH NORMALIZATION")
    print("-" * 70)

    if "month" not in df.columns:
        raise ValueError(
            "month column not found."
        )

    month_map = {
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

    # Convert everything to string first.
    month_as_string = (
        df["month"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Replace textual months.
    normalized = month_as_string.replace(month_map)

    # Numeric conversion.
    normalized = pd.to_numeric(
        normalized,
        errors="coerce",
    )

    invalid_before = int(
        normalized.isna().sum()
    )

    print(
        "INVALID MONTHS AFTER NORMALIZATION:",
        invalid_before,
    )

    if invalid_before:
        print(
            "INVALID MONTH VALUES:"
        )
        print(
            df.loc[
                normalized.isna(),
                "month"
            ].value_counts()
        )

        raise ValueError(
            "Unable to normalize all month values."
        )

    normalized = normalized.astype(int)

    invalid_range = (
        ~normalized.between(1, 12)
    )

    invalid_count = int(
        invalid_range.sum()
    )

    if invalid_count:
        print(
            "INVALID MONTH RANGE VALUES:"
        )
        print(
            normalized[
                invalid_range
            ].value_counts()
        )

        raise ValueError(
            "Month values must be between 1 and 12."
        )

    df["month"] = normalized

    print(
        "MONTH RANGE:",
        int(df["month"].min()),
        "-",
        int(df["month"].max()),
    )

    print(
        "MONTH VALIDATION: PASS"
    )

    return df


# ======================================================================
# SEASON NORMALIZATION
# ======================================================================

def season_from_month(month: int) -> str:

    if month in [12, 1, 2]:
        return "WINTER"

    if month in [3, 4, 5]:
        return "PRE_MONSOON"

    if month in [6, 7, 8, 9]:
        return "MONSOON"

    if month in [10, 11]:
        return "POST_MONSOON"

    raise ValueError(
        f"Invalid month: {month}"
    )


def normalize_season(df: pd.DataFrame) -> pd.DataFrame:

    print()
    print("SEASON VALIDATION")
    print("-" * 70)

    expected_season = (
        df["month"]
        .apply(season_from_month)
    )

    existing_season = (
        df["season"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    inconsistencies = int(
        (
            existing_season
            != expected_season
        ).sum()
    )

    print(
        "SEASON INCONSISTENCIES:",
        inconsistencies,
    )

    if inconsistencies:
        print(
            "Repairing season from month."
        )

    df["season"] = expected_season

    allowed = {
        "MONSOON",
        "POST_MONSOON",
        "PRE_MONSOON",
        "WINTER",
    }

    invalid = (
        ~df["season"].isin(allowed)
    )

    if invalid.any():
        raise ValueError(
            "Invalid season values found."
        )

    print(
        "SEASON VALIDATION: PASS"
    )

    return df


# ======================================================================
# DATASET VALIDATION
# ======================================================================

def validate_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    banner("DATASET VALIDATION")

    required = set(
        FEATURE_COLUMNS
        + [TARGET]
    )

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            f"{sorted(missing)}"
        )

    print(
        "REQUIRED FEATURE SCHEMA: PASS"
    )

    leakage_columns = {
        "target_3m_stress",
        "rainfall_stress",
        "persistent_drought_signal",
        "environmental_risk_score",
        "environmental_risk_level",
    }

    present_leakage = (
        leakage_columns
        & set(df.columns)
    )

    print(
        "LEGACY LEAKAGE COLUMNS:"
    )
    print(
        sorted(present_leakage)
    )

    if present_leakage:
        raise ValueError(
            "Leakage columns found:\n"
            f"{sorted(present_leakage)}"
        )

    target_values = sorted(
        pd.to_numeric(
            df[TARGET],
            errors="coerce",
        )
        .dropna()
        .unique()
        .tolist()
    )

    print(
        "TARGET VALUES:",
        target_values,
    )

    if set(target_values) != {0, 1}:
        raise ValueError(
            "Target must contain exactly 0 and 1."
        )

    print(
        "TARGET DISTRIBUTION:"
    )
    print(
        df[TARGET].value_counts()
    )

    print(
        "TARGET RATE:",
        float(df[TARGET].mean()),
    )

    duplicates = int(
        df.duplicated().sum()
    )

    print(
        "EXACT DUPLICATES:",
        duplicates,
    )

    if duplicates:
        raise ValueError(
            f"Exact duplicates found: {duplicates}"
        )

    df = normalize_month(df)

    df = normalize_season(df)

    print(
        "DATASET VALIDATION: PASS"
    )

    return df


# ======================================================================
# NUMERIC BIN CREATION
# ======================================================================

def make_numeric_bins(
    reference: pd.Series,
    production: pd.Series,
    bins: int = 10,
) -> np.ndarray:

    reference = pd.to_numeric(
        reference,
        errors="coerce",
    ).dropna()

    production = pd.to_numeric(
        production,
        errors="coerce",
    ).dropna()

    combined = pd.concat(
        [reference, production],
        ignore_index=True,
    )

    if combined.empty:
        return np.array(
            [-np.inf, np.inf],
            dtype=float,
        )

    quantiles = np.linspace(
        0,
        1,
        bins + 1,
    )

    edges = (
        combined
        .quantile(quantiles)
        .to_numpy(
            dtype=float
        )
    )

    edges = np.unique(edges)

    if len(edges) < 2:
        value = float(
            combined.iloc[0]
        )

        return np.array(
            [
                -np.inf,
                value - EPSILON,
                value + EPSILON,
                np.inf,
            ]
        )

    edges[0] = -np.inf
    edges[-1] = np.inf

    return edges


# ======================================================================
# PSI
# ======================================================================

def calculate_psi(
    reference: pd.Series,
    production: pd.Series,
    bins: int = 10,
) -> float:

    reference = pd.to_numeric(
        reference,
        errors="coerce",
    )

    production = pd.to_numeric(
        production,
        errors="coerce",
    )

    reference = reference.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    production = production.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if reference.empty or production.empty:
        return float("nan")

    edges = make_numeric_bins(
        reference,
        production,
        bins=bins,
    )

    reference_counts, _ = np.histogram(
        reference,
        bins=edges,
    )

    production_counts, _ = np.histogram(
        production,
        bins=edges,
    )

    reference_pct = (
        reference_counts
        / max(
            reference_counts.sum(),
            1,
        )
    )

    production_pct = (
        production_counts
        / max(
            production_counts.sum(),
            1,
        )
    )

    reference_pct = np.clip(
        reference_pct,
        EPSILON,
        None,
    )

    production_pct = np.clip(
        production_pct,
        EPSILON,
        None,
    )

    psi = np.sum(
        (
            production_pct
            - reference_pct
        )
        * np.log(
            production_pct
            / reference_pct
        )
    )

    return float(psi)


# ======================================================================
# CATEGORICAL PSI
# ======================================================================

def calculate_categorical_psi(
    reference: pd.Series,
    production: pd.Series,
) -> float:

    reference = (
        reference
        .astype(str)
        .fillna("__MISSING__")
    )

    production = (
        production
        .astype(str)
        .fillna("__MISSING__")
    )

    categories = sorted(
        set(reference.unique())
        | set(production.unique())
    )

    if not categories:
        return float("nan")

    reference_counts = (
        reference
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
    )

    production_counts = (
        production
        .value_counts()
        .reindex(
            categories,
            fill_value=0,
        )
    )

    reference_pct = (
        reference_counts
        / max(
            reference_counts.sum(),
            1,
        )
    )

    production_pct = (
        production_counts
        / max(
            production_counts.sum(),
            1,
        )
    )

    reference_pct = np.clip(
        reference_pct.to_numpy(
            dtype=float
        ),
        EPSILON,
        None,
    )

    production_pct = np.clip(
        production_pct.to_numpy(
            dtype=float
        ),
        EPSILON,
        None,
    )

    psi = np.sum(
        (
            production_pct
            - reference_pct
        )
        * np.log(
            production_pct
            / reference_pct
        )
    )

    return float(psi)


# ======================================================================
# DRIFT CLASSIFICATION
# ======================================================================

def classify_drift(
    psi: float,
) -> str:

    if not np.isfinite(psi):
        return "UNKNOWN"

    if psi >= PSI_SIGNIFICANT:
        return "SIGNIFICANT"

    if psi >= PSI_MODERATE:
        return "MODERATE"

    return "STABLE"


# ======================================================================
# FEATURE DRIFT
# ======================================================================

def calculate_feature_drift(
    train_df: pd.DataFrame,
    production_df: pd.DataFrame,
) -> pd.DataFrame:

    banner("CALCULATING FEATURE DRIFT")

    rows = []

    for feature in FEATURE_COLUMNS:

        reference = train_df[feature]
        production = production_df[feature]

        missing_reference = float(
            reference.isna().mean()
        )

        missing_production = float(
            production.isna().mean()
        )

        missing_rate_change = (
            missing_production
            - missing_reference
        )

        if feature in CATEGORICAL_FEATURES:

            psi = calculate_categorical_psi(
                reference,
                production,
            )

            feature_type = "categorical"

        else:

            psi = calculate_psi(
                reference,
                production,
            )

            feature_type = "numeric"

        drift_status = classify_drift(
            psi
        )

        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "reference_rows": int(
                    len(reference)
                ),
                "production_rows": int(
                    len(production)
                ),
                "reference_missing_rate": missing_reference,
                "production_missing_rate": missing_production,
                "missing_rate_change": missing_rate_change,
                "psi": psi,
                "drift_status": drift_status,
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "psi",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    result["rank"] = (
        np.arange(len(result))
        + 1
    )

    return result


# ======================================================================
# SUMMARY
# ======================================================================

def create_summary(
    drift_df: pd.DataFrame,
    train_df: pd.DataFrame,
    production_df: pd.DataFrame,
) -> dict:

    significant = drift_df[
        drift_df["drift_status"]
        == "SIGNIFICANT"
    ]

    moderate = drift_df[
        drift_df["drift_status"]
        == "MODERATE"
    ]

    stable = drift_df[
        drift_df["drift_status"]
        == "STABLE"
    ]

    max_psi = (
        float(
            drift_df["psi"]
            .dropna()
            .max()
        )
        if drift_df["psi"].notna().any()
        else None
    )

    max_psi_feature = None

    if drift_df["psi"].notna().any():

        max_index = (
            drift_df["psi"]
            .idxmax()
        )

        max_psi_feature = (
            str(
                drift_df.loc[
                    max_index,
                    "feature",
                ]
            )
        )

    summary = {
        "step": "11",
        "project_stage": "MODEL MONITORING AND DRIFT V4",
        "input_file": str(INPUT_FILE),
        "reference_period": {
            "start_year": int(
                train_df[YEAR_COLUMN].min()
            ),
            "end_year": int(
                train_df[YEAR_COLUMN].max()
            ),
            "rows": int(
                len(train_df)
            ),
        },
        "production_period": {
            "start_year": int(
                production_df[YEAR_COLUMN].min()
            ),
            "end_year": int(
                production_df[YEAR_COLUMN].max()
            ),
            "rows": int(
                len(production_df)
            ),
        },
        "feature_count": int(
            len(FEATURE_COLUMNS)
        ),
        "categorical_feature_count": int(
            len(CATEGORICAL_FEATURES)
        ),
        "numeric_feature_count": int(
            len(NUMERIC_FEATURES)
        ),
        "psi_thresholds": {
            "stable_below": PSI_MODERATE,
            "moderate_from": PSI_MODERATE,
            "significant_from": PSI_SIGNIFICANT,
        },
        "drift_counts": {
            "stable": int(
                len(stable)
            ),
            "moderate": int(
                len(moderate)
            ),
            "significant": int(
                len(significant)
            ),
        },
        "maximum_psi": max_psi,
        "maximum_psi_feature": max_psi_feature,
        "significant_drift_features": (
            significant["feature"]
            .tolist()
        ),
        "moderate_drift_features": (
            moderate["feature"]
            .tolist()
        ),
        "production_readiness": (
            "REVIEW"
            if len(significant) > 0
            else "PASS"
        ),
    }

    return summary


# ======================================================================
# MARKDOWN REPORT
# ======================================================================

def create_markdown_report(
    summary: dict,
    drift_df: pd.DataFrame,
) -> str:

    lines = []

    lines.append(
        "# Model Monitoring & Drift V4"
    )
    lines.append("")

    lines.append(
        "## Monitoring Scope"
    )
    lines.append("")

    lines.append(
        f"- Reference period: "
        f"{summary['reference_period']['start_year']}"
        f"-"
        f"{summary['reference_period']['end_year']}"
    )

    lines.append(
        f"- Production period: "
        f"{summary['production_period']['start_year']}"
        f"-"
        f"{summary['production_period']['end_year']}"
    )

    lines.append(
        f"- Reference rows: "
        f"{summary['reference_period']['rows']}"
    )

    lines.append(
        f"- Production rows: "
        f"{summary['production_period']['rows']}"
    )

    lines.append(
        f"- Features monitored: "
        f"{summary['feature_count']}"
    )

    lines.append("")

    lines.append(
        "## PSI Thresholds"
    )
    lines.append("")

    lines.append(
        "| PSI | Interpretation |"
    )
    lines.append(
        "|---:|---|"
    )
    lines.append(
        "| < 0.10 | Stable |"
    )
    lines.append(
        "| 0.10 - 0.25 | Moderate |"
    )
    lines.append(
        "| >= 0.25 | Significant |"
    )

    lines.append("")

    lines.append(
        "## Drift Summary"
    )
    lines.append("")

    lines.append(
        f"- Stable features: "
        f"{summary['drift_counts']['stable']}"
    )

    lines.append(
        f"- Moderate drift features: "
        f"{summary['drift_counts']['moderate']}"
    )

    lines.append(
        f"- Significant drift features: "
        f"{summary['drift_counts']['significant']}"
    )

    lines.append(
        f"- Maximum PSI: "
        f"{summary['maximum_psi']}"
    )

    lines.append(
        f"- Maximum PSI feature: "
        f"{summary['maximum_psi_feature']}"
    )

    lines.append("")

    lines.append(
        "## Feature Drift"
    )
    lines.append("")

    lines.append(
        "| Rank | Feature | Type | PSI | Status |"
    )
    lines.append(
        "|---:|---|---|---:|---|"
    )

    for _, row in drift_df.iterrows():

        psi = row["psi"]

        if pd.isna(psi):
            psi_text = "N/A"
        else:
            psi_text = f"{float(psi):.6f}"

        lines.append(
            f"| {int(row['rank'])} "
            f"| {row['feature']} "
            f"| {row['feature_type']} "
            f"| {psi_text} "
            f"| {row['drift_status']} |"
        )

    lines.append("")

    lines.append(
        "## Production Recommendation"
    )
    lines.append("")

    if (
        summary["drift_counts"]["significant"]
        > 0
    ):

        lines.append(
            "Significant feature drift was detected. "
            "Production deployment should remain under "
            "review until the affected features are "
            "investigated."
        )

    else:

        lines.append(
            "No significant feature drift was detected "
            "under the configured PSI threshold."
        )

    lines.append("")

    return "\n".join(lines)


# ======================================================================
# SAVE OUTPUTS
# ======================================================================

def save_outputs(
    drift_df: pd.DataFrame,
    summary: dict,
) -> None:

    banner("SAVING MONITORING OUTPUTS")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    drift_df.to_csv(
        DRIFT_FILE,
        index=False,
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            allow_nan=False,
        )

    report = create_markdown_report(
        summary,
        drift_df,
    )

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print(
        "feature_drift.csv: PASS"
    )
    print(
        DRIFT_FILE
    )

    print(
        "drift_summary.json: PASS"
    )
    print(
        SUMMARY_FILE
    )

    print(
        "drift_report.md: PASS"
    )
    print(
        REPORT_FILE
    )


# ======================================================================
# OUTPUT VALIDATION
# ======================================================================

def validate_outputs(
    drift_df: pd.DataFrame,
    summary: dict,
) -> None:

    banner("OUTPUT VALIDATION")

    if not DRIFT_FILE.exists():
        raise ValueError(
            "feature_drift.csv was not created."
        )

    if not SUMMARY_FILE.exists():
        raise ValueError(
            "drift_summary.json was not created."
        )

    if not REPORT_FILE.exists():
        raise ValueError(
            "drift_report.md was not created."
        )

    required_drift_columns = [
        "feature",
        "feature_type",
        "reference_rows",
        "production_rows",
        "reference_missing_rate",
        "production_missing_rate",
        "missing_rate_change",
        "psi",
        "drift_status",
        "rank",
    ]

    missing = (
        set(required_drift_columns)
        - set(drift_df.columns)
    )

    if missing:
        raise ValueError(
            "Missing drift output columns:\n"
            f"{sorted(missing)}"
        )

    if len(drift_df) != len(
        FEATURE_COLUMNS
    ):
        raise ValueError(
            "Unexpected drift row count."
        )

    if not set(
        drift_df["drift_status"]
    ).issubset(
        {
            "STABLE",
            "MODERATE",
            "SIGNIFICANT",
            "UNKNOWN",
        }
    ):
        raise ValueError(
            "Invalid drift status found."
        )

    print(
        "DRIFT FILE: PASS"
    )

    print(
        "DRIFT ROWS:",
        len(drift_df),
    )

    print(
        "SUMMARY FILE: PASS"
    )

    print(
        "REPORT FILE: PASS"
    )

    print(
        "FEATURE COUNT: PASS"
    )

    print(
        "OUTPUT VALIDATION: PASS"
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    banner(
        "11. MODEL MONITORING & DRIFT V4"
    )

    # --------------------------------------------------------------
    # Load
    # --------------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------------
    # Validate
    # --------------------------------------------------------------

    df = validate_dataset(df)

    # --------------------------------------------------------------
    # Validate year
    # --------------------------------------------------------------

    if not pd.api.types.is_numeric_dtype(
        df[YEAR_COLUMN]
    ):

        df[YEAR_COLUMN] = pd.to_numeric(
            df[YEAR_COLUMN],
            errors="coerce",
        )

    if df[YEAR_COLUMN].isna().any():
        raise ValueError(
            "Invalid year values found."
        )

    df[YEAR_COLUMN] = (
        df[YEAR_COLUMN]
        .astype(int)
    )

    print()
    print(
        "YEAR RANGE:",
        int(df[YEAR_COLUMN].min()),
        "-",
        int(df[YEAR_COLUMN].max()),
    )

    # --------------------------------------------------------------
    # Reference / production split
    # --------------------------------------------------------------

    banner(
        "TEMPORAL MONITORING SPLIT"
    )

    reference_df = df[
        df[YEAR_COLUMN]
        <= TRAIN_END_YEAR
    ].copy()

    production_df = df[
        df[YEAR_COLUMN]
        >= PRODUCTION_START_YEAR
    ].copy()

    if reference_df.empty:
        raise ValueError(
            "Reference dataset is empty."
        )

    if production_df.empty:
        raise ValueError(
            "Production dataset is empty."
        )

    print(
        "REFERENCE ROWS:",
        len(reference_df),
    )

    print(
        "PRODUCTION ROWS:",
        len(production_df),
    )

    print()

    print(
        "REFERENCE YEAR RANGE:",
        int(
            reference_df[
                YEAR_COLUMN
            ].min()
        ),
        "-",
        int(
            reference_df[
                YEAR_COLUMN
            ].max()
        ),
    )

    print(
        "PRODUCTION YEAR RANGE:",
        int(
            production_df[
                YEAR_COLUMN
            ].min()
        ),
        "-",
        int(
            production_df[
                YEAR_COLUMN
            ].max()
        ),
    )

    # --------------------------------------------------------------
    # Drift
    # --------------------------------------------------------------

    drift_df = calculate_feature_drift(
        reference_df,
        production_df,
    )

    # --------------------------------------------------------------
    # Print results
    # --------------------------------------------------------------

    banner(
        "DRIFT RESULTS"
    )

    print(
        drift_df[
            [
                "rank",
                "feature",
                "feature_type",
                "psi",
                "drift_status",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    summary = create_summary(
        drift_df,
        reference_df,
        production_df,
    )

    # --------------------------------------------------------------
    # Summary display
    # --------------------------------------------------------------

    banner(
        "DRIFT SUMMARY"
    )

    print(
        "STABLE FEATURES:",
        summary[
            "drift_counts"
        ]["stable"],
    )

    print(
        "MODERATE DRIFT FEATURES:",
        summary[
            "drift_counts"
        ]["moderate"],
    )

    print(
        "SIGNIFICANT DRIFT FEATURES:",
        summary[
            "drift_counts"
        ]["significant"],
    )

    print(
        "MAXIMUM PSI:",
        summary[
            "maximum_psi"
        ],
    )

    print(
        "MAXIMUM PSI FEATURE:",
        summary[
            "maximum_psi_feature"
        ],
    )

    if summary[
        "significant_drift_features"
    ]:

        print()
        print(
            "SIGNIFICANT DRIFT FEATURES:"
        )

        for feature in summary[
            "significant_drift_features"
        ]:

            print(
                f"- {feature}"
            )

    if summary[
        "moderate_drift_features"
    ]:

        print()
        print(
            "MODERATE DRIFT FEATURES:"
        )

        for feature in summary[
            "moderate_drift_features"
        ]:

            print(
                f"- {feature}"
            )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------

    save_outputs(
        drift_df,
        summary,
    )

    # --------------------------------------------------------------
    # Validate
    # --------------------------------------------------------------

    validate_outputs(
        drift_df,
        summary,
    )

    # --------------------------------------------------------------
    # Final status
    # --------------------------------------------------------------

    banner(
        "11. MODEL MONITORING & DRIFT V4 COMPLETE"
    )

    if (
        summary[
            "drift_counts"
        ]["significant"]
        > 0
    ):

        print(
            "STATUS: REVIEW"
        )

        print(
            "SIGNIFICANT DRIFT DETECTED."
        )

    else:

        print(
            "STATUS: PASS"
        )

        print(
            "NO SIGNIFICANT DRIFT DETECTED."
        )

    print()
    print(
        "OUTPUT DIRECTORY:"
    )
    print(
        OUTPUT_DIR
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()