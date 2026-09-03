from pathlib import Path
import json
import math
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd


# ================================================================
# 3.13 FINAL PROJECT VALIDATION & PRODUCTION READINESS
# ================================================================
#
# PURPOSE
# -------
# Final end-to-end validation of the Bharat Earth ML project.
#
# IMPORTANT
# ---------
# This validator automatically prefers:
#
#   final_risk_predictions_fixed.csv
#
# over:
#
#   final_risk_predictions.csv
#
# because the fixed deployment artifact contains corrected
# 1-based calendar months and recalculated seasons.
#
# ================================================================


# ================================================================
# PROJECT CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(
    r"C:\Users\subha\Downloads\bharat-earth"
)

FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

DEPLOYMENT_DIR = (
    FEATURE_DIR
    / "deployment"
)

MONITORING_DIR = (
    FEATURE_DIR
    / "monitoring"
)

FINAL_VALIDATION_DIR = (
    FEATURE_DIR
    / "final_validation"
)


# ================================================================
# INPUT FILES
# ================================================================

FEATURE_FILE = (
    FEATURE_DIR
    / "ml_dataset_v2.csv"
)

TARGET_FILE = (
    FEATURE_DIR
    / "severe_anomaly_target.csv"
)

ORIGINAL_DEPLOYMENT_FILE = (
    DEPLOYMENT_DIR
    / "final_risk_predictions.csv"
)

FIXED_DEPLOYMENT_FILE = (
    DEPLOYMENT_DIR
    / "final_risk_predictions_fixed.csv"
)

DRIFT_FILE = (
    MONITORING_DIR
    / "feature_drift.csv"
)


# ================================================================
# OUTPUT FILES
# ================================================================

KEY_VALIDATION_FILE = (
    FINAL_VALIDATION_DIR
    / "key_validation.csv"
)

LEAKAGE_VALIDATION_FILE = (
    FINAL_VALIDATION_DIR
    / "leakage_validation.csv"
)

MISSING_VALIDATION_FILE = (
    FINAL_VALIDATION_DIR
    / "missing_value_validation.csv"
)

MONTH_VALIDATION_FILE = (
    FINAL_VALIDATION_DIR
    / "month_validation.csv"
)

SEASON_VALIDATION_FILE = (
    FINAL_VALIDATION_DIR
    / "season_validation.csv"
)

PERFORMANCE_VALIDATION_FILE = (
    FINAL_VALIDATION_DIR
    / "performance_validation.csv"
)

POLICY_VALIDATION_FILE = (
    FINAL_VALIDATION_DIR
    / "policy_validation.csv"
)

DRIFT_VALIDATION_FILE = (
    FINAL_VALIDATION_DIR
    / "drift_validation.csv"
)

SCHEMA_VALIDATION_FILE = (
    FINAL_VALIDATION_DIR
    / "schema_validation.csv"
)

ARTIFACT_VALIDATION_FILE = (
    FINAL_VALIDATION_DIR
    / "artifact_validation.csv"
)

REPRODUCIBILITY_FILE = (
    FINAL_VALIDATION_DIR
    / "reproducibility.csv"
)

TEMPORAL_VALIDATION_FILE = (
    FINAL_VALIDATION_DIR
    / "temporal_validation.csv"
)

FINAL_REPORT_FILE = (
    FINAL_VALIDATION_DIR
    / "final_validation_report.csv"
)

SUMMARY_JSON_FILE = (
    FINAL_VALIDATION_DIR
    / "validation_summary.json"
)

MANIFEST_JSON_FILE = (
    FINAL_VALIDATION_DIR
    / "deployment_manifest.json"
)

MARKDOWN_REPORT_FILE = (
    FINAL_VALIDATION_DIR
    / "production_readiness_report.md"
)

EXCEL_REPORT_FILE = (
    FINAL_VALIDATION_DIR
    / "production_readiness_report.xlsx"
)


# ================================================================
# EXPECTED PROJECT VALUES
# ================================================================

EXPECTED_DEPLOYMENT_ROWS = 7668

EXPECTED_TOTAL_ROWS = 50133

TARGET_COLUMN = (
    "target_3m_severe_anomaly"
)

PROBABILITY_COLUMN = (
    "risk_probability"
)

ACTUAL_COLUMN = (
    "actual"
)

MONTH_COLUMN = (
    "month"
)

SEASON_COLUMN = (
    "season"
)


# ================================================================
# PERFORMANCE GATES
# ================================================================

MIN_PR_AUC = 0.10

MIN_ROC_AUC = 0.70

MIN_PRECISION = 0.10

MIN_RECALL = 0.25

MIN_F1 = 0.15

MAX_BRIER = 0.10


# ================================================================
# POLICY GATES
# ================================================================

MIN_ALERT_RATE = 0.01

MAX_ALERT_RATE = 0.50


# ================================================================
# DRIFT THRESHOLDS
# ================================================================

MODERATE_PSI_THRESHOLD = 0.10

SIGNIFICANT_PSI_THRESHOLD = 0.25


# ================================================================
# EXPECTED SEASONS
# ================================================================

EXPECTED_SEASONS = {
    "WINTER",
    "PRE_MONSOON",
    "MONSOON",
    "POST_MONSOON",
}


# ================================================================
# REQUIRED PRODUCTION DEPLOYMENT COLUMNS
# ================================================================

REQUIRED_DEPLOYMENT_COLUMNS = [
    "subdivision",
    "year",
    "month",
    "season",
    "actual",
    "final_probability",
    "risk_probability",
    "risk_level",
    "risk_alert",
    "alert_priority",
    "policy_threshold",
    "prediction_status",
]


# ================================================================
# REQUIRED MODEL FEATURES
# ================================================================

REQUIRED_FEATURES = [
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


# ================================================================
# EXPECTED ARTIFACTS
# ================================================================

EXPECTED_ARTIFACTS = {
    "deployment_predictions": (
        DEPLOYMENT_DIR
        / "final_risk_predictions_fixed.csv"
    ),

    "final_predictions": (
        FEATURE_DIR
        / "final_model"
        / "final_predictions.csv"
    ),

    "feature_importance": (
        FEATURE_DIR
        / "explainability"
        / "xgboost_feature_importance.csv"
    ),

    "permutation_importance": (
        FEATURE_DIR
        / "explainability"
        / "permutation_importance.csv"
    ),

    "shap_importance": (
        FEATURE_DIR
        / "explainability"
        / "shap_importance.csv"
    ),

    "high_risk_records": (
        FEATURE_DIR
        / "explainability"
        / "top_100_high_risk_records.csv"
    ),

    "seasonal_policy": (
        FEATURE_DIR
        / "seasonal_policy_results.csv"
    ),

    "risk_alert_policy": (
        FEATURE_DIR
        / "risk_alert_policy.csv"
    ),

    "regional_risk_profile": (
        FEATURE_DIR
        / "regional_risk_profile.csv"
    ),

    "seasonal_risk_profile": (
        FEATURE_DIR
        / "seasonal_risk_profile.csv"
    ),
}


# ================================================================
# REPRODUCIBILITY SCRIPTS
# ================================================================

EXPECTED_SCRIPTS = [
    "feature_ablation.py",
    "model_explainability.py",
    "seasonal_policy_optimization.py",
    "model_monitoring_drift.py",
    "final_prediction_deployment_report.py",
    "final_project_validation.py",
]


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def normalize_text(series):
    return (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
    )


def normalize_year(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def normalize_month(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def month_to_season(month):

    if pd.isna(month):
        return "UNKNOWN"

    month = int(month)

    if month in [12, 1, 2]:
        return "WINTER"

    if month in [3, 4, 5]:
        return "PRE_MONSOON"

    if month in [6, 7, 8, 9]:
        return "MONSOON"

    if month in [10, 11]:
        return "POST_MONSOON"

    return "UNKNOWN"


def make_key_frame(
    df,
    columns
):
    """
    Safe key generation.

    IMPORTANT:
    Never uses '|'.join directly on raw dataframe
    values because pandas NAType causes:

        TypeError:
        sequence item expected str instance, NAType found

    Every value is converted safely to string.
    """

    temp = pd.DataFrame(index=df.index)

    for column in columns:

        if column not in df.columns:

            temp[column] = (
                "__MISSING_COLUMN__"
            )

        else:

            values = df[column]

            values = (
                values
                .astype("string")
                .fillna("__NA__")
                .str.strip()
            )

            temp[column] = values

    key = temp.astype(str).agg(
        "|".join,
        axis=1
    )

    return key


def result_row(
    section,
    check,
    status,
    value="",
    details=""
):

    return {
        "section": section,
        "check": check,
        "status": status,
        "value": value,
        "details": details,
    }


def ensure_output_dir():
    FINAL_VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ================================================================
# LOAD DATA
# ================================================================

def load_data():

    print("=" * 70)
    print(
        "3.13 FINAL PROJECT VALIDATION "
        "& PRODUCTION READINESS"
    )
    print("=" * 70)

    ensure_output_dir()

    # ------------------------------------------------------------
    # Deployment selection
    # ------------------------------------------------------------

    if FIXED_DEPLOYMENT_FILE.exists():

        deployment_file = (
            FIXED_DEPLOYMENT_FILE
        )

        print()
        print(
            "REPAIRED DEPLOYMENT FILE FOUND."
        )

        print(
            "USING:",
            deployment_file
        )

    elif ORIGINAL_DEPLOYMENT_FILE.exists():

        deployment_file = (
            ORIGINAL_DEPLOYMENT_FILE
        )

        print()
        print(
            "WARNING: REPAIRED DEPLOYMENT "
            "FILE NOT FOUND."
        )

        print(
            "USING ORIGINAL:",
            deployment_file
        )

    else:

        raise FileNotFoundError(
            "No deployment file found."
        )

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{FEATURE_FILE}"
        )

    if not TARGET_FILE.exists():

        raise FileNotFoundError(
            f"Target file not found:\n"
            f"{TARGET_FILE}"
        )

    features = pd.read_csv(
        FEATURE_FILE,
        low_memory=False
    )

    target = pd.read_csv(
        TARGET_FILE,
        low_memory=False
    )

    deployment = pd.read_csv(
        deployment_file,
        low_memory=False
    )

    print()
    print("=" * 70)
    print(
        "LOADING FINAL VALIDATION DATA"
    )
    print("=" * 70)

    print(
        "FEATURE FILE:",
        FEATURE_FILE
    )

    print(
        "FEATURE ROWS:",
        len(features)
    )

    print(
        "TARGET FILE:",
        TARGET_FILE
    )

    print(
        "TARGET ROWS:",
        len(target)
    )

    print(
        "DEPLOYMENT FILE:",
        deployment_file
    )

    print(
        "DEPLOYMENT ROWS:",
        len(deployment)
    )

    print(
        "FINAL PREDICTIONS:",
        len(deployment)
    )

    return (
        features,
        target,
        deployment,
        deployment_file
    )


# ================================================================
# 1. DATASET INTEGRITY
# ================================================================

def validate_dataset_integrity(
    features,
    target,
    deployment
):

    print()
    print("=" * 70)
    print("1. DATASET INTEGRITY")
    print("=" * 70)

    results = []

    # ------------------------------------------------------------
    # Feature file
    # ------------------------------------------------------------

    results.append(
        result_row(
            "DATASET INTEGRITY",
            "FEATURE DATASET",
            "PASS",
            str(FEATURE_FILE)
        )
    )

    print(
        f"{'FEATURE DATASET':40}"
        f"PASS"
        f"               {FEATURE_FILE}"
    )

    # ------------------------------------------------------------
    # Target file
    # ------------------------------------------------------------

    results.append(
        result_row(
            "DATASET INTEGRITY",
            "TARGET DATASET",
            "PASS",
            str(TARGET_FILE)
        )
    )

    print(
        f"{'TARGET DATASET':40}"
        f"PASS"
        f"               {TARGET_FILE}"
    )

    # ------------------------------------------------------------
    # Deployment
    # ------------------------------------------------------------

    results.append(
        result_row(
            "DATASET INTEGRITY",
            "DEPLOYMENT DATASET",
            "PASS",
            "fixed deployment selected"
        )
    )

    print(
        f"{'DEPLOYMENT DATASET':40}"
        f"PASS"
        f"               fixed deployment"
    )

    # ------------------------------------------------------------
    # Target column
    # ------------------------------------------------------------

    target_column_pass = (
        TARGET_COLUMN in target.columns
    )

    status = (
        "PASS"
        if target_column_pass
        else "FAIL"
    )

    results.append(
        result_row(
            "DATASET INTEGRITY",
            "TARGET COLUMN",
            status
        )
    )

    print(
        f"{'TARGET COLUMN':40}"
        f"{status}"
    )

    if not target_column_pass:

        return results

    # ------------------------------------------------------------
    # Binary target
    # ------------------------------------------------------------

    target_values = set(
        safe_numeric(
            target[TARGET_COLUMN]
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    binary_pass = (
        target_values.issubset({0, 1})
    )

    positive = int(
        (
            safe_numeric(
                target[TARGET_COLUMN]
            )
            == 1
        ).sum()
    )

    negative = int(
        (
            safe_numeric(
                target[TARGET_COLUMN]
            )
            == 0
        ).sum()
    )

    status = (
        "PASS"
        if binary_pass
        else "FAIL"
    )

    results.append(
        result_row(
            "DATASET INTEGRITY",
            "TARGET BINARY VALIDATION",
            status,
            f"positive={positive}, negative={negative}"
        )
    )

    print(
        f"{'TARGET BINARY VALIDATION':40}"
        f"{status}"
        f"               positive={positive}, "
        f"negative={negative}"
    )

    if not binary_pass:

        return results

    # ------------------------------------------------------------
    # Target rate
    # ------------------------------------------------------------

    total_target = positive + negative

    target_rate = (
        positive / total_target
        if total_target > 0
        else np.nan
    )

    results.append(
        result_row(
            "DATASET INTEGRITY",
            "TARGET RATE",
            "PASS",
            f"{target_rate:.6f}"
        )
    )

    print(
        f"{'TARGET RATE':40}"
        f"PASS"
        f"               {target_rate:.6f}"
    )

    # ------------------------------------------------------------
    # Required feature schema
    # ------------------------------------------------------------

    missing_features = [
        feature
        for feature in REQUIRED_FEATURES
        if feature not in features.columns
    ]

    schema_pass = (
        len(missing_features) == 0
    )

    status = (
        "PASS"
        if schema_pass
        else "FAIL"
    )

    details = (
        "All required features present"
        if schema_pass
        else str(missing_features)
    )

    results.append(
        result_row(
            "DATASET INTEGRITY",
            "REQUIRED FEATURE SCHEMA",
            status,
            details=details
        )
    )

    print(
        f"{'REQUIRED FEATURE SCHEMA':40}"
        f"{status}"
        f"               {details}"
    )

    # ------------------------------------------------------------
    # Exact duplicates
    # ------------------------------------------------------------

    for name, dataframe in [
        ("FEATURE", features),
        ("TARGET", target),
        ("DEPLOYMENT", deployment),
    ]:

        duplicate_count = int(
            dataframe.duplicated()
            .sum()
        )

        status = (
            "PASS"
            if duplicate_count == 0
            else "REVIEW"
        )

        results.append(
            result_row(
                "DATASET INTEGRITY",
                f"{name} EXACT DUPLICATES",
                status,
                duplicate_count
            )
        )

        print(
            f"{name + ' EXACT DUPLICATES':40}"
            f"{status}"
            f"               {duplicate_count}"
        )

    return results


# ================================================================
# 2. KEY STRUCTURE VALIDATION
# ================================================================

def validate_key_structure(
    features,
    target,
    deployment
):

    print()
    print("=" * 70)
    print("2. KEY STRUCTURE VALIDATION")
    print("=" * 70)

    results = []

    key_columns = [
        "subdivision",
        "year",
        "month",
    ]

    duplicate_outputs = {
        "FEATURES":
            FINAL_VALIDATION_DIR
            / "features_duplicate_keys.csv",

        "TARGET":
            FINAL_VALIDATION_DIR
            / "target_duplicate_keys.csv",
    }

    for name, dataframe in [
        ("FEATURES", features),
        ("TARGET", target),
        ("DEPLOYMENT", deployment),
    ]:

        print()
        print(
            f"DATASET: {name}"
        )

        missing_keys = [
            column
            for column in key_columns
            if column not in dataframe.columns
        ]

        if missing_keys:

            results.append(
                result_row(
                    "KEY STRUCTURE",
                    f"{name} KEY COLUMNS",
                    "FAIL",
                    details=str(missing_keys)
                )
            )

            print(
                "KEY COLUMNS: FAIL",
                missing_keys
            )

            continue

        # --------------------------------------------------------
        # Null key rate
        # --------------------------------------------------------

        null_key_mask = (
            dataframe[key_columns]
            .isna()
            .any(axis=1)
        )

        null_key_count = int(
            null_key_mask.sum()
        )

        null_key_rate = (
            null_key_count
            /
            len(dataframe)
            if len(dataframe) > 0
            else 0
        )

        if name == "DEPLOYMENT":

            status = (
                "PASS"
                if null_key_count == 0
                else "FAIL"
            )

        else:

            # Feature and target datasets can contain
            # missing structural keys. This is a review item
            # unless it blocks deployment.
            status = (
                "PASS"
                if null_key_count == 0
                else "REVIEW"
            )

        results.append(
            result_row(
                "KEY STRUCTURE",
                f"{name} NULL KEY ROWS",
                status,
                f"rate={null_key_rate:.6f}"
            )
        )

        print(
            f"{name} NULL KEY ROWS:",
            status,
            f"rate={null_key_rate:.6f}"
        )

        # --------------------------------------------------------
        # Safe key generation
        # --------------------------------------------------------

        keys = make_key_frame(
            dataframe,
            key_columns
        )

        key_counts = (
            keys.value_counts()
        )

        repeated_keys = (
            key_counts[
                key_counts > 1
            ]
        )

        repeated_group_count = int(
            len(repeated_keys)
        )

        if name == "DEPLOYMENT":

            # Deployment should contain one record per
            # subdivision/year/month.
            unique_pass = (
                repeated_group_count == 0
            )

            status = (
                "PASS"
                if unique_pass
                else "FAIL"
            )

        else:

            # Historical feature/target datasets may have
            # repeated structures. Keep as review.
            status = (
                "PASS"
                if repeated_group_count == 0
                else "REVIEW"
            )

        results.append(
            result_row(
                "KEY STRUCTURE",
                f"{name} UNIQUE KEY",
                status,
                f"groups={repeated_group_count}"
            )
        )

        print(
            f"{name} UNIQUE KEY:",
            status,
            f"groups={repeated_group_count}"
        )

        # --------------------------------------------------------
        # Save duplicate keys
        # --------------------------------------------------------

        if (
            repeated_group_count > 0
            and
            name in duplicate_outputs
        ):

            duplicate_file = (
                duplicate_outputs[name]
            )

            temp = dataframe.copy()

            temp["_validation_key"] = keys

            duplicate_temp = (
                temp[
                    temp["_validation_key"]
                    .isin(
                        repeated_keys.index
                    )
                ]
                .copy()
            )

            duplicate_temp.to_csv(
                duplicate_file,
                index=False
            )

            print(
                "DUPLICATE KEY FILE:",
                duplicate_file
            )

        # --------------------------------------------------------
        # Month validity
        # --------------------------------------------------------

        if "month" in dataframe.columns:

            months = normalize_month(
                dataframe["month"]
            )

            invalid_months = (
                ~months.between(
                    1,
                    12
                )
            )

            invalid_count = int(
                invalid_months.sum()
            )

            status = (
                "PASS"
                if invalid_count == 0
                else (
                    "FAIL"
                    if name == "DEPLOYMENT"
                    else "REVIEW"
                )
            )

            results.append(
                result_row(
                    "KEY STRUCTURE",
                    f"{name} INVALID MONTH ROWS",
                    status,
                    invalid_count
                )
            )

            print(
                f"{name} INVALID MONTH ROWS:",
                status,
                invalid_count
            )

    key_df = pd.DataFrame(
        results
    )

    key_df.to_csv(
        KEY_VALIDATION_FILE,
        index=False
    )

    print()
    print(
        "KEY VALIDATION SAVED:",
        KEY_VALIDATION_FILE
    )

    return results


# ================================================================
# 3. LEAKAGE VALIDATION
# ================================================================

def validate_leakage(
    features
):

    print()
    print("=" * 70)
    print("3. LEAKAGE VALIDATION")
    print("=" * 70)

    results = []

    # ------------------------------------------------------------
    # Explicit target
    # ------------------------------------------------------------

    explicit_target = (
        TARGET_COLUMN in features.columns
    )

    status = (
        "FAIL"
        if explicit_target
        else "PASS"
    )

    results.append(
        result_row(
            "LEAKAGE",
            "EXPLICIT TARGET IN FEATURES",
            status,
            details=(
                "Target present"
                if explicit_target
                else "Target absent"
            )
        )
    )

    print(
        "EXPLICIT TARGET IN FEATURES:",
        status
    )

    # ------------------------------------------------------------
    # Future feature names
    # ------------------------------------------------------------

    future_tokens = [
        "future",
        "next_month",
        "next_3m",
        "next_6m",
        "future_rainfall",
        "lead_",
        "_lead",
    ]

    future_features = []

    for column in features.columns:

        lower = str(
            column
        ).lower()

        if any(
            token in lower
            for token in future_tokens
        ):

            future_features.append(
                column
            )

    status = (
        "PASS"
        if len(future_features) == 0
        else "FAIL"
    )

    results.append(
        result_row(
            "LEAKAGE",
            "EXPLICIT FUTURE FEATURES",
            status,
            details=str(
                future_features
            )
        )
    )

    print(
        "EXPLICIT FUTURE FEATURES:",
        status,
        future_features
    )

    # ------------------------------------------------------------
    # Target-like names
    # ------------------------------------------------------------

    target_like = []

    for column in features.columns:

        lower = str(
            column
        ).lower()

        if (
            "target" in lower
            or "label" in lower
            or "outcome" in lower
        ):

            if column != TARGET_COLUMN:

                target_like.append(
                    column
                )

    # Target-like variables are REVIEW, not automatically
    # production failures, because some may be legitimate
    # intermediate variables.
    status = (
        "PASS"
        if len(target_like) == 0
        else "REVIEW"
    )

    results.append(
        result_row(
            "LEAKAGE",
            "TARGET-LIKE FEATURE NAMES",
            status,
            details=str(
                target_like
            )
        )
    )

    print(
        "TARGET-LIKE FEATURE NAMES:",
        status,
        target_like
    )

    pd.DataFrame(
        results
    ).to_csv(
        LEAKAGE_VALIDATION_FILE,
        index=False
    )

    return results


# ================================================================
# 4. TEMPORAL VALIDATION
# ================================================================

def validate_temporal(
    features,
    target,
    deployment
):

    print()
    print("=" * 70)
    print("4. TEMPORAL VALIDATION")
    print("=" * 70)

    results = []

    datasets = [
        ("FEATURES", features),
        ("TARGET", target),
        ("DEPLOYMENT", deployment),
    ]

    for name, dataframe in datasets:

        if "year" not in dataframe.columns:

            results.append(
                result_row(
                    "TEMPORAL",
                    f"{name} YEAR VALIDITY",
                    "FAIL",
                    details="year column missing"
                )
            )

            continue

        years = normalize_year(
            dataframe["year"]
        )

        invalid = (
            years.isna()
            |
            (years < 1800)
            |
            (years > 2100)
        )

        invalid_count = int(
            invalid.sum()
        )

        status = (
            "PASS"
            if invalid_count == 0
            else "FAIL"
        )

        results.append(
            result_row(
                "TEMPORAL",
                f"{name} YEAR VALIDITY",
                status,
                invalid_count
            )
        )

        print(
            f"{name} YEAR VALIDITY:",
            status
        )

        if invalid_count == 0:

            min_year = int(
                years.min()
            )

            max_year = int(
                years.max()
            )

            results.append(
                result_row(
                    "TEMPORAL",
                    f"{name} YEAR RANGE",
                    "PASS",
                    f"{min_year}-{max_year}"
                )
            )

            print(
                f"{name} YEAR RANGE:",
                "PASS",
                f"{min_year}-{max_year}"
            )

        else:

            results.append(
                result_row(
                    "TEMPORAL",
                    f"{name} YEAR RANGE",
                    "FAIL"
                )
            )

    temporal_df = pd.DataFrame(
        results
    )

    temporal_df.to_csv(
        TEMPORAL_VALIDATION_FILE,
        index=False
    )

    return results


# ================================================================
# 5. MONTH VALIDATION
# ================================================================

def validate_month(
    deployment
):

    print()
    print("=" * 70)
    print("5. MONTH VALIDATION")
    print("=" * 70)

    months = normalize_month(
        deployment["month"]
    )

    invalid_mask = (
        ~months.between(
            1,
            12
        )
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    invalid_rate = (
        invalid_count
        /
        len(deployment)
        if len(deployment) > 0
        else 0
    )

    invalid_values = sorted(
        months[
            invalid_mask
        ]
        .dropna()
        .unique()
        .tolist()
    )

    zero_count = int(
        (
            months == 0
        ).sum()
    )

    print(
        "INVALID MONTH COUNT:",
        invalid_count
    )

    print(
        "INVALID MONTH RATE:",
        f"{invalid_rate:.6f}"
    )

    print(
        "INVALID MONTH VALUES:",
        invalid_values
    )

    print(
        "MONTH ZERO COUNT:",
        zero_count
    )

    status = (
        "PASS"
        if invalid_count == 0
        else "FAIL"
    )

    print(
        "MONTH VALIDITY:",
        status
    )

    results = [
        result_row(
            "MONTH",
            "INVALID MONTH COUNT",
            "PASS"
            if invalid_count == 0
            else "FAIL",
            invalid_count
        ),
        result_row(
            "MONTH",
            "INVALID MONTH RATE",
            "PASS"
            if invalid_rate == 0
            else "FAIL",
            f"{invalid_rate:.6f}"
        ),
        result_row(
            "MONTH",
            "INVALID MONTH VALUES",
            "PASS"
            if invalid_count == 0
            else "FAIL",
            str(invalid_values)
        ),
        result_row(
            "MONTH",
            "MONTH ZERO RECORDS",
            "PASS"
            if zero_count == 0
            else "FAIL",
            zero_count
        ),
    ]

    pd.DataFrame(
        results
    ).to_csv(
        MONTH_VALIDATION_FILE,
        index=False
    )

    return results


# ================================================================
# 6. SEASON VALIDATION
# ================================================================

def validate_season(
    deployment
):

    print()
    print("=" * 70)
    print("6. SEASON VALIDATION")
    print("=" * 70)

    df = deployment.copy()

    months = normalize_month(
        df["month"]
    )

    derived_season = (
        months.apply(
            month_to_season
        )
    )

    # ------------------------------------------------------------
    # Existing season
    # ------------------------------------------------------------

    existing_season = (
        df["season"]
        .astype("string")
        .fillna("UNKNOWN")
        .str.strip()
        .str.upper()
    )

    unknown_existing = int(
        existing_season
        .eq("UNKNOWN")
        .sum()
    )

    # ------------------------------------------------------------
    # Domain
    # ------------------------------------------------------------

    invalid_domain = (
        ~existing_season.isin(
            EXPECTED_SEASONS
        )
    )

    invalid_domain_count = int(
        invalid_domain.sum()
    )

    # ------------------------------------------------------------
    # Month-season consistency
    # ------------------------------------------------------------

    consistency_mask = (
        existing_season
        ==
        derived_season
    )

    inconsistent_count = int(
        (~consistency_mask).sum()
    )

    # ------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------

    season_pass = (
        unknown_existing == 0
        and
        invalid_domain_count == 0
        and
        inconsistent_count == 0
    )

    status = (
        "PASS"
        if season_pass
        else "FAIL"
    )

    print(
        "UNKNOWN SEASONS:",
        unknown_existing
    )

    print(
        "INVALID SEASON VALUES:",
        invalid_domain_count
    )

    print(
        "MONTH/SEASON INCONSISTENCIES:",
        inconsistent_count
    )

    print()
    print(
        "SEASON DISTRIBUTION:"
    )

    print(
        existing_season
        .value_counts()
    )

    results = [
        result_row(
            "SEASON",
            "UNKNOWN SEASONS",
            "PASS"
            if unknown_existing == 0
            else "FAIL",
            unknown_existing
        ),
        result_row(
            "SEASON",
            "SEASON DOMAIN",
            "PASS"
            if invalid_domain_count == 0
            else "FAIL",
            invalid_domain_count
        ),
        result_row(
            "SEASON",
            "MONTH SEASON CONSISTENCY",
            "PASS"
            if inconsistent_count == 0
            else "FAIL",
            inconsistent_count
        ),
    ]

    pd.DataFrame(
        results
    ).to_csv(
        SEASON_VALIDATION_FILE,
        index=False
    )

    return results


# ================================================================
# 7. MISSING VALUE VALIDATION
# ================================================================

def validate_missing_values(
    features,
    target,
    deployment
):

    print()
    print("=" * 70)
    print("7. MISSING VALUE VALIDATION")
    print("=" * 70)

    results = []

    datasets = [
        ("FEATURES", features),
        ("TARGET", target),
        ("DEPLOYMENT", deployment),
    ]

    for name, dataframe in datasets:

        missing_rate = (
            dataframe
            .isna()
            .mean()
        )

        maximum_rate = float(
            missing_rate.max()
        )

        worst_column = (
            missing_rate
            .idxmax()
        )

        # Missing values are REVIEW rather than automatic
        # failure because rainfall datasets can legitimately
        # contain missing environmental observations.
        status = (
            "PASS"
            if maximum_rate == 0
            else "REVIEW"
        )

        print(
            f"{name}:",
            status,
            f"maximum_rate={maximum_rate:.6f}",
            f"column={worst_column}"
        )

        results.append(
            result_row(
                "MISSING VALUES",
                f"{name} MISSING VALUE RATE",
                status,
                f"{maximum_rate:.6f}",
                str(worst_column)
            )
        )

    missing_df = pd.DataFrame(
        results
    )

    missing_df.to_csv(
        MISSING_VALIDATION_FILE,
        index=False
    )

    return results


# ================================================================
# 8. PROBABILITY VALIDATION
# ================================================================

def validate_probability(
    deployment
):

    print()
    print("=" * 70)
    print("8. PROBABILITY VALIDATION")
    print("=" * 70)

    results = []

    if PROBABILITY_COLUMN not in deployment.columns:

        results.append(
            result_row(
                "PROBABILITY",
                "PROBABILITY COLUMN",
                "FAIL",
                details="Column missing"
            )
        )

        return results

    probability = safe_numeric(
        deployment[
            PROBABILITY_COLUMN
        ]
    )

    null_count = int(
        probability.isna().sum()
    )

    invalid_range = (
        probability.notna()
        &
        (
            (probability < 0)
            |
            (probability > 1)
        )
    )

    invalid_range_count = int(
        invalid_range.sum()
    )

    min_probability = (
        float(probability.min())
        if probability.notna().any()
        else np.nan
    )

    max_probability = (
        float(probability.max())
        if probability.notna().any()
        else np.nan
    )

    results.append(
        result_row(
            "PROBABILITY",
            "PROBABILITY COLUMN",
            "PASS",
            PROBABILITY_COLUMN
        )
    )

    results.append(
        result_row(
            "PROBABILITY",
            "PROBABILITY NULL",
            "PASS"
            if null_count == 0
            else "FAIL",
            null_count
        )
    )

    results.append(
        result_row(
            "PROBABILITY",
            "PROBABILITY RANGE",
            "PASS"
            if invalid_range_count == 0
            else "FAIL",
            invalid_range_count
        )
    )

    results.append(
        result_row(
            "PROBABILITY",
            "PROBABILITY MINIMUM",
            "PASS",
            min_probability
        )
    )

    results.append(
        result_row(
            "PROBABILITY",
            "PROBABILITY MAXIMUM",
            "PASS",
            max_probability
        )
    )

    print(
        "PROBABILITY COLUMN:",
        "PASS"
    )

    print(
        "PROBABILITY NULL:",
        null_count
    )

    print(
        "PROBABILITY RANGE:",
        "PASS"
        if invalid_range_count == 0
        else "FAIL"
    )

    print(
        "PROBABILITY MINIMUM:",
        min_probability
    )

    print(
        "PROBABILITY MAXIMUM:",
        max_probability
    )

    return results


# ================================================================
# 9. POLICY VALIDATION
# ================================================================

def validate_policy(
    deployment
):

    print()
    print("=" * 70)
    print("9. POLICY VALIDATION")
    print("=" * 70)

    results = []

    probability = safe_numeric(
        deployment[
            PROBABILITY_COLUMN
        ]
    )

    threshold = safe_numeric(
        deployment[
            "policy_threshold"
        ]
    )

    alert = safe_numeric(
        deployment[
            "risk_alert"
        ]
    )

    # ------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------

    threshold_valid = (
        threshold.notna().all()
        and
        threshold.between(
            0,
            1
        ).all()
    )

    status = (
        "PASS"
        if threshold_valid
        else "FAIL"
    )

    print(
        "POLICY THRESHOLD:",
        status,
        f"column={PROBABILITY_COLUMN}"
    )

    results.append(
        result_row(
            "POLICY",
            "POLICY THRESHOLD",
            status,
            f"column={PROBABILITY_COLUMN}"
        )
    )

    # ------------------------------------------------------------
    # Alert consistency
    #
    # The policy can use:
    #
    # probability >= threshold
    #
    # ------------------------------------------------------------

    expected_alert = (
        probability >= threshold
    ).astype(int)

    alert_numeric = (
        alert
        .fillna(-999)
        .astype(int)
    )

    consistency = (
        expected_alert
        ==
        alert_numeric
    )

    consistency_pass = (
        consistency.all()
    )

    consistency_errors = int(
        (~consistency).sum()
    )

    status = (
        "PASS"
        if consistency_pass
        else "FAIL"
    )

    print(
        "POLICY ALERT CONSISTENCY:",
        status
    )

    results.append(
        result_row(
            "POLICY",
            "POLICY ALERT CONSISTENCY",
            status,
            consistency_errors
        )
    )

    # ------------------------------------------------------------
    # Alert rate
    # ------------------------------------------------------------

    alert_rate = float(
        alert_numeric.mean()
    )

    alert_rate_pass = (
        MIN_ALERT_RATE
        <=
        alert_rate
        <=
        MAX_ALERT_RATE
    )

    status = (
        "PASS"
        if alert_rate_pass
        else "REVIEW"
    )

    print(
        "ALERT RATE:",
        status,
        f"{alert_rate:.6f}"
    )

    results.append(
        result_row(
            "POLICY",
            "ALERT RATE",
            status,
            f"{alert_rate:.6f}"
        )
    )

    pd.DataFrame(
        results
    ).to_csv(
        POLICY_VALIDATION_FILE,
        index=False
    )

    return results


# ================================================================
# 10. MODEL PERFORMANCE
# ================================================================

def calculate_binary_metrics(
    actual,
    probability,
    threshold
):

    actual = (
        safe_numeric(actual)
        .fillna(0)
        .astype(int)
        .to_numpy()
    )

    probability = (
        safe_numeric(probability)
        .fillna(0)
        .to_numpy()
    )

    prediction = (
        probability >= threshold
    ).astype(int)

    tp = int(
        (
            (actual == 1)
            &
            (prediction == 1)
        ).sum()
    )

    tn = int(
        (
            (actual == 0)
            &
            (prediction == 0)
        ).sum()
    )

    fp = int(
        (
            (actual == 0)
            &
            (prediction == 1)
        ).sum()
    )

    fn = int(
        (
            (actual == 1)
            &
            (prediction == 0)
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0
    )

    f1 = (
        2
        * precision
        * recall
        /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    # Brier
    brier = float(
        np.mean(
            (
                probability
                -
                actual
            ) ** 2
        )
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def safe_auc_metrics(
    actual,
    probability
):

    try:

        from sklearn.metrics import (
            average_precision_score,
            roc_auc_score,
        )

        actual = (
            safe_numeric(actual)
            .fillna(0)
            .astype(int)
        )

        probability = (
            safe_numeric(probability)
            .fillna(0)
        )

        pr_auc = float(
            average_precision_score(
                actual,
                probability
            )
        )

        roc_auc = float(
            roc_auc_score(
                actual,
                probability
            )
        )

        return (
            pr_auc,
            roc_auc
        )

    except Exception as exc:

        print(
            "AUC calculation failed:",
            exc
        )

        return (
            np.nan,
            np.nan
        )


def validate_performance(
    deployment
):

    print()
    print("=" * 70)
    print("10. MODEL PERFORMANCE")
    print("=" * 70)

    results = []

    actual = deployment[
        ACTUAL_COLUMN
    ]

    probability = deployment[
        PROBABILITY_COLUMN
    ]

    threshold = float(
        safe_numeric(
            deployment[
                "policy_threshold"
            ]
        )
        .median()
    )

    pr_auc, roc_auc = (
        safe_auc_metrics(
            actual,
            probability
        )
    )

    metrics = calculate_binary_metrics(
        actual,
        probability,
        threshold
    )

    gates = {
        "PR-AUC": (
            pr_auc >= MIN_PR_AUC
        ),
        "ROC-AUC": (
            roc_auc >= MIN_ROC_AUC
        ),
        "PRECISION": (
            metrics["precision"]
            >=
            MIN_PRECISION
        ),
        "RECALL": (
            metrics["recall"]
            >=
            MIN_RECALL
        ),
        "F1": (
            metrics["f1"]
            >=
            MIN_F1
        ),
        "BRIER SCORE": (
            metrics["brier"]
            <=
            MAX_BRIER
        ),
    }

    values = {
        "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc,
        "PRECISION": metrics["precision"],
        "RECALL": metrics["recall"],
        "F1": metrics["f1"],
        "BRIER SCORE": metrics["brier"],
    }

    for metric, passed in gates.items():

        status = (
            "PASS"
            if passed
            else "FAIL"
        )

        results.append(
            result_row(
                "PERFORMANCE",
                metric,
                status,
                values[metric]
            )
        )

        print(
            f"{metric}:",
            status,
            f"{values[metric]:.6f}"
        )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    print()
    print(
        "CONFUSION MATRIX"
    )

    print(
        "[[TN FP]"
    )

    print(
        f" [{metrics['tn']} "
        f"{metrics['fp']}]"
    )

    print(
        f" [{metrics['fn']} "
        f"{metrics['tp']}]]"
    )

    return results


# ================================================================
# 11. DRIFT VALIDATION
# ================================================================

def validate_drift():

    print()
    print("=" * 70)
    print("11. DRIFT VALIDATION")
    print("=" * 70)

    results = []

    if not DRIFT_FILE.exists():

        print(
            "DRIFT FILE NOT FOUND:",
            DRIFT_FILE
        )

        results.append(
            result_row(
                "DRIFT",
                "DRIFT FILE",
                "REVIEW",
                details="File not found"
            )
        )

        return results

    drift = pd.read_csv(
        DRIFT_FILE,
        low_memory=False
    )

    print(
        "DRIFT FILE:",
        DRIFT_FILE
    )

    print(
        "DRIFT ROWS:",
        len(drift)
    )

    # ------------------------------------------------------------
    # Detect PSI column
    # ------------------------------------------------------------

    psi_candidates = [
        column
        for column in drift.columns
        if "psi" in str(column).lower()
    ]

    if not psi_candidates:

        print(
            "PSI COLUMN: NOT FOUND"
        )

        results.append(
            result_row(
                "DRIFT",
                "PSI COLUMN",
                "REVIEW",
                details="No PSI column detected"
            )
        )

        return results

    psi_column = psi_candidates[0]

    psi = safe_numeric(
        drift[psi_column]
    )

    maximum_psi = float(
        psi.max()
    )

    significant_count = int(
        (
            psi
            >=
            SIGNIFICANT_PSI_THRESHOLD
        ).sum()
    )

    moderate_count = int(
        (
            (psi >= MODERATE_PSI_THRESHOLD)
            &
            (psi < SIGNIFICANT_PSI_THRESHOLD)
        ).sum()
    )

    print(
        "PSI COLUMN:",
        psi_column
    )

    print(
        "MAXIMUM PSI:",
        f"{maximum_psi:.6f}"
    )

    print(
        "SIGNIFICANT DRIFT FEATURES:",
        significant_count
    )

    print(
        "MODERATE DRIFT FEATURES:",
        moderate_count
    )

    # Significant drift is REVIEW rather than hard FAIL.
    results.append(
        result_row(
            "DRIFT",
            "MAXIMUM PSI",
            "REVIEW"
            if maximum_psi
            >= SIGNIFICANT_PSI_THRESHOLD
            else "PASS",
            f"{maximum_psi:.6f}"
        )
    )

    results.append(
        result_row(
            "DRIFT",
            "SIGNIFICANT DRIFT FEATURES",
            "REVIEW"
            if significant_count > 0
            else "PASS",
            f"threshold={SIGNIFICANT_PSI_THRESHOLD}"
        )
    )

    results.append(
        result_row(
            "DRIFT",
            "MODERATE DRIFT FEATURES",
            "PASS",
            f"threshold={MODERATE_PSI_THRESHOLD}"
        )
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    drift_output = pd.DataFrame(
        results
    )

    drift_output.to_csv(
        DRIFT_VALIDATION_FILE,
        index=False
    )

    return results


# ================================================================
# 12. PRODUCTION SCHEMA
# ================================================================

def validate_production_schema(
    deployment
):

    print()
    print("=" * 70)
    print("12. PRODUCTION SCHEMA VALIDATION")
    print("=" * 70)

    missing = [
        column
        for column
        in REQUIRED_DEPLOYMENT_COLUMNS
        if column not in deployment.columns
    ]

    status = (
        "PASS"
        if len(missing) == 0
        else "FAIL"
    )

    details = (
        "All production columns present"
        if not missing
        else str(missing)
    )

    print(
        "DEPLOYMENT SCHEMA:",
        status,
        details
    )

    results = [
        result_row(
            "PRODUCTION SCHEMA",
            "DEPLOYMENT SCHEMA",
            status,
            details=details
        )
    ]

    pd.DataFrame(
        results
    ).to_csv(
        SCHEMA_VALIDATION_FILE,
        index=False
    )

    return results


# ================================================================
# 13. DEPLOYMENT ARTIFACT VALIDATION
# ================================================================

def validate_artifacts():

    print()
    print("=" * 70)
    print("13. DEPLOYMENT ARTIFACT VALIDATION")
    print("=" * 70)

    results = []

    for artifact_name, path in (
        EXPECTED_ARTIFACTS.items()
    ):

        exists = path.exists()

        status = (
            "PASS"
            if exists
            else "FAIL"
        )

        print(
            f"ARTIFACT {artifact_name}:",
            status
        )

        results.append(
            result_row(
                "ARTIFACTS",
                f"ARTIFACT {artifact_name}",
                status,
                str(path)
            )
        )

    artifact_df = pd.DataFrame(
        results
    )

    artifact_df.to_csv(
        ARTIFACT_VALIDATION_FILE,
        index=False
    )

    return results


# ================================================================
# 14. REPRODUCIBILITY VALIDATION
# ================================================================

def validate_reproducibility():

    print()
    print("=" * 70)
    print("14. REPRODUCIBILITY VALIDATION")
    print("=" * 70)

    results = []

    script_dir = (
        PROJECT_ROOT
        / "src"
        / "ml"
    )

    for script_name in (
        EXPECTED_SCRIPTS
    ):

        path = (
            script_dir
            / script_name
        )

        exists = path.exists()

        status = (
            "PASS"
            if exists
            else "FAIL"
        )

        print(
            f"SCRIPT {script_name}:",
            status
        )

        results.append(
            result_row(
                "REPRODUCIBILITY",
                f"SCRIPT {script_name}",
                status,
                str(path)
            )
        )

    reproducibility_df = pd.DataFrame(
        results
    )

    reproducibility_df.to_csv(
        REPRODUCIBILITY_FILE,
        index=False
    )

    return results


# ================================================================
# 15. FINAL DECISION
# ================================================================

def determine_final_decision(
    all_results
):

    hard_failures = [
        row
        for row in all_results
        if row["status"] == "FAIL"
    ]

    reviews = [
        row
        for row in all_results
        if row["status"] == "REVIEW"
    ]

    if len(hard_failures) > 0:

        decision = "NO_GO"

    elif len(reviews) > 0:

        decision = "CONDITIONAL_GO"

    else:

        decision = "GO"

    return (
        decision,
        hard_failures,
        reviews
    )


# ================================================================
# FINAL CSV REPORT
# ================================================================

def save_final_report(
    all_results
):

    report_df = pd.DataFrame(
        all_results
    )

    report_df.to_csv(
        FINAL_REPORT_FILE,
        index=False
    )

    return report_df


# ================================================================
# JSON SUMMARY
# ================================================================

def save_json_summary(
    decision,
    failures,
    reviews,
    deployment_file,
    deployment
):

    summary = {
        "project":
            "Bharat Earth",

        "validation_version":
            "3.13",

        "timestamp":
            datetime.now().isoformat(),

        "project_root":
            str(PROJECT_ROOT),

        "deployment_file":
            str(deployment_file),

        "deployment_rows":
            int(len(deployment)),

        "decision":
            decision,

        "failure_count":
            len(failures),

        "review_count":
            len(reviews),

        "failures":
            failures,

        "reviews":
            reviews,

        "status":
            (
                "PRODUCTION_READY"
                if decision == "GO"
                else (
                    "CONDITIONAL_PRODUCTION_READY"
                    if decision
                    == "CONDITIONAL_GO"
                    else "NOT_PRODUCTION_READY"
                )
            ),
    }

    with open(
        SUMMARY_JSON_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            default=str
        )

    return summary


# ================================================================
# DEPLOYMENT MANIFEST
# ================================================================

def save_manifest(
    decision,
    deployment_file
):

    manifest = {
        "project":
            "Bharat Earth",

        "version":
            "3.13",

        "deployment_file":
            str(deployment_file),

        "deployment_file_exists":
            deployment_file.exists(),

        "final_decision":
            decision,

        "generated_at":
            datetime.now().isoformat(),

        "model":
            {
                "algorithm":
                    "XGBoost",

                "probability_column":
                    PROBABILITY_COLUMN,
            },

        "policy":
            {
                "threshold_source":
                    "deployment policy_threshold",

                "alert_column":
                    "risk_alert",
            },

        "month_encoding":
            "calendar months 1-12",

        "season_mapping":
            {
                "WINTER":
                    [12, 1, 2],

                "PRE_MONSOON":
                    [3, 4, 5],

                "MONSOON":
                    [6, 7, 8, 9],

                "POST_MONSOON":
                    [10, 11],
            },
    }

    with open(
        MANIFEST_JSON_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2
        )


# ================================================================
# MARKDOWN REPORT
# ================================================================

def save_markdown_report(
    decision,
    failures,
    reviews,
    deployment_file,
    report_df
):

    lines = []

    lines.append(
        "# Bharat Earth - Production Readiness Report"
    )

    lines.append("")

    lines.append(
        f"**Validation Version:** 3.13"
    )

    lines.append("")

    lines.append(
        f"**Generated:** "
        f"{datetime.now().isoformat()}"
    )

    lines.append("")

    lines.append(
        f"**Deployment File:** "
        f"`{deployment_file}`"
    )

    lines.append("")

    lines.append(
        f"## Final Decision: `{decision}`"
    )

    lines.append("")

    if decision == "GO":

        lines.append(
            "All hard validation gates passed."
        )

    elif decision == "CONDITIONAL_GO":

        lines.append(
            "All hard validation gates passed, "
            "but review items remain."
        )

    else:

        lines.append(
            "One or more hard validation gates failed."
        )

    lines.append("")

    lines.append(
        f"**Failures:** {len(failures)}"
    )

    lines.append("")

    lines.append(
        f"**Review Items:** {len(reviews)}"
    )

    lines.append("")

    # ------------------------------------------------------------
    # Failures
    # ------------------------------------------------------------

    if failures:

        lines.append(
            "## Hard Failures"
        )

        lines.append("")

        for failure in failures:

            lines.append(
                f"- **{failure['check']}**: "
                f"{failure['value']} "
                f"{failure['details']}"
            )

        lines.append("")

    # ------------------------------------------------------------
    # Reviews
    # ------------------------------------------------------------

    if reviews:

        lines.append(
            "## Review Items"
        )

        lines.append("")

        for review in reviews:

            lines.append(
                f"- **{review['check']}**: "
                f"{review['value']} "
                f"{review['details']}"
            )

        lines.append("")

    # ------------------------------------------------------------
    # Section summary
    # ------------------------------------------------------------

    lines.append(
        "## Validation Summary"
    )

    lines.append("")

    section_summary = (
        report_df
        .groupby(
            [
                "section",
                "status"
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    for _, row in (
        section_summary.iterrows()
    ):

        lines.append(
            f"- {row['section']} / "
            f"{row['status']}: "
            f"{row['count']}"
        )

    lines.append("")

    MARKDOWN_REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ================================================================
# EXCEL REPORT
# ================================================================

def save_excel_report(
    report_df
):

    try:

        with pd.ExcelWriter(
            EXCEL_REPORT_FILE,
            engine="openpyxl"
        ) as writer:

            report_df.to_excel(
                writer,
                sheet_name="Final Validation",
                index=False
            )

            # Section sheets
            for section in (
                report_df[
                    "section"
                ]
                .dropna()
                .unique()
            ):

                safe_name = str(
                    section
                )[:31]

                section_df = report_df[
                    report_df[
                        "section"
                    ]
                    == section
                ]

                section_df.to_excel(
                    writer,
                    sheet_name=safe_name,
                    index=False
                )

        print(
            "EXCEL REPORT SAVED:",
            EXCEL_REPORT_FILE
        )

    except ImportError:

        print()
        print(
            "WARNING: openpyxl is not installed."
        )

        print(
            "Excel report skipped."
        )

        print(
            "Install with:"
        )

        print(
            "python -m pip install openpyxl"
        )


# ================================================================
# PRINT FINAL SUMMARY
# ================================================================

def print_final_summary(
    decision,
    failures,
    reviews
):

    print()
    print("=" * 70)
    print("15. FINAL PRODUCTION DECISION")
    print("=" * 70)

    print()
    print(
        "FINAL DECISION:",
        decision
    )

    print(
        "TOTAL FAILURES:",
        len(failures)
    )

    print(
        "TOTAL REVIEW ITEMS:",
        len(reviews)
    )

    if failures:

        print()
        print(
            "HARD FAILURES:"
        )

        for failure in failures:

            print(
                "-",
                failure["check"],
                ":",
                failure["value"],
                failure["details"]
            )

    if reviews:

        print()
        print(
            "REVIEW ITEMS:"
        )

        for review in reviews:

            print(
                "-",
                review["check"],
                ":",
                review["value"],
                review["details"]
            )

    print()

    if decision == "GO":

        print(
            "PRODUCTION STATUS:"
        )

        print(
            "READY FOR PRODUCTION"
        )

    elif decision == "CONDITIONAL_GO":

        print(
            "PRODUCTION STATUS:"
        )

        print(
            "CONDITIONAL - REVIEW ITEMS REQUIRE "
            "OPERATIONAL SIGN-OFF"
        )

    else:

        print(
            "PRODUCTION STATUS:"
        )

        print(
            "NOT READY FOR PRODUCTION"
        )


# ================================================================
# MAIN
# ================================================================

def main():

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    (
        features,
        target,
        deployment,
        deployment_file
    ) = load_data()

    all_results = []

    # ------------------------------------------------------------
    # 1. Integrity
    # ------------------------------------------------------------

    all_results.extend(
        validate_dataset_integrity(
            features,
            target,
            deployment
        )
    )

    # ------------------------------------------------------------
    # 2. Key structure
    # ------------------------------------------------------------

    all_results.extend(
        validate_key_structure(
            features,
            target,
            deployment
        )
    )

    # ------------------------------------------------------------
    # 3. Leakage
    # ------------------------------------------------------------

    all_results.extend(
        validate_leakage(
            features
        )
    )

    # ------------------------------------------------------------
    # 4. Temporal
    # ------------------------------------------------------------

    all_results.extend(
        validate_temporal(
            features,
            target,
            deployment
        )
    )

    # ------------------------------------------------------------
    # 5. Month
    # ------------------------------------------------------------

    all_results.extend(
        validate_month(
            deployment
        )
    )

    # ------------------------------------------------------------
    # 6. Season
    # ------------------------------------------------------------

    all_results.extend(
        validate_season(
            deployment
        )
    )

    # ------------------------------------------------------------
    # 7. Missing
    # ------------------------------------------------------------

    all_results.extend(
        validate_missing_values(
            features,
            target,
            deployment
        )
    )

    # ------------------------------------------------------------
    # 8. Probability
    # ------------------------------------------------------------

    all_results.extend(
        validate_probability(
            deployment
        )
    )

    # ------------------------------------------------------------
    # 9. Policy
    # ------------------------------------------------------------

    all_results.extend(
        validate_policy(
            deployment
        )
    )

    # ------------------------------------------------------------
    # 10. Performance
    # ------------------------------------------------------------

    all_results.extend(
        validate_performance(
            deployment
        )
    )

    # ------------------------------------------------------------
    # 11. Drift
    # ------------------------------------------------------------

    all_results.extend(
        validate_drift()
    )

    # ------------------------------------------------------------
    # 12. Production schema
    # ------------------------------------------------------------

    all_results.extend(
        validate_production_schema(
            deployment
        )
    )

    # ------------------------------------------------------------
    # 13. Artifacts
    # ------------------------------------------------------------

    all_results.extend(
        validate_artifacts()
    )

    # ------------------------------------------------------------
    # 14. Reproducibility
    # ------------------------------------------------------------

    all_results.extend(
        validate_reproducibility()
    )

    # ------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------

    (
        decision,
        failures,
        reviews
    ) = determine_final_decision(
        all_results
    )

    # ------------------------------------------------------------
    # Save final report
    # ------------------------------------------------------------

    report_df = save_final_report(
        all_results
    )

    # ------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------

    save_json_summary(
        decision,
        failures,
        reviews,
        deployment_file,
        deployment
    )

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------

    save_manifest(
        decision,
        deployment_file
    )

    # ------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------

    save_markdown_report(
        decision,
        failures,
        reviews,
        deployment_file,
        report_df
    )

    # ------------------------------------------------------------
    # Excel
    # ------------------------------------------------------------

    save_excel_report(
        report_df
    )

    # ------------------------------------------------------------
    # Final console
    # ------------------------------------------------------------

    print_final_summary(
        decision,
        failures,
        reviews
    )

    print()
    print("=" * 70)
    print(
        "SAVING FINAL REPORTS"
    )
    print("=" * 70)

    print(
        "CSV:",
        FINAL_REPORT_FILE
    )

    print(
        "JSON:",
        SUMMARY_JSON_FILE
    )

    print(
        "MANIFEST:",
        MANIFEST_JSON_FILE
    )

    print(
        "MARKDOWN:",
        MARKDOWN_REPORT_FILE
    )

    print(
        "EXCEL:",
        EXCEL_REPORT_FILE
    )

    print()
    print("=" * 70)
    print(
        "3.13 FINAL PROJECT VALIDATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "FINAL DECISION:",
        decision
    )

    print(
        "FAILURES:",
        len(failures)
    )

    print(
        "REVIEW ITEMS:",
        len(reviews)
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()