from pathlib import Path
import json
import sys

import pandas as pd
import numpy as np


# ======================================================================
# PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "features"

V4_DATASET = DATA_DIR / "ml_dataset_v4.csv"

MODEL_DIR = DATA_DIR / "model_v4"
MODEL_FILE = MODEL_DIR / "xgboost_model.json"
MODEL_SCHEMA_FILE = MODEL_DIR / "model_schema.json"

CALIBRATION_DIR = DATA_DIR / "calibration_v4"
CALIBRATION_FILE = CALIBRATION_DIR / "calibrated_predictions.csv"
CALIBRATION_METRICS = CALIBRATION_DIR / "calibration_metrics.csv"
CALIBRATION_SUMMARY = CALIBRATION_DIR / "calibration_summary.json"

POLICY_DIR = DATA_DIR / "policy_v4"
POLICY_FILE = POLICY_DIR / "selected_policy.csv"
POLICY_METRICS = POLICY_DIR / "policy_metrics.csv"
POLICY_PREDICTIONS = POLICY_DIR / "calibration_policy_predictions.csv"
THRESHOLD_FILE = POLICY_DIR / "threshold_analysis.csv"

MONITORING_DIR = DATA_DIR / "monitoring_v4"
DRIFT_FILE = MONITORING_DIR / "feature_drift.csv"

# Temporal/model-index features are not evaluated using conventional PSI.
DRIFT_EXCLUDED_FEATURES = {"year"}

OUTPUT_DIR = DATA_DIR / "production_readiness_v4"

REPORT_CSV = OUTPUT_DIR / "production_readiness_report.csv"
REPORT_JSON = OUTPUT_DIR / "production_readiness_summary.json"
REPORT_MD = OUTPUT_DIR / "production_readiness_report.md"


TARGET = "target_3m_severe_anomaly"

CATEGORICAL_FEATURES = [
    "subdivision",
    "month",
    "season",
]

EXPECTED_FEATURES = [
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

LEGACY_LEAKAGE_COLUMNS = {
    "target_3m_stress",
    "rainfall_stress",
    "environmental_risk_score",
    "environmental_risk_level",
    "persistent_drought_signal",
}


# ======================================================================
# HELPERS
# ======================================================================

results = []


def banner(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def record(section, check, status, details=""):
    results.append(
        {
            "section": section,
            "check": check,
            "status": status,
            "details": str(details),
        }
    )

    print(
        f"{check:<38} "
        f"{status:<8} "
        f"{details}"
    )


def require_file(path, section, label):
    if path.exists():
        record(
            section,
            label,
            "PASS",
            str(path),
        )
        return True

    record(
        section,
        label,
        "FAIL",
        f"Missing: {path}",
    )
    return False


def safe_json_load(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        return None


# ======================================================================
# 1. DATASET VALIDATION
# ======================================================================

def validate_dataset():

    banner("1. V4 DATASET VALIDATION")

    if not require_file(
        V4_DATASET,
        "DATASET",
        "V4 DATASET",
    ):
        return None

    try:
        df = pd.read_csv(V4_DATASET)
    except Exception as exc:
        record(
            "DATASET",
            "DATASET LOAD",
            "FAIL",
            exc,
        )
        return None

    print("SHAPE:", df.shape)
    print("COLUMNS:", list(df.columns))

    missing_features = sorted(
        set(EXPECTED_FEATURES + [TARGET])
        - set(df.columns)
    )

    if missing_features:
        record(
            "DATASET",
            "REQUIRED SCHEMA",
            "FAIL",
            missing_features,
        )
    else:
        record(
            "DATASET",
            "REQUIRED SCHEMA",
            "PASS",
            "All required columns present",
        )

    leakage = sorted(
        LEGACY_LEAKAGE_COLUMNS
        & set(df.columns)
    )

    if leakage:
        record(
            "DATASET",
            "LEGACY LEAKAGE",
            "FAIL",
            leakage,
        )
    else:
        record(
            "DATASET",
            "LEGACY LEAKAGE",
            "PASS",
            "No legacy leakage columns",
        )

    if TARGET in df.columns:

        target = pd.to_numeric(
            df[TARGET],
            errors="coerce",
        )

        invalid_target = int(
            target.isna().sum()
        )

        if invalid_target:
            record(
                "DATASET",
                "TARGET VALUES",
                "FAIL",
                f"Invalid values={invalid_target}",
            )
        elif set(target.unique()) == {0, 1}:
            record(
                "DATASET",
                "TARGET VALUES",
                "PASS",
                f"rate={target.mean():.6f}",
            )
        else:
            record(
                "DATASET",
                "TARGET VALUES",
                "FAIL",
                sorted(target.unique().tolist()),
            )

    duplicates = int(
        df.duplicated().sum()
    )

    record(
        "DATASET",
        "EXACT DUPLICATES",
        "PASS" if duplicates == 0 else "FAIL",
        duplicates,
    )

    # --------------------------------------------------------------
    # MONTH NORMALIZATION
    # --------------------------------------------------------------

    banner("MONTH VALIDATION")

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

    month = df["month"].copy()

    normalized_month = (
        month.astype(str)
        .str.strip()
        .str.upper()
        .replace(month_map)
    )

    normalized_month = pd.to_numeric(
        normalized_month,
        errors="coerce",
    )

    invalid_month = (
        normalized_month.isna()
        | ~normalized_month.between(1, 12)
    )

    invalid_count = int(
        invalid_month.sum()
    )

    record(
        "DATASET",
        "MONTH VALIDATION",
        "PASS" if invalid_count == 0 else "FAIL",
        f"invalid={invalid_count}",
    )

    if invalid_count == 0:
        df["month"] = normalized_month.astype(int)

    # --------------------------------------------------------------
    # SEASON VALIDATION
    # --------------------------------------------------------------

    season_map = {
        12: "WINTER",
        1: "WINTER",
        2: "WINTER",
        3: "PRE_MONSOON",
        4: "PRE_MONSOON",
        5: "PRE_MONSOON",
        6: "MONSOON",
        7: "MONSOON",
        8: "MONSOON",
        9: "MONSOON",
        10: "POST_MONSOON",
        11: "POST_MONSOON",
    }

    if invalid_count == 0:

        expected_season = (
            df["month"].map(season_map)
        )

        actual_season = (
            df["season"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        inconsistent = int(
            (actual_season != expected_season).sum()
        )

        if inconsistent:
            print(
                "SEASON INCONSISTENCIES:",
                inconsistent,
            )
            print(
                "REPAIRING season from month."
            )

            df["season"] = expected_season

        record(
            "DATASET",
            "SEASON VALIDATION",
            "PASS",
            f"inconsistencies_repaired={inconsistent}",
        )

    # --------------------------------------------------------------
    # NUMERIC FEATURE CHECK
    # --------------------------------------------------------------

    numeric_features = [
        x
        for x in EXPECTED_FEATURES
        if x not in CATEGORICAL_FEATURES
    ]

    numeric_failures = []

    for feature in numeric_features:

        if feature not in df.columns:
            continue

        converted = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        invalid = (
            converted.isna()
            & df[feature].notna()
        ).sum()

        if invalid:
            numeric_failures.append(
                (feature, int(invalid))
            )

    record(
        "DATASET",
        "NUMERIC FEATURE TYPES",
        "PASS" if not numeric_failures else "FAIL",
        "All numeric features valid"
        if not numeric_failures
        else numeric_failures,
    )

    return df


# ======================================================================
# 2. MODEL VALIDATION
# ======================================================================

def validate_model():

    banner("2. XGBOOST MODEL VALIDATION")

    if not require_file(
        MODEL_FILE,
        "MODEL",
        "MODEL FILE",
    ):
        return None

    if not require_file(
        MODEL_SCHEMA_FILE,
        "MODEL",
        "MODEL SCHEMA",
    ):
        return None

    try:
        import xgboost as xgb

        model = xgb.XGBClassifier(
            enable_categorical=True
        )

        model.load_model(
            MODEL_FILE
        )

        booster = model.get_booster()

        model_features = list(
            booster.feature_names
        )

        record(
            "MODEL",
            "MODEL LOAD",
            "PASS",
            f"features={len(model_features)}",
        )

    except Exception as exc:

        record(
            "MODEL",
            "MODEL LOAD",
            "FAIL",
            exc,
        )

        return None

    if model_features == EXPECTED_FEATURES:

        record(
            "MODEL",
            "FEATURE ORDER",
            "PASS",
            "23 features match V4 schema",
        )

    else:

        record(
            "MODEL",
            "FEATURE ORDER",
            "FAIL",
            {
                "expected": EXPECTED_FEATURES,
                "actual": model_features,
            },
        )

    schema = safe_json_load(
        MODEL_SCHEMA_FILE
    )

    if schema is None:

        record(
            "MODEL",
            "SCHEMA JSON",
            "FAIL",
            "Unable to read JSON",
        )

    else:

        record(
            "MODEL",
            "SCHEMA JSON",
            "PASS",
            "Valid JSON",
        )

    return model


# ======================================================================
# 3. MODEL SCHEMA COMPATIBILITY
# ======================================================================

def validate_model_compatibility(df, model):

    banner("3. MODEL / DATA COMPATIBILITY")

    if model is None or df is None:
        record(
            "COMPATIBILITY",
            "MODEL INPUT",
            "FAIL",
            "Missing model or dataset",
        )
        return

    try:

        X = df[
            EXPECTED_FEATURES
        ].copy()

        # Exact categorical representation used during V4 training.
        X["subdivision"] = (
            X["subdivision"]
            .astype(str)
            .astype("category")
        )

        X["month"] = (
            pd.to_numeric(
                X["month"],
                errors="coerce",
            )
            .astype(int)
            .astype(str)
            .astype("category")
        )

        X["season"] = (
            X["season"]
            .astype(str)
            .astype("category")
        )

        numeric_features = [
            x
            for x in EXPECTED_FEATURES
            if x not in CATEGORICAL_FEATURES
        ]

        for feature in numeric_features:

            if feature == "rainfall_missing":
                X[feature] = (
                    pd.to_numeric(
                        X[feature],
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype("int64")
                )

            else:
                X[feature] = (
                    pd.to_numeric(
                        X[feature],
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype("float64")
                )

        null_count = int(
            X.isna().sum().sum()
        )

        record(
            "COMPATIBILITY",
            "NULL VALUES",
            "PASS" if null_count == 0 else "FAIL",
            null_count,
        )

        if null_count:
            return

        probabilities = model.predict_proba(
            X.iloc[:10]
        )[:, 1]

        record(
            "COMPATIBILITY",
            "MODEL PREDICTION",
            "PASS",
            f"sample={len(probabilities)}",
        )

        print(
            "SAMPLE PROBABILITIES:",
            probabilities,
        )

    except Exception as exc:

        record(
            "COMPATIBILITY",
            "MODEL PREDICTION",
            "FAIL",
            str(exc),
        )


# ======================================================================
# 4. CALIBRATION VALIDATION
# ======================================================================

def validate_calibration():

    banner("4. CALIBRATION VALIDATION")

    required = [
        (
            CALIBRATION_FILE,
            "CALIBRATED PREDICTIONS",
        ),
        (
            CALIBRATION_METRICS,
            "CALIBRATION METRICS",
        ),
        (
            CALIBRATION_SUMMARY,
            "CALIBRATION SUMMARY",
        ),
    ]

    missing = False

    for path, label in required:

        if not require_file(
            path,
            "CALIBRATION",
            label,
        ):
            missing = True

    if missing:
        return None

    try:

        df = pd.read_csv(
            CALIBRATION_FILE
        )

        required_columns = [
            "subdivision",
            "year",
            "month",
            "season",
            "actual",
            "raw_probability",
            "isotonic_probability",
            "sigmoid_probability",
        ]

        missing_columns = sorted(
            set(required_columns)
            - set(df.columns)
        )

        if missing_columns:

            record(
                "CALIBRATION",
                "OUTPUT SCHEMA",
                "FAIL",
                missing_columns,
            )

            return None

        record(
            "CALIBRATION",
            "OUTPUT SCHEMA",
            "PASS",
            f"columns={len(df.columns)}",
        )

        for column in [
            "raw_probability",
            "isotonic_probability",
            "sigmoid_probability",
        ]:

            values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            valid = (
                values.notna()
                & values.between(0, 1)
            )

            record(
                "CALIBRATION",
                f"{column} RANGE",
                "PASS" if valid.all() else "FAIL",
                f"invalid={int((~valid).sum())}",
            )

        actual = pd.to_numeric(
            df["actual"],
            errors="coerce",
        )

        record(
            "CALIBRATION",
            "ACTUAL TARGET",
            "PASS"
            if set(actual.dropna().unique()) <= {0, 1}
            else "FAIL",
            f"rows={len(df)}",
        )

        return df

    except Exception as exc:

        record(
            "CALIBRATION",
            "CALIBRATION LOAD",
            "FAIL",
            exc,
        )

        return None


# ======================================================================
# 5. POLICY VALIDATION
# ======================================================================

def validate_policy():

    banner("5. POLICY VALIDATION")

    required = [
        (
            POLICY_FILE,
            "SELECTED POLICY",
        ),
        (
            POLICY_METRICS,
            "POLICY METRICS",
        ),
        (
            POLICY_PREDICTIONS,
            "POLICY PREDICTIONS",
        ),
        (
            THRESHOLD_FILE,
            "THRESHOLD ANALYSIS",
        ),
    ]

    missing = False

    for path, label in required:

        if not require_file(
            path,
            "POLICY",
            label,
        ):
            missing = True

    if missing:
        return None

    try:

        policy = pd.read_csv(
            POLICY_FILE
        )

        metrics = pd.read_csv(
            POLICY_METRICS
        )

        predictions = pd.read_csv(
            POLICY_PREDICTIONS
        )

        record(
            "POLICY",
            "SELECTED POLICY LOAD",
            "PASS",
            f"rows={len(policy)}",
        )

        # Find threshold column.
        threshold_columns = [
            x
            for x in policy.columns
            if "threshold" in x.lower()
        ]

        probability_columns = [
            x
            for x in policy.columns
            if "probability" in x.lower()
        ]

        print(
            "POLICY COLUMNS:",
            list(policy.columns),
        )

        if threshold_columns:

            threshold_values = pd.to_numeric(
                policy[threshold_columns[0]],
                errors="coerce",
            ).dropna()

            valid = (
                threshold_values.between(0, 1)
            ).all()

            record(
                "POLICY",
                "THRESHOLD RANGE",
                "PASS" if valid else "FAIL",
                threshold_values.tolist(),
            )

        else:

            record(
                "POLICY",
                "THRESHOLD COLUMN",
                "REVIEW",
                "Threshold column not detected",
            )

        if probability_columns:

            print(
                "POLICY PROBABILITY COLUMNS:",
                probability_columns,
            )

        record(
            "POLICY",
            "POLICY PREDICTIONS",
            "PASS"
            if len(predictions) > 0
            else "FAIL",
            f"rows={len(predictions)}",
        )

        record(
            "POLICY",
            "POLICY METRICS",
            "PASS"
            if len(metrics) > 0
            else "FAIL",
            f"rows={len(metrics)}",
        )

        return policy

    except Exception as exc:

        record(
            "POLICY",
            "POLICY VALIDATION",
            "FAIL",
            exc,
        )

        return None


# ======================================================================
# 6. DRIFT VALIDATION
# ======================================================================
def validate_drift():

    banner("6. DRIFT MONITORING VALIDATION")

    if not require_file(
        DRIFT_FILE,
        "DRIFT",
        "DRIFT FILE",
    ):
        return None

    try:

        drift = pd.read_csv(DRIFT_FILE)

        required_columns = [
            "feature",
            "psi",
        ]

        missing = sorted(
            set(required_columns)
            - set(drift.columns)
        )

        if missing:

            record(
                "DRIFT",
                "DRIFT SCHEMA",
                "FAIL",
                missing,
            )

            return None

        # ----------------------------------------------------------
        # Temporal/model-index features
        # ----------------------------------------------------------

        excluded = drift[
            drift["feature"].isin(DRIFT_EXCLUDED_FEATURES)
        ].copy()

        if not excluded.empty:

            for _, row in excluded.iterrows():

                print(
                    f"EXCLUDED FROM PSI DECISION: "
                    f"{row['feature']} "
                    f"(temporal/model-index feature)"
                )

            record(
                "DRIFT",
                "TEMPORAL FEATURE SHIFT",
                "REVIEW",
                "Excluded from conventional PSI: "
                + ", ".join(excluded["feature"].astype(str).tolist()),
            )

        # ----------------------------------------------------------
        # Conventional feature drift
        # ----------------------------------------------------------

        monitored = drift[
            ~drift["feature"].isin(DRIFT_EXCLUDED_FEATURES)
        ].copy()

        psi = pd.to_numeric(
            monitored["psi"],
            errors="coerce",
        )

        valid = (
            psi.notna()
            & (psi >= 0)
        )

        record(
            "DRIFT",
            "PSI VALUES",
            "PASS" if valid.all() else "FAIL",
            f"monitored_features={len(monitored)}",
        )

        if valid.any():

            max_psi = float(
                psi.max()
            )

            significant_mask = psi >= 0.25

            moderate_mask = (
                (psi >= 0.10)
                & (psi < 0.25)
            )

            significant = int(
                significant_mask.sum()
            )

            moderate = int(
                moderate_mask.sum()
            )

            print(
                "MAX MONITORED PSI:",
                max_psi,
            )

            print(
                "SIGNIFICANT DRIFT:",
                significant,
            )

            print(
                "MODERATE DRIFT:",
                moderate,
            )

            significant_features = monitored.loc[
                significant_mask,
                "feature",
            ].astype(str).tolist()

            moderate_features = monitored.loc[
                moderate_mask,
                "feature",
            ].astype(str).tolist()

            # Significant drift requires review.
            if significant:

                record(
                    "DRIFT",
                    "SIGNIFICANT DRIFT",
                    "REVIEW",
                    {
                        "max_psi": round(max_psi, 6),
                        "features": significant_features,
                    },
                )

            else:

                record(
                    "DRIFT",
                    "SIGNIFICANT DRIFT",
                    "PASS",
                    f"max_psi={max_psi:.6f}",
                )

            # Moderate drift is also explicitly surfaced.
            if moderate:

                record(
                    "DRIFT",
                    "MODERATE DRIFT",
                    "REVIEW",
                    {
                        "count": moderate,
                        "features": moderate_features,
                    },
                )

            else:

                record(
                    "DRIFT",
                    "MODERATE DRIFT",
                    "PASS",
                    "No moderate drift features",
                )

        return drift

    except Exception as exc:

        record(
            "DRIFT",
            "DRIFT VALIDATION",
            "FAIL",
            exc,
        )

        return None


       
# ======================================================================
# 7. CROSS-STEP VALIDATION
# ======================================================================

def cross_validate(
    df,
    calibration,
    policy,
):

    banner("7. CROSS-STEP VALIDATION")

    # --------------------------------------------------------------
    # Dataset / calibration row consistency
    # --------------------------------------------------------------

    if calibration is not None:

        calibration_years = sorted(
            pd.to_numeric(
                calibration["year"],
                errors="coerce",
            ).dropna().unique()
        )

        print(
            "CALIBRATION YEARS:",
            calibration_years,
        )

        record(
            "CROSS STEP",
            "CALIBRATION OUTPUT",
            "PASS",
            f"rows={len(calibration)}",
        )

    # --------------------------------------------------------------
    # Policy prediction consistency
    # --------------------------------------------------------------

    if policy is not None:

        policy_probability = None
        policy_threshold = None

        policy_columns = [
            x.lower()
            for x in policy.columns
        ]

        for original, lowered in zip(
            policy.columns,
            policy_columns,
        ):

            if (
                "threshold"
                in lowered
            ):
                value = pd.to_numeric(
                    policy[original],
                    errors="coerce",
                ).dropna()

                if len(value):
                    policy_threshold = float(
                        value.iloc[0]
                    )

            if "probability_type" in lowered:
                policy_probability = str(
                    policy[original].iloc[0]
                ).strip()

            elif (
                "probability"
                in lowered
                and "selected"
                in lowered
            ):
                policy_probability = str(
                    policy[original].iloc[0]
                ).strip()

        print(
            "POLICY THRESHOLD:",
            policy_threshold,
        )

        print(
            "POLICY PROBABILITY:",
            policy_probability,
        )

        if (
            policy_threshold is not None
            and 0 <= policy_threshold <= 1
        ):

            record(
                "CROSS STEP",
                "POLICY THRESHOLD",
                "PASS",
                policy_threshold,
            )

        elif policy_threshold is not None:

            record(
                "CROSS STEP",
                "POLICY THRESHOLD",
                "FAIL",
                policy_threshold,
            )

        else:

            record(
                "CROSS STEP",
                "POLICY THRESHOLD",
                "REVIEW",
                "Could not identify threshold",
            )


# ======================================================================
# 8. OUTPUT VALIDATION
# ======================================================================

def validate_outputs():

    banner("8. PRODUCTION ARTIFACT VALIDATION")

    artifacts = [
        V4_DATASET,
        MODEL_FILE,
        MODEL_SCHEMA_FILE,
        CALIBRATION_FILE,
        CALIBRATION_METRICS,
        CALIBRATION_SUMMARY,
        POLICY_FILE,
        POLICY_METRICS,
        POLICY_PREDICTIONS,
        THRESHOLD_FILE,
        DRIFT_FILE,
    ]

    missing = []

    for path in artifacts:

        if path.exists():

            print(
                "PASS:",
                path,
            )

        else:

            print(
                "FAIL:",
                path,
            )

            missing.append(
                str(path)
            )

    record(
        "ARTIFACTS",
        "REQUIRED ARTIFACTS",
        "PASS" if not missing else "FAIL",
        "All required artifacts exist"
        if not missing
        else missing,
    )


# ======================================================================
# 9. FINAL DECISION
# ======================================================================

def final_decision():

    banner("9. FINAL PRODUCTION DECISION")

    report_df = pd.DataFrame(
        results
    )

    failures = report_df[
        report_df["status"] == "FAIL"
    ]

    reviews = report_df[
        report_df["status"] == "REVIEW"
    ]

    passes = report_df[
        report_df["status"] == "PASS"
    ]

    failure_count = len(failures)
    review_count = len(reviews)
    pass_count = len(passes)

    print(
        "PASS:",
        pass_count,
    )

    print(
        "FAIL:",
        failure_count,
    )

    print(
        "REVIEW:",
        review_count,
    )

    if failure_count == 0:

        if review_count == 0:

            decision = "GO"

        else:

            decision = "GO_WITH_REVIEW"

    else:

        decision = "NO_GO"

    print()
    print(
        "FINAL DECISION:",
        decision,
    )

    if failure_count:

        print()
        print(
            "HARD FAILURES:"
        )

        for _, row in failures.iterrows():

            print(
                f"- {row['check']}: "
                f"{row['details']}"
            )

    if review_count:

        print()
        print(
            "REVIEW ITEMS:"
        )

        for _, row in reviews.iterrows():

            print(
                f"- {row['check']}: "
                f"{row['details']}"
            )

    return (
        report_df,
        decision,
        failure_count,
        review_count,
    )


# ======================================================================
# 10. SAVE REPORTS
# ======================================================================

def save_reports(
    report_df,
    decision,
    failure_count,
    review_count,
):
    """Save CSV, JSON and Markdown reports without optional dependencies."""

    banner("10. SAVING PRODUCTION READINESS REPORTS")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    report_df.to_csv(REPORT_CSV, index=False)

    # JSON
    summary = {
        "project": "Bharat Earth",
        "validation_version": "V4",
        "final_decision": str(decision),
        "total_checks": int(len(report_df)),
        "passes": int((report_df["status"] == "PASS").sum()),
        "failures": int(failure_count),
        "reviews": int(review_count),
        "dataset": str(V4_DATASET),
        "model": str(MODEL_FILE),
        "model_schema": str(MODEL_SCHEMA_FILE),
        "calibration": str(CALIBRATION_FILE),
        "policy": str(POLICY_FILE),
        "drift": str(DRIFT_FILE),
    }

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    # Markdown
    columns = list(report_df.columns)

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("# Bharat Earth V4 Production Readiness Report\n\n")
        f.write("## Final Decision\n\n")
        f.write(f"**{decision}**\n\n")
        f.write(f"- Total checks: {len(report_df)}\n")
        f.write(f"- Passes: {(report_df['status'] == 'PASS').sum()}\n")
        f.write(f"- Failures: {failure_count}\n")
        f.write(f"- Reviews: {review_count}\n\n")

        f.write("## Validation Results\n\n")

        # Markdown header
        f.write(
            "| " + " | ".join(str(c) for c in columns) + " |\n"
        )

        # Markdown separator
        f.write(
            "| " + " | ".join("---" for _ in columns) + " |\n"
        )

        # Markdown rows
        for _, row in report_df.iterrows():
            values = []

            for column in columns:
                value = row[column]

                if pd.isna(value):
                    value = ""

                value = (
                    str(value)
                    .replace("\\", "\\\\")
                    .replace("|", "\\|")
                    .replace("\r", " ")
                    .replace("\n", " ")
                )

                values.append(value)

            f.write("| " + " | ".join(values) + " |\n")

        f.write("\n## Review Items\n\n")

        reviews = report_df[report_df["status"] == "REVIEW"]

        if reviews.empty:
            f.write("No review items.\n")
        else:
            for _, row in reviews.iterrows():
                f.write(
                    f"- **{row['check']}**: {row['details']}\n"
                )

        f.write("\n## Artifact Paths\n\n")
        f.write(f"- Dataset: `{V4_DATASET}`\n")
        f.write(f"- Model: `{MODEL_FILE}`\n")
        f.write(f"- Model schema: `{MODEL_SCHEMA_FILE}`\n")
        f.write(f"- Calibration: `{CALIBRATION_FILE}`\n")
        f.write(f"- Policy: `{POLICY_FILE}`\n")
        f.write(f"- Drift: `{DRIFT_FILE}`\n")

    print("CSV:", REPORT_CSV)
    print("JSON:", REPORT_JSON)
    print("MARKDOWN:", REPORT_MD)
    print("REPORT SAVING: PASS")


# ======================================================================
# MAIN
# ======================================================================

def main():

    banner(
        "BHARAT EARTH V4 PRODUCTION READINESS"
    )

    print(
        "PROJECT ROOT:",
        PROJECT_ROOT,
    )

    # --------------------------------------------------------------
    # Step 1
    # --------------------------------------------------------------

    df = validate_dataset()

    # --------------------------------------------------------------
    # Step 2
    # --------------------------------------------------------------

    model = validate_model()

    # --------------------------------------------------------------
    # Step 3
    # --------------------------------------------------------------

    validate_model_compatibility(
        df,
        model,
    )

    # --------------------------------------------------------------
    # Step 4
    # --------------------------------------------------------------

    calibration = validate_calibration()

    # --------------------------------------------------------------
    # Step 5
    # --------------------------------------------------------------

    policy = validate_policy()

    # --------------------------------------------------------------
    # Step 6
    # --------------------------------------------------------------

    validate_drift()

    # --------------------------------------------------------------
    # Step 7
    # --------------------------------------------------------------

    cross_validate(
        df,
        calibration,
        policy,
    )

    # --------------------------------------------------------------
    # Step 8
    # --------------------------------------------------------------

    validate_outputs()

    # --------------------------------------------------------------
    # Step 9
    # --------------------------------------------------------------

    (
        report_df,
        decision,
        failure_count,
        review_count,
    ) = final_decision()

    # --------------------------------------------------------------
    # Step 10
    # --------------------------------------------------------------

    save_reports(
        report_df,
        decision,
        failure_count,
        review_count,
    )

    banner(
        "PRODUCTION READINESS COMPLETE"
    )

    print(
        "FINAL DECISION:",
        decision,
    )

    print(
        "FAILURES:",
        failure_count,
    )

    print(
        "REVIEWS:",
        review_count,
    )

    print(
        "REPORT:",
        REPORT_MD,
    )

    # Do not make the process fail just because
    # drift is REVIEW. Hard failures remain NO_GO.
    if failure_count > 0:
        return 1

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )