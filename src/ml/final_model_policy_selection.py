from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# 3.10 FINAL MODEL & POLICY SELECTION
# ============================================================

print("=" * 70)
print("3.10 FINAL MODEL & POLICY SELECTION")
print("=" * 70)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v2.csv"
)

TARGET_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "severe_anomaly_target.csv"
)

CALIBRATED_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "calibrated_predictions.csv"
)

SEASONAL_POLICY_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "seasonal_policy_results.csv"
)

POLICY_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "policy_stress_test.csv"
)

ERROR_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "error_analysis"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "final_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET = "target_3m_severe_anomaly"


# ============================================================
# POLICY CONSTRAINTS
# ============================================================

# We do not want a policy that alerts on most observations.
MAX_ALERT_RATE = 0.25

# Minimum acceptable recall for the final operational policy.
MIN_RECALL = 0.30

# Minimum acceptable precision.
MIN_PRECISION = 0.10

# Candidate thresholds.
THRESHOLDS = np.round(
    np.arange(
        0.01,
        0.171,
        0.005,
    ),
    3,
)


# ============================================================
# LOAD CALIBRATED PREDICTIONS
# ============================================================

def load_calibrated_predictions():

    print()
    print("=" * 70)
    print("LOADING CALIBRATED PREDICTIONS")
    print("=" * 70)

    if not CALIBRATED_FILE.exists():

        raise FileNotFoundError(
            f"Missing file:\n{CALIBRATED_FILE}"
        )

    df = pd.read_csv(
        CALIBRATED_FILE
    )

    print(
        "INPUT:",
        CALIBRATED_FILE
    )

    print(
        "ROWS:",
        len(df)
    )

    print(
        "COLUMNS:",
        list(df.columns)
    )

    required = [
        "subdivision",
        "year",
        "month",
        "actual",
        "raw_probability",
        "isotonic_probability",
        "sigmoid_probability",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    ).astype(int)

    for column in [
        "raw_probability",
        "isotonic_probability",
        "sigmoid_probability",
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "actual",
            "raw_probability",
            "isotonic_probability",
            "sigmoid_probability",
        ]
    ).copy()

    return df


# ============================================================
# LOAD EXISTING POLICY RESULTS
# ============================================================

def load_existing_results():

    print()
    print("=" * 70)
    print("LOADING EXISTING POLICY RESULTS")
    print("=" * 70)

    seasonal = None
    policy = None
    error_summary = None

    if SEASONAL_POLICY_FILE.exists():

        seasonal = pd.read_csv(
            SEASONAL_POLICY_FILE
        )

        print(
            "SEASONAL POLICY:",
            seasonal.shape
        )

    else:

        print(
            "WARNING: seasonal_policy_results.csv not found."
        )

    if POLICY_FILE.exists():

        policy = pd.read_csv(
            POLICY_FILE
        )

        print(
            "POLICY STRESS TEST:",
            policy.shape
        )

    else:

        print(
            "WARNING: policy_stress_test.csv not found."
        )

    error_file = (
        ERROR_DIR
        / "global_error_summary.csv"
    )

    if error_file.exists():

        error_summary = pd.read_csv(
            error_file
        )

        print(
            "ERROR SUMMARY:",
            error_summary.shape
        )

    return seasonal, policy, error_summary


# ============================================================
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    y,
    probability,
    threshold,
):

    prediction = (
        probability
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        prediction,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y,
        prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        prediction,
        zero_division=0,
    )

    alert_rate = (
        prediction.mean()
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": alert_rate,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "false_positive_rate": false_positive_rate,
    }


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_probability_models(
    df
):

    print()
    print("=" * 70)
    print("MODEL PROBABILITY COMPARISON")
    print("=" * 70)

    y = df["actual"].values

    rows = []

    models = {
        "RAW_XGBOOST":
            "raw_probability",

        "ISOTONIC":
            "isotonic_probability",

        "SIGMOID":
            "sigmoid_probability",
    }

    for model_name, column in models.items():

        probability = (
            df[column]
            .values
        )

        pr_auc = average_precision_score(
            y,
            probability,
        )

        roc_auc = roc_auc_score(
            y,
            probability,
        )

        brier = np.mean(
            (
                probability
                - y
            ) ** 2
        )

        rows.append(
            {
                "model": model_name,
                "probability_column": column,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "brier_score": brier,
                "mean_probability":
                    probability.mean(),
                "max_probability":
                    probability.max(),
            }
        )

    result = pd.DataFrame(
        rows
    )

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def threshold_search(
    df,
    probability_column,
):

    y = df["actual"].values

    probability = (
        df[probability_column]
        .values
    )

    rows = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            y,
            probability,
            threshold,
        )

        rows.append(
            metrics
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# POLICY SCORING
# ============================================================

def select_operational_policy(
    threshold_results
):

    candidates = threshold_results[
        (
            threshold_results[
                "alert_rate"
            ]
            <= MAX_ALERT_RATE
        )
        &
        (
            threshold_results[
                "recall"
            ]
            >= MIN_RECALL
        )
        &
        (
            threshold_results[
                "precision"
            ]
            >= MIN_PRECISION
        )
    ].copy()

    # --------------------------------------------------------
    # Primary objective:
    # maximize F1.
    #
    # Secondary:
    # maximize precision.
    #
    # Tertiary:
    # maximize recall.
    # --------------------------------------------------------

    if len(candidates) > 0:

        candidates = candidates.sort_values(
            [
                "f1",
                "precision",
                "recall",
            ],
            ascending=False,
        )

        selected = candidates.iloc[
            0
        ].copy()

        selection_reason = (
            "Meets alert-rate, precision "
            "and recall constraints while "
            "maximizing F1."
        )

    else:

        # ----------------------------------------------------
        # No threshold satisfies every constraint.
        # Choose the best F1 among thresholds
        # below the alert-rate ceiling.
        # ----------------------------------------------------

        candidates = threshold_results[
            threshold_results[
                "alert_rate"
            ]
            <= MAX_ALERT_RATE
        ].copy()

        if len(candidates) == 0:

            candidates = (
                threshold_results.copy()
            )

        candidates = candidates.sort_values(
            [
                "f1",
                "precision",
                "recall",
            ],
            ascending=False,
        )

        selected = candidates.iloc[
            0
        ].copy()

        selection_reason = (
            "No threshold satisfies all "
            "constraints. Selected the best "
            "F1 subject to the alert-rate "
            "constraint where possible."
        )

    return (
        selected,
        candidates,
        selection_reason,
    )


# ============================================================
# SEASON-AWARE POLICY EVALUATION
# ============================================================

def evaluate_season_policy(
    df,
    thresholds,
):

    print()
    print("=" * 70)
    print("SEASON-AWARE POLICY EVALUATION")
    print("=" * 70)

    season_thresholds = {}

    for _, row in thresholds.iterrows():

        season = str(
            row["season"]
        )

        threshold = float(
            row["threshold"]
        )

        season_thresholds[
            season
        ] = threshold

    result = df.copy()

    result["season_threshold"] = (
        result["season"]
        .astype(str)
        .map(season_thresholds)
    )

    # If season is unavailable,
    # use the global threshold later.

    return result


# ============================================================
# FIND SEASON COLUMN
# ============================================================

def add_season(
    df
):

    data = df.copy()

    if "season" in data.columns:

        return data

    month = pd.to_numeric(
        data["month"],
        errors="coerce",
    )

    data["season"] = np.select(
        [
            month.isin(
                [12, 1, 2]
            ),

            month.isin(
                [3, 4, 5]
            ),

            month.isin(
                [6, 7, 8, 9]
            ),

            month.isin(
                [10, 11]
            ),
        ],
        [
            "WINTER",
            "PRE_MONSOON",
            "MONSOON",
            "POST_MONSOON",
        ],
        default="UNKNOWN",
    )

    return data


# ============================================================
# EVALUATE GLOBAL POLICY
# ============================================================

def evaluate_global_policy(
    df,
    probability_column,
    threshold,
):

    y = df["actual"].values

    probability = (
        df[probability_column]
        .values
    )

    metrics = calculate_metrics(
        y,
        probability,
        threshold,
    )

    metrics[
        "policy"
    ] = "GLOBAL"

    metrics[
        "probability_model"
    ] = probability_column

    return metrics


# ============================================================
# EVALUATE SEASON-AWARE POLICY
# ============================================================

def evaluate_season_aware_policy(
    df,
    probability_column,
    season_thresholds,
):

    data = add_season(
        df
    ).copy()

    data["season_threshold"] = (
        data["season"]
        .map(
            season_thresholds
        )
    )

    # Fallback to global threshold
    # if an unknown season exists.

    fallback = np.mean(
        list(
            season_thresholds.values()
        )
    )

    data[
        "season_threshold"
    ] = data[
        "season_threshold"
    ].fillna(
        fallback
    )

    probability = (
        data[
            probability_column
        ].values
    )

    y = data[
        "actual"
    ].values

    prediction = (
        probability
        >= data[
            "season_threshold"
        ].values
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "policy":
            "SEASON_AWARE",

        "probability_model":
            probability_column,

        "threshold":
            "SEASON_SPECIFIC",

        "precision":
            precision_score(
                y,
                prediction,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y,
                prediction,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y,
                prediction,
                zero_division=0,
            ),

        "alert_rate":
            prediction.mean(),

        "true_positive":
            int(tp),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_negative":
            int(tn),
    }


# ============================================================
# SEASON THRESHOLD FILE
# ============================================================

def extract_season_thresholds(
    seasonal
):

    if seasonal is None:

        return None

    required = [
        "season",
        "threshold",
    ]

    if not all(
        c in seasonal.columns
        for c in required
    ):

        return None

    result = {}

    for _, row in seasonal.iterrows():

        result[
            str(row["season"])
        ] = float(
            row["threshold"]
        )

    return result


# ============================================================
# POLICY DECISION
# ============================================================

def choose_final_policy(
    global_metrics,
    season_metrics,
):

    candidates = []

    if global_metrics is not None:

        candidates.append(
            global_metrics
        )

    if season_metrics is not None:

        candidates.append(
            season_metrics
        )

    comparison = pd.DataFrame(
        candidates
    )

    print()
    print("=" * 70)
    print("FINAL POLICY CANDIDATES")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Operational policy selection
    #
    # Prefer:
    # 1. Alert rate <= 25%
    # 2. Precision >= 10%
    # 3. Recall >= 30%
    # 4. Highest F1
    # --------------------------------------------------------

    valid = comparison[
        (
            comparison[
                "alert_rate"
            ]
            <= MAX_ALERT_RATE
        )
        &
        (
            comparison[
                "precision"
            ]
            >= MIN_PRECISION
        )
        &
        (
            comparison[
                "recall"
            ]
            >= MIN_RECALL
        )
    ].copy()

    if len(valid) > 0:

        valid = valid.sort_values(
            [
                "f1",
                "precision",
                "recall",
            ],
            ascending=False,
        )

        final = valid.iloc[
            0
        ].copy()

    else:

        valid = comparison[
            comparison[
                "alert_rate"
            ]
            <= MAX_ALERT_RATE
        ].copy()

        if len(valid) == 0:

            valid = comparison.copy()

        valid = valid.sort_values(
            [
                "f1",
                "precision",
            ],
            ascending=False,
        )

        final = valid.iloc[
            0
        ].copy()

    return (
        final,
        comparison,
    )


# ============================================================
# BUILD FINAL POLICY DATASET
# ============================================================

def build_final_predictions(
    df,
    probability_column,
    policy_type,
    threshold,
    season_thresholds=None,
):

    result = add_season(
        df
    ).copy()

    result[
        "final_probability"
    ] = result[
        probability_column
    ]

    if policy_type == "GLOBAL":

        result[
            "final_threshold"
        ] = float(
            threshold
        )

    else:

        result[
            "final_threshold"
        ] = (
            result["season"]
            .map(
                season_thresholds
            )
        )

        result[
            "final_threshold"
        ] = result[
            "final_threshold"
        ].fillna(
            float(
                threshold
                if not isinstance(
                    threshold,
                    str
                )
                else 0.09
            )
        )

    result[
        "final_alert"
    ] = (
        result[
            "final_probability"
        ]
        >= result[
            "final_threshold"
        ]
    ).astype(int)

    result[
        "correct_prediction"
    ] = (
        result[
            "final_alert"
        ]
        == result[
            "actual"
        ]
    ).astype(int)

    result[
        "error_type"
    ] = np.select(
        [
            (
                result["actual"]
                == 1
            )
            &
            (
                result["final_alert"]
                == 1
            ),

            (
                result["actual"]
                == 0
            )
            &
            (
                result["final_alert"]
                == 0
            ),

            (
                result["actual"]
                == 0
            )
            &
            (
                result["final_alert"]
                == 1
            ),

            (
                result["actual"]
                == 1
            )
            &
            (
                result["final_alert"]
                == 0
            ),
        ],
        [
            "TRUE_POSITIVE",
            "TRUE_NEGATIVE",
            "FALSE_POSITIVE",
            "FALSE_NEGATIVE",
        ],
        default="UNKNOWN",
    )

    return result


# ============================================================
# FINAL REPORT
# ============================================================

def create_final_report(
    model_comparison,
    threshold_results,
    final_policy,
    comparison,
):

    report = []

    report.append(
        {
            "section":
                "FINAL_POLICY",

            "metric":
                "policy",

            "value":
                final_policy[
                    "policy"
                ],
        }
    )

    for column in [
        "probability_model",
        "threshold",
        "precision",
        "recall",
        "f1",
        "alert_rate",
        "true_positive",
        "false_positive",
        "false_negative",
        "true_negative",
    ]:

        if column in final_policy.index:

            report.append(
                {
                    "section":
                        "FINAL_POLICY",

                    "metric":
                        column,

                    "value":
                        final_policy[
                            column
                        ],
                }
            )

    report.append(
        {
            "section":
                "CONSTRAINTS",

            "metric":
                "max_alert_rate",

            "value":
                MAX_ALERT_RATE,
        }
    )

    report.append(
        {
            "section":
                "CONSTRAINTS",

            "metric":
                "minimum_precision",

            "value":
                MIN_PRECISION,
        }
    )

    report.append(
        {
            "section":
                "CONSTRAINTS",

            "metric":
                "minimum_recall",

            "value":
                MIN_RECALL,
        }
    )

    return pd.DataFrame(
        report
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_calibrated_predictions()

    seasonal, policy, error_summary = (
        load_existing_results()
    )

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    model_comparison = (
        compare_probability_models(
            df
        )
    )

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("GLOBAL THRESHOLD ANALYSIS")
    print("=" * 70)

    threshold_tables = []

    for model_name, column in {
        "RAW_XGBOOST":
            "raw_probability",

        "ISOTONIC":
            "isotonic_probability",

        "SIGMOID":
            "sigmoid_probability",
    }.items():

        table = threshold_search(
            df,
            column,
        )

        table[
            "model"
        ] = model_name

        table[
            "probability_column"
        ] = column

        threshold_tables.append(
            table
        )

    all_thresholds = pd.concat(
        threshold_tables,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Print sigmoid because calibration
    # is the preferred probability source.
    # --------------------------------------------------------

    sigmoid_thresholds = (
        all_thresholds[
            all_thresholds[
                "model"
            ]
            == "SIGMOID"
        ]
        .sort_values(
            [
                "f1",
                "precision",
                "recall",
            ],
            ascending=False,
        )
    )

    print(
        sigmoid_thresholds.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Select calibrated operational threshold.
    # --------------------------------------------------------

    selected_threshold, candidates, reason = (
        select_operational_policy(
            sigmoid_thresholds
        )
    )

    print()
    print("=" * 70)
    print("SELECTED CALIBRATED THRESHOLD")
    print("=" * 70)

    print(
        "THRESHOLD:",
        selected_threshold[
            "threshold"
        ]
    )

    print(
        "PRECISION:",
        selected_threshold[
            "precision"
        ]
    )

    print(
        "RECALL:",
        selected_threshold[
            "recall"
        ]
    )

    print(
        "F1:",
        selected_threshold[
            "f1"
        ]
    )

    print(
        "ALERT RATE:",
        selected_threshold[
            "alert_rate"
        ]
    )

    print(
        "REASON:",
        reason
    )

    # --------------------------------------------------------
    # Global sigmoid policy
    # --------------------------------------------------------

    global_metrics = evaluate_global_policy(
        df,
        "sigmoid_probability",
        float(
            selected_threshold[
                "threshold"
            ]
        ),
    )

    # --------------------------------------------------------
    # Existing seasonal policy
    # --------------------------------------------------------

    season_thresholds = (
        extract_season_thresholds(
            seasonal
        )
    )

    season_metrics = None

    if season_thresholds is not None:

        season_metrics = (
            evaluate_season_aware_policy(
                df,
                "sigmoid_probability",
                season_thresholds,
            )
        )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    final_policy, comparison = (
        choose_final_policy(
            global_metrics,
            season_metrics,
        )
    )

    print()
    print("=" * 70)
    print("FINAL POLICY")
    print("=" * 70)

    for column in final_policy.index:

        print(
            f"{column}: "
            f"{final_policy[column]}"
        )

    # --------------------------------------------------------
    # Build final predictions
    # --------------------------------------------------------

    final_policy_type = (
        final_policy[
            "policy"
        ]
    )

    if final_policy_type == "GLOBAL":

        final_predictions = (
            build_final_predictions(
                df,
                "sigmoid_probability",
                "GLOBAL",
                float(
                    final_policy[
                        "threshold"
                    ]
                ),
            )
        )

    else:

        final_predictions = (
            build_final_predictions(
                df,
                "sigmoid_probability",
                "SEASON_AWARE",
                0.09,
                season_thresholds,
            )
        )

    # --------------------------------------------------------
    # Final error counts
    # --------------------------------------------------------

    error_counts = (
        final_predictions[
            "error_type"
        ]
        .value_counts()
        .rename_axis(
            "error_type"
        )
        .reset_index(
            name="count"
        )
    )

    error_counts[
        "percentage"
    ] = (
        error_counts[
            "count"
        ]
        / len(final_predictions)
    )

    print()
    print("=" * 70)
    print("FINAL ERROR COMPOSITION")
    print("=" * 70)

    print(
        error_counts.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Final season performance
    # --------------------------------------------------------

    season_final = []

    for season, group in (
        final_predictions.groupby(
            "season"
        )
    ):

        y = group["actual"]

        p = group["final_alert"]

        season_final.append(
            {
                "season":
                    season,

                "observations":
                    len(group),

                "events":
                    int(y.sum()),

                "event_rate":
                    y.mean(),

                "alerts":
                    int(p.sum()),

                "alert_rate":
                    p.mean(),

                "precision":
                    precision_score(
                        y,
                        p,
                        zero_division=0,
                    ),

                "recall":
                    recall_score(
                        y,
                        p,
                        zero_division=0,
                    ),

                "f1":
                    f1_score(
                        y,
                        p,
                        zero_division=0,
                    ),
            }
        )

    season_final = pd.DataFrame(
        season_final
    )

    print()
    print("=" * 70)
    print("FINAL POLICY BY SEASON")
    print("=" * 70)

    print(
        season_final.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    report = create_final_report(
        model_comparison,
        sigmoid_thresholds,
        final_policy,
        comparison,
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    final_predictions.to_csv(
        OUTPUT_DIR
        / "final_predictions.csv",
        index=False,
    )

    model_comparison.to_csv(
        OUTPUT_DIR
        / "model_comparison.csv",
        index=False,
    )

    all_thresholds.to_csv(
        OUTPUT_DIR
        / "all_threshold_results.csv",
        index=False,
    )

    sigmoid_thresholds.to_csv(
        OUTPUT_DIR
        / "sigmoid_threshold_results.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "final_policy_comparison.csv",
        index=False,
    )

    error_counts.to_csv(
        OUTPUT_DIR
        / "final_error_composition.csv",
        index=False,
    )

    season_final.to_csv(
        OUTPUT_DIR
        / "final_season_performance.csv",
        index=False,
    )

    report.to_csv(
        OUTPUT_DIR
        / "final_model_policy_report.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    try:

        with pd.ExcelWriter(
            OUTPUT_DIR
            / "final_model_policy_results.xlsx",
            engine="openpyxl",
        ) as writer:

            model_comparison.to_excel(
                writer,
                sheet_name="model_comparison",
                index=False,
            )

            all_thresholds.to_excel(
                writer,
                sheet_name="thresholds",
                index=False,
            )

            sigmoid_thresholds.to_excel(
                writer,
                sheet_name="sigmoid_thresholds",
                index=False,
            )

            comparison.to_excel(
                writer,
                sheet_name="policy_comparison",
                index=False,
            )

            error_counts.to_excel(
                writer,
                sheet_name="error_composition",
                index=False,
            )

            season_final.to_excel(
                writer,
                sheet_name="season_performance",
                index=False,
            )

            report.to_excel(
                writer,
                sheet_name="final_report",
                index=False,
            )

            final_predictions[
                [
                    "subdivision",
                    "year",
                    "month",
                    "season",
                    "actual",
                    "sigmoid_probability",
                    "final_probability",
                    "final_threshold",
                    "final_alert",
                    "error_type",
                ]
            ].to_excel(
                writer,
                sheet_name="final_predictions",
                index=False,
            )

        print()
        print(
            "Excel report saved."
        )

    except ImportError:

        print()
        print(
            "openpyxl is not installed."
        )

        print(
            "CSV files were saved successfully."
        )

        print(
            "Install with:"
        )

        print(
            "pip install openpyxl"
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):

        if path.is_file():

            print(path)

    print()
    print("=" * 70)
    print("3.10 FINAL MODEL & POLICY SELECTION COMPLETE")
    print("=" * 70)

    print(
        "FINAL POLICY:",
        final_policy[
            "policy"
        ]
    )

    print(
        "FINAL MODEL:",
        final_policy[
            "probability_model"
        ]
    )

    print(
        "FINAL PRECISION:",
        f"{float(final_policy['precision']):.6f}"
    )

    print(
        "FINAL RECALL:",
        f"{float(final_policy['recall']):.6f}"
    )

    print(
        "FINAL F1:",
        f"{float(final_policy['f1']):.6f}"
    )

    print(
        "FINAL ALERT RATE:",
        f"{float(final_policy['alert_rate']):.6f}"
    )

    print()
    print(
        "NEXT STAGE: 3.11 FINAL PREDICTION & DEPLOYMENT REPORT"
    )


if __name__ == "__main__":
    main()