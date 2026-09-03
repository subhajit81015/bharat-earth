from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)


# ================================================================
# 9. FINAL POLICY / THRESHOLD OPTIMIZATION V4
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ================================================================
# INPUT
# ================================================================

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "calibration_v4"
    / "calibrated_predictions.csv"
)


# ================================================================
# OUTPUT
# ================================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "policy_v4"
)

THRESHOLD_RESULTS_FILE = (
    OUTPUT_DIR
    / "threshold_analysis.csv"
)

POLICY_METRICS_FILE = (
    OUTPUT_DIR
    / "policy_metrics.csv"
)

SELECTED_POLICY_FILE = (
    OUTPUT_DIR
    / "selected_policy.csv"
)

CALIBRATION_POLICY_FILE = (
    OUTPUT_DIR
    / "calibration_policy_predictions.csv"
)


# ================================================================
# TARGET
# ================================================================

TARGET = "actual"


# ================================================================
# PROBABILITIES
# ================================================================

PROBABILITY_COLUMNS = [
    "raw_probability",
    "isotonic_probability",
    "sigmoid_probability",
]


# ================================================================
# POLICY SEARCH
# ================================================================

THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.50
THRESHOLD_STEP = 0.005

MIN_RECALL = 0.40
MIN_PRECISION = 0.10


# ================================================================
# MONTH NORMALIZATION
# ================================================================

MONTH_MAP = {
    "JAN": 1,
    "JANUARY": 1,
    "FEB": 2,
    "FEBRUARY": 2,
    "MAR": 3,
    "MARCH": 3,
    "APR": 4,
    "APRIL": 4,
    "MAY": 5,
    "JUN": 6,
    "JUNE": 6,
    "JUL": 7,
    "JULY": 7,
    "AUG": 8,
    "AUGUST": 8,
    "SEP": 9,
    "SEPT": 9,
    "SEPTEMBER": 9,
    "OCT": 10,
    "OCTOBER": 10,
    "NOV": 11,
    "NOVEMBER": 11,
    "DEC": 12,
    "DECEMBER": 12,
}


# ================================================================
# HEADER
# ================================================================

def header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ================================================================
# NORMALIZE MONTH
# ================================================================

def normalize_month(series):

    original = series.copy()

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    result = numeric.copy()

    unresolved = result.isna()

    if unresolved.any():

        text = (
            series
            .astype("string")
            .str.strip()
            .str.upper()
        )

        mapped = (
            text.map(MONTH_MAP)
        )

        result.loc[
            unresolved
        ] = mapped.loc[
            unresolved
        ]

    # ------------------------------------------------------------
    # Support zero-based encoding if present
    # ------------------------------------------------------------

    zero_based_mask = (
        result.notna()
        & result.between(0, 11)
    )

    if (
        zero_based_mask.any()
        and not result.between(
            1,
            12,
        ).all()
    ):

        result.loc[
            zero_based_mask
            & (result == 0)
        ] = 1

        for old_month in range(1, 12):

            mask = (
                zero_based_mask
                & (result == old_month)
            )

            result.loc[
                mask
            ] = old_month + 1

    return result


# ================================================================
# LOAD
# ================================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nCalibration file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    header(
        "LOADING CALIBRATED V4 PREDICTIONS"
    )

    print(
        "INPUT:"
    )

    print(
        INPUT_FILE
    )

    print(
        "ROWS:",
        len(df)
    )

    print(
        "COLUMNS:",
        len(df.columns)
    )

    print(
        "\nCOLUMNS:"
    )

    print(
        df.columns.tolist()
    )

    return df


# ================================================================
# VALIDATE INPUT
# ================================================================

def validate_input(df):

    header(
        "INPUT VALIDATION"
    )

    required_columns = {
        "subdivision",
        "year",
        "month",
        "season",
        TARGET,
        *PROBABILITY_COLUMNS,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing required columns:\n"
            f"{sorted(missing)}"
        )

    # ------------------------------------------------------------
    # Leakage
    # ------------------------------------------------------------

    forbidden = {
        "target_3m_stress",
        "rainfall_stress",
    }

    present_forbidden = (
        forbidden
        & set(df.columns)
    )

    if present_forbidden:

        raise ValueError(
            "Forbidden leakage columns found:\n"
            f"{sorted(present_forbidden)}"
        )

    print(
        "LEAKAGE CHECK: PASS"
    )

    # ------------------------------------------------------------
    # Target
    # ------------------------------------------------------------

    target = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    if target.isna().any():

        raise ValueError(
            "Target contains NULL/non-numeric values."
        )

    target_values = sorted(
        target.unique().tolist()
    )

    print(
        "TARGET VALUES:",
        target_values
    )

    if not set(
        target_values
    ).issubset({0, 1}):

        raise ValueError(
            "Target must contain only 0 and 1."
        )

    if len(
        target_values
    ) != 2:

        raise ValueError(
            "Target must contain both classes."
        )

    print(
        "TARGET RATE:",
        f"{target.mean():.6f}"
    )

    print(
        "TARGET VALIDATION: PASS"
    )

    # ------------------------------------------------------------
    # Probability columns
    # ------------------------------------------------------------

    for column in PROBABILITY_COLUMNS:

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        if values.isna().any():

            raise ValueError(
                f"{column} contains NULL/non-numeric values."
            )

        if (
            values < 0
        ).any() or (
            values > 1
        ).any():

            raise ValueError(
                f"{column} contains values outside [0,1]."
            )

        print(
            f"{column}: PASS"
        )

    # ------------------------------------------------------------
    # Normalize month
    # ------------------------------------------------------------

    original_month = (
        df["month"].copy()
    )

    normalized_month = (
        normalize_month(
            df["month"]
        )
    )

    invalid = (
        normalized_month.isna()
        | ~normalized_month.between(
            1,
            12,
        )
    )

    print(
        "INVALID MONTHS BEFORE NORMALIZATION:",
        int(
            (
                pd.to_numeric(
                    original_month,
                    errors="coerce",
                ).isna()
            ).sum()
        )
    )

    print(
        "INVALID MONTHS AFTER NORMALIZATION:",
        int(invalid.sum())
    )

    if invalid.any():

        print(
            "\nUNRESOLVED MONTH VALUES:"
        )

        print(
            original_month[
                invalid
            ]
            .value_counts()
            .head(20)
        )

        raise ValueError(
            "Month normalization failed."
        )

    df["month"] = (
        normalized_month
        .astype(int)
    )

    print(
        "MONTH RANGE:",
        int(df["month"].min()),
        "-",
        int(df["month"].max())
    )

    print(
        "MONTH VALIDATION: PASS"
    )

    # ------------------------------------------------------------
    # Year
    # ------------------------------------------------------------

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    if df["year"].isna().any():

        raise ValueError(
            "Invalid year values found."
        )

    print(
        "YEAR RANGE:",
        int(df["year"].min()),
        "-",
        int(df["year"].max())
    )

    # ------------------------------------------------------------
    # Keys
    # ------------------------------------------------------------

    key_columns = [
        "subdivision",
        "year",
        "month",
    ]

    duplicate_keys = int(
        df.duplicated(
            subset=key_columns
        ).sum()
    )

    print(
        "DUPLICATE KEYS:",
        duplicate_keys
    )

    if duplicate_keys:

        raise ValueError(
            "Duplicate subdivision/year/month keys found."
        )

    # ------------------------------------------------------------
    # Season
    # ------------------------------------------------------------

    expected_season = derive_season(
        df["month"]
    )

    existing_season = (
        df["season"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    season_mismatch = (
        existing_season
        != expected_season
    )

    print(
        "SEASON INCONSISTENCIES:",
        int(season_mismatch.sum())
    )

    if season_mismatch.any():

        print(
            "REPAIRING season from normalized month."
        )

        df["season"] = expected_season

    else:

        df["season"] = existing_season

    unknown_seasons = (
        ~df["season"].isin(
            [
                "WINTER",
                "PRE_MONSOON",
                "MONSOON",
                "POST_MONSOON",
            ]
        )
    )

    if unknown_seasons.any():

        raise ValueError(
            "Unknown seasons remain."
        )

    print(
        "SEASON VALIDATION: PASS"
    )

    return df


# ================================================================
# SEASON DERIVATION
# ================================================================

def derive_season(month):

    month = pd.to_numeric(
        month,
        errors="coerce",
    )

    return pd.Series(
        np.select(
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
        ),
        index=month.index,
    )


# ================================================================
# CONFUSION MATRIX
# ================================================================

def calculate_confusion(
    y_true,
    y_pred,
):

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        matrix.ravel()
    )

    return (
        int(tn),
        int(fp),
        int(fn),
        int(tp),
    )


# ================================================================
# THRESHOLD METRICS
# ================================================================

def calculate_threshold_metrics(
    y_true,
    probability,
    threshold,
):

    predictions = (
        probability
        >= threshold
    ).astype(int)

    tn, fp, fn, tp = (
        calculate_confusion(
            y_true,
            predictions,
        )
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    alert_rate = (
        predictions.mean()
    )

    return {
        "threshold": float(threshold),
        "observations": int(
            len(y_true)
        ),
        "events": int(
            np.sum(y_true)
        ),
        "alerts": int(
            np.sum(predictions)
        ),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "alert_rate": float(
            alert_rate
        ),
    }


# ================================================================
# OPTIMIZE ONE PROBABILITY
# ================================================================

def optimize_probability(
    y_true,
    probability,
    probability_name,
):

    header(
        f"THRESHOLD OPTIMIZATION: {probability_name}"
    )

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX
        + THRESHOLD_STEP / 2,
        THRESHOLD_STEP,
    )

    thresholds = np.round(
        thresholds,
        6,
    )

    rows = []

    for threshold in thresholds:

        metrics = (
            calculate_threshold_metrics(
                y_true,
                probability,
                threshold,
            )
        )

        metrics[
            "probability_type"
        ] = probability_name

        rows.append(
            metrics
        )

    result = pd.DataFrame(
        rows
    )

    # ------------------------------------------------------------
    # Prefer recall >= minimum
    # ------------------------------------------------------------

    recall_candidates = result[
        result["recall"]
        >= MIN_RECALL
    ].copy()

    # ------------------------------------------------------------
    # Prefer precision >= minimum if available
    # ------------------------------------------------------------

    precision_candidates = (
        recall_candidates[
            recall_candidates["precision"]
            >= MIN_PRECISION
        ].copy()
    )

    if not precision_candidates.empty:

        candidates = (
            precision_candidates
        )

        selection_rule = (
            "precision_and_recall_constraints"
        )

    elif not recall_candidates.empty:

        candidates = (
            recall_candidates
        )

        selection_rule = (
            "recall_constraint"
        )

    else:

        candidates = result

        selection_rule = (
            "best_f1_fallback"
        )

    # ------------------------------------------------------------
    # Select highest F1
    # ------------------------------------------------------------

    selected = (
        candidates
        .sort_values(
            [
                "f1",
                "precision",
                "recall",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .iloc[0]
        .copy()
    )

    selected[
        "selection_rule"
    ] = selection_rule

    print(
        "\nSELECTED THRESHOLD:",
        f"{selected['threshold']:.6f}"
    )

    print(
        "PRECISION:",
        f"{selected['precision']:.6f}"
    )

    print(
        "RECALL:",
        f"{selected['recall']:.6f}"
    )

    print(
        "F1:",
        f"{selected['f1']:.6f}"
    )

    print(
        "ALERT RATE:",
        f"{selected['alert_rate']:.6f}"
    )

    print(
        "ALERTS:",
        int(selected["alerts"])
    )

    print(
        "SELECTION RULE:",
        selection_rule
    )

    return (
        result,
        selected,
    )


# ================================================================
# PROBABILITY QUALITY
# ================================================================

def calculate_probability_quality(
    y_true,
    probability,
    name,
):

    return {
        "probability_type":
            name,

        "observations":
            int(len(y_true)),

        "events":
            int(np.sum(y_true)),

        "event_rate":
            float(np.mean(y_true)),

        "pr_auc":
            float(
                average_precision_score(
                    y_true,
                    probability,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y_true,
                    probability,
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    y_true,
                    probability,
                )
            ),

        "mean_probability":
            float(
                probability.mean()
            ),

        "minimum_probability":
            float(
                probability.min()
            ),

        "maximum_probability":
            float(
                probability.max()
            ),
    }


# ================================================================
# SELECT FINAL CALIBRATED POLICY
# ================================================================

def select_final_policy(
    quality_df,
    selected_df,
):

    header(
        "FINAL POLICY SELECTION"
    )

    calibrated = (
        quality_df[
            quality_df[
                "probability_type"
            ].isin(
                [
                    "isotonic_probability",
                    "sigmoid_probability",
                ]
            )
        ]
        .copy()
    )

    if calibrated.empty:

        raise ValueError(
            "No calibrated probabilities found."
        )

    # ------------------------------------------------------------
    # Lowest Brier score is preferred.
    # If Brier is effectively tied, use PR-AUC.
    # ------------------------------------------------------------

    calibrated = (
        calibrated
        .sort_values(
            [
                "brier_score",
                "pr_auc",
            ],
            ascending=[
                True,
                False,
            ],
        )
    )

    best_probability_type = (
        calibrated.iloc[0][
            "probability_type"
        ]
    )

    candidate = (
        selected_df[
            selected_df[
                "probability_type"
            ]
            == best_probability_type
        ]
    )

    if candidate.empty:

        raise ValueError(
            "No threshold candidate for selected "
            "calibrated probability."
        )

    final_policy = (
        candidate
        .sort_values(
            [
                "f1",
                "precision",
                "recall",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .iloc[0]
        .copy()
    )

    print(
        "SELECTED CALIBRATED PROBABILITY:",
        best_probability_type
    )

    print(
        "SELECTED THRESHOLD:",
        f"{float(final_policy['threshold']):.6f}"
    )

    return (
        best_probability_type,
        final_policy,
    )


# ================================================================
# CREATE POLICY PREDICTIONS
# ================================================================

def create_policy_predictions(
    df,
    probability_type,
    threshold,
):

    result = df[
        [
            "subdivision",
            "year",
            "month",
            "season",
            "actual",
        ]
    ].copy()

    result[
        "selected_probability"
    ] = (
        df[
            probability_type
        ]
        .astype(float)
    )

    result[
        "policy_threshold"
    ] = float(threshold)

    result[
        "policy_alert"
    ] = (
        result[
            "selected_probability"
        ]
        >= threshold
    ).astype(int)

    # ------------------------------------------------------------
    # Risk level
    # ------------------------------------------------------------

    result[
        "alert_level"
    ] = np.select(
        [
            result[
                "selected_probability"
            ]
            >= threshold,

            result[
                "selected_probability"
            ]
            >= threshold * 0.75,

            result[
                "selected_probability"
            ]
            >= threshold * 0.50,
        ],
        [
            "HIGH",
            "WATCH",
            "LOW",
        ],
        default="NORMAL",
    )

    # ------------------------------------------------------------
    # Prediction status
    # ------------------------------------------------------------

    result[
        "prediction_status"
    ] = np.where(
        result[
            "policy_alert"
        ] == 1,

        np.where(
            result["actual"] == 1,
            "TRUE_POSITIVE",
            "FALSE_POSITIVE",
        ),

        np.where(
            result["actual"] == 1,
            "FALSE_NEGATIVE",
            "TRUE_NEGATIVE",
        ),
    )

    return result


# ================================================================
# MAIN
# ================================================================

def main():

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    df = load_data()

    # ------------------------------------------------------------
    # VALIDATE AND NORMALIZE
    # ------------------------------------------------------------

    df = validate_input(
        df
    )

    # ------------------------------------------------------------
    # TARGET
    # ------------------------------------------------------------

    y = (
        df[TARGET]
        .astype(int)
        .to_numpy()
    )

    # ------------------------------------------------------------
    # PROBABILITY QUALITY
    # ------------------------------------------------------------

    header(
        "PROBABILITY QUALITY"
    )

    quality_rows = []

    for probability_type in (
        PROBABILITY_COLUMNS
    ):

        probability = (
            df[
                probability_type
            ]
            .astype(float)
            .to_numpy()
        )

        metrics = (
            calculate_probability_quality(
                y,
                probability,
                probability_type,
            )
        )

        quality_rows.append(
            metrics
        )

        print(
            f"\n{probability_type}"
        )

        print(
            "PR-AUC:",
            f"{metrics['pr_auc']:.6f}"
        )

        print(
            "ROC-AUC:",
            f"{metrics['roc_auc']:.6f}"
        )

        print(
            "BRIER:",
            f"{metrics['brier_score']:.6f}"
        )

        print(
            "MEAN PROBABILITY:",
            f"{metrics['mean_probability']:.6f}"
        )

    quality_df = pd.DataFrame(
        quality_rows
    )

    # ------------------------------------------------------------
    # THRESHOLD SEARCH
    # ------------------------------------------------------------

    threshold_results = []
    selected_rows = []

    for probability_type in (
        PROBABILITY_COLUMNS
    ):

        probability = (
            df[
                probability_type
            ]
            .astype(float)
            .to_numpy()
        )

        result, selected = (
            optimize_probability(
                y,
                probability,
                probability_type,
            )
        )

        threshold_results.append(
            result
        )

        selected_rows.append(
            selected.to_dict()
        )

    threshold_analysis = pd.concat(
        threshold_results,
        ignore_index=True,
    )

    selected_candidates = pd.DataFrame(
        selected_rows
    )

    # ------------------------------------------------------------
    # FINAL POLICY
    # ------------------------------------------------------------

    (
        final_probability_type,
        final_policy,
    ) = select_final_policy(
        quality_df,
        selected_candidates,
    )

    final_threshold = float(
        final_policy[
            "threshold"
        ]
    )

    # ------------------------------------------------------------
    # POLICY PREDICTIONS
    # ------------------------------------------------------------

    policy_predictions = (
        create_policy_predictions(
            df,
            final_probability_type,
            final_threshold,
        )
    )

    # ------------------------------------------------------------
    # FINAL METRICS
    # ------------------------------------------------------------

    final_probability = (
        df[
            final_probability_type
        ]
        .astype(float)
        .to_numpy()
    )

    final_prediction = (
        final_probability
        >= final_threshold
    ).astype(int)

    tn, fp, fn, tp = (
        calculate_confusion(
            y,
            final_prediction,
        )
    )

    final_metrics = {

        "target":
            "target_3m_severe_anomaly",

        "probability_type":
            final_probability_type,

        "threshold":
            final_threshold,

        "observations":
            int(len(y)),

        "events":
            int(y.sum()),

        "alerts":
            int(final_prediction.sum()),

        "alert_rate":
            float(final_prediction.mean()),

        "true_negatives":
            tn,

        "false_positives":
            fp,

        "false_negatives":
            fn,

        "true_positives":
            tp,

        "precision":
            float(
                precision_score(
                    y,
                    final_prediction,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    y,
                    final_prediction,
                    zero_division=0,
                )
            ),

        "f1":
            float(
                f1_score(
                    y,
                    final_prediction,
                    zero_division=0,
                )
            ),

        "pr_auc":
            float(
                average_precision_score(
                    y,
                    final_probability,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    y,
                    final_probability,
                )
            ),

        "brier_score":
            float(
                brier_score_loss(
                    y,
                    final_probability,
                )
            ),

        "selection_period":
            "2014-2015",

        "test_period":
            "2016-2017",
    }

    # ------------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # SAVE THRESHOLD ANALYSIS
    # ------------------------------------------------------------

    threshold_analysis.to_csv(
        THRESHOLD_RESULTS_FILE,
        index=False,
    )

    # ------------------------------------------------------------
    # SAVE POLICY METRICS
    # ------------------------------------------------------------

    pd.DataFrame(
        [final_metrics]
    ).to_csv(
        POLICY_METRICS_FILE,
        index=False,
    )

    # ------------------------------------------------------------
    # SAVE SELECTED POLICY
    # ------------------------------------------------------------

    selected_output = pd.DataFrame(
        [
            final_metrics
        ]
    )

    selected_output[
        "minimum_recall_requirement"
    ] = MIN_RECALL

    selected_output[
        "minimum_precision_requirement"
    ] = MIN_PRECISION

    selected_output.to_csv(
        SELECTED_POLICY_FILE,
        index=False,
    )

    # ------------------------------------------------------------
    # SAVE POLICY PREDICTIONS
    # ------------------------------------------------------------

    policy_predictions.to_csv(
        CALIBRATION_POLICY_FILE,
        index=False,
    )

    # ------------------------------------------------------------
    # FINAL REPORT
    # ------------------------------------------------------------

    header(
        "FINAL POLICY VALIDATION"
    )

    print(
        "TARGET:",
        "target_3m_severe_anomaly"
    )

    print(
        "PROBABILITY:",
        final_probability_type
    )

    print(
        "THRESHOLD:",
        f"{final_threshold:.6f}"
    )

    print(
        "PR-AUC:",
        f"{final_metrics['pr_auc']:.6f}"
    )

    print(
        "ROC-AUC:",
        f"{final_metrics['roc_auc']:.6f}"
    )

    print(
        "BRIER:",
        f"{final_metrics['brier_score']:.6f}"
    )

    print(
        "PRECISION:",
        f"{final_metrics['precision']:.6f}"
    )

    print(
        "RECALL:",
        f"{final_metrics['recall']:.6f}"
    )

    print(
        "F1:",
        f"{final_metrics['f1']:.6f}"
    )

    print(
        "ALERT RATE:",
        f"{final_metrics['alert_rate']:.6f}"
    )

    print(
        "\nCONFUSION MATRIX:"
    )

    print(
        np.array(
            [
                [tn, fp],
                [fn, tp],
            ]
        )
    )

    print(
        "\nPREDICTION STATUS:"
    )

    print(
        policy_predictions[
            "prediction_status"
        ]
        .value_counts()
    )

    # ------------------------------------------------------------
    # OUTPUT CHECKS
    # ------------------------------------------------------------

    header(
        "OUTPUT VALIDATION"
    )

    output_files = [
        THRESHOLD_RESULTS_FILE,
        POLICY_METRICS_FILE,
        SELECTED_POLICY_FILE,
        CALIBRATION_POLICY_FILE,
    ]

    for file in output_files:

        if not file.exists():

            raise RuntimeError(
                f"Output file was not created:\n{file}"
            )

        print(
            "PASS:",
            file
        )

    # ------------------------------------------------------------
    # Policy prediction checks
    # ------------------------------------------------------------

    if (
        len(policy_predictions)
        != len(df)
    ):

        raise ValueError(
            "Policy prediction row count mismatch."
        )

    if (
        policy_predictions["month"]
        .min()
        < 1
    ):

        raise ValueError(
            "Invalid month remains in policy output."
        )

    if (
        policy_predictions["month"]
        .max()
        > 12
    ):

        raise ValueError(
            "Invalid month remains in policy output."
        )

    if (
        policy_predictions[
            "selected_probability"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "NULL selected probabilities found."
        )

    print(
        "POLICY ROW COUNT: PASS"
    )

    print(
        "POLICY MONTH RANGE: PASS"
    )

    print(
        "POLICY PROBABILITIES: PASS"
    )

    # ------------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------------

    header(
        "STEP 9 POLICY OPTIMIZATION COMPLETE"
    )

    print(
        "STATUS: PASS"
    )

    print(
        "\nFINAL POLICY:"
    )

    print(
        f"{final_probability_type}"
        f" >= "
        f"{final_threshold:.6f}"
    )

    print(
        "\nOUTPUT DIRECTORY:"
    )

    print(
        OUTPUT_DIR
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    main()