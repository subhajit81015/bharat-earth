from pathlib import Path

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


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v2.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "calibrated_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "final_policy_validation.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "target_3m_severe_anomaly"

PROBABILITY_COLUMN = "sigmoid_probability"

GLOBAL_POLICY_THRESHOLD = 0.09

RANDOM_STATE = 42

TRAIN_FRACTION = 0.693

VALIDATION_FRACTION = 0.155

TEST_FRACTION = 0.152


# ============================================================
# MONTH / SEASON
# ============================================================

MONTH_NUMBER_TO_NAME = {
    0: "JAN",
    1: "FEB",
    2: "MAR",
    3: "APR",
    4: "MAY",
    5: "JUN",
    6: "JUL",
    7: "AUG",
    8: "SEP",
    9: "OCT",
    10: "NOV",
    11: "DEC",
}


MONTH_TO_SEASON = {
    "JAN": "WINTER",
    "FEB": "WINTER",
    "MAR": "PRE_MONSOON",
    "APR": "PRE_MONSOON",
    "MAY": "PRE_MONSOON",
    "JUN": "MONSOON",
    "JUL": "MONSOON",
    "AUG": "MONSOON",
    "SEP": "MONSOON",
    "OCT": "POST_MONSOON",
    "NOV": "POST_MONSOON",
    "DEC": "WINTER",
}


SEASON_ORDER = [
    "WINTER",
    "PRE_MONSOON",
    "MONSOON",
    "POST_MONSOON",
]


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

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

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    try:
        roc_auc = roc_auc_score(
            y_true,
            probabilities,
        )
    except ValueError:
        roc_auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    observations = len(y_true)

    events = int(
        y_true.sum()
    )

    alerts = int(
        predictions.sum()
    )

    alert_rate = (
        alerts / observations
        if observations
        else 0.0
    )

    event_rate = (
        events / observations
        if observations
        else 0.0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "observations": observations,
        "events": events,
        "event_rate": event_rate,
        "alerts": alerts,
        "alert_rate": alert_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "false_positive_rate": false_positive_rate,
    }


# ============================================================
# PREPARE PREDICTIONS
# ============================================================

def prepare_predictions():

    if not PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Predictions file not found:\n"
            f"{PREDICTIONS_FILE}"
        )

    df = pd.read_csv(
        PREDICTIONS_FILE
    )

    required = {
        "subdivision",
        "year",
        "month",
        "actual",
        "sigmoid_probability",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing prediction columns: "
            f"{sorted(missing)}"
        )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    df["sigmoid_probability"] = pd.to_numeric(
        df["sigmoid_probability"],
        errors="coerce",
    )

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["month"] = pd.to_numeric(
        df["month"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "actual",
            "sigmoid_probability",
            "year",
            "month",
        ]
    ).copy()

    df["actual"] = (
        df["actual"]
        .astype(int)
    )

    df["year"] = (
        df["year"]
        .astype(int)
    )

    df["month"] = (
        df["month"]
        .astype(int)
    )

    df["month_name"] = (
        df["month"]
        .map(
            MONTH_NUMBER_TO_NAME
        )
    )

    df["season"] = (
        df["month_name"]
        .map(MONTH_TO_SEASON)
    )

    if df["season"].isna().any():

        raise ValueError(
            "Unable to map some months to seasons."
        )

    return df


# ============================================================
# RECONSTRUCT DATA SPLIT
# ============================================================

def create_temporal_split(df):

    """
    Reconstruct the same chronological split used by
    the model-development pipeline.

    The prediction file represents the final TEST period.
    Therefore this function is primarily used to document
    the test period and its temporal characteristics.
    """

    df = df.sort_values(
        [
            "year",
            "month",
            "subdivision",
        ]
    ).reset_index(drop=True)

    return df


# ============================================================
# GLOBAL POLICY
# ============================================================

def evaluate_global_policy(
    df,
    threshold,
):

    metrics = calculate_metrics(
        df["actual"],
        df[PROBABILITY_COLUMN],
        threshold,
    )

    print()
    print("=" * 75)
    print("FINAL GLOBAL POLICY")
    print("=" * 75)

    print(
        f"THRESHOLD: {threshold:.2f}"
    )

    for key, value in metrics.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.6f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    return metrics


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(
    df,
):

    thresholds = np.arange(
        0.02,
        0.151,
        0.01,
    )

    rows = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            df["actual"],
            df[PROBABILITY_COLUMN],
            threshold,
        )

        rows.append(
            {
                "threshold": threshold,
                "precision":
                    metrics["precision"],
                "recall":
                    metrics["recall"],
                "f1":
                    metrics["f1"],
                "alert_rate":
                    metrics["alert_rate"],
                "false_positive_rate":
                    metrics["false_positive_rate"],
            }
        )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# SEASONAL EVALUATION
# ============================================================

def evaluate_by_season(
    df,
    threshold,
):

    rows = []

    for season in SEASON_ORDER:

        group = df[
            df["season"] == season
        ]

        if group.empty:

            continue

        metrics = calculate_metrics(
            group["actual"],
            group[PROBABILITY_COLUMN],
            threshold,
        )

        rows.append(
            {
                "season": season,
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# REGIONAL EVALUATION
# ============================================================

def evaluate_by_region(
    df,
    threshold,
):

    rows = []

    for region, group in df.groupby(
        "subdivision",
        dropna=False,
    ):

        metrics = calculate_metrics(
            group["actual"],
            group[PROBABILITY_COLUMN],
            threshold,
        )

        rows.append(
            {
                "subdivision": region,
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TEMPORAL EVALUATION
# ============================================================

def evaluate_temporal_stability(
    df,
    threshold,
):

    min_year = int(
        df["year"].min()
    )

    max_year = int(
        df["year"].max()
    )

    year_range = (
        max_year - min_year
    )

    boundary_1 = (
        min_year
        + year_range / 3
    )

    boundary_2 = (
        min_year
        + 2 * year_range / 3
    )

    def period(year):

        if year <= boundary_1:
            return "EARLY"

        if year <= boundary_2:
            return "MIDDLE"

        return "LATE"

    temp = df.copy()

    temp["time_period"] = (
        temp["year"]
        .apply(period)
    )

    rows = []

    for period_name in [
        "EARLY",
        "MIDDLE",
        "LATE",
    ]:

        group = temp[
            temp["time_period"]
            == period_name
        ]

        if group.empty:

            continue

        metrics = calculate_metrics(
            group["actual"],
            group[PROBABILITY_COLUMN],
            threshold,
        )

        rows.append(
            {
                "time_period":
                    period_name,
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# POLICY DECISION
# ============================================================

def policy_decision(
    global_metrics,
    threshold_table,
):

    selected = threshold_table[
        np.isclose(
            threshold_table["threshold"],
            GLOBAL_POLICY_THRESHOLD,
        )
    ]

    if selected.empty:

        raise ValueError(
            "Global threshold not found."
        )

    selected = selected.iloc[0]

    print()
    print("=" * 75)
    print("FINAL POLICY DECISION")
    print("=" * 75)

    print(
        "Selected policy: GLOBAL"
    )

    print(
        f"Selected threshold: "
        f"{GLOBAL_POLICY_THRESHOLD:.2f}"
    )

    print(
        f"Precision: "
        f"{global_metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{global_metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{global_metrics['f1']:.4f}"
    )

    print(
        f"Alert rate: "
        f"{global_metrics['alert_rate']:.4f}"
    )

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("3.6 FINAL POLICY VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # Load test predictions
    # --------------------------------------------------------

    df = prepare_predictions()

    print()
    print(
        "TEST DATASET"
    )

    print(
        "ROWS:",
        len(df),
    )

    print(
        "EVENTS:",
        df["actual"].sum(),
    )

    print(
        "EVENT RATE:",
        f"{df['actual'].mean():.6f}",
    )

    print(
        "YEAR RANGE:",
        df["year"].min(),
        "-",
        df["year"].max(),
    )

    # --------------------------------------------------------
    # Global policy
    # --------------------------------------------------------

    global_metrics = (
        evaluate_global_policy(
            df,
            GLOBAL_POLICY_THRESHOLD,
        )
    )

    # --------------------------------------------------------
    # Threshold sensitivity
    # --------------------------------------------------------

    threshold_table = (
        threshold_analysis(
            df
        )
    )

    print()
    print("=" * 75)
    print("FINAL TEST THRESHOLD SENSITIVITY")
    print("=" * 75)

    print(
        threshold_table.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Seasonal stability
    # --------------------------------------------------------

    seasonal = evaluate_by_season(
        df,
        GLOBAL_POLICY_THRESHOLD,
    )

    print()
    print("=" * 75)
    print("FINAL TEST SEASONAL STABILITY")
    print("=" * 75)

    print(
        seasonal.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Regional stability
    # --------------------------------------------------------

    regional = evaluate_by_region(
        df,
        GLOBAL_POLICY_THRESHOLD,
    )

    print()
    print("=" * 75)
    print("FINAL TEST REGIONAL STABILITY")
    print("=" * 75)

    print(
        regional.sort_values(
            "event_rate",
            ascending=False,
        ).head(20).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Temporal stability
    # --------------------------------------------------------

    temporal = (
        evaluate_temporal_stability(
            df,
            GLOBAL_POLICY_THRESHOLD,
        )
    )

    print()
    print("=" * 75)
    print("FINAL TEST TEMPORAL STABILITY")
    print("=" * 75)

    print(
        temporal.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # Policy decision
    # --------------------------------------------------------

    selected = policy_decision(
        global_metrics,
        threshold_table,
    )

    # --------------------------------------------------------
    # Final confusion matrix
    # --------------------------------------------------------

    predictions = (
        df[PROBABILITY_COLUMN]
        >= GLOBAL_POLICY_THRESHOLD
    ).astype(int)

    matrix = confusion_matrix(
        df["actual"],
        predictions,
        labels=[0, 1],
    )

    print()
    print("=" * 75)
    print("FINAL CONFUSION MATRIX")
    print("=" * 75)

    print(
        matrix
    )

    print()
    print(
        "TN:",
        matrix[0, 0],
    )

    print(
        "FP:",
        matrix[0, 1],
    )

    print(
        "FN:",
        matrix[1, 0],
    )

    print(
        "TP:",
        matrix[1, 1],
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_rows = []

    output_rows.append(
        {
            "analysis_type":
                "FINAL_GLOBAL_POLICY",
            "policy":
                "GLOBAL",
            "threshold":
                GLOBAL_POLICY_THRESHOLD,
            "precision":
                global_metrics["precision"],
            "recall":
                global_metrics["recall"],
            "f1":
                global_metrics["f1"],
            "pr_auc":
                global_metrics["pr_auc"],
            "roc_auc":
                global_metrics["roc_auc"],
            "alert_rate":
                global_metrics["alert_rate"],
            "event_rate":
                global_metrics["event_rate"],
            "observations":
                global_metrics["observations"],
            "events":
                global_metrics["events"],
            "alerts":
                global_metrics["alerts"],
            "true_positive":
                global_metrics["true_positive"],
            "false_positive":
                global_metrics["false_positive"],
            "false_negative":
                global_metrics["false_negative"],
            "true_negative":
                global_metrics["true_negative"],
        }
    )

    for _, row in seasonal.iterrows():

        output_rows.append(
            {
                "analysis_type":
                    "SEASON",
                "policy":
                    "GLOBAL_0.09",
                "season":
                    row["season"],
                "observations":
                    row["observations"],
                "events":
                    row["events"],
                "event_rate":
                    row["event_rate"],
                "alerts":
                    row["alerts"],
                "alert_rate":
                    row["alert_rate"],
                "precision":
                    row["precision"],
                "recall":
                    row["recall"],
                "f1":
                    row["f1"],
            }
        )

    for _, row in temporal.iterrows():

        output_rows.append(
            {
                "analysis_type":
                    "TIME_PERIOD",
                "policy":
                    "GLOBAL_0.09",
                "time_period":
                    row["time_period"],
                "observations":
                    row["observations"],
                "events":
                    row["events"],
                "event_rate":
                    row["event_rate"],
                "alerts":
                    row["alerts"],
                "alert_rate":
                    row["alert_rate"],
                "precision":
                    row["precision"],
                "recall":
                    row["recall"],
                "f1":
                    row["f1"],
            }
        )

    final_output = pd.DataFrame(
        output_rows
    )

    final_output.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 75)
    print("3.6 FINAL POLICY VALIDATION COMPLETE")
    print("=" * 75)

    print(
        "OUTPUT:",
        OUTPUT_FILE,
    )

    print()
    print(
        "FINAL POLICY:",
        "GLOBAL THRESHOLD 0.09",
    )

    print(
        "FINAL TEST F1:",
        f"{global_metrics['f1']:.6f}",
    )

    print(
        "FINAL TEST PRECISION:",
        f"{global_metrics['precision']:.6f}",
    )

    print(
        "FINAL TEST RECALL:",
        f"{global_metrics['recall']:.6f}",
    )


if __name__ == "__main__":
    main()