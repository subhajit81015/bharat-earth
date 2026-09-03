from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "calibrated_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "risk_alert_policy.csv"
)


TARGET = "actual"

PROBABILITY_COLUMN = "sigmoid_probability"


# ---------------------------------------------------------
# ALERT LEVELS
# ---------------------------------------------------------

def assign_alert_level(probability):
    """
    Convert calibrated severe-anomaly probability
    into an operational risk level.
    """

    if probability < 0.05:
        return "LOW"

    elif probability < 0.10:
        return "MODERATE"

    elif probability < 0.15:
        return "ELEVATED"

    elif probability < 0.25:
        return "HIGH"

    else:
        return "CRITICAL"


# ---------------------------------------------------------
# THRESHOLD ANALYSIS
# ---------------------------------------------------------

def threshold_analysis(
    y_true,
    probabilities,
):
    """
    Evaluate calibrated probability thresholds.
    """

    thresholds = np.arange(
        0.01,
        0.51,
        0.01,
    )

    results = []

    for threshold in thresholds:

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

        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1],
            ).ravel()
        )

        alert_rate = (
            predictions.mean()
        )

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        )

        event_capture_rate = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        results.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "alert_rate": alert_rate,
                "false_positive_rate":
                    false_positive_rate,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "event_capture_rate":
                    event_capture_rate,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------
# POLICY SELECTION
# ---------------------------------------------------------

def select_operational_threshold(
    results,
):
    """
    Select a practical alert threshold.

    We prefer:
    - reasonable precision
    - useful recall
    - manageable alert rate

    The primary selection criterion is F1.
    """

    candidates = results[
        (results["precision"] >= 0.10)
        & (results["recall"] >= 0.30)
        & (results["alert_rate"] <= 0.30)
    ].copy()

    if candidates.empty:

        print(
            "\nNO THRESHOLD SATISFIED "
            "ALL OPERATIONAL CONSTRAINTS."
        )

        best = results.loc[
            results["f1"].idxmax()
        ]

    else:

        best = candidates.loc[
            candidates["f1"].idxmax()
        ]

    return best


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Calibrated predictions not found: "
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "DATASET"
    )

    print(
        "ROWS:",
        len(df),
    )

    print(
        "COLUMNS:",
        list(df.columns),
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    required_columns = {
        TARGET,
        PROBABILITY_COLUMN,
        "subdivision",
        "year",
        "month",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # -----------------------------------------------------
    # CLEAN PROBABILITIES
    # -----------------------------------------------------

    df[PROBABILITY_COLUMN] = (
        pd.to_numeric(
            df[PROBABILITY_COLUMN],
            errors="coerce",
        )
    )

    df[TARGET] = (
        pd.to_numeric(
            df[TARGET],
            errors="coerce",
        )
    )

    df = df.dropna(
        subset=[
            TARGET,
            PROBABILITY_COLUMN,
        ]
    ).copy()

    # -----------------------------------------------------
    # BASIC INFORMATION
    # -----------------------------------------------------

    y_true = (
        df[TARGET]
        .astype(int)
        .to_numpy()
    )

    probabilities = (
        df[PROBABILITY_COLUMN]
        .to_numpy()
    )

    print(
        "\nTARGET"
    )

    print(
        "POSITIVE EVENTS:",
        int(y_true.sum()),
    )

    print(
        "TOTAL:",
        len(y_true),
    )

    print(
        "EVENT RATE:",
        y_true.mean(),
    )

    print(
        "\nCALIBRATED PROBABILITY"
    )

    print(
        "MIN:",
        probabilities.min(),
    )

    print(
        "MEAN:",
        probabilities.mean(),
    )

    print(
        "MEDIAN:",
        np.median(probabilities),
    )

    print(
        "MAX:",
        probabilities.max(),
    )

    # -----------------------------------------------------
    # THRESHOLD ANALYSIS
    # -----------------------------------------------------

    results = threshold_analysis(
        y_true,
        probabilities,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "CALIBRATED PROBABILITY "
        "THRESHOLD ANALYSIS"
    )

    print(
        "=" * 75
    )

    print(
        results.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # -----------------------------------------------------
    # BEST F1
    # -----------------------------------------------------

    best_f1 = results.loc[
        results["f1"].idxmax()
    ]

    print(
        "\n"
        + "=" * 75
    )

    print(
        "BEST F1 THRESHOLD"
    )

    print(
        "=" * 75
    )

    print(
        "THRESHOLD:",
        best_f1["threshold"],
    )

    print(
        "PRECISION:",
        best_f1["precision"],
    )

    print(
        "RECALL:",
        best_f1["recall"],
    )

    print(
        "F1:",
        best_f1["f1"],
    )

    print(
        "ALERT RATE:",
        best_f1["alert_rate"],
    )

    print(
        "EVENT CAPTURE:",
        best_f1["event_capture_rate"],
    )

    # -----------------------------------------------------
    # OPERATIONAL THRESHOLD
    # -----------------------------------------------------

    operational = (
        select_operational_threshold(
            results
        )
    )

    operational_threshold = float(
        operational["threshold"]
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SELECTED OPERATIONAL THRESHOLD"
    )

    print(
        "=" * 75
    )

    print(
        "THRESHOLD:",
        operational_threshold,
    )

    print(
        "PRECISION:",
        operational["precision"],
    )

    print(
        "RECALL:",
        operational["recall"],
    )

    print(
        "F1:",
        operational["f1"],
    )

    print(
        "ALERT RATE:",
        operational["alert_rate"],
    )

    print(
        "EVENT CAPTURE:",
        operational["event_capture_rate"],
    )

    # -----------------------------------------------------
    # ASSIGN ALERT LEVEL
    # -----------------------------------------------------

    df["alert_level"] = (
        df[PROBABILITY_COLUMN]
        .apply(assign_alert_level)
    )

    # -----------------------------------------------------
    # FINAL ALERT
    # -----------------------------------------------------

    df["severe_anomaly_alert"] = (
        df[PROBABILITY_COLUMN]
        >= operational_threshold
    ).astype(int)

    # -----------------------------------------------------
    # RISK LEVEL DISTRIBUTION
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 75
    )

    print(
        "ALERT LEVEL DISTRIBUTION"
    )

    print(
        "=" * 75
    )

    print(
        df["alert_level"]
        .value_counts()
        .to_string()
    )

    # -----------------------------------------------------
    # EVENT RATE BY ALERT LEVEL
    # -----------------------------------------------------

    event_by_level = (
        df.groupby(
            "alert_level",
            observed=True,
        )[TARGET]
        .agg(
            observations="count",
            events="sum",
            event_rate="mean",
        )
        .sort_values(
            "event_rate",
            ascending=False,
        )
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "ACTUAL EVENT RATE BY ALERT LEVEL"
    )

    print(
        "=" * 75
    )

    print(
        event_by_level.to_string(
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # -----------------------------------------------------
    # SELECTED ALERT CONFUSION MATRIX
    # -----------------------------------------------------

    selected_predictions = (
        df["severe_anomaly_alert"]
        .to_numpy()
    )

    cm = confusion_matrix(
        y_true,
        selected_predictions,
        labels=[0, 1],
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "SELECTED ALERT CONFUSION MATRIX"
    )

    print(
        "=" * 75
    )

    print(cm)

    # -----------------------------------------------------
    # REGION PERFORMANCE
    # -----------------------------------------------------

    region_summary = (
        df.groupby(
            "subdivision",
            observed=True,
        )
        .agg(
            observations=(
                TARGET,
                "count",
            ),
            actual_events=(
                TARGET,
                "sum",
            ),
            average_probability=(
                PROBABILITY_COLUMN,
                "mean",
            ),
            maximum_probability=(
                PROBABILITY_COLUMN,
                "max",
            ),
            alerts=(
                "severe_anomaly_alert",
                "sum",
            ),
        )
        .reset_index()
    )

    region_summary[
        "actual_event_rate"
    ] = (
        region_summary[
            "actual_events"
        ]
        / region_summary[
            "observations"
        ]
    )

    region_summary[
        "alert_rate"
    ] = (
        region_summary[
            "alerts"
        ]
        / region_summary[
            "observations"
        ]
    )

    region_summary = (
        region_summary
        .sort_values(
            "average_probability",
            ascending=False,
        )
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "TOP 15 REGIONS BY CALIBRATED RISK"
    )

    print(
        "=" * 75
    )

    print(
        region_summary.head(15).to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    # -----------------------------------------------------
    # SAVE OUTPUT
    # -----------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "RISK ALERT DATASET CREATED"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()