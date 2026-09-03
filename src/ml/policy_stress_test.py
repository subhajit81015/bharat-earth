from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# =========================================================
# PATHS
# =========================================================

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
    / "policy_stress_test.csv"
)


# =========================================================
# CONFIG
# =========================================================

TARGET = "actual"

PROBABILITY_COLUMN = "sigmoid_probability"

SELECTED_THRESHOLD = 0.09


# =========================================================
# MONTH MAPPING
# =========================================================

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


MONTH_NAME_TO_SEASON = {
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


# =========================================================
# METRICS
# =========================================================

def calculate_metrics(
    group,
    threshold,
):

    y_true = (
        group[TARGET]
        .astype(int)
        .to_numpy()
    )

    probability = (
        group[PROBABILITY_COLUMN]
        .astype(float)
        .to_numpy()
    )

    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0,
    )

    event_rate = (
        y_true.mean()
        if len(y_true) > 0
        else 0
    )

    alert_rate = (
        prediction.mean()
        if len(prediction) > 0
        else 0
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    return {
        "observations": len(group),
        "events": int(y_true.sum()),
        "event_rate": event_rate,
        "alerts": int(prediction.sum()),
        "alert_rate": alert_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_negative": int(tn),
        "false_positive_rate":
            false_positive_rate,
        "event_capture_rate":
            recall,
        "average_probability":
            probability.mean(),
        "maximum_probability":
            probability.max(),
    }


# =========================================================
# GROUP EVALUATION
# =========================================================

def evaluate_by_column(
    df,
    column,
    threshold,
):

    results = []

    for value, group in df.groupby(
        column,
        dropna=False,
        observed=True,
    ):

        metrics = calculate_metrics(
            group,
            threshold,
        )

        row = {
            column: value,
        }

        row.update(metrics)

        results.append(row)

    return pd.DataFrame(results)


# =========================================================
# PRINT
# =========================================================

def print_section(title):

    print(
        "\n"
        + "=" * 75
    )

    print(title)

    print(
        "=" * 75
    )


def print_dataframe(df):

    if df.empty:

        print("NO DATA")

        return

    print(
        df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # LOAD
    # =====================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "POLICY STRESS TEST"
    )

    print(
        "ROWS:",
        len(df),
    )

    print(
        "COLUMNS:",
        list(df.columns),
    )

    # =====================================================
    # VALIDATE
    # =====================================================

    required = {
        "subdivision",
        "year",
        "month",
        TARGET,
        PROBABILITY_COLUMN,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing columns: "
            f"{sorted(missing)}"
        )

    # =====================================================
    # CLEAN
    # =====================================================

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    df[PROBABILITY_COLUMN] = pd.to_numeric(
        df[PROBABILITY_COLUMN],
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
            TARGET,
            PROBABILITY_COLUMN,
            "year",
            "month",
        ]
    ).copy()

    df[TARGET] = (
        df[TARGET]
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

    # =====================================================
    # RESTORE MONTH NAME
    # =====================================================

    df["month_name"] = (
        df["month"]
        .map(MONTH_NUMBER_TO_NAME)
    )

    unknown_months = (
        df.loc[
            df["month_name"].isna(),
            "month",
        ]
        .unique()
        .tolist()
    )

    if unknown_months:

        raise ValueError(
            "Unknown encoded month values: "
            f"{unknown_months}"
        )

    # =====================================================
    # CREATE SEASON
    # =====================================================

    df["season"] = (
        df["month_name"]
        .map(MONTH_NAME_TO_SEASON)
    )

    if df["season"].isna().any():

        raise ValueError(
            "Unable to determine season."
        )

    # =====================================================
    # ALERT
    # =====================================================

    df["alert"] = (
        df[PROBABILITY_COLUMN]
        >= SELECTED_THRESHOLD
    ).astype(int)

    print(
        "\nSELECTED THRESHOLD:",
        SELECTED_THRESHOLD,
    )

    # =====================================================
    # OVERALL
    # =====================================================

    overall = calculate_metrics(
        df,
        SELECTED_THRESHOLD,
    )

    print_section(
        "OVERALL POLICY PERFORMANCE"
    )

    for key, value in overall.items():

        print(
            f"{key}: {value}"
        )

    # =====================================================
    # SEASON
    # =====================================================

    season_results = evaluate_by_column(
        df,
        "season",
        SELECTED_THRESHOLD,
    )

    season_results["season"] = pd.Categorical(
        season_results["season"],
        categories=SEASON_ORDER,
        ordered=True,
    )

    season_results = (
        season_results
        .sort_values("season")
        .reset_index(drop=True)
    )

    print_section(
        "SEASONAL POLICY PERFORMANCE"
    )

    print_dataframe(
        season_results[
            [
                "season",
                "observations",
                "events",
                "event_rate",
                "alerts",
                "alert_rate",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "event_capture_rate",
                "average_probability",
                "maximum_probability",
            ]
        ]
    )

    # =====================================================
    # REGION
    # =====================================================

    region_results = evaluate_by_column(
        df,
        "subdivision",
        SELECTED_THRESHOLD,
    )

    print_section(
        "TOP 15 REGIONS BY ACTUAL EVENT RATE"
    )

    top_regions = (
        region_results
        .sort_values(
            [
                "event_rate",
                "events",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(15)
    )

    print_dataframe(
        top_regions[
            [
                "subdivision",
                "observations",
                "events",
                "event_rate",
                "alerts",
                "alert_rate",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "event_capture_rate",
                "average_probability",
                "maximum_probability",
            ]
        ]
    )

    # =====================================================
    # LOWEST PRECISION
    # =====================================================

    print_section(
        "LOWEST PRECISION REGIONS"
    )

    lowest_precision = (
        region_results[
            region_results["alerts"] > 0
        ]
        .sort_values(
            "precision",
            ascending=True,
        )
        .head(15)
    )

    print_dataframe(
        lowest_precision[
            [
                "subdivision",
                "observations",
                "events",
                "event_rate",
                "alerts",
                "alert_rate",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
            ]
        ]
    )

    # =====================================================
    # BEST RECALL
    # =====================================================

    print_section(
        "BEST RECALL REGIONS"
    )

    best_recall = (
        region_results[
            region_results["events"] > 0
        ]
        .sort_values(
            "recall",
            ascending=False,
        )
        .head(15)
    )

    print_dataframe(
        best_recall[
            [
                "subdivision",
                "observations",
                "events",
                "event_rate",
                "alerts",
                "alert_rate",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
            ]
        ]
    )

    # =====================================================
    # WORST RECALL
    # =====================================================

    print_section(
        "WORST RECALL REGIONS"
    )

    worst_recall = (
        region_results[
            region_results["events"] > 0
        ]
        .sort_values(
            "recall",
            ascending=True,
        )
        .head(15)
    )

    print_dataframe(
        worst_recall[
            [
                "subdivision",
                "observations",
                "events",
                "event_rate",
                "alerts",
                "alert_rate",
                "precision",
                "recall",
                "f1",
            ]
        ]
    )

    # =====================================================
    # YEARLY
    # =====================================================

    yearly_results = evaluate_by_column(
        df,
        "year",
        SELECTED_THRESHOLD,
    )

    yearly_results = (
        yearly_results
        .sort_values("year")
        .reset_index(drop=True)
    )

    print_section(
        "YEARLY POLICY PERFORMANCE"
    )

    print_dataframe(
        yearly_results[
            [
                "year",
                "observations",
                "events",
                "event_rate",
                "alerts",
                "alert_rate",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "event_capture_rate",
            ]
        ]
    )

    # =====================================================
    # TIME PERIOD
    # =====================================================

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

    def assign_period(year):

        if year <= boundary_1:

            return "EARLY"

        if year <= boundary_2:

            return "MIDDLE"

        return "LATE"

    df["time_period"] = (
        df["year"]
        .apply(assign_period)
    )

    period_results = evaluate_by_column(
        df,
        "time_period",
        SELECTED_THRESHOLD,
    )

    period_order = [
        "EARLY",
        "MIDDLE",
        "LATE",
    ]

    period_results["time_period"] = (
        pd.Categorical(
            period_results["time_period"],
            categories=period_order,
            ordered=True,
        )
    )

    period_results = (
        period_results
        .sort_values("time_period")
        .reset_index(drop=True)
    )

    print_section(
        "TEMPORAL STABILITY"
    )

    print_dataframe(
        period_results[
            [
                "time_period",
                "observations",
                "events",
                "event_rate",
                "alerts",
                "alert_rate",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "event_capture_rate",
                "average_probability",
                "maximum_probability",
            ]
        ]
    )

    # =====================================================
    # PROBABILITY BANDS
    # =====================================================

    band_definitions = [
        (
            "VERY_LOW_<2%",
            0.00,
            0.02,
        ),
        (
            "LOW_2-5%",
            0.02,
            0.05,
        ),
        (
            "MODERATE_5-10%",
            0.05,
            0.10,
        ),
        (
            "ELEVATED_10-15%",
            0.10,
            0.15,
        ),
        (
            "HIGH_15-20%",
            0.15,
            0.20,
        ),
        (
            "CRITICAL_>=20%",
            0.20,
            1.01,
        ),
    ]

    band_rows = []

    for name, lower, upper in band_definitions:

        if upper > 1:

            mask = (
                df[PROBABILITY_COLUMN]
                >= lower
            )

        else:

            mask = (
                (df[PROBABILITY_COLUMN] >= lower)
                & (
                    df[PROBABILITY_COLUMN]
                    < upper
                )
            )

        group = df.loc[mask]

        if len(group) == 0:

            continue

        actual_rate = (
            group[TARGET].mean()
        )

        predicted_rate = (
            group[PROBABILITY_COLUMN]
            .mean()
        )

        band_rows.append(
            {
                "probability_band": name,
                "observations": len(group),
                "events": int(
                    group[TARGET].sum()
                ),
                "actual_event_rate":
                    actual_rate,
                "mean_predicted_probability":
                    predicted_rate,
                "calibration_gap":
                    abs(
                        actual_rate
                        - predicted_rate
                    ),
            }
        )

    band_results = pd.DataFrame(
        band_rows
    )

    print_section(
        "PROBABILITY BAND VALIDATION"
    )

    print_dataframe(
        band_results
    )

    # =====================================================
    # THRESHOLD SENSITIVITY
    # =====================================================

    thresholds = [
        0.05,
        0.06,
        0.07,
        0.08,
        0.09,
        0.10,
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
    ]

    threshold_rows = []

    for threshold in thresholds:

        metrics = calculate_metrics(
            df,
            threshold,
        )

        threshold_rows.append(
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
                "event_capture_rate":
                    metrics[
                        "event_capture_rate"
                    ],
                "false_positive_rate":
                    metrics[
                        "false_positive_rate"
                    ],
            }
        )

    threshold_results = pd.DataFrame(
        threshold_rows
    )

    print_section(
        "THRESHOLD SENSITIVITY"
    )

    print_dataframe(
        threshold_results
    )

    # =====================================================
    # STABILITY
    # =====================================================

    min_season_recall = (
        season_results["recall"]
        .min()
    )

    min_temporal_recall = (
        period_results["recall"]
        .min()
    )

    min_season_f1 = (
        season_results["f1"]
        .min()
    )

    min_temporal_f1 = (
        period_results["f1"]
        .min()
    )

    print_section(
        "POLICY STABILITY CHECK"
    )

    print(
        "Minimum seasonal recall:",
        f"{min_season_recall:.4f}",
    )

    print(
        "Minimum seasonal F1:",
        f"{min_season_f1:.4f}",
    )

    print(
        "Minimum temporal recall:",
        f"{min_temporal_recall:.4f}",
    )

    print(
        "Minimum temporal F1:",
        f"{min_temporal_f1:.4f}",
    )

    if (
        min_season_recall >= 0.30
        and min_temporal_recall >= 0.30
    ):

        policy_status = "STABLE"

    elif (
        min_season_recall >= 0.15
        and min_temporal_recall >= 0.15
    ):

        policy_status = (
            "MODERATELY_STABLE"
        )

    else:

        policy_status = "UNSTABLE"

    print(
        "POLICY STATUS:",
        policy_status,
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    output_frames = []

    season_save = season_results.copy()
    season_save["analysis_type"] = "SEASON"
    output_frames.append(season_save)

    region_save = region_results.copy()
    region_save["analysis_type"] = "REGION"
    output_frames.append(region_save)

    yearly_save = yearly_results.copy()
    yearly_save["analysis_type"] = "YEAR"
    output_frames.append(yearly_save)

    period_save = period_results.copy()
    period_save["analysis_type"] = "TIME_PERIOD"
    output_frames.append(period_save)

    band_save = band_results.copy()
    band_save["analysis_type"] = "PROBABILITY_BAND"
    output_frames.append(band_save)

    threshold_save = threshold_results.copy()
    threshold_save["analysis_type"] = "THRESHOLD"
    output_frames.append(threshold_save)

    combined = pd.concat(
        output_frames,
        ignore_index=True,
        sort=False,
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # =====================================================
    # COMPLETE
    # =====================================================

    print_section(
        "POLICY STRESS TEST COMPLETE"
    )

    print(
        "OUTPUT:",
        OUTPUT_FILE,
    )

    print(
        "THRESHOLD:",
        SELECTED_THRESHOLD,
    )

    print(
        "POLICY STATUS:",
        policy_status,
    )


if __name__ == "__main__":
    main()