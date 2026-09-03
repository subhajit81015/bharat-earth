from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

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
    / "seasonal_policy_results.csv"
)


# ============================================================
# CONFIG
# ============================================================

TARGET = "actual"

PROBABILITY_COLUMN = "sigmoid_probability"

GLOBAL_THRESHOLD = 0.09

THRESHOLDS = np.arange(
    0.02,
    0.151,
    0.01,
)


# ============================================================
# MONTH / SEASON MAPPING
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

    alerts = int(
        predictions.sum()
    )

    observations = len(
        y_true
    )

    events = int(
        y_true.sum()
    )

    alert_rate = (
        alerts / observations
        if observations
        else 0
    )

    return {
        "observations": observations,
        "events": events,
        "event_rate": (
            events / observations
            if observations
            else 0
        ),
        "alerts": alerts,
        "alert_rate": alert_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "INPUT:",
        INPUT_FILE,
    )

    print(
        "ROWS:",
        len(df),
    )

    print(
        "COLUMNS:",
        list(df.columns),
    )

    required_columns = {
        "subdivision",
        "year",
        "month",
        TARGET,
        PROBABILITY_COLUMN,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Missing columns: "
            f"{sorted(missing)}"
        )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    df = df.copy()

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    df[PROBABILITY_COLUMN] = pd.to_numeric(
        df[PROBABILITY_COLUMN],
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
            "month",
        ]
    ).copy()

    df[TARGET] = (
        df[TARGET]
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

    unknown = (
        df.loc[
            df["month_name"].isna(),
            "month",
        ]
        .unique()
        .tolist()
    )

    if unknown:

        raise ValueError(
            "Unknown month values: "
            f"{unknown}"
        )

    df["season"] = (
        df["month_name"]
        .map(MONTH_TO_SEASON)
    )

    if df["season"].isna().any():

        raise ValueError(
            "Some rows have no season."
        )

    return df


# ============================================================
# GLOBAL POLICY
# ============================================================

def evaluate_global_policy(df):

    metrics = calculate_metrics(
        df[TARGET],
        df[PROBABILITY_COLUMN],
        GLOBAL_THRESHOLD,
    )

    print()
    print("=" * 70)
    print("GLOBAL POLICY")
    print("=" * 70)

    print(
        f"THRESHOLD: {GLOBAL_THRESHOLD:.2f}"
    )

    for key, value in metrics.items():

        print(
            f"{key}: {value:.6f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    return metrics


# ============================================================
# SEASON THRESHOLD OPTIMIZATION
# ============================================================

def optimize_season(
    season_df,
    season,
):

    y_true = season_df[
        TARGET
    ].to_numpy()

    probabilities = season_df[
        PROBABILITY_COLUMN
    ].to_numpy()

    rows = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            y_true,
            probabilities,
            threshold,
        )

        rows.append(
            {
                "season": season,
                "threshold": threshold,
                **metrics,
            }
        )

    results = pd.DataFrame(
        rows
    )

    # Select maximum F1.
    best = results.sort_values(
        [
            "f1",
            "recall",
            "precision",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).iloc[0]

    return (
        results,
        best,
    )


# ============================================================
# MAIN SEASON OPTIMIZATION
# ============================================================

def run_season_optimization(df):

    all_results = []

    best_results = []

    print()
    print("=" * 70)
    print("SEASON-AWARE THRESHOLD OPTIMIZATION")
    print("=" * 70)

    for season in SEASON_ORDER:

        season_df = df[
            df["season"] == season
        ].copy()

        if season_df.empty:

            continue

        results, best = optimize_season(
            season_df,
            season,
        )

        all_results.append(
            results
        )

        best_results.append(
            best.to_dict()
        )

        print()
        print(
            f"SEASON: {season}"
        )

        print(
            f"OBSERVATIONS: "
            f"{len(season_df)}"
        )

        print(
            f"EVENTS: "
            f"{season_df[TARGET].sum()}"
        )

        print(
            f"EVENT RATE: "
            f"{season_df[TARGET].mean():.6f}"
        )

        print(
            f"BEST THRESHOLD: "
            f"{best['threshold']:.2f}"
        )

        print(
            f"PRECISION: "
            f"{best['precision']:.6f}"
        )

        print(
            f"RECALL: "
            f"{best['recall']:.6f}"
        )

        print(
            f"F1: "
            f"{best['f1']:.6f}"
        )

        print(
            f"ALERT RATE: "
            f"{best['alert_rate']:.6f}"
        )

    all_results_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    best_results_df = pd.DataFrame(
        best_results
    )

    return (
        all_results_df,
        best_results_df,
    )


# ============================================================
# APPLY SEASON POLICY
# ============================================================

def apply_season_policy(
    df,
    best_results_df,
):

    threshold_map = dict(
        zip(
            best_results_df["season"],
            best_results_df["threshold"],
        )
    )

    df = df.copy()

    df["season_threshold"] = (
        df["season"]
        .map(threshold_map)
    )

    df["season_alert"] = (
        df[PROBABILITY_COLUMN]
        >= df["season_threshold"]
    ).astype(int)

    return df


# ============================================================
# EVALUATE SEASON POLICY
# ============================================================

def evaluate_season_policy(
    df,
):

    y_true = df[
        TARGET
    ].to_numpy()

    predictions = df[
        "season_alert"
    ].to_numpy()

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

    print()
    print("=" * 70)
    print("SEASON-AWARE POLICY")
    print("=" * 70)

    print(
        f"OBSERVATIONS: {len(df)}"
    )

    print(
        f"EVENTS: {y_true.sum()}"
    )

    print(
        f"ALERTS: {predictions.sum()}"
    )

    print(
        f"PRECISION: {precision:.6f}"
    )

    print(
        f"RECALL: {recall:.6f}"
    )

    print(
        f"F1: {f1:.6f}"
    )

    print(
        f"ALERT RATE: {alert_rate:.6f}"
    )

    return {
        "policy": "SEASON_AWARE",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": alert_rate,
    }


# ============================================================
# COMPARE GLOBAL VS SEASON
# ============================================================

def compare_policies(
    global_metrics,
    seasonal_metrics,
):

    comparison = pd.DataFrame(
        [
            {
                "policy": "GLOBAL_0.09",
                "precision":
                    global_metrics[
                        "precision"
                    ],
                "recall":
                    global_metrics[
                        "recall"
                    ],
                "f1":
                    global_metrics[
                        "f1"
                    ],
                "alert_rate":
                    global_metrics[
                        "alert_rate"
                    ],
            },
            seasonal_metrics,
        ]
    )

    print()
    print("=" * 70)
    print("POLICY COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    return comparison


# ============================================================
# SAVE
# ============================================================

def save_results(
    threshold_results,
    best_results,
    comparison,
):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        OUTPUT_FILE.with_suffix(".xlsx")
    ) as writer:

        threshold_results.to_excel(
            writer,
            sheet_name="threshold_analysis",
            index=False,
        )

        best_results.to_excel(
            writer,
            sheet_name="best_thresholds",
            index=False,
        )

        comparison.to_excel(
            writer,
            sheet_name="policy_comparison",
            index=False,
        )

    threshold_results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(
        "RESULTS SAVED:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        OUTPUT_FILE.with_suffix(
            ".xlsx"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------

    df = prepare_data(
        df
    )

    print()
    print(
        "SEASONS:"
    )

    print(
        df["season"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # GLOBAL POLICY
    # --------------------------------------------------------

    global_metrics = (
        evaluate_global_policy(
            df
        )
    )

    # --------------------------------------------------------
    # SEASON OPTIMIZATION
    # --------------------------------------------------------

    (
        threshold_results,
        best_results,
    ) = run_season_optimization(
        df
    )

    # --------------------------------------------------------
    # BEST THRESHOLDS
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("BEST SEASON THRESHOLDS")
    print("=" * 70)

    print(
        best_results[
            [
                "season",
                "threshold",
                "precision",
                "recall",
                "f1",
                "alert_rate",
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # APPLY
    # --------------------------------------------------------

    policy_df = apply_season_policy(
        df,
        best_results,
    )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    seasonal_metrics = (
        evaluate_season_policy(
            policy_df
        )
    )

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    comparison = compare_policies(
        global_metrics,
        seasonal_metrics,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_results(
        threshold_results,
        best_results,
        comparison,
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3.5 SEASON-AWARE POLICY OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()