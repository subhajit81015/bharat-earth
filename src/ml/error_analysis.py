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
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 3.9 ERROR ANALYSIS
# ============================================================

print("=" * 70)
print("3.9 MODEL ERROR ANALYSIS")
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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "target_3m_severe_anomaly"

TEST_SIZE = 7668

THRESHOLD = 0.09

RANDOM_STATE = 42


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
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


# ============================================================
# SEASON ENCODING
# ============================================================

SEASON_MAPPING = {
    "WINTER": 0,
    "PRE_MONSOON": 1,
    "MONSOON": 2,
    "POST_MONSOON": 3,
}


SEASON_REVERSE = {
    0: "WINTER",
    1: "PRE_MONSOON",
    2: "MONSOON",
    3: "POST_MONSOON",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    features = pd.read_csv(
        FEATURE_FILE
    )

    target = pd.read_csv(
        TARGET_FILE
    )

    print(
        "FEATURE SHAPE:",
        features.shape
    )

    print(
        "TARGET SHAPE:",
        target.shape
    )

    merge_keys = [
        "subdivision",
        "year",
        "month",
    ]

    target_small = target[
        merge_keys + [TARGET]
    ].copy()

    df = features.merge(
        target_small,
        on=merge_keys,
        how="inner",
        validate="one_to_one",
    )

    print(
        "MERGED SHAPE:",
        df.shape
    )

    print(
        "TARGET RATE:",
        df[TARGET].mean()
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    data = df.copy()

    # --------------------------------------------------------
    # Preserve reporting fields
    # --------------------------------------------------------

    data["original_month"] = data["month"]

    # --------------------------------------------------------
    # Convert month
    # --------------------------------------------------------

    month_numbers = pd.to_numeric(
        data["month"],
        errors="coerce",
    )

    month_names = {
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

    month_text = (
        data["month"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    month_from_name = (
        month_text.map(month_names)
    )

    data["month"] = (
        month_numbers
        .fillna(month_from_name)
    )

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    data["season"] = (
        data["season"]
        .astype(str)
        .str.upper()
        .str.strip()
        .map(SEASON_MAPPING)
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    for column in FEATURES:

        if column == "season":
            continue

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if data["month"].isna().any():

        raise ValueError(
            "Month conversion failed."
        )

    if not data["month"].between(
        1,
        12,
    ).all():

        raise ValueError(
            "Month contains values outside 1-12."
        )

    if data["season"].isna().any():

        raise ValueError(
            "Unknown season detected."
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="coerce",
    )

    data = data.dropna(
        subset=[TARGET]
    ).copy()

    data[TARGET] = (
        data[TARGET]
        .astype(int)
    )

    # --------------------------------------------------------
    # Missing numerical values
    # --------------------------------------------------------

    for column in FEATURES:

        if data[column].isna().any():

            median_value = (
                data[column]
                .median()
            )

            if pd.isna(
                median_value
            ):
                median_value = 0

            data[column] = (
                data[column]
                .fillna(median_value)
            )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    data = data.sort_values(
        [
            "year",
            "month",
            "subdivision",
        ]
    ).reset_index(
        drop=True
    )

    return data


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(data):

    split_index = (
        len(data)
        - TEST_SIZE
    )

    train = data.iloc[
        :split_index
    ].copy()

    test = data.iloc[
        split_index:
    ].copy()

    print()
    print("=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(
        "TRAIN ROWS:",
        len(train)
    )

    print(
        "TEST ROWS:",
        len(test)
    )

    print(
        "TRAIN YEARS:",
        train["year"].min(),
        "->",
        train["year"].max(),
    )

    print(
        "TEST YEARS:",
        test["year"].min(),
        "->",
        test["year"].max(),
    )

    print(
        "TRAIN POSITIVE RATE:",
        train[TARGET].mean(),
    )

    print(
        "TEST POSITIVE RATE:",
        test[TARGET].mean(),
    )

    return train, test


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(train):

    X_train = train[FEATURES]
    y_train = train[TARGET]

    positive = int(
        y_train.sum()
    )

    negative = (
        len(y_train)
        - positive
    )

    scale_pos_weight = (
        negative / positive
    )

    print()
    print("=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    print(
        "POSITIVE:",
        positive
    )

    print(
        "NEGATIVE:",
        negative
    )

    print(
        "SCALE POS WEIGHT:",
        scale_pos_weight
    )

    model = XGBClassifier(
        n_estimators=127,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    return model


# ============================================================
# CREATE ERROR LABELS
# ============================================================

def create_predictions(
    model,
    test,
):

    X_test = test[FEATURES]

    y_test = (
        test[TARGET]
        .astype(int)
        .values
    )

    probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    prediction = (
        probability
        >= THRESHOLD
    ).astype(int)

    result = test.copy()

    result["actual"] = y_test

    result[
        "predicted_probability"
    ] = probability

    result[
        "predicted"
    ] = prediction

    # --------------------------------------------------------
    # Error type
    # --------------------------------------------------------

    result["error_type"] = np.select(
        [
            (y_test == 1)
            & (prediction == 1),

            (y_test == 0)
            & (prediction == 0),

            (y_test == 0)
            & (prediction == 1),

            (y_test == 1)
            & (prediction == 0),
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
# GLOBAL ERROR SUMMARY
# ============================================================

def global_error_summary(
    result
):

    y = result["actual"]
    p = result["predicted"]

    tp = int(
        ((y == 1) & (p == 1)).sum()
    )

    tn = int(
        ((y == 0) & (p == 0)).sum()
    )

    fp = int(
        ((y == 0) & (p == 1)).sum()
    )

    fn = int(
        ((y == 1) & (p == 0)).sum()
    )

    summary = {
        "observations": len(result),
        "events": int(y.sum()),
        "event_rate": y.mean(),
        "alerts": int(p.sum()),
        "alert_rate": p.mean(),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision_score(
            y,
            p,
            zero_division=0,
        ),
        "recall": recall_score(
            y,
            p,
            zero_division=0,
        ),
        "f1": f1_score(
            y,
            p,
            zero_division=0,
        ),
        "false_positive_rate": (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        ),
        "false_negative_rate": (
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0
        ),
        "average_probability": (
            result[
                "predicted_probability"
            ].mean()
        ),
        "maximum_probability": (
            result[
                "predicted_probability"
            ].max()
        ),
    }

    return pd.DataFrame(
        [summary]
    )


# ============================================================
# FALSE POSITIVES
# ============================================================

def false_positive_analysis(
    result
):

    fp = result[
        result["error_type"]
        == "FALSE_POSITIVE"
    ].copy()

    fp = fp.sort_values(
        "predicted_probability",
        ascending=False,
    )

    return fp


# ============================================================
# FALSE NEGATIVES
# ============================================================

def false_negative_analysis(
    result
):

    fn = result[
        result["error_type"]
        == "FALSE_NEGATIVE"
    ].copy()

    fn = fn.sort_values(
        "predicted_probability",
        ascending=False,
    )

    return fn


# ============================================================
# TRUE POSITIVES
# ============================================================

def true_positive_analysis(
    result
):

    tp = result[
        result["error_type"]
        == "TRUE_POSITIVE"
    ].copy()

    return tp.sort_values(
        "predicted_probability",
        ascending=False,
    )


# ============================================================
# TRUE NEGATIVES
# ============================================================

def true_negative_analysis(
    result
):

    tn = result[
        result["error_type"]
        == "TRUE_NEGATIVE"
    ].copy()

    return tn.sort_values(
        "predicted_probability",
        ascending=False,
    )


# ============================================================
# ERROR BY GROUP
# ============================================================

def group_error_analysis(
    result,
    group_column,
):

    rows = []

    for value, group in result.groupby(
        group_column
    ):

        y = group["actual"]
        p = group["predicted"]

        tp = (
            (y == 1)
            & (p == 1)
        ).sum()

        tn = (
            (y == 0)
            & (p == 0)
        ).sum()

        fp = (
            (y == 0)
            & (p == 1)
        ).sum()

        fn = (
            (y == 1)
            & (p == 0)
        ).sum()

        events = int(
            y.sum()
        )

        alerts = int(
            p.sum()
        )

        rows.append(
            {
                group_column: value,
                "observations": len(group),
                "events": events,
                "event_rate": y.mean(),
                "alerts": alerts,
                "alert_rate": p.mean(),
                "true_positive": int(tp),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_negative": int(tn),
                "precision": precision_score(
                    y,
                    p,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y,
                    p,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y,
                    p,
                    zero_division=0,
                ),
                "false_positive_rate": (
                    fp / (fp + tn)
                    if (fp + tn) > 0
                    else 0
                ),
                "false_negative_rate": (
                    fn / (fn + tp)
                    if (fn + tp) > 0
                    else 0
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PROBABILITY BINS
# ============================================================

def probability_error_analysis(
    result
):

    bins = [
        0.00,
        0.02,
        0.04,
        0.06,
        0.08,
        0.09,
        0.10,
        0.12,
        0.15,
        0.20,
        0.30,
        0.50,
        0.70,
        1.01,
    ]

    labels = [
        "0.00-0.02",
        "0.02-0.04",
        "0.04-0.06",
        "0.06-0.08",
        "0.08-0.09",
        "0.09-0.10",
        "0.10-0.12",
        "0.12-0.15",
        "0.15-0.20",
        "0.20-0.30",
        "0.30-0.50",
        "0.50-0.70",
        "0.70+",
    ]

    data = result.copy()

    data["probability_bin"] = pd.cut(
        data[
            "predicted_probability"
        ],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    rows = []

    for probability_bin, group in data.groupby(
        "probability_bin",
        observed=False,
    ):

        y = group["actual"]
        p = group["predicted"]

        rows.append(
            {
                "probability_bin":
                    str(probability_bin),

                "observations":
                    len(group),

                "events":
                    int(y.sum()),

                "actual_event_rate":
                    y.mean(),

                "alerts":
                    int(p.sum()),

                "alert_rate":
                    p.mean(),

                "average_probability":
                    group[
                        "predicted_probability"
                    ].mean(),

                "false_positives":
                    int(
                        (
                            (y == 0)
                            & (p == 1)
                        ).sum()
                    ),

                "false_negatives":
                    int(
                        (
                            (y == 1)
                            & (p == 0)
                        ).sum()
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# HIGH CONFIDENCE FALSE POSITIVES
# ============================================================

def high_confidence_false_positives(
    result,
    n=100,
):

    fp = false_positive_analysis(
        result
    )

    columns = [
        "subdivision",
        "year",
        "month",
        "actual",
        "predicted_probability",
        "rainfall_mm",
        "rainfall_3m",
        "rainfall_6m",
        "rainfall_12m",
        "rainfall_anomaly",
        "rainfall_anomaly_pct",
        "rainfall_deficit_mm",
        "rainfall_zscore",
        "rainfall_prev_3m",
        "rainfall_prev_6m",
        "rainfall_prev_12m",
        "rainfall_trend_3m",
        "season",
    ]

    columns = [
        c for c in columns
        if c in fp.columns
    ]

    return fp[
        columns
    ].head(n)


# ============================================================
# HIGH CONFIDENCE FALSE NEGATIVES
# ============================================================

def high_confidence_false_negatives(
    result,
    n=100,
):

    fn = false_negative_analysis(
        result
    )

    columns = [
        "subdivision",
        "year",
        "month",
        "actual",
        "predicted_probability",
        "rainfall_mm",
        "rainfall_3m",
        "rainfall_6m",
        "rainfall_12m",
        "rainfall_anomaly",
        "rainfall_anomaly_pct",
        "rainfall_deficit_mm",
        "rainfall_zscore",
        "rainfall_prev_3m",
        "rainfall_prev_6m",
        "rainfall_prev_12m",
        "rainfall_trend_3m",
        "season",
    ]

    columns = [
        c for c in columns
        if c in fn.columns
    ]

    return fn[
        columns
    ].head(n)


# ============================================================
# ERROR COMPOSITION
# ============================================================

def error_composition(
    result
):

    counts = (
        result["error_type"]
        .value_counts()
        .rename_axis(
            "error_type"
        )
        .reset_index(
            name="count"
        )
    )

    counts["percentage"] = (
        counts["count"]
        / len(result)
    )

    return counts


# ============================================================
# THRESHOLD ERROR ANALYSIS
# ============================================================

def threshold_analysis(
    result
):

    y = result["actual"].values

    probability = (
        result[
            "predicted_probability"
        ].values
    )

    thresholds = [
        0.02,
        0.04,
        0.06,
        0.08,
        0.09,
        0.10,
        0.12,
        0.15,
        0.20,
        0.25,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
    ]

    rows = []

    for threshold in thresholds:

        p = (
            probability
            >= threshold
        ).astype(int)

        tp = (
            (y == 1)
            & (p == 1)
        ).sum()

        tn = (
            (y == 0)
            & (p == 0)
        ).sum()

        fp = (
            (y == 0)
            & (p == 1)
        ).sum()

        fn = (
            (y == 1)
            & (p == 0)
        ).sum()

        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y,
                    p,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y,
                    p,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y,
                    p,
                    zero_division=0,
                ),
                "alert_rate": p.mean(),
                "true_positive": int(tp),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_negative": int(tn),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PRINT RESULTS
# ============================================================

def print_global_summary(
    summary
):

    print()
    print("=" * 70)
    print("GLOBAL ERROR SUMMARY")
    print("=" * 70)

    row = summary.iloc[0]

    for column in summary.columns:

        value = row[column]

        if isinstance(
            value,
            (float, np.floating),
        ):

            print(
                f"{column}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{column}: "
                f"{value}"
            )


def print_top_errors(
    fp,
    fn,
):

    print()
    print("=" * 70)
    print("TOP 20 HIGH-CONFIDENCE FALSE POSITIVES")
    print("=" * 70)

    if len(fp) > 0:

        print(
            fp.head(20).to_string(
                index=False
            )
        )

    else:

        print(
            "NO FALSE POSITIVES"
        )

    print()
    print("=" * 70)
    print("TOP 20 FALSE NEGATIVES")
    print("=" * 70)

    if len(fn) > 0:

        print(
            fn.head(20).to_string(
                index=False
            )
        )

    else:

        print(
            "NO FALSE NEGATIVES"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    result,
    summary,
    fp,
    fn,
    tp,
    tn,
    error_comp,
    yearly,
    seasonal,
    subdivision,
    monthly,
    probability_bins,
    threshold_results,
):

    # --------------------------------------------------------
    # Full predictions
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_DIR
        / "all_test_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Error groups
    # --------------------------------------------------------

    summary.to_csv(
        OUTPUT_DIR
        / "global_error_summary.csv",
        index=False,
    )

    fp.to_csv(
        OUTPUT_DIR
        / "false_positives.csv",
        index=False,
    )

    fn.to_csv(
        OUTPUT_DIR
        / "false_negatives.csv",
        index=False,
    )

    tp.to_csv(
        OUTPUT_DIR
        / "true_positives.csv",
        index=False,
    )

    tn.to_csv(
        OUTPUT_DIR
        / "true_negatives.csv",
        index=False,
    )

    error_comp.to_csv(
        OUTPUT_DIR
        / "error_composition.csv",
        index=False,
    )

    yearly.to_csv(
        OUTPUT_DIR
        / "yearly_error_analysis.csv",
        index=False,
    )

    seasonal.to_csv(
        OUTPUT_DIR
        / "seasonal_error_analysis.csv",
        index=False,
    )

    subdivision.to_csv(
        OUTPUT_DIR
        / "subdivision_error_analysis.csv",
        index=False,
    )

    monthly.to_csv(
        OUTPUT_DIR
        / "monthly_error_analysis.csv",
        index=False,
    )

    probability_bins.to_csv(
        OUTPUT_DIR
        / "probability_error_analysis.csv",
        index=False,
    )

    threshold_results.to_csv(
        OUTPUT_DIR
        / "threshold_error_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    try:

        with pd.ExcelWriter(
            OUTPUT_DIR
            / "error_analysis_results.xlsx",
            engine="openpyxl",
        ) as writer:

            summary.to_excel(
                writer,
                sheet_name="global",
                index=False,
            )

            error_comp.to_excel(
                writer,
                sheet_name="error_composition",
                index=False,
            )

            fp.head(1000).to_excel(
                writer,
                sheet_name="false_positives",
                index=False,
            )

            fn.head(1000).to_excel(
                writer,
                sheet_name="false_negatives",
                index=False,
            )

            tp.head(1000).to_excel(
                writer,
                sheet_name="true_positives",
                index=False,
            )

            tn.head(1000).to_excel(
                writer,
                sheet_name="true_negatives",
                index=False,
            )

            yearly.to_excel(
                writer,
                sheet_name="yearly",
                index=False,
            )

            seasonal.to_excel(
                writer,
                sheet_name="seasonal",
                index=False,
            )

            subdivision.to_excel(
                writer,
                sheet_name="subdivision",
                index=False,
            )

            monthly.to_excel(
                writer,
                sheet_name="monthly",
                index=False,
            )

            probability_bins.to_excel(
                writer,
                sheet_name="probability_bins",
                index=False,
            )

            threshold_results.to_excel(
                writer,
                sheet_name="thresholds",
                index=False,
            )

        print()
        print(
            "Excel file saved."
        )

    except ImportError:

        print()
        print(
            "openpyxl not installed."
        )

        print(
            "CSV files were still saved."
        )

        print(
            "Install with:"
        )

        print(
            "pip install openpyxl"
        )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    files = [
        "all_test_predictions.csv",
        "global_error_summary.csv",
        "false_positives.csv",
        "false_negatives.csv",
        "true_positives.csv",
        "true_negatives.csv",
        "error_composition.csv",
        "yearly_error_analysis.csv",
        "seasonal_error_analysis.csv",
        "subdivision_error_analysis.csv",
        "monthly_error_analysis.csv",
        "probability_error_analysis.csv",
        "threshold_error_analysis.csv",
        "error_analysis_results.xlsx",
    ]

    for filename in files:

        path = (
            OUTPUT_DIR
            / filename
        )

        if path.exists():

            print(path)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    data = prepare_data(
        df
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, test = temporal_split(
        data
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        train
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    result = create_predictions(
        model,
        test,
    )

    # --------------------------------------------------------
    # Global summary
    # --------------------------------------------------------

    summary = global_error_summary(
        result
    )

    print_global_summary(
        summary
    )

    # --------------------------------------------------------
    # Error groups
    # --------------------------------------------------------

    fp = false_positive_analysis(
        result
    )

    fn = false_negative_analysis(
        result
    )

    tp = true_positive_analysis(
        result
    )

    tn = true_negative_analysis(
        result
    )

    # --------------------------------------------------------
    # Error composition
    # --------------------------------------------------------

    error_comp = error_composition(
        result
    )

    print()
    print("=" * 70)
    print("ERROR COMPOSITION")
    print("=" * 70)

    print(
        error_comp.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Year
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ERROR ANALYSIS BY YEAR")
    print("=" * 70)

    yearly = group_error_analysis(
        result,
        "year",
    )

    print(
        yearly.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ERROR ANALYSIS BY SEASON")
    print("=" * 70)

    seasonal = group_error_analysis(
        result,
        "season",
    )

    seasonal["season"] = (
        seasonal["season"]
        .map(SEASON_REVERSE)
        .fillna(
            seasonal["season"]
            .astype(str)
        )
    )

    print(
        seasonal.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Subdivision
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ERROR ANALYSIS BY SUBDIVISION")
    print("=" * 70)

    subdivision = group_error_analysis(
        result,
        "subdivision",
    )

    print(
        subdivision
        .sort_values(
            "false_positive_rate",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Month
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ERROR ANALYSIS BY MONTH")
    print("=" * 70)

    monthly = group_error_analysis(
        result,
        "month",
    )

    monthly = monthly.sort_values(
        "month"
    )

    print(
        monthly.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROBABILITY ERROR ANALYSIS")
    print("=" * 70)

    probability_bins = (
        probability_error_analysis(
            result
        )
    )

    print(
        probability_bins.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("THRESHOLD ERROR ANALYSIS")
    print("=" * 70)

    threshold_results = (
        threshold_analysis(
            result
        )
    )

    print(
        threshold_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Top errors
    # --------------------------------------------------------

    top_fp = (
        high_confidence_false_positives(
            result,
            n=100,
        )
    )

    top_fn = (
        high_confidence_false_negatives(
            result,
            n=100,
        )
    )

    print_top_errors(
        top_fp,
        top_fn,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        result,
        summary,
        fp,
        fn,
        tp,
        tn,
        error_comp,
        yearly,
        seasonal,
        subdivision,
        monthly,
        probability_bins,
        threshold_results,
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    row = summary.iloc[0]

    print()
    print("=" * 70)
    print("3.9 ERROR ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"THRESHOLD: {THRESHOLD:.2f}"
    )

    print(
        f"PRECISION: "
        f"{row['precision']:.6f}"
    )

    print(
        f"RECALL: "
        f"{row['recall']:.6f}"
    )

    print(
        f"F1: "
        f"{row['f1']:.6f}"
    )

    print(
        f"FALSE POSITIVES: "
        f"{int(row['false_positive'])}"
    )

    print(
        f"FALSE NEGATIVES: "
        f"{int(row['false_negative'])}"
    )

    print()
    print(
        "NEXT STAGE: 3.10 FINAL MODEL & POLICY SELECTION"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()