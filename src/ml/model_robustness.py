from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


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
    / "robustness"
)


# ============================================================
# CONFIG
# ============================================================

TARGET = "target_3m_severe_anomaly"

RANDOM_STATE = 42

TEST_SIZE = 7668

THRESHOLD = 0.09

MIN_YEAR_SAMPLES = 50


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


SEASON_MAPPING = {
    "WINTER": 0,
    "PRE_MONSOON": 1,
    "MONSOON": 2,
    "POST_MONSOON": 3,
}


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("=" * 70)
    print("3.8 MODEL ROBUSTNESS & TEMPORAL VALIDATION")
    print("=" * 70)

    features = pd.read_csv(
        FEATURE_FILE
    )

    target = pd.read_csv(
        TARGET_FILE
    )

    keys = [
        "subdivision",
        "year",
        "month",
    ]

    print()
    print("FEATURE DATASET:", features.shape)
    print("TARGET DATASET:", target.shape)

    target_small = target[
        keys + [TARGET]
    ].copy()

    df = features.merge(
        target_small,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    print()
    print("MERGED DATASET:", df.shape)
    print(
        "TARGET RATE:",
        df[TARGET].mean(),
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    data = df.copy()

    # --------------------------------------------------------
    # Preserve original month for reporting
    # --------------------------------------------------------

    data["month_original"] = data["month"]

    # --------------------------------------------------------
    # Convert month
    # --------------------------------------------------------

    month_numeric = pd.to_numeric(
        data["month"],
        errors="coerce",
    )

    # If month is string month names
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

    month_string = (
        data["month"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    month_from_name = (
        month_string.map(month_names)
    )

    month_numeric = (
        month_numeric
        .fillna(month_from_name)
    )

    data["month"] = month_numeric

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    data["season"] = (
        data["season"]
        .astype(str)
        .str.upper()
        .map(SEASON_MAPPING)
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for column in FEATURES:

        if column == "season":
            continue

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Validate month
    # --------------------------------------------------------

    if data["month"].isna().any():

        bad_count = int(
            data["month"].isna().sum()
        )

        raise ValueError(
            f"Could not convert {bad_count} month values."
        )

    if not data["month"].between(
        1,
        12,
    ).all():

        raise ValueError(
            "Month contains values outside 1-12."
        )

    # --------------------------------------------------------
    # Validate season
    # --------------------------------------------------------

    if data["season"].isna().any():

        raise ValueError(
            "Unknown season values detected."
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
    # Fill numeric missing values
    # --------------------------------------------------------

    for column in FEATURES:

        if data[column].isna().any():

            median = (
                data[column]
                .median()
            )

            if pd.isna(median):
                median = 0

            data[column] = (
                data[column]
                .fillna(median)
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
# TRAIN / TEST SPLIT
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
    print("TEMPORAL SPLIT")
    print("=" * 70)

    print(
        "TRAIN:",
        len(train),
    )

    print(
        "TEST:",
        len(test),
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
        "TRAIN TARGET RATE:",
        train[TARGET].mean(),
    )

    print(
        "TEST TARGET RATE:",
        test[TARGET].mean(),
    )

    return train, test


# ============================================================
# MODEL
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
    print("TRAINING ROBUSTNESS MODEL")
    print("=" * 70)

    print(
        "POSITIVE:",
        positive,
    )

    print(
        "NEGATIVE:",
        negative,
    )

    print(
        "SCALE POS WEIGHT:",
        scale_pos_weight,
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
# GLOBAL EVALUATION
# ============================================================

def evaluate_global(
    model,
    test,
):

    X_test = test[FEATURES]
    y_test = test[TARGET]

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        probabilities
        >= THRESHOLD
    ).astype(int)

    result = {
        "observations": len(test),
        "events": int(y_test.sum()),
        "event_rate": y_test.mean(),
        "alerts": int(predictions.sum()),
        "alert_rate": predictions.mean(),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    print()
    print("=" * 70)
    print("GLOBAL TEST PERFORMANCE")
    print("=" * 70)

    for key, value in result.items():

        print(
            f"{key}: {value:.6f}"
            if isinstance(value, float)
            else f"{key}: {value}"
        )

    return result, probabilities


# ============================================================
# YEAR ROBUSTNESS
# ============================================================

def evaluate_by_year(
    model,
    test,
):

    rows = []

    for year, group in test.groupby(
        "year"
    ):

        if len(group) < MIN_YEAR_SAMPLES:
            continue

        X = group[FEATURES]
        y = group[TARGET]

        probabilities = (
            model.predict_proba(X)[:, 1]
        )

        predictions = (
            probabilities
            >= THRESHOLD
        ).astype(int)

        events = int(
            y.sum()
        )

        if events > 0:

            pr_auc = (
                average_precision_score(
                    y,
                    probabilities,
                )
            )

            roc_auc = (
                roc_auc_score(
                    y,
                    probabilities,
                )
            )

        else:

            pr_auc = np.nan
            roc_auc = np.nan

        rows.append(
            {
                "year": year,
                "observations": len(group),
                "events": events,
                "event_rate": y.mean(),
                "alerts": int(
                    predictions.sum()
                ),
                "alert_rate": predictions.mean(),
                "precision": precision_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            }
        )

    result = pd.DataFrame(rows)

    return result


# ============================================================
# SEASON ROBUSTNESS
# ============================================================

def evaluate_by_season(
    model,
    test,
):

    rows = []

    season_names = {
        0: "WINTER",
        1: "PRE_MONSOON",
        2: "MONSOON",
        3: "POST_MONSOON",
    }

    for season_id, group in test.groupby(
        "season"
    ):

        X = group[FEATURES]
        y = group[TARGET]

        probabilities = (
            model.predict_proba(X)[:, 1]
        )

        predictions = (
            probabilities
            >= THRESHOLD
        ).astype(int)

        events = int(
            y.sum()
        )

        if events > 0:

            pr_auc = (
                average_precision_score(
                    y,
                    probabilities,
                )
            )

            if y.nunique() > 1:

                roc_auc = (
                    roc_auc_score(
                        y,
                        probabilities,
                    )
                )

            else:

                roc_auc = np.nan

        else:

            pr_auc = np.nan
            roc_auc = np.nan

        rows.append(
            {
                "season": season_names.get(
                    season_id,
                    str(season_id),
                ),
                "observations": len(group),
                "events": events,
                "event_rate": y.mean(),
                "alerts": int(
                    predictions.sum()
                ),
                "alert_rate": predictions.mean(),
                "precision": precision_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SUBDIVISION ROBUSTNESS
# ============================================================

def evaluate_by_subdivision(
    model,
    test,
):

    rows = []

    for subdivision, group in test.groupby(
        "subdivision"
    ):

        X = group[FEATURES]
        y = group[TARGET]

        probabilities = (
            model.predict_proba(X)[:, 1]
        )

        predictions = (
            probabilities
            >= THRESHOLD
        ).astype(int)

        events = int(
            y.sum()
        )

        if events > 0:

            pr_auc = (
                average_precision_score(
                    y,
                    probabilities,
                )
            )

        else:

            pr_auc = np.nan

        if (
            events > 0
            and y.nunique() > 1
        ):

            roc_auc = (
                roc_auc_score(
                    y,
                    probabilities,
                )
            )

        else:

            roc_auc = np.nan

        rows.append(
            {
                "subdivision": subdivision,
                "observations": len(group),
                "events": events,
                "event_rate": y.mean(),
                "alerts": int(
                    predictions.sum()
                ),
                "alert_rate": predictions.mean(),
                "precision": precision_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "event_rate",
            ascending=False,
        )
    )


# ============================================================
# FEATURE STABILITY
# ============================================================

def feature_importance(
    model,
):

    result = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance":
                model.feature_importances_,
        }
    )

    result = (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result["rank"] = (
        np.arange(
            len(result)
        ) + 1
    )

    return result


# ============================================================
# MONTH ROBUSTNESS
# ============================================================

def evaluate_by_month(
    model,
    test,
):

    rows = []

    for month, group in test.groupby(
        "month"
    ):

        X = group[FEATURES]
        y = group[TARGET]

        probabilities = (
            model.predict_proba(X)[:, 1]
        )

        predictions = (
            probabilities
            >= THRESHOLD
        ).astype(int)

        events = int(
            y.sum()
        )

        if events > 0:

            pr_auc = (
                average_precision_score(
                    y,
                    probabilities,
                )
            )

        else:

            pr_auc = np.nan

        rows.append(
            {
                "month": int(month),
                "observations": len(group),
                "events": events,
                "event_rate": y.mean(),
                "alerts": int(
                    predictions.sum()
                ),
                "alert_rate": predictions.mean(),
                "precision": precision_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "pr_auc": pr_auc,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SAVE
# ============================================================

def save_results(
    global_result,
    yearly,
    seasonal,
    subdivision,
    monthly,
    importance,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [global_result]
    ).to_csv(
        OUTPUT_DIR
        / "global_robustness.csv",
        index=False,
    )

    yearly.to_csv(
        OUTPUT_DIR
        / "yearly_robustness.csv",
        index=False,
    )

    seasonal.to_csv(
        OUTPUT_DIR
        / "seasonal_robustness.csv",
        index=False,
    )

    subdivision.to_csv(
        OUTPUT_DIR
        / "subdivision_robustness.csv",
        index=False,
    )

    monthly.to_csv(
        OUTPUT_DIR
        / "monthly_robustness.csv",
        index=False,
    )

    importance.to_csv(
        OUTPUT_DIR
        / "feature_stability.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Excel
    # --------------------------------------------------------

    try:

        with pd.ExcelWriter(
            OUTPUT_DIR
            / "model_robustness_results.xlsx",
            engine="openpyxl",
        ) as writer:

            pd.DataFrame(
                [global_result]
            ).to_excel(
                writer,
                sheet_name="global",
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

            importance.to_excel(
                writer,
                sheet_name="features",
                index=False,
            )

        print(
            "Excel saved."
        )

    except ImportError:

        print()
        print(
            "openpyxl not installed."
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

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        OUTPUT_DIR
        / "global_robustness.csv"
    )

    print(
        OUTPUT_DIR
        / "yearly_robustness.csv"
    )

    print(
        OUTPUT_DIR
        / "seasonal_robustness.csv"
    )

    print(
        OUTPUT_DIR
        / "subdivision_robustness.csv"
    )

    print(
        OUTPUT_DIR
        / "monthly_robustness.csv"
    )

    print(
        OUTPUT_DIR
        / "feature_stability.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

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
    # Global
    # --------------------------------------------------------

    (
        global_result,
        probabilities,
    ) = evaluate_global(
        model,
        test,
    )

    # --------------------------------------------------------
    # Year
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("YEAR-WISE ROBUSTNESS")
    print("=" * 70)

    yearly = evaluate_by_year(
        model,
        test,
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
    print("SEASON-WISE ROBUSTNESS")
    print("=" * 70)

    seasonal = evaluate_by_season(
        model,
        test,
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
    print("SUBDIVISION-WISE ROBUSTNESS")
    print("=" * 70)

    subdivision = (
        evaluate_by_subdivision(
            model,
            test,
        )
    )

    print(
        subdivision.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Month
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MONTH-WISE ROBUSTNESS")
    print("=" * 70)

    monthly = evaluate_by_month(
        model,
        test,
    )

    print(
        monthly.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = feature_importance(
        model
    )

    print()
    print("=" * 70)
    print("FEATURE STABILITY")
    print("=" * 70)

    print(
        importance
        .head(20)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        global_result,
        yearly,
        seasonal,
        subdivision,
        monthly,
        importance,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3.8 MODEL ROBUSTNESS & TEMPORAL VALIDATION COMPLETE")
    print("=" * 70)

    print(
        f"TEST PR-AUC: "
        f"{global_result['pr_auc']:.6f}"
    )

    print(
        f"TEST ROC-AUC: "
        f"{global_result['roc_auc']:.6f}"
    )

    print(
        f"TEST F1 @ {THRESHOLD}: "
        f"{global_result['f1']:.6f}"
    )


if __name__ == "__main__":
    main()