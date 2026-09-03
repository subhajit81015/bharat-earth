from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v4.csv"
)

TARGET = "target_3m_severe_anomaly"


FEATURES = [
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
    "month",
    "season",
    "subdivision",
]


CATEGORICAL_FEATURES = [
    "month",
    "season",
    "subdivision",
]


MONTH_ORDER = {
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


def encode_categories(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
):
    """
    Encode categorical columns using mappings learned
    from the training dataset only.
    """

    train = train.copy()
    validation = validation.copy()
    test = test.copy()

    for column in CATEGORICAL_FEATURES:

        categories = sorted(
            train[column]
            .dropna()
            .astype(str)
            .unique()
        )

        mapping = {
            value: index
            for index, value in enumerate(categories)
        }

        train[column] = (
            train[column]
            .astype(str)
            .map(mapping)
        )

        validation[column] = (
            validation[column]
            .astype(str)
            .map(mapping)
        )

        test[column] = (
            test[column]
            .astype(str)
            .map(mapping)
        )

    for column in FEATURES:

        train[column] = pd.to_numeric(
            train[column],
            errors="coerce",
        )

        validation[column] = pd.to_numeric(
            validation[column],
            errors="coerce",
        )

        test[column] = pd.to_numeric(
            test[column],
            errors="coerce",
        )

    return train, validation, test


def create_temporal_split(
    df: pd.DataFrame,
):
    """
    Create chronological 70/15/15 train-validation-test split.
    """

    df = df.copy()

    df["month_number"] = (
        df["month"].map(MONTH_ORDER)
    )

    df = df.sort_values(
        ["year", "month_number"]
    ).reset_index(drop=True)

    years = sorted(
        df["year"].unique()
    )

    train_end = int(
        len(years) * 0.70
    )

    validation_end = int(
        len(years) * 0.85
    )

    train_years = years[:train_end]

    validation_years = years[
        train_end:validation_end
    ]

    test_years = years[
        validation_end:
    ]

    train = df[
        df["year"].isin(train_years)
    ].copy()

    validation = df[
        df["year"].isin(validation_years)
    ].copy()

    test = df[
        df["year"].isin(test_years)
    ].copy()

    return train, validation, test


def evaluate_thresholds(
    y_true,
    probabilities,
):
    """
    Evaluate classification thresholds.
    """

    thresholds = [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

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

        alert_rate = predictions.mean()

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "alert_rate": alert_rate,
            }
        )

    return pd.DataFrame(results)


def main():

    # ---------------------------------------------------------
    # 1. CHECK INPUT
    # ---------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    # ---------------------------------------------------------
    # 2. LOAD DATA
    # ---------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    print("DATASET")
    print(
        "TOTAL ROWS:",
        len(df),
    )

    print(
        "TARGET:",
        TARGET,
    )

    print(
        "TARGET RATE:",
        df[TARGET].mean(),
    )

    # ---------------------------------------------------------
    # 3. VALIDATE REQUIRED COLUMNS
    # ---------------------------------------------------------

    required_columns = (
        set(FEATURES)
        | {TARGET}
    )

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ---------------------------------------------------------
    # 4. CREATE TEMPORAL SPLIT
    # ---------------------------------------------------------

    (
        train,
        validation,
        test,
    ) = create_temporal_split(df)

    print(
        "\nDATA SPLIT"
    )

    print(
        "TRAIN ROWS:",
        len(train),
    )

    print(
        "VALIDATION ROWS:",
        len(validation),
    )

    print(
        "TEST ROWS:",
        len(test),
    )

    # ---------------------------------------------------------
    # 5. TARGET RATES
    # ---------------------------------------------------------

    print(
        "\nTARGET RATES"
    )

    print(
        "TRAIN:",
        train[TARGET].mean(),
    )

    print(
        "VALIDATION:",
        validation[TARGET].mean(),
    )

    print(
        "TEST:",
        test[TARGET].mean(),
    )

    # ---------------------------------------------------------
    # 6. ENCODE CATEGORICAL FEATURES
    # ---------------------------------------------------------

    (
        train_encoded,
        validation_encoded,
        test_encoded,
    ) = encode_categories(
        train,
        validation,
        test,
    )

    X_train = train_encoded[
        FEATURES
    ]

    y_train = train_encoded[
        TARGET
    ]

    X_validation = validation_encoded[
        FEATURES
    ]

    y_validation = validation_encoded[
        TARGET
    ]

    X_test = test_encoded[
        FEATURES
    ]

    y_test = test_encoded[
        TARGET
    ]

    # ---------------------------------------------------------
    # 7. CLASS IMBALANCE
    # ---------------------------------------------------------

    positive_count = y_train.sum()

    negative_count = (
        len(y_train)
        - positive_count
    )

    if positive_count == 0:

        raise ValueError(
            "Training dataset contains "
            "no positive target examples."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print(
        "\nCLASS IMBALANCE"
    )

    print(
        "POSITIVE COUNT:",
        int(positive_count),
    )

    print(
        "NEGATIVE COUNT:",
        int(negative_count),
    )

    print(
        "SCALE POS WEIGHT:",
        scale_pos_weight,
    )

    # ---------------------------------------------------------
    # 8. XGBOOST MODEL
    # ---------------------------------------------------------

    model = XGBClassifier(
        n_estimators=2000,
        max_depth=5,
        learning_rate=0.03,
        min_child_weight=5,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",

        # IMPORTANT:
        # XGBoost 3.4.1 supports this parameter
        # directly in XGBClassifier.
        early_stopping_rounds=75,
    )

    print(
        "\nTRAINING XGBOOST..."
    )

    # ---------------------------------------------------------
    # 9. TRAIN WITH VALIDATION SET
    # ---------------------------------------------------------

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    # ---------------------------------------------------------
    # 10. BEST ITERATION
    # ---------------------------------------------------------

    print(
        "\nBOOSTING INFORMATION"
    )

    print(
        "BEST ITERATION:",
        model.best_iteration,
    )

    print(
        "BEST SCORE:",
        model.best_score,
    )

    # ---------------------------------------------------------
    # 11. PREDICTIONS
    # ---------------------------------------------------------

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # ---------------------------------------------------------
    # 12. VALIDATION METRICS
    # ---------------------------------------------------------

    validation_pr_auc = (
        average_precision_score(
            y_validation,
            validation_probabilities,
        )
    )

    validation_roc_auc = (
        roc_auc_score(
            y_validation,
            validation_probabilities,
        )
    )

    print(
        "\nVALIDATION"
    )

    print(
        "PR-AUC:",
        validation_pr_auc,
    )

    print(
        "ROC-AUC:",
        validation_roc_auc,
    )

    # ---------------------------------------------------------
    # 13. THRESHOLD ANALYSIS
    # ---------------------------------------------------------

    threshold_df = evaluate_thresholds(
        y_validation,
        validation_probabilities,
    )

    print(
        "\nTHRESHOLD ANALYSIS"
    )

    print(
        threshold_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 14. SELECT BEST VALIDATION THRESHOLD
    # ---------------------------------------------------------

    best_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    print(
        "\nBEST VALIDATION THRESHOLD:",
        best_threshold,
    )

    print(
        "VALIDATION PRECISION:",
        best_row["precision"],
    )

    print(
        "VALIDATION RECALL:",
        best_row["recall"],
    )

    print(
        "VALIDATION F1:",
        best_row["f1"],
    )

    print(
        "VALIDATION ALERT RATE:",
        best_row["alert_rate"],
    )

    # ---------------------------------------------------------
    # 15. FINAL TEST
    # ---------------------------------------------------------

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    test_pr_auc = (
        average_precision_score(
            y_test,
            test_probabilities,
        )
    )

    test_roc_auc = (
        roc_auc_score(
            y_test,
            test_probabilities,
        )
    )

    test_precision = (
        precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    test_recall = (
        recall_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    test_f1 = (
        f1_score(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    test_alert_rate = (
        test_predictions.mean()
    )

    # ---------------------------------------------------------
    # 16. FINAL RESULTS
    # ---------------------------------------------------------

    print(
        "\nFINAL TEST RESULTS"
    )

    print(
        "PR-AUC:",
        test_pr_auc,
    )

    print(
        "ROC-AUC:",
        test_roc_auc,
    )

    print(
        "PRECISION:",
        test_precision,
    )

    print(
        "RECALL:",
        test_recall,
    )

    print(
        "F1:",
        test_f1,
    )

    print(
        "ALERT RATE:",
        test_alert_rate,
    )

    # ---------------------------------------------------------
    # 17. CONFUSION MATRIX
    # ---------------------------------------------------------

    print(
        "\nCONFUSION MATRIX"
    )

    print(
        confusion_matrix(
            y_test,
            test_predictions,
        )
    )

    # ---------------------------------------------------------
    # 18. FEATURE IMPORTANCE
    # ---------------------------------------------------------

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(
        ascending=False
    )

    print(
        "\nTOP 20 FEATURE IMPORTANCE"
    )

    print(
        importance.head(20).to_string()
    )

    # ---------------------------------------------------------
    # 19. FINAL BENCHMARK
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "MODEL BENCHMARK"
    )

    print(
        "=" * 70
    )

    print(
        "CURRENT CHAMPION:"
    )

    print(
        "XGBoost V4 - 500 trees"
    )

    print(
        "PR-AUC:  0.148238"
    )

    print(
        "ROC-AUC: 0.796542"
    )

    print(
        "F1:      0.227397"
    )

    print(
        "\nEARLY-STOPPING MODEL:"
    )

    print(
        f"PR-AUC:  {test_pr_auc:.6f}"
    )

    print(
        f"ROC-AUC: {test_roc_auc:.6f}"
    )

    print(
        f"F1:      {test_f1:.6f}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()