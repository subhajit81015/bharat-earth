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


NUMERIC_FEATURES = [
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


CATEGORICAL_FEATURES = [
    "subdivision",
    "month",
    "season",
]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for XGBoost."""

    result = df.copy()

    for column in CATEGORICAL_FEATURES:
        result[column] = (
            result[column]
            .astype("category")
            .cat.codes
            .replace(-1, np.nan)
        )

    return result


def evaluate_thresholds(
    y_true,
    probabilities,
):
    """Evaluate classification thresholds."""

    results = []

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

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(results)


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values(
        ["year", "month"]
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

    features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    train = prepare_data(train)
    validation = prepare_data(validation)
    test = prepare_data(test)

    X_train = train[features]
    y_train = train[TARGET]

    X_validation = validation[features]
    y_validation = validation[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    positive_count = y_train.sum()
    negative_count = len(y_train) - positive_count

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print(
        "SCALE POS WEIGHT:",
        scale_pos_weight,
    )

    print("\nDATA SPLIT")

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

    print("\nTARGET RATES")

    print(
        "TRAIN:",
        y_train.mean(),
    )

    print(
        "VALIDATION:",
        y_validation.mean(),
    )

    print(
        "TEST:",
        y_test.mean(),
    )

    model = XGBClassifier(
        n_estimators=500,
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
    )

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

    print("\nVALIDATION")

    print(
        "PR-AUC:",
        average_precision_score(
            y_validation,
            validation_probabilities,
        ),
    )

    print(
        "ROC-AUC:",
        roc_auc_score(
            y_validation,
            validation_probabilities,
        ),
    )

    print("\nTHRESHOLD ANALYSIS")

    results = evaluate_thresholds(
        y_validation,
        validation_probabilities,
    )

    print(
        results.to_string(
            index=False
        )
    )

    best_row = results.loc[
        results["f1"].idxmax()
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

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    print(
        "\nFINAL TEST RESULTS"
    )

    print(
        "PR-AUC:",
        average_precision_score(
            y_test,
            test_probabilities,
        ),
    )

    print(
        "ROC-AUC:",
        roc_auc_score(
            y_test,
            test_probabilities,
        ),
    )

    print(
        "PRECISION:",
        precision_score(
            y_test,
            test_predictions,
            zero_division=0,
        ),
    )

    print(
        "RECALL:",
        recall_score(
            y_test,
            test_predictions,
            zero_division=0,
        ),
    )

    print(
        "F1:",
        f1_score(
            y_test,
            test_predictions,
            zero_division=0,
        ),
    )

    print("\nCONFUSION MATRIX")

    print(
        confusion_matrix(
            y_test,
            test_predictions,
        )
    )

    print("\nFEATURE IMPORTANCE")

    importance = pd.Series(
        model.feature_importances_,
        index=features,
    ).sort_values(
        ascending=False
    )

    print(
        importance.head(20).to_string()
    )


if __name__ == "__main__":
    main()