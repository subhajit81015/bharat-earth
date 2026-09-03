from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
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


RAINFALL_FEATURES = [
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
]


SEASONAL_FEATURES = [
    "month",
    "season",
    "month_sin",
    "month_cos",
]


REGION_FEATURE = [
    "subdivision",
]


CATEGORICAL_FEATURES = [
    "month",
    "season",
    "subdivision",
]


def prepare_split(
    df: pd.DataFrame,
):
    """Create identical chronological train/validation/test splits."""

    df = df.copy()

    month_order = {
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

    df["month_number"] = df[
        "month"
    ].map(month_order)

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


def encode_categories(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
):
    """
    Encode categorical columns using mappings learned
    only from the training dataset.
    """

    train_result = train[features].copy()
    validation_result = validation[features].copy()
    test_result = test[features].copy()

    for column in CATEGORICAL_FEATURES:

        if column not in features:
            continue

        categories = sorted(
            train_result[column]
            .dropna()
            .astype(str)
            .unique()
        )

        mapping = {
            value: index
            for index, value
            in enumerate(categories)
        }

        train_result[column] = (
            train_result[column]
            .astype(str)
            .map(mapping)
        )

        validation_result[column] = (
            validation_result[column]
            .astype(str)
            .map(mapping)
        )

        test_result[column] = (
            test_result[column]
            .astype(str)
            .map(mapping)
        )

    # Explicitly convert every feature to numeric.
    for column in features:

        train_result[column] = pd.to_numeric(
            train_result[column],
            errors="coerce",
        )

        validation_result[column] = pd.to_numeric(
            validation_result[column],
            errors="coerce",
        )

        test_result[column] = pd.to_numeric(
            test_result[column],
            errors="coerce",
        )

    return (
        train_result,
        validation_result,
        test_result,
    )


def train_and_evaluate(
    model_name: str,
    features: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
):
    """Train and evaluate one feature configuration."""

    (
        X_train,
        X_validation,
        X_test,
    ) = encode_categories(
        train,
        validation,
        test,
        features,
    )

    y_train = train[TARGET]

    y_validation = validation[TARGET]

    y_test = test[TARGET]

    positive_count = y_train.sum()

    negative_count = (
        len(y_train)
        - positive_count
    )

    scale_pos_weight = (
        negative_count
        / positive_count
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

    threshold_results = []

    for threshold in np.arange(
        0.10,
        0.91,
        0.05,
    ):

        validation_predictions = (
            validation_probabilities
            >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            validation_predictions,
            zero_division=0,
        )

        threshold_results.append(
            {
                "threshold": round(
                    float(threshold),
                    2,
                ),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    best_row = threshold_df.loc[
        threshold_df["f1"].idxmax()
    ]

    best_threshold = float(
        best_row["threshold"]
    )

    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    return {
        "model": model_name,
        "features": len(features),
        "validation_pr_auc":
            average_precision_score(
                y_validation,
                validation_probabilities,
            ),
        "validation_roc_auc":
            roc_auc_score(
                y_validation,
                validation_probabilities,
            ),
        "best_threshold":
            best_threshold,
        "validation_f1":
            best_row["f1"],
        "test_pr_auc":
            average_precision_score(
                y_test,
                test_probabilities,
            ),
        "test_roc_auc":
            roc_auc_score(
                y_test,
                test_probabilities,
            ),
        "test_precision":
            precision_score(
                y_test,
                test_predictions,
                zero_division=0,
            ),
        "test_recall":
            recall_score(
                y_test,
                test_predictions,
                zero_division=0,
            ),
        "test_f1":
            f1_score(
                y_test,
                test_predictions,
                zero_division=0,
            ),
    }


def main():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    train, validation, test = (
        prepare_split(df)
    )

    print("DATASET")

    print(
        "TOTAL ROWS:",
        len(df),
    )

    print(
        "TRAIN:",
        len(train),
    )

    print(
        "VALIDATION:",
        len(validation),
    )

    print(
        "TEST:",
        len(test),
    )

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

    experiments = {
        "A_FULL_MODEL": (
            RAINFALL_FEATURES
            + SEASONAL_FEATURES
            + REGION_FEATURE
        ),

        "B_NO_SEASONALITY": (
            RAINFALL_FEATURES
            + REGION_FEATURE
        ),

        "C_RAINFALL_ONLY": (
            RAINFALL_FEATURES
        ),
    }

    results = []

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FEATURE ABLATION EXPERIMENT"
    )

    print(
        "=" * 70
    )

    for model_name, features in (
        experiments.items()
    ):

        print(
            f"\nRUNNING: {model_name}"
        )

        print(
            "FEATURE COUNT:",
            len(features),
        )

        result = train_and_evaluate(
            model_name,
            features,
            train,
            validation,
            test,
        )

        results.append(result)

        print(
            "VALIDATION PR-AUC:",
            f"{result['validation_pr_auc']:.6f}",
        )

        print(
            "VALIDATION ROC-AUC:",
            f"{result['validation_roc_auc']:.6f}",
        )

        print(
            "BEST THRESHOLD:",
            result["best_threshold"],
        )

        print(
            "TEST PR-AUC:",
            f"{result['test_pr_auc']:.6f}",
        )

        print(
            "TEST ROC-AUC:",
            f"{result['test_roc_auc']:.6f}",
        )

        print(
            "TEST PRECISION:",
            f"{result['test_precision']:.6f}",
        )

        print(
            "TEST RECALL:",
            f"{result['test_recall']:.6f}",
        )

        print(
            "TEST F1:",
            f"{result['test_f1']:.6f}",
        )

    results_df = pd.DataFrame(
        results
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL ABLATION RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )


if __name__ == "__main__":
    main()