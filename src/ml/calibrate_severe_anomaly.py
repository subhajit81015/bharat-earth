from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
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
    train,
    validation,
    test,
):
    """
    Learn categorical mappings from training data only.
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
            for index, value
            in enumerate(categories)
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


def create_split(df):

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


def evaluate(
    name,
    y_true,
    probabilities,
):

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    brier = brier_score_loss(
        y_true,
        probabilities,
    )

    print(
        f"\n{name}"
    )

    print(
        "PR-AUC:",
        f"{pr_auc:.6f}",
    )

    print(
        "ROC-AUC:",
        f"{roc_auc:.6f}",
    )

    print(
        "BRIER SCORE:",
        f"{brier:.6f}",
    )

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
    }


def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "DATASET"
    )

    print(
        "TOTAL ROWS:",
        len(df),
    )

    print(
        "TARGET RATE:",
        df[TARGET].mean(),
    )

    # --------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------

    train, validation, test = (
        create_split(df)
    )

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

    # --------------------------------------------------
    # ENCODING
    # --------------------------------------------------

    (
        train,
        validation,
        test,
    ) = encode_categories(
        train,
        validation,
        test,
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # --------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------

    positive_count = y_train.sum()

    negative_count = (
        len(y_train)
        - positive_count
    )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print(
        "\nSCALE POS WEIGHT:",
        scale_pos_weight,
    )

    # --------------------------------------------------
    # TRAIN ORIGINAL XGBOOST
    # --------------------------------------------------

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
        early_stopping_rounds=75,
    )

    print(
        "\nTRAINING XGBOOST..."
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

    print(
        "BEST ITERATION:",
        model.best_iteration,
    )

    # --------------------------------------------------
    # RAW PROBABILITIES
    # --------------------------------------------------

    validation_raw = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    test_raw = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------
    # RAW MODEL EVALUATION
    # --------------------------------------------------

    evaluate(
        "RAW XGBOOST - VALIDATION",
        y_validation,
        validation_raw,
    )

    evaluate(
        "RAW XGBOOST - TEST",
        y_test,
        test_raw,
    )

    # --------------------------------------------------
    # ISOTONIC CALIBRATION
    # --------------------------------------------------

    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    isotonic.fit(
        validation_raw,
        y_validation,
    )

    validation_isotonic = (
        isotonic.predict(
            validation_raw
        )
    )

    test_isotonic = (
        isotonic.predict(
            test_raw
        )
    )

    # --------------------------------------------------
    # ISOTONIC RESULTS
    # --------------------------------------------------

    evaluate(
        "ISOTONIC - VALIDATION",
        y_validation,
        validation_isotonic,
    )

    evaluate(
        "ISOTONIC - TEST",
        y_test,
        test_isotonic,
    )

    # --------------------------------------------------
    # PLATT / SIGMOID CALIBRATION
    # --------------------------------------------------

    sigmoid = LogisticRegression(
        max_iter=2000,
    )

    sigmoid.fit(
        validation_raw.reshape(-1, 1),
        y_validation,
    )

    validation_sigmoid = (
        sigmoid.predict_proba(
            validation_raw.reshape(-1, 1)
        )[:, 1]
    )

    test_sigmoid = (
        sigmoid.predict_proba(
            test_raw.reshape(-1, 1)
        )[:, 1]
    )

    # --------------------------------------------------
    # SIGMOID RESULTS
    # --------------------------------------------------

    evaluate(
        "SIGMOID - VALIDATION",
        y_validation,
        validation_sigmoid,
    )

    evaluate(
        "SIGMOID - TEST",
        y_test,
        test_sigmoid,
    )

    # --------------------------------------------------
    # PROBABILITY DISTRIBUTION
    # --------------------------------------------------

    print(
        "\nPROBABILITY SUMMARY"
    )

    summary = pd.DataFrame(
        {
            "raw": test_raw,
            "isotonic": test_isotonic,
            "sigmoid": test_sigmoid,
        }
    )

    print(
        summary.describe().to_string()
    )

    # --------------------------------------------------
    # CALIBRATION BINS
    # --------------------------------------------------

    print(
        "\nCALIBRATION BINS"
    )

    calibration_df = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "raw_probability": test_raw,
            "isotonic_probability":
                test_isotonic,
            "sigmoid_probability":
                test_sigmoid,
        }
    )

    calibration_df[
        "probability_bin"
    ] = pd.qcut(
        calibration_df[
            "raw_probability"
        ],
        q=10,
        duplicates="drop",
    )

    calibration_summary = (
        calibration_df
        .groupby(
            "probability_bin",
            observed=True,
        )
        .agg(
            observations=(
                "actual",
                "size",
            ),
            actual_rate=(
                "actual",
                "mean",
            ),
            raw_mean=(
                "raw_probability",
                "mean",
            ),
            isotonic_mean=(
                "isotonic_probability",
                "mean",
            ),
            sigmoid_mean=(
                "sigmoid_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        calibration_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------
    # SAVE CALIBRATED TEST OUTPUT
    # --------------------------------------------------

    output_file = (
        PROJECT_ROOT
        / "data"
        / "features"
        / "calibrated_predictions.csv"
    )

    output = pd.DataFrame(
        {
            "subdivision":
                test["subdivision"],
            "year":
                test["year"],
            "month":
                test["month"],
            "actual":
                y_test,
            "raw_probability":
                test_raw,
            "isotonic_probability":
                test_isotonic,
            "sigmoid_probability":
                test_sigmoid,
        }
    )

    output.to_csv(
        output_file,
        index=False,
    )

    print(
        "\nCALIBRATED PREDICTIONS SAVED:"
    )

    print(
        output_file
    )


if __name__ == "__main__":
    main()