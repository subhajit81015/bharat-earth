from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset.csv"
)

TARGET = "target_3m_stress"


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
    "year",
]


CATEGORICAL_FEATURES = [
    "subdivision",
    "month",
    "season",
]


def train_xgboost() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"ML dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values(
        ["year", "month"]
    ).reset_index(drop=True)

    # Convert categorical variables to category codes.
    for column in CATEGORICAL_FEATURES:
        df[column] = (
            df[column]
            .astype("category")
            .cat.codes
        )

    features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    # Time-based split.
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

    X_train = train[features]
    y_train = train[TARGET]

    X_validation = validation[features]
    y_validation = validation[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    # Handle severe class imbalance.
    negative_count = (
        y_train == 0
    ).sum()

    positive_count = (
        y_train == 1
    ).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        "SCALE POS WEIGHT:",
        scale_pos_weight,
    )

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
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

    print("\nVALIDATION")
    print(
        "POSITIVE RATE:",
        y_validation.mean(),
    )

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

    print("\nTEST")
    print(
        "POSITIVE RATE:",
        y_test.mean(),
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

    print("\nTHRESHOLD ANALYSIS")

    results = []

    for threshold in [
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
    ]:

        predictions = (
            validation_probabilities
            >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
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

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    best_row = results_df.loc[
        results_df["f1"].idxmax()
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

    # Evaluate the untouched test period
    # using the validation-selected threshold.
    test_predictions = (
        test_probabilities
        >= best_threshold
    ).astype(int)

    print(
        "\nFINAL TEST RESULTS"
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

    print(
        "\nCONFUSION MATRIX:"
    )

    print(
        confusion_matrix(
            y_test,
            test_predictions,
        )
    )

    print(
        "\nCLASSIFICATION REPORT:"
    )

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    train_xgboost()