from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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


def train_baseline() -> None:
    """Train and evaluate a time-based logistic regression baseline."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"ML dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values(
        ["year", "month"]
    ).reset_index(drop=True)

    # Time-based split.
    split_year = df["year"].quantile(0.80)

    train = df[
        df["year"] <= split_year
    ].copy()

    test = df[
        df["year"] > split_year
    ].copy()

    X_train = train[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]

    y_train = train[TARGET]

    X_test = test[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ]

    y_test = test[TARGET]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    print("TRAIN ROWS:", len(train))
    print("TEST ROWS:", len(test))

    print(
        "TRAIN POSITIVE RATE:",
        y_train.mean(),
    )

    print(
        "TEST POSITIVE RATE:",
        y_test.mean(),
    )

    print(
        "\nPRECISION:",
        precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    )

    print(
        "RECALL:",
        recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    )

    print(
        "F1:",
        f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    )

    print(
        "PR-AUC:",
        average_precision_score(
            y_test,
            probabilities,
        ),
    )

    print(
        "ROC-AUC:",
        roc_auc_score(
            y_test,
            probabilities,
        ),
    )

    print("\nCONFUSION MATRIX:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print("\nCLASSIFICATION REPORT:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    train_baseline()