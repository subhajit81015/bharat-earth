from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
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
    / "ml_dataset_v3.csv"
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
]


CATEGORICAL_FEATURES = [
    "subdivision",
    "month",
    "season",
]


def run_analysis() -> None:

    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values(
        ["year", "month"]
    ).reset_index(drop=True)

    years = sorted(df["year"].unique())

    train_end = int(len(years) * 0.70)
    validation_end = int(len(years) * 0.85)

    train_years = years[:train_end]
    validation_years = years[
        train_end:validation_end
    ]
    test_years = years[
        validation_end:
    ]

    train = df[
        df["year"].isin(train_years)
    ]

    validation = df[
        df["year"].isin(validation_years)
    ]

    test = df[
        df["year"].isin(test_years)
    ]

    features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    X_train = train[features]
    y_train = train[TARGET]

    X_validation = validation[features]
    y_validation = validation[TARGET]

    X_test = test[features]
    y_test = test[TARGET]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
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
                SimpleImputer(
                    strategy="most_frequent"
                ),
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
        max_iter=3000,
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

    validation_probabilities = (
        pipeline.predict_proba(
            X_validation
        )[:, 1]
    )

    test_probabilities = (
        pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    print("VALIDATION PR-AUC:")
    print(
        average_precision_score(
            y_validation,
            validation_probabilities,
        )
    )

    print(
        "\nVALIDATION ROC-AUC:"
    )

    print(
        roc_auc_score(
            y_validation,
            validation_probabilities,
        )
    )

    print(
        "\nTHRESHOLD POLICY"
    )

    rows = []

    for threshold in [
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

        predicted_positive_rate = (
            predictions.mean()
        )

        false_positive_count = (
            (
                (predictions == 1)
                & (y_validation == 0)
            )
            .sum()
        )

        true_positive_count = (
            (
                (predictions == 1)
                & (y_validation == 1)
            )
            .sum()
        )

        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "predicted_alert_rate":
                    predicted_positive_rate,
                "true_positives":
                    true_positive_count,
                "false_positives":
                    false_positive_count,
            }
        )

    results = pd.DataFrame(rows)

    print(
        results.to_string(
            index=False
        )
    )

    print(
        "\nFINAL TEST AT SELECTED THRESHOLDS"
    )

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]:

        predictions = (
            test_probabilities
            >= threshold
        ).astype(int)

        print(
            f"\nTHRESHOLD: {threshold}"
        )

        print(
            "PRECISION:",
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        print(
            "RECALL:",
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        print(
            "ALERT RATE:",
            predictions.mean(),
        )


if __name__ == "__main__":
    run_analysis()