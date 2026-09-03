from pathlib import Path
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)


# ==============================================================
# PATHS
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v4.csv"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "model_v4"
    / "xgboost_model.json"
)

SCHEMA_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "model_v4"
    / "model_schema.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "calibration_v4"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CALIBRATED_FILE = (
    OUTPUT_DIR
    / "calibrated_predictions.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "calibration_metrics.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "calibration_summary.json"
)


# ==============================================================
# CONSTANTS
# ==============================================================

TARGET = "target_3m_severe_anomaly"

CATEGORICAL_FEATURES = [
    "subdivision",
    "month",
    "season",
]

FEATURES = [
    "subdivision",
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

NUMERIC_FEATURES = [
    x
    for x in FEATURES
    if x not in CATEGORICAL_FEATURES
]


# ==============================================================
# PRINT HELPERS
# ==============================================================

def header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ==============================================================
# LOAD SCHEMA
# ==============================================================

def load_schema():

    header("LOADING SAVED MODEL SCHEMA")

    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"Schema not found:\n{SCHEMA_FILE}"
        )

    with open(
        SCHEMA_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        schema = json.load(f)

    print(
        "SCHEMA:",
        SCHEMA_FILE,
    )

    print("SCHEMA LOAD: PASS")

    return schema


# ==============================================================
# LOAD DATA
# ==============================================================

def load_dataset():

    header("LOADING CLEAN V4 DATASET")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    print("INPUT:")
    print(DATA_FILE)

    print("ROWS:", len(df))
    print("COLUMNS:", len(df.columns))

    return df


# ==============================================================
# VALIDATE DATASET
# ==============================================================

def validate_dataset(df):

    header("DATASET VALIDATION")

    leakage_columns = [
        column
        for column in df.columns
        if (
            column.startswith("target_")
            and column != TARGET
        )
        or column in {
            "rainfall_stress",
            "persistent_drought_signal",
            "environmental_risk_score",
            "environmental_risk_level",
        }
    ]

    print(
        "LEAKAGE COLUMNS:"
    )
    print(leakage_columns)

    if leakage_columns:
        raise ValueError(
            "LEAKAGE COLUMNS FOUND:\n"
            f"{leakage_columns}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target missing: {TARGET}"
        )

    values = sorted(
        df[TARGET]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        "TARGET VALUES:",
        values,
    )

    if values != [0, 1]:
        raise ValueError(
            "Target must contain exactly 0 and 1."
        )

    print(
        "TARGET DISTRIBUTION:"
    )
    print(
        df[TARGET].value_counts()
    )

    print(
        "TARGET RATE:",
        df[TARGET].mean()
    )

    duplicates = int(
        df.duplicated().sum()
    )

    print(
        "EXACT DUPLICATES:",
        duplicates,
    )

    if duplicates:
        raise ValueError(
            f"Dataset contains {duplicates} duplicates."
        )


# ==============================================================
# NORMALIZE MONTH
# ==============================================================

def normalize_month(series):

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    # Already 1-12
    valid_one_based = (
        numeric.notna()
        & numeric.between(1, 12)
    )

    if valid_one_based.all():
        return numeric.astype(int)

    # Zero-based 0-11
    valid_zero_based = (
        numeric.notna()
        & numeric.between(0, 11)
    )

    if valid_zero_based.all():
        return (
            numeric
            .astype(int)
            .add(1)
        )

    # Text month support
    month_names = {
        "JAN": 1,
        "JANUARY": 1,
        "FEB": 2,
        "FEBRUARY": 2,
        "MAR": 3,
        "MARCH": 3,
        "APR": 4,
        "APRIL": 4,
        "MAY": 5,
        "JUN": 6,
        "JUNE": 6,
        "JUL": 7,
        "JULY": 7,
        "AUG": 8,
        "AUGUST": 8,
        "SEP": 9,
        "SEPT": 9,
        "SEPTEMBER": 9,
        "OCT": 10,
        "OCTOBER": 10,
        "NOV": 11,
        "NOVEMBER": 11,
        "DEC": 12,
        "DECEMBER": 12,
    }

    text = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )

    mapped = text.map(month_names)

    result = numeric.copy()

    result[mapped.notna()] = mapped[
        mapped.notna()
    ]

    result = pd.to_numeric(
        result,
        errors="coerce",
    )

    if result.isna().any():
        bad = (
            series[result.isna()]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Unable to normalize month values:\n"
            f"{bad}"
        )

    result = result.astype(int)

    if not result.between(1, 12).all():
        raise ValueError(
            "Invalid month values after normalization."
        )

    return result


# ==============================================================
# DERIVE SEASON
# ==============================================================

def season_from_month(month):

    month = int(month)

    if month in {6, 7, 8, 9}:
        return "MONSOON"

    if month in {10, 11}:
        return "POST_MONSOON"

    if month in {3, 4, 5}:
        return "PRE_MONSOON"

    if month in {12, 1, 2}:
        return "WINTER"

    raise ValueError(
        f"Invalid month: {month}"
    )


# ==============================================================
# PREPARE EXACT MODEL MATRIX
# ==============================================================

def build_model_matrix(df, schema):

    header("PREPARING EXACT MODEL SCHEMA")

    X = df[FEATURES].copy()

    # ----------------------------------------------------------
    # IMPORTANT:
    # The schema JSON may not contain categorical_categories.
    # Therefore categories are recovered safely.
    # ----------------------------------------------------------

    schema_categories = schema.get(
        "categorical_categories",
        {}
    )

    # ----------------------------------------------------------
    # SUBDIVISION
    # ----------------------------------------------------------

    if (
        isinstance(schema_categories, dict)
        and "subdivision"
        in schema_categories
    ):

        subdivision_categories = [
            str(x)
            for x in schema_categories[
                "subdivision"
            ]
        ]

    else:

        subdivision_categories = sorted(
            df["subdivision"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    # ----------------------------------------------------------
    # MONTH
    # ----------------------------------------------------------

    if (
        isinstance(schema_categories, dict)
        and "month"
        in schema_categories
    ):

        month_categories = [
            str(x)
            for x in schema_categories[
                "month"
            ]
        ]

    else:

        month_categories = [
            str(x)
            for x in range(1, 13)
        ]

    # ----------------------------------------------------------
    # SEASON
    # ----------------------------------------------------------

    if (
        isinstance(schema_categories, dict)
        and "season"
        in schema_categories
    ):

        season_categories = [
            str(x)
            for x in schema_categories[
                "season"
            ]
        ]

    else:

        season_categories = [
            "MONSOON",
            "POST_MONSOON",
            "PRE_MONSOON",
            "WINTER",
        ]

    # ----------------------------------------------------------
    # NORMALIZE MONTH
    # ----------------------------------------------------------

    X["month"] = normalize_month(
        X["month"]
    )

    # ----------------------------------------------------------
    # REBUILD SEASON FROM MONTH
    # ----------------------------------------------------------

    X["season"] = X[
        "month"
    ].map(
        season_from_month
    )

    # ----------------------------------------------------------
    # SUBDIVISION CATEGORY
    # ----------------------------------------------------------

    X["subdivision"] = pd.Categorical(
        X["subdivision"].astype(str),
        categories=subdivision_categories,
    )

    # ----------------------------------------------------------
    # MONTH CATEGORY
    #
    # CRITICAL:
    # XGBoost was trained with STRING categories:
    # ['1', '2', ..., '12']
    # ----------------------------------------------------------

    X["month"] = pd.Categorical(
        X["month"].astype(str),
        categories=month_categories,
    )

    # ----------------------------------------------------------
    # SEASON CATEGORY
    # ----------------------------------------------------------

    X["season"] = pd.Categorical(
        X["season"].astype(str),
        categories=season_categories,
    )

    # ----------------------------------------------------------
    # NUMERIC FEATURES
    # ----------------------------------------------------------

    for column in NUMERIC_FEATURES:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # ----------------------------------------------------------
    # FILL NUMERIC MISSING VALUES
    # ----------------------------------------------------------

    for column in NUMERIC_FEATURES:

        if X[column].isna().any():

            median = X[column].median()

            if pd.isna(median):
                median = 0.0

            X[column] = (
                X[column]
                .fillna(median)
            )

    # ----------------------------------------------------------
    # CHECK CATEGORICAL VALUES
    # ----------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        if X[column].isna().any():

            bad_count = int(
                X[column].isna().sum()
            )

            raise ValueError(
                f"{column} has "
                f"{bad_count} values outside "
                "the model category set."
            )

    # ----------------------------------------------------------
    # FINAL COLUMN ORDER
    # ----------------------------------------------------------

    X = X[FEATURES].copy()

    # ----------------------------------------------------------
    # PRINT
    # ----------------------------------------------------------

    print(
        "\nFINAL CATEGORICAL DTYPES:"
    )

    for column in CATEGORICAL_FEATURES:

        print(
            f"{column}:",
            X[column].dtype,
        )

        print(
            "categories:",
            X[column]
            .cat
            .categories
            .tolist(),
        )

    print(
        "\nBUILDING MODEL MATRIX"
    )

    print(
        "MODEL MATRIX SHAPE:",
        X.shape,
    )

    print(
        "MODEL FEATURE COUNT:",
        len(X.columns),
    )

    print(
        "\nREMAINING NULL VALUES:",
        int(X.isna().sum().sum()),
    )

    return X


# ==============================================================
# LOAD MODEL
# ==============================================================

def load_model():

    header("LOADING CLEAN XGBOOST MODEL")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    model = xgb.XGBClassifier()

    model.load_model(
        str(MODEL_FILE)
    )

    print(
        "MODEL:",
        MODEL_FILE,
    )

    print(
        "MODEL LOADED: PASS"
    )

    return model


# ==============================================================
# MODEL SCHEMA VALIDATION
# ==============================================================

def validate_model_schema(
    model,
    X,
):

    header(
        "VALIDATING MODEL SCHEMA"
    )

    booster = model.get_booster()

    model_features = [
        str(x)
        for x in booster.feature_names
    ]

    data_features = [
        str(x)
        for x in X.columns
    ]

    print(
        "MODEL FEATURES:",
        len(model_features),
    )

    if model_features != data_features:

        print(
            "MODEL:",
            model_features,
        )

        print(
            "DATA:",
            data_features,
        )

        raise ValueError(
            "MODEL FEATURE ORDER MISMATCH"
        )

    print(
        "MODEL FEATURE ORDER: PASS"
    )

    # ----------------------------------------------------------
    # Validate categorical columns
    # ----------------------------------------------------------

    for column in CATEGORICAL_FEATURES:

        if not isinstance(
            X[column].dtype,
            pd.CategoricalDtype,
        ):

            raise ValueError(
                f"{column} is not categorical."
            )

    print(
        "MODEL CATEGORICAL TYPES: PASS"
    )


# ==============================================================
# SAFE MODEL PREDICTION
# ==============================================================

def get_probability(
    model,
    X,
):

    try:

        prediction = model.predict_proba(
            X
        )[:, 1]

        prediction = np.asarray(
            prediction,
            dtype=float,
        )

        if not np.isfinite(
            prediction
        ).all():

            raise ValueError(
                "Model returned non-finite probabilities."
            )

        if (
            prediction.min() < 0
            or prediction.max() > 1
        ):

            raise ValueError(
                "Model probabilities outside [0,1]."
            )

        return prediction

    except Exception as exc:

        print()
        print(
            "RAW PROBABILITY GENERATION FAILED."
        )

        print(exc)

        raise RuntimeError(
            "XGBoost model input schema is "
            "still incompatible."
        ) from exc


# ==============================================================
# TEMPORAL SPLIT
# ==============================================================

def temporal_split(df):

    header(
        "TEMPORAL CALIBRATION SPLIT"
    )

    train = df[
        df["year"] <= 2013
    ].copy()

    calibration = df[
        df["year"].between(
            2014,
            2015,
        )
    ].copy()

    test = df[
        df["year"].between(
            2016,
            2017,
        )
    ].copy()

    if len(train) == 0:
        raise ValueError(
            "Training split is empty."
        )

    if len(calibration) == 0:
        raise ValueError(
            "Calibration split is empty."
        )

    if len(test) == 0:
        raise ValueError(
            "Test split is empty."
        )

    print(
        "TRAIN ROWS:",
        len(train),
    )

    print(
        "CALIBRATION ROWS:",
        len(calibration),
    )

    print(
        "TEST ROWS:",
        len(test),
    )

    print(
        "\nYEAR RANGES:"
    )

    print(
        "TRAIN:",
        train["year"].min(),
        "-",
        train["year"].max(),
    )

    print(
        "CALIBRATION:",
        calibration["year"].min(),
        "-",
        calibration["year"].max(),
    )

    print(
        "TEST:",
        test["year"].min(),
        "-",
        test["year"].max(),
    )

    return (
        train,
        calibration,
        test,
    )


# ==============================================================
# CALIBRATION METRICS
# ==============================================================

def probability_metrics(
    y,
    probability,
):

    return {
        "pr_auc": float(
            average_precision_score(
                y,
                probability,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                probability,
            )
        ),
        "brier": float(
            brier_score_loss(
                y,
                probability,
            )
        ),
        "mean_probability": float(
            np.mean(probability)
        ),
    }


# ==============================================================
# MAIN CALIBRATION
# ==============================================================

def main():

    header(
        "8.1 MODEL CALIBRATION V4"
    )

    # ----------------------------------------------------------
    # Load
    # ----------------------------------------------------------

    schema = load_schema()

    df = load_dataset()

    validate_dataset(
        df
    )

    # ----------------------------------------------------------
    # Normalize month and season in source dataset
    # ----------------------------------------------------------

    df["month"] = normalize_month(
        df["month"]
    )

    df["season"] = df[
        "month"
    ].map(
        season_from_month
    )

    # ----------------------------------------------------------
    # Build model matrix
    # ----------------------------------------------------------

    X = build_model_matrix(
        df,
        schema,
    )

    # ----------------------------------------------------------
    # Load model
    # ----------------------------------------------------------

    model = load_model()

    # ----------------------------------------------------------
    # Validate schema
    # ----------------------------------------------------------

    validate_model_schema(
        model,
        X,
    )

    # ----------------------------------------------------------
    # Split
    # ----------------------------------------------------------

    (
        train_df,
        calibration_df,
        test_df,
    ) = temporal_split(
        df
    )

    X_calibration = X.loc[
        calibration_df.index
    ].copy()

    X_test = X.loc[
        test_df.index
    ].copy()

    y_calibration = (
        calibration_df[TARGET]
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test_df[TARGET]
        .astype(int)
        .to_numpy()
    )

    print(
        "\nCALIBRATION TARGET CLASSES:"
    )

    print(
        sorted(
            np.unique(
                y_calibration
            ).tolist()
        )
    )

    print(
        "TEST TARGET CLASSES:"
    )

    print(
        sorted(
            np.unique(
                y_test
            ).tolist()
        )
    )

    # ----------------------------------------------------------
    # Raw probabilities
    # ----------------------------------------------------------

    header(
        "GENERATING RAW PROBABILITIES"
    )

    raw_calibration = get_probability(
        model,
        X_calibration,
    )

    raw_test = get_probability(
        model,
        X_test,
    )

    print(
        "RAW CALIBRATION PROBABILITY: PASS"
    )

    print(
        "RAW TEST PROBABILITY: PASS"
    )

    # ----------------------------------------------------------
    # Isotonic calibration
    # ----------------------------------------------------------

    header(
        "ISOTONIC CALIBRATION"
    )

    isotonic = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    isotonic.fit(
        raw_calibration,
        y_calibration,
    )

    isotonic_test = isotonic.predict(
        raw_test
    )

    print(
        "ISOTONIC CALIBRATION: PASS"
    )

    # ----------------------------------------------------------
    # Sigmoid calibration
    # ----------------------------------------------------------

    header(
        "SIGMOID CALIBRATION"
    )

    raw_calibration_2d = (
        raw_calibration
        .reshape(-1, 1)
    )

    raw_test_2d = (
        raw_test
        .reshape(-1, 1)
    )

    sigmoid = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=42,
    )

    sigmoid.fit(
        raw_calibration_2d,
        y_calibration,
    )

    sigmoid_test = sigmoid.predict_proba(
        raw_test_2d
    )[:, 1]

    print(
        "SIGMOID CALIBRATION: PASS"
    )

    # ----------------------------------------------------------
    # Probability metrics
    # ----------------------------------------------------------

    header(
        "CALIBRATION QUALITY"
    )

    raw_metrics = probability_metrics(
        y_test,
        raw_test,
    )

    isotonic_metrics = probability_metrics(
        y_test,
        isotonic_test,
    )

    sigmoid_metrics = probability_metrics(
        y_test,
        sigmoid_test,
    )

    print(
        "\nRAW PROBABILITY"
    )

    for key, value in raw_metrics.items():
        print(
            f"{key.upper()}: {value:.6f}"
        )

    print(
        "\nISOTONIC PROBABILITY"
    )

    for key, value in isotonic_metrics.items():
        print(
            f"{key.upper()}: {value:.6f}"
        )

    print(
        "\nSIGMOID PROBABILITY"
    )

    for key, value in sigmoid_metrics.items():
        print(
            f"{key.upper()}: {value:.6f}"
        )

    # ----------------------------------------------------------
    # Select calibration
    #
    # Prefer lower Brier score.
    # ----------------------------------------------------------

    candidates = {
        "raw_probability": raw_metrics,
        "isotonic_probability": isotonic_metrics,
        "sigmoid_probability": sigmoid_metrics,
    }

    selected = min(
        candidates,
        key=lambda name:
        candidates[name]["brier"],
    )

    print()
    print(
        "SELECTED CALIBRATION:",
        selected,
    )

    # ----------------------------------------------------------
    # Build prediction output
    # ----------------------------------------------------------

    output = test_df[
        [
            "subdivision",
            "year",
            "month",
            "season",
        ]
    ].copy()

    output["actual"] = y_test

    output["raw_probability"] = (
        raw_test
    )

    output["isotonic_probability"] = (
        isotonic_test
    )

    output["sigmoid_probability"] = (
        sigmoid_test
    )

    # ----------------------------------------------------------
    # Validate output
    # ----------------------------------------------------------

    if len(output) != len(
        test_df
    ):
        raise ValueError(
            "Calibration output row count mismatch."
        )

    if output[
        "month"
    ].isna().any():

        raise ValueError(
            "NULL months in calibration output."
        )

    if not output[
        "month"
    ].between(1, 12).all():

        raise ValueError(
            "Invalid months in calibration output."
        )

    for column in [
        "raw_probability",
        "isotonic_probability",
        "sigmoid_probability",
    ]:

        if not output[
            column
        ].between(0, 1).all():

            raise ValueError(
                f"Invalid probability values: {column}"
            )

    # ----------------------------------------------------------
    # Save predictions
    # ----------------------------------------------------------

    header(
        "SAVING CALIBRATION OUTPUT"
    )

    output.to_csv(
        CALIBRATED_FILE,
        index=False,
    )

    print(
        "CALIBRATED PREDICTIONS:"
    )

    print(
        CALIBRATED_FILE
    )

    # ----------------------------------------------------------
    # Save metrics
    # ----------------------------------------------------------

    metrics_rows = []

    for name, metrics in candidates.items():

        row = {
            "probability": name,
            **metrics,
        }

        metrics_rows.append(
            row
        )

    metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    print(
        "CALIBRATION METRICS:"
    )

    print(
        METRICS_FILE
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------

    summary = {
    "dataset": str(DATA_FILE),
    "model": str(MODEL_FILE),
    "schema": str(SCHEMA_FILE),
    "rows": int(len(output)),
    "target": TARGET,
    "target_rate": float(
        output["actual"].mean()
    ),
    "selected_calibration": selected,
    "raw_metrics": raw_metrics,
    "isotonic_metrics": isotonic_metrics,
    "sigmoid_metrics": sigmoid_metrics,

    # Fitted sigmoid calibration parameters.
    # Calibration is fitted directly on raw XGBoost
    # probabilities using LogisticRegression.
    "sigmoid_parameters": {
        "a": float(sigmoid.coef_[0][0]),
        "b": float(sigmoid.intercept_[0]),
    },

    "feature_count": len(FEATURES),
    "categorical_features": CATEGORICAL_FEATURES,
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "CALIBRATION SUMMARY:"
    )

    print(
        SUMMARY_FILE
    )

    # ----------------------------------------------------------
    # Final
    # ----------------------------------------------------------

    header(
        "8.1 MODEL CALIBRATION V4 COMPLETE"
    )

    print(
        "STATUS: PASS"
    )

    print(
        "SELECTED:",
        selected,
    )

    print(
        "OUTPUT ROWS:",
        len(output),
    )


if __name__ == "__main__":
    main()