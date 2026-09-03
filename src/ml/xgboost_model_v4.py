from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v4.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "model_v4"
)

MODEL_FILE = (
    MODEL_DIR
    / "xgboost_model.json"
)

TEST_FILE = (
    MODEL_DIR
    / "test_predictions.csv"
)

METRICS_FILE = (
    MODEL_DIR
    / "model_metrics.csv"
)

SCHEMA_FILE = (
    MODEL_DIR
    / "model_schema.json"
)


# ======================================================================
# TARGET
# ======================================================================

TARGET = "target_3m_severe_anomaly"


# ======================================================================
# EXACT MODEL FEATURE ORDER
# ======================================================================

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


CATEGORICAL_FEATURES = [
    "subdivision",
    "month",
    "season",
]


NUMERIC_FEATURES = [
    feature
    for feature in FEATURES
    if feature not in CATEGORICAL_FEATURES
]


# ======================================================================
# HEADER
# ======================================================================

def header(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ======================================================================
# MONTH NORMALIZATION
# ======================================================================

def normalize_month(series):

    month_map = {
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

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    text = (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )

    mapped = text.map(month_map)

    result = numeric.copy()

    missing = result.isna()

    result.loc[missing] = mapped.loc[missing]

    # Detect zero-based encoding.
    valid = result.dropna()

    if not valid.empty:

        if (
            valid.min() == 0
            and valid.max() <= 11
        ):
            result = result + 1

    return result


# ======================================================================
# SEASON
# ======================================================================

def derive_season(month):

    result = pd.Series(
        "UNKNOWN",
        index=month.index,
        dtype="string",
    )

    result.loc[
        month.isin([12, 1, 2])
    ] = "WINTER"

    result.loc[
        month.isin([3, 4, 5])
    ] = "PRE_MONSOON"

    result.loc[
        month.isin([6, 7, 8, 9])
    ] = "MONSOON"

    result.loc[
        month.isin([10, 11])
    ] = "POST_MONSOON"

    return result


# ======================================================================
# LOAD
# ======================================================================

def load_data():

    header(
        "LOADING V4 DATASET"
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        "INPUT:",
        INPUT_FILE
    )

    print(
        "SHAPE:",
        df.shape
    )

    required = set(
        FEATURES + [TARGET]
    )

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    return df


# ======================================================================
# PREPARE DATA
# ======================================================================

def prepare_data(df):

    header(
        "PREPARING V4 TRAINING DATA"
    )

    df = df.copy()

    # --------------------------------------------------------------
    # MONTH
    # --------------------------------------------------------------

    df["month"] = normalize_month(
        df["month"]
    )

    if df["month"].isna().any():

        raise ValueError(
            "NULL month values found."
        )

    if (
        (df["month"] < 1)
        | (df["month"] > 12)
    ).any():

        raise ValueError(
            "Invalid month values found."
        )

    df["month"] = (
        df["month"]
        .astype(int)
    )

    # --------------------------------------------------------------
    # YEAR
    # --------------------------------------------------------------

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    if df["year"].isna().any():

        raise ValueError(
            "Invalid year values."
        )

    df["year"] = (
        df["year"]
        .astype(int)
    )

    # --------------------------------------------------------------
    # SUBDIVISION
    # --------------------------------------------------------------

    df["subdivision"] = (
        df["subdivision"]
        .astype("string")
        .fillna("UNKNOWN")
        .str.strip()
    )

    # --------------------------------------------------------------
    # SEASON
    # --------------------------------------------------------------

    df["season"] = derive_season(
        df["month"]
    )

    if (
        df["season"]
        == "UNKNOWN"
    ).any():

        raise ValueError(
            "UNKNOWN season generated."
        )

    # --------------------------------------------------------------
    # NUMERIC FEATURES
    # --------------------------------------------------------------

    for feature in NUMERIC_FEATURES:

        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

        median = df[feature].median()

        if pd.isna(median):

            raise ValueError(
                f"Feature has no valid values: {feature}"
            )

        df[feature] = (
            df[feature]
            .fillna(median)
        )

    # --------------------------------------------------------------
    # CRITICAL:
    # CONVERT CATEGORICAL VALUES TO STRINGS FIRST
    # --------------------------------------------------------------

    df["subdivision"] = (
        df["subdivision"]
        .astype(str)
    )

    df["month"] = (
        df["month"]
        .astype(int)
        .astype(str)
    )

    df["season"] = (
        df["season"]
        .astype(str)
    )

    # --------------------------------------------------------------
    # FIXED CATEGORY DEFINITIONS
    # --------------------------------------------------------------

    subdivision_categories = sorted(
        df["subdivision"]
        .unique()
        .tolist()
    )

    month_categories = [
        str(x)
        for x in range(1, 13)
    ]

    season_categories = [
        "MONSOON",
        "POST_MONSOON",
        "PRE_MONSOON",
        "WINTER",
    ]

    df["subdivision"] = pd.Categorical(
        df["subdivision"],
        categories=subdivision_categories,
    )

    df["month"] = pd.Categorical(
        df["month"],
        categories=month_categories,
    )

    df["season"] = pd.Categorical(
        df["season"],
        categories=season_categories,
    )

    print(
        "SUBDIVISION CATEGORIES:",
        len(subdivision_categories)
    )

    print(
        "MONTH CATEGORIES:",
        month_categories
    )

    print(
        "SEASON CATEGORIES:",
        season_categories
    )

    return df


# ======================================================================
# BUILD X
# ======================================================================

def build_X(df):

    X = df[
        FEATURES
    ].copy()

    # Exact categorical dtype.
    X["subdivision"] = pd.Categorical(
        X["subdivision"],
        categories=df[
            "subdivision"
        ].cat.categories,
    )

    X["month"] = pd.Categorical(
        X["month"],
        categories=[
            str(x)
            for x in range(1, 13)
        ],
    )

    X["season"] = pd.Categorical(
        X["season"],
        categories=[
            "MONSOON",
            "POST_MONSOON",
            "PRE_MONSOON",
            "WINTER",
        ],
    )

    # Numeric dtype.
    for feature in NUMERIC_FEATURES:

        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

        X[feature] = (
            X[feature]
            .fillna(
                X[feature].median()
            )
        )

    X = X[
        FEATURES
    ]

    if X.isna().any().any():

        raise ValueError(
            "NULL values remain in model matrix."
        )

    return X


# ======================================================================
# VALIDATE
# ======================================================================

def validate_matrix(X):

    header(
        "MODEL MATRIX VALIDATION"
    )

    print(
        "FEATURE COUNT:",
        len(X.columns)
    )

    print(
        "FEATURE ORDER:"
    )

    print(
        list(X.columns)
    )

    if list(X.columns) != FEATURES:

        raise ValueError(
            "Feature order mismatch."
        )

    print(
        "\nFINAL DTYPES:"
    )

    for column in X.columns:

        print(
            f"{column}: {X[column].dtype}"
        )

    print(
        "\nCATEGORICAL FEATURES:"
    )

    for column in CATEGORICAL_FEATURES:

        if not pd.api.types.is_categorical_dtype(
            X[column]
        ):

            raise ValueError(
                f"{column} is not categorical."
            )

        print(
            column,
            "PASS"
        )

    print(
        "\nMODEL MATRIX: PASS"
    )


# ======================================================================
# TRAIN
# ======================================================================

def train_model(
    X_train,
    y_train,
    X_valid,
    y_valid,
):

    header(
        "TRAINING XGBOOST V4"
    )

    positive = int(
        y_train.sum()
    )

    negative = int(
        len(y_train)
        - positive
    )

    scale_pos_weight = (
        negative / positive
    )

    print(
        "POSITIVE COUNT:",
        positive
    )

    print(
        "NEGATIVE COUNT:",
        negative
    )

    print(
        "SCALE POS WEIGHT:",
        scale_pos_weight
    )

    model = xgb.XGBClassifier(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_valid,
                y_valid,
            )
        ],
        verbose=False,
    )

    return model


# ======================================================================
# METRICS
# ======================================================================

def calculate_metrics(
    y,
    probability,
):

    return {
        "pr_auc":
            average_precision_score(
                y,
                probability,
            ),

        "roc_auc":
            roc_auc_score(
                y,
                probability,
            ),

        "brier":
            brier_score_loss(
                y,
                probability,
            ),
    }


# ======================================================================
# SAVE SCHEMA
# ======================================================================

def save_schema():

    schema = {

        "features":
            FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_categories": {

            "month":
                [
                    str(x)
                    for x in range(1, 13)
                ],

            "season":
                [
                    "MONSOON",
                    "POST_MONSOON",
                    "PRE_MONSOON",
                    "WINTER",
                ],
        },
    }

    with open(
        SCHEMA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            schema,
            file,
            indent=2,
        )


# ======================================================================
# MAIN
# ======================================================================

def main():

    header(
        "XGBOOST V4 MODEL TRAINING"
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # LOAD
    # --------------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------------
    # PREPARE
    # --------------------------------------------------------------

    df = prepare_data(
        df
    )

    # --------------------------------------------------------------
    # TARGET
    # --------------------------------------------------------------

    y = (
        df[TARGET]
        .astype(int)
    )

    X = build_X(
        df
    )

    validate_matrix(
        X
    )

    print(
        "\nMODEL FEATURES:",
        len(FEATURES)
    )

    print(
        "TARGET:",
        TARGET
    )

    print(
        "TARGET DISTRIBUTION:"
    )

    print(
        y.value_counts()
    )

    print(
        "TARGET RATE:",
        y.mean()
    )

    # --------------------------------------------------------------
    # TEMPORAL SPLIT
    # --------------------------------------------------------------

    header(
        "TEMPORAL TRAIN / VALIDATION / TEST SPLIT"
    )

    train_mask = (
        df["year"] <= 2013
    )

    valid_mask = (
        df["year"].between(
            2014,
            2015,
        )
    )

    test_mask = (
        df["year"].between(
            2016,
            2017,
        )
    )

    X_train = X.loc[
        train_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    X_valid = X.loc[
        valid_mask
    ]

    y_valid = y.loc[
        valid_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    y_test = y.loc[
        test_mask
    ]

    print(
        "TRAIN ROWS:",
        len(X_train)
    )

    print(
        "VALIDATION ROWS:",
        len(X_valid)
    )

    print(
        "TEST ROWS:",
        len(X_test)
    )

    print()

    print(
        "YEAR RANGES:"
    )

    print(
        "TRAIN:",
        df.loc[
            train_mask,
            "year"
        ].min(),
        "-",
        df.loc[
            train_mask,
            "year"
        ].max(),
    )

    print(
        "VALIDATION:",
        df.loc[
            valid_mask,
            "year"
        ].min(),
        "-",
        df.loc[
            valid_mask,
            "year"
        ].max(),
    )

    print(
        "TEST:",
        df.loc[
            test_mask,
            "year"
        ].min(),
        "-",
        df.loc[
            test_mask,
            "year"
        ].max(),
    )

    # --------------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
        X_valid,
        y_valid,
    )

    # --------------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------------

    header(
        "XGBOOST V4 - VALIDATION"
    )

    valid_probability = (
        model.predict_proba(
            X_valid
        )[:, 1]
    )

    valid_metrics = calculate_metrics(
        y_valid,
        valid_probability,
    )

    print(
        f"PR-AUC: {valid_metrics['pr_auc']:.6f}"
    )

    print(
        f"ROC-AUC: {valid_metrics['roc_auc']:.6f}"
    )

    print(
        f"BRIER: {valid_metrics['brier']:.6f}"
    )

    # --------------------------------------------------------------
    # TEST
    # --------------------------------------------------------------

    header(
        "XGBOOST V4 - TEST"
    )

    test_probability = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probability,
    )

    print(
        f"PR-AUC: {test_metrics['pr_auc']:.6f}"
    )

    print(
        f"ROC-AUC: {test_metrics['roc_auc']:.6f}"
    )

    print(
        f"BRIER: {test_metrics['brier']:.6f}"
    )

    # --------------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------------

    header(
        "SAVING MODEL"
    )

    model.save_model(
        str(MODEL_FILE)
    )

    print(
        "MODEL SAVED:"
    )

    print(
        MODEL_FILE
    )

    # --------------------------------------------------------------
    # SAVE SCHEMA
    # --------------------------------------------------------------

    save_schema()

    print(
        "SCHEMA SAVED:"
    )

    print(
        SCHEMA_FILE
    )

    # --------------------------------------------------------------
    # RELOAD
    # --------------------------------------------------------------

    header(
        "MODEL COMPATIBILITY TEST"
    )

    reloaded = xgb.XGBClassifier()

    reloaded.load_model(
        str(MODEL_FILE)
    )

    # IMPORTANT:
    # use EXACT categorical representation
    compatibility_probability = (
        reloaded.predict_proba(
            X_test.iloc[:10].copy()
        )[:, 1]
    )

    print(
        "PREDICTION: PASS"
    )

    print(
        "PROBABILITIES:"
    )

    print(
        compatibility_probability
    )

    # --------------------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------------------

    predictions = df.loc[
        test_mask,
        [
            "subdivision",
            "year",
            "month",
            "season",
        ],
    ].copy()

    # Convert categorical output to normal values.
    predictions["subdivision"] = (
        predictions["subdivision"]
        .astype(str)
    )

    predictions["month"] = (
        predictions["month"]
        .astype(str)
    )

    predictions["season"] = (
        predictions["season"]
        .astype(str)
    )

    predictions["actual"] = (
        y_test.to_numpy()
    )

    predictions["raw_probability"] = (
        test_probability
    )

    predictions.to_csv(
        TEST_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # METRICS
    # --------------------------------------------------------------

    metrics_df = pd.DataFrame(
        [
            {
                "dataset": "validation",
                **valid_metrics,
            },
            {
                "dataset": "test",
                **test_metrics,
            },
        ]
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # FINAL
    # --------------------------------------------------------------

    header(
        "XGBOOST V4 COMPLETE"
    )

    print(
        "STATUS: PASS"
    )

    print(
        "MODEL:",
        MODEL_FILE
    )

    print(
        "FEATURE COUNT:",
        len(FEATURES)
    )

    print(
        "CATEGORICAL:",
        CATEGORICAL_FEATURES
    )

    print(
        "SCHEMA:",
        SCHEMA_FILE
    )


if __name__ == "__main__":
    main()