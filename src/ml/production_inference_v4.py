# ======================================================================
# BHARAT EARTH V4 - PRODUCTION INFERENCE
# ======================================================================
#
# Purpose:
#   Run production inference using the validated V4 model pipeline.
#
# Pipeline:
#   Dataset
#       ↓
#   XGBoost V4
#       ↓
#   Sigmoid Calibration
#       ↓
#   Policy Threshold
#       ↓
#   Production Prediction
#
# Final validated policy:
#   sigmoid_probability >= 0.09
#
# ======================================================================

from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


# ======================================================================
# PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "features"

DATA_FILE = DATA_DIR / "ml_dataset_v4.csv"

MODEL_DIR = DATA_DIR / "model_v4"
MODEL_FILE = MODEL_DIR / "xgboost_model.json"
SCHEMA_FILE = MODEL_DIR / "model_schema.json"

CALIBRATION_DIR = DATA_DIR / "calibration_v4"
CALIBRATION_SUMMARY_FILE = (
    CALIBRATION_DIR / "calibration_summary.json"
)

POLICY_DIR = DATA_DIR / "policy_v4"
POLICY_FILE = POLICY_DIR / "selected_policy.csv"

OUTPUT_DIR = DATA_DIR / "inference_v4"

PREDICTIONS_FILE = OUTPUT_DIR / "production_predictions.csv"
SUMMARY_FILE = OUTPUT_DIR / "inference_summary.json"
METRICS_FILE = OUTPUT_DIR / "inference_metrics.csv"


# ======================================================================
# CONSTANTS
# ======================================================================

TARGET = "target_3m_severe_anomaly"

EXPECTED_THRESHOLD = 0.09

EXPECTED_PROBABILITY_TYPE = "sigmoid_probability"

EXPECTED_CATEGORICAL_FEATURES = [
    "subdivision",
    "month",
    "season",
]

EXPECTED_FEATURE_COUNT = 23


# ======================================================================
# LOGGING
# ======================================================================

def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def status(name: str, result: str, detail: str = "") -> None:
    print(
        f"{name:<38}"
        f"{result:<8}"
        f"{detail}"
    )


# ======================================================================
# UTILITY
# ======================================================================

def fail(message: str) -> None:
    raise RuntimeError(message)


def json_safe(value: Any) -> Any:
    """
    Convert numpy/pandas objects into JSON-safe Python values.
    """

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if pd.isna(value):
        return None

    return value


# ======================================================================
# LOAD SCHEMA
# ======================================================================

def load_schema() -> dict:
    banner("LOADING SAVED MODEL SCHEMA")

    if not SCHEMA_FILE.exists():
        fail(
            f"Schema file not found:\n{SCHEMA_FILE}"
        )

    with open(
        SCHEMA_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        schema = json.load(f)

    status(
        "SCHEMA LOAD",
        "PASS",
        str(SCHEMA_FILE),
    )

    return schema


# ======================================================================
# EXTRACT SCHEMA INFORMATION
# ======================================================================

def extract_feature_schema(
    schema: dict,
    model: xgb.Booster | None = None,
) -> tuple[list[str], dict[str, list[Any]]]:
    """
    Extract feature names and categorical categories.

    Supports several reasonable schema layouts so that the inference
    script is robust to the existing V4 model_schema.json format.
    """

    feature_names: list[str] = []
    categories: dict[str, list[Any]] = {}

    # --------------------------------------------------------------
    # Feature names
    # --------------------------------------------------------------

    possible_feature_keys = [
        "feature_names",
        "features",
        "model_features",
        "columns",
    ]

    for key in possible_feature_keys:
        value = schema.get(key)

        if isinstance(value, list):
            if all(
                isinstance(x, str)
                for x in value
            ):
                feature_names = value
                break

    # --------------------------------------------------------------
    # Features stored as dictionaries
    # --------------------------------------------------------------

    if not feature_names:
        features_obj = schema.get("features")

        if isinstance(features_obj, dict):
            feature_names = list(
                features_obj.keys()
            )

    # --------------------------------------------------------------
    # Model fallback
    # --------------------------------------------------------------

    if not feature_names and model is not None:
        feature_names = list(
            model.feature_names or []
        )

    if not feature_names:
        fail(
            "Unable to determine model feature names "
            "from model_schema.json or XGBoost model."
        )

    # --------------------------------------------------------------
    # Categories
    # --------------------------------------------------------------

    categorical_schema = schema.get(
        "categorical_features",
        {},
    )

    if isinstance(
        categorical_schema,
        dict,
    ):
        for key, value in categorical_schema.items():
            if isinstance(value, list):
                categories[key] = value

    # Alternate schema layout
    if not categories:
        category_obj = schema.get(
            "categories",
            {},
        )

        if isinstance(
            category_obj,
            dict,
        ):
            for key, value in category_obj.items():
                if isinstance(value, list):
                    categories[key] = value

    return feature_names, categories


# ======================================================================
# LOAD MODEL
# ======================================================================

def load_model() -> xgb.Booster:
    banner("LOADING XGBOOST V4 MODEL")

    if not MODEL_FILE.exists():
        fail(
            f"Model file not found:\n{MODEL_FILE}"
        )

    model = xgb.Booster()

    model.load_model(
        str(MODEL_FILE)
    )

    feature_count = len(
        model.feature_names or []
    )

    status(
        "MODEL LOAD",
        "PASS",
        f"features={feature_count}",
    )

    return model


# ======================================================================
# LOAD CALIBRATION
# ======================================================================

def load_calibration_summary() -> dict:
    banner("LOADING SIGMOID CALIBRATION")

    if not CALIBRATION_SUMMARY_FILE.exists():
        fail(
            "Calibration summary not found:\n"
            f"{CALIBRATION_SUMMARY_FILE}"
        )

    with open(
        CALIBRATION_SUMMARY_FILE,
        "r",
        encoding="utf-8",
    ) as f:
        summary = json.load(f)

    status(
        "CALIBRATION SUMMARY",
        "PASS",
        str(CALIBRATION_SUMMARY_FILE),
    )

    return summary


# ======================================================================
# LOAD POLICY
# ======================================================================

def load_policy() -> tuple[float, str]:
    banner("LOADING PRODUCTION POLICY")

    if not POLICY_FILE.exists():
        fail(
            "Policy file not found:\n"
            f"{POLICY_FILE}"
        )

    policy = pd.read_csv(
        POLICY_FILE
    )

    if policy.empty:
        fail(
            "Selected policy file is empty."
        )

    if "threshold" not in policy.columns:
        fail(
            "Policy file does not contain "
            "'threshold'."
        )

    threshold = float(
        policy.iloc[0]["threshold"]
    )

    probability_type = (
        str(
            policy.iloc[0].get(
                "probability_type",
                EXPECTED_PROBABILITY_TYPE,
            )
        )
    )

    status(
        "POLICY LOAD",
        "PASS",
        f"type={probability_type}, "
        f"threshold={threshold:.6f}",
    )

    return threshold, probability_type


# ======================================================================
# VALIDATE POLICY
# ======================================================================

def validate_policy(
    threshold: float,
    probability_type: str,
) -> None:
    banner("POLICY VALIDATION")

    if not math.isfinite(threshold):
        fail(
            "Policy threshold is not finite."
        )

    if not 0.0 < threshold < 1.0:
        fail(
            f"Invalid policy threshold: {threshold}"
        )

    if abs(
        threshold - EXPECTED_THRESHOLD
    ) > 1e-9:
        fail(
            "Policy threshold differs from "
            f"validated V4 threshold.\n"
            f"Expected: {EXPECTED_THRESHOLD}\n"
            f"Found: {threshold}"
        )

    if (
        probability_type
        != EXPECTED_PROBABILITY_TYPE
    ):
        fail(
            "Unexpected probability type.\n"
            f"Expected: {EXPECTED_PROBABILITY_TYPE}\n"
            f"Found: {probability_type}"
        )

    status(
        "PROBABILITY TYPE",
        "PASS",
        probability_type,
    )

    status(
        "THRESHOLD",
        "PASS",
        f"{threshold:.6f}",
    )


# ======================================================================
# LOAD DATASET
# ======================================================================

def load_dataset() -> pd.DataFrame:
    banner("LOADING V4 DATASET")

    if not DATA_FILE.exists():
        fail(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print("INPUT:")
    print(DATA_FILE)

    print("SHAPE:")
    print(df.shape)

    print("COLUMNS:")
    print(list(df.columns))

    return df


# ======================================================================
# MONTH NORMALIZATION
# ======================================================================

def normalize_month(
    series: pd.Series,
) -> pd.Series:
    """
    Convert month representations such as:

        1
        '1'
        JAN
        January
        'JANUARY'

    into canonical string categories:

        '1' ... '12'
    """

    month_map = {
        "JAN": "1",
        "JANUARY": "1",

        "FEB": "2",
        "FEBRUARY": "2",

        "MAR": "3",
        "MARCH": "3",

        "APR": "4",
        "APRIL": "4",

        "MAY": "5",

        "JUN": "6",
        "JUNE": "6",

        "JUL": "7",
        "JULY": "7",

        "AUG": "8",
        "AUGUST": "8",

        "SEP": "9",
        "SEPT": "9",
        "SEPTEMBER": "9",

        "OCT": "10",
        "OCTOBER": "10",

        "NOV": "11",
        "NOVEMBER": "11",

        "DEC": "12",
        "DECEMBER": "12",
    }

    raw = (
        series
        .astype("string")
        .str.strip()
    )

    upper = raw.str.upper()

    mapped = upper.map(
        month_map
    )

    numeric = pd.to_numeric(
        upper,
        errors="coerce",
    )

    numeric_valid = numeric.where(
        numeric.between(1, 12)
    )

    result = mapped.copy()

    missing_mapping = result.isna()

    result.loc[
        missing_mapping
    ] = (
        numeric_valid
        .loc[missing_mapping]
        .astype("Int64")
        .astype("string")
    )

    return result


# ======================================================================
# SEASON NORMALIZATION
# ======================================================================

def season_from_month(
    month: pd.Series,
) -> pd.Series:
    month_num = pd.to_numeric(
        month,
        errors="coerce",
    )

    result = pd.Series(
        pd.NA,
        index=month.index,
        dtype="string",
    )

    # Winter
    result.loc[
        month_num.isin([12, 1, 2])
    ] = "WINTER"

    # Pre-monsoon
    result.loc[
        month_num.isin([3, 4, 5])
    ] = "PRE_MONSOON"

    # Monsoon
    result.loc[
        month_num.isin([6, 7, 8, 9])
    ] = "MONSOON"

    # Post-monsoon
    result.loc[
        month_num.isin([10, 11])
    ] = "POST_MONSOON"

    return result


# ======================================================================
# VALIDATE DATASET
# ======================================================================

def validate_dataset(
    df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:

    banner("DATASET VALIDATION")

    if TARGET not in df.columns:
        fail(
            f"Target column not found: {TARGET}"
        )

    duplicates = int(
        df.duplicated().sum()
    )

    print(
        "EXACT DUPLICATES:",
        duplicates,
    )

    if duplicates:
        fail(
            f"Exact duplicate rows found: "
            f"{duplicates}"
        )

    # --------------------------------------------------------------
    # Month
    # --------------------------------------------------------------

    print()
    print("MONTH NORMALIZATION")

    normalized_month = normalize_month(
        df["month"]
    )

    invalid_before = int(
        normalized_month.isna().sum()
    )

    print(
        "INVALID MONTHS:",
        invalid_before,
    )

    if invalid_before:
        invalid_values = (
            df.loc[
                normalized_month.isna(),
                "month",
            ]
            .value_counts()
            .head(20)
        )

        print(
            "INVALID MONTH VALUES:"
        )
        print(invalid_values)

        fail(
            "Unable to normalize all month values."
        )

    df["month"] = normalized_month

    invalid_after = int(
        pd.to_numeric(
            df["month"],
            errors="coerce",
        )
        .isna()
        .sum()
    )

    if invalid_after:
        fail(
            "Invalid month values remain."
        )

    month_numbers = pd.to_numeric(
        df["month"],
        errors="coerce",
    )

    if not month_numbers.between(
        1,
        12,
    ).all():
        fail(
            "Month values outside 1-12 found."
        )

    status(
        "MONTH VALIDATION",
        "PASS",
        "1-12",
    )

    # --------------------------------------------------------------
    # Season
    # --------------------------------------------------------------

    expected_season = season_from_month(
        df["month"]
    )

    actual_season = (
        df["season"]
        .astype("string")
        .str.strip()
        .str.upper()
        .str.replace(
            " ",
            "_",
            regex=False,
        )
    )

    inconsistencies = int(
        (
            actual_season
            != expected_season
        ).sum()
    )

    print(
        "SEASON INCONSISTENCIES:",
        inconsistencies,
    )

    if inconsistencies:
        print(
            "REPAIRING season from month."
        )

        df["season"] = (
            expected_season
        )
    else:
        df["season"] = actual_season

    valid_seasons = {
        "MONSOON",
        "POST_MONSOON",
        "PRE_MONSOON",
        "WINTER",
    }

    if not set(
        df["season"].dropna().unique()
    ).issubset(valid_seasons):
        fail(
            "Invalid season values found."
        )

    status(
        "SEASON VALIDATION",
        "PASS",
        f"inconsistencies_repaired="
        f"{inconsistencies}",
    )

    # --------------------------------------------------------------
    # Target
    # --------------------------------------------------------------

    target_numeric = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    if target_numeric.isna().any():
        fail(
            "Target contains non-numeric values."
        )

    target_values = sorted(
        target_numeric
        .unique()
        .tolist()
    )

    if not set(target_values).issubset(
        {0, 1}
    ):
        fail(
            f"Invalid target values: "
            f"{target_values}"
        )

    df[TARGET] = (
        target_numeric
        .astype("int64")
    )

    print(
        "TARGET VALUES:",
        target_values,
    )

    print(
        "TARGET RATE:",
        float(df[TARGET].mean()),
    )

    # --------------------------------------------------------------
    # Required model features
    # --------------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_names
        if feature not in df.columns
    ]

    if missing_features:
        fail(
            "Missing model features:\n"
            f"{missing_features}"
        )

    status(
        "MODEL FEATURE SCHEMA",
        "PASS",
        f"features={len(feature_names)}",
    )

    return df


# ======================================================================
# BUILD MODEL MATRIX
# ======================================================================

def build_model_matrix(
    df: pd.DataFrame,
    feature_names: list[str],
    categories: dict[str, list[Any]],
) -> pd.DataFrame:

    banner("BUILDING EXACT MODEL MATRIX")

    X = df[
        feature_names
    ].copy()

    # --------------------------------------------------------------
    # subdivision
    # --------------------------------------------------------------

    if "subdivision" in X.columns:
        if "subdivision" in categories:
            subdivision_categories = [
                str(x)
                for x in categories[
                    "subdivision"
                ]
            ]
        else:
            subdivision_categories = sorted(
                X["subdivision"]
                .astype("string")
                .dropna()
                .unique()
                .tolist()
            )

        X["subdivision"] = pd.Categorical(
            X["subdivision"]
            .astype("string"),
            categories=subdivision_categories,
        )

    # --------------------------------------------------------------
    # month
    # --------------------------------------------------------------

    if "month" in X.columns:
        month_categories = [
            str(x)
            for x in categories.get(
                "month",
                [
                    str(i)
                    for i in range(1, 13)
                ],
            )
        ]

        X["month"] = pd.Categorical(
            X["month"]
            .astype("string"),
            categories=month_categories,
        )

    # --------------------------------------------------------------
    # season
    # --------------------------------------------------------------

    if "season" in X.columns:
        season_categories = [
            str(x)
            for x in categories.get(
                "season",
                [
                    "MONSOON",
                    "POST_MONSOON",
                    "PRE_MONSOON",
                    "WINTER",
                ],
            )
        ]

        X["season"] = pd.Categorical(
            X["season"]
            .astype("string"),
            categories=season_categories,
        )

    # --------------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------------

    categorical = set(
        EXPECTED_CATEGORICAL_FEATURES
    )

    for feature in feature_names:
        if feature in categorical:
            continue

        X[feature] = pd.to_numeric(
            X[feature],
            errors="coerce",
        )

    # --------------------------------------------------------------
    # Missing values
    # --------------------------------------------------------------

    null_before = int(
        X.isna().sum().sum()
    )

    print(
        "MISSING VALUES BEFORE:",
        null_before,
    )

    # Numeric missing values
    for feature in feature_names:
        if feature in categorical:
            continue

        if X[feature].isna().any():
            median = X[feature].median()

            if pd.isna(median):
                median = 0.0

            X[feature] = (
                X[feature]
                .fillna(median)
            )

    # Categorical missing values
    for feature in categorical:
        if feature not in X.columns:
            continue

        if X[feature].isna().any():
            fail(
                f"Missing categorical values "
                f"found in {feature}."
            )

    null_after = int(
        X.isna().sum().sum()
    )

    print(
        "MISSING VALUES AFTER:",
        null_after,
    )

    if null_after:
        fail(
            "Null values remain in model matrix."
        )

    # --------------------------------------------------------------
    # Final order
    # --------------------------------------------------------------

    X = X[
        feature_names
    ]

    if list(X.columns) != feature_names:
        fail(
            "Feature order mismatch."
        )

    if len(X.columns) != EXPECTED_FEATURE_COUNT:
        fail(
            f"Expected {EXPECTED_FEATURE_COUNT} "
            f"features, found {len(X.columns)}."
        )

    print(
        "MODEL MATRIX SHAPE:",
        X.shape,
    )

    print(
        "MODEL FEATURE COUNT:",
        len(X.columns),
    )

    print()
    print("FINAL DTYPES:")

    for column in X.columns:
        print(
            f"{column}: "
            f"{X[column].dtype}"
        )

    return X


# ======================================================================
# VALIDATE CATEGORICAL COMPATIBILITY
# ======================================================================

def validate_categories(
    X: pd.DataFrame,
    model: xgb.Booster,
) -> None:

    banner("CATEGORICAL SCHEMA VALIDATION")

    categorical_features = [
        feature
        for feature in EXPECTED_CATEGORICAL_FEATURES
        if feature in X.columns
    ]

    model_features = list(
        model.feature_names or []
    )

    if model_features:
        if model_features != list(
            X.columns
        ):
            fail(
                "Model feature order does not "
                "match inference matrix."
            )

    print(
        "MODEL FEATURES:",
        len(model_features),
    )

    print(
        "INFERENCE FEATURES:",
        len(X.columns),
    )

    for feature in categorical_features:

        if not pd.api.types.is_categorical_dtype(
            X[feature]
        ):
            fail(
                f"{feature} must have "
                "categorical dtype."
            )

        categories = (
            X[feature]
            .cat.categories
            .tolist()
        )

        print(
            f"{feature}: "
            f"{len(categories)} categories"
        )

    status(
        "CATEGORICAL TYPES",
        "PASS",
    )

    status(
        "FEATURE ORDER",
        "PASS",
    )


# ======================================================================
# RAW MODEL PREDICTION
# ======================================================================

def generate_raw_probability(
    model: xgb.Booster,
    X: pd.DataFrame,
) -> np.ndarray:

    banner("GENERATING RAW MODEL PROBABILITIES")

    try:
        dmatrix = xgb.DMatrix(
            X,
            enable_categorical=True,
        )

        probabilities = model.predict(
            dmatrix
        )

    except Exception as exc:

        print()
        print(
            "XGBOOST PREDICTION FAILED."
        )
        print(exc)

        raise RuntimeError(
            "Production inference failed. "
            "The model input schema is not "
            "compatible with the saved V4 model."
        ) from exc

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(probabilities) != len(X):
        fail(
            "Prediction row count does not "
            "match input row count."
        )

    if not np.isfinite(
        probabilities
    ).all():
        fail(
            "Model generated non-finite "
            "probabilities."
        )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    print(
        "RAW PROBABILITY: PASS"
    )

    print(
        "MIN:",
        float(probabilities.min()),
    )

    print(
        "MAX:",
        float(probabilities.max()),
    )

    print(
        "MEAN:",
        float(probabilities.mean()),
    )

    return probabilities


# ======================================================================
# SIGMOID CALIBRATION
# ======================================================================

def extract_sigmoid_parameters(
    summary: dict,
) -> tuple[float, float] | None:
    """
    Extract the fitted sklearn LogisticRegression sigmoid parameters.

    The V4 calibration code fits LogisticRegression directly on the
    raw XGBoost probability:

        sigmoid.fit(raw_probability.reshape(-1, 1), y)

    Therefore the fitted transformation is:

        calibrated_probability =
            sigmoid(a * raw_probability + b)

    where:
        a = sigmoid.coef_[0][0]
        b = sigmoid.intercept_[0]

    The function accepts several JSON layouts for backward compatibility.
    """

    candidates: list[dict] = []

    for key in (
        "sigmoid_parameters",
        "sigmoid_calibration",
        "sigmoid",
        "parameters",
    ):
        value = summary.get(key)
        if isinstance(value, dict):
            candidates.append(value)

    candidates.append(summary)

    for obj in candidates:
        if not isinstance(obj, dict):
            continue

        a = None
        b = None

        for key in (
            "a",
            "A",
            "sigmoid_a",
            "slope",
            "coef",
            "coefficient",
            "coefficient_",
        ):
            if key in obj:
                a = obj[key]
                break

        for key in (
            "b",
            "B",
            "sigmoid_b",
            "intercept",
            "intercept_",
        ):
            if key in obj:
                b = obj[key]
                break

        if a is None or b is None:
            continue

        try:
            a = float(a)
            b = float(b)
        except (TypeError, ValueError):
            continue

        if math.isfinite(a) and math.isfinite(b):
            return a, b

    return None


def apply_sigmoid_calibration(
    raw_probability: np.ndarray,
    calibration_summary: dict,
) -> np.ndarray:
    """
    Apply the exact V4 sigmoid calibration.

    IMPORTANT:
    calibrate_model_v4.py fits LogisticRegression directly on raw
    XGBoost probabilities, not on logit(raw_probability).

    Therefore:

        z = a * raw_probability + b
        p = 1 / (1 + exp(-z))

    This must match sklearn LogisticRegression.predict_proba().
    """

    banner("APPLYING SIGMOID CALIBRATION")

    params = extract_sigmoid_parameters(
        calibration_summary
    )

    if params is None:
        fail(
            "Sigmoid calibration parameters were not found in "
            "calibration_summary.json. Re-run calibrate_model_v4.py "
            "after adding sigmoid_parameters to the saved summary."
        )

    a, b = params

    print(
        "SIGMOID A:",
        a,
    )

    print(
        "SIGMOID B:",
        b,
    )

    # --------------------------------------------------------------
    # Exact sklearn LogisticRegression transformation
    #
    # Calibration training:
    #
    #   sigmoid.fit(
    #       raw_calibration.reshape(-1, 1),
    #       y_calibration,
    #   )
    #
    # Therefore sklearn computes:
    #
    #   z = a * raw_probability + b
    #   p = sigmoid(z)
    #
    # DO NOT convert raw_probability to logit first.
    # --------------------------------------------------------------

    raw_probability = np.asarray(
        raw_probability,
        dtype=float,
    )

    if not np.isfinite(raw_probability).all():
        fail(
            "Raw probabilities contain non-finite values "
            "before sigmoid calibration."
        )

    raw_probability = np.clip(
        raw_probability,
        0.0,
        1.0,
    )

    z = (
        a * raw_probability
        + b
    )

    # Numerically stable sigmoid.
    calibrated = np.empty_like(
        z,
        dtype=float,
    )

    positive = z >= 0

    calibrated[positive] = (
        1.0
        / (
            1.0
            + np.exp(
                -z[positive]
            )
        )
    )

    exp_z = np.exp(
        z[~positive]
    )

    calibrated[~positive] = (
        exp_z
        / (
            1.0
            + exp_z
        )
    )

    calibrated = np.asarray(
        calibrated,
        dtype=float,
    )

    calibrated = np.clip(
        calibrated,
        0.0,
        1.0,
    )

    if not np.isfinite(
        calibrated
    ).all():
        fail(
            "Sigmoid calibration generated "
            "non-finite probabilities."
        )

    print(
        "SIGMOID CALIBRATION: PASS"
    )

    print(
        "CALIBRATED MIN:",
        float(calibrated.min()),
    )

    print(
        "CALIBRATED MAX:",
        float(calibrated.max()),
    )

    print(
        "CALIBRATED MEAN:",
        float(calibrated.mean()),
    )

    return calibrated


def determine_calibrated_probability(
    raw_probability: np.ndarray,
    calibration_summary: dict,
) -> np.ndarray:

    """
    Apply sigmoid calibration.

    If the calibration summary explicitly states
    that sigmoid_probability is identical to the
    raw probability, preserve the raw probability.

    Otherwise use the stored sigmoid parameters.
    """

    selected = str(
        calibration_summary.get(
            "selected_calibration",
            calibration_summary.get(
                "selected",
                "",
            ),
        )
    ).lower()

    print(
        "SELECTED CALIBRATION:",
        selected,
    )

    # If summary contains explicit parameters,
    # use them.
    params = extract_sigmoid_parameters(
        calibration_summary
    )

    if params is not None:
        return apply_sigmoid_calibration(
            raw_probability,
            calibration_summary,
        )

    # If there are no parameters, do not invent
    # calibration coefficients.
    #
    # This is only acceptable if the summary says
    # sigmoid is equivalent to raw probability.
    identity_markers = [
        "raw_probability",
        "identity",
        "none",
        "no_calibration",
    ]

    if selected in identity_markers:
        print(
            "SIGMOID CALIBRATION: "
            "IDENTITY / RAW PROBABILITY"
        )

        return raw_probability.copy()

    fail(
        "Unable to reproduce sigmoid calibration. "
        "No calibration parameters were found."
    )


# ======================================================================
# RISK LEVEL
# ======================================================================

def risk_level(
    probability: float,
    threshold: float,
) -> str:

    """
    Operational risk classification.

    < 0.50 threshold:
        NORMAL

    >= threshold:
        ALERT

    Additional severity levels are based on
    probability relative to the policy threshold.
    """

    if probability >= threshold * 2.0:
        return "HIGH"

    if probability >= threshold:
        return "ALERT"

    if probability >= threshold * 0.5:
        return "WATCH"

    return "NORMAL"


# ======================================================================
# CREATE PRODUCTION PREDICTIONS
# ======================================================================

def create_predictions(
    df: pd.DataFrame,
    raw_probability: np.ndarray,
    sigmoid_probability: np.ndarray,
    threshold: float,
) -> pd.DataFrame:

    banner("CREATING PRODUCTION PREDICTIONS")

    result = pd.DataFrame()

    # --------------------------------------------------------------
    # Identity fields
    # --------------------------------------------------------------

    identity_columns = [
        "subdivision",
        "year",
        "month",
        "season",
    ]

    for column in identity_columns:
        if column in df.columns:
            result[column] = df[
                column
            ].astype(
                "string"
            )

    # --------------------------------------------------------------
    # Prediction fields
    # --------------------------------------------------------------

    result[
        "raw_probability"
    ] = raw_probability

    result[
        "sigmoid_probability"
    ] = sigmoid_probability

    result[
        "policy_threshold"
    ] = threshold

    result[
        "alert"
    ] = (
        sigmoid_probability
        >= threshold
    )

    result[
        "risk_level"
    ] = [
        risk_level(
            float(p),
            threshold,
        )
        for p in sigmoid_probability
    ]

    # --------------------------------------------------------------
    # Actual target
    #
    # Kept for offline validation when the
    # inference dataset contains the target.
    # In real production input this column may
    # be absent.
    # --------------------------------------------------------------

    if TARGET in df.columns:
        result[
            "actual"
        ] = df[TARGET].astype(
            "int64"
        )

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    timestamp = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    result[
        "model_version"
    ] = "bharat-earth-v4"

    result[
        "inference_timestamp_utc"
    ] = timestamp

    result[
        "probability_type"
    ] = EXPECTED_PROBABILITY_TYPE

    result[
        "target"
    ] = TARGET

    # --------------------------------------------------------------
    # Final validation
    # --------------------------------------------------------------

    if len(result) != len(df):
        fail(
            "Prediction output row count "
            "does not match input."
        )

    probability_columns = [
        "raw_probability",
        "sigmoid_probability",
        "policy_threshold",
    ]

    for column in probability_columns:

        values = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        if values.isna().any():
            fail(
                f"Invalid values in {column}."
            )

        if not values.between(
            0.0,
            1.0,
        ).all():
            fail(
                f"Probability outside "
                f"[0,1] in {column}."
            )

    expected_alert = (
        result[
            "sigmoid_probability"
        ]
        >= threshold
    )

    if not (
        expected_alert
        == result["alert"]
    ).all():
        fail(
            "Alert decisions do not match "
            "the production threshold."
        )

    status(
        "PREDICTION OUTPUT",
        "PASS",
        f"rows={len(result)}",
    )

    return result


# ======================================================================
# VALIDATE PREDICTIONS
# ======================================================================

def validate_predictions(
    predictions: pd.DataFrame,
    threshold: float,
) -> dict:

    banner("PRODUCTION PREDICTION VALIDATION")

    required = [
        "raw_probability",
        "sigmoid_probability",
        "policy_threshold",
        "alert",
        "risk_level",
        "model_version",
        "inference_timestamp_utc",
        "probability_type",
        "target",
    ]

    missing = [
        column
        for column in required
        if column not in predictions.columns
    ]

    if missing:
        fail(
            f"Missing prediction columns: "
            f"{missing}"
        )

    if len(predictions) == 0:
        fail(
            "No prediction rows generated."
        )

    # --------------------------------------------------------------
    # Probability validation
    # --------------------------------------------------------------

    for column in [
        "raw_probability",
        "sigmoid_probability",
    ]:

        values = pd.to_numeric(
            predictions[column],
            errors="coerce",
        )

        if values.isna().any():
            fail(
                f"NaN values found in {column}."
            )

        if not values.between(
            0,
            1,
        ).all():
            fail(
                f"Invalid probability range "
                f"in {column}."
            )

    # --------------------------------------------------------------
    # Threshold validation
    # --------------------------------------------------------------

    policy_values = pd.to_numeric(
        predictions[
            "policy_threshold"
        ],
        errors="coerce",
    )

    if not (
        np.isclose(
            policy_values,
            threshold,
        ).all()
    ):
        fail(
            "Prediction threshold does not "
            "match selected production policy."
        )

    # --------------------------------------------------------------
    # Alert validation
    # --------------------------------------------------------------

    expected_alert = (
        predictions[
            "sigmoid_probability"
        ]
        >= threshold
    )

    actual_alert = (
        predictions["alert"]
        .astype(bool)
    )

    if not (
        expected_alert
        == actual_alert
    ).all():
        fail(
            "Alert decisions are inconsistent "
            "with the selected policy."
        )

    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------

    observations = len(
        predictions
    )

    alerts = int(
        predictions["alert"]
        .sum()
    )

    alert_rate = (
        alerts / observations
        if observations
        else 0.0
    )

    summary = {
        "observations": observations,
        "alerts": alerts,
        "alert_rate": alert_rate,
        "mean_raw_probability": float(
            predictions[
                "raw_probability"
            ].mean()
        ),
        "mean_sigmoid_probability": float(
            predictions[
                "sigmoid_probability"
            ].mean()
        ),
        "maximum_sigmoid_probability": float(
            predictions[
                "sigmoid_probability"
            ].max()
        ),
        "minimum_sigmoid_probability": float(
            predictions[
                "sigmoid_probability"
            ].min()
        ),
        "threshold": threshold,
    }

    status(
        "PROBABILITY RANGE",
        "PASS",
    )

    status(
        "ALERT POLICY",
        "PASS",
        f"alerts={alerts}, "
        f"rate={alert_rate:.6f}",
    )

    # --------------------------------------------------------------
    # Offline metrics if actual target exists
    # --------------------------------------------------------------

    if "actual" in predictions.columns:

        actual = pd.to_numeric(
            predictions["actual"],
            errors="coerce",
        ).astype(int)

        predicted = (
            predictions["alert"]
            .astype(int)
        )

        tp = int(
            (
                (actual == 1)
                & (predicted == 1)
            ).sum()
        )

        tn = int(
            (
                (actual == 0)
                & (predicted == 0)
            ).sum()
        )

        fp = int(
            (
                (actual == 0)
                & (predicted == 1)
            ).sum()
        )

        fn = int(
            (
                (actual == 1)
                & (predicted == 0)
            ).sum()
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

        summary.update(
            {
                "actual_events": int(
                    actual.sum()
                ),
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

        print()
        print(
            "OFFLINE VALIDATION"
        )

        print(
            "TRUE POSITIVES:",
            tp,
        )

        print(
            "TRUE NEGATIVES:",
            tn,
        )

        print(
            "FALSE POSITIVES:",
            fp,
        )

        print(
            "FALSE NEGATIVES:",
            fn,
        )

        print(
            "PRECISION:",
            f"{precision:.6f}",
        )

        print(
            "RECALL:",
            f"{recall:.6f}",
        )

        print(
            "F1:",
            f"{f1:.6f}",
        )

    return summary


# ======================================================================
# SAVE OUTPUTS
# ======================================================================

def save_outputs(
    predictions: pd.DataFrame,
    summary: dict,
    calibration_summary: dict,
    threshold: float,
    probability_type: str,
) -> None:

    banner("SAVING PRODUCTION INFERENCE OUTPUT")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    if not PREDICTIONS_FILE.exists():
        fail(
            "Prediction CSV was not created."
        )

    status(
        "PREDICTIONS CSV",
        "PASS",
        str(PREDICTIONS_FILE),
    )

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    metrics = {
        "model_version": "bharat-earth-v4",
        "target": TARGET,
        "probability_type": probability_type,
        "threshold": threshold,
        "calibration_method": "sigmoid",
        **summary,
    }

    metrics_df = pd.DataFrame(
        [json_safe(metrics)]
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False,
    )

    status(
        "INFERENCE METRICS",
        "PASS",
        str(METRICS_FILE),
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    final_summary = {
        "project": "Bharat Earth",
        "model_version": "v4",
        "target": TARGET,
        "model_file": str(
            MODEL_FILE
        ),
        "schema_file": str(
            SCHEMA_FILE
        ),
        "calibration_summary": str(
            CALIBRATION_SUMMARY_FILE
        ),
        "policy_file": str(
            POLICY_FILE
        ),
        "probability_type": probability_type,
        "threshold": threshold,
        "calibration_method": "sigmoid",
        "production_status": "PASS",
        "summary": summary,
    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_safe(final_summary),
            f,
            indent=2,
        )

    status(
        "INFERENCE SUMMARY",
        "PASS",
        str(SUMMARY_FILE),
    )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    banner(
        "12. BHARAT EARTH V4 "
        "PRODUCTION INFERENCE"
    )

    print(
        "PROJECT ROOT:",
        PROJECT_ROOT,
    )

    # --------------------------------------------------------------
    # Load model first so model feature names
    # can be used to validate schema.
    # --------------------------------------------------------------

    model = load_model()

    schema = load_schema()

    calibration_summary = (
        load_calibration_summary()
    )

    threshold, probability_type = (
        load_policy()
    )

    validate_policy(
        threshold,
        probability_type,
    )

    # --------------------------------------------------------------
    # Extract schema
    # --------------------------------------------------------------

    feature_names, categories = (
        extract_feature_schema(
            schema,
            model,
        )
    )

    # If schema does not explicitly provide
    # categories, use the known V4 categorical
    # schema from the validated model.
    #
    # These categories are taken from the V4
    # training schema shown in the previous
    # validation steps.
    # --------------------------------------------------------------

    if "subdivision" not in categories:

        categories[
            "subdivision"
        ] = [
            "Andaman & Nicobar Islands",
            "Arunachal Pradesh",
            "Assam & Meghalaya",
            "Bihar",
            "Chhattisgarh",
            "Coastal Andhra Pradesh",
            "Coastal Karnataka",
            "East Madhya Pradesh",
            "East Rajasthan",
            "East Uttar Pradesh",
            "Gangetic West Bengal",
            "Gujarat Region",
            "Haryana Delhi & Chandigarh",
            "Himachal Pradesh",
            "Jammu & Kashmir",
            "Jharkhand",
            "Kerala",
            "Konkan & Goa",
            "Lakshadweep",
            "Madhya Maharashtra",
            "Matathwada",
            "Naga Mani Mizo Tripura",
            "North Interior Karnataka",
            "Orissa",
            "Punjab",
            "Rayalseema",
            "Saurashtra & Kutch",
            "South Interior Karnataka",
            "Sub Himalayan West Bengal & Sikkim",
            "Tamil Nadu",
            "Telangana",
            "Uttarakhand",
            "Vidarbha",
            "West Madhya Pradesh",
            "West Rajasthan",
            "West Uttar Pradesh",
        ]

    if "month" not in categories:
        categories[
            "month"
        ] = [
            str(i)
            for i in range(1, 13)
        ]

    if "season" not in categories:
        categories[
            "season"
        ] = [
            "MONSOON",
            "POST_MONSOON",
            "PRE_MONSOON",
            "WINTER",
        ]

    # --------------------------------------------------------------
    # Dataset
    # --------------------------------------------------------------

    df = load_dataset()

    df = validate_dataset(
        df,
        feature_names,
    )

    # --------------------------------------------------------------
    # Model matrix
    # --------------------------------------------------------------

    X = build_model_matrix(
        df,
        feature_names,
        categories,
    )

    validate_categories(
        X,
        model,
    )

    # --------------------------------------------------------------
    # Prediction
    # --------------------------------------------------------------

    raw_probability = (
        generate_raw_probability(
            model,
            X,
        )
    )

    # --------------------------------------------------------------
    # Calibration
    # --------------------------------------------------------------

    sigmoid_probability = (
        determine_calibrated_probability(
            raw_probability,
            calibration_summary,
        )
    )

    # --------------------------------------------------------------
    # Production output
    # --------------------------------------------------------------

    predictions = create_predictions(
        df,
        raw_probability,
        sigmoid_probability,
        threshold,
    )

    summary = validate_predictions(
        predictions,
        threshold,
    )

    save_outputs(
        predictions,
        summary,
        calibration_summary,
        threshold,
        probability_type,
    )

    # --------------------------------------------------------------
    # Final
    # --------------------------------------------------------------

    banner(
        "12. PRODUCTION INFERENCE V4 COMPLETE"
    )

    print(
        "STATUS: PASS"
    )

    print(
        "MODEL: XGBoost V4"
    )

    print(
        "CALIBRATION:",
        "sigmoid",
    )

    print(
        "POLICY:",
        f"sigmoid_probability >= "
        f"{threshold:.6f}",
    )

    print(
        "OBSERVATIONS:",
        len(predictions),
    )

    print(
        "ALERTS:",
        int(
            predictions["alert"].sum()
        ),
    )

    print(
        "OUTPUT DIRECTORY:"
    )

    print(
        OUTPUT_DIR
    )


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print(
            "Process interrupted by user."
        )
        sys.exit(130)

    except Exception as exc:
        print()
        print("=" * 70)
        print("PRODUCTION INFERENCE FAILED")
        print("=" * 70)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        sys.exit(1)