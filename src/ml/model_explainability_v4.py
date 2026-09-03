from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import average_precision_score
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings("ignore")


# ======================================================================
# PROJECT PATHS
# ======================================================================

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
    / "explainability_v4"
)

FEATURE_IMPORTANCE_FILE = (
    OUTPUT_DIR / "feature_importance.csv"
)

PERMUTATION_IMPORTANCE_FILE = (
    OUTPUT_DIR / "permutation_importance.csv"
)

SHAP_IMPORTANCE_FILE = (
    OUTPUT_DIR / "shap_importance.csv"
)

EXPLAINABILITY_SUMMARY_FILE = (
    OUTPUT_DIR / "explainability_summary.json"
)

HIGH_RISK_FILE = (
    OUTPUT_DIR / "high_risk_records.csv"
)

TARGET = "target_3m_severe_anomaly"

LEGACY_LEAKAGE_COLUMNS = {
    "target_3m_stress",
    "rainfall_stress",
    "persistent_drought_signal",
    "environmental_risk_score",
    "environmental_risk_level",
}


# ======================================================================
# EXPECTED V4 CATEGORIES
# ======================================================================

EXPECTED_MONTH_CATEGORIES = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "11",
    "12",
]

EXPECTED_SEASON_CATEGORIES = [
    "MONSOON",
    "POST_MONSOON",
    "PRE_MONSOON",
    "WINTER",
]

MONTH_NAME_MAP = {
    "JAN": "1",
    "FEB": "2",
    "MAR": "3",
    "APR": "4",
    "MAY": "5",
    "JUN": "6",
    "JUL": "7",
    "AUG": "8",
    "SEP": "9",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}


# ======================================================================
# HELPERS
# ======================================================================

def banner(text: str) -> None:
    print()
    print("=" * 70)
    print(text)
    print("=" * 70)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_string_series(series: pd.Series) -> pd.Series:
    return (
        series
        .astype("string")
        .str.strip()
        .str.upper()
    )


# ======================================================================
# LOAD MODEL
# ======================================================================

def load_model() -> xgb.XGBClassifier:

    banner("LOADING XGBOOST V4 MODEL")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    print("MODEL:")
    print(MODEL_FILE)

    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_FILE))

    print("MODEL LOAD: PASS")

    return model


# ======================================================================
# READ MODEL SCHEMA
# ======================================================================

def read_model_schema(model: xgb.XGBClassifier) -> dict:

    banner("READING TRAINED MODEL FEATURE TYPES")

    booster = model.get_booster()

    feature_names = list(
        booster.feature_names
    )

    if not feature_names:
        raise ValueError(
            "Saved XGBoost model contains no feature names."
        )

    raw_types = list(
        booster.feature_types
    )

    if len(raw_types) != len(feature_names):
        raise ValueError(
            "Model feature names and feature types "
            "have different lengths."
        )

    feature_types = dict(
        zip(
            feature_names,
            raw_types,
        )
    )

    print(
        "MODEL FEATURE COUNT:",
        len(feature_names),
    )

    print()
    print("TRAINED MODEL FEATURE TYPES:")

    for name in feature_names:
        print(
            f"{name}: {feature_types[name]}"
        )

    categorical_features = [
        name
        for name in feature_names
        if feature_types[name] == "c"
    ]

    numeric_features = [
        name
        for name in feature_names
        if feature_types[name] != "c"
    ]

    print()
    print(
        "CATEGORICAL FEATURES DETECTED:"
    )
    print(categorical_features)

    print()
    print(
        "NUMERIC FEATURES DETECTED:"
    )
    print(numeric_features)

    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"Model schema not found:\n{SCHEMA_FILE}"
        )

    schema = load_json(
        SCHEMA_FILE
    )

    print()
    print("SCHEMA FILE:")
    print(SCHEMA_FILE)
    print("SCHEMA LOAD: PASS")

    schema["_runtime_feature_names"] = (
        feature_names
    )

    schema["_runtime_feature_types"] = (
        feature_types
    )

    schema["_runtime_categorical_features"] = (
        categorical_features
    )

    schema["_runtime_numeric_features"] = (
        numeric_features
    )

    return schema


# ======================================================================
# LOAD DATA
# ======================================================================

def load_dataset() -> pd.DataFrame:

    banner("LOADING V4 DATASET")

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print("DATA FILE:")
    print(DATA_FILE)

    print(
        "SHAPE:",
        df.shape,
    )

    print("COLUMNS:")
    print(list(df.columns))

    return df


# ======================================================================
# SEASON LOGIC
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

    result.loc[
        month_num.isin([12, 1, 2])
    ] = "WINTER"

    result.loc[
        month_num.isin([3, 4, 5])
    ] = "PRE_MONSOON"

    result.loc[
        month_num.isin([6, 7, 8, 9])
    ] = "MONSOON"

    result.loc[
        month_num.isin([10, 11])
    ] = "POST_MONSOON"

    return result


# ======================================================================
# MONTH NORMALIZATION
# ======================================================================

def normalize_month(
    series: pd.Series,
) -> pd.Series:

    raw = normalize_string_series(
        series
    )

    # --------------------------------------------------------------
    # Month names
    # --------------------------------------------------------------

    mapped = raw.map(
        MONTH_NAME_MAP
    )

    # --------------------------------------------------------------
    # Numeric values
    #
    # Supports:
    # 1
    # 1.0
    # "1"
    # "1.0"
    # --------------------------------------------------------------

    numeric = pd.to_numeric(
        raw,
        errors="coerce",
    )

    numeric_strings = (
        numeric
        .where(
            numeric.notna(),
            np.nan,
        )
    )

    numeric_strings = (
        numeric_strings
        .round()
        .astype("Int64")
        .astype("string")
    )

    # --------------------------------------------------------------
    # Prefer month names.
    # Otherwise use numeric representation.
    # --------------------------------------------------------------

    normalized = mapped.where(
        mapped.notna(),
        numeric_strings,
    )

    valid = set(
        EXPECTED_MONTH_CATEGORIES
    )

    invalid_mask = (
        normalized.isna()
        | ~normalized.isin(valid)
    )

    invalid_count = int(
        invalid_mask.sum()
    )

    print(
        "INVALID MONTHS:",
        invalid_count,
    )

    if invalid_count:

        print(
            "INVALID MONTH VALUES:"
        )

        print(
            raw[
                invalid_mask
            ]
            .value_counts(
                dropna=False
            )
            .head(30)
        )

        raise ValueError(
            "Invalid month values found."
        )

    return normalized


# ======================================================================
# DATASET VALIDATION
# ======================================================================

def validate_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

    banner("DATASET VALIDATION")

    # ==============================================================
    # LEAKAGE
    # ==============================================================

    leakage = sorted(
        LEGACY_LEAKAGE_COLUMNS
        & set(df.columns)
    )

    print(
        "LEGACY LEAKAGE COLUMNS:"
    )
    print(leakage)

    if leakage:
        raise ValueError(
            f"Legacy leakage columns found: "
            f"{leakage}"
        )

    # ==============================================================
    # TARGET
    # ==============================================================

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column not found: {TARGET}"
        )

    target_numeric = pd.to_numeric(
        df[TARGET],
        errors="coerce",
    )

    if target_numeric.isna().any():
        raise ValueError(
            "Target contains NULL or "
            "non-numeric values."
        )

    df[TARGET] = (
        target_numeric
        .astype("int64")
    )

    print(
        "TARGET DISTRIBUTION:"
    )
    print(
        df[TARGET].value_counts()
    )

    print(
        "TARGET RATE:",
        df[TARGET].mean(),
    )

    target_values = sorted(
        df[TARGET]
        .unique()
        .tolist()
    )

    print(
        "TARGET VALUES:",
        target_values,
    )

    if set(target_values) != {0, 1}:
        raise ValueError(
            "Target must contain both 0 and 1."
        )

    # ==============================================================
    # DUPLICATES
    # ==============================================================

    duplicates = int(
        df.duplicated().sum()
    )

    print(
        "EXACT DUPLICATES:",
        duplicates,
    )

    if duplicates:
        raise ValueError(
            f"Exact duplicate rows found: "
            f"{duplicates}"
        )

    # ==============================================================
    # MONTH
    # ==============================================================

    print()
    print(
        "MONTH NORMALIZATION"
    )

    df["month"] = normalize_month(
        df["month"]
    )

    print(
        "MONTH RANGE:",
        df["month"].min(),
        "-",
        df["month"].max(),
    )

    print(
        "MONTH VALIDATION: PASS"
    )

    # ==============================================================
    # SEASON
    # ==============================================================

    expected_season = (
        season_from_month(
            df["month"]
        )
    )

    actual_season = (
        normalize_string_series(
            df["season"]
        )
    )

    inconsistent = (
        actual_season
        != expected_season
    )

    inconsistency_count = int(
        inconsistent.sum()
    )

    print(
        "SEASON INCONSISTENCIES:",
        inconsistency_count,
    )

    if inconsistency_count:
        print(
            "Repairing season from month."
        )

    df["season"] = (
        expected_season
    )

    valid_seasons = set(
        EXPECTED_SEASON_CATEGORIES
    )

    invalid_season = (
        df["season"].isna()
        | ~df["season"].isin(
            valid_seasons
        )
    )

    if invalid_season.any():
        raise ValueError(
            "Invalid season values found."
        )

    print(
        "SEASON VALIDATION: PASS"
    )

    # ==============================================================
    # YEAR
    # ==============================================================

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    if df["year"].isna().any():
        raise ValueError(
            "Invalid year values found."
        )

    df["year"] = (
        df["year"]
        .astype("int64")
    )

    # ==============================================================
    # SUBDIVISION
    # ==============================================================

    if "subdivision" not in df.columns:
        raise ValueError(
            "subdivision column not found."
        )

    if df["subdivision"].isna().any():
        raise ValueError(
            "NULL subdivision values found."
        )

    df["subdivision"] = (
        df["subdivision"]
        .astype("string")
        .str.strip()
    )

    print(
        "DATASET VALIDATION: PASS"
    )

    return df


# ======================================================================
# EXTRACT CATEGORIES FROM SCHEMA
# ======================================================================

def _recursive_find_categories(
    obj,
    column: str,
):

    # --------------------------------------------------------------
    # Direct dictionary
    # --------------------------------------------------------------

    if isinstance(obj, dict):

        # Exact column key
        if column in obj:

            value = obj[column]

            if isinstance(
                value,
                (list, tuple),
            ):
                return list(value)

            if isinstance(
                value,
                dict,
            ):

                for key in [
                    "categories",
                    "values",
                    "category_values",
                    "levels",
                    "unique_values",
                ]:

                    candidate = value.get(
                        key
                    )

                    if isinstance(
                        candidate,
                        (list, tuple),
                    ):
                        return list(
                            candidate
                        )

        # Search known category containers
        for key in [
            "categories",
            "categorical_categories",
            "category_values",
            "feature_categories",
            "categorical_values",
            "category_schema",
            "features",
            "feature_schema",
            "columns",
        ]:

            if key in obj:

                result = (
                    _recursive_find_categories(
                        obj[key],
                        column,
                    )
                )

                if result is not None:
                    return result

        # Generic recursive search
        for value in obj.values():

            if isinstance(
                value,
                (dict, list),
            ):

                result = (
                    _recursive_find_categories(
                        value,
                        column,
                    )
                )

                if result is not None:
                    return result

    # --------------------------------------------------------------
    # Lists
    # --------------------------------------------------------------

    elif isinstance(
        obj,
        list,
    ):

        for item in obj:

            if isinstance(
                item,
                (dict, list),
            ):

                result = (
                    _recursive_find_categories(
                        item,
                        column,
                    )
                )

                if result is not None:
                    return result

    return None


def get_schema_categories(
    schema: dict,
    column: str,
    df: pd.DataFrame,
) -> list:

    categories = (
        _recursive_find_categories(
            schema,
            column,
        )
    )

    # ==============================================================
    # MONTH
    # ==============================================================

    if column == "month":

        categories = (
            EXPECTED_MONTH_CATEGORIES
        )

    # ==============================================================
    # SEASON
    # ==============================================================

    elif column == "season":

        categories = (
            EXPECTED_SEASON_CATEGORIES
        )

    # ==============================================================
    # SUBDIVISION
    #
    # Prefer saved schema.
    # If the schema does not contain subdivision categories,
    # use deterministic sorted dataset categories.
    # ==============================================================

    elif column == "subdivision":

        if categories is None:

            categories = sorted(
                df["subdivision"]
                .astype(str)
                .unique()
                .tolist()
            )

    if categories is None:
        raise ValueError(
            f"Cannot determine categories "
            f"for categorical feature: {column}"
        )

    categories = [
        str(x).strip()
        for x in categories
    ]

    # Remove duplicates while preserving order.
    categories = list(
        dict.fromkeys(categories)
    )

    return categories


# ======================================================================
# BUILD EXACT MODEL MATRIX
# ======================================================================

def build_model_matrix(
    df: pd.DataFrame,
    model: xgb.XGBClassifier,
    schema: dict,
) -> pd.DataFrame:

    banner(
        "BUILDING MODEL MATRIX FROM TRAINED MODEL SCHEMA"
    )

    booster = (
        model.get_booster()
    )

    feature_names = list(
        booster.feature_names
    )

    feature_types = dict(
        zip(
            feature_names,
            booster.feature_types,
        )
    )

    # ==============================================================
    # FEATURE EXISTENCE
    # ==============================================================

    missing = [
        col
        for col in feature_names
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    # ==============================================================
    # EXACT FEATURE ORDER
    # ==============================================================

    X = df[
        feature_names
    ].copy()

    # ==============================================================
    # CATEGORICAL FEATURES
    # ==============================================================

    categorical_features = [
        col
        for col in feature_names
        if feature_types[col] == "c"
    ]

    for column in categorical_features:

        categories = (
            get_schema_categories(
                schema,
                column,
                df,
            )
        )

        # ----------------------------------------------------------
        # MONTH
        # ----------------------------------------------------------

        if column == "month":

            values = (
                normalize_month(
                    X[column]
                )
            )

        # ----------------------------------------------------------
        # Other categorical columns
        # ----------------------------------------------------------

        else:

            values = (
                X[column]
                .astype("string")
                .str.strip()
            )

            if column == "season":
                values = (
                    values.str.upper()
                )

        # ----------------------------------------------------------
        # Validate category membership BEFORE creating
        # pandas Categorical.
        # ----------------------------------------------------------

        unknown_mask = (
            values.isna()
            | ~values.isin(
                categories
            )
        )

        if unknown_mask.any():

            bad_values = (
                values[
                    unknown_mask
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Unknown categorical values "
                f"in {column}: "
                f"{bad_values[:20]}"
            )

        # ----------------------------------------------------------
        # IMPORTANT:
        #
        # XGBoost categorical input requires:
        #
        # dtype = category
        #
        # and identical category labels/order.
        # ----------------------------------------------------------

        X[column] = pd.Series(
            pd.Categorical(
                values,
                categories=categories,
                ordered=False,
            ),
            index=X.index,
        )

    # ==============================================================
    # NUMERIC FEATURES
    # ==============================================================

    numeric_features = [
        col
        for col in feature_names
        if feature_types[col] != "c"
    ]

    for column in numeric_features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

        if X[column].isna().any():

            if column == "rainfall_missing":
                X[column] = (
                    X[column]
                    .fillna(0)
                )

            else:

                median = (
                    X[column]
                    .median()
                )

                if pd.isna(median):
                    raise ValueError(
                        f"Cannot impute numeric "
                        f"feature: {column}"
                    )

                X[column] = (
                    X[column]
                    .fillna(median)
                )

        if column == "year":

            X[column] = (
                X[column]
                .astype("int64")
            )

        elif column == "rainfall_missing":

            X[column] = (
                X[column]
                .astype("int64")
            )

        else:

            X[column] = (
                X[column]
                .astype("float64")
            )

    # ==============================================================
    # FINAL UNKNOWN CATEGORICAL CHECK
    # ==============================================================

    for column in categorical_features:

        if X[column].isna().any():

            raise ValueError(
                f"Categorical feature "
                f"{column} contains unknown "
                f"or missing values."
            )

    # ==============================================================
    # FINAL NULL CHECK
    # ==============================================================

    null_count = int(
        X.isna().sum().sum()
    )

    print(
        "MISSING VALUES AFTER:",
        null_count,
    )

    if null_count:
        raise ValueError(
            "NULL values remain in model matrix."
        )

    # ==============================================================
    # OUTPUT
    # ==============================================================

    print(
        "MODEL MATRIX SHAPE:",
        X.shape,
    )

    print(
        "MODEL FEATURE COUNT:",
        len(X.columns),
    )

    print()
    print(
        "FINAL MODEL DTYPES:"
    )

    for column in X.columns:

        print(
            f"{column}: {X[column].dtype}"
        )

    print()
    print(
        "CATEGORICAL VALUES:"
    )

    for column in categorical_features:

        print(
            f"{column}: "
            f"{list(X[column].cat.categories)}"
        )

    return X


# ======================================================================
# MODEL MATRIX VALIDATION
# ======================================================================

def validate_model_matrix(
    X: pd.DataFrame,
    model: xgb.XGBClassifier,
) -> None:

    banner(
        "MODEL MATRIX VALIDATION"
    )

    booster = (
        model.get_booster()
    )

    model_features = list(
        booster.feature_names
    )

    data_features = list(
        X.columns
    )

    # ==============================================================
    # FEATURE NAMES
    # ==============================================================

    if model_features == data_features:

        print(
            "FEATURE NAMES: PASS"
        )

    else:

        print(
            "FEATURE NAMES: FAIL"
        )

        print(
            "MODEL:",
            model_features,
        )

        print(
            "DATA:",
            data_features,
        )

        raise ValueError(
            "Feature order does not match model."
        )

    # ==============================================================
    # TYPES
    # ==============================================================

    model_types = dict(
        zip(
            model_features,
            booster.feature_types,
        )
    )

    for column in model_features:

        expected = (
            model_types[column]
        )

        actual = X[column].dtype

        if expected == "c":

            if not isinstance(
                actual,
                pd.CategoricalDtype,
            ):
                raise ValueError(
                    f"{column} must be categorical."
                )

        elif expected == "int":

            if not pd.api.types.is_integer_dtype(
                actual
            ):
                raise ValueError(
                    f"{column} must be integer."
                )

        elif expected == "float":

            if not pd.api.types.is_float_dtype(
                actual
            ):
                raise ValueError(
                    f"{column} must be float."
                )

    print(
        "FEATURE ORDER: PASS"
    )

    print(
        "FEATURE DTYPES: PASS"
    )


# ======================================================================
# XGBOOST PREDICTION
# ======================================================================

def predict_probability(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> np.ndarray:

    booster = (
        model.get_booster()
    )

    try:

        dmatrix = xgb.DMatrix(
            X,
            enable_categorical=True,
        )

        probability = (
            booster.predict(
                dmatrix
            )
        )

    except Exception as exc:

        print()
        print(
            "XGBOOST PREDICTION FAILED."
        )

        print(
            str(exc)
        )

        raise RuntimeError(
            "XGBoost prediction failed. "
            "The model matrix does not exactly "
            "match the saved categorical schema."
        ) from exc

    probability = np.asarray(
        probability,
        dtype=float,
    )

    probability = np.clip(
        probability,
        0.0,
        1.0,
    )

    return probability


# ======================================================================
# COMPATIBILITY TEST
# ======================================================================

def compatibility_test(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> None:

    banner(
        "MODEL PREDICTION COMPATIBILITY TEST"
    )

    sample = (
        X.iloc[:10]
        .copy()
    )

    probability = (
        predict_probability(
            model,
            sample,
        )
    )

    print(
        "PREDICTION: PASS"
    )

    print(
        "PROBABILITIES:"
    )

    print(
        probability
    )


# ======================================================================
# NATIVE XGBOOST FEATURE IMPORTANCE
# ======================================================================

def xgboost_feature_importance(
    model: xgb.XGBClassifier,
) -> pd.DataFrame:

    banner(
        "XGBOOST FEATURE IMPORTANCE"
    )

    booster = (
        model.get_booster()
    )

    names = list(
        booster.feature_names
    )

    importance = (
        booster.get_score(
            importance_type="gain"
        )
    )

    rows = []

    for feature in names:

        rows.append(
            {
                "feature": feature,
                "importance": float(
                    importance.get(
                        feature,
                        0.0,
                    )
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    total = (
        result["importance"]
        .sum()
    )

    if total > 0:

        result["importance"] = (
            result["importance"]
            / total
        )

    result = (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    result["rank"] = (
        np.arange(
            len(result)
        )
        + 1
    )

    print()
    print("TOP 20:")
    print(
        result.head(20)
        .to_string(
            index=False
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )

    return result


# ======================================================================
# SKLEARN COMPATIBLE WRAPPER
# ======================================================================

class XGBCategoricalWrapper(
    BaseEstimator,
    ClassifierMixin,
):

    def __init__(
        self,
        model,
    ):

        self.model = model

        self.classes_ = np.array(
            [0, 1]
        )

    def fit(
        self,
        X,
        y=None,
    ):

        return self

    def predict_proba(
        self,
        X,
    ):

        probability = (
            predict_probability(
                self.model,
                X,
            )
        )

        return np.column_stack(
            [
                1.0 - probability,
                probability,
            ]
        )

    def predict(
        self,
        X,
    ):

        probability = (
            self.predict_proba(X)
            [:, 1]
        )

        return (
            probability >= 0.5
        ).astype(int)


# ======================================================================
# PERMUTATION IMPORTANCE
# ======================================================================

def permutation_importance_v4(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:

    banner(
        "PERMUTATION IMPORTANCE"
    )

    sample_size = min(
        3000,
        len(X),
    )

    rng = np.random.RandomState(
        42
    )

    indices = rng.choice(
        len(X),
        size=sample_size,
        replace=False,
    )

    X_sample = (
        X.iloc[
            indices
        ]
        .copy()
    )

    y_sample = (
        y.iloc[
            indices
        ]
        .copy()
    )

    print(
        "SAMPLE SIZE:",
        len(X_sample),
    )

    baseline_probability = (
        predict_probability(
            model,
            X_sample,
        )
    )

    baseline_score = (
        average_precision_score(
            y_sample,
            baseline_probability,
        )
    )

    print(
        "BASELINE PR-AUC:",
        f"{baseline_score:.6f}",
    )

    rows = []

    for feature in X_sample.columns:

        scores = []

        for repeat in range(3):

            permuted = (
                X_sample.copy()
            )

            values = (
                permuted[
                    feature
                ].copy()
            )

            # ------------------------------------------------------
            # Preserve categorical metadata.
            # ------------------------------------------------------

            if isinstance(
                values.dtype,
                pd.CategoricalDtype,
            ):

                shuffled_codes = (
                    values
                    .cat.codes
                    .to_numpy()
                    .copy()
                )

                rng.shuffle(
                    shuffled_codes
                )

                shuffled = (
                    pd.Categorical.from_codes(
                        shuffled_codes,
                        categories=(
                            values
                            .cat.categories
                        ),
                        ordered=(
                            values
                            .cat.ordered
                        ),
                    )
                )

                permuted[
                    feature
                ] = pd.Series(
                    shuffled,
                    index=permuted.index,
                )

            else:

                shuffled = (
                    values
                    .to_numpy()
                    .copy()
                )

                rng.shuffle(
                    shuffled
                )

                permuted[
                    feature
                ] = shuffled

            probability = (
                predict_probability(
                    model,
                    permuted,
                )
            )

            score = (
                average_precision_score(
                    y_sample,
                    probability,
                )
            )

            scores.append(
                baseline_score
                - score
            )

        rows.append(
            {
                "feature": feature,
                "importance_mean": float(
                    np.mean(scores)
                ),
                "importance_std": float(
                    np.std(scores)
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result = (
        result
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    result["rank"] = (
        np.arange(
            len(result)
        )
        + 1
    )

    print()
    print(
        result.head(20)
        .to_string(
            index=False
        )
    )

    result.to_csv(
        PERMUTATION_IMPORTANCE_FILE,
        index=False,
    )

    return result


# ======================================================================
# SHAP IMPORTANCE
# ======================================================================

def shap_importance_v4(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
) -> pd.DataFrame:

    banner(
        "SHAP IMPORTANCE"
    )

    try:

        import shap

    except ImportError:

        print(
            "SHAP NOT INSTALLED."
        )

        print(
            "Saving empty SHAP result."
        )

        result = pd.DataFrame(
            columns=[
                "feature",
                "mean_abs_shap",
                "rank",
            ]
        )

        result.to_csv(
            SHAP_IMPORTANCE_FILE,
            index=False,
        )

        return result

    sample_size = min(
        3000,
        len(X),
    )

    sample = (
        X.iloc[
            :sample_size
        ]
        .copy()
    )

    try:

        # ----------------------------------------------------------
        # Native booster is more reliable for this categorical
        # XGBoost model than sklearn-level SHAP prediction.
        # ----------------------------------------------------------

        booster = (
            model.get_booster()
        )

        explainer = (
            shap.TreeExplainer(
                booster
            )
        )

        shap_values = (
            explainer.shap_values(
                sample
            )
        )

        if isinstance(
            shap_values,
            list,
        ):

            shap_values = (
                shap_values[-1]
            )

        shap_values = np.asarray(
            shap_values
        )

        # Some SHAP versions return
        # (rows, features, classes).
        if (
            shap_values.ndim == 3
        ):

            shap_values = (
                shap_values[:, :, -1]
            )

        if (
            shap_values.ndim != 2
        ):

            raise ValueError(
                "Unexpected SHAP output shape: "
                f"{shap_values.shape}"
            )

        if (
            shap_values.shape[1]
            != len(X.columns)
        ):

            raise ValueError(
                "SHAP feature count does not "
                "match model feature count."
            )

        mean_abs = (
            np.mean(
                np.abs(
                    shap_values
                ),
                axis=0,
            )
        )

        result = pd.DataFrame(
            {
                "feature": X.columns,
                "mean_abs_shap": mean_abs,
            }
        )

        result = (
            result
            .sort_values(
                "mean_abs_shap",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        result["rank"] = (
            np.arange(
                len(result)
            )
            + 1
        )

        print()
        print(
            result.head(20)
            .to_string(
                index=False
            )
        )

        result.to_csv(
            SHAP_IMPORTANCE_FILE,
            index=False,
        )

        return result

    except Exception as exc:

        print()
        print(
            "SHAP CALCULATION WARNING:"
        )

        print(
            str(exc)
        )

        print(
            "Saving empty SHAP result."
        )

        result = pd.DataFrame(
            columns=[
                "feature",
                "mean_abs_shap",
                "rank",
            ]
        )

        result.to_csv(
            SHAP_IMPORTANCE_FILE,
            index=False,
        )

        return result


# ======================================================================
# HIGH RISK RECORDS
# ======================================================================

def create_high_risk_records(
    df: pd.DataFrame,
    probability: np.ndarray,
) -> pd.DataFrame:

    result = df[
        [
            "subdivision",
            "year",
            "month",
            "season",
        ]
    ].copy()

    result["probability"] = (
        probability
    )

    result = (
        result
        .sort_values(
            "probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    result["risk_rank"] = (
        np.arange(
            len(result)
        )
        + 1
    )

    # --------------------------------------------------------------
    # Step 9 selected policy:
    #
    # sigmoid_probability >= 0.09
    #
    # Here Step 10 uses raw model probability because this
    # explainability script explains the XGBoost model itself.
    #
    # The threshold is therefore kept as the same operational
    # threshold for high-risk inspection.
    # --------------------------------------------------------------

    high_risk = result[
        result["probability"]
        >= 0.09
    ].copy()

    high_risk.to_csv(
        HIGH_RISK_FILE,
        index=False,
    )

    return high_risk


# ======================================================================
# SUMMARY
# ======================================================================

def save_summary(
    df: pd.DataFrame,
    X: pd.DataFrame,
    probability: np.ndarray,
    feature_importance: pd.DataFrame,
    permutation_importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
) -> None:

    categorical_features = [
        column
        for column in X.columns
        if isinstance(
            X[column].dtype,
            pd.CategoricalDtype,
        )
    ]

    numeric_features = [
        column
        for column in X.columns
        if not isinstance(
            X[column].dtype,
            pd.CategoricalDtype,
        )
    ]

    summary = {

        "project": "Bharat Earth",

        "version": "V4",

        "rows": int(
            len(df)
        ),

        "feature_count": int(
            X.shape[1]
        ),

        "target": TARGET,

        "target_rate": float(
            df[TARGET].mean()
        ),

        "model": str(
            MODEL_FILE
        ),

        "schema": str(
            SCHEMA_FILE
        ),

        "categorical_features":
            categorical_features,

        "numeric_features":
            numeric_features,

        "prediction_mean": float(
            np.mean(probability)
        ),

        "prediction_min": float(
            np.min(probability)
        ),

        "prediction_max": float(
            np.max(probability)
        ),

        "feature_importance_rows":
            int(
                len(feature_importance)
            ),

        "permutation_importance_rows":
            int(
                len(
                    permutation_importance
                )
            ),

        "shap_importance_rows":
            int(
                len(
                    shap_importance
                )
            ),

        "status": "PASS",
    }

    with open(
        EXPLAINABILITY_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )


# ======================================================================
# MAIN
# ======================================================================

def main():

    banner(
        "10. MODEL EXPLAINABILITY V4"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==============================================================
    # MODEL
    # ==============================================================

    model = load_model()

    schema = read_model_schema(
        model
    )

    # ==============================================================
    # DATA
    # ==============================================================

    df = load_dataset()

    df = validate_dataset(
        df
    )

    # ==============================================================
    # MODEL MATRIX
    # ==============================================================

    X = build_model_matrix(
        df,
        model,
        schema,
    )

    y = (
        pd.to_numeric(
            df[TARGET],
            errors="raise",
        )
        .astype(int)
    )

    # ==============================================================
    # VALIDATE MODEL MATRIX
    # ==============================================================

    validate_model_matrix(
        X,
        model,
    )

    # ==============================================================
    # COMPATIBILITY
    # ==============================================================

    compatibility_test(
        model,
        X,
    )

    # ==============================================================
    # MODEL PROBABILITIES
    # ==============================================================

    banner(
        "GENERATING MODEL PROBABILITIES"
    )

    probability = (
        predict_probability(
            model,
            X,
        )
    )

    print(
        "PREDICTION GENERATION: PASS"
    )

    print(
        "PROBABILITY MIN:",
        float(
            probability.min()
        ),
    )

    print(
        "PROBABILITY MAX:",
        float(
            probability.max()
        ),
    )

    print(
        "PROBABILITY MEAN:",
        float(
            probability.mean()
        ),
    )

    # ==============================================================
    # FEATURE IMPORTANCE
    # ==============================================================

    feature_importance = (
        xgboost_feature_importance(
            model
        )
    )

    # ==============================================================
    # PERMUTATION IMPORTANCE
    # ==============================================================

    permutation_importance = (
        permutation_importance_v4(
            model,
            X,
            y,
        )
    )

    # ==============================================================
    # SHAP
    # ==============================================================

    shap_importance = (
        shap_importance_v4(
            model,
            X,
        )
    )

    # ==============================================================
    # HIGH RISK
    # ==============================================================

    banner(
        "HIGH RISK RECORD ANALYSIS"
    )

    high_risk = (
        create_high_risk_records(
            df,
            probability,
        )
    )

    print(
        "HIGH RISK RECORDS:",
        len(high_risk),
    )

    # ==============================================================
    # SUMMARY
    # ==============================================================

    save_summary(
        df,
        X,
        probability,
        feature_importance,
        permutation_importance,
        shap_importance,
    )

    # ==============================================================
    # OUTPUT VALIDATION
    # ==============================================================

    banner(
        "OUTPUT VALIDATION"
    )

    output_files = [
        FEATURE_IMPORTANCE_FILE,
        PERMUTATION_IMPORTANCE_FILE,
        SHAP_IMPORTANCE_FILE,
        HIGH_RISK_FILE,
        EXPLAINABILITY_SUMMARY_FILE,
    ]

    for path in output_files:

        status = (
            "PASS"
            if path.exists()
            else "FAIL"
        )

        print(
            f"{path.name}: {status}"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Expected output not created: "
                f"{path}"
            )

    # ==============================================================
    # FINAL
    # ==============================================================

    banner(
        "10. MODEL EXPLAINABILITY V4 COMPLETE"
    )

    print(
        "STATUS: PASS"
    )

    print(
        "MODEL FEATURE COUNT:",
        X.shape[1],
    )

    print(
        "CATEGORICAL FEATURES:",
        [
            c
            for c in X.columns
            if isinstance(
                X[c].dtype,
                pd.CategoricalDtype,
            )
        ],
    )

    print()
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
    main()