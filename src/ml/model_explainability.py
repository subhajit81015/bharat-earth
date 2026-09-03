from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v2.csv"
)

TARGET_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "severe_anomaly_target.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "explainability"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "target_3m_severe_anomaly"

TEST_SIZE = 7668

RANDOM_STATE = 42

SHAP_SAMPLE_SIZE = 1000

PERMUTATION_SAMPLE_SIZE = 3000

TOP_N = 20

POLICY_THRESHOLD = 0.09


# ============================================================
# FEATURES
# ============================================================

# IMPORTANT:
# subdivision is intentionally excluded here initially.
# It is a string categorical variable and requires explicit
# encoding before being passed to XGBoost.

FEATURES = [
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


# ============================================================
# SEASON ENCODING
# ============================================================

SEASON_MAPPING = {
    "WINTER": 0,
    "PRE_MONSOON": 1,
    "MONSOON": 2,
    "POST_MONSOON": 3,
}


# ============================================================
# LOAD + MERGE DATASETS
# ============================================================

def load_dataset():

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature dataset not found:\n"
            f"{FEATURE_FILE}"
        )

    if not TARGET_FILE.exists():

        raise FileNotFoundError(
            f"Target dataset not found:\n"
            f"{TARGET_FILE}"
        )

    features = pd.read_csv(
        FEATURE_FILE
    )

    target = pd.read_csv(
        TARGET_FILE
    )

    print()
    print("=" * 70)
    print("INPUT DATASETS")
    print("=" * 70)

    print(
        "FEATURE FILE:",
        FEATURE_FILE,
    )

    print(
        "FEATURE SHAPE:",
        features.shape,
    )

    print(
        "TARGET FILE:",
        TARGET_FILE,
    )

    print(
        "TARGET SHAPE:",
        target.shape,
    )

    # --------------------------------------------------------
    # Merge keys
    # --------------------------------------------------------

    keys = [
        "subdivision",
        "year",
        "month",
    ]

    required_feature_keys = set(keys)

    required_target_keys = set(
        keys + [TARGET]
    )

    missing_feature_keys = (
        required_feature_keys
        - set(features.columns)
    )

    missing_target_keys = (
        required_target_keys
        - set(target.columns)
    )

    if missing_feature_keys:

        raise ValueError(
            "Missing feature keys:\n"
            f"{sorted(missing_feature_keys)}"
        )

    if missing_target_keys:

        raise ValueError(
            "Missing target columns:\n"
            f"{sorted(missing_target_keys)}"
        )

    # --------------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------------

    feature_duplicates = (
        features
        .duplicated(keys)
        .sum()
    )

    target_duplicates = (
        target
        .duplicated(keys)
        .sum()
    )

    print()
    print(
        "FEATURE DUPLICATES:",
        feature_duplicates,
    )

    print(
        "TARGET DUPLICATES:",
        target_duplicates,
    )

    if feature_duplicates > 0:

        raise ValueError(
            "Duplicate feature keys detected."
        )

    if target_duplicates > 0:

        raise ValueError(
            "Duplicate target keys detected."
        )

    # --------------------------------------------------------
    # Target subset
    # --------------------------------------------------------

    target_small = target[
        keys + [TARGET]
    ].copy()

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    df = features.merge(
        target_small,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    print()
    print("=" * 70)
    print("MERGED DATASET")
    print("=" * 70)

    print(
        "SHAPE:",
        df.shape,
    )

    print(
        "TARGET RATE:",
        df[TARGET].mean(),
    )

    print()
    print(
        "TARGET COUNTS:"
    )

    print(
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Check merge did not lose rows
    # --------------------------------------------------------

    if len(df) != len(features):

        raise ValueError(
            "Merge changed feature row count.\n"
            f"Feature rows: {len(features)}\n"
            f"Merged rows: {len(df)}"
        )

    return df


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(df):

    data = df.copy()

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_features = [
        feature
        for feature in FEATURES
        if feature != "season"
    ]

    for column in numeric_features:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    data["season"] = (
        data["season"]
        .astype(str)
        .map(SEASON_MAPPING)
    )

    if data["season"].isna().any():

        raise ValueError(
            "Unknown season values found."
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="coerce",
    )

    data = data.dropna(
        subset=[TARGET]
    ).copy()

    data[TARGET] = (
        data[TARGET]
        .astype(int)
    )

    # --------------------------------------------------------
    # Validate target
    # --------------------------------------------------------

    invalid_target_values = set(
        data[TARGET].unique()
    ) - {0, 1}

    if invalid_target_values:

        raise ValueError(
            "Invalid target values: "
            f"{sorted(invalid_target_values)}"
        )

    # --------------------------------------------------------
    # Chronological ordering
    # --------------------------------------------------------

    data = data.sort_values(
        [
            "year",
            "month",
            "subdivision",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X = data[
        FEATURES
    ].copy()

    y = data[
        TARGET
    ].copy()

    # --------------------------------------------------------
    # Missing value handling
    # --------------------------------------------------------

    for column in X.columns:

        if X[column].isna().any():

            median_value = (
                X[column]
                .median()
            )

            if pd.isna(
                median_value
            ):

                median_value = 0.0

            X[column] = (
                X[column]
                .fillna(median_value)
            )

    return data, X, y


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def split_data(
    data,
    X,
    y,
):

    if len(data) <= TEST_SIZE:

        raise ValueError(
            "Dataset is smaller than TEST_SIZE."
        )

    split_index = (
        len(data)
        - TEST_SIZE
    )

    X_train = X.iloc[
        :split_index
    ].copy()

    y_train = y.iloc[
        :split_index
    ].copy()

    X_test = X.iloc[
        split_index:
    ].copy()

    y_test = y.iloc[
        split_index:
    ].copy()

    metadata_test = data.iloc[
        split_index:
    ].copy()

    print()
    print("=" * 70)
    print("DATA SPLIT")
    print("=" * 70)

    print(
        "TRAIN ROWS:",
        len(X_train),
    )

    print(
        "TEST ROWS:",
        len(X_test),
    )

    print(
        "TRAIN POSITIVE RATE:",
        f"{y_train.mean():.6f}",
    )

    print(
        "TEST POSITIVE RATE:",
        f"{y_test.mean():.6f}",
    )

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        metadata_test,
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
):

    positive_count = int(
        y_train.sum()
    )

    negative_count = int(
        len(y_train)
        - positive_count
    )

    if positive_count == 0:

        raise ValueError(
            "Training set contains no positive events."
        )

    scale_pos_weight = (
        negative_count
        / positive_count
    )

    print()
    print("=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    print(
        "POSITIVE COUNT:",
        positive_count,
    )

    print(
        "NEGATIVE COUNT:",
        negative_count,
    )

    print(
        "SCALE POS WEIGHT:",
        scale_pos_weight,
    )

    model = XGBClassifier(
        n_estimators=127,
        max_depth=5,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    return model


# ============================================================
# MODEL PERFORMANCE
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    pr_auc = (
        average_precision_score(
            y_test,
            probabilities,
        )
    )

    roc_auc = (
        roc_auc_score(
            y_test,
            probabilities,
        )
    )

    print()
    print("=" * 70)
    print("MODEL TEST PERFORMANCE")
    print("=" * 70)

    print(
        f"PR-AUC: {pr_auc:.6f}"
    )

    print(
        f"ROC-AUC: {roc_auc:.6f}"
    )

    return probabilities


# ============================================================
# XGBOOST IMPORTANCE
# ============================================================

def calculate_xgboost_importance(
    model,
):

    result = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance":
                model.feature_importances_,
        }
    )

    result = (
        result
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    result["rank"] = (
        np.arange(
            len(result)
        )
        + 1
    )

    return result


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

def calculate_permutation_importance(
    model,
    X_test,
    y_test,
):

    sample_size = min(
        PERMUTATION_SAMPLE_SIZE,
        len(X_test),
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    indices = rng.choice(
        len(X_test),
        size=sample_size,
        replace=False,
    )

    X_sample = X_test.iloc[
        indices
    ].copy()

    y_sample = y_test.iloc[
        indices
    ].copy()

    print()
    print("=" * 70)
    print("PERMUTATION IMPORTANCE")
    print("=" * 70)

    print(
        "SAMPLE SIZE:",
        sample_size,
    )

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="average_precision",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean":
                result.importances_mean,
            "importance_std":
                result.importances_std,
        }
    )

    importance = (
        importance
        .sort_values(
            "importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance["rank"] = (
        np.arange(
            len(importance)
        )
        + 1
    )

    return importance


# ============================================================
# SHAP
# ============================================================

def calculate_shap(
    model,
    X_test,
):

    try:

        import shap

    except ImportError:

        print()
        print(
            "SHAP IS NOT INSTALLED."
        )

        print(
            "Run:"
        )

        print(
            "pip install shap"
        )

        return None, None

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(X_test),
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    indices = rng.choice(
        len(X_test),
        size=sample_size,
        replace=False,
    )

    X_sample = X_test.iloc[
        indices
    ].copy()

    print()
    print("=" * 70)
    print("SHAP ANALYSIS")
    print("=" * 70)

    print(
        "SHAP SAMPLE SIZE:",
        sample_size,
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = (
        explainer.shap_values(
            X_sample
        )
    )

    # --------------------------------------------------------
    # SHAP version compatibility
    # --------------------------------------------------------

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

    if shap_values.ndim == 3:

        shap_values = (
            shap_values[:, :, -1]
        )

    if shap_values.shape[1] != len(
        FEATURES
    ):

        raise ValueError(
            "Unexpected SHAP shape: "
            f"{shap_values.shape}"
        )

    mean_abs_shap = (
        np.abs(
            shap_values
        )
        .mean(axis=0)
    )

    importance = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean_abs_shap":
                mean_abs_shap,
        }
    )

    importance = (
        importance
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    importance["rank"] = (
        np.arange(
            len(importance)
        )
        + 1
    )

    shap_values_df = pd.DataFrame(
        shap_values,
        columns=FEATURES,
    )

    return (
        importance,
        shap_values_df,
    )


# ============================================================
# HIGH RISK RECORDS
# ============================================================

def identify_high_risk_records(
    model,
    X_test,
    y_test,
    metadata_test,
):

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    result = metadata_test.copy()

    result[
        "predicted_probability"
    ] = probabilities

    result["actual"] = (
        y_test.to_numpy()
    )

    result["risk_alert"] = (
        probabilities
        >= POLICY_THRESHOLD
    ).astype(int)

    result = (
        result
        .sort_values(
            "predicted_probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# SAVE RESULTS
# ============================================================

def save_outputs(
    xgb_importance,
    permutation,
    shap_importance,
    shap_values,
    high_risk,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    xgb_importance.to_csv(
        OUTPUT_DIR
        / "xgboost_feature_importance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Permutation
    # --------------------------------------------------------

    permutation.to_csv(
        OUTPUT_DIR
        / "permutation_importance.csv",
        index=False,
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    if shap_importance is not None:

        shap_importance.to_csv(
            OUTPUT_DIR
            / "shap_importance.csv",
            index=False,
        )

    if shap_values is not None:

        shap_values.to_csv(
            OUTPUT_DIR
            / "shap_values.csv",
            index=False,
        )

    # --------------------------------------------------------
    # High risk
    # --------------------------------------------------------

    high_risk.head(
        100
    ).to_csv(
        OUTPUT_DIR
        / "top_100_high_risk_records.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Combined importance
    # --------------------------------------------------------

    combined = xgb_importance[
        [
            "feature",
            "importance",
        ]
    ].rename(
        columns={
            "importance":
                "xgboost_importance",
        }
    )

    combined = combined.merge(
        permutation[
            [
                "feature",
                "importance_mean",
                "importance_std",
            ]
        ],
        on="feature",
        how="left",
    )

    if shap_importance is not None:

        combined = combined.merge(
            shap_importance[
                [
                    "feature",
                    "mean_abs_shap",
                ]
            ],
            on="feature",
            how="left",
        )

    combined.to_csv(
        OUTPUT_DIR
        / "combined_feature_importance.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        OUTPUT_DIR
        / "xgboost_feature_importance.csv"
    )

    print(
        OUTPUT_DIR
        / "permutation_importance.csv"
    )

    if shap_importance is not None:

        print(
            OUTPUT_DIR
            / "shap_importance.csv"
        )

    print(
        OUTPUT_DIR
        / "top_100_high_risk_records.csv"
    )

    print(
        OUTPUT_DIR
        / "combined_feature_importance.csv"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("3.7 MODEL EXPLAINABILITY")
    print("=" * 70)

    # --------------------------------------------------------
    # Load and merge
    # --------------------------------------------------------

    df = load_dataset()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    (
        data,
        X,
        y,
    ) = prepare_data(
        df
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_test,
        y_test,
        metadata_test,
    ) = split_data(
        data,
        X,
        y,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    probabilities = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # --------------------------------------------------------
    # XGBoost importance
    # --------------------------------------------------------

    xgb_importance = (
        calculate_xgboost_importance(
            model
        )
    )

    print()
    print("=" * 70)
    print("TOP 20 XGBOOST FEATURES")
    print("=" * 70)

    print(
        xgb_importance
        .head(TOP_N)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Permutation
    # --------------------------------------------------------

    permutation = (
        calculate_permutation_importance(
            model,
            X_test,
            y_test,
        )
    )

    print()
    print("=" * 70)
    print("TOP 20 PERMUTATION FEATURES")
    print("=" * 70)

    print(
        permutation
        .head(TOP_N)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    (
        shap_importance,
        shap_values,
    ) = calculate_shap(
        model,
        X_test,
    )

    if shap_importance is not None:

        print()
        print("=" * 70)
        print("TOP 20 SHAP FEATURES")
        print("=" * 70)

        print(
            shap_importance
            .head(TOP_N)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # High risk
    # --------------------------------------------------------

    high_risk = (
        identify_high_risk_records(
            model,
            X_test,
            y_test,
            metadata_test,
        )
    )

    print()
    print("=" * 70)
    print("TOP 20 HIGH-RISK RECORDS")
    print("=" * 70)

    display_columns = [
        "subdivision",
        "year",
        "month",
        "actual",
        "predicted_probability",
        "risk_alert",
    ]

    print(
        high_risk[
            display_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        xgb_importance,
        permutation,
        shap_importance,
        shap_values,
        high_risk,
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3.7 MODEL EXPLAINABILITY COMPLETE")
    print("=" * 70)

    print(
        "POLICY THRESHOLD:",
        POLICY_THRESHOLD,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()