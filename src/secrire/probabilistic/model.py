from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.secrire.probabilistic.schema import CATEGORY_COLUMNS, MODELS_DIR, RANDOM_SEED, TARGET_COLUMN


def _prepare_model_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    feature_frame = frame[feature_columns].copy()
    for column in CATEGORY_COLUMNS:
        if column in feature_frame.columns:
            feature_frame[column] = feature_frame[column].astype(str)
    numeric_columns = [column for column in feature_frame.columns if column not in CATEGORY_COLUMNS]
    for column in numeric_columns:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce")
    encoded = pd.get_dummies(feature_frame, columns=[column for column in CATEGORY_COLUMNS if column in feature_frame.columns], dummy_na=False)
    return encoded, list(encoded.columns)


def _fit_train_medians(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    feature_frame = frame[feature_columns].copy()
    numeric_columns = [column for column in feature_frame.columns if column not in CATEGORY_COLUMNS]
    medians: dict[str, float] = {}
    for column in numeric_columns:
        medians[column] = pd.to_numeric(feature_frame[column], errors="coerce").median()
    return medians


def _apply_train_medians(frame: pd.DataFrame, feature_columns: list[str], medians: dict[str, float]) -> pd.DataFrame:
    feature_frame = frame[feature_columns].copy()
    for column in CATEGORY_COLUMNS:
        if column in feature_frame.columns:
            feature_frame[column] = feature_frame[column].astype(str)
    for column, median_value in medians.items():
        if column in feature_frame.columns:
            feature_frame[column] = pd.to_numeric(feature_frame[column], errors="coerce").fillna(median_value)
    return feature_frame


def prepare_model_matrix_for_prediction(
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_medians: dict[str, float],
    train_columns: pd.Index | None = None,
) -> pd.DataFrame:
    feature_frame = _apply_train_medians(frame, feature_columns, train_medians)
    encoded, columns = _prepare_model_matrix(feature_frame, feature_columns)
    if train_columns is not None:
        encoded = encoded.reindex(columns=train_columns, fill_value=0.0)
    return encoded, columns


def fit_probabilistic_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_columns: list[str],
    model_name: str = "strict_regime_logistic_regression",
    random_seed: int = RANDOM_SEED,
) -> tuple[LogisticRegression, dict[str, object], pd.Index, pd.Index]:
    train_medians = _fit_train_medians(train, feature_columns)
    X_train_raw = _apply_train_medians(train, feature_columns, train_medians)
    X_valid_raw = _apply_train_medians(validation, feature_columns, train_medians)
    X_train, train_columns = _prepare_model_matrix(X_train_raw, feature_columns)
    X_valid, _ = _prepare_model_matrix(X_valid_raw, feature_columns)
    X_valid = X_valid.reindex(columns=train_columns, fill_value=0.0)

    y_train = train[TARGET_COLUMN].astype(int).to_numpy()
    y_valid = validation[TARGET_COLUMN].astype(int).to_numpy()

    model = LogisticRegression(
        solver="liblinear",
        C=0.1,
        max_iter=5000,
        random_state=random_seed,
    )
    model.fit(X_train, y_train)

    metadata = {
        "model_type": "LogisticRegression",
        "solver": "liblinear",
        "regularization": {"type": "L2", "C": 0.1},
        "convergence_status": "converged" if int(model.n_iter_[0]) < 5000 else "not_converged",
        "n_iterations": int(model.n_iter_[0]),
        "convergence_change": "Changed from lbfgs to liblinear with stronger L2 regularization after lbfgs reached 2000 iterations; no feature or data-window change.",
        "random_seed": int(random_seed),
        "selected_features": feature_columns,
        "train_window": "1901-2013",
        "validation_window": "2014-2015",
        "test_window": "2016-2017",
        "model_name": model_name,
        "feature_count": len(train_columns),
        "target": TARGET_COLUMN,
        "train_numeric_medians": train_medians,
    }

    path = Path(MODELS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    model_path = path / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    with (path / f"{model_name}_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    return model, metadata, pd.Index(train_columns), pd.Index(X_valid.columns)
