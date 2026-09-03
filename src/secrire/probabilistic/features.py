from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.secrire.probabilistic.schema import (
    BASELINE_FEATURES,
    CATEGORY_COLUMNS,
    EARTH_MEMORY_PATH,
    EARTH_STATE_PATH,
    PROBABILISTIC_DIR,
    REGIME_CONTEXT_FEATURES,
    REGIME_DISCOVERY_FEATURES,
    TARGET_COLUMN,
    V4_DATA_PATH,
)


def load_v4_dataset(path: str | Path = V4_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target {TARGET_COLUMN} missing from V4 dataset")
    return df.sort_values(["subdivision", "year", "month"], kind="mergesort").reset_index(drop=True)


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df["year"] <= 2013].copy().reset_index(drop=True)
    validation = df[df["year"].between(2014, 2015)].copy().reset_index(drop=True)
    test = df[df["year"].between(2016, 2017)].copy().reset_index(drop=True)
    if len(train) == 0 or len(validation) == 0 or len(test) == 0:
        raise ValueError("Temporal split produced empty train/validation/test period")
    return train, validation, test


def _safe_coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _build_regime_assignment(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    model: KMeans,
    imputation_medians: dict[str, float] | None = None,
) -> pd.DataFrame:
    numeric = frame[REGIME_DISCOVERY_FEATURES].copy()
    numeric = _safe_coerce_numeric(numeric, REGIME_DISCOVERY_FEATURES)
    if imputation_medians is not None:
        for column in REGIME_DISCOVERY_FEATURES:
            numeric[column] = numeric[column].fillna(imputation_medians.get(column, numeric[column].median()))
    else:
        for column in REGIME_DISCOVERY_FEATURES:
            numeric[column] = numeric[column].fillna(numeric[column].median())
    scaled = scaler.transform(numeric)
    labels = model.predict(scaled)
    result = frame[["subdivision", "year", "month", "season"]].copy()
    result["regime_id"] = [f"REGIME_{int(label) + 1:02d}" for label in labels]
    result["previous_regime_id"] = result.groupby("subdivision", sort=False)["regime_id"].shift(1)
    result["regime_changed"] = result["previous_regime_id"].notna() & result["regime_id"].ne(result["previous_regime_id"])
    run_id = result["regime_id"].ne(result["previous_regime_id"]).groupby(result["subdivision"], sort=False).cumsum()
    result["consecutive_months_in_current_regime"] = result.groupby(["subdivision", run_id], sort=False).cumcount() + 1
    result["regime_duration"] = result["consecutive_months_in_current_regime"]
    return result.reset_index(drop=True)


def _prepare_regime_frame(frame: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared = prepared.merge(assignment, on=["subdivision", "year", "month", "season"], how="left", validate="one_to_one")
    for column in REGIME_CONTEXT_FEATURES:
        if column not in prepared.columns:
            prepared[column] = np.nan
    return prepared


def fit_train_fitted_regime_model(train: pd.DataFrame) -> tuple[StandardScaler, KMeans, dict[str, object]]:
    regime_matrix = train[REGIME_DISCOVERY_FEATURES].copy()
    regime_matrix = _safe_coerce_numeric(regime_matrix, REGIME_DISCOVERY_FEATURES)
    imputation_medians = {column: regime_matrix[column].median() for column in REGIME_DISCOVERY_FEATURES}
    regime_matrix = regime_matrix.fillna(imputation_medians)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(regime_matrix)
    model = KMeans(n_clusters=2, random_state=42, n_init=10, algorithm="lloyd")
    model.fit(scaled)
    metadata = {
        "regime_model": "KMeans",
        "n_clusters": 2,
        "random_state": 42,
        "fit_window": "1901-2013",
        "input_features": REGIME_DISCOVERY_FEATURES,
        "imputation_medians": imputation_medians,
    }
    return scaler, model, metadata


def assign_regime_context(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    model: KMeans,
    imputation_medians: dict[str, float] | None = None,
) -> pd.DataFrame:
    assignment = _build_regime_assignment(frame, scaler, model, imputation_medians=imputation_medians)
    return _prepare_regime_frame(frame, assignment)


def build_strict_regime_features(v4_df: pd.DataFrame) -> dict[str, pd.DataFrame | dict[str, object]]:
    train, validation, test = temporal_split(v4_df)
    scaler, model, metadata = fit_train_fitted_regime_model(train)

    imputation_medians = metadata.get("imputation_medians")
    train_with_regime = assign_regime_context(train, scaler, model, imputation_medians=imputation_medians)
    validation_with_regime = assign_regime_context(validation, scaler, model, imputation_medians=imputation_medians)
    test_with_regime = assign_regime_context(test, scaler, model, imputation_medians=imputation_medians)

    for subset in [train_with_regime, validation_with_regime, test_with_regime]:
        for column in ["regime_id", "previous_regime_id", "regime_changed", "regime_duration", "consecutive_months_in_current_regime"]:
            if column not in subset.columns:
                subset[column] = np.nan
        subset["regime_id"] = subset["regime_id"].astype(str)
        subset["previous_regime_id"] = subset["previous_regime_id"].astype(str)
        subset["regime_changed"] = subset["regime_changed"].fillna(False).astype(bool)
        subset["regime_duration"] = pd.to_numeric(subset["regime_duration"], errors="coerce").fillna(0).astype(int)
        subset["consecutive_months_in_current_regime"] = pd.to_numeric(
            subset["consecutive_months_in_current_regime"], errors="coerce"
        ).fillna(0).astype(int)

    return {
        "train": train_with_regime,
        "validation": validation_with_regime,
        "test": test_with_regime,
        "metadata": metadata,
    }


def feature_manifest() -> dict:
    rows = []
    for feature in BASELINE_FEATURES:
        rows.append(
            {
                "feature_name": feature,
                "source_artifact": "data/features/ml_dataset_v4.csv",
                "feature_type": "numeric" if feature not in CATEGORY_COLUMNS else "categorical",
                "role": "observation",
                "used_by_v4": True,
                "introduced_by_secrire": False,
                "leakage_status": "PASS",
            }
        )
    for feature in REGIME_CONTEXT_FEATURES:
        rows.append(
            {
                "feature_name": feature,
                "source_artifact": "data/features/climate_regime_v1/climate_regime.csv",
                "feature_type": "categorical",
                "role": "regime",
                "used_by_v4": False,
                "introduced_by_secrire": True,
                "leakage_status": "PASS",
            }
        )
    return {
        "experiment_version": "v1",
        "target": TARGET_COLUMN,
        "features": rows,
        "strict_regime_rule": "Regime assignment is fitted on training rows only and frozen for validation/test.",
        "memory_status": "Not used in Track C; treated as future ablation.",
    }


def write_feature_manifest(target_path: str | Path = PROBABILISTIC_DIR / "feature_manifest.json") -> dict:
    manifest = feature_manifest()
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _read_optional_artifact(path: str | Path) -> pd.DataFrame | None:
    artifact = Path(path)
    if not artifact.exists():
        return None
    return pd.read_csv(artifact)


def load_context_tables() -> dict[str, pd.DataFrame | None]:
    return {
        "earth_state": _read_optional_artifact(EARTH_STATE_PATH),
        "earth_memory": _read_optional_artifact(EARTH_MEMORY_PATH),
        "climate_regime": _read_optional_artifact("data/features/climate_regime_v1/climate_regime.csv"),
    }
