from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.secrire.probabilistic.schema import PROBABILISTIC_DIR, TARGET_COLUMN


def _safe_calibration_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"raw_probability", TARGET_COLUMN}
    if not required.issubset(df.columns):
        raise ValueError(f"Calibration input must include raw_probability and {TARGET_COLUMN}")
    result = df[["raw_probability", TARGET_COLUMN]].copy()
    result["raw_probability"] = pd.to_numeric(result["raw_probability"], errors="coerce")
    result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN], errors="coerce").astype(int)
    return result.dropna().reset_index(drop=True)


def fit_calibration(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, object]:
    train_df = _safe_calibration_frame(train)
    validation_df = _safe_calibration_frame(validation)

    raw_validation = validation_df["raw_probability"].to_numpy(dtype=float)
    y_validation = validation_df[TARGET_COLUMN].to_numpy(dtype=int)

    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    isotonic.fit(raw_validation, y_validation)

    sigmoid = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    sigmoid.fit(raw_validation.reshape(-1, 1), y_validation)

    isotonic_probability = isotonic.predict(raw_validation)
    sigmoid_probability = sigmoid.predict_proba(raw_validation.reshape(-1, 1))[:, 1]

    def calibration_error(probability: np.ndarray) -> float:
        error = 0.0
        for index in range(10):
            lower = index / 10.0
            upper = (index + 1) / 10.0
            mask = (raw_validation >= lower) & (raw_validation < upper if index < 9 else raw_validation <= upper)
            if mask.any():
                error += float(mask.mean()) * abs(float(probability[mask].mean()) - float(y_validation[mask].mean()))
        return error

    selected = "sigmoid" if np.nanmean(np.abs(sigmoid_probability - y_validation)) <= np.nanmean(np.abs(isotonic_probability - y_validation)) else "isotonic"

    calibration = {
        "selected": selected,
        "fit_window": "2014-2015",
        "raw_validation_brier": float(np.mean((raw_validation - y_validation) ** 2)),
        "raw_validation_ece": calibration_error(raw_validation),
        "isotonic_validation_brier": float(np.mean((isotonic_probability - y_validation) ** 2)),
        "isotonic_validation_ece": calibration_error(isotonic_probability),
        "sigmoid_validation_brier": float(np.mean((sigmoid_probability - y_validation) ** 2)),
        "sigmoid_validation_ece": calibration_error(sigmoid_probability),
        "selection_metric": "validation mean absolute calibration error",
        "isotonic": isotonic,
        "sigmoid": sigmoid,
    }

    output_path = Path(PROBABILISTIC_DIR) / "calibration_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected": selected,
                "fit_window": "2014-2015",
                "selection_metric": calibration["selection_metric"],
                "candidates": {
                    "raw": {"brier": calibration["raw_validation_brier"], "ece": calibration["raw_validation_ece"]},
                    "isotonic": {"brier": calibration["isotonic_validation_brier"], "ece": calibration["isotonic_validation_ece"]},
                    "sigmoid": {"brier": calibration["sigmoid_validation_brier"], "ece": calibration["sigmoid_validation_ece"]},
                },
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    return calibration


def calibrate_probabilities(probabilities: pd.Series, calibration: dict[str, object]) -> pd.Series:
    raw = pd.to_numeric(probabilities, errors="coerce").to_numpy(dtype=float)
    isotonic = calibration["isotonic"]
    sigmoid = calibration["sigmoid"]
    selected = calibration["selected"]
    if selected == "isotonic":
        calibrated = isotonic.predict(raw)
    else:
        calibrated = sigmoid.predict_proba(raw.reshape(-1, 1))[:, 1]
    return pd.Series(np.clip(calibrated, 0.0, 1.0), index=probabilities.index)


def select_calibration(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, object]:
    return fit_calibration(train, validation)
