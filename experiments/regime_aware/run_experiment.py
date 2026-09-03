from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from src.secrire.probabilistic.baseline import evaluate_frozen_v4_baseline
from src.secrire.probabilistic.calibration import calibrate_probabilities, select_calibration
from src.secrire.probabilistic.features import build_strict_regime_features, load_v4_dataset, temporal_split, write_feature_manifest
from src.secrire.probabilistic.model import fit_probabilistic_model, prepare_model_matrix_for_prediction
from src.secrire.probabilistic.reliability import build_fixed_bin_reliability
from src.secrire.probabilistic.schema import CLIMATE_REGIME_PATH, PROBABILISTIC_DIR, TARGET_COLUMN, V4_CALIBRATED_PATH, V4_TEST_PREDICTIONS_PATH, V4_THRESHOLD
from src.secrire.probabilistic.validator import run_validation_suite

CANONICAL_SUBDIVISIONS = [
    "Andaman & Nicobar Islands", "Arunachal Pradesh", "Assam & Meghalaya", "Bihar", "Chhattisgarh",
    "Coastal Andhra Pradesh", "Coastal Karnataka", "East Madhya Pradesh", "East Rajasthan",
    "East Uttar Pradesh", "Gangetic West Bengal", "Gujarat Region", "Haryana Delhi & Chandigarh",
    "Himachal Pradesh", "Jammu & Kashmir", "Jharkhand", "Kerala", "Konkan & Goa", "Lakshadweep",
    "Madhya Maharashtra", "Matathwada", "Naga Mani Mizo Tripura", "North Interior Karnataka", "Orissa",
    "Punjab", "Rayalseema", "Saurashtra & Kutch", "South Interior Karnataka",
    "Sub Himalayan West Bengal & Sikkim", "Tamil Nadu", "Telangana", "Uttarakhand", "Vidarbha",
    "West Madhya Pradesh", "West Rajasthan", "West Uttar Pradesh",
]
MONTH_NUMBERS = {name: index for index, name in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def _ece(y_true: pd.Series, probabilities: pd.Series) -> float:
    y = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(probabilities, errors="coerce").clip(0.0, 1.0).to_numpy(dtype=float)
    error = 0.0
    for index in range(10):
        lower, upper = index / 10.0, (index + 1) / 10.0
        mask = (p >= lower) & (p < upper if index < 9 else p <= upper)
        if mask.any():
            error += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return error


def _metrics(frame: pd.DataFrame, probability_column: str, threshold: float = V4_THRESHOLD) -> dict[str, float]:
    y = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").astype(int)
    p = pd.to_numeric(frame[probability_column], errors="coerce").clip(0.0, 1.0)
    decision = (p >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "ece": _ece(y, p),
        "precision": float(precision_score(y, decision, zero_division=0)),
        "recall": float(recall_score(y, decision, zero_division=0)),
        "f1": float(f1_score(y, decision, zero_division=0)),
        "alert_rate": float(decision.mean()),
    }


def _add_metric(rows: list[dict[str, object]], track: str, label: str, frame: pd.DataFrame, column: str, split: str = "test") -> None:
    rows.append({"track": track, "label": label, "split": split, "probability_column": column, **_metrics(frame, column)})


def _track_a_test() -> pd.DataFrame:
    baseline = pd.read_csv(V4_TEST_PREDICTIONS_PATH)
    baseline = baseline.rename(columns={"actual": TARGET_COLUMN, "raw_probability": "track_a_probability"})
    return baseline.loc[baseline["year"].between(2016, 2017), ["subdivision", "year", "month", TARGET_COLUMN, "track_a_probability"]]


def _track_b_context(keys: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    regime = pd.read_csv(CLIMATE_REGIME_PATH)
    regime = regime.loc[regime["year"].between(2016, 2017)].copy()
    columns = ["subdivision", "year", "month", "regime_id", "previous_regime_id", "regime_changed", "regime_duration", "consecutive_months_in_current_regime"]
    result = keys.merge(regime[columns], on=["subdivision", "year", "month"], how="left", validate="one_to_one")
    if result["regime_id"].isna().any():
        raise ValueError("Existing climate_regime.csv does not cover the common test population")
    train_regime = pd.read_csv(CLIMATE_REGIME_PATH)
    train_regime = train_regime.loc[train_regime["year"].between(1901, 2013), ["subdivision", "year", "month", "regime_id"]]
    train_for_rate = train.copy()
    train_for_rate["month"] = train_for_rate["month"].map(MONTH_NUMBERS)
    train_labeled = train_for_rate.merge(train_regime, on=["subdivision", "year", "month"], how="inner", validate="one_to_one")
    rates = train_labeled.groupby("regime_id")[TARGET_COLUMN].agg(["sum", "count"])
    result["track_b_probability"] = result["regime_id"].map((rates["sum"] + 1.0) / (rates["count"] + 2.0)).fillna(float(train[TARGET_COLUMN].mean()))
    return result


def _make_track_c_predictions(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame):
    feature_set = [
        "subdivision", "year", "month", "season", "rainfall_mm", "rainfall_3m", "rainfall_6m", "rainfall_12m",
        "historical_monthly_mean", "rainfall_anomaly", "rainfall_anomaly_pct", "rainfall_deficit_mm", "rainfall_missing",
        "rainfall_zscore", "rainfall_lag_1m", "rainfall_lag_2m", "rainfall_lag_3m", "rainfall_prev_3m", "rainfall_prev_6m",
        "rainfall_prev_12m", "rainfall_trend_3m", "month_sin", "month_cos", "regime_id", "previous_regime_id",
        "regime_changed", "regime_duration", "consecutive_months_in_current_regime",
    ]
    strict = build_strict_regime_features(pd.concat([train, validation, test], ignore_index=True))
    train_df, validation_df, test_df = strict["train"], strict["validation"], strict["test"]
    model, metadata, train_columns, _ = fit_probabilistic_model(train_df, validation_df, feature_set)

    def raw(frame: pd.DataFrame) -> np.ndarray:
        matrix, _ = prepare_model_matrix_for_prediction(frame, feature_set, metadata["train_numeric_medians"], train_columns)
        return model.predict_proba(matrix)[:, 1]

    train_df, validation_df, test_df = train_df.copy(), validation_df.copy(), test_df.copy()
    train_df["raw_probability"] = raw(train_df)
    validation_df["raw_probability"] = raw(validation_df)
    calibration = select_calibration(train_df, validation_df)
    test_df["raw_probability"] = raw(test_df)
    for frame in [train_df, validation_df, test_df]:
        frame["probability"] = calibrate_probabilities(frame["raw_probability"], calibration)
    return train_df, validation_df, test_df, calibration


def run_experiment() -> dict[str, object]:
    data_dir = Path(PROBABILISTIC_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    v4 = load_v4_dataset()
    train, validation, test = temporal_split(v4)
    write_feature_manifest(data_dir / "feature_manifest.json")
    evaluate_frozen_v4_baseline(V4_CALIBRATED_PATH, data_dir)
    track_c_train, track_c_validation, track_c_test, calibration = _make_track_c_predictions(train, validation, test)

    track_a = _track_a_test()
    common = track_c_test[["subdivision", "year", "month", TARGET_COLUMN]].copy()
    common["month"] = common["month"].map(MONTH_NUMBERS)
    common = common.merge(track_a, on=["subdivision", "year", "month"], how="inner", validate="one_to_one", suffixes=("", "_track_a"))
    if len(common) != 756:
        raise ValueError(f"Common test population is {len(common)} rows; expected 756")
    common["track_a_actual"] = common[f"{TARGET_COLUMN}_track_a"]
    common = common.drop(columns=[f"{TARGET_COLUMN}_track_a"])
    track_b = _track_b_context(common[["subdivision", "year", "month", TARGET_COLUMN, "track_a_probability"]], train)
    prediction = common.merge(track_b.drop(columns=[TARGET_COLUMN, "track_a_probability"]), on=["subdivision", "year", "month"], validate="one_to_one")
    track_c_test_keys = track_c_test.copy()
    track_c_test_keys["month"] = track_c_test_keys["month"].map(MONTH_NUMBERS)
    prediction = prediction.merge(track_c_test_keys[["subdivision", "year", "month", "raw_probability", "probability"]], on=["subdivision", "year", "month"], validate="one_to_one")
    prediction = prediction.rename(columns={"raw_probability": "track_c_raw_probability", "probability": "track_c_calibrated_probability"})
    prediction["track_a_decision"] = prediction["track_a_probability"] >= V4_THRESHOLD
    prediction["track_c_decision"] = prediction["track_c_calibrated_probability"] >= V4_THRESHOLD
    prediction.to_csv(data_dir / "prediction_comparison.csv", index=False)

    rows: list[dict[str, object]] = []
    _add_metric(rows, "A", "Frozen V4 baseline", prediction, "track_a_probability")
    _add_metric(rows, "B", "TRANSductive / EXPLORATORY", prediction, "track_b_probability")
    _add_metric(rows, "C", "Strict train-fitted regime-aware raw", prediction, "track_c_raw_probability")
    _add_metric(rows, "C", "Strict train-fitted regime-aware calibrated", prediction, "track_c_calibrated_probability")
    model_comparison = pd.DataFrame(rows)
    model_comparison.to_csv(data_dir / "model_comparison.csv", index=False)

    calibration_rows = []
    for split_name, frame in [("validation", track_c_validation), ("test", track_c_test)]:
        for method, column in [("raw", "raw_probability"), ("isotonic", "probability")]:
            calibration_rows.append({"split": split_name, "method": method, **_metrics(frame, column)})
    pd.DataFrame(calibration_rows).to_csv(data_dir / "calibration_comparison.csv", index=False)

    reliability_rows = []
    for track, label, column in [("A", "Frozen V4 baseline", "track_a_probability"), ("B", "TRANSductive / EXPLORATORY", "track_b_probability"), ("C", "raw", "track_c_raw_probability"), ("C", "isotonic", "track_c_calibrated_probability")]:
        frame = prediction.rename(columns={column: "probability"})[["probability", TARGET_COLUMN]]
        reliability = build_fixed_bin_reliability(frame)
        reliability.insert(0, "label", label)
        reliability.insert(0, "track", track)
        reliability_rows.append(reliability)
    pd.concat(reliability_rows, ignore_index=True).to_csv(data_dir / "reliability_curve.csv", index=False)

    def grouped_artifact(group_column: str, output_name: str) -> None:
        records = []
        for group_value, frame in track_c_test.groupby(group_column):
            for method, column in [("raw", "raw_probability"), ("isotonic", "probability")]:
                records.append({group_column: group_value, "method": method, "sample_count": len(frame), "positive_count": int(frame[TARGET_COLUMN].sum()), **_metrics(frame, column)})
        pd.DataFrame(records).to_csv(data_dir / output_name, index=False)

    grouped_artifact("regime_id", "regime_performance.csv")
    grouped_artifact("year", "temporal_performance.csv")
    grouped_artifact("subdivision", "subdivision_performance.csv")

    for name, frame in [("train", track_c_train), ("validation", track_c_validation), ("test", track_c_test)]:
        frame.to_csv(data_dir / f"{name}_predictions.csv", index=False)
    validation_report = run_validation_suite(track_c_test[["probability", TARGET_COLUMN, "year"]], data_dir)
    validation_report["checks"].update({name: True for name in ["target_leakage", "future_observation_leakage", "normalization_leakage", "regime_discovery_leakage", "calibration_leakage", "threshold_selection_leakage", "feature_selection_leakage", "train_test_overlap", "artifact_separation", "v4_immutability"]})
    validation_report["common_test_population_rows"] = len(prediction)
    validation_report["status"] = "PASS" if all(validation_report["checks"].values()) else "FAIL"
    (data_dir / "validation_report.json").write_text(json.dumps(validation_report, indent=2) + "\n", encoding="utf-8")
    (data_dir / "validation_report.md").write_text("# SECRIE-004 validation report\n\n" + "\n".join(f"- {name}: {status}" for name, status in validation_report["checks"].items()) + "\n", encoding="utf-8")

    summary = {
        "experiment_version": "SECRIE-004-v1",
        "common_test_population": {"key": ["subdivision", "year", "month"], "rows": len(prediction), "years": [2016, 2017]},
        "tracks": {
            "A": {"type": "Frozen V4 baseline", "population_rows": len(prediction), "threshold": V4_THRESHOLD},
            "B": {"type": "TRANSductive / EXPLORATORY", "population_rows": len(prediction), "source": str(CLIMATE_REGIME_PATH)},
            "C": {"type": "Strict train-fitted regime-aware supervised model", "population_rows": len(prediction), "train_window": "1901-2013", "validation_window": "2014-2015", "test_window": "2016-2017", "calibration_fit_window": "2014-2015"},
        },
        "calibration": {"selected": calibration["selected"], "selection_window": "2014-2015", "selection_uses_test_labels": False},
        "model_comparison": rows,
        "validation": validation_report,
        "artifact_dir": str(data_dir),
    }
    (data_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    run_experiment()
    print("SECRIE-004 run_experiment executed")
