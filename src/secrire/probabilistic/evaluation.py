from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from src.secrire.probabilistic.schema import PROBABILISTIC_DIR, TARGET_COLUMN


def _metric_summary(y_true: pd.Series, probabilities: pd.Series, threshold: float = 0.09) -> dict[str, float]:
    y = pd.Series(y_true).astype(int)
    p = pd.Series(probabilities).astype(float)
    preds = (p >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "alert_rate": float(preds.mean()),
    }


def evaluate_predictions(frame: pd.DataFrame, threshold: float = 0.09) -> pd.DataFrame:
    metrics = _metric_summary(frame[TARGET_COLUMN], frame["probability"], threshold)
    row = {"threshold": threshold, **metrics}
    return pd.DataFrame([row])


def build_reliability_bins(frame: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    probs = pd.to_numeric(frame["probability"], errors="coerce").clip(0.0, 1.0)
    y = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").astype(int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    records = []
    for idx in range(bins):
        left = edges[idx]
        right = edges[idx + 1]
        mask = (probs >= left) & (probs < right)
        if idx == bins - 1:
            mask = (probs >= left) & (probs <= right)
        bin_probs = probs[mask]
        bin_y = y[mask]
        records.append(
            {
                "bin_index": idx,
                "bin_start": float(left),
                "bin_end": float(right),
                "sample_count": int(len(bin_probs)),
                "mean_predicted_probability": float(bin_probs.mean()) if len(bin_probs) else 0.0,
                "observed_positive_rate": float(bin_y.mean()) if len(bin_y) else 0.0,
                "calibration_gap": float((bin_y.mean() - bin_probs.mean()) if len(bin_probs) else 0.0),
            }
        )
    return pd.DataFrame(records)


def write_reference_artifacts(frame: pd.DataFrame, output_dir: str | Path = PROBABILISTIC_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = evaluate_predictions(frame)
    reliability = build_reliability_bins(frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "model_comparison.csv", index=False)
    reliability.to_csv(output_dir / "reliability_curve.csv", index=False)
    return metrics, reliability, frame
