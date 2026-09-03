from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.secrire.probabilistic.schema import PROBABILISTIC_DIR, TARGET_COLUMN, V4_CALIBRATED_PATH, V4_THRESHOLD


def evaluate_frozen_v4_baseline(
    calibrated_path: str | Path = V4_CALIBRATED_PATH,
    output_dir: str | Path = PROBABILISTIC_DIR,
) -> pd.DataFrame:
    df = pd.read_csv(calibrated_path)
    if "raw_probability" not in df.columns:
        raise ValueError("Frozen V4 calibration file does not contain raw_probability")
    if "actual" not in df.columns:
        raise ValueError("Frozen V4 calibration file does not contain actual")

    result = df.copy()
    result["predicted_positive"] = result["raw_probability"] >= V4_THRESHOLD
    result["tp"] = ((result["predicted_positive"]) & (result["actual"] == 1)).astype(int)
    result["fp"] = ((result["predicted_positive"]) & (result["actual"] == 0)).astype(int)
    result["fn"] = ((~result["predicted_positive"]) & (result["actual"] == 1)).astype(int)
    tp = int(result["tp"].sum())
    fp = int(result["fp"].sum())
    fn = int(result["fn"].sum())
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    recall = tp / (tp + fn) if (tp + fn) else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else np.nan
    alert_rate = float(result["predicted_positive"].mean())

    summary = {
        "track": "A",
        "model": "Frozen V4",
        "threshold": float(V4_THRESHOLD),
        "precision": float(precision) if pd.notna(precision) else np.nan,
        "recall": float(recall) if pd.notna(recall) else np.nan,
        "f1": float(f1) if pd.notna(f1) else np.nan,
        "alert_rate": float(alert_rate),
        "positive_rate": float(result["actual"].mean()),
        "source_artifact": str(calibrated_path),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return pd.DataFrame([summary])
