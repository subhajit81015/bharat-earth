from __future__ import annotations

import pandas as pd

from src.secrire.probabilistic.schema import TARGET_COLUMN


def build_fixed_bin_reliability(frame: pd.DataFrame) -> pd.DataFrame:
    values = pd.to_numeric(frame["probability"], errors="coerce").clip(0.0, 1.0)
    targets = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").astype(int)
    records = []
    for idx in range(10):
        left = idx / 10.0
        right = (idx + 1) / 10.0
        mask = (values >= left) & (values < right)
        if idx == 9:
            mask = (values >= left) & (values <= right)
        sample = values[mask]
        observed = targets[mask]
        records.append(
            {
                "bin_index": idx,
                "sample_count": int(len(sample)),
                "mean_predicted_probability": float(sample.mean()) if len(sample) else 0.0,
                "observed_positive_rate": float(observed.mean()) if len(observed) else 0.0,
                "calibration_gap": float((observed.mean() - sample.mean()) if len(sample) else 0.0),
            }
        )
    return pd.DataFrame(records)
