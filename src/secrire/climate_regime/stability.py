from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_stability(
    labels: np.ndarray,
    centroids: np.ndarray,
    normalized_values: np.ndarray,
    rerun_labels: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for index in range(len(centroids)):
        mask = labels == index
        rerun_mask = rerun_labels == index
        distances = np.linalg.norm(normalized_values[mask] - centroids[index], axis=1) if mask.any() else np.array([np.inf])
        centroid_stability = float(1.0 / (1.0 + np.mean(distances)))
        agreement = float(np.mean(mask == rerun_mask))
        rows.append({
            "regime_id": f"REGIME_{index + 1:02d}",
            "assignment_count": int(mask.sum()),
            "minimum_cluster_size": int(mask.sum()),
            "centroid_stability": centroid_stability,
            "rerun_assignment_agreement": agreement,
            "stability_score": float((centroid_stability + agreement) / 2),
            "stability_interpretation": "algorithmic stability, not physical climate stability",
        })
    return pd.DataFrame(rows)