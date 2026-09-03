from __future__ import annotations

import math
from typing import Any

import pandas as pd

SIMILARITY_FEATURES = [
    "rainfall_mm",
    "rainfall_anomaly",
    "rainfall_deficit_mm",
    "rainfall_trend_3m",
    "rainfall_3m",
    "rainfall_6m",
    "rainfall_12m",
    "rainfall_zscore",
]
SIMILARITY_THRESHOLD = 0.75
HISTORY_HORIZON_MONTHS = 36


def _as_numeric(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return numeric


def compute_similarity_stats(df: pd.DataFrame, features: list[str] | None = None) -> dict[str, tuple[float, float]]:
    """Create deterministic normalized feature statistics from the valid Earth State dataset."""

    selected = features or SIMILARITY_FEATURES
    stats: dict[str, tuple[float, float]] = {}
    for feature in selected:
        if feature not in df.columns:
            continue
        values = pd.to_numeric(df[feature], errors="coerce").fillna(0.0)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std == 0.0:
            std = 1.0
        stats[feature] = (mean, std)
    return stats


def similarity_score(
    current_row: dict[str, Any],
    prior_row: dict[str, Any],
    stats: dict[str, tuple[float, float]] | None = None,
    features: list[str] | None = None,
) -> float:
    """Compute a normalized transparent similarity between two Earth States.

    The metric is a deterministic average normalized Euclidean distance across a
    small set of rainfall/anomaly/hydrological observation features. It is based
    on state variables available at the current timestamp and does not use target
    leakage or future states.
    """

    selected = features or SIMILARITY_FEATURES
    if stats is None:
        stats = {}

    dimension_diffs: list[float] = []
    for feature in selected:
        if feature not in current_row or feature not in prior_row:
            continue
        current_value = _as_numeric(current_row.get(feature, 0.0))
        prior_value = _as_numeric(prior_row.get(feature, 0.0))
        mean, std = stats.get(feature, (0.0, 1.0))
        current_norm = (current_value - mean) / max(std, 1e-6)
        prior_norm = (prior_value - mean) / max(std, 1e-6)
        dimension_diffs.append((current_norm - prior_norm) ** 2)

    if not dimension_diffs:
        return 1.0

    distance = math.sqrt(sum(dimension_diffs) / len(dimension_diffs))
    similarity = 1.0 / (1.0 + distance)
    return max(0.0, min(1.0, similarity))


__all__ = [
    "SIMILARITY_FEATURES",
    "SIMILARITY_THRESHOLD",
    "HISTORY_HORIZON_MONTHS",
    "compute_similarity_stats",
    "similarity_score",
]
