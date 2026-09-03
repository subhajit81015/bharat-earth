from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DEFAULT_CANDIDATE_K = (2, 3, 4, 5, 6)
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class NormalizedFeatures:
    values: np.ndarray
    medians: dict[str, float]
    scales: dict[str, float]


@dataclass(frozen=True)
class DiscoveryResult:
    labels: np.ndarray
    centroids: np.ndarray
    normalized: NormalizedFeatures
    candidate_metrics: list[dict[str, float | int]]
    selected_k: int
    random_state: int
    method: str = "KMeans"


def robust_normalize(frame: pd.DataFrame, features: tuple[str, ...]) -> NormalizedFeatures:
    values = frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    medians = values.median().fillna(0.0)
    q1 = values.quantile(0.25).fillna(0.0)
    q3 = values.quantile(0.75).fillna(0.0)
    scales = (q3 - q1).replace(0, 1.0).fillna(1.0)
    normalized = ((values.fillna(medians) - medians) / scales).to_numpy(dtype=float)
    return NormalizedFeatures(
        values=normalized,
        medians={name: float(medians[name]) for name in features},
        scales={name: float(scales[name]) for name in features},
    )


def _canonicalize(labels: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(range(len(centroids)), key=lambda index: tuple(np.round(centroids[index], 12)))
    mapping = {old: new for new, old in enumerate(order)}
    return np.array([mapping[int(label)] for label in labels], dtype=int), centroids[order]


def discover_regimes(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    candidate_k: tuple[int, ...] = DEFAULT_CANDIDATE_K,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> DiscoveryResult:
    normalized = robust_normalize(frame, features)
    metrics: list[dict[str, float | int]] = []
    models: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k in candidate_k:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10, algorithm="lloyd")
        labels = model.fit_predict(normalized.values)
        canonical_labels, canonical_centroids = _canonicalize(labels, model.cluster_centers_)
        metrics.append({
            "k": int(k),
            "inertia": float(model.inertia_),
            "silhouette_score": float(silhouette_score(normalized.values, canonical_labels)),
            "minimum_cluster_size": int(np.bincount(canonical_labels, minlength=k).min()),
        })
        models[k] = (canonical_labels, canonical_centroids)
    selected = max(metrics, key=lambda item: (float(item["silhouette_score"]), -int(item["k"])))
    selected_k = int(selected["k"])
    labels, centroids = models[selected_k]
    return DiscoveryResult(labels, centroids, normalized, metrics, selected_k, random_state)