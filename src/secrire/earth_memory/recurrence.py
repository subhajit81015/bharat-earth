from __future__ import annotations

from typing import Any

import pandas as pd

from src.secrire.earth_memory.similarity import SIMILARITY_FEATURES, SIMILARITY_THRESHOLD, similarity_score


def find_similar_prior_states(
    current_row: dict[str, Any],
    history_rows: list[dict[str, Any]],
    stats: dict[str, tuple[float, float]] | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
    features: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return prior states that are similar to the current state within the chosen threshold."""

    matches: list[dict[str, Any]] = []
    selected = features or SIMILARITY_FEATURES
    for prior_row in history_rows:
        score = similarity_score(current_row, prior_row, stats=stats, features=selected)
        if score >= threshold:
            months_ago = int(current_row["state_month_index"] - prior_row["state_month_index"])
            matches.append({
                "state_id": prior_row["state_id"],
                "months_ago": months_ago,
                "similarity": score,
            })
    return sorted(matches, key=lambda item: item["months_ago"])


def summarize_recurrence(
    current_row: dict[str, Any],
    history_rows: list[dict[str, Any]],
    stats: dict[str, tuple[float, float]] | None = None,
    threshold: float = SIMILARITY_THRESHOLD,
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Compute recurrence metadata using only prior states in the same subdivision."""

    matches = find_similar_prior_states(current_row, history_rows, stats=stats, threshold=threshold, features=features)

    if not matches:
        return {
            "recurrence_count": 0,
            "months_since_similar_state": None,
            "historical_similarity_count": 0,
            "similarity_matches": [],
        }

    latest_match = matches[-1]
    return {
        "recurrence_count": len(matches),
        "months_since_similar_state": latest_match["months_ago"],
        "historical_similarity_count": len(matches),
        "similarity_matches": matches,
    }


__all__ = ["find_similar_prior_states", "summarize_recurrence"]
