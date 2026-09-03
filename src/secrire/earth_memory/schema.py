from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MEMORY_VERSION = "v1"
HISTORY_HORIZON_MONTHS = 36
SIMILARITY_THRESHOLD = 0.75
SIMILARITY_METHOD = "normalized_observation_distance"


@dataclass(slots=True)
class MemoryQualityState:
    """Memory completeness and quality metadata."""

    memory_complete: bool = False
    history_sufficient: bool = False
    missing_history_count: int = 0
    memory_quality_status: str = "UNKNOWN"


@dataclass(slots=True)
class EarthMemory:
    """Temporal memory representation for a validated Earth State.

    This is an experimental memory layer that uses only prior information and
    state-level observations. It never uses future observations or target labels.
    """

    memory_id: str
    state_id: str
    subdivision: str
    year: int
    month: int
    season: str
    previous_state_id: str | None = None
    memory_window_months: int = 0
    available_history_months: int = 0
    historical_state_count: int = 0
    anomaly_persistence: float = 0.0
    deficit_persistence: float = 0.0
    rainfall_condition_persistence: float = 0.0
    consecutive_anomaly_direction: int = 0
    recurrence_count: int = 0
    months_since_similar_state: int | None = None
    historical_similarity_count: int = 0
    similarity_score: float = 0.0
    memory_complete: bool = False
    history_sufficient: bool = False
    missing_history_count: int = 0
    memory_quality_status: str = "UNKNOWN"
    source_state_id: str | None = None
    source_dataset: str = "data/features/earth_state_v1/earth_state.csv"
    source_artifact: str = "earth_state_v1/earth_state.csv"
    memory_generation_version: str = MEMORY_VERSION
    lineage_metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "EarthMemory",
    "MemoryQualityState",
    "MEMORY_VERSION",
    "HISTORY_HORIZON_MONTHS",
    "SIMILARITY_THRESHOLD",
    "SIMILARITY_METHOD",
]
