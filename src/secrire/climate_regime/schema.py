from __future__ import annotations

REGIME_ENGINE_VERSION = "v1"
REGIME_ASSIGNMENT_VERSION = "v1"
TARGET_COLUMN = "target_3m_severe_anomaly"

ASSIGNMENT_COLUMNS = [
    "state_id", "subdivision", "year", "month", "season", "regime_id",
    "regime_assignment_version", "previous_regime_id", "regime_duration",
    "consecutive_months_in_current_regime", "regime_changed",
]
TRANSITION_COLUMNS = [
    "subdivision", "previous_state_id", "current_state_id", "previous_regime_id",
    "current_regime_id", "regime_changed", "months_in_previous_regime",
]