from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

VALID_SEASONS = {
    "WINTER",
    "PRE_MONSOON",
    "MONSOON",
    "POST_MONSOON",
    "SUMMER",
    "AUTUMN",
    "SPRING",
}


@dataclass(slots=True)
class PredictionState:
    """Prediction-time Earth State fields.

    This object holds the features available before an observed outcome is known.
    It intentionally remains empty by default to prevent accidental leakage from
    the target field or any post-outcome analysis signal.
    """

    features: dict[str, Any] = field(default_factory=dict)
    model_version: str | None = None
    prediction_timestamp: str | None = None


@dataclass(slots=True)
class OutcomeState:
    """Post-outcome analysis state.

    This state is explicitly separate from prediction-time observation and must
    never be used when constructing a prediction-time Earth State.
    """

    target_3m_severe_anomaly: int | None = None
    observed_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DataQualityState:
    """Data quality and validation metadata for a single Earth State."""

    source_row_id: int | None = None
    validation_status: str = "UNKNOWN"
    missing_fields: list[str] = field(default_factory=list)
    invalid_fields: list[str] = field(default_factory=list)
    repaired_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ObservationState:
    """Observation-only Earth State used before prediction.

    This represents the available rainfall and environmental observation state at
    time t, including memory and anomaly features. The target variable used for
    future anomaly classification is intentionally excluded from this model.
    """

    state_id: str
    subdivision: str
    year: int
    month: int
    season: str
    temporal_position: int = 0

    rainfall_mm: float | None = None
    rainfall_missing: int | float = 0
    rainfall_3m: float | None = None
    rainfall_6m: float | None = None
    rainfall_12m: float | None = None
    rainfall_lag_1m: float | None = None
    rainfall_lag_2m: float | None = None
    rainfall_lag_3m: float | None = None
    rainfall_prev_3m: float | None = None
    rainfall_prev_6m: float | None = None
    rainfall_prev_12m: float | None = None

    historical_monthly_mean: float | None = None
    rainfall_anomaly: float | None = None
    rainfall_anomaly_pct: float | None = None
    rainfall_deficit_mm: float | None = None
    rainfall_zscore: float | None = None

    rainfall_trend_3m: float | None = None

    month_sin: float | None = None
    month_cos: float | None = None


@dataclass(slots=True)
class EarthState:
    """Structured Earth State representation for the SECRIE engine.

    The prediction-time state is intentionally decoupled from outcome analysis so
    future target leakage does not enter the state used for forecasting.
    """

    observation: ObservationState
    prediction: PredictionState = field(default_factory=PredictionState)
    outcome: OutcomeState = field(default_factory=OutcomeState)
    data_quality: DataQualityState = field(default_factory=DataQualityState)


__all__ = [
    "EarthState",
    "ObservationState",
    "PredictionState",
    "OutcomeState",
    "DataQualityState",
    "VALID_SEASONS",
]
