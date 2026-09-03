from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROBABILISTIC_DIR = PROJECT_ROOT / "data" / "features" / "probabilistic_v1"
MODELS_DIR = PROBABILISTIC_DIR / "models"

V4_DATA_PATH = PROJECT_ROOT / "data" / "features" / "ml_dataset_v4.csv"
V4_CALIBRATED_PATH = PROJECT_ROOT / "data" / "features" / "calibrated_predictions.csv"
V4_TEST_PREDICTIONS_PATH = PROJECT_ROOT / "data" / "features" / "model_v4" / "test_predictions.csv"
EARTH_STATE_PATH = PROJECT_ROOT / "data" / "features" / "earth_state_v1" / "earth_state.csv"
EARTH_MEMORY_PATH = PROJECT_ROOT / "data" / "features" / "earth_memory_v1" / "earth_memory.csv"
CLIMATE_REGIME_PATH = PROJECT_ROOT / "data" / "features" / "climate_regime_v1" / "climate_regime.csv"
TARGET_PATH = PROJECT_ROOT / "data" / "features" / "severe_anomaly_target.csv"

TARGET_COLUMN = "target_3m_severe_anomaly"
V4_THRESHOLD = 0.09
RANDOM_SEED = 42

BASELINE_FEATURES = [
    "subdivision",
    "year",
    "month",
    "season",
    "rainfall_mm",
    "rainfall_3m",
    "rainfall_6m",
    "rainfall_12m",
    "historical_monthly_mean",
    "rainfall_anomaly",
    "rainfall_anomaly_pct",
    "rainfall_deficit_mm",
    "rainfall_missing",
    "rainfall_zscore",
    "rainfall_lag_1m",
    "rainfall_lag_2m",
    "rainfall_lag_3m",
    "rainfall_prev_3m",
    "rainfall_prev_6m",
    "rainfall_prev_12m",
    "rainfall_trend_3m",
    "month_sin",
    "month_cos",
]

REGIME_DISCOVERY_FEATURES = [
    "rainfall_mm",
    "rainfall_3m",
    "rainfall_6m",
    "rainfall_12m",
    "historical_monthly_mean",
    "rainfall_anomaly",
    "rainfall_anomaly_pct",
    "rainfall_deficit_mm",
    "rainfall_zscore",
    "rainfall_lag_1m",
    "rainfall_lag_2m",
    "rainfall_lag_3m",
    "rainfall_prev_3m",
    "rainfall_prev_6m",
    "rainfall_prev_12m",
    "rainfall_trend_3m",
    "month_sin",
    "month_cos",
]

REGIME_CONTEXT_FEATURES = [
    "regime_id",
    "previous_regime_id",
    "regime_changed",
    "regime_duration",
    "consecutive_months_in_current_regime",
]

MEMORY_CONTEXT_FEATURES = [
    "anomaly_persistence",
    "deficit_persistence",
    "rainfall_condition_persistence",
    "consecutive_anomaly_direction",
    "recurrence_count",
    "months_since_similar_state",
    "historical_similarity_count",
    "similarity_score",
    "memory_complete",
    "history_sufficient",
    "missing_history_count",
    "memory_quality_status",
]

CATEGORY_COLUMNS = ["subdivision", "month", "season", "regime_id", "previous_regime_id"]

TRAIN_YEARS = (1901, 2013)
VALIDATION_YEARS = (2014, 2015)
TEST_YEARS = (2016, 2017)

PROTECTED_V4_FILES = [
    "src/ml",
    "data/features/ml_dataset_v4.csv",
    "data/features/model_v4",
    "data/features/calibration_v4",
    "data/features/policy_v4",
    "data/features/monitoring_v4",
    "data/features/production_readiness_v4",
    "data/features/inference_v4",
]
