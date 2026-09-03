from pathlib import Path

import pandas as pd

from src.secrire.earth_state.builder import build_earth_state, compute_state_id
from src.secrire.earth_state.schema import (
    DataQualityState,
    EarthState,
    ObservationState,
    OutcomeState,
    PredictionState,
)
from src.secrire.earth_state.transition import build_state_transitions
from src.secrire.earth_state.validator import validate_earth_state

SOURCE_PATH = Path("data/features/ml_dataset_v4.csv")


def test_valid_earth_state():
    df = build_earth_state(SOURCE_PATH)
    result = validate_earth_state(df)
    assert result["status"] == "PASS"


def test_invalid_month():
    df = build_earth_state(SOURCE_PATH)
    df.loc[0, "month"] = 13
    result = validate_earth_state(df)
    assert result["status"] in {"WARN", "FAIL"}


def test_invalid_season():
    df = build_earth_state(SOURCE_PATH)
    df.loc[0, "season"] = "SPRING"
    result = validate_earth_state(df)
    assert result["status"] in {"WARN", "FAIL"}


def test_missing_required_field():
    df = build_earth_state(SOURCE_PATH)
    df = df.drop(columns=["rainfall_mm"])
    result = validate_earth_state(df)
    assert result["status"] in {"WARN", "FAIL"}


def test_duplicate_state():
    df = build_earth_state(SOURCE_PATH)
    df.loc[1, "state_id"] = df.loc[0, "state_id"]
    result = validate_earth_state(df)
    assert result["status"] in {"WARN", "FAIL"}


def test_invalid_numeric_value():
    df = build_earth_state(SOURCE_PATH)
    df.loc[0, "rainfall_anomaly"] = float("nan")
    result = validate_earth_state(df)
    assert result["status"] in {"WARN", "FAIL"}


def test_deterministic_state_id():
    a = compute_state_id("Andaman & Nicobar Islands", 1901, 4)
    b = compute_state_id("Andaman & Nicobar Islands", 1901, 4)
    assert a == b
    assert len(a) > 8


def test_temporal_ordering():
    df = build_earth_state(SOURCE_PATH)
    for _, group in df.groupby("subdivision"):
        ordered = group.sort_values(["year", "month"]).reset_index(drop=True)
        assert ordered["year"].is_monotonic_increasing
        assert ordered["month"].notna().all()


def test_transition_calculation():
    df = build_earth_state(SOURCE_PATH)
    transitions = build_state_transitions(df)
    assert not transitions.empty
    required = {
        "subdivision",
        "previous_state_id",
        "state_id",
        "rainfall_change",
        "anomaly_change",
        "deficit_change",
        "rainfall_trend_change",
        "rainfall_3m_change",
        "rainfall_6m_change",
        "persistence",
        "volatility",
    }
    assert required.issubset(transitions.columns)


def test_target_leakage_prevention():
    df = build_earth_state(SOURCE_PATH)
    assert "target_3m_severe_anomaly" not in df.columns
    result = validate_earth_state(df)
    assert result["status"] == "PASS"


def test_prediction_outcome_separation():
    prediction = PredictionState()
    outcome = OutcomeState(target_3m_severe_anomaly=1)
    assert prediction.features == {}
    assert outcome.target_3m_severe_anomaly == 1


def test_builder_output_schema():
    df = build_earth_state(SOURCE_PATH)
    expected = {
        "state_id",
        "subdivision",
        "year",
        "month",
        "season",
        "temporal_position",
        "rainfall_mm",
        "rainfall_missing",
    }
    assert expected.issubset(df.columns)
    assert df["subdivision"].notna().all()
    assert df["month"].between(1, 12).all()


def test_schema_models():
    observation = ObservationState(
        state_id="obs-1",
        subdivision="Test",
        year=2024,
        month=6,
        season="MONSOON",
        rainfall_mm=100.0,
    )
    earth = EarthState(
        observation=observation,
        prediction=PredictionState(),
        outcome=OutcomeState(target_3m_severe_anomaly=0),
        data_quality=DataQualityState(validation_status="PASS"),
    )
    assert earth.observation.state_id == "obs-1"
    assert earth.prediction.features == {}
    assert earth.outcome.target_3m_severe_anomaly == 0


if __name__ == "__main__":
    for name in [
        "test_valid_earth_state",
        "test_invalid_month",
        "test_invalid_season",
        "test_missing_required_field",
        "test_duplicate_state",
        "test_invalid_numeric_value",
        "test_deterministic_state_id",
        "test_temporal_ordering",
        "test_transition_calculation",
        "test_target_leakage_prevention",
        "test_prediction_outcome_separation",
        "test_builder_output_schema",
        "test_schema_models",
    ]:
        globals()[name]()
        print(f"PASS: {name}")
