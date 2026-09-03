from pathlib import Path

import pandas as pd

from src.secrire.earth_memory.builder import build_earth_memory, compute_memory_id
from src.secrire.earth_memory.schema import EarthMemory, MEMORY_VERSION
from src.secrire.earth_memory.similarity import similarity_score
from src.secrire.earth_memory.validator import validate_earth_memory

SOURCE_PATH = Path("data/features/earth_state_v1/earth_state.csv")


def test_schema_validity():
    df = build_earth_memory(SOURCE_PATH)
    assert {"memory_id", "state_id", "subdivision", "year", "month", "season"}.issubset(df.columns)


def test_deterministic_memory_ids():
    a = compute_memory_id("state_123", "Andaman & Nicobar Islands", 1901, 2)
    b = compute_memory_id("state_123", "Andaman & Nicobar Islands", 1901, 2)
    assert a == b
    assert a.startswith("andaman_nicobar_islands_1901_02")


def test_one_to_one_state_memory():
    state_df = pd.read_csv(SOURCE_PATH)
    mem_df = build_earth_memory(SOURCE_PATH)
    assert len(mem_df) == len(state_df)
    assert mem_df["state_id"].nunique() == len(state_df)


def test_subdivision_isolation():
    mem_df = build_earth_memory(SOURCE_PATH)
    grouped = mem_df.groupby("subdivision")
    for _, group in grouped:
        assert group["state_id"].nunique() == len(group)


def test_no_future_leakage():
    mem_df = build_earth_memory(SOURCE_PATH)
    assert "target_3m_severe_anomaly" not in mem_df.columns


def test_target_leakage_protection():
    mem_df = build_earth_memory(SOURCE_PATH)
    validation = validate_earth_memory(mem_df)
    assert validation["status"] == "PASS"


def test_first_observation_history_handling():
    mem_df = build_earth_memory(SOURCE_PATH)
    first_group = mem_df[mem_df["state_id"].str.endswith("_01")]
    assert first_group["historical_state_count"].ge(0).all()


def test_recurrence_calculation():
    mem_df = build_earth_memory(SOURCE_PATH)
    assert "recurrence_count" in mem_df.columns
    assert mem_df["recurrence_count"].between(0, 1000).all()


def test_similarity_calculation():
    current = {"rainfall_mm": 100.0, "rainfall_anomaly": 20.0, "rainfall_deficit_mm": 0.0, "rainfall_trend_3m": 1.0, "rainfall_3m": 95.0, "rainfall_6m": 200.0, "rainfall_12m": 240.0, "rainfall_zscore": 0.2}
    prior = {"rainfall_mm": 95.0, "rainfall_anomaly": 18.0, "rainfall_deficit_mm": 0.0, "rainfall_trend_3m": 0.5, "rainfall_3m": 90.0, "rainfall_6m": 210.0, "rainfall_12m": 250.0, "rainfall_zscore": 0.1}
    score = similarity_score(current, prior)
    assert 0.0 <= score <= 1.0


def test_missing_value_handling():
    mem_df = build_earth_memory(SOURCE_PATH)
    assert mem_df["missing_history_count"].ge(0).all()
    assert mem_df["memory_complete"].isin([True, False]).all()


def test_deterministic_repeated_execution():
    a = build_earth_memory(SOURCE_PATH)
    b = build_earth_memory(SOURCE_PATH)
    pd.testing.assert_frame_equal(a, b)


def test_validator_pass_on_valid_output():
    df = build_earth_memory(SOURCE_PATH)
    result = validate_earth_memory(df)
    assert result["status"] == "PASS"


if __name__ == "__main__":
    for name in [
        "test_schema_validity",
        "test_deterministic_memory_ids",
        "test_one_to_one_state_memory",
        "test_subdivision_isolation",
        "test_no_future_leakage",
        "test_target_leakage_protection",
        "test_first_observation_history_handling",
        "test_recurrence_calculation",
        "test_similarity_calculation",
        "test_missing_value_handling",
        "test_deterministic_repeated_execution",
        "test_validator_pass_on_valid_output",
    ]:
        globals()[name]()
        print(f"PASS: {name}")
