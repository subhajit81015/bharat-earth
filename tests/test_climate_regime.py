from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.secrire.climate_regime.assignment import assign_regimes
from src.secrire.climate_regime.discovery import discover_regimes, robust_normalize
from src.secrire.climate_regime.features import TARGET_COLUMN, build_regime_features
from src.secrire.climate_regime.stability import calculate_stability
from src.secrire.climate_regime.transition import build_transition_records, summarize_transitions
from src.secrire.climate_regime.validator import validate_climate_regime

STATE_PATH = Path("data/features/earth_state_v1/earth_state.csv")
MEMORY_PATH = Path("data/features/earth_memory_v1/earth_memory.csv")


def _pipeline():
    state = pd.read_csv(STATE_PATH).groupby("subdivision", sort=False).head(36).reset_index(drop=True)
    memory = pd.read_csv(MEMORY_PATH)
    memory = memory[memory.state_id.isin(state.state_id)].reset_index(drop=True)
    features = build_regime_features(state, memory)
    discovery = discover_regimes(features.frame, features.selected_features)
    assignments = assign_regimes(features.frame, discovery)
    return state, features, discovery, assignments


def test_schema_and_feature_selection():
    _, features, _, assignments = _pipeline()
    assert len(assignments) > 0
    assert {"state_id", "subdivision", "year", "month", "regime_id"}.issubset(assignments.columns)
    assert TARGET_COLUMN not in features.selected_features
    assert TARGET_COLUMN not in features.frame.columns


def test_deterministic_normalization_and_discovery():
    _, features, first, _ = _pipeline()
    second = discover_regimes(features.frame, features.selected_features)
    normalized = robust_normalize(features.frame, features.selected_features)
    assert normalized.medians == first.normalized.medians
    assert normalized.scales == first.normalized.scales
    assert first.selected_k == second.selected_k
    assert (first.labels == second.labels).all()
    assert np.allclose(first.centroids, second.centroids)


def test_assignment_and_subdivision_preservation():
    state, _, _, assignments = _pipeline()
    assert len(assignments) == len(state)
    assert assignments.state_id.nunique() == len(state)
    assert assignments.subdivision.nunique() == state.subdivision.nunique()
    assert assignments.regime_id.astype(str).str.match(r"^REGIME_[0-9]{2}$").all()


def test_persistence_and_transitions_are_temporal():
    _, _, _, assignments = _pipeline()
    records = build_transition_records(assignments)
    summary = summarize_transitions(assignments)
    assert len(records) == len(assignments) - assignments.subdivision.nunique()
    assert records.previous_state_id.notna().all()
    assert records.current_state_id.notna().all()
    assert summary.transition_probability.between(0, 1).all()
    first_rows = assignments.groupby("subdivision", sort=False).head(1)
    assert first_rows.previous_state_id.isna().all()


def test_stability_and_missing_value_handling():
    _, features, discovery, _ = _pipeline()
    rerun = discover_regimes(features.frame, features.selected_features)
    stability = calculate_stability(
        discovery.labels, discovery.centroids, discovery.normalized.values, rerun.labels
    )
    assert len(stability) == discovery.selected_k
    assert stability.stability_score.between(0, 1).all()
    assert not pd.isna(discovery.normalized.values).any()


def test_validator_pass():
    state, features, _, assignments = _pipeline()
    transitions = build_transition_records(assignments)
    result = validate_climate_regime(
        assignments, transitions, features.selected_features, state.state_id
    )
    assert result["status"] == "PASS"


if __name__ == "__main__":
    for name in [
        "test_schema_and_feature_selection",
        "test_deterministic_normalization_and_discovery",
        "test_assignment_and_subdivision_preservation",
        "test_persistence_and_transitions_are_temporal",
        "test_stability_and_missing_value_handling",
        "test_validator_pass",
    ]:
        globals()[name]()
        print(f"PASS: {name}")
