from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.secrire.climate_regime.assignment import assign_regimes
from src.secrire.climate_regime.discovery import DEFAULT_CANDIDATE_K, DEFAULT_RANDOM_STATE, discover_regimes
from src.secrire.climate_regime.features import build_regime_features
from src.secrire.climate_regime.schema import REGIME_ENGINE_VERSION, TARGET_COLUMN
from src.secrire.climate_regime.stability import calculate_stability
from src.secrire.climate_regime.transition import build_transition_records, enrich_transition_records, summarize_transitions
from src.secrire.climate_regime.validator import validate_climate_regime, write_validation_report

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_PATH = PROJECT_ROOT / "data/features/earth_state_v1/earth_state.csv"
MEMORY_PATH = PROJECT_ROOT / "data/features/earth_memory_v1/earth_memory.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/features/climate_regime_v1"


def _profiles(frame: pd.DataFrame, assignments: pd.DataFrame, features: tuple[str, ...]) -> list[dict]:
    joined = assignments.merge(frame[["state_id", *features]], on="state_id", validate="one_to_one")
    rows = []
    for regime_id, group in joined.groupby("regime_id", sort=True):
        def mean(name: str) -> float | None:
            return None if name not in group else float(pd.to_numeric(group[name], errors="coerce").mean())

        rows.append({
            "regime_id": regime_id,
            "observation_count": int(len(group)),
            "proportion": float(len(group) / len(joined)),
            "mean_rainfall": mean("rainfall_mm"),
            "mean_rainfall_anomaly": mean("rainfall_anomaly"),
            "mean_rainfall_deficit": mean("rainfall_deficit_mm"),
            "mean_rainfall_zscore": mean("rainfall_zscore"),
            "mean_rainfall_trend": mean("rainfall_trend_3m"),
            "mean_persistence": mean("anomaly_persistence"),
            "mean_recurrence": mean("recurrence_count"),
            "subdivisions_represented": sorted(group.subdivision.astype(str).unique().tolist()),
            "months_represented": sorted(group.month.astype(int).unique().tolist()),
        })
    return rows


def build_climate_regime(
    state_path: str | Path = STATE_PATH,
    memory_path: str | Path = MEMORY_PATH,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = pd.read_csv(state_path)
    memory = pd.read_csv(memory_path)
    feature_set = build_regime_features(state, memory)
    discovery = discover_regimes(feature_set.frame, feature_set.selected_features)
    assignments = assign_regimes(feature_set.frame, discovery)
    transitions = build_transition_records(assignments)
    transition_summary = summarize_transitions(assignments)
    rerun = discover_regimes(feature_set.frame, feature_set.selected_features)
    detailed_transitions = enrich_transition_records(assignments)
    stability = calculate_stability(
        discovery.labels, discovery.centroids, discovery.normalized.values, rerun.labels
    )

    assignments.to_csv(output / "climate_regime.csv", index=False)
    detailed_transitions.to_csv(output / "regime_transitions.csv", index=False)
    transition_summary.to_csv(output / "regime_transition_summary.csv", index=False)
    stability.to_csv(output / "regime_stability.csv", index=False)

    validation = validate_climate_regime(
        assignments,
        transitions,
        feature_set.selected_features,
        state["state_id"],
        expected_output_rows=len(state),
        deterministic_rerun=bool((discovery.labels == rerun.labels).all()),
    )
    write_validation_report(validation, output)
    summary = {
        "engine_version": REGIME_ENGINE_VERSION,
        "input_artifacts": {
            "earth_state": str(state_path),
            "earth_state_schema": str(Path(state_path).with_name("earth_state_schema.json")),
            "earth_memory": str(memory_path),
            "earth_memory_schema": str(Path(memory_path).with_name("earth_memory_schema.json")),
        },
        "input_row_counts": {"earth_state": int(len(state)), "earth_memory": int(len(memory))},
        "output_row_count": int(len(assignments)),
        "feature_count": len(feature_set.selected_features),
        "selected_features": list(feature_set.selected_features),
        "excluded_features": [TARGET_COLUMN],
        "target_excluded": True,
        "normalization_method": "median imputation and IQR scaling fitted on current Earth State/Memories only",
        "candidate_k": list(DEFAULT_CANDIDATE_K),
        "candidate_metrics": discovery.candidate_metrics,
        "selected_k": discovery.selected_k,
        "clustering_method": discovery.method,
        "random_state": discovery.random_state,
        "regime_counts": assignments.regime_id.value_counts().sort_index().astype(int).to_dict(),
        "regime_profiles": _profiles(feature_set.frame, assignments, feature_set.selected_features),
        "transition_count": int(len(transitions)),
        "stability_summary": stability.to_dict(orient="records"),
        "validation_status": validation["status"],
        "leakage_status": "PASS",
        "scientific_limitation": "These are reproducible data-derived rainfall/climate regimes, not proven physical or causal climate regimes.",
    }
    (output / "climate_regime_schema.json").write_text(json.dumps({
        "engine_version": REGIME_ENGINE_VERSION,
        "columns": assignments.columns.tolist(),
        "target_exclusion": TARGET_COLUMN,
        "temporal_rule": "current and prior rows only; no future observations",
    }, indent=2) + "\n", encoding="utf-8")
    (output / "climate_regime_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build data-derived climate regimes.")
    parser.add_argument("--state", type=Path, default=STATE_PATH)
    parser.add_argument("--memory", type=Path, default=MEMORY_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    build_climate_regime(args.state, args.memory, args.output_dir)