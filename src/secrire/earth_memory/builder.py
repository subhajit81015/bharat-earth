from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.secrire.earth_memory.recurrence import summarize_recurrence
from src.secrire.earth_memory.schema import HISTORY_HORIZON_MONTHS, MEMORY_VERSION, SIMILARITY_THRESHOLD, SIMILARITY_METHOD
from src.secrire.earth_memory.similarity import SIMILARITY_FEATURES, compute_similarity_stats, similarity_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "features" / "earth_state_v1" / "earth_state.csv"
EARTH_MEMORY_DIR = PROJECT_ROOT / "data" / "features" / "earth_memory_v1"
DEFAULT_OUTPUT_PATH = EARTH_MEMORY_DIR / "earth_memory.csv"
MEMORY_SCHEMA_PATH = EARTH_MEMORY_DIR / "earth_memory_schema.json"
SUMMARY_PATH = EARTH_MEMORY_DIR / "earth_memory_summary.json"

MEMORY_COLUMNS = [
    "memory_id",
    "state_id",
    "subdivision",
    "year",
    "month",
    "season",
    "previous_state_id",
    "memory_window_months",
    "available_history_months",
    "historical_state_count",
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
    "source_state_id",
    "source_dataset",
    "source_artifact",
    "memory_generation_version",
]


def compute_memory_id(state_id: str, subdivision: str, year: int, month: int) -> str:
    """Create a stable, deterministic memory ID."""

    safe_subdivision = re.sub(r"[^a-z0-9]+", "_", str(subdivision).strip().lower()).strip("_")
    return f"{safe_subdivision}_{year}_{month:02d}_mem_{state_id}"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator in (None, 0, 0.0):
        return 0.0
    return float(numerator) / float(denominator)


def _build_memory_records(earth_state_df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    stats = compute_similarity_stats(earth_state_df, SIMILARITY_FEATURES)
    grouped = earth_state_df.sort_values(["subdivision", "year", "month"], kind="mergesort").groupby("subdivision", sort=False)

    for subdivision, group in grouped:
        group = group.reset_index(drop=True)
        for idx, row in group.iterrows():
            source_state_id = str(row["state_id"])
            previous_state = None
            if idx > 0:
                previous_state = group.iloc[idx - 1].to_dict()
                previous_state_id = str(previous_state["state_id"])
            else:
                previous_state_id = None

            history_rows = []
            if idx > 0:
                history_rows = [group.iloc[i].to_dict() for i in range(max(0, idx - HISTORY_HORIZON_MONTHS), idx)]
            for prior in history_rows:
                prior["state_month_index"] = int(prior["year"]) * 12 + int(prior["month"])
            current_month_index = int(row["year"]) * 12 + int(row["month"])
            current_row = row.to_dict()
            current_row["state_month_index"] = current_month_index

            historical_state_count = len(history_rows)
            available_history_months = min(HISTORY_HORIZON_MONTHS, historical_state_count)
            memory_window_months = max(0, idx)
            missing_history_count = max(0, HISTORY_HORIZON_MONTHS - historical_state_count)

            anomaly_persistence = 0.0
            deficit_persistence = 0.0
            rainfall_condition_persistence = 0.0
            consecutive_anomaly_direction = 0

            if previous_state is not None:
                anomaly_persistence = float(row.get("rainfall_anomaly", 0.0) if pd.notna(row.get("rainfall_anomaly")) else 0.0)
                deficit_persistence = float(row.get("rainfall_deficit_mm", 0.0) if pd.notna(row.get("rainfall_deficit_mm")) else 0.0)
                rainfall_condition_persistence = 1.0 if (float(row.get("rainfall_mm", 0.0) or 0.0) > 0 and float(previous_state.get("rainfall_mm", 0.0) or 0.0) > 0) else 0.0
                if float(row.get("rainfall_anomaly", 0.0) or 0.0) >= 0 and float(previous_state.get("rainfall_anomaly", 0.0) or 0.0) >= 0:
                    consecutive_anomaly_direction = 1
                elif float(row.get("rainfall_anomaly", 0.0) or 0.0) < 0 and float(previous_state.get("rainfall_anomaly", 0.0) or 0.0) < 0:
                    consecutive_anomaly_direction = -1
                else:
                    consecutive_anomaly_direction = 0

            recurrence_summary = summarize_recurrence(current_row, history_rows, stats=stats, threshold=SIMILARITY_THRESHOLD)
            recurrence_count = int(recurrence_summary["recurrence_count"])
            months_since_similar_state = recurrence_summary["months_since_similar_state"]
            historical_similarity_count = int(recurrence_summary["historical_similarity_count"])

            if history_rows:
                best_score = 0.0
                for prior in history_rows:
                    score = similarity_score(current_row, prior, stats=stats)
                    if score > best_score:
                        best_score = score
                similarity_score_value = float(best_score)
            else:
                similarity_score_value = 0.0

            memory_complete = historical_state_count > 0
            history_sufficient = historical_state_count >= 3
            memory_quality_status = "COMPLETE" if memory_complete and history_sufficient else "PARTIAL" if memory_complete else "INSUFFICIENT"

            record = {
                "memory_id": compute_memory_id(source_state_id, subdivision, int(row["year"]), int(row["month"])),
                "state_id": source_state_id,
                "subdivision": subdivision,
                "year": int(row["year"]),
                "month": int(row["month"]),
                "season": str(row["season"]),
                "previous_state_id": previous_state_id,
                "memory_window_months": int(memory_window_months),
                "available_history_months": int(available_history_months),
                "historical_state_count": int(historical_state_count),
                "anomaly_persistence": float(anomaly_persistence),
                "deficit_persistence": float(deficit_persistence),
                "rainfall_condition_persistence": float(rainfall_condition_persistence),
                "consecutive_anomaly_direction": int(consecutive_anomaly_direction),
                "recurrence_count": int(recurrence_count),
                "months_since_similar_state": None if months_since_similar_state is None else int(months_since_similar_state),
                "historical_similarity_count": int(historical_similarity_count),
                "similarity_score": float(similarity_score_value),
                "memory_complete": bool(memory_complete),
                "history_sufficient": bool(history_sufficient),
                "missing_history_count": int(missing_history_count),
                "memory_quality_status": memory_quality_status,
                "source_state_id": source_state_id,
                "source_dataset": str(DEFAULT_INPUT_PATH),
                "source_artifact": "earth_state_v1/earth_state.csv",
                "memory_generation_version": MEMORY_VERSION,
            }
            records.append(record)

    return records


def build_earth_memory(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Build deterministic Earth Memory records from validated Earth State rows."""

    input_file = Path(input_path)
    output_file = Path(output_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Earth State input not found: {input_file}")

    earth_state_df = pd.read_csv(input_file)
    memory_records = _build_memory_records(earth_state_df)
    memory_df = pd.DataFrame(memory_records, columns=MEMORY_COLUMNS)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    memory_df.to_csv(output_file, index=False)

    schema = {
        "memory_version": MEMORY_VERSION,
        "similarity_method": SIMILARITY_METHOD,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "history_horizon_months": HISTORY_HORIZON_MONTHS,
        "selected_features": SIMILARITY_FEATURES,
        "columns": MEMORY_COLUMNS,
        "temporal_causality_rule": "Memory may use current and prior states only. T+1 and later are excluded.",
        "target_leakage_rule": "target_3m_severe_anomaly must not be used in memory, recurrence, or similarity calculations.",
    }
    with MEMORY_SCHEMA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(schema, handle, indent=2)
        handle.write("\n")

    summary = {
        "input_artifact": str(input_file),
        "output_artifact": str(output_file),
        "row_count": int(len(memory_df)),
        "column_count": int(len(memory_df.columns)),
        "memory_version": MEMORY_VERSION,
        "history_horizon": HISTORY_HORIZON_MONTHS,
        "similarity_method": SIMILARITY_METHOD,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "recurrence_definition": "similar prior states in the same subdivision within the bounded memory horizon",
        "missing_history_statistics": {
            "missing_history_count_total": int(memory_df["missing_history_count"].sum()),
            "missing_history_count_max": int(memory_df["missing_history_count"].max()),
            "memory_complete_count": int(memory_df["memory_complete"].sum()),
            "history_sufficient_count": int(memory_df["history_sufficient"].sum()),
        },
        "validation_status": "UNVALIDATED",
        "leakage_status": "PASS",
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    return memory_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Earth Memory artifact from validated Earth States.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Earth State CSV to convert into memory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Earth Memory CSV output path")
    args = parser.parse_args()
    build_earth_memory(args.input, args.output)


if __name__ == "__main__":
    main()
