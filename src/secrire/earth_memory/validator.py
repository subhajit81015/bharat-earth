from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.secrire.earth_memory.builder import DEFAULT_INPUT_PATH, DEFAULT_OUTPUT_PATH, EARTH_MEMORY_DIR, compute_memory_id

REQUIRED_COLUMNS = {
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
}


def validate_earth_memory(df: pd.DataFrame) -> dict:
    """Return PASS/FAIL checks for the Earth Memory artifact."""

    checks: list[dict[str, str]] = []
    issues: list[str] = []

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        checks.append({"name": "required_columns", "status": "FAIL", "message": f"Missing required columns: {missing_columns}"})
        issues.append(f"Missing required columns: {missing_columns}")
    else:
        checks.append({"name": "required_columns", "status": "PASS", "message": "Required Earth Memory columns are present."})

    bad_ids = df.loc[~df["memory_id"].apply(lambda x: str(x).startswith(str(x).split('_mem_')[0]) if '_mem_' in str(x) else True), "memory_id"]
    if not bad_ids.empty:
        checks.append({"name": "memory_id", "status": "FAIL", "message": "Memory IDs are not deterministic or not linked to source state ids."})
        issues.append("memory_id is not deterministic.")
    else:
        checks.append({"name": "memory_id", "status": "PASS", "message": "Memory IDs are deterministic."})

    duplicates = df[df["memory_id"].duplicated(keep=False)]
    if not duplicates.empty:
        checks.append({"name": "unique_memory_ids", "status": "FAIL", "message": f"Duplicate memory IDs found: {duplicates['memory_id'].nunique()}"})
        issues.append("Duplicate memory IDs were found.")
    else:
        checks.append({"name": "unique_memory_ids", "status": "PASS", "message": "Memory IDs are unique."})

    if len(df) == 0:
        checks.append({"name": "one_to_one_state_memory", "status": "FAIL", "message": "No memory rows were generated."})
        issues.append("Earth Memory has no rows.")
    else:
        checks.append({"name": "one_to_one_state_memory", "status": "PASS", "message": "Memory records were generated for the Earth State input."})

    if "target_3m_severe_anomaly" in set(df.columns):
        checks.append({"name": "target_leakage", "status": "FAIL", "message": "Target leakage present in Earth Memory output."})
        issues.append("target_3m_severe_anomaly is present in the memory artifact.")
    else:
        checks.append({"name": "target_leakage", "status": "PASS", "message": "Target leakage not present."})

    if "subdivision" in df.columns and df["subdivision"].astype(str).str.strip().eq("").any():
        checks.append({"name": "subdivision_isolation", "status": "FAIL", "message": "Blank subdivision labels found."})
        issues.append("Subdivision values are blank.")
    else:
        checks.append({"name": "subdivision_isolation", "status": "PASS", "message": "Subdivision values are populated."})

    invalid_months = df.loc[df["month"].notna(), "month"][~df.loc[df["month"].notna(), "month"].between(1, 12)]
    if not invalid_months.empty:
        checks.append({"name": "temporal_ordering", "status": "FAIL", "message": f"Invalid months: {invalid_months.tolist()}"})
        issues.append("Memory month values are invalid.")
    else:
        checks.append({"name": "temporal_ordering", "status": "PASS", "message": "Temporal month values are valid."})

    recurrence_issues = []
    if "recurrence_count" in df.columns:
        if (df["recurrence_count"] < 0).any():
            recurrence_issues.append("recurrence_count < 0")
    if "months_since_similar_state" in df.columns:
        if df["months_since_similar_state"].notna().any() and (df["months_since_similar_state"].fillna(0) < 0).any():
            recurrence_issues.append("months_since_similar_state < 0")
    if recurrence_issues:
        checks.append({"name": "recurrence_values", "status": "FAIL", "message": "; ".join(recurrence_issues)})
        issues.append("Recurrence values are invalid.")
    else:
        checks.append({"name": "recurrence_values", "status": "PASS", "message": "Recurrence values are valid."})

    similarity_issues = []
    if "similarity_score" in df.columns and (df["similarity_score"].notna() & ((df["similarity_score"] < 0) | (df["similarity_score"] > 1))).any():
        similarity_issues.append("similarity_score out of range")
    if similarity_issues:
        checks.append({"name": "similarity_values", "status": "FAIL", "message": "; ".join(similarity_issues)})
        issues.append("Similarity values are invalid.")
    else:
        checks.append({"name": "similarity_values", "status": "PASS", "message": "Similarity values are within [0, 1]."})

    missing_history_issues = []
    if "missing_history_count" in df.columns and (df["missing_history_count"] < 0).any():
        missing_history_issues.append("missing_history_count < 0")
    if missing_history_issues:
        checks.append({"name": "missing_history_handling", "status": "FAIL", "message": "; ".join(missing_history_issues)})
        issues.append("Missing history is invalid.")
    else:
        checks.append({"name": "missing_history_handling", "status": "PASS", "message": "Missing-history values are valid."})

    lineage_ok = all(bool(str(x)) for x in df["source_dataset"].dropna()) if "source_dataset" in df.columns else False
    if lineage_ok:
        checks.append({"name": "source_lineage", "status": "PASS", "message": "Source lineage metadata is present."})
    else:
        checks.append({"name": "source_lineage", "status": "FAIL", "message": "Source lineage metadata is missing."})
        issues.append("Source lineage metadata is missing.")

    if any(check["status"] == "FAIL" for check in checks):
        overall_status = "FAIL"
    else:
        overall_status = "PASS"

    return {
        "status": overall_status,
        "row_count": int(len(df)),
        "checks": checks,
        "issues": issues,
    }


def write_validation_report(df: pd.DataFrame, output_dir: str | Path = EARTH_MEMORY_DIR) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = validate_earth_memory(df)
    json_path = output_dir / "validation_report.json"
    md_path = output_dir / "validation_report.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    lines = [
        "# Earth Memory Validation Report",
        "",
        f"Status: {result['status']}",
        f"Rows: {result['row_count']}",
        "",
        "## Checks",
        "",
    ]
    for check in result["checks"]:
        lines.append(f"- {check['name']}: {check['status']} - {check['message']}")
    if result["issues"]:
        lines.append("")
        lines.append("## Issues")
        lines.append("")
        for issue in result["issues"]:
            lines.append(f"- {issue}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate the Earth Memory artifact.")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_PATH, help="Earth Memory CSV to validate")
    parser.add_argument("--output-dir", type=Path, default=EARTH_MEMORY_DIR, help="Directory to write validation reports")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    write_validation_report(df, args.output_dir)
