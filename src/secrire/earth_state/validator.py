from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.secrire.earth_state.builder import DEFAULT_OUTPUT_PATH, EARTH_STATE_DIR

REQUIRED_COLUMNS = {
    "state_id",
    "subdivision",
    "year",
    "month",
    "season",
    "temporal_position",
    "rainfall_mm",
    "rainfall_missing",
    "rainfall_3m",
    "rainfall_6m",
    "rainfall_12m",
    "rainfall_lag_1m",
    "rainfall_lag_2m",
    "rainfall_lag_3m",
    "rainfall_prev_3m",
    "rainfall_prev_6m",
    "rainfall_prev_12m",
    "historical_monthly_mean",
    "rainfall_anomaly",
    "rainfall_anomaly_pct",
    "rainfall_deficit_mm",
    "rainfall_zscore",
    "rainfall_trend_3m",
    "month_sin",
    "month_cos",
}

VALID_SEASONS = {"WINTER", "PRE_MONSOON", "MONSOON", "POST_MONSOON"}
NUMERIC_COLUMNS = [
    "year",
    "month",
    "temporal_position",
    "rainfall_mm",
    "rainfall_missing",
    "rainfall_3m",
    "rainfall_6m",
    "rainfall_12m",
    "rainfall_lag_1m",
    "rainfall_lag_2m",
    "rainfall_lag_3m",
    "rainfall_prev_3m",
    "rainfall_prev_6m",
    "rainfall_prev_12m",
    "historical_monthly_mean",
    "rainfall_anomaly",
    "rainfall_anomaly_pct",
    "rainfall_deficit_mm",
    "rainfall_zscore",
    "rainfall_trend_3m",
    "month_sin",
    "month_cos",
]


def _make_check(name: str, status: str, message: str) -> dict[str, str]:
    return {"name": name, "status": status.upper(), "message": message}


def validate_earth_state(df: pd.DataFrame) -> dict:
    """Validate the generated Earth State against the required schema and logic."""

    checks: list[dict[str, str]] = []
    issues: list[str] = []

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        checks.append(_make_check("schema", "FAIL", f"Missing required columns: {missing_columns}"))
        issues.append(f"Missing required columns: {missing_columns}")
    else:
        checks.append(_make_check("schema", "PASS", "Schema matches the required Earth State layout."))

    if "target_3m_severe_anomaly" in set(df.columns):
        checks.append(_make_check("target_leakage", "FAIL", "Target leakage present in prediction-time Earth State."))
        issues.append("target_3m_severe_anomaly is present in the Earth State feature set.")
    else:
        checks.append(_make_check("target_leakage", "PASS", "Target leakage check passed."))

    required_non_null = ["state_id", "subdivision", "year", "month", "season"]
    missing_values = [column for column in required_non_null if column in df.columns and df[column].isna().any()]
    rainfall_missing_mask = df["rainfall_missing"].astype(float).fillna(0).isin([1.0]) if "rainfall_missing" in df.columns else pd.Series(False, index=df.index)
    rainfall_mm_missing = df["rainfall_mm"].isna() if "rainfall_mm" in df.columns else pd.Series(False, index=df.index)
    invalid_rainfall_missing = rainfall_mm_missing & (~rainfall_missing_mask)
    if invalid_rainfall_missing.any():
        missing_values.append("rainfall_mm")
    if missing_values:
        checks.append(_make_check("required_fields", "FAIL", f"Missing values in: {missing_values}"))
        issues.extend(f"Missing value in {column}" for column in missing_values)
    else:
        checks.append(_make_check("required_fields", "PASS", "Required fields are populated."))

    invalid_months = df.loc[df["month"].notna(), "month"][~df.loc[df["month"].notna(), "month"].between(1, 12)]
    if not invalid_months.empty:
        checks.append(_make_check("month", "FAIL", f"Invalid month values found: {invalid_months.tolist()}"))
        issues.append("Month values must be between 1 and 12.")
    else:
        checks.append(_make_check("month", "PASS", "Month values are within the valid 1-12 range."))

    bad_season = df.loc[df["season"].notna(), "season"][~df.loc[df["season"].notna(), "season"].isin(VALID_SEASONS)]
    if not bad_season.empty:
        checks.append(_make_check("season", "FAIL", f"Unexpected season values: {bad_season.tolist()}"))
        issues.append("Season values are outside the approved set.")
    else:
        checks.append(_make_check("season", "PASS", "Season values are valid."))

    empty_subdivision = df.loc[df["subdivision"].astype(str).str.strip() == "", "subdivision"]
    if not empty_subdivision.empty:
        checks.append(_make_check("subdivision", "FAIL", "Blank subdivision names found."))
        issues.append("Subdivision names are required and cannot be blank.")
    else:
        checks.append(_make_check("subdivision", "PASS", "Subdivision values are populated."))

    duplicate_state_ids = df[df["state_id"].duplicated(keep=False)]
    if not duplicate_state_ids.empty:
        checks.append(_make_check("duplicates", "FAIL", f"Duplicate state IDs found: {duplicate_state_ids['state_id'].nunique()}"))
        issues.append("Earth State IDs are not unique.")
    else:
        checks.append(_make_check("duplicates", "PASS", "State IDs are unique."))

    chronological_issues = []
    for subdivision, group in df.groupby("subdivision", sort=False):
        ordered = group.sort_values(["year", "month"], kind="mergesort")
        if len(ordered) > 1:
            prev = ordered.iloc[0].to_dict()
            for _, current in ordered.iloc[1:].iterrows():
                current_row = current.to_dict()
                if (current_row["year"], current_row["month"]) <= (prev["year"], prev["month"]):
                    chronological_issues.append(subdivision)
                    break
                prev = current_row
    if chronological_issues:
        checks.append(_make_check("chronology", "FAIL", f"Chronological ordering issues for: {chronological_issues}"))
        issues.append("Chronological order is inconsistent within a subdivision.")
    else:
        checks.append(_make_check("chronology", "PASS", "Chronology is consistent within each subdivision."))

    numeric_violations = []
    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        non_missing = series[series.notna()]
        if not non_missing.empty and not np.isfinite(non_missing.to_numpy(dtype=float)).all():
            numeric_violations.append(column)
    if numeric_violations:
        checks.append(_make_check("numeric_values", "FAIL", f"Non-finite numeric values in: {numeric_violations}"))
        issues.append("Numeric fields contain invalid values.")
    else:
        checks.append(_make_check("numeric_values", "PASS", "Numeric fields are valid."))

    impossible_values = []
    if "rainfall_mm" in df.columns:
        if (df["rainfall_mm"] < 0).any():
            impossible_values.append("rainfall_mm < 0")
    if "rainfall_deficit_mm" in df.columns:
        if (df["rainfall_deficit_mm"] < 0).any():
            impossible_values.append("rainfall_deficit_mm < 0")
    if "rainfall_missing" in df.columns:
        if (~df["rainfall_missing"].isin([0, 1])).any():
            impossible_values.append("rainfall_missing not in {0,1}")
    if impossible_values:
        checks.append(_make_check("impossible_values", "FAIL", "; ".join(impossible_values)))
        issues.append("Impossible values were detected in the Earth State.")
    else:
        checks.append(_make_check("impossible_values", "PASS", "Observed values are within plausible ranges."))

    if any(check["status"] == "FAIL" for check in checks):
        overall_status = "FAIL"
    elif any(check["status"] == "WARN" for check in checks):
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    return {
        "status": overall_status,
        "row_count": int(len(df)),
        "checks": checks,
        "issues": issues,
    }


def write_validation_report(df: pd.DataFrame, output_dir: str | Path = EARTH_STATE_DIR) -> dict:
    """Write validation outputs to JSON and Markdown files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = validate_earth_state(df)
    json_path = output_dir / "validation_report.json"
    markdown_path = output_dir / "validation_report.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    lines = [
        "# Earth State Validation Report",
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

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate the SECRIE Earth State dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_PATH, help="Earth State CSV to validate")
    parser.add_argument("--output-dir", type=Path, default=EARTH_STATE_DIR, help="Directory to write reports")
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    write_validation_report(df, args.output_dir)
