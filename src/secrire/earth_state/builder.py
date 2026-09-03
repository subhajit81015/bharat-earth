from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_PATH = PROJECT_ROOT / "data" / "features" / "ml_dataset_v4.csv"
EARTH_STATE_DIR = PROJECT_ROOT / "data" / "features" / "earth_state_v1"
DEFAULT_OUTPUT_PATH = EARTH_STATE_DIR / "earth_state.csv"
SUMMARY_PATH = EARTH_STATE_DIR / "earth_state_summary.json"

SOURCE_REQUIRED_COLUMNS = {
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
    "target_3m_severe_anomaly",
    "rainfall_lag_1m",
    "rainfall_lag_2m",
    "rainfall_lag_3m",
    "rainfall_prev_3m",
    "rainfall_prev_6m",
    "rainfall_prev_12m",
    "rainfall_trend_3m",
    "month_sin",
    "month_cos",
}

OUTPUT_COLUMNS = [
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
]

MONTH_MAP = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}


def compute_state_id(subdivision: str, year: int, month: int) -> str:
    """Create a deterministic Earth State identifier from stable metadata."""

    safe_subdivision = re.sub(r"[^a-z0-9]+", "_", str(subdivision).strip().lower()).strip("_")
    return f"{safe_subdivision}_{int(year)}_{int(month):02d}"


def _normalize_month(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("Month value is missing.")

    raw = str(value).strip().upper()
    if raw.isdigit():
        month = int(raw)
        if 1 <= month <= 12:
            return month
        raise ValueError(f"Invalid month value: {value}")

    if raw in MONTH_MAP:
        return MONTH_MAP[raw]

    raise ValueError(f"Unsupported month format: {value!r}")


def _write_summary(source_path: Path, source_df: pd.DataFrame, output_df: pd.DataFrame) -> dict[str, Any]:
    EarthStateVersion = "v1"
    summary = {
        "system_version": f"Python {__import__('platform').python_version()}",
        "earth_state_version": EarthStateVersion,
        "source_dataset": str(source_path),
        "source_row_count": int(len(source_df)),
        "output_row_count": int(len(output_df)),
        "source_schema": list(source_df.columns),
        "output_schema": list(output_df.columns),
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_configuration_version": "not_available",
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def build_earth_state(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Construct a prediction-time Earth State table from the V4 rainfall dataset.

    The source data is read without modification and copied into a dedicated
    decomposition of the live state representation. The target outcome is excluded
    from the output to enforce prediction-time separation.
    """

    source_file = Path(source_path)
    output_file = Path(output_path)

    if not source_file.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_file}")

    source_df = pd.read_csv(source_file)
    missing_columns = sorted(SOURCE_REQUIRED_COLUMNS - set(source_df.columns))
    if missing_columns:
        raise ValueError(f"Missing source columns: {missing_columns}")

    normalized = source_df.copy()
    normalized["subdivision"] = normalized["subdivision"].astype(str).str.strip()
    normalized["season"] = normalized["season"].astype(str).str.strip().str.upper()
    normalized["year"] = pd.to_numeric(normalized["year"], errors="raise").astype(int)
    normalized["month"] = normalized["month"].map(_normalize_month)
    normalized["state_id"] = normalized.apply(
        lambda row: compute_state_id(row["subdivision"], row["year"], row["month"]),
        axis=1,
    )

    duplicate_ids = normalized[normalized["state_id"].duplicated(keep=False)]
    if not duplicate_ids.empty:
        # The deterministic ID is stable; duplicates here indicate a source-data issue.
        raise ValueError(f"Duplicate state IDs detected in source data: {duplicate_ids['state_id'].nunique()} unique duplicates")

    ordered = normalized.sort_values(["subdivision", "year", "month"], kind="mergesort").copy()
    ordered["temporal_position"] = ordered.groupby("subdivision").cumcount()

    output_df = ordered[
        [
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
        ]
    ].copy()

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_file, index=False)
    _write_summary(source_file, source_df, output_df)
    return output_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SECRIE Earth State dataset.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE_PATH, help="Path to the V4 source CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path for the Earth State CSV")
    args = parser.parse_args()
    build_earth_state(args.source, args.output)


if __name__ == "__main__":
    main()
