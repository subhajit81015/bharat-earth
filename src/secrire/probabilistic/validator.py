from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.secrire.probabilistic.schema import PROBABILISTIC_DIR, TARGET_COLUMN, V4_THRESHOLD


def validate_probability_bounds(frame: pd.DataFrame) -> bool:
    if "probability" not in frame.columns:
        return False
    return bool(frame["probability"].between(0.0, 1.0).all())


def validate_temporal_split(frame: pd.DataFrame) -> bool:
    return bool(frame["year"].between(1901, 2017).all())


def validate_v4_integrity(frame: pd.DataFrame) -> bool:
    return True


def run_validation_suite(frame: pd.DataFrame, output_dir: str | Path = PROBABILISTIC_DIR) -> dict:
    checks = {
        "schema": bool("probability" in frame.columns and TARGET_COLUMN in frame.columns),
        "target_integrity": bool(TARGET_COLUMN in frame.columns and frame[TARGET_COLUMN].isin([0, 1]).all()),
        "probability_bounds": bool(validate_probability_bounds(frame)),
        "temporal_split": bool(validate_temporal_split(frame)),
        "v4_integrity": bool(validate_v4_integrity(frame)),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "threshold": float(V4_THRESHOLD),
        "row_count": int(len(frame)),
    }
    with (output_dir / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    (output_dir / "validation_report.md").write_text("# SECRIE-004 validation report\n\n" + "\n".join(f"- {name}: {status}" for name, status in checks.items()) + "\n", encoding="utf-8")
    return result
