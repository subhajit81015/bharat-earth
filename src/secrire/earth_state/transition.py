from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.secrire.earth_state.builder import DEFAULT_OUTPUT_PATH, EARTH_STATE_DIR

REQUIRED_COLUMNS = {
    "state_id",
    "subdivision",
    "year",
    "month",
    "rainfall_mm",
    "rainfall_anomaly",
    "rainfall_deficit_mm",
    "rainfall_trend_3m",
    "rainfall_3m",
    "rainfall_6m",
}


def build_state_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute state-to-state changes for each subdivision."""

    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Transition input missing required columns: {missing}")

    ordered = df.copy()
    ordered = ordered.sort_values(["subdivision", "year", "month"], kind="mergesort").reset_index(drop=True)

    rows: list[dict[str, float | str | int | None]] = []
    for subdivision, group in ordered.groupby("subdivision", sort=False):
        for previous, current in zip(group.iloc[:-1].itertuples(index=False), group.iloc[1:].itertuples(index=False)):
            previous_rainfall = getattr(previous, "rainfall_mm", 0.0) or 0.0
            current_rainfall = getattr(current, "rainfall_mm", 0.0) or 0.0
            previous_anomaly = getattr(previous, "rainfall_anomaly", 0.0) or 0.0
            current_anomaly = getattr(current, "rainfall_anomaly", 0.0) or 0.0
            previous_deficit = getattr(previous, "rainfall_deficit_mm", 0.0) or 0.0
            current_deficit = getattr(current, "rainfall_deficit_mm", 0.0) or 0.0
            previous_trend = getattr(previous, "rainfall_trend_3m", 0.0) or 0.0
            current_trend = getattr(current, "rainfall_trend_3m", 0.0) or 0.0
            previous_3m = getattr(previous, "rainfall_3m", 0.0) or 0.0
            current_3m = getattr(current, "rainfall_3m", 0.0) or 0.0
            previous_6m = getattr(previous, "rainfall_6m", 0.0) or 0.0
            current_6m = getattr(current, "rainfall_6m", 0.0) or 0.0

            rainfall_change = current_rainfall - previous_rainfall
            anomaly_change = current_anomaly - previous_anomaly
            deficit_change = current_deficit - previous_deficit
            rainfall_trend_change = current_trend - previous_trend
            rainfall_3m_change = current_3m - previous_3m
            rainfall_6m_change = current_6m - previous_6m

            previous_value = previous_rainfall if previous_rainfall not in (0, 0.0) else 1.0
            persistence = current_rainfall / previous_value if previous_value else 0.0
            volatility = abs(rainfall_change)

            rows.append(
                {
                    "subdivision": subdivision,
                    "previous_state_id": getattr(previous, "state_id", ""),
                    "state_id": getattr(current, "state_id", ""),
                    "rainfall_change": rainfall_change,
                    "anomaly_change": anomaly_change,
                    "deficit_change": deficit_change,
                    "rainfall_trend_change": rainfall_trend_change,
                    "rainfall_3m_change": rainfall_3m_change,
                    "rainfall_6m_change": rainfall_6m_change,
                    "persistence": persistence,
                    "volatility": volatility,
                }
            )

    transitions = pd.DataFrame(
        rows,
        columns=[
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
        ],
    )
    return transitions


def write_state_transitions(df: pd.DataFrame, output_path: str | Path = EARTH_STATE_DIR / "state_transitions.csv") -> pd.DataFrame:
    transitions = build_state_transitions(df)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    transitions.to_csv(output_file, index=False)
    return transitions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Earth State transitions.")
    parser.add_argument("--input", type=Path, default=DEFAULT_OUTPUT_PATH, help="Earth State CSV used for transitions")
    parser.add_argument("--output", type=Path, default=EARTH_STATE_DIR / "state_transitions.csv", help="Output transition CSV")
    args = parser.parse_args()
    source_df = pd.read_csv(args.input)
    write_state_transitions(source_df, args.output)
