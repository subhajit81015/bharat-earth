from __future__ import annotations

import pandas as pd

from src.secrire.climate_regime.schema import TRANSITION_COLUMNS


def build_transition_records(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = assignments.copy()
    rows["months_in_previous_regime"] = rows.groupby("subdivision", sort=False)[
        "consecutive_months_in_current_regime"
    ].shift(1)
    rows = rows[rows["previous_state_id"].notna()].copy()
    rows["months_in_previous_regime"] = rows["months_in_previous_regime"].astype(int)
    return rows[[
        "subdivision", "previous_state_id", "state_id", "previous_regime_id", "regime_id",
        "regime_changed", "months_in_previous_regime",
    ]].rename(columns={"state_id": "current_state_id", "regime_id": "current_regime_id"})[TRANSITION_COLUMNS]


def summarize_transitions(assignments: pd.DataFrame) -> pd.DataFrame:
    records = build_transition_records(assignments)
    summary = records.groupby(
        ["subdivision", "previous_regime_id", "current_regime_id"], as_index=False
    ).size().rename(columns={
        "previous_regime_id": "from_regime",
        "current_regime_id": "to_regime",
        "size": "transition_count",
    })
    totals = summary.groupby("subdivision")["transition_count"].transform("sum")
    summary["transition_probability"] = summary["transition_count"] / totals
    return summary


def enrich_transition_records(assignments: pd.DataFrame) -> pd.DataFrame:
    records = build_transition_records(assignments)
    summary = summarize_transitions(assignments)
    return records.merge(
        summary,
        left_on=["subdivision", "previous_regime_id", "current_regime_id"],
        right_on=["subdivision", "from_regime", "to_regime"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["from_regime", "to_regime"])