from __future__ import annotations

import pandas as pd

from src.secrire.climate_regime.discovery import DiscoveryResult
from src.secrire.climate_regime.schema import REGIME_ASSIGNMENT_VERSION


def assign_regimes(frame: pd.DataFrame, discovery: DiscoveryResult) -> pd.DataFrame:
    result = frame[["state_id", "subdivision", "year", "month", "season"]].copy()
    result["regime_id"] = [f"REGIME_{int(label) + 1:02d}" for label in discovery.labels]
    result["regime_assignment_version"] = REGIME_ASSIGNMENT_VERSION
    result = result.sort_values(["subdivision", "year", "month", "state_id"], kind="mergesort")
    result["previous_state_id"] = result.groupby("subdivision", sort=False)["state_id"].shift(1)
    result["previous_regime_id"] = result.groupby("subdivision", sort=False)["regime_id"].shift(1)
    result["regime_changed"] = result["previous_regime_id"].notna() & result["regime_id"].ne(result["previous_regime_id"])
    run_id = result["regime_id"].ne(result["previous_regime_id"]).groupby(result["subdivision"], sort=False).cumsum()
    result["consecutive_months_in_current_regime"] = result.groupby(["subdivision", run_id], sort=False).cumcount() + 1
    result["regime_duration"] = result["consecutive_months_in_current_regime"]
    result["regime_changed"] = result["regime_changed"].astype(bool)
    return result.reset_index(drop=True)