from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TARGET_COLUMN = "target_3m_severe_anomaly"
FEATURE_CANDIDATES = [
	"rainfall_mm",
	"rainfall_3m",
	"rainfall_6m",
	"rainfall_12m",
	"rainfall_anomaly",
	"rainfall_anomaly_pct",
	"rainfall_deficit_mm",
	"rainfall_zscore",
	"rainfall_trend_3m",
	"rainfall_lag_1m",
	"rainfall_lag_2m",
	"rainfall_lag_3m",
	"anomaly_persistence",
	"deficit_persistence",
	"rainfall_condition_persistence",
	"recurrence_count",
	"months_since_similar_state",
	"historical_similarity_count",
	"similarity_score",
]
IDENTITY_COLUMNS = ["state_id", "subdivision", "year", "month", "season"]


@dataclass(frozen=True)
class FeatureSet:
	frame: pd.DataFrame
	selected_features: tuple[str, ...]
	excluded_features: tuple[str, ...]
	normalization_input: str = "current Earth State and Earth Memory rows only"


def select_feature_columns(
	earth_state_df: pd.DataFrame,
	earth_memory_df: pd.DataFrame,
) -> tuple[str, ...]:
	"""Select the explicit, available, target-free regime feature contract."""

	available = set(earth_state_df.columns) | set(earth_memory_df.columns)
	selected = tuple(column for column in FEATURE_CANDIDATES if column in available)
	if not selected:
		raise ValueError("No approved climate regime features are present in the inputs.")
	if TARGET_COLUMN in selected:
		raise ValueError(f"Target column {TARGET_COLUMN} cannot be a regime feature.")
	return selected


def build_regime_features(
	earth_state_df: pd.DataFrame,
	earth_memory_df: pd.DataFrame,
) -> FeatureSet:
	"""Join validated observations and memory without introducing future rows."""

	selected = select_feature_columns(earth_state_df, earth_memory_df)
	state_columns = IDENTITY_COLUMNS + [column for column in selected if column in earth_state_df.columns]
	state = earth_state_df[state_columns].copy()
	memory_columns = ["state_id"] + [column for column in selected if column in earth_memory_df.columns]
	memory = earth_memory_df[memory_columns].copy()
	frame = state.merge(memory, on="state_id", how="left", suffixes=("", "_memory"), validate="one_to_one")
	for column in selected:
		memory_column = f"{column}_memory"
		if memory_column in frame.columns:
			frame[column] = frame[column].fillna(frame[memory_column]) if column in frame else frame[memory_column]
			frame = frame.drop(columns=memory_column)
		if column not in frame.columns:
			frame[column] = pd.NA
	frame = frame.sort_values(["subdivision", "year", "month", "state_id"], kind="mergesort")
	frame = frame.reset_index(drop=True)
	numeric = frame[list(selected)].apply(pd.to_numeric, errors="coerce")
	frame.loc[:, list(selected)] = numeric
	return FeatureSet(
		frame=frame,
		selected_features=selected,
		excluded_features=(TARGET_COLUMN,),
	)
