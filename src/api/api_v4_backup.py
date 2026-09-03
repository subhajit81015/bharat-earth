"""
Bharat Earth V4 Production Prediction API

Endpoints:
    GET  /health
    POST /predict
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict


PROJECT_ROOT = Path(
    os.getenv(
        "BHARAT_EARTH_ROOT",
        Path(__file__).resolve().parents[2],
    )
)

MODEL_FILE = PROJECT_ROOT / "data" / "features" / "model_v4" / "xgboost_model.json"
SCHEMA_FILE = PROJECT_ROOT / "data" / "features" / "model_v4" / "model_schema.json"
CALIBRATION_FILE = (
    PROJECT_ROOT
    / "data" / "features" / "calibration_v4"
    / "calibration_summary.json"
)
POLICY_FILE = (
    PROJECT_ROOT
    / "data" / "features" / "policy_v4"
    / "selected_policy.csv"
)

MODEL_VERSION = "bharat-earth-v4"
TARGET = "target_3m_severe_anomaly"

FEATURES = [
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
    "rainfall_lag_1m",
    "rainfall_lag_2m",
    "rainfall_lag_3m",
    "rainfall_prev_3m",
    "rainfall_prev_6m",
    "rainfall_prev_12m",
    "rainfall_trend_3m",
    "month_sin",
    "month_cos",
]

CATEGORICAL_FEATURES = {
    "subdivision",
    "month",
    "season",
}

SEASON_BY_MONTH = {
    12: "WINTER",
    1: "WINTER",
    2: "WINTER",
    3: "PRE_MONSOON",
    4: "PRE_MONSOON",
    5: "PRE_MONSOON",
    6: "MONSOON",
    7: "MONSOON",
    8: "MONSOON",
    9: "MONSOON",
    10: "POST_MONSOON",
    11: "POST_MONSOON",
}

SUBDIVISIONS = []  # Populated from the trained XGBoost model categories.

MONTH_CATEGORIES = [str(i) for i in range(1, 13)]

SEASON_CATEGORIES = [
    "MONSOON",
    "POST_MONSOON",
    "PRE_MONSOON",
    "WINTER",
]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subdivision: str
    year: int
    month: int = Field(ge=1, le=12)
    season: str

    rainfall_mm: float
    rainfall_3m: float
    rainfall_6m: float
    rainfall_12m: float
    historical_monthly_mean: float
    rainfall_anomaly: float
    rainfall_anomaly_pct: float
    rainfall_deficit_mm: float
    rainfall_missing: int = Field(ge=0, le=1)
    rainfall_zscore: float
    rainfall_lag_1m: float
    rainfall_lag_2m: float
    rainfall_lag_3m: float
    rainfall_prev_3m: float
    rainfall_prev_6m: float
    rainfall_prev_12m: float
    rainfall_trend_3m: float
    month_sin: float
    month_cos: float


class PredictionResponse(BaseModel):
    model: str
    target: str
    subdivision: str
    year: int
    month: int
    season: str
    raw_probability: float
    calibrated_probability: float
    calibration: str
    threshold: float
    alert: bool
    decision: str


def _load_json(path: Path) -> dict[str, Any]:

    if not path.exists():
        raise RuntimeError(
            f"Required artifact missing: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def _extract_sigmoid_parameters(
    summary: dict[str, Any],
) -> tuple[float, float]:

    candidates = [
        summary.get("sigmoid_parameters"),
        summary.get("sigmoid"),
        summary.get("parameters"),
        summary,
    ]

    for item in candidates:

        if not isinstance(item, dict):
            continue

        a = item.get("a")
        b = item.get("b")

        if a is None:
            a = item.get("coef")

        if b is None:
            b = item.get("intercept")

        try:
            a = float(a)
            b = float(b)

        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            math.isfinite(a)
            and math.isfinite(b)
        ):
            return a, b

    raise RuntimeError(
        "Unable to reproduce sigmoid calibration: "
        "no valid parameters found."
    )


def _load_policy() -> tuple[float, str]:

    if not POLICY_FILE.exists():
        raise RuntimeError(
            f"Required policy artifact missing: "
            f"{POLICY_FILE}"
        )

    policy = pd.read_csv(
        POLICY_FILE
    )

    if len(policy) != 1:
        raise RuntimeError(
            f"Expected exactly one selected policy row, "
            f"found {len(policy)}."
        )

    row = policy.iloc[0]

    threshold = float(
        row["threshold"]
    )

    probability_type = str(
        row["probability_type"]
    ).strip()

    if probability_type != "sigmoid_probability":
        raise RuntimeError(
            "Unsupported production probability type: "
            f"{probability_type}"
        )

    if not 0.0 < threshold < 1.0:
        raise RuntimeError(
            f"Invalid production threshold: {threshold}"
        )

    return (
        threshold,
        probability_type,
    )


def _load_schema_categories() -> dict[str, list[str]]:
    """Load categorical vocabularies.

    Priority:
      1. Categories stored in the trained XGBoost model.
      2. model_schema.json categorical_categories.
      3. Safe month/season fallbacks.

    Using the model's stored categories avoids categorical-code mismatches
    between training and production inference.
    """
    categories: dict[str, list[str]] = {}

    # XGBoost >= 3.1 can expose the categories stored in the model.
    try:
        booster = MODEL.get_booster()
        stored = booster.get_categories(export_to_arrow=True).to_arrow()

        for feature_name, values in stored:
            if values is not None:
                categories[str(feature_name)] = [
                    str(v) for v in values.to_pylist()
                ]
    except Exception:
        # Fall back to the schema if model category export is unavailable.
        pass

    # Schema uses "categorical_categories" in V4.
    schema = _load_json(SCHEMA_FILE)
    raw = schema.get("categorical_categories", {})

    if isinstance(raw, dict):
        for key, values in raw.items():
            if key not in categories and isinstance(values, list):
                categories[key] = [str(v) for v in values]

    # Backward-compatible support for a generic "categories" key.
    raw_generic = schema.get("categories", {})
    if isinstance(raw_generic, dict):
        for key, values in raw_generic.items():
            if key not in categories and isinstance(values, list):
                categories[key] = [str(v) for v in values]

    # These are deterministic and also present in the trained model.
    categories.setdefault("month", MONTH_CATEGORIES)
    categories.setdefault("season", SEASON_CATEGORIES)

    if "subdivision" not in categories or not categories["subdivision"]:
        raise RuntimeError(
            "No trained subdivision categories were found in the "
            "XGBoost model or V4 schema."
        )

    return categories


try:

    MODEL = xgb.XGBClassifier(
    enable_categorical=True
    
    )

    MODEL.load_model(
        str(MODEL_FILE)
    )

    CALIBRATION_SUMMARY = _load_json(
        CALIBRATION_FILE
    )

    SIGMOID_A, SIGMOID_B = (
        _extract_sigmoid_parameters(
            CALIBRATION_SUMMARY
        )
    )

    THRESHOLD, PROBABILITY_TYPE = (
        _load_policy()
    )

    CATEGORIES = (
        _load_schema_categories()
    )

    if not CATEGORIES.get("subdivision"):
        raise RuntimeError("Subdivision categories are empty.")

    if len(CATEGORIES["subdivision"]) != 36:
        raise RuntimeError(
            "Expected 36 trained subdivision categories, found "
            f"{len(CATEGORIES['subdivision'])}."
        )

    model_features = list(
        getattr(
            MODEL,
            "feature_names",
            [],
        )
        or []
    )

    if (
        model_features
        and model_features != FEATURES
    ):

        raise RuntimeError(
            "Model feature order does not "
            "match V4 production schema."
        )

except Exception as exc:

    raise RuntimeError(
        "Failed to initialize "
        f"Bharat Earth V4 API: {exc}"
    ) from exc


def apply_sigmoid(
    raw_probability: float,
) -> float:

    z = (
        SIGMOID_A
        * float(raw_probability)
        + SIGMOID_B
    )

    if z >= 0:

        calibrated = (
            1.0
            / (
                1.0
                + math.exp(-z)
            )
        )

    else:

        ez = math.exp(z)

        calibrated = (
            ez
            / (
                1.0
                + ez
            )
        )

    if not math.isfinite(
        calibrated
    ):

        raise RuntimeError(
            "Non-finite calibrated probability."
        )

    return float(
        calibrated
    )


def build_feature_frame(
    request: PredictionRequest,
) -> pd.DataFrame:

    data = request.model_dump()

    expected_season = (
        SEASON_BY_MONTH[
            int(data["month"])
        ]
    )

    supplied_season = str(
        data["season"]
    ).strip().upper()

    if supplied_season != expected_season:

        supplied_season = (
            expected_season
        )

    data["season"] = (
        supplied_season
    )

    data["month"] = str(
        int(data["month"])
    )

    for feature in FEATURES:

        if feature not in data:

            raise HTTPException(
                status_code=422,
                detail=(
                    "Missing required feature: "
                    f"{feature}"
                ),
            )

    frame = pd.DataFrame(
        [data],
        columns=FEATURES,
    )

    for feature in CATEGORICAL_FEATURES:

        allowed = CATEGORIES.get(
            feature
        )

        if allowed:

            value = str(
                frame.loc[
                    0,
                    feature,
                ]
            )

            if value not in allowed:

                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Invalid {feature}="
                        f"'{value}'. "
                        "Expected one of the "
                        "V4 categories."
                    ),
                )

            frame[feature] = pd.Categorical(
                frame[feature].astype(str),
                categories=[str(v) for v in allowed],
            )

    numeric_features = [
        feature
        for feature in FEATURES
        if feature
        not in CATEGORICAL_FEATURES
    ]

    for feature in numeric_features:

        frame[feature] = pd.to_numeric(
            frame[feature],
            errors="coerce",
        )

    if frame[
        numeric_features
    ].isna().any().any():

        bad = [
            feature
            for feature in numeric_features
            if pd.isna(
                frame.loc[
                    0,
                    feature,
                ]
            )
        ]

        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid or missing numeric "
                f"features: {bad}"
            ),
        )

    return frame[
        FEATURES
    ]


app = FastAPI(
    title="Bharat Earth V4 API",
    description=(
        "Production prediction service "
        "for Bharat Earth V4 severe "
        "3-month rainfall anomaly detection."
    ),
    version="4.0.0",
)


@app.get("/health")
def health() -> dict[str, Any]:

    return {
        "status": "healthy",
        "model": MODEL_VERSION,
        "calibration": "sigmoid",
        "probability_type": PROBABILITY_TYPE,
        "threshold": THRESHOLD,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
) -> PredictionResponse:

    try:

        frame = build_feature_frame(
            request
        )

        raw_probability = float(
            MODEL.predict_proba(
                frame
            )[:, 1][0]
        )

        calibrated_probability = (
            apply_sigmoid(
                raw_probability
            )
        )

        alert = (
            calibrated_probability
            >= THRESHOLD
        )

        return PredictionResponse(

            model=MODEL_VERSION,

            target=TARGET,

            subdivision=request.subdivision,

            year=request.year,

            month=request.month,

            season=str(
                frame.loc[
                    0,
                    "season",
                ]
            ),

            raw_probability=(
                raw_probability
            ),

            calibrated_probability=(
                calibrated_probability
            ),

            calibration="sigmoid",

            threshold=THRESHOLD,

            alert=bool(alert),

            decision=(
                "ALERT"
                if alert
                else "NO_ALERT"
            ),
        )

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {exc}"
            ),
        ) from exc