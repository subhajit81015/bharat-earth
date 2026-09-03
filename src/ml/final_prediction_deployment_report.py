from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# 3.11 FINAL PREDICTION & DEPLOYMENT REPORT
# ============================================================

print("=" * 70)
print("3.11 FINAL PREDICTION & DEPLOYMENT REPORT")
print("=" * 70)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FINAL_PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "final_model"
    / "final_predictions.csv"
)

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "ml_dataset_v2.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "deployment"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FINAL OPERATIONAL POLICY
# ============================================================

FINAL_THRESHOLD = 0.09


# ============================================================
# VALID SEASONS
# ============================================================

VALID_SEASONS = {
    "WINTER",
    "PRE_MONSOON",
    "MONSOON",
    "POST_MONSOON",
}


# ============================================================
# SEASON FROM MONTH
# ============================================================

def season_from_month(month):

    if pd.isna(month):
        return pd.NA

    try:
        month = int(month)
    except Exception:
        return pd.NA

    if month in [12, 1, 2]:
        return "WINTER"

    if month in [3, 4, 5]:
        return "PRE_MONSOON"

    if month in [6, 7, 8, 9]:
        return "MONSOON"

    if month in [10, 11]:
        return "POST_MONSOON"

    # IMPORTANT:
    # 0 is NOT assigned to any season.
    return pd.NA


# ============================================================
# LOAD FINAL PREDICTIONS
# ============================================================

def load_predictions():

    print()
    print("=" * 70)
    print("LOADING FINAL PREDICTIONS")
    print("=" * 70)

    if not FINAL_PREDICTIONS_FILE.exists():

        raise FileNotFoundError(
            "\nFinal prediction file not found:\n"
            f"{FINAL_PREDICTIONS_FILE}\n\n"
            "Run the final model prediction stage first."
        )

    df = pd.read_csv(
        FINAL_PREDICTIONS_FILE
    )

    print(
        "INPUT:",
        FINAL_PREDICTIONS_FILE
    )

    print(
        "ROWS:",
        len(df)
    )

    print(
        "COLUMNS:",
        list(df.columns)
    )

    required_columns = [
        "subdivision",
        "year",
        "month",
        "actual",
        "final_probability",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    df["year"] = pd.to_numeric(
        df["year"],
        errors="coerce",
    )

    df["month"] = pd.to_numeric(
        df["month"],
        errors="coerce",
    )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    df["final_probability"] = pd.to_numeric(
        df["final_probability"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Remove impossible target/probability rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "actual",
            "final_probability",
        ]
    ).copy()

    df["actual"] = (
        df["actual"]
        .astype(int)
    )

    return df


# ============================================================
# LOAD ORIGINAL FEATURE DATA
# ============================================================

def load_features():

    print()
    print("=" * 70)
    print("LOADING ORIGINAL FEATURE DATA")
    print("=" * 70)

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            "\nFeature dataset not found:\n"
            f"{FEATURE_FILE}"
        )

    features = pd.read_csv(
        FEATURE_FILE
    )

    print(
        "INPUT:",
        FEATURE_FILE
    )

    print(
        "ROWS:",
        len(features)
    )

    required = [
        "subdivision",
        "year",
        "month",
        "season",
    ]

    missing = [
        col
        for col in required
        if col not in features.columns
    ]

    if missing:

        raise ValueError(
            f"Missing feature columns: {missing}"
        )

    features["year"] = pd.to_numeric(
        features["year"],
        errors="coerce",
    )

    features["month"] = pd.to_numeric(
        features["month"],
        errors="coerce",
    )

    features["season"] = (
        features["season"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    return features[
        [
            "subdivision",
            "year",
            "month",
            "season",
        ]
    ].copy()


# ============================================================
# SEASON RECOVERY
# ============================================================

def recover_season(
    predictions,
    features,
):

    df = predictions.copy()

    print()
    print("=" * 70)
    print("SEASON RECOVERY")
    print("=" * 70)

    # --------------------------------------------------------
    # Count month 0
    # --------------------------------------------------------

    month_numeric = pd.to_numeric(
        df["month"],
        errors="coerce",
    )

    zero_month_mask = (
        month_numeric == 0
    )

    print(
        "MONTH == 0 ROWS:",
        int(zero_month_mask.sum())
    )

    # --------------------------------------------------------
    # Create STRING season column from the beginning.
    # This fixes the pandas dtype error.
    # --------------------------------------------------------

    recovered = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    # --------------------------------------------------------
    # STEP 1
    # Preserve valid season already contained in
    # final_predictions.csv.
    # --------------------------------------------------------

    if "season" in df.columns:

        existing_season = (
            df["season"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        valid_existing = (
            existing_season.isin(
                VALID_SEASONS
            )
        )

        recovered.loc[
            valid_existing
        ] = existing_season.loc[
            valid_existing
        ]

        print(
            "VALID EXISTING SEASONS:",
            int(valid_existing.sum())
        )

    else:

        existing_season = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )

        print(
            "EXISTING SEASON COLUMN: NOT PRESENT"
        )

    # --------------------------------------------------------
    # STEP 2
    # Derive season from valid month.
    #
    # This applies ONLY to months 1-12.
    # --------------------------------------------------------

    derived = month_numeric.apply(
        season_from_month
    ).astype("string")

    valid_month_mask = (
        month_numeric.between(
            1,
            12,
        )
    )

    fill_from_month = (
        recovered.isna()
        &
        valid_month_mask
        &
        derived.notna()
    )

    recovered.loc[
        fill_from_month
    ] = derived.loc[
        fill_from_month
    ]

    print(
        "SEASONS DERIVED FROM MONTH:",
        int(fill_from_month.sum())
    )

    # --------------------------------------------------------
    # STEP 3
    # Use original feature data as a fallback.
    #
    # IMPORTANT:
    # We only use exact subdivision/year/month matches.
    # --------------------------------------------------------

    feature_lookup = features[
        features["month"].between(
            1,
            12,
        )
    ].copy()

    feature_lookup = (
        feature_lookup[
            [
                "subdivision",
                "year",
                "month",
                "season",
            ]
        ]
        .dropna(
            subset=[
                "subdivision",
                "year",
                "month",
                "season",
            ]
        )
        .drop_duplicates(
            subset=[
                "subdivision",
                "year",
                "month",
            ]
        )
    )

    # --------------------------------------------------------
    # Merge original season.
    # --------------------------------------------------------

    df["_row_id_311"] = np.arange(
        len(df)
    )

    merged = df.merge(
        feature_lookup,
        on=[
            "subdivision",
            "year",
            "month",
        ],
        how="left",
        suffixes=(
            "",
            "_feature",
        ),
    )

    # --------------------------------------------------------
    # Important:
    # merged index no longer corresponds to original index,
    # so use _row_id_311 for safe assignment.
    # --------------------------------------------------------

    merged["_recovered_season"] = pd.Series(
        pd.NA,
        index=merged.index,
        dtype="string",
    )

    # Existing valid season
    if "season" in merged.columns:

        existing = (
            merged["season"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        mask = existing.isin(
            VALID_SEASONS
        )

        merged.loc[
            mask,
            "_recovered_season",
        ] = existing.loc[
            mask
        ]

    # Feature season
    if "season_feature" in merged.columns:

        feature_season = (
            merged["season_feature"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        feature_mask = (
            merged[
                "_recovered_season"
            ].isna()
            &
            feature_season.isin(
                VALID_SEASONS
            )
        )

        merged.loc[
            feature_mask,
            "_recovered_season",
        ] = feature_season.loc[
            feature_mask
        ]

    # Derive from valid month
    merged_month = pd.to_numeric(
        merged["month"],
        errors="coerce",
    )

    derived_merged = (
        merged_month
        .apply(
            season_from_month
        )
        .astype("string")
    )

    month_mask = (
        merged[
            "_recovered_season"
        ].isna()
        &
        merged_month.between(
            1,
            12,
        )
        &
        derived_merged.notna()
    )

    merged.loc[
        month_mask,
        "_recovered_season",
    ] = derived_merged.loc[
        month_mask
    ]

    # --------------------------------------------------------
    # Return to original order
    # --------------------------------------------------------

    merged = merged.sort_values(
        "_row_id_311"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Final season column explicitly STRING.
    # --------------------------------------------------------

    df = merged.copy()

    df["season"] = (
        df[
            "_recovered_season"
        ]
        .astype("string")
    )

    # --------------------------------------------------------
    # Remove helper columns
    # --------------------------------------------------------

    helper_columns = [
        "_row_id_311",
        "_recovered_season",
        "season_feature",
    ]

    for col in helper_columns:

        if col in df.columns:

            df.drop(
                columns=col,
                inplace=True,
            )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    unknown_mask = (
        df["season"].isna()
        |
        ~df["season"].isin(
            VALID_SEASONS
        )
    )

    unknown_count = int(
        unknown_mask.sum()
    )

    print()
    print(
        "UNKNOWN / UNRESOLVED SEASONS:",
        unknown_count
    )

    if unknown_count > 0:

        print()
        print(
            "UNRESOLVED MONTH VALUES:"
        )

        print(
            sorted(
                df.loc[
                    unknown_mask,
                    "month",
                ]
                .dropna()
                .unique()
                .tolist()
            )
        )

        print()
        print(
            "UNRESOLVED RECORD SAMPLE:"
        )

        print(
            df.loc[
                unknown_mask,
                [
                    "subdivision",
                    "year",
                    "month",
                    "season",
                ],
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

        # ----------------------------------------------------
        # We do NOT invent the season.
        # Keep UNKNOWN explicitly.
        # ----------------------------------------------------

        df.loc[
            unknown_mask,
            "season",
        ] = "UNKNOWN"

    # --------------------------------------------------------
    # Final distribution
    # --------------------------------------------------------

    print()
    print(
        "FINAL SEASON DISTRIBUTION:"
    )

    print(
        df[
            "season"
        ]
        .value_counts(
            dropna=False
        )
    )

    return df


# ============================================================
# BUILD FINAL RISK OUTPUT
# ============================================================

def build_risk_output(
    df
):

    data = df.copy()

    data[
        "risk_probability"
    ] = (
        pd.to_numeric(
            data[
                "final_probability"
            ],
            errors="coerce",
        )
        .clip(
            0,
            1,
        )
    )

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    data[
        "policy_threshold"
    ] = FINAL_THRESHOLD

    data[
        "risk_alert"
    ] = (
        data[
            "risk_probability"
        ]
        >= FINAL_THRESHOLD
    ).astype(int)

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    data[
        "risk_level"
    ] = np.select(
        [
            data[
                "risk_probability"
            ] >= 0.15,

            data[
                "risk_probability"
            ] >= 0.10,

            data[
                "risk_probability"
            ] >= 0.09,

            data[
                "risk_probability"
            ] >= 0.05,
        ],
        [
            "CRITICAL",
            "HIGH",
            "ELEVATED",
            "MODERATE",
        ],
        default="LOW",
    )

    data[
        "alert_priority"
    ] = data[
        "risk_level"
    ]

    # --------------------------------------------------------
    # Prediction status
    # --------------------------------------------------------

    data[
        "prediction_status"
    ] = np.select(
        [
            (
                data["actual"] == 1
            )
            &
            (
                data["risk_alert"] == 1
            ),

            (
                data["actual"] == 0
            )
            &
            (
                data["risk_alert"] == 0
            ),

            (
                data["actual"] == 0
            )
            &
            (
                data["risk_alert"] == 1
            ),

            (
                data["actual"] == 1
            )
            &
            (
                data["risk_alert"] == 0
            ),
        ],
        [
            "TRUE_POSITIVE",
            "TRUE_NEGATIVE",
            "FALSE_POSITIVE",
            "FALSE_NEGATIVE",
        ],
        default="UNKNOWN",
    )

    return data


# ============================================================
# OVERALL METRICS
# ============================================================

def calculate_overall_metrics(
    df
):

    y = df[
        "actual"
    ]

    predictions = df[
        "risk_alert"
    ]

    probabilities = df[
        "risk_probability"
    ]

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    result = {
        "observations":
            len(df),

        "events":
            int(y.sum()),

        "event_rate":
            float(y.mean()),

        "alerts":
            int(predictions.sum()),

        "alert_rate":
            float(predictions.mean()),

        "precision":
            precision_score(
                y,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y,
                predictions,
                zero_division=0,
            ),

        "pr_auc":
            average_precision_score(
                y,
                probabilities,
            ),

        "roc_auc":
            roc_auc_score(
                y,
                probabilities,
            ),

        "true_positive":
            int(tp),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_negative":
            int(tn),

        "false_positive_rate":
            (
                fp / (fp + tn)
                if (fp + tn) > 0
                else 0
            ),

        "average_probability":
            float(
                probabilities.mean()
            ),

        "maximum_probability":
            float(
                probabilities.max()
            ),
    }

    return pd.DataFrame(
        [result]
    )


# ============================================================
# RISK DISTRIBUTION
# ============================================================

def risk_distribution(
    df
):

    result = (
        df.groupby(
            "risk_level"
        )
        .agg(
            observations=(
                "actual",
                "size",
            ),
            events=(
                "actual",
                "sum",
            ),
            alerts=(
                "risk_alert",
                "sum",
            ),
            average_probability=(
                "risk_probability",
                "mean",
            ),
        )
        .reset_index()
    )

    result[
        "percentage"
    ] = (
        result[
            "observations"
        ]
        / len(df)
    )

    result[
        "event_rate"
    ] = (
        result[
            "events"
        ]
        / result[
            "observations"
        ]
    )

    order = [
        "CRITICAL",
        "HIGH",
        "ELEVATED",
        "MODERATE",
        "LOW",
    ]

    result[
        "risk_level"
    ] = pd.Categorical(
        result[
            "risk_level"
        ],
        categories=order,
        ordered=True,
    )

    return result.sort_values(
        "risk_level"
    )


# ============================================================
# REGIONAL SUMMARY
# ============================================================

def regional_summary(
    df
):

    rows = []

    for region, group in (
        df.groupby(
            "subdivision"
        )
    ):

        y = group[
            "actual"
        ]

        p = group[
            "risk_alert"
        ]

        rows.append(
            {
                "subdivision":
                    region,

                "observations":
                    len(group),

                "events":
                    int(y.sum()),

                "event_rate":
                    float(y.mean()),

                "alerts":
                    int(p.sum()),

                "alert_rate":
                    float(p.mean()),

                "average_probability":
                    float(
                        group[
                            "risk_probability"
                        ].mean()
                    ),

                "maximum_probability":
                    float(
                        group[
                            "risk_probability"
                        ].max()
                    ),

                "precision":
                    precision_score(
                        y,
                        p,
                        zero_division=0,
                    ),

                "recall":
                    recall_score(
                        y,
                        p,
                        zero_division=0,
                    ),

                "f1":
                    f1_score(
                        y,
                        p,
                        zero_division=0,
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result[
        "risk_rank"
    ] = (
        result[
            "average_probability"
        ]
        .rank(
            ascending=False,
            method="dense",
        )
        .astype(int)
    )

    return result.sort_values(
        "risk_rank"
    )


# ============================================================
# SEASONAL SUMMARY
# ============================================================

def seasonal_summary(
    df
):

    rows = []

    for season, group in (
        df.groupby(
            "season",
            dropna=False,
        )
    ):

        y = group[
            "actual"
        ]

        p = group[
            "risk_alert"
        ]

        rows.append(
            {
                "season":
                    season,

                "observations":
                    len(group),

                "events":
                    int(y.sum()),

                "event_rate":
                    float(y.mean()),

                "alerts":
                    int(p.sum()),

                "alert_rate":
                    float(p.mean()),

                "average_probability":
                    float(
                        group[
                            "risk_probability"
                        ].mean()
                    ),

                "maximum_probability":
                    float(
                        group[
                            "risk_probability"
                        ].max()
                    ),

                "precision":
                    precision_score(
                        y,
                        p,
                        zero_division=0,
                    ),

                "recall":
                    recall_score(
                        y,
                        p,
                        zero_division=0,
                    ),

                "f1":
                    f1_score(
                        y,
                        p,
                        zero_division=0,
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    order = [
        "WINTER",
        "PRE_MONSOON",
        "MONSOON",
        "POST_MONSOON",
        "UNKNOWN",
    ]

    result[
        "_order"
    ] = result[
        "season"
    ].map(
        {
            name: i
            for i, name in enumerate(order)
        }
    ).fillna(999)

    result = result.sort_values(
        "_order"
    ).drop(
        columns="_order"
    )

    return result


# ============================================================
# MONTHLY SUMMARY
# ============================================================

def monthly_summary(
    df
):

    rows = []

    for month, group in (
        df.groupby(
            "month",
            dropna=False,
        )
    ):

        y = group[
            "actual"
        ]

        p = group[
            "risk_alert"
        ]

        rows.append(
            {
                "month":
                    month,

                "observations":
                    len(group),

                "events":
                    int(y.sum()),

                "event_rate":
                    float(y.mean()),

                "alerts":
                    int(p.sum()),

                "alert_rate":
                    float(p.mean()),

                "average_probability":
                    float(
                        group[
                            "risk_probability"
                        ].mean()
                    ),

                "maximum_probability":
                    float(
                        group[
                            "risk_probability"
                        ].max()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "month"
    )


# ============================================================
# HIGH-RISK ALERTS
# ============================================================

def high_risk_alerts(
    df
):

    alerts = df[
        df[
            "risk_alert"
        ] == 1
    ].copy()

    alerts = alerts.sort_values(
        [
            "risk_probability",
            "year",
            "month",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    columns = [
        "subdivision",
        "year",
        "month",
        "season",
        "actual",
        "risk_probability",
        "risk_level",
        "alert_priority",
        "policy_threshold",
        "prediction_status",
    ]

    return alerts[
        columns
    ]


# ============================================================
# ERROR SUMMARY
# ============================================================

def error_summary(
    df
):

    result = (
        df[
            "prediction_status"
        ]
        .value_counts()
        .rename_axis(
            "prediction_status"
        )
        .reset_index(
            name="count"
        )
    )

    result[
        "percentage"
    ] = (
        result["count"]
        / len(df)
    )

    return result


# ============================================================
# MODEL CARD
# ============================================================

def create_model_card(
    metrics
):

    m = metrics.iloc[
        0
    ]

    text = f"""
# Bharat Earth
## Final Model Card

### Model

- Model: XGBoost
- Probability calibration: Sigmoid
- Policy: Global probability threshold
- Operational threshold: {FINAL_THRESHOLD:.2f}

### Test Performance

- Observations: {int(m['observations'])}
- Events: {int(m['events'])}
- Event rate: {m['event_rate']:.6f}
- Alerts: {int(m['alerts'])}
- Alert rate: {m['alert_rate']:.6f}
- Precision: {m['precision']:.6f}
- Recall: {m['recall']:.6f}
- F1: {m['f1']:.6f}
- PR-AUC: {m['pr_auc']:.6f}
- ROC-AUC: {m['roc_auc']:.6f}

### Confusion Matrix

- True Positive: {int(m['true_positive'])}
- False Positive: {int(m['false_positive'])}
- False Negative: {int(m['false_negative'])}
- True Negative: {int(m['true_negative'])}

### Operational Policy

Generate an environmental severe-anomaly alert when:

    calibrated_probability >= 0.09

### Risk Levels

- CRITICAL: probability >= 0.15
- HIGH: probability >= 0.10
- ELEVATED: probability >= 0.09
- MODERATE: probability >= 0.05
- LOW: probability < 0.05

### Important Limitation

The model is an early-warning decision-support system.
It is not a deterministic drought or rainfall forecast.

False positives and false negatives remain material.
The operational threshold should be periodically
revalidated using newly observed data.

### Monitoring

Monitor:

1. PR-AUC
2. ROC-AUC
3. Precision
4. Recall
5. Alert rate
6. Probability calibration
7. Regional drift
8. Seasonal drift
9. Feature distribution drift
10. Target/event-rate drift
"""

    return text.strip()


# ============================================================
# SAVE EXCEL
# ============================================================

def save_excel(
    metrics,
    distribution,
    regional,
    seasonal,
    monthly,
    alerts,
    errors,
    predictions,
):

    output_file = (
        OUTPUT_DIR
        / "deployment_report.xlsx"
    )

    try:

        with pd.ExcelWriter(
            output_file,
            engine="openpyxl",
        ) as writer:

            metrics.to_excel(
                writer,
                sheet_name="metrics",
                index=False,
            )

            distribution.to_excel(
                writer,
                sheet_name="risk_distribution",
                index=False,
            )

            regional.to_excel(
                writer,
                sheet_name="regional",
                index=False,
            )

            seasonal.to_excel(
                writer,
                sheet_name="seasonal",
                index=False,
            )

            monthly.to_excel(
                writer,
                sheet_name="monthly",
                index=False,
            )

            alerts.to_excel(
                writer,
                sheet_name="alerts",
                index=False,
            )

            errors.to_excel(
                writer,
                sheet_name="errors",
                index=False,
            )

            predictions.to_excel(
                writer,
                sheet_name="predictions",
                index=False,
            )

        print()
        print(
            "EXCEL REPORT SAVED:"
        )

        print(
            output_file
        )

    except ImportError:

        print()
        print(
            "WARNING: openpyxl is not installed."
        )

        print(
            "Install it with:"
        )

        print(
            "pip install openpyxl"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load predictions
    # --------------------------------------------------------

    predictions = load_predictions()

    # --------------------------------------------------------
    # 2. Load original feature data
    # --------------------------------------------------------

    features = load_features()

    # --------------------------------------------------------
    # 3. Recover season
    # --------------------------------------------------------

    predictions = recover_season(
        predictions,
        features,
    )

    # --------------------------------------------------------
    # 4. Build final risk predictions
    # --------------------------------------------------------

    df = build_risk_output(
        predictions
    )

    # --------------------------------------------------------
    # 5. Overall metrics
    # --------------------------------------------------------

    metrics = calculate_overall_metrics(
        df
    )

    print()
    print("=" * 70)
    print("GLOBAL DEPLOYMENT POLICY")
    print("=" * 70)

    print(
        f"THRESHOLD: {FINAL_THRESHOLD:.2f}"
    )

    print(
        metrics.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 6. Risk distribution
    # --------------------------------------------------------

    distribution = risk_distribution(
        df
    )

    print()
    print("=" * 70)
    print("RISK DISTRIBUTION")
    print("=" * 70)

    print(
        distribution.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 7. Regional
    # --------------------------------------------------------

    regional = regional_summary(
        df
    )

    print()
    print("=" * 70)
    print("TOP 15 REGIONAL RISK")
    print("=" * 70)

    print(
        regional.head(
            15
        ).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 8. Seasonal
    # --------------------------------------------------------

    seasonal = seasonal_summary(
        df
    )

    print()
    print("=" * 70)
    print("SEASONAL RISK")
    print("=" * 70)

    print(
        seasonal.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 9. Monthly
    # --------------------------------------------------------

    monthly = monthly_summary(
        df
    )

    # --------------------------------------------------------
    # 10. Alerts
    # --------------------------------------------------------

    alerts = high_risk_alerts(
        df
    )

    print()
    print("=" * 70)
    print("HIGH-RISK ALERTS")
    print("=" * 70)

    print(
        "TOTAL ALERTS:",
        len(alerts)
    )

    print()

    if len(alerts) > 0:

        print(
            alerts.head(
                20
            ).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 11. Error analysis
    # --------------------------------------------------------

    errors = error_summary(
        df
    )

    print()
    print("=" * 70)
    print("PREDICTION ERROR SUMMARY")
    print("=" * 70)

    print(
        errors.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 12. Final prediction output
    # --------------------------------------------------------

    final_columns = [
        "subdivision",
        "year",
        "month",
        "season",
        "actual",
        "final_probability",
        "risk_probability",
        "risk_level",
        "risk_alert",
        "alert_priority",
        "policy_threshold",
        "prediction_status",
    ]

    final_predictions = df[
        [
            col
            for col in final_columns
            if col in df.columns
        ]
    ].copy()

    # --------------------------------------------------------
    # 13. Save CSV files
    # --------------------------------------------------------

    final_predictions_file = (
        OUTPUT_DIR
        / "final_risk_predictions.csv"
    )

    alerts_file = (
        OUTPUT_DIR
        / "high_risk_alerts.csv"
    )

    regional_file = (
        OUTPUT_DIR
        / "regional_risk_summary.csv"
    )

    seasonal_file = (
        OUTPUT_DIR
        / "seasonal_risk_summary.csv"
    )

    monthly_file = (
        OUTPUT_DIR
        / "monthly_risk_summary.csv"
    )

    distribution_file = (
        OUTPUT_DIR
        / "risk_distribution.csv"
    )

    errors_file = (
        OUTPUT_DIR
        / "deployment_error_summary.csv"
    )

    metrics_file = (
        OUTPUT_DIR
        / "deployment_metrics.csv"
    )

    final_predictions.to_csv(
        final_predictions_file,
        index=False,
    )

    alerts.to_csv(
        alerts_file,
        index=False,
    )

    regional.to_csv(
        regional_file,
        index=False,
    )

    seasonal.to_csv(
        seasonal_file,
        index=False,
    )

    monthly.to_csv(
        monthly_file,
        index=False,
    )

    distribution.to_csv(
        distribution_file,
        index=False,
    )

    errors.to_csv(
        errors_file,
        index=False,
    )

    metrics.to_csv(
        metrics_file,
        index=False,
    )

    # --------------------------------------------------------
    # 14. Model card
    # --------------------------------------------------------

    model_card_text = create_model_card(
        metrics
    )

    model_card_file = (
        OUTPUT_DIR
        / "FINAL_MODEL_CARD.md"
    )

    with open(
        model_card_file,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            model_card_text
        )

    # --------------------------------------------------------
    # 15. Excel
    # --------------------------------------------------------

    save_excel(
        metrics,
        distribution,
        regional,
        seasonal,
        monthly,
        alerts,
        errors,
        final_predictions,
    )

    # --------------------------------------------------------
    # 16. Final summary
    # --------------------------------------------------------

    m = metrics.iloc[
        0
    ]

    print()
    print("=" * 70)
    print("3.11 FINAL DEPLOYMENT SUMMARY")
    print("=" * 70)

    print(
        "MODEL: XGBoost + Sigmoid Calibration"
    )

    print(
        f"POLICY THRESHOLD: {FINAL_THRESHOLD:.2f}"
    )

    print(
        f"TEST OBSERVATIONS: {int(m['observations'])}"
    )

    print(
        f"TEST EVENTS: {int(m['events'])}"
    )

    print(
        f"PR-AUC: {m['pr_auc']:.6f}"
    )

    print(
        f"ROC-AUC: {m['roc_auc']:.6f}"
    )

    print(
        f"PRECISION: {m['precision']:.6f}"
    )

    print(
        f"RECALL: {m['recall']:.6f}"
    )

    print(
        f"F1: {m['f1']:.6f}"
    )

    print(
        f"ALERT RATE: {m['alert_rate']:.6f}"
    )

    print(
        f"TOTAL ALERTS: {int(m['alerts'])}"
    )

    print(
        f"TRUE POSITIVE: {int(m['true_positive'])}"
    )

    print(
        f"FALSE POSITIVE: {int(m['false_positive'])}"
    )

    print(
        f"FALSE NEGATIVE: {int(m['false_negative'])}"
    )

    print(
        f"TRUE NEGATIVE: {int(m['true_negative'])}"
    )

    # --------------------------------------------------------
    # 17. Saved files
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    for path in sorted(
        OUTPUT_DIR.iterdir()
    ):

        if path.is_file():

            print(
                path
            )

    print()
    print("=" * 70)
    print("3.11 FINAL PREDICTION & DEPLOYMENT REPORT COMPLETE")
    print("=" * 70)

    print()
    print(
        "NEXT STAGE: 3.12 MODEL MONITORING & DRIFT VALIDATION"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()