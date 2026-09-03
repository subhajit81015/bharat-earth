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
    brier_score_loss,
)

warnings.filterwarnings("ignore")


# ============================================================
# 3.12 MODEL MONITORING & DRIFT VALIDATION
# ============================================================

print("=" * 70)
print("3.12 MODEL MONITORING & DRIFT VALIDATION")
print("=" * 70)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "data" / "features"

FINAL_MODEL_DIR = FEATURE_DIR / "final_model"

DEPLOYMENT_DIR = FEATURE_DIR / "deployment"

MONITORING_DIR = FEATURE_DIR / "monitoring"

MONITORING_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# INPUT FILES
# ============================================================

PREDICTIONS_FILE = (
    FINAL_MODEL_DIR
    / "final_predictions.csv"
)

DEPLOYMENT_FILE = (
    DEPLOYMENT_DIR
    / "final_risk_predictions.csv"
)

FEATURE_FILE = (
    FEATURE_DIR
    / "ml_dataset_v2.csv"
)

TARGET_FILE = (
    FEATURE_DIR
    / "severe_anomaly_target.csv"
)


# ============================================================
# POLICY CONFIGURATION
# ============================================================

POLICY_THRESHOLD = 0.09

PSI_STABLE = 0.10
PSI_MODERATE = 0.25

MIN_GROUP_SIZE = 20


# ============================================================
# FEATURE LIST
# ============================================================

MONITORED_NUMERIC_FEATURES = [
    "rainfall_mm",
    "rainfall_3m",
    "rainfall_6m",
    "rainfall_12m",
    "historical_monthly_mean",
    "rainfall_anomaly",
    "rainfall_anomaly_pct",
    "rainfall_deficit_mm",
    "rainfall_zscore",
    "rainfall_lag_1m",
    "rainfall_lag_2m",
    "rainfall_lag_3m",
    "rainfall_prev_3m",
    "rainfall_prev_6m",
    "rainfall_prev_12m",
    "rainfall_trend_3m",
]


VALID_SEASONS = {
    "WINTER",
    "PRE_MONSOON",
    "MONSOON",
    "POST_MONSOON",
    "UNKNOWN",
}


# ============================================================
# PRINT HELPERS
# ============================================================

def print_section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_table(
    df,
    max_rows=None,
):

    if df is None or len(df) == 0:

        print("NO DATA")

        return

    display_df = df

    if max_rows is not None:

        display_df = df.head(
            max_rows
        )

    print(
        display_df.to_string(
            index=False
        )
    )


# ============================================================
# SAFE METRICS
# ============================================================

def safe_precision(
    y_true,
    y_pred,
):

    try:

        return precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )

    except Exception:

        return np.nan


def safe_recall(
    y_true,
    y_pred,
):

    try:

        return recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )

    except Exception:

        return np.nan


def safe_f1(
    y_true,
    y_pred,
):

    try:

        return f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

    except Exception:

        return np.nan


def safe_pr_auc(
    y_true,
    probability,
):

    try:

        if len(
            np.unique(y_true)
        ) < 2:

            return np.nan

        return average_precision_score(
            y_true,
            probability,
        )

    except Exception:

        return np.nan


def safe_roc_auc(
    y_true,
    probability,
):

    try:

        if len(
            np.unique(y_true)
        ) < 2:

            return np.nan

        return roc_auc_score(
            y_true,
            probability,
        )

    except Exception:

        return np.nan


def safe_brier(
    y_true,
    probability,
):

    try:

        return brier_score_loss(
            y_true,
            np.clip(
                probability,
                0,
                1,
            ),
        )

    except Exception:

        return np.nan


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print_section(
        "LOADING MONITORING DATA"
    )

    # --------------------------------------------------------
    # Deployment file preferred
    # --------------------------------------------------------

    if DEPLOYMENT_FILE.exists():

        input_file = DEPLOYMENT_FILE

        print(
            "USING DEPLOYMENT FILE:"
        )

    elif PREDICTIONS_FILE.exists():

        input_file = PREDICTIONS_FILE

        print(
            "USING FINAL PREDICTIONS FILE:"
        )

    else:

        raise FileNotFoundError(
            "\nPrediction file not found.\n\n"
            f"Checked:\n"
            f"{DEPLOYMENT_FILE}\n"
            f"{PREDICTIONS_FILE}"
        )

    print(
        input_file
    )

    df = pd.read_csv(
        input_file
    )

    print(
        "ROWS:",
        len(df)
    )

    print(
        "COLUMNS:",
        list(df.columns)
    )

    # --------------------------------------------------------
    # Feature dataset
    # --------------------------------------------------------

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"\nFeature file not found:\n"
            f"{FEATURE_FILE}"
        )

    features = pd.read_csv(
        FEATURE_FILE
    )

    print()
    print(
        "FEATURE DATASET:",
        FEATURE_FILE
    )

    print(
        "FEATURE ROWS:",
        len(features)
    )

    # --------------------------------------------------------
    # Target dataset
    # --------------------------------------------------------

    target = None

    if TARGET_FILE.exists():

        target = pd.read_csv(
            TARGET_FILE
        )

        print(
            "TARGET DATASET:",
            TARGET_FILE
        )

        print(
            "TARGET ROWS:",
            len(target)
        )

    return (
        df,
        features,
        target,
    )


# ============================================================
# STANDARDIZE PREDICTION DATA
# ============================================================

def standardize_predictions(
    df,
):

    df = df.copy()

    # --------------------------------------------------------
    # Subdivision
    # --------------------------------------------------------

    if "subdivision" in df.columns:

        df["subdivision"] = (
            df["subdivision"]
            .astype("string")
            .str.strip()
        )

    # --------------------------------------------------------
    # Year
    # --------------------------------------------------------

    if "year" in df.columns:

        df["year"] = pd.to_numeric(
            df["year"],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Month
    # --------------------------------------------------------

    if "month" in df.columns:

        df["month"] = pd.to_numeric(
            df["month"],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Actual
    # --------------------------------------------------------

    if "actual" not in df.columns:

        raise ValueError(
            "Prediction file does not contain 'actual'."
        )

    df["actual"] = pd.to_numeric(
        df["actual"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability_candidates = [
        "risk_probability",
        "final_probability",
        "sigmoid_probability",
        "isotonic_probability",
        "raw_probability",
    ]

    probability_column = None

    for column in probability_candidates:

        if column in df.columns:

            probability_column = column

            break

    if probability_column is None:

        raise ValueError(
            "No probability column found.\n"
            f"Expected one of: "
            f"{probability_candidates}"
        )

    df[
        "monitor_probability"
    ] = pd.to_numeric(
        df[
            probability_column
        ],
        errors="coerce",
    )

    df[
        "monitor_probability"
    ] = (
        df[
            "monitor_probability"
        ]
        .clip(
            0,
            1,
        )
    )

    print()
    print(
        "PROBABILITY COLUMN:",
        probability_column
    )

    # --------------------------------------------------------
    # Alert
    # --------------------------------------------------------

    if "risk_alert" in df.columns:

        df[
            "monitor_alert"
        ] = pd.to_numeric(
            df[
                "risk_alert"
            ],
            errors="coerce",
        ).fillna(
            0
        ).astype(
            int
        )

    elif "final_alert" in df.columns:

        df[
            "monitor_alert"
        ] = pd.to_numeric(
            df[
                "final_alert"
            ],
            errors="coerce",
        ).fillna(
            0
        ).astype(
            int
        )

    else:

        df[
            "monitor_alert"
        ] = (
            df[
                "monitor_probability"
            ]
            >= POLICY_THRESHOLD
        ).astype(
            int
        )

    # --------------------------------------------------------
    # Policy threshold
    # --------------------------------------------------------

    df[
        "monitor_threshold"
    ] = POLICY_THRESHOLD

    return df


# ============================================================
# STANDARDIZE FEATURE DATA
# ============================================================

def standardize_features(
    features,
):

    features = features.copy()

    if "subdivision" in features.columns:

        features[
            "subdivision"
        ] = (
            features[
                "subdivision"
            ]
            .astype("string")
            .str.strip()
        )

    if "year" in features.columns:

        features[
            "year"
        ] = pd.to_numeric(
            features[
                "year"
            ],
            errors="coerce",
        )

    if "month" in features.columns:

        features[
            "month"
        ] = pd.to_numeric(
            features[
                "month"
            ],
            errors="coerce",
        )

    if "season" in features.columns:

        features[
            "season"
        ] = (
            features[
                "season"
            ]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    return features


# ============================================================
# SEASON RECOVERY
# ============================================================

def recover_season(
    df,
    features,
):

    print_section(
        "SEASON VALIDATION"
    )

    df = df.copy()

    # ========================================================
    # NORMALIZE MAIN DATASET
    # ========================================================

    if "subdivision" in df.columns:

        df[
            "subdivision"
        ] = (
            df[
                "subdivision"
            ]
            .astype("string")
            .str.strip()
        )

    df[
        "year"
    ] = pd.to_numeric(
        df[
            "year"
        ],
        errors="coerce",
    )

    df[
        "month"
    ] = pd.to_numeric(
        df[
            "month"
        ],
        errors="coerce",
    )

    # ========================================================
    # NORMALIZE EXISTING SEASON
    # ========================================================

    if "season" in df.columns:

        df[
            "season"
        ] = (
            df[
                "season"
            ]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    else:

        df[
            "season"
        ] = pd.Series(
            pd.NA,
            index=df.index,
            dtype="string",
        )

    # ========================================================
    # CREATE RECOVERED SEASON
    # ========================================================

    recovered = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    existing = df[
        "season"
    ]

    valid_existing = existing.isin(
        {
            "WINTER",
            "PRE_MONSOON",
            "MONSOON",
            "POST_MONSOON",
        }
    )

    recovered.loc[
        valid_existing
    ] = existing.loc[
        valid_existing
    ]

    print(
        "VALID EXISTING SEASONS:",
        int(
            valid_existing.sum()
        )
    )

    # ========================================================
    # DERIVE FROM MONTH
    # ========================================================

    month = df[
        "month"
    ]

    valid_month = month.between(
        1,
        12,
    )

    derived = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    derived.loc[
        month.isin(
            [
                12,
                1,
                2,
            ]
        )
    ] = "WINTER"

    derived.loc[
        month.isin(
            [
                3,
                4,
                5,
            ]
        )
    ] = "PRE_MONSOON"

    derived.loc[
        month.isin(
            [
                6,
                7,
                8,
                9,
            ]
        )
    ] = "MONSOON"

    derived.loc[
        month.isin(
            [
                10,
                11,
            ]
        )
    ] = "POST_MONSOON"

    fill_from_month = (
        recovered.isna()
        &
        valid_month
        &
        derived.notna()
    )

    recovered.loc[
        fill_from_month
    ] = derived.loc[
        fill_from_month
    ]

    print(
        "DERIVED FROM MONTH:",
        int(
            fill_from_month.sum()
        )
    )

    # ========================================================
    # FEATURE DATA FALLBACK
    # ========================================================

    required_columns = {
        "subdivision",
        "year",
        "month",
        "season",
    }

    if required_columns.issubset(
        set(features.columns)
    ):

        lookup = features[
            [
                "subdivision",
                "year",
                "month",
                "season",
            ]
        ].copy()

        # ----------------------------------------------------
        # CRITICAL:
        # BOTH DATASETS USE STRING SUBDIVISION
        # ----------------------------------------------------

        lookup[
            "subdivision"
        ] = (
            lookup[
                "subdivision"
            ]
            .astype("string")
            .str.strip()
        )

        lookup[
            "year"
        ] = pd.to_numeric(
            lookup[
                "year"
            ],
            errors="coerce",
        )

        lookup[
            "month"
        ] = pd.to_numeric(
            lookup[
                "month"
            ],
            errors="coerce",
        )

        lookup[
            "season"
        ] = (
            lookup[
                "season"
            ]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        lookup = lookup[
            lookup[
                "season"
            ].isin(
                {
                    "WINTER",
                    "PRE_MONSOON",
                    "MONSOON",
                    "POST_MONSOON",
                }
            )
        ].copy()

        # ----------------------------------------------------
        # Remove duplicate keys
        # ----------------------------------------------------

        lookup = lookup.drop_duplicates(
            subset=[
                "subdivision",
                "year",
                "month",
            ],
            keep="first",
        )

        # ----------------------------------------------------
        # Main temporary keys
        # ----------------------------------------------------

        temp = df[
            [
                "subdivision",
                "year",
                "month",
            ]
        ].copy()

        temp[
            "_row_id"
        ] = np.arange(
            len(temp)
        )

        # ----------------------------------------------------
        # SAFE MERGE
        # ----------------------------------------------------

        temp = temp.merge(
            lookup,
            on=[
                "subdivision",
                "year",
                "month",
            ],
            how="left",
            sort=False,
        )

        temp = (
            temp
            .sort_values(
                "_row_id"
            )
            .reset_index(
                drop=True
            )
        )

        feature_season = (
            temp[
                "season"
            ]
            .astype("string")
        )

        fill_from_features = (
            recovered.isna()
            &
            feature_season.isin(
                {
                    "WINTER",
                    "PRE_MONSOON",
                    "MONSOON",
                    "POST_MONSOON",
                }
            )
        )

        recovered.loc[
            fill_from_features
        ] = feature_season.loc[
            fill_from_features
        ].values

        print(
            "RECOVERED FROM FEATURE DATA:",
            int(
                fill_from_features.sum()
            )
        )

    # ========================================================
    # UNKNOWN
    # ========================================================

    unresolved = recovered.isna()

    print(
        "UNRESOLVED BEFORE UNKNOWN:",
        int(
            unresolved.sum()
        )
    )

    recovered.loc[
        unresolved
    ] = "UNKNOWN"

    df[
        "season"
    ] = recovered.astype(
        "string"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    invalid = ~df[
        "season"
    ].isin(
        VALID_SEASONS
    )

    if invalid.any():

        raise ValueError(
            "Invalid season values remain:\n"
            + str(
                df.loc[
                    invalid,
                    "season"
                ].unique()
            )
        )

    print()
    print(
        "FINAL SEASON DISTRIBUTION:"
    )

    print(
        df[
            "season"
        ].value_counts(
            dropna=False
        )
    )

    print()

    print(
        "UNKNOWN MONTH COUNT:",
        int(
            (
                df[
                    "month"
                ]
                == 0
            ).sum()
        )
    )

    print(
        "SEASON VALIDATION: PASS"
    )

    return df


# ============================================================
# PSI
# ============================================================

def calculate_psi(
    reference,
    current,
    bins=10,
):

    reference = pd.to_numeric(
        pd.Series(
            reference
        ),
        errors="coerce",
    )

    current = pd.to_numeric(
        pd.Series(
            current
        ),
        errors="coerce",
    )

    reference = reference[
        np.isfinite(
            reference
        )
    ]

    current = current[
        np.isfinite(
            current
        )
    ]

    if (
        len(reference) < 20
        or len(current) < 20
    ):

        return np.nan

    try:

        quantiles = np.linspace(
            0,
            1,
            bins + 1,
        )

        edges = np.unique(
            reference.quantile(
                quantiles
            ).values
        )

        if len(edges) < 3:

            return 0.0

        edges[0] = -np.inf
        edges[-1] = np.inf

        reference_bins = pd.cut(
            reference,
            bins=edges,
            include_lowest=True,
        )

        current_bins = pd.cut(
            current,
            bins=edges,
            include_lowest=True,
        )

        reference_pct = (
            reference_bins
            .value_counts(
                normalize=True,
                sort=False,
            )
        )

        current_pct = (
            current_bins
            .value_counts(
                normalize=True,
                sort=False,
            )
        )

        current_pct = (
            current_pct
            .reindex(
                reference_pct.index
            )
            .fillna(0)
        )

        reference_pct = (
            reference_pct
            .fillna(0)
        )

        epsilon = 1e-6

        reference_pct = (
            reference_pct
            .clip(
                lower=epsilon
            )
        )

        current_pct = (
            current_pct
            .clip(
                lower=epsilon
            )
        )

        psi = np.sum(
            (
                current_pct
                - reference_pct
            )
            *
            np.log(
                current_pct
                / reference_pct
            )
        )

        return float(
            psi
        )

    except Exception:

        return np.nan


def classify_psi(
    value,
):

    if pd.isna(value):

        return "UNAVAILABLE"

    if value < PSI_STABLE:

        return "STABLE"

    if value < PSI_MODERATE:

        return "MODERATE_DRIFT"

    return "SIGNIFICANT_DRIFT"


# ============================================================
# FEATURE DRIFT
# ============================================================

def calculate_feature_drift(
    features,
):

    print_section(
        "FEATURE DRIFT MONITORING"
    )

    if "year" not in features.columns:

        print(
            "Year column unavailable."
        )

        return pd.DataFrame()

    years = (
        pd.to_numeric(
            features[
                "year"
            ],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    years = sorted(
        years
    )

    if len(years) < 2:

        print(
            "Not enough years for drift analysis."
        )

        return pd.DataFrame()

    latest_year = years[-1]

    reference = features[
        pd.to_numeric(
            features[
                "year"
            ],
            errors="coerce",
        )
        < latest_year
    ].copy()

    current = features[
        pd.to_numeric(
            features[
                "year"
            ],
            errors="coerce",
        )
        == latest_year
    ].copy()

    print(
        "REFERENCE YEARS:",
        f"<= {latest_year - 1}"
    )

    print(
        "CURRENT YEAR:",
        latest_year
    )

    print(
        "REFERENCE ROWS:",
        len(reference)
    )

    print(
        "CURRENT ROWS:",
        len(current)
    )

    rows = []

    for feature in MONITORED_NUMERIC_FEATURES:

        if (
            feature not in reference.columns
            or feature not in current.columns
        ):

            continue

        reference_values = pd.to_numeric(
            reference[
                feature
            ],
            errors="coerce",
        )

        current_values = pd.to_numeric(
            current[
                feature
            ],
            errors="coerce",
        )

        psi = calculate_psi(
            reference_values,
            current_values,
        )

        reference_mean = (
            reference_values.mean()
        )

        current_mean = (
            current_values.mean()
        )

        mean_change = (
            current_mean
            - reference_mean
        )

        mean_change_pct = (
            mean_change
            /
            (
                abs(
                    reference_mean
                )
                + 1e-9
            )
        )

        rows.append(
            {
                "feature":
                    feature,

                "reference_mean":
                    reference_mean,

                "current_mean":
                    current_mean,

                "reference_std":
                    reference_values.std(),

                "current_std":
                    current_values.std(),

                "mean_change":
                    mean_change,

                "mean_change_pct":
                    mean_change_pct,

                "reference_missing_rate":
                    reference_values.isna().mean(),

                "current_missing_rate":
                    current_values.isna().mean(),

                "psi":
                    psi,

                "psi_status":
                    classify_psi(
                        psi
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if len(result) > 0:

        result = result.sort_values(
            "psi",
            ascending=False,
        )

    print()

    print_table(
        result,
        max_rows=20,
    )

    return result


# ============================================================
# PROBABILITY MONITORING
# ============================================================

def probability_monitoring(
    df,
):

    print_section(
        "PROBABILITY DISTRIBUTION MONITORING"
    )

    probability = df[
        "monitor_probability"
    ]

    alert = (
        probability
        >= POLICY_THRESHOLD
    )

    result = pd.DataFrame(
        [
            {
                "observations":
                    len(df),

                "mean_probability":
                    probability.mean(),

                "median_probability":
                    probability.median(),

                "std_probability":
                    probability.std(),

                "minimum_probability":
                    probability.min(),

                "p25_probability":
                    probability.quantile(
                        0.25
                    ),

                "p75_probability":
                    probability.quantile(
                        0.75
                    ),

                "p90_probability":
                    probability.quantile(
                        0.90
                    ),

                "p95_probability":
                    probability.quantile(
                        0.95
                    ),

                "p99_probability":
                    probability.quantile(
                        0.99
                    ),

                "maximum_probability":
                    probability.max(),

                "policy_threshold":
                    POLICY_THRESHOLD,

                "alerts":
                    int(
                        alert.sum()
                    ),

                "alert_rate":
                    alert.mean(),
            }
        ]
    )

    print_table(
        result
    )

    return result


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

def overall_performance(
    df,
):

    print_section(
        "OVERALL MODEL PERFORMANCE"
    )

    y = (
        df[
            "actual"
        ]
        .astype(int)
    )

    probability = (
        df[
            "monitor_probability"
        ]
    )

    alert = (
        probability
        >= POLICY_THRESHOLD
    ).astype(int)

    tn = int(
        (
            (y == 0)
            &
            (alert == 0)
        ).sum()
    )

    fp = int(
        (
            (y == 0)
            &
            (alert == 1)
        ).sum()
    )

    fn = int(
        (
            (y == 1)
            &
            (alert == 0)
        ).sum()
    )

    tp = int(
        (
            (y == 1)
            &
            (alert == 1)
        ).sum()
    )

    result = pd.DataFrame(
        [
            {
                "observations":
                    len(df),

                "events":
                    int(y.sum()),

                "event_rate":
                    y.mean(),

                "alerts":
                    int(alert.sum()),

                "alert_rate":
                    alert.mean(),

                "true_positive":
                    tp,

                "false_positive":
                    fp,

                "false_negative":
                    fn,

                "true_negative":
                    tn,

                "precision":
                    safe_precision(
                        y,
                        alert,
                    ),

                "recall":
                    safe_recall(
                        y,
                        alert,
                    ),

                "f1":
                    safe_f1(
                        y,
                        alert,
                    ),

                "pr_auc":
                    safe_pr_auc(
                        y,
                        probability,
                    ),

                "roc_auc":
                    safe_roc_auc(
                        y,
                        probability,
                    ),

                "brier_score":
                    safe_brier(
                        y,
                        probability,
                    ),

                "false_positive_rate":
                    (
                        fp
                        /
                        (
                            fp
                            + tn
                            + 1e-12
                        )
                    ),
            }
        ]
    )

    print_table(
        result
    )

    return result


# ============================================================
# YEARLY TARGET DRIFT
# ============================================================

def yearly_target_drift(
    df,
):

    print_section(
        "YEARLY TARGET DRIFT"
    )

    rows = []

    for year, group in (
        df.groupby(
            "year",
            dropna=False,
        )
    ):

        y = group[
            "actual"
        ].astype(int)

        rows.append(
            {
                "year":
                    year,

                "observations":
                    len(group),

                "events":
                    int(y.sum()),

                "event_rate":
                    y.mean(),

                "mean_probability":
                    group[
                        "monitor_probability"
                    ].mean(),

                "alert_rate":
                    (
                        group[
                            "monitor_probability"
                        ]
                        >= POLICY_THRESHOLD
                    ).mean(),
            }
        )

    result = pd.DataFrame(
        rows
    ).sort_values(
        "year"
    )

    if len(result) > 0:

        global_rate = (
            df[
                "actual"
            ].mean()
        )

        result[
            "event_rate_vs_global"
        ] = (
            result[
                "event_rate"
            ]
            /
            (
                global_rate
                + 1e-12
            )
        )

        result[
            "absolute_event_rate_change"
        ] = (
            result[
                "event_rate"
            ]
            - global_rate
        )

    print_table(
        result,
        max_rows=30,
    )

    return result


# ============================================================
# SEASONAL MONITORING
# ============================================================

def seasonal_monitoring(
    df,
):

    print_section(
        "SEASONAL MONITORING"
    )

    rows = []

    for season, group in (
        df.groupby(
            "season",
            dropna=False,
        )
    ):

        y = group[
            "actual"
        ].astype(int)

        probability = group[
            "monitor_probability"
        ]

        alert = (
            probability
            >= POLICY_THRESHOLD
        ).astype(int)

        rows.append(
            {
                "season":
                    season,

                "observations":
                    len(group),

                "events":
                    int(y.sum()),

                "event_rate":
                    y.mean(),

                "alerts":
                    int(alert.sum()),

                "alert_rate":
                    alert.mean(),

                "precision":
                    safe_precision(
                        y,
                        alert,
                    ),

                "recall":
                    safe_recall(
                        y,
                        alert,
                    ),

                "f1":
                    safe_f1(
                        y,
                        alert,
                    ),

                "pr_auc":
                    safe_pr_auc(
                        y,
                        probability,
                    ),

                "roc_auc":
                    safe_roc_auc(
                        y,
                        probability,
                    ),

                "average_probability":
                    probability.mean(),

                "maximum_probability":
                    probability.max(),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if len(result) > 0:

        result[
            "sample_warning"
        ] = np.where(
            result[
                "observations"
            ]
            < MIN_GROUP_SIZE,
            "SMALL_SAMPLE",
            "",
        )

    print_table(
        result
    )

    return result


# ============================================================
# MONTHLY MONITORING
# ============================================================

def monthly_monitoring(
    df,
):

    print_section(
        "MONTHLY MONITORING"
    )

    rows = []

    for month, group in (
        df.groupby(
            "month",
            dropna=False,
        )
    ):

        y = group[
            "actual"
        ].astype(int)

        probability = group[
            "monitor_probability"
        ]

        alert = (
            probability
            >= POLICY_THRESHOLD
        ).astype(int)

        rows.append(
            {
                "month":
                    month,

                "observations":
                    len(group),

                "events":
                    int(y.sum()),

                "event_rate":
                    y.mean(),

                "alerts":
                    int(alert.sum()),

                "alert_rate":
                    alert.mean(),

                "precision":
                    safe_precision(
                        y,
                        alert,
                    ),

                "recall":
                    safe_recall(
                        y,
                        alert,
                    ),

                "f1":
                    safe_f1(
                        y,
                        alert,
                    ),

                "average_probability":
                    probability.mean(),
            }
        )

    result = pd.DataFrame(
        rows
    ).sort_values(
        "month"
    )

    print_table(
        result
    )

    return result


# ============================================================
# REGIONAL MONITORING
# ============================================================

def regional_monitoring(
    df,
):

    print_section(
        "REGIONAL MONITORING"
    )

    rows = []

    for subdivision, group in (
        df.groupby(
            "subdivision",
            dropna=False,
        )
    ):

        y = group[
            "actual"
        ].astype(int)

        probability = group[
            "monitor_probability"
        ]

        alert = (
            probability
            >= POLICY_THRESHOLD
        ).astype(int)

        rows.append(
            {
                "subdivision":
                    subdivision,

                "observations":
                    len(group),

                "events":
                    int(y.sum()),

                "event_rate":
                    y.mean(),

                "alerts":
                    int(alert.sum()),

                "alert_rate":
                    alert.mean(),

                "precision":
                    safe_precision(
                        y,
                        alert,
                    ),

                "recall":
                    safe_recall(
                        y,
                        alert,
                    ),

                "f1":
                    safe_f1(
                        y,
                        alert,
                    ),

                "pr_auc":
                    safe_pr_auc(
                        y,
                        probability,
                    ),

                "roc_auc":
                    safe_roc_auc(
                        y,
                        probability,
                    ),

                "average_probability":
                    probability.mean(),

                "maximum_probability":
                    probability.max(),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if len(result) > 0:

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

        result = result.sort_values(
            "risk_rank"
        )

    print_table(
        result,
        max_rows=30,
    )

    return result


# ============================================================
# RECENT VS BASELINE
# ============================================================

def recent_vs_baseline(
    df,
):

    print_section(
        "RECENT VS BASELINE"
    )

    if "year" not in df.columns:

        return pd.DataFrame()

    years = (
        pd.to_numeric(
            df[
                "year"
            ],
            errors="coerce",
        )
        .dropna()
        .unique()
    )

    years = sorted(
        years
    )

    if len(years) < 2:

        print(
            "Not enough years."
        )

        return pd.DataFrame()

    latest_year = years[-1]

    previous_year = years[-2]

    recent = df[
        df[
            "year"
        ]
        == latest_year
    ].copy()

    baseline = df[
        df[
            "year"
        ]
        < latest_year
    ].copy()

    if len(baseline) == 0:

        baseline = df[
            df[
                "year"
            ]
            == previous_year
        ].copy()

    def summarize(
        group
    ):

        y = group[
            "actual"
        ].astype(int)

        probability = group[
            "monitor_probability"
        ]

        alert = (
            probability
            >= POLICY_THRESHOLD
        ).astype(int)

        return {
            "observations":
                len(group),

            "event_rate":
                y.mean(),

            "mean_probability":
                probability.mean(),

            "alert_rate":
                alert.mean(),

            "precision":
                safe_precision(
                    y,
                    alert,
                ),

            "recall":
                safe_recall(
                    y,
                    alert,
                ),

            "f1":
                safe_f1(
                    y,
                    alert,
                ),

            "pr_auc":
                safe_pr_auc(
                    y,
                    probability,
                ),

            "roc_auc":
                safe_roc_auc(
                    y,
                    probability,
                ),

            "brier_score":
                safe_brier(
                    y,
                    probability,
                ),
        }

    baseline_summary = summarize(
        baseline
    )

    recent_summary = summarize(
        recent
    )

    result = pd.DataFrame(
        [
            {
                "period":
                    "BASELINE",

                **baseline_summary,
            },

            {
                "period":
                    f"RECENT_{latest_year}",

                **recent_summary,
            },
        ]
    )

    print_table(
        result
    )

    return result


# ============================================================
# ALERT STABILITY
# ============================================================

def alert_stability(
    df,
):

    print_section(
        "ALERT RATE STABILITY"
    )

    rows = []

    for year, group in (
        df.groupby(
            "year",
            dropna=False,
        )
    ):

        probability = group[
            "monitor_probability"
        ]

        alert = (
            probability
            >= POLICY_THRESHOLD
        )

        rows.append(
            {
                "year":
                    year,

                "observations":
                    len(group),

                "alerts":
                    int(
                        alert.sum()
                    ),

                "alert_rate":
                    alert.mean(),

                "mean_probability":
                    probability.mean(),

                "p95_probability":
                    probability.quantile(
                        0.95
                    ),

                "maximum_probability":
                    probability.max(),
            }
        )

    result = pd.DataFrame(
        rows
    ).sort_values(
        "year"
    )

    print_table(
        result,
        max_rows=30,
    )

    return result


# ============================================================
# UNKNOWN / INVALID MONTH MONITORING
# ============================================================

def unknown_month_monitoring(
    df,
):

    print_section(
        "UNKNOWN / INVALID MONTH MONITORING"
    )

    invalid_month = (
        ~df[
            "month"
        ].between(
            1,
            12,
        )
    )

    group = df[
        invalid_month
    ].copy()

    if len(group) == 0:

        result = pd.DataFrame(
            [
                {
                    "observations":
                        0,

                    "events":
                        0,

                    "event_rate":
                        0,

                    "alerts":
                        0,

                    "alert_rate":
                        0,

                    "average_probability":
                        np.nan,
                }
            ]
        )

    else:

        y = group[
            "actual"
        ].astype(int)

        probability = group[
            "monitor_probability"
        ]

        alert = (
            probability
            >= POLICY_THRESHOLD
        )

        result = pd.DataFrame(
            [
                {
                    "observations":
                        len(group),

                    "events":
                        int(
                            y.sum()
                        ),

                    "event_rate":
                        y.mean(),

                    "alerts":
                        int(
                            alert.sum()
                        ),

                    "alert_rate":
                        alert.mean(),

                    "average_probability":
                        probability.mean(),

                    "maximum_probability":
                        probability.max(),
                }
            ]
        )

    print_table(
        result
    )

    return result


# ============================================================
# DRIFT SUMMARY
# ============================================================

def create_drift_summary(
    feature_drift,
    performance,
    recent_comparison,
    probability,
):

    rows = []

    # --------------------------------------------------------
    # Feature PSI
    # --------------------------------------------------------

    if (
        feature_drift is not None
        and len(feature_drift) > 0
    ):

        significant_count = int(
            (
                feature_drift[
                    "psi"
                ]
                >= PSI_MODERATE
            ).sum()
        )

        moderate_count = int(
            (
                (
                    feature_drift[
                        "psi"
                    ]
                    >= PSI_STABLE
                )
                &
                (
                    feature_drift[
                        "psi"
                    ]
                    < PSI_MODERATE
                )
            ).sum()
        )

        max_psi = feature_drift[
            "psi"
        ].max()

        if significant_count > 0:

            status = (
                "SIGNIFICANT_DRIFT"
            )

        elif moderate_count > 0:

            status = (
                "MODERATE_DRIFT"
            )

        else:

            status = "STABLE"

        rows.append(
            {
                "monitor":
                    "FEATURE_PSI",

                "value":
                    max_psi,

                "status":
                    status,

                "details":
                    (
                        f"{significant_count} "
                        f"significant, "
                        f"{moderate_count} "
                        f"moderate"
                    ),
            }
        )

    # --------------------------------------------------------
    # Alert rate
    # --------------------------------------------------------

    if len(probability) > 0:

        alert_rate = (
            probability[
                "alert_rate"
            ].iloc[0]
        )

        if alert_rate > 0.30:

            alert_status = (
                "HIGH_ALERT_RATE"
            )

        else:

            alert_status = "NORMAL"

        rows.append(
            {
                "monitor":
                    "ALERT_RATE",

                "value":
                    alert_rate,

                "status":
                    alert_status,

                "details":
                    (
                        f"Threshold="
                        f"{POLICY_THRESHOLD:.2f}"
                    ),
            }
        )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    if (
        performance is not None
        and len(performance) > 0
    ):

        f1 = performance[
            "f1"
        ].iloc[0]

        pr_auc = performance[
            "pr_auc"
        ].iloc[0]

        rows.append(
            {
                "monitor":
                    "F1",

                "value":
                    f1,

                "status":
                    (
                        "REVIEW"
                        if f1 < 0.15
                        else "PASS"
                    ),

                "details":
                    "Policy performance",
            }
        )

        rows.append(
            {
                "monitor":
                    "PR_AUC",

                "value":
                    pr_auc,

                "status":
                    (
                        "REVIEW"
                        if pr_auc < 0.10
                        else "PASS"
                    ),

                "details":
                    "Ranking performance",
            }
        )

    # --------------------------------------------------------
    # Recent event drift
    # --------------------------------------------------------

    if (
        recent_comparison is not None
        and len(
            recent_comparison
        ) == 2
    ):

        baseline = (
            recent_comparison.iloc[0]
        )

        recent = (
            recent_comparison.iloc[1]
        )

        event_change = (
            recent[
                "event_rate"
            ]
            -
            baseline[
                "event_rate"
            ]
        )

        alert_change = (
            recent[
                "alert_rate"
            ]
            -
            baseline[
                "alert_rate"
            ]
        )

        rows.append(
            {
                "monitor":
                    "EVENT_RATE_CHANGE",

                "value":
                    event_change,

                "status":
                    (
                        "REVIEW"
                        if abs(
                            event_change
                        ) > 0.03
                        else "STABLE"
                    ),

                "details":
                    "Recent minus baseline",
            }
        )

        rows.append(
            {
                "monitor":
                    "ALERT_RATE_CHANGE",

                "value":
                    alert_change,

                "status":
                    (
                        "REVIEW"
                        if abs(
                            alert_change
                        ) > 0.10
                        else "STABLE"
                    ),

                "details":
                    "Recent minus baseline",
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# DEPLOYMENT DECISION
# ============================================================

def deployment_decision(
    summary,
):

    significant = summary[
        summary[
            "status"
        ]
        == "SIGNIFICANT_DRIFT"
    ]

    review = summary[
        summary[
            "status"
        ].isin(
            [
                "REVIEW",
                "HIGH_ALERT_RATE",
                "MODERATE_DRIFT",
            ]
        )
    ]

    if len(significant) > 0:

        decision = (
            "RETRAIN_OR_INVESTIGATE"
        )

    elif len(review) > 0:

        decision = (
            "MONITOR_CLOSELY"
        )

    else:

        decision = (
            "STABLE_FOR_CONTINUED_MONITORING"
        )

    result = pd.DataFrame(
        [
            {
                "decision":
                    decision,

                "policy_threshold":
                    POLICY_THRESHOLD,

                "significant_issues":
                    len(significant),

                "review_items":
                    len(review),

                "monitoring_status":
                    (
                        "ATTENTION_REQUIRED"
                        if len(review) > 0
                        else "STABLE"
                    ),
            }
        ]
    )

    return result


# ============================================================
# SAVE CSV OUTPUTS
# ============================================================

def save_csv(
    name,
    dataframe,
):

    if dataframe is None:

        return None

    path = (
        MONITORING_DIR
        / name
    )

    dataframe.to_csv(
        path,
        index=False,
    )

    print(
        path
    )

    return path


# ============================================================
# SAVE EXCEL
# ============================================================

def save_excel(
    datasets,
):

    excel_file = (
        MONITORING_DIR
        / "model_monitoring_report.xlsx"
    )

    try:

        with pd.ExcelWriter(
            excel_file,
            engine="openpyxl",
        ) as writer:

            for sheet_name, dataframe in datasets.items():

                if dataframe is None:

                    continue

                if len(dataframe) == 0:

                    continue

                safe_sheet = (
                    sheet_name[:31]
                )

                dataframe.to_excel(
                    writer,
                    sheet_name=safe_sheet,
                    index=False,
                )

        print(
            excel_file
        )

        return excel_file

    except ImportError:

        print()
        print(
            "WARNING: openpyxl is not installed."
        )

        print(
            "CSV files will still be available."
        )

        print(
            "Install with:"
        )

        print(
            "pip install openpyxl"
        )

        return None


# ============================================================
# SAVE MARKDOWN REPORT
# ============================================================

def save_markdown_report(
    performance,
    probability,
    feature_drift,
    target_drift,
    seasonal,
    monthly,
    regional,
    recent,
    alert_history,
    unknown_months,
    summary,
    decision,
):

    report_file = (
        MONITORING_DIR
        / "MODEL_MONITORING_REPORT.md"
    )

    lines = []

    lines.append(
        "# Bharat Earth"
    )

    lines.append(
        "## 3.12 Model Monitoring & Drift Validation"
    )

    lines.append("")

    lines.append(
        "## Policy"
    )

    lines.append("")

    lines.append(
        f"- Threshold: `{POLICY_THRESHOLD:.2f}`"
    )

    lines.append("")

    lines.append(
        "## Deployment Decision"
    )

    lines.append("")

    if (
        decision is not None
        and len(decision) > 0
    ):

        lines.append(
            str(
                decision[
                    "decision"
                ].iloc[0]
            )
        )

    lines.append("")

    sections = [
        (
            "Overall Performance",
            performance,
        ),
        (
            "Probability Monitoring",
            probability,
        ),
        (
            "Drift Summary",
            summary,
        ),
        (
            "Feature Drift",
            feature_drift,
        ),
        (
            "Target Drift",
            target_drift,
        ),
        (
            "Seasonal Monitoring",
            seasonal,
        ),
        (
            "Monthly Monitoring",
            monthly,
        ),
        (
            "Regional Monitoring",
            regional,
        ),
        (
            "Recent vs Baseline",
            recent,
        ),
        (
            "Alert Stability",
            alert_history,
        ),
        (
            "Unknown Month Monitoring",
            unknown_months,
        ),
    ]

    for title, dataframe in sections:

        lines.append(
            f"## {title}"
        )

        lines.append("")

        if (
            dataframe is None
            or len(dataframe) == 0
        ):

            lines.append(
                "No data available."
            )

        else:

            try:

                lines.append(
                    dataframe.to_markdown(
                        index=False
                    )
                )

            except Exception:

                lines.append(
                    dataframe.to_string(
                        index=False
                    )
                )

        lines.append("")

    lines.append(
        "## Monitoring Interpretation"
    )

    lines.append("")

    lines.append(
        "- PSI < 0.10: stable"
    )

    lines.append(
        "- PSI 0.10-0.25: moderate drift"
    )

    lines.append(
        "- PSI >= 0.25: significant drift"
    )

    lines.append(
        "- High alert rate should trigger operational review."
    )

    lines.append(
        "- Significant feature drift should trigger investigation before retraining."
    )

    lines.append(
        "- Small seasonal/regional groups should not be interpreted as statistically reliable without additional data."
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(
                lines
            )
        )

    print(
        report_file
    )

    return report_file


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. LOAD
    # ========================================================

    (
        df,
        features,
        target_source,
    ) = load_data()

    # ========================================================
    # 2. STANDARDIZE
    # ========================================================

    df = standardize_predictions(
        df
    )

    features = standardize_features(
        features
    )

    # ========================================================
    # 3. RECOVER SEASON
    # ========================================================

    df = recover_season(
        df,
        features,
    )

    # ========================================================
    # 4. CLEAN MONITORING DATA
    # ========================================================

    before = len(df)

    df = df.dropna(
        subset=[
            "actual",
            "monitor_probability",
        ]
    ).copy()

    after = len(df)

    print()
    print(
        "ROWS REMOVED FOR INVALID TARGET/PROBABILITY:",
        before - after
    )

    df[
        "actual"
    ] = (
        df[
            "actual"
        ]
        .astype(int)
    )

    df[
        "monitor_probability"
    ] = (
        df[
            "monitor_probability"
        ]
        .clip(
            0,
            1,
        )
    )

    # ========================================================
    # 5. PROBABILITY
    # ========================================================

    probability = probability_monitoring(
        df
    )

    # ========================================================
    # 6. OVERALL PERFORMANCE
    # ========================================================

    performance = overall_performance(
        df
    )

    # ========================================================
    # 7. YEARLY TARGET DRIFT
    # ========================================================

    target_drift = yearly_target_drift(
        df
    )

    # ========================================================
    # 8. SEASONAL
    # ========================================================

    seasonal = seasonal_monitoring(
        df
    )

    # ========================================================
    # 9. MONTHLY
    # ========================================================

    monthly = monthly_monitoring(
        df
    )

    # ========================================================
    # 10. REGIONAL
    # ========================================================

    regional = regional_monitoring(
        df
    )

    # ========================================================
    # 11. RECENT VS BASELINE
    # ========================================================

    recent = recent_vs_baseline(
        df
    )

    # ========================================================
    # 12. ALERT STABILITY
    # ========================================================

    alert_history = alert_stability(
        df
    )

    # ========================================================
    # 13. UNKNOWN MONTH
    # ========================================================

    unknown_months = (
        unknown_month_monitoring(
            df
        )
    )

    # ========================================================
    # 14. FEATURE DRIFT
    # ========================================================

    feature_drift = calculate_feature_drift(
        features
    )

    # ========================================================
    # 15. DRIFT SUMMARY
    # ========================================================

    summary = create_drift_summary(
        feature_drift,
        performance,
        recent,
        probability,
    )

    # ========================================================
    # 16. DECISION
    # ========================================================

    decision = deployment_decision(
        summary
    )

    # ========================================================
    # 17. PRINT FINAL SUMMARY
    # ========================================================

    print_section(
        "3.12 DRIFT SUMMARY"
    )

    print_table(
        summary
    )

    print_section(
        "DEPLOYMENT DECISION"
    )

    print_table(
        decision
    )

    # ========================================================
    # 18. SAVE CSV FILES
    # ========================================================

    print_section(
        "SAVING MONITORING OUTPUTS"
    )

    save_csv(
        "monitoring_predictions.csv",
        df,
    )

    save_csv(
        "probability_monitoring.csv",
        probability,
    )

    save_csv(
        "performance_monitoring.csv",
        performance,
    )

    save_csv(
        "target_drift.csv",
        target_drift,
    )

    save_csv(
        "seasonal_monitoring.csv",
        seasonal,
    )

    save_csv(
        "monthly_monitoring.csv",
        monthly,
    )

    save_csv(
        "regional_monitoring.csv",
        regional,
    )

    save_csv(
        "recent_vs_baseline.csv",
        recent,
    )

    save_csv(
        "alert_stability.csv",
        alert_history,
    )

    save_csv(
        "unknown_month_monitoring.csv",
        unknown_months,
    )

    save_csv(
        "feature_drift.csv",
        feature_drift,
    )

    save_csv(
        "drift_summary.csv",
        summary,
    )

    save_csv(
        "deployment_decision.csv",
        decision,
    )

    # ========================================================
    # 19. EXCEL
    # ========================================================

    datasets = {
        "performance":
            performance,

        "probability":
            probability,

        "drift_summary":
            summary,

        "feature_drift":
            feature_drift,

        "target_drift":
            target_drift,

        "seasonal":
            seasonal,

        "monthly":
            monthly,

        "regional":
            regional,

        "recent_baseline":
            recent,

        "alert_stability":
            alert_history,

        "unknown_month":
            unknown_months,

        "decision":
            decision,
    }

    save_excel(
        datasets
    )

    # ========================================================
    # 20. MARKDOWN
    # ========================================================

    save_markdown_report(
        performance,
        probability,
        feature_drift,
        target_drift,
        seasonal,
        monthly,
        regional,
        recent,
        alert_history,
        unknown_months,
        summary,
        decision,
    )

    # ========================================================
    # 21. FINAL
    # ========================================================

    print()
    print("=" * 70)
    print(
        "3.12 MODEL MONITORING & DRIFT VALIDATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "OUTPUT DIRECTORY:"
    )

    print(
        MONITORING_DIR
    )

    print()
    print(
        "NEXT STAGE:"
    )

    print(
        "3.13 FINAL PROJECT VALIDATION & PRODUCTION READINESS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()