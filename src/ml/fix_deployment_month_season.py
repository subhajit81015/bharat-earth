from pathlib import Path
import pandas as pd
import numpy as np


# ================================================================
# 3.13.1 DEPLOYMENT MONTH / SEASON REPAIR
# FINAL PRODUCTION VERSION
#
# PURPOSE:
#   The deployment file contains month values 0..11.
#   This script converts:
#
#       0  -> January
#       1  -> February
#       ...
#       11 -> December
#
#   Then derives the correct meteorological season.
#
#   IMPORTANT:
#   - Original deployment file is NOT overwritten.
#   - An audit file is generated.
#   - Hard validation is performed before output.
# ================================================================


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(
    r"C:\Users\subha\Downloads\bharat-earth"
)

DEPLOYMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "deployment"
    / "final_risk_predictions.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "deployment"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "final_risk_predictions_fixed.csv"
)

AUDIT_FILE = (
    OUTPUT_DIR
    / "month_season_repair_audit.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "month_season_validation.csv"
)


# ================================================================
# EXPECTED VALUES
# ================================================================

EXPECTED_ROWS = 7668

VALID_MONTHS = set(range(1, 13))

ZERO_BASED_MONTHS = set(range(0, 12))

EXPECTED_SEASONS = {
    "WINTER",
    "PRE_MONSOON",
    "MONSOON",
    "POST_MONSOON",
}


# ================================================================
# SEASON MAPPING
# ================================================================

def month_to_season(month):
    """
    Convert conventional calendar month to season.

    1,2,12  -> WINTER
    3,4,5   -> PRE_MONSOON
    6,7,8,9 -> MONSOON
    10,11   -> POST_MONSOON
    """

    if pd.isna(month):
        return "UNKNOWN"

    month = int(month)

    if month in [12, 1, 2]:
        return "WINTER"

    if month in [3, 4, 5]:
        return "PRE_MONSOON"

    if month in [6, 7, 8, 9]:
        return "MONSOON"

    if month in [10, 11]:
        return "POST_MONSOON"

    return "UNKNOWN"


# ================================================================
# SAFE NUMERIC CONVERSION
# ================================================================

def convert_month_numeric(series):

    """
    Safely convert month column to numeric.

    Invalid strings become NaN.
    """

    return pd.to_numeric(
        series,
        errors="coerce"
    )


# ================================================================
# LOAD DEPLOYMENT DATA
# ================================================================

def load_deployment():

    print("=" * 70)
    print("3.13.1 DEPLOYMENT MONTH / SEASON REPAIR")
    print("=" * 70)

    print()
    print("LOADING DEPLOYMENT DATA")

    if not DEPLOYMENT_FILE.exists():

        raise FileNotFoundError(
            f"Deployment file not found:\n"
            f"{DEPLOYMENT_FILE}"
        )

    df = pd.read_csv(
        DEPLOYMENT_FILE,
        low_memory=False
    )

    print()
    print(
        "INPUT FILE:"
    )

    print(
        DEPLOYMENT_FILE
    )

    print()
    print(
        "ROWS:",
        len(df)
    )

    print()
    print(
        "COLUMNS:"
    )

    print(
        df.columns.tolist()
    )

    return df


# ================================================================
# INPUT VALIDATION
# ================================================================

def validate_input_schema(df):

    print()
    print("=" * 70)
    print("INPUT SCHEMA VALIDATION")
    print("=" * 70)

    required_columns = [
        "subdivision",
        "year",
        "month",
        "actual",
        "final_probability",
        "risk_probability",
        "risk_level",
        "risk_alert",
        "alert_priority",
        "policy_threshold",
        "prediction_status",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required deployment columns:\n"
            + "\n".join(missing)
        )

    print(
        "REQUIRED COLUMNS: PASS"
    )

    if len(df) != EXPECTED_ROWS:

        raise ValueError(
            f"Unexpected deployment row count.\n"
            f"Expected: {EXPECTED_ROWS}\n"
            f"Actual:   {len(df)}"
        )

    print(
        f"ROW COUNT: PASS ({len(df)})"
    )


# ================================================================
# ANALYZE ORIGINAL MONTH
# ================================================================

def analyze_original_month(df):

    print()
    print("=" * 70)
    print("ORIGINAL MONTH ANALYSIS")
    print("=" * 70)

    original = convert_month_numeric(
        df["month"]
    )

    print()
    print(
        "ORIGINAL MONTH VALUES:"
    )

    print(
        sorted(
            original
            .dropna()
            .unique()
            .tolist()
        )
    )

    print()
    print(
        "ORIGINAL MONTH DISTRIBUTION:"
    )

    print(
        original
        .value_counts(
            dropna=False
        )
        .sort_index()
    )

    # ------------------------------------------------------------
    # Check for null/non-numeric values
    # ------------------------------------------------------------

    null_count = int(
        original.isna().sum()
    )

    print()
    print(
        "NON-NUMERIC / NULL MONTHS:",
        null_count
    )

    if null_count > 0:

        raise ValueError(
            f"Month contains {null_count} "
            "non-numeric/null records. "
            "Automatic month conversion is unsafe."
        )

    return original


# ================================================================
# DETECT ZERO-BASED ENCODING
# ================================================================

def detect_zero_based_encoding(
    original_month
):

    print()
    print("=" * 70)
    print("MONTH ENCODING DETECTION")
    print("=" * 70)

    values = set(
        original_month
        .astype(int)
        .unique()
        .tolist()
    )

    print(
        "UNIQUE VALUES:",
        sorted(values)
    )

    # ------------------------------------------------------------
    # Exact expected zero-based range
    # ------------------------------------------------------------

    if not values.issubset(
        ZERO_BASED_MONTHS
    ):

        raise ValueError(
            "Month values are not contained "
            "within 0..11.\n"
            f"Found: {sorted(values)}\n"
            "Automatic conversion stopped."
        )

    if 0 not in values:

        raise ValueError(
            "Month=0 was not found. "
            "The deployment file does not match "
            "the expected zero-based encoding."
        )

    # ------------------------------------------------------------
    # Absence of 12 is important
    # ------------------------------------------------------------

    if 12 in values:

        raise ValueError(
            "Month=12 exists together with 0..11. "
            "Encoding is ambiguous. "
            "Automatic conversion stopped."
        )

    print()
    print(
        "DETECTED ENCODING:"
    )

    print(
        "ZERO-BASED MONTHS"
    )

    print(
        "0 = January"
    )

    print(
        "1 = February"
    )

    print(
        "2 = March"
    )

    print(
        "3 = April"
    )

    print(
        "4 = May"
    )

    print(
        "5 = June"
    )

    print(
        "6 = July"
    )

    print(
        "7 = August"
    )

    print(
        "8 = September"
    )

    print(
        "9 = October"
    )

    print(
        "10 = November"
    )

    print(
        "11 = December"
    )

    return True


# ================================================================
# APPLY MONTH CONVERSION
# ================================================================

def convert_months(
    df,
    original_month
):

    print()
    print("=" * 70)
    print("CONVERTING MONTHS")
    print("=" * 70)

    df = df.copy()

    # Preserve original for audit
    df["_original_month"] = (
        original_month.astype(int)
    )

    # Zero-based -> conventional month
    df["month"] = (
        df["_original_month"] + 1
    ).astype(int)

    print()
    print(
        "CONVERSION:"
    )

    print(
        "0 -> 1"
    )

    print(
        "1 -> 2"
    )

    print(
        "..."
    )

    print(
        "11 -> 12"
    )

    print()
    print(
        "CORRECTED MONTH DISTRIBUTION:"
    )

    print(
        df["month"]
        .value_counts()
        .sort_index()
    )

    return df


# ================================================================
# DERIVE SEASON
# ================================================================

def derive_season(df):

    print()
    print("=" * 70)
    print("DERIVING SEASON")
    print("=" * 70)

    df = df.copy()

    df["season"] = (
        df["month"]
        .apply(
            month_to_season
        )
    )

    print()
    print(
        "SEASON DISTRIBUTION:"
    )

    print(
        df["season"]
        .value_counts(
            dropna=False
        )
    )

    return df


# ================================================================
# BUILD AUDIT
# ================================================================

def build_audit(df):

    print()
    print("=" * 70)
    print("BUILDING REPAIR AUDIT")
    print("=" * 70)

    audit = (
        df[
            [
                "subdivision",
                "year",
                "_original_month",
                "month",
                "season"
            ]
        ]
        .copy()
    )

    audit.rename(
        columns={
            "_original_month":
                "original_month",

            "month":
                "corrected_month",
        },
        inplace=True
    )

    audit["conversion"] = (
        audit["original_month"]
        .astype(str)
        + " -> "
        + audit["corrected_month"]
        .astype(str)
    )

    audit["repair_method"] = (
        "ZERO_BASED_TO_CALENDAR"
    )

    audit["confidence"] = (
        "HIGH"
    )

    return audit


# ================================================================
# HARD VALIDATION
# ================================================================

def validate_repaired_data(
    df
):

    print()
    print("=" * 70)
    print("FINAL REPAIR VALIDATION")
    print("=" * 70)

    validation_results = []

    # ------------------------------------------------------------
    # Row count
    # ------------------------------------------------------------

    row_count_pass = (
        len(df) == EXPECTED_ROWS
    )

    validation_results.append(
        {
            "check":
                "ROW_COUNT",

            "status":
                "PASS"
                if row_count_pass
                else "FAIL",

            "value":
                len(df)
        }
    )

    print(
        "ROW COUNT:",
        "PASS"
        if row_count_pass
        else "FAIL",
        len(df)
    )

    if not row_count_pass:

        raise ValueError(
            "Row count changed during repair."
        )

    # ------------------------------------------------------------
    # Month range
    # ------------------------------------------------------------

    invalid_month_mask = (
        ~df["month"].between(
            1,
            12
        )
    )

    invalid_month_count = int(
        invalid_month_mask.sum()
    )

    month_pass = (
        invalid_month_count == 0
    )

    validation_results.append(
        {
            "check":
                "MONTH_RANGE",

            "status":
                "PASS"
                if month_pass
                else "FAIL",

            "value":
                invalid_month_count
        }
    )

    print(
        "INVALID MONTHS:",
        invalid_month_count
    )

    if not month_pass:

        raise ValueError(
            "Invalid calendar months remain."
        )

    # ------------------------------------------------------------
    # Unknown season
    # ------------------------------------------------------------

    unknown_season_mask = (
        df["season"]
        .eq("UNKNOWN")
    )

    unknown_season_count = int(
        unknown_season_mask.sum()
    )

    season_pass = (
        unknown_season_count == 0
    )

    validation_results.append(
        {
            "check":
                "UNKNOWN_SEASON",

            "status":
                "PASS"
                if season_pass
                else "FAIL",

            "value":
                unknown_season_count
        }
    )

    print(
        "UNKNOWN SEASONS:",
        unknown_season_count
    )

    if not season_pass:

        raise ValueError(
            "UNKNOWN seasons remain."
        )

    # ------------------------------------------------------------
    # Expected season domain
    # ------------------------------------------------------------

    actual_seasons = set(
        df["season"]
        .dropna()
        .unique()
        .tolist()
    )

    invalid_seasons = (
        actual_seasons
        -
        EXPECTED_SEASONS
    )

    season_domain_pass = (
        len(invalid_seasons) == 0
    )

    validation_results.append(
        {
            "check":
                "SEASON_DOMAIN",

            "status":
                "PASS"
                if season_domain_pass
                else "FAIL",

            "value":
                ",".join(
                    sorted(
                        actual_seasons
                    )
                )
        }
    )

    print(
        "SEASON DOMAIN:",
        "PASS"
        if season_domain_pass
        else "FAIL"
    )

    if not season_domain_pass:

        raise ValueError(
            f"Invalid seasons: {invalid_seasons}"
        )

    # ------------------------------------------------------------
    # Month -> season consistency
    # ------------------------------------------------------------

    expected_seasons = (
        df["month"]
        .apply(
            month_to_season
        )
    )

    inconsistent = (
        df["season"]
        !=
        expected_seasons
    )

    inconsistent_count = int(
        inconsistent.sum()
    )

    consistency_pass = (
        inconsistent_count == 0
    )

    validation_results.append(
        {
            "check":
                "MONTH_SEASON_CONSISTENCY",

            "status":
                "PASS"
                if consistency_pass
                else "FAIL",

            "value":
                inconsistent_count
        }
    )

    print(
        "MONTH/SEASON CONSISTENCY:",
        "PASS"
        if consistency_pass
        else "FAIL"
    )

    if not consistency_pass:

        raise ValueError(
            "Month/season consistency failed."
        )

    # ------------------------------------------------------------
    # Subdivision null check
    # ------------------------------------------------------------

    null_subdivision = int(
        df["subdivision"]
        .isna()
        .sum()
    )

    subdivision_pass = (
        null_subdivision == 0
    )

    validation_results.append(
        {
            "check":
                "SUBDIVISION_NULL",

            "status":
                "PASS"
                if subdivision_pass
                else "FAIL",

            "value":
                null_subdivision
        }
    )

    print(
        "NULL SUBDIVISIONS:",
        null_subdivision
    )

    if not subdivision_pass:

        raise ValueError(
            "Null subdivision values detected."
        )

    # ------------------------------------------------------------
    # Year null check
    # ------------------------------------------------------------

    null_year = int(
        df["year"]
        .isna()
        .sum()
    )

    year_pass = (
        null_year == 0
    )

    validation_results.append(
        {
            "check":
                "YEAR_NULL",

            "status":
                "PASS"
                if year_pass
                else "FAIL",

            "value":
                null_year
        }
    )

    print(
        "NULL YEARS:",
        null_year
    )

    if not year_pass:

        raise ValueError(
            "Null year values detected."
        )

    # ------------------------------------------------------------
    # Duplicate rows
    # ------------------------------------------------------------

    duplicate_count = int(
        df.duplicated(
            subset=[
                "subdivision",
                "year",
                "month"
            ]
        ).sum()
    )

    duplicate_pass = (
        duplicate_count == 0
    )

    validation_results.append(
        {
            "check":
                "SUBDIVISION_YEAR_MONTH_DUPLICATES",

            "status":
                "PASS"
                if duplicate_pass
                else "REVIEW",

            "value":
                duplicate_count
        }
    )

    print(
        "SUBDIVISION/YEAR/MONTH DUPLICATES:",
        duplicate_count
    )

    # ------------------------------------------------------------
    # Validation dataframe
    # ------------------------------------------------------------

    validation_df = pd.DataFrame(
        validation_results
    )

    return validation_df


# ================================================================
# SAVE AUDIT + OUTPUT
# ================================================================

def save_files(
    df,
    audit,
    validation_df
):

    print()
    print("=" * 70)
    print("SAVING OUTPUT FILES")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Remove internal audit column
    # ------------------------------------------------------------

    output_df = df.drop(
        columns=[
            "_original_month"
        ]
    )

    # ------------------------------------------------------------
    # Save repaired deployment
    # ------------------------------------------------------------

    output_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ------------------------------------------------------------
    # Save audit
    # ------------------------------------------------------------

    audit.to_csv(
        AUDIT_FILE,
        index=False
    )

    # ------------------------------------------------------------
    # Save validation
    # ------------------------------------------------------------

    validation_df.to_csv(
        VALIDATION_FILE,
        index=False
    )

    print()
    print(
        "REPAIRED DEPLOYMENT:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "REPAIR AUDIT:"
    )

    print(
        AUDIT_FILE
    )

    print()
    print(
        "VALIDATION REPORT:"
    )

    print(
        VALIDATION_FILE
    )


# ================================================================
# FINAL SUMMARY
# ================================================================

def print_final_summary(
    df
):

    print()
    print("=" * 70)
    print(
        "3.13.1 FINAL SUMMARY"
    )
    print("=" * 70)

    print()
    print(
        "ROWS:",
        len(df)
    )

    print()
    print(
        "MONTH DISTRIBUTION:"
    )

    print(
        df["month"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "SEASON DISTRIBUTION:"
    )

    print(
        df["season"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "MONTH MIN:",
        int(df["month"].min())
    )

    print(
        "MONTH MAX:",
        int(df["month"].max())
    )

    print()
    print(
        "UNKNOWN SEASONS:",
        int(
            df["season"]
            .eq("UNKNOWN")
            .sum()
        )
    )

    print()
    print(
        "INVALID MONTHS:",
        int(
            (~df["month"].between(1, 12))
            .sum()
        )
    )

    print()
    print("=" * 70)
    print(
        "STATUS: PASS"
    )
    print("=" * 70)


# ================================================================
# MAIN
# ================================================================

def main():

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    df = load_deployment()

    # ------------------------------------------------------------
    # Schema validation
    # ------------------------------------------------------------

    validate_input_schema(
        df
    )

    # ------------------------------------------------------------
    # Analyze original month
    # ------------------------------------------------------------

    original_month = (
        analyze_original_month(
            df
        )
    )

    # ------------------------------------------------------------
    # Detect zero-based encoding
    # ------------------------------------------------------------

    detect_zero_based_encoding(
        original_month
    )

    # ------------------------------------------------------------
    # Convert
    # ------------------------------------------------------------

    repaired = (
        convert_months(
            df,
            original_month
        )
    )

    # ------------------------------------------------------------
    # Derive season
    # ------------------------------------------------------------

    repaired = (
        derive_season(
            repaired
        )
    )

    # ------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------

    audit = (
        build_audit(
            repaired
        )
    )

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    validation_df = (
        validate_repaired_data(
            repaired
        )
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    save_files(
        repaired,
        audit,
        validation_df
    )

    # ------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------

    print_final_summary(
        repaired
    )

    print()
    print("=" * 70)
    print(
        "3.13.1 DEPLOYMENT MONTH / SEASON REPAIR COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Run the final project validation using:"
    )

    print(
        OUTPUT_FILE
    )


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()