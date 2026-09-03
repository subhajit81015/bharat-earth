# Earth State Validation Report

Status: PASS
Rows: 50133

## Checks

- schema: PASS - Schema matches the required Earth State layout.
- target_leakage: PASS - Target leakage check passed.
- required_fields: PASS - Required fields are populated.
- month: PASS - Month values are within the valid 1-12 range.
- season: PASS - Season values are valid.
- subdivision: PASS - Subdivision values are populated.
- duplicates: PASS - State IDs are unique.
- chronology: PASS - Chronology is consistent within each subdivision.
- numeric_values: PASS - Numeric fields are valid.
- impossible_values: PASS - Observed values are within plausible ranges.
