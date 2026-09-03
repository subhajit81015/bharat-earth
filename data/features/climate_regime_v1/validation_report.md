# Climate Regime Validation Report

Status: PASS

## Checks

- schema_validity: PASS - Required assignment columns are present.
- one_to_one_assignment: PASS - Each Earth State has one regime assignment.
- target_leakage: PASS - The prediction target is excluded from regime discovery and output.
- feature_provenance: PASS - Selected regime features are explicitly recorded.
- source_lineage: PASS - Every assignment retains source Earth State lineage.
- deterministic_rerun_equivalence: PASS - Deterministic rerun produces equivalent regime assignments.
- subdivision_consistency: PASS - Subdivision identity is preserved.
- valid_regime_ids: PASS - Regime IDs use the canonical format.
- no_empty_regime: PASS - Every discovered regime has assignments.
- missing_value_handling: PASS - Assignment identity and regime values are complete.
- temporal_ordering: PASS - Assignments are chronologically ordered within subdivisions.
- future_state_exclusion: PASS - Persistence uses only the current and previous ordered state.
- transition_validity: PASS - Transitions contain valid adjacent regime pairs.
- v4_integrity: PASS - V4 integrity is verified separately by git checks.
