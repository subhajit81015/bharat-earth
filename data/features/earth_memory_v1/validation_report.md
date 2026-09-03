# Earth Memory Validation Report

Status: PASS
Rows: 50133

## Checks

- required_columns: PASS - Required Earth Memory columns are present.
- memory_id: PASS - Memory IDs are deterministic.
- unique_memory_ids: PASS - Memory IDs are unique.
- one_to_one_state_memory: PASS - Memory records were generated for the Earth State input.
- target_leakage: PASS - Target leakage not present.
- subdivision_isolation: PASS - Subdivision values are populated.
- temporal_ordering: PASS - Temporal month values are valid.
- recurrence_values: PASS - Recurrence values are valid.
- similarity_values: PASS - Similarity values are within [0, 1].
- missing_history_handling: PASS - Missing-history values are valid.
- source_lineage: PASS - Source lineage metadata is present.
