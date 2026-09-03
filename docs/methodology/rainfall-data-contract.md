# Rainfall Data Contract

## Purpose

Define the canonical schema for rainfall observations
used by BHARAT-EARTH.

## Canonical Fields

| Field | Type | Required |
|---|---|---|
| district | string | yes |
| state | string | yes |
| observation_date | date | yes |
| rainfall_mm | float | yes |
| source | string | yes |
| source_record_id | string | no |
| ingestion_timestamp | datetime | yes |

## Rules

- district must not be null
- state must not be null
- observation_date must be valid
- rainfall_mm must be >= 0
- source must not be null
- duplicate district/date records must be detected
- source provenance must be retained

## Data Layers

raw
→ processed
→ features

## Quality

A dataset failing mandatory validation must not
enter the feature engineering stage.