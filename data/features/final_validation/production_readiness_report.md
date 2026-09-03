# Bharat Earth - Production Readiness Report

**Validation Version:** 3.13

**Generated:** 2026-09-01T02:29:21.573881

**Deployment File:** `C:\Users\subha\Downloads\bharat-earth\data\features\deployment\final_risk_predictions_fixed.csv`

## Final Decision: `NO_GO`

One or more hard validation gates failed.

**Failures:** 1

**Review Items:** 7

## Hard Failures

- **REQUIRED FEATURE SCHEMA**:  ['rainfall_stress']

## Review Items

- **FEATURES INVALID MONTH ROWS**: 50133 
- **TARGET INVALID MONTH ROWS**: 50133 
- **TARGET-LIKE FEATURE NAMES**:  ['target_3m_stress']
- **FEATURES MISSING VALUE RATE**: 0.004169 rainfall_trend_3m
- **TARGET MISSING VALUE RATE**: 0.001257 rainfall_mm
- **MAXIMUM PSI**: 0.472711 
- **SIGNIFICANT DRIFT FEATURES**: threshold=0.25 

## Validation Summary

- ARTIFACTS / PASS: 10
- DATASET INTEGRITY / FAIL: 1
- DATASET INTEGRITY / PASS: 9
- DRIFT / PASS: 1
- DRIFT / REVIEW: 2
- KEY STRUCTURE / PASS: 7
- KEY STRUCTURE / REVIEW: 2
- LEAKAGE / PASS: 2
- LEAKAGE / REVIEW: 1
- MISSING VALUES / PASS: 1
- MISSING VALUES / REVIEW: 2
- MONTH / PASS: 4
- PERFORMANCE / PASS: 6
- POLICY / PASS: 3
- PROBABILITY / PASS: 5
- PRODUCTION SCHEMA / PASS: 1
- REPRODUCIBILITY / PASS: 6
- SEASON / PASS: 3
- TEMPORAL / PASS: 6
