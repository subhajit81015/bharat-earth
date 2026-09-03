# Bharat Earth V4 Production Readiness Report

## Final Decision

**GO_WITH_REVIEW**

- Total checks: 39
- Passes: 37
- Failures: 0
- Reviews: 2

## Validation Results

| section | check | status | details |
| --- | --- | --- | --- |
| DATASET | V4 DATASET | PASS | C:\\Users\\subha\\Downloads\\bharat-earth\\data\\features\\ml_dataset_v4.csv |
| DATASET | REQUIRED SCHEMA | PASS | All required columns present |
| DATASET | LEGACY LEAKAGE | PASS | No legacy leakage columns |
| DATASET | TARGET VALUES | PASS | rate=0.044701 |
| DATASET | EXACT DUPLICATES | PASS | 0 |
| DATASET | MONTH VALIDATION | PASS | invalid=0 |
| DATASET | SEASON VALIDATION | PASS | inconsistencies_repaired=4147 |
| DATASET | NUMERIC FEATURE TYPES | PASS | All numeric features valid |
| MODEL | MODEL FILE | PASS | C:\\Users\\subha\\Downloads\\bharat-earth\\data\\features\\model_v4\\xgboost_model.json |
| MODEL | MODEL SCHEMA | PASS | C:\\Users\\subha\\Downloads\\bharat-earth\\data\\features\\model_v4\\model_schema.json |
| MODEL | MODEL LOAD | PASS | features=23 |
| MODEL | FEATURE ORDER | PASS | 23 features match V4 schema |
| MODEL | SCHEMA JSON | PASS | Valid JSON |
| COMPATIBILITY | NULL VALUES | PASS | 0 |
| COMPATIBILITY | MODEL PREDICTION | PASS | sample=10 |
| CALIBRATION | CALIBRATED PREDICTIONS | PASS | C:\\Users\\subha\\Downloads\\bharat-earth\\data\\features\\calibration_v4\\calibrated_predictions.csv |
| CALIBRATION | CALIBRATION METRICS | PASS | C:\\Users\\subha\\Downloads\\bharat-earth\\data\\features\\calibration_v4\\calibration_metrics.csv |
| CALIBRATION | CALIBRATION SUMMARY | PASS | C:\\Users\\subha\\Downloads\\bharat-earth\\data\\features\\calibration_v4\\calibration_summary.json |
| CALIBRATION | OUTPUT SCHEMA | PASS | columns=8 |
| CALIBRATION | raw_probability RANGE | PASS | invalid=0 |
| CALIBRATION | isotonic_probability RANGE | PASS | invalid=0 |
| CALIBRATION | sigmoid_probability RANGE | PASS | invalid=0 |
| CALIBRATION | ACTUAL TARGET | PASS | rows=756 |
| POLICY | SELECTED POLICY | PASS | C:\\Users\\subha\\Downloads\\bharat-earth\\data\\features\\policy_v4\\selected_policy.csv |
| POLICY | POLICY METRICS | PASS | C:\\Users\\subha\\Downloads\\bharat-earth\\data\\features\\policy_v4\\policy_metrics.csv |
| POLICY | POLICY PREDICTIONS | PASS | C:\\Users\\subha\\Downloads\\bharat-earth\\data\\features\\policy_v4\\calibration_policy_predictions.csv |
| POLICY | THRESHOLD ANALYSIS | PASS | C:\\Users\\subha\\Downloads\\bharat-earth\\data\\features\\policy_v4\\threshold_analysis.csv |
| POLICY | SELECTED POLICY LOAD | PASS | rows=1 |
| POLICY | THRESHOLD RANGE | PASS | [0.09] |
| POLICY | POLICY PREDICTIONS | PASS | rows=756 |
| POLICY | POLICY METRICS | PASS | rows=1 |
| DRIFT | DRIFT FILE | PASS | C:\\Users\\subha\\Downloads\\bharat-earth\\data\\features\\monitoring_v4\\feature_drift.csv |
| DRIFT | TEMPORAL FEATURE SHIFT | REVIEW | Excluded from conventional PSI: year |
| DRIFT | PSI VALUES | PASS | monitored_features=22 |
| DRIFT | SIGNIFICANT DRIFT | PASS | max_psi=0.110893 |
| DRIFT | MODERATE DRIFT | REVIEW | {'count': 3, 'features': ['rainfall_prev_12m', 'rainfall_12m', 'rainfall_trend_3m']} |
| CROSS STEP | CALIBRATION OUTPUT | PASS | rows=756 |
| CROSS STEP | POLICY THRESHOLD | PASS | 0.09 |
| ARTIFACTS | REQUIRED ARTIFACTS | PASS | All required artifacts exist |

## Review Items

- **TEMPORAL FEATURE SHIFT**: Excluded from conventional PSI: year
- **MODERATE DRIFT**: {'count': 3, 'features': ['rainfall_prev_12m', 'rainfall_12m', 'rainfall_trend_3m']}

## Artifact Paths

- Dataset: `C:\Users\subha\Downloads\bharat-earth\data\features\ml_dataset_v4.csv`
- Model: `C:\Users\subha\Downloads\bharat-earth\data\features\model_v4\xgboost_model.json`
- Model schema: `C:\Users\subha\Downloads\bharat-earth\data\features\model_v4\model_schema.json`
- Calibration: `C:\Users\subha\Downloads\bharat-earth\data\features\calibration_v4\calibrated_predictions.csv`
- Policy: `C:\Users\subha\Downloads\bharat-earth\data\features\policy_v4\selected_policy.csv`
- Drift: `C:\Users\subha\Downloads\bharat-earth\data\features\monitoring_v4\feature_drift.csv`
