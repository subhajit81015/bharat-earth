# Model Monitoring & Drift V4

## Monitoring Scope

- Reference period: 1901-2013
- Production period: 2016-2017
- Reference rows: 48513
- Production rows: 756
- Features monitored: 23

## PSI Thresholds

| PSI | Interpretation |
|---:|---|
| < 0.10 | Stable |
| 0.10 - 0.25 | Moderate |
| >= 0.25 | Significant |

## Drift Summary

- Stable features: 19
- Moderate drift features: 3
- Significant drift features: 1
- Maximum PSI: 12.702809535185633
- Maximum PSI feature: year

## Feature Drift

| Rank | Feature | Type | PSI | Status |
|---:|---|---|---:|---|
| 1 | year | numeric | 12.702810 | SIGNIFICANT |
| 2 | rainfall_prev_12m | numeric | 0.110893 | MODERATE |
| 3 | rainfall_12m | numeric | 0.104711 | MODERATE |
| 4 | rainfall_trend_3m | numeric | 0.100730 | MODERATE |
| 5 | month | categorical | 0.073973 | STABLE |
| 6 | rainfall_anomaly_pct | numeric | 0.066410 | STABLE |
| 7 | rainfall_lag_3m | numeric | 0.060371 | STABLE |
| 8 | rainfall_prev_3m | numeric | 0.058571 | STABLE |
| 9 | month_sin | numeric | 0.056135 | STABLE |
| 10 | season | categorical | 0.051476 | STABLE |
| 11 | rainfall_prev_6m | numeric | 0.047758 | STABLE |
| 12 | rainfall_zscore | numeric | 0.046648 | STABLE |
| 13 | rainfall_3m | numeric | 0.044953 | STABLE |
| 14 | rainfall_6m | numeric | 0.039623 | STABLE |
| 15 | month_cos | numeric | 0.038298 | STABLE |
| 16 | rainfall_lag_2m | numeric | 0.038269 | STABLE |
| 17 | rainfall_anomaly | numeric | 0.034987 | STABLE |
| 18 | rainfall_deficit_mm | numeric | 0.032820 | STABLE |
| 19 | rainfall_lag_1m | numeric | 0.022331 | STABLE |
| 20 | rainfall_mm | numeric | 0.018381 | STABLE |
| 21 | historical_monthly_mean | numeric | 0.006484 | STABLE |
| 22 | subdivision | categorical | 0.000831 | STABLE |
| 23 | rainfall_missing | numeric | 0.000000 | STABLE |

## Production Recommendation

Significant feature drift was detected. Production deployment should remain under review until the affected features are investigated.
