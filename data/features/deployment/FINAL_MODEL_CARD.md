# Bharat Earth
## Final Model Card

### Model

- Model: XGBoost
- Probability calibration: Sigmoid
- Policy: Global probability threshold
- Operational threshold: 0.09

### Test Performance

- Observations: 7668
- Events: 424
- Event rate: 0.055295
- Alerts: 1349
- Alert rate: 0.175926
- Precision: 0.160860
- Recall: 0.511792
- F1: 0.244783
- PR-AUC: 0.153547
- ROC-AUC: 0.804733

### Confusion Matrix

- True Positive: 217
- False Positive: 1132
- False Negative: 207
- True Negative: 6112

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