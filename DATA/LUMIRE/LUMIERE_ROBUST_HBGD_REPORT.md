# LUMIERE ROBUST HBGD PREDICTION

## 1. Objective

Step 4 showed that the one-interval individualized HBGD model did not outperform the no-change baseline. This experiment investigates whether the poor performance is caused by unstable single-interval rate estimation.

Multiple historical observations are therefore used to estimate a more robust exponential growth rate.

## 2. Dataset Split

| Metric | Value |
|---|---:|
| Total patients | 67 |
| Training patients | 54 |
| Test patients | 13 |
| Test prediction cases | 75 |

Patient-level splitting is maintained. No test-patient observations are used to estimate the global rate.

## 3. Models

### No-change

D_future = D_current

### Global exponential

Uses one population-level rate estimated only from training patients.

### One-interval HBGD

Uses the immediately preceding observation pair.

### Two-interval average HBGD

Averages the two most recent interval-specific growth rates.

### Multi-point HBGD

Fits ln(D) = a + r*t using all observations available before the target observation.

### Recent-window HBGD

Fits the exponential model using the most recent 4 historical observations.

## 4. Results

| Model | MAE (mm) | RMSE (mm) | Median AE (mm) | MAPE (%) | N |
|---|---:|---:|---:|---:|---:|
| No-change | 8.9621 | 13.3902 | 4.6577 | 22.54 | 75 |
| Global exponential | 9.0534 | 13.2598 | 5.3329 | 22.95 | 75 |
| One-interval HBGD | 16.9799 | 28.8923 | 7.9192 | 39.20 | 75 |
| Two-interval average HBGD | 13.5171 | 22.2479 | 5.9660 | 32.35 | 62 |
| Multi-point HBGD | 13.6811 | 20.6367 | 6.3202 | 32.95 | 75 |
| Recent-window HBGD | 13.6689 | 20.7365 | 5.6035 | 32.76 | 75 |

### Best MAE

**No-change** achieved the lowest MAE at 8.9621 mm.

One-interval HBGD vs no-change: -89.46% improvement.
Two-interval average HBGD vs no-change: -50.82% improvement.
Multi-point HBGD vs no-change: -52.65% improvement.
Recent-window HBGD vs no-change: -52.52% improvement.

## 5. Growth-State Analysis

| State | N | No-change MAE | One-interval MAE | Multi-point MAE | Recent-window MAE |
|---|---:|---:|---:|---:|---:|
| GROWTH | 33 | 7.2517 | 19.0219 | 13.3930 | 13.3766 |
| STABLE | 21 | 5.4272 | 5.7151 | 6.8031 | 6.9269 |
| SHRINKAGE | 21 | 15.1848 | 25.0358 | 21.0117 | 20.8702 |

## 6. Rate Stability Diagnosis

| Rate estimator | N | Mean r/day | Median r/day |
|---|---:|---:|---:|
| One-interval | 75 | -0.00338824 | 0.00014602 |
| Multi-point | 75 | -0.00305595 | 0.00018038 |
| Recent-window | 75 | -0.00288253 | 0.00045099 |

## 7. Interpretation

**The robust multi-timepoint HBGD estimators did not outperform the no-change baseline.** This would suggest that simply stabilizing the exponential growth-rate estimate is insufficient for reliable GBM forward prediction.

This experiment does not establish that HBGD is universally predictive or non-predictive. It evaluates the specific exponential diameter formulation on this LUMIERE test split.

## 8. Temporal Assumption

The current implementation converts differences in LUMIERE week identifiers to days using 7 days per week. This assumption should be verified against the official LUMIERE documentation.

## 9. Next Decision

If a robust HBGD estimator improves prediction, it can become the HBGD baseline for the cross-cancer transfer experiment.

If no HBGD estimator improves prediction, the project should not claim that HBGD alone predicts GBM progression. The subsequent transfer experiment should instead test whether learned growth representations or auxiliary HBGD features improve a supervised predictive model.