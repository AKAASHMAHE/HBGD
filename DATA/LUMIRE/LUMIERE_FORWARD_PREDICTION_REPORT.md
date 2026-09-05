# LUMIERE HBGD FORWARD PREDICTION

## 1. Objective

This experiment evaluates whether an individualized HBGD growth rate calculated from previous longitudinal tumor observations can predict the next observed tumor diameter.

The evaluation is performed using a patient-level train/test split to prevent observations from the same patient appearing in both sets.

## 2. Dataset

| Metric | Value |
|---|---:|
| Total patients | 67 |
| Training patients | 54 |
| Test patients | 13 |
| Test prediction cases | 75 |

## 3. Patient-Level Split

Random seed: 42

### Training patients

Patient-002, Patient-003, Patient-004, Patient-006, Patient-007, Patient-008, Patient-009, Patient-011, Patient-013, Patient-014, Patient-015, Patient-017, Patient-018, Patient-023, Patient-024, Patient-025, Patient-028, Patient-029, Patient-030, Patient-031, Patient-032, Patient-034, Patient-035, Patient-036, Patient-040, Patient-041, Patient-043, Patient-046, Patient-048, Patient-049, Patient-050, Patient-051, Patient-052, Patient-054, Patient-055, Patient-058, Patient-060, Patient-061, Patient-062, Patient-063, Patient-066, Patient-067, Patient-068, Patient-070, Patient-071, Patient-072, Patient-074, Patient-075, Patient-077, Patient-078, Patient-080, Patient-086, Patient-089, Patient-091

### Test patients

Patient-010, Patient-012, Patient-019, Patient-022, Patient-033, Patient-037, Patient-038, Patient-042, Patient-059, Patient-064, Patient-065, Patient-073, Patient-085

## 4. Global HBGD Rate

The global exponential rate is estimated only from training patients.

```text
r_global = sum[ln(D2/D1)] / sum[Δt]
```

Global rate: 0.00036611 per day
Training intervals used: 321

## 5. Forward Prediction Results

| Model | MAE (mm) | RMSE (mm) | Median AE (mm) | MAPE (%) |
|---|---:|---:|---:|---:|
| No-change | 8.9621 | 13.3902 | 4.6577 | 22.54 |
| Global exponential | 9.0534 | 13.2598 | 5.3329 | 22.95 |
| Individualized HBGD | 16.9799 | 28.8923 | 7.9192 | 39.20 |

## 6. HBGD Improvement

| Comparison | Improvement from HBGD |
|---|---:|
| vs No-change | -89.46% |
| vs Global exponential | -87.55% |

HBGD produced lower absolute error than no-change in 26 / 75 test cases (34.67%).

HBGD produced lower absolute error than global exponential prediction in 23 / 75 test cases (30.67%).

## 7. Interpretation

**Individualized HBGD did not outperform the no-change baseline on the test set.** The HBGD formulation therefore requires further analysis before claiming predictive superiority.

## 8. Important Limitation

The temporal conversion currently assumes one LUMIERE week identifier corresponds to seven days. This must be verified against official dataset documentation before interpreting the estimated rates as biologically calibrated growth rates.

The experiment is a one-step-ahead prediction evaluation. It does not yet establish long-horizon forecasting performance.

The individualized rate is estimated from the immediately preceding pair of observations. More robust versions can later use multiple historical observations, regression-based rate estimation, dynamic state information, or learned representations.

## 9. Next Stage

If within-LUMIERE prediction is successful, the next stage is cross-cancer transfer: learn growth-related representations from the existing HBGD source-cancer dataset and evaluate whether those representations improve prediction on LUMIERE GBM patients.

This transfer experiment must also use patient-level separation and must not use future LUMIERE measurements during model training.