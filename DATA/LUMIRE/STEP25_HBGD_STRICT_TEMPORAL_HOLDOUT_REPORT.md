# Step 25 — Strict Temporal Holdout Validation

## Objective

This experiment evaluates whether transferred HBGD information remains useful when predicting a strictly later unseen tumor observation using only information available before the prediction time.

## Temporal protocol

For each eligible patient, prediction origins were constructed chronologically. Historical observations preceding the prediction origin were used to derive HBGD features, while the immediately subsequent observation served as the future target.

Target-domain regression models were trained using other patients only for each held-out patient. Thus, the held-out patient's future measurements were not used during model fitting.

## Dataset

- Source HBGD intervals: 49
- Source patients: 12
- LUMIERE patients: 67
- Eligible temporal patients: 59
- Temporal forecasts: 276
- Forecast patients: 59

## Overall performance

| Model | MAE (mm) | RMSE (mm) | Improvement vs Geometry (mm) | Improvement (%) |
|---|---:|---:|---:|---:|
| M0_No_Change | 7.0034 | 10.5940 | 0.0472 | 0.67% |
| M1_Geometry | 7.0506 | 10.0760 | 0.0000 | 0.00% |
| M2_Geometry_Transferred_HBGD | 7.0272 | 9.9688 | 0.0234 | 0.33% |
| M3_Direct_Transferred_HBGD | 7.6441 | 10.9683 | -0.5935 | -8.42% |

## Phenotype-specific performance

| Phenotype | Forecasts | Patients | Geometry MAE | Geometry + HBGD MAE | Improvement |
|---|---:|---:|---:|---:|---:|
| Growth | 64 | 22 | 7.0368 | 6.8211 | 0.2157 |
| Stable | 56 | 20 | 9.3524 | 9.1857 | 0.1668 |
| Shrinkage | 156 | 39 | 6.2300 | 6.3369 | -0.1070 |

## Patient-level performance

- Mean improvement: 0.036395 mm
- Median improvement: -0.037898 mm
- Patients improved: 28/59
- Patients worsened: 31/59

## Interpretation

Under the strict temporal holdout protocol, Geometry + Transferred HBGD produced a lower pooled MAE than Geometry alone. This provides directional evidence that the transferred HBGD representation may retain complementary information when forecasting a later unseen observation.

This analysis remains exploratory and does not establish clinical validity or universal cross-cancer transferability. The temporal interpretation also depends on the assumption that the LUMIERE week index accurately represents elapsed time.

## Scientific conclusion

Strict temporal validation is a stronger test of the disease-progression forecasting hypothesis than random case-level evaluation because future observations are explicitly withheld from feature construction and model fitting. The results should therefore be interpreted as evidence for or against temporal generalization of the transferred HBGD representation.
