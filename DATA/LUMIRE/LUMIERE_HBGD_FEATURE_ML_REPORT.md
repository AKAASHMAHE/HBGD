# LUMIERE HBGD FEATURE MACHINE-LEARNING ANALYSIS

## 1. Objective

This experiment tests whether HBGD-derived temporal features improve prediction of future tumor diameter when used as machine-learning features.

The experiment does not assume that the HBGD exponential equation itself is sufficient to forecast GBM progression.

## 2. Patient-Level Split

| Split | Patients |
|---|---:|
| Training | 47 |
| Validation | 7 |
| Test | 13 |

The test patients were not used for model training or feature/model selection.

## 3. Prediction Cases

| Split | Prediction cases |
|---|---:|
| Training | 208 |
| Validation | 59 |
| Test | 75 |

## 4. Models

### No-change

Future diameter is assumed to equal the current diameter.

### Geometry-only

Uses tumor size, recent changes, interval duration, and longitudinal history information without HBGD rate/state features.

### HBGD-only

Uses recent HBGD rate, multi-point HBGD rate, rate variability, state code, and overall HBGD rate.

### Geometry + HBGD

Combines tumor geometry/history with HBGD temporal features.

## 5. Test Results

| Model | MAE (mm) | RMSE (mm) | MAPE (%) |
|---|---:|---:|---:|
| No_change | 8.9621 | 13.3902 | 22.54 |
| Geometry_only | 8.0192 | 11.6371 | 18.96 |
| HBGD_only | 11.4525 | 15.4494 | 34.03 |
| Geometry_plus_HBGD | 8.1553 | 11.7812 | 19.35 |

## 6. Best ML Model

**Geometry_only** achieved the lowest test MAE among the machine-learning models: 8.0192 mm.

Improvement of the best ML model versus no-change: 10.52%.

## 7. HBGD Contribution

**Adding HBGD features did not improve test MAE over geometry/history-only features.** Therefore, this experiment does not establish an incremental predictive contribution from HBGD features.

## 8. Combined Model Feature Importance

| Feature | Importance |
|---|---:|
| current_volume_cm3 | 0.274647 |
| current_diameter_mm | 0.274102 |
| total_history_weeks | 0.094854 |
| multi_point_hbgd_rate_per_day | 0.074448 |
| overall_hbgd_rate_per_day | 0.037203 |
| recent_interval_weeks | 0.036489 |
| diameter_change_mm | 0.035383 |
| hbgd_rate_variability | 0.029201 |
| previous_volume_cm3 | 0.026857 |
| previous_diameter_mm | 0.026465 |
| recent_hbgd_rate_per_day | 0.023513 |
| volume_change_cm3 | 0.022189 |
| number_of_previous_observations | 0.019462 |
| volume_percent_change | 0.013579 |
| diameter_percent_change | 0.010358 |
| hbgd_state_code | 0.001248 |

## 9. Limitations

The target is future equivalent spherical tumor diameter derived from segmentation volume.

The current temporal representation assumes seven days per LUMIERE week identifier.

The dataset is limited to the validated LUMIERE longitudinal cohort and the test set contains only 13 patients.

Random Forest performance should therefore be interpreted as an experiment on this cohort rather than proof of general clinical performance.

## 10. Next Stage

If the combined HBGD model demonstrates improvement, the next experiment should test whether HBGD representations transfer from the existing source-cancer dataset to LUMIERE GBM.

If HBGD does not improve the combined model, the transfer experiment should be framed as a representation-transfer investigation rather than assuming that HBGD provides predictive benefit.