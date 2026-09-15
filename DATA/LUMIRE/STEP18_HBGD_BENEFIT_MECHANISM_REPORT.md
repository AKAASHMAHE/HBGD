# STEP 18 - HBGD BENEFIT MECHANISM ANALYSIS

## Objective

Determine whether HBGD-derived growth-rate information provides greater predictive benefit under particular longitudinal or tumor-growth conditions.

## Experimental Integrity

The analysis uses the exact Step-12 held-out predictions recovered in Step 16.

No model retraining was performed.

No new train/test split was created.

Phenotype and longitudinal characteristics were analyzed post-hoc and were not supplied as predictive inputs.

## Phenotype-Level Benefit

| Phenotype | N | Improved | Worsened | Mean Improvement (mm) | Median Improvement (mm) |
|---|---:|---:|---:|---:|---:|
| Growth | 11 | 6 | 5 | +0.1512 | +0.2934 |
| Stable | 9 | 7 | 2 | +0.1575 | +0.0669 |
| Shrinkage | 12 | 7 | 5 | +0.0475 | +0.0510 |

## Feature Associations

| Feature | N | Pearson r | Spearman r |
|---|---:|---:|---:|
| mean_interval_rate_per_day | 32 | +0.2385 | +0.1001 |
| mean_absolute_interval_rate_per_day | 32 | -0.3291 | -0.2716 |
| interval_rate_std_per_day | 23 | -0.1543 | -0.1937 |
| mean_interval_weeks | 32 | +0.3554 | +0.3396 |
| mean_diameter_change_mm | 32 | +0.0647 | +0.0685 |
| mean_absolute_diameter_change_mm | 32 | +0.0422 | -0.0535 |
| mean_percent_diameter_change | 32 | +0.1030 | +0.0696 |
| mean_current_diameter_mm | 32 | +0.0316 | -0.0238 |
| n_timepoints | 32 | +0.0187 | +0.0397 |
| history_weeks | 32 | +0.0733 | +0.0961 |
| patient_mean_rate_per_day | 32 | +0.2040 | +0.0708 |
| patient_rate_variability_per_day | 32 | -0.2924 | -0.3208 |

## Strongest Descriptive Associations

- **mean_interval_weeks**: Spearman r = +0.3396
- **patient_rate_variability_per_day**: Spearman r = -0.3208
- **mean_absolute_interval_rate_per_day**: Spearman r = -0.2716
- **interval_rate_std_per_day**: Spearman r = -0.1937
- **mean_interval_rate_per_day**: Spearman r = +0.1001

## History Analysis

| Group | N | Mean Improvement (mm) | Improved | Worsened |
|---|---:|---:|---:|---:|
| Shorter_History | 17 | +0.1289 | 12 | 5 |
| Longer_History | 15 | +0.0972 | 8 | 7 |

## Interpretation

Positive HBGD improvement means that adding HBGD-derived growth-rate information reduced prediction MAE relative to geometry alone.

Correlation results are exploratory and descriptive. They do not establish causality or statistical significance.

The test cohort contains only 13 patients, and some patients contribute to multiple phenotype groups. Therefore, feature associations should be considered hypothesis-generating.

## Conclusion

Step 18 identifies longitudinal and growth characteristics associated with the incremental predictive benefit of HBGD-derived growth-rate information, while preserving the original held-out predictions.
