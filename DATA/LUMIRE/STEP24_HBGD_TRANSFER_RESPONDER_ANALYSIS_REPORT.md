# Step 24 — HBGD Transfer Responder / Non-Responder Analysis

## Objective

Step 24 investigates whether the incremental benefit of transferred HBGD information is concentrated in specific longitudinal tumor-growth characteristics.

A patient-seed evaluation is considered a **Responder** when Geometry + Transferred HBGD has lower MAE than Geometry alone. Otherwise it is classified as a **Non-responder**.

## Leakage control

All longitudinal characteristics were calculated from observations available before the corresponding prediction origin. Future tumor diameter was not used to construct the historical HBGD features or phenotype labels.

## Response distribution

- Total patient-seed evaluations: 130
- Responders: 80
- Non-responders: 50
- Responder fraction: 61.54%

## Responder versus non-responder features

| Feature | Responder mean | Non-responder mean | Difference |
|---|---:|---:|---:|
| current_diameter_mm_mean | 36.896104 | 36.154501 | 0.741603 |
| current_volume_cm3_mean | 37.274839 | 36.706011 | 0.568827 |
| history_n_timepoints_mean | 3.993750 | 3.910000 | 0.083750 |
| history_n_intervals_mean | 2.993750 | 2.910000 | 0.083750 |
| history_weeks_mean | 38.478511 | 32.599909 | 5.878602 |
| mean_hbgd_rate_per_day_mean | -0.009179 | -0.010025 | 0.000847 |
| absolute_mean_hbgd_rate_per_day_mean | 0.011455 | 0.015869 | -0.004414 |
| hbgd_rate_std_per_day_mean | 0.006144 | 0.009510 | -0.003366 |
| positive_rate_fraction_mean | 0.469290 | 0.443413 | 0.025877 |
| negative_rate_fraction_mean | 0.530710 | 0.556587 | -0.025877 |
| last_interval_rate_per_day_mean | -0.006959 | -0.006400 | -0.000559 |
| last_interval_abs_rate_per_day_mean | 0.009794 | 0.012085 | -0.002291 |
| rate_range_per_day_mean | 0.012414 | 0.017533 | -0.005119 |
| history_diameter_change_mm_mean | -2.884737 | -2.276321 | -0.608416 |
| history_percent_change_mean | 0.198318 | 3.774316 | -3.575999 |

## Phenotype response

| Phenotype | N | Responders | Non-responders | Responder fraction | Mean improvement |
|---|---:|---:|---:|---:|---:|
| Growth | 45 | 27 | 18 | 60.00% | 0.191891 mm |
| Stable | 13 | 10 | 3 | 76.92% | -0.032292 mm |
| Shrinkage | 72 | 43 | 29 | 59.72% | 0.156343 mm |

## Descriptive benefit correlations

| Feature | N | Pearson r |
|---|---:|---:|
| current_diameter_mm_mean | 130 | 0.1827 |
| current_volume_cm3_mean | 130 | 0.1580 |
| hbgd_rate_std_per_day_mean | 130 | -0.1498 |
| rate_range_per_day_mean | 130 | -0.1436 |
| last_interval_rate_per_day_mean | 130 | -0.1277 |
| history_n_timepoints_mean | 130 | -0.1240 |
| history_n_intervals_mean | 130 | -0.1240 |
| history_percent_change_mean | 130 | -0.0662 |
| mean_hbgd_rate_per_day_mean | 130 | -0.0560 |
| last_interval_abs_rate_per_day_mean | 130 | 0.0520 |
| negative_rate_fraction_mean | 130 | -0.0371 |
| positive_rate_fraction_mean | 130 | 0.0371 |
| history_weeks_mean | 130 | -0.0287 |
| history_diameter_change_mm_mean | 130 | -0.0100 |
| absolute_mean_hbgd_rate_per_day_mean | 130 | -0.0033 |

## Interpretation

This responder analysis is exploratory and hypothesis-generating. Differences between responders and non-responders do not establish causality, statistical significance, or clinical utility.

The analysis is intended to identify longitudinal characteristics that may explain heterogeneity in the incremental contribution of HBGD information.

## Scientific conclusion

The cross-cancer HBGD transfer effect should not be interpreted as uniformly beneficial across all patients. Instead, Step 24 evaluates whether its usefulness may depend on the underlying longitudinal growth phenotype, growth-rate stability, temporal history, or tumor characteristics.

Any observed responder pattern should be considered hypothesis-generating until independently reproduced in a larger cohort.
