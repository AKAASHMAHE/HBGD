# STEP 13 - PATIENT-LEVEL HBGD ROBUSTNESS AND STATISTICAL TESTING

## Objective

Step 13 tests whether the small predictive improvement from adding HBGD rate features is consistent across held-out patients rather than being driven by a small number of prediction cases.

## Primary Comparison

Geometry-only versus Geometry + HBGD rate.

## Patient Split

| Partition | Patients |
|---|---:|
| Train | 47 |
| Validation | 7 |
| Test | 13 |

## Overall Test Performance

| Model | Test MAE |
|---|---:|
| Geometry-only | 7.4912 mm |
| Geometry + HBGD rate | 7.1862 mm |

## Patient-Level Results

| Metric | Value |
|---|---:|
| Patients improved | 8 |
| Patients worsened | 5 |
| Patients unchanged | 0 |
| Mean improvement | 0.7685 mm |
| Median improvement | 0.0797 mm |
| Improvement SD | 2.1665 mm |
| Paired effect size | 0.3547 |

## Bootstrap Analysis

The patient-level improvement was resampled 5000 times.

Bootstrap mean: 0.7500 mm

Bootstrap 95% CI: [-0.2496, 1.9629] mm

## Paired Permutation Test

Null hypothesis: the mean patient-level MAE improvement is zero.

Observed mean improvement: 0.7685 mm

Permutation p-value: 0.274745

Wilcoxon signed-rank p-value: 0.273438

## Interpretation

The mean patient-level improvement favors the geometry + HBGD-rate model.

More held-out patients improved than worsened when HBGD rate was added.

The bootstrap 95% confidence interval includes zero. Therefore, the small observed HBGD improvement cannot be considered statistically robust based on this bootstrap analysis.

The paired permutation test does not reject the zero-improvement null hypothesis at the 0.05 level.

## Scientific Interpretation

This analysis evaluates incremental predictive information, not biological validity of the exponential HBGD model.

Because only 13 test patients are available, statistical power is limited and the result should be considered preliminary even if a nominal statistical threshold is reached.

## Conclusion

Step 13 determines whether the small HBGD-related improvement observed in Step 12 is reproducible at the patient level. The result should be interpreted using the direction of the effect, the number of improved versus worsened patients, the bootstrap interval, and the paired statistical tests together.