# STEP 10 - HBGD STATISTICAL ALIGNMENT AND CROSS-CANCER TRANSFER

## 1. Objective

Step 10 evaluates whether the cross-cancer HBGD phenotype similarity observed in Step 9 is statistically robust and whether a source-cancer growth-pattern representation can provide useful transfer information for LUMIERE GBM.

## 2. Statistical Alignment

| Metric | Value |
|---|---:|
| Observed standardized phenotype distance | 2.659276 |
| Permutation p-value | 1.000000 |
| Bootstrap mean distance | 2.633921 |
| Bootstrap 95% CI lower | 1.891733 |
| Bootstrap 95% CI upper | 3.534498 |

A smaller phenotype distance indicates greater similarity between the source and target domains.

## 3. Feature-Level Comparison

| Feature | Source mean | GBM mean | Standardized difference |
|---|---:|---:|---:|
| growth_rate | -0.000893 | -0.003728 | -0.4653 |
| volume_growth_rate | -0.001758 | -0.011183 | -0.5205 |
| state_rate_mean | -0.000693 | -0.003728 | -0.5005 |
| state_rate_std | 0.001336 | 0.011663 | 1.1619 |
| growth_fraction | 0.200000 | 0.471868 | 0.9321 |
| stable_fraction | 0.420000 | 0.201403 | -0.6817 |
| shrinkage_fraction | 0.380000 | 0.326729 | -0.1559 |
| state_persistence | 0.400000 | 0.410026 | 0.0247 |

## 4. Transfer Experiment

| Model | Validation MAE | Test MAE | Test RMSE |
|---|---:|---:|---:|
| Geometry_only | 7.3863 | 7.9989 | 11.6383 |
| Transferred_HBGD_only | 11.6363 | 12.0527 | 15.6637 |
| Geometry_plus_transferred_HBGD | 7.2641 | 7.9970 | 11.5722 |

## 5. Transfer Interpretation

The geometry + transferred-HBGD model improved test MAE over geometry-only by 0.02%. This provides preliminary evidence that the transferred growth phenotype contains predictive information beyond the geometry features used in the baseline.

## 6. Statistical Caution

The source phenotype cohort is small. Therefore, alignment estimates and transfer performance should be considered preliminary.

The permutation analysis tests the observed feature-space distance under random reassignment of domain labels; it does not establish biological equivalence.

The bootstrap interval reflects resampling uncertainty within the available cohorts and should not be interpreted as a clinical confidence interval.

## 7. Temporal Assumption

LUMIERE week identifiers are converted using seven days per week. This assumption should be verified against the official dataset documentation before treating rates as clinically calibrated temporal measurements.

## 8. Scientific Conclusion

Step 10 separates phenotype alignment from predictive transfer. Similarity between source and GBM growth phenotypes is evaluated independently from whether transferred information improves future-diameter prediction.

A positive transfer result would support further development of a cancer-adaptive HBGD representation. A negative transfer result would indicate that the current source-to-GBM mapping is insufficient and that more explicit domain adaptation is required.