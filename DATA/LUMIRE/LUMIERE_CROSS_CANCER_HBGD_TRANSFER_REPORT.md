# LUMIERE CROSS-CANCER HBGD TRANSFER REPORT

## Objective

This experiment evaluates whether growth-related information derived from the existing source-cancer HBGD pipeline can be transferred to the LUMIERE glioblastoma cohort.

The source-cancer absolute growth rate was not directly imposed on LUMIERE. Instead, LUMIERE growth-rate features were standardized relative to the source-cancer HBGD growth-rate distribution.

## Patient Split

| Split | Patients | Cases |
|---|---:|---:|
| Training | 47 | 234 |
| Validation | 7 | 33 |
| Test | 13 | 75 |

## Source HBGD Profile

| Parameter | Value |
|---|---:|
| growth_rate_count | 25 |
| growth_rate_mean | -0.0008932260209198254 |
| growth_rate_median | -0.00028194102829629496 |
| growth_rate_std | 0.002733575106964497 |
| normalized_diameter_mean | 0.9330246064774796 |
| normalized_diameter_std | 0.1903148036481223 |
| normalized_volume_mean | 0.9358692493604348 |
| normalized_volume_std | 0.45376517776812675 |
| state_growth_count | 8 |
| state_interval_count | 49 |
| state_rate_mean | -0.0011366497615692016 |
| state_rate_std | 0.0031182667215001176 |
| state_shrinkage_count | 21 |
| state_stable_count | 20 |
| trajectory_error_mean | 0.7023544805099613 |
| volume_rate_count | 25 |
| volume_rate_mean | -0.0017584824155085518 |
| volume_rate_std | 0.007404673107302771 |

## Model Comparison

| Model | Validation MAE | Test MAE | Test RMSE |
|---|---:|---:|---:|
| Geometry_only | 5.8738 | 7.5529 | 11.2184 |
| Transferred_HBGD_only | 7.6869 | 10.1261 | 13.4215 |
| Geometry_plus_transferred_HBGD | 5.6962 | 7.7850 | 11.3331 |

## Interpretation

The geometry + transferred-HBGD model did not improve test MAE over the geometry-only model. Therefore, this experiment does not establish incremental predictive benefit from the transferred HBGD representation.

## Scientific Caution

This experiment is a preliminary cross-cancer transfer test. It does not establish clinical validity, biological equivalence of growth rates between cancer types, or generalization to unseen institutions.

The source-cancer HBGD distribution is used as a transfer reference rather than assuming that its absolute growth rates are directly applicable to GBM.

## Next Step

If transferred HBGD improves performance, the next stage should investigate a learned domain-adaptation or representation-transfer model and evaluate the effect across repeated patient-level splits.

If transferred HBGD does not improve performance, the result should be reported as evidence that direct source-cancer HBGD distribution transfer is insufficient, motivating a more explicit domain-adaptation strategy.