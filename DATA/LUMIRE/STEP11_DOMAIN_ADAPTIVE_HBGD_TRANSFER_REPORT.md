# STEP 11 - DOMAIN-ADAPTIVE HBGD TRANSFER

## Objective

Step 11 tests whether a growth-prediction model learned from the source cancer can be adapted to the LUMIERE glioblastoma domain without assuming identical growth-rate distributions.

## Patient Split

| Partition | Patients |
|---|---:|
| Train | 47 |
| Validation | 7 |
| Test | 13 |

The target test patients were not used to fit the GBM-specific domain adaptation model.

## Results

| Model | Validation MAE (mm) | Test MAE (mm) | Test RMSE (mm) |
|---|---:|---:|---:|
| Direct_HBGD | 15.7307 | 15.8693 | 27.0294 |
| Source_HBGD_Transfer | 8.9896 | 10.7697 | 15.9796 |
| GBM_Geometry_Only | 7.5282 | 7.9925 | 11.0089 |
| Domain_Adapted_HBGD | 7.3078 | 8.3593 | 11.8635 |
| Geometry_Plus_Adapted_HBGD | 8.6511 | 9.5397 | 12.8673 |

## Domain Adaptation

The adaptation model learns a GBM-specific residual correction to the source-cancer prediction using only the GBM training patients.

Mean validation correction: 3.0152 mm

Mean test correction: 2.1549 mm

## Comparison Against Geometry Baseline

Geometry-only test MAE: 7.9925 mm

Geometry + adapted HBGD test MAE: 9.5397 mm

Difference: -1.5472 mm

Relative change: -19.36%

The geometry + adapted HBGD model did not outperform the geometry-only baseline on the held-out test cohort. Therefore, this experiment does not establish predictive benefit from the current domain-adaptive HBGD representation.

## Scientific Interpretation

This experiment is intentionally more conservative than direct cross-cancer rate transfer. The source model provides a transferable growth prior, while the GBM training cohort learns a target-specific correction.

A successful result would support the hypothesis that source-cancer growth dynamics can be adapted to another cancer domain. A negative result would indicate that the current HBGD representation or adaptation strategy is insufficient.

## Limitations

The source cohort is substantially smaller than an ideal multi-cancer training cohort, and the LUMIERE cohort contains only 67 patients with longitudinal phenotype trajectories.

The current experiment uses equivalent tumor diameter derived from segmentation volume for LUMIERE rather than an identical measurement process across cancer datasets.

The seven-days-per-week conversion of LUMIERE week identifiers remains a dataset-temporal assumption and should be verified against the official documentation.

The held-out test cohort contains 13 patients, so performance estimates remain preliminary.

## Final Conclusion

Step 11 evaluates domain adaptation as a mechanism for cross-cancer HBGD transfer. The primary endpoint is held-out GBM prediction performance relative to a geometry-only baseline.