# STEP 12 - HBGD INCREMENTAL INFORMATION AND ABLATION

## Objective

This experiment evaluates whether HBGD-derived growth dynamics provide predictive information beyond tumor geometry and temporal history in LUMIERE glioblastoma.

## Patient Split

| Partition | Patients |
|---|---:|
| Train | 47 |
| Validation | 7 |
| Test | 13 |

The test cohort was not used for model fitting or feature selection.

## Model Results

| Model | Validation MAE | Test MAE | Test RMSE | Features |
|---|---:|---:|---:|---:|
| M0_No_Change | 7.4820 | 9.0360 | 13.1131 | 0 |
| M1_Geometry | 7.5334 | 8.0170 | 11.2588 | 13 |
| M2_Geometry_Temporal | 7.6712 | 7.9180 | 11.1904 | 13 |
| M3_Geometry_HBGD_Rate | 7.7155 | 7.9392 | 11.1780 | 13 |
| M4_Geometry_HBGD_Variability | 7.7118 | 8.0126 | 11.2018 | 13 |
| M5_Geometry_All_HBGD | 7.6013 | 7.9664 | 11.1475 | 13 |

## Incremental HBGD Effect

Geometry-only test MAE: 8.0170 mm

Geometry + HBGD-rate test MAE: 7.9392 mm

Relative change from adding HBGD rate: 0.97%

Geometry + HBGD-variability test MAE: 8.0126 mm

Relative change from adding HBGD variability: 0.06%

Geometry + all HBGD test MAE: 7.9664 mm

Relative change from adding all HBGD features: 0.63%

## Interpretation

The full HBGD feature set improved held-out test MAE relative to geometry-only. This provides preliminary evidence that HBGD contains incremental predictive information beyond the geometry baseline.

The HBGD-rate ablation improved the geometry baseline, suggesting that the estimated growth rate may contain useful temporal information.

Temporal-history features improved the geometry baseline, indicating that part of the predictive signal may be attributable to longitudinal observation history rather than the HBGD formulation itself.

## Important Scientific Distinction

An improvement from HBGD features should not automatically be interpreted as proof that the exponential HBGD model is biologically correct. The ablation only evaluates whether HBGD-derived variables contain incremental predictive information.

Conversely, failure to improve prediction does not invalidate the mathematical HBGD formulation; it indicates that the current HBGD features may not provide sufficient additional predictive information for this dataset and task.

## Limitations

The experiment uses one target cancer cohort and a relatively small held-out patient cohort.

The HBGD features are derived from the available longitudinal segmentation measurements.

LUMIERE week identifiers are converted using seven days per week; this temporal assumption should be verified against the official dataset documentation.

## Conclusion

Step 12 isolates the incremental contribution of HBGD-derived features from general geometry and longitudinal-history information. The resulting ablation determines whether HBGD provides measurable additional predictive value for future tumor diameter.