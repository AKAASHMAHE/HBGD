# STEP 16 - EXACT STEP-12 TEST COHORT PHENOTYPE ANALYSIS

## Cohort Recovery

The exact Step-12 test cohort was recovered from the patient IDs contained in the Step-13 patient-level results.

Step 13 was generated using the established Step-12 patient split. Therefore, these IDs were used only to identify the same held-out cohort.

Test patients: 13

Prediction cases: 88

Prediction rows: 440

Phenotype matched rows: 440

Unmatched rows: 0

## Test Patients

- Patient-010
- Patient-012
- Patient-019
- Patient-022
- Patient-033
- Patient-037
- Patient-038
- Patient-042
- Patient-059
- Patient-064
- Patient-065
- Patient-073
- Patient-085

## Phenotype-Stratified Results

| Phenotype | N Cases | Model | MAE (mm) | RMSE (mm) |
|---|---:|---|---:|---:|
| Growth | 38 | M1_Geometry | 8.9442 | 13.7778 |
| Growth | 38 | M2_Geometry_Temporal | 8.7309 | 13.6339 |
| Growth | 38 | M3_Geometry_HBGD_Rate | 8.8829 | 13.7112 |
| Growth | 38 | M4_Geometry_HBGD_Variability | 9.0117 | 13.7285 |
| Growth | 38 | M5_Geometry_All_HBGD | 8.8785 | 13.6044 |
| Stable | 26 | M1_Geometry | 5.5331 | 6.1460 |
| Stable | 26 | M2_Geometry_Temporal | 5.5084 | 6.1174 |
| Stable | 26 | M3_Geometry_HBGD_Rate | 5.4270 | 6.0666 |
| Stable | 26 | M4_Geometry_HBGD_Variability | 5.5787 | 6.2038 |
| Stable | 26 | M5_Geometry_All_HBGD | 5.4821 | 6.0736 |
| Shrinkage | 24 | M1_Geometry | 9.2399 | 11.1043 |
| Shrinkage | 24 | M2_Geometry_Temporal | 9.2414 | 11.1490 |
| Shrinkage | 24 | M3_Geometry_HBGD_Rate | 9.1665 | 10.9822 |
| Shrinkage | 24 | M4_Geometry_HBGD_Variability | 9.0675 | 10.9536 |
| Shrinkage | 24 | M5_Geometry_All_HBGD | 9.2135 | 11.0742 |

## Interpretation

Phenotype was assigned post-hoc from the observed interval growth rate and was not provided as a predictive model input.

The analysis therefore evaluates whether the already-generated Step-12 models behave differently across observed growth phenotypes.

These results are descriptive and should not be interpreted as statistically significant without an appropriate patient-level statistical analysis.
