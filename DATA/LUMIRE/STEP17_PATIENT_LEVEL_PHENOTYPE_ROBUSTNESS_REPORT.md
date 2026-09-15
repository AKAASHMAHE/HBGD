# STEP 17 - PATIENT-LEVEL PHENOTYPE ROBUSTNESS

## Objective

Evaluate whether HBGD-derived growth-rate information provides consistent patient-level predictive benefit within Growth, Stable and Shrinkage phenotypes.

## Data Integrity

The analysis uses the exact Step-12 predictions recovered in Step 16.

No model was retrained and no patient split was reconstructed.

Phenotype was assigned from observed interval growth dynamics and was used only for post-hoc stratification.

## Results

| Phenotype | Patients | Improved | Worsened | Mean Improvement (mm) | Relative Improvement | Bootstrap 95% CI | Permutation p |
|---|---:|---:|---:|---:|---:|---|---:|
| Growth | 11 | 6 | 5 | 0.1512 | 1.33% | [-0.0791, 0.3934] | 0.264774 |
| Stable | 9 | 7 | 2 | 0.1575 | 2.87% | [0.0249, 0.2993] | 0.086191 |
| Shrinkage | 12 | 7 | 5 | 0.0475 | 0.56% | [-0.1886, 0.2798] | 0.702130 |

## Interpretation

Positive improvement values indicate lower MAE for Geometry + HBGD Rate compared with Geometry alone.

Negative values indicate that adding HBGD rate worsened prediction.

Bootstrap confidence intervals and paired permutation tests are patient-level analyses and should be interpreted cautiously because the test cohort contains only 13 patients.

The phenotype-specific analysis is therefore evidence of robustness or heterogeneity of the observed effect, rather than definitive statistical validation.

## Conclusion

Step 17 evaluates whether the incremental contribution of HBGD-derived growth-rate information is consistent across individual patients within each observed growth phenotype.
