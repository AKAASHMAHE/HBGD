# Step 22 — Cross-Cancer HBGD Transfer Split Robustness

## Objective

This experiment evaluates whether the incremental predictive contribution of the transferred HBGD representation remains stable across multiple patient-level target-domain train/validation/test splits.

The source HBGD representation was learned once from 49 validated source intervals representing 12 source patients. It was then transferred unchanged across all target-domain splits.

## Leakage control

- Splitting was performed at the patient level.
- Test patients were excluded from model fitting.
- Future target diameter was not used as an input.
- The source HBGD model was fixed across target splits.
- Geometry and fusion models were independently fitted within each target training split.

## Split-level results

| Seed | Train patients | Validation patients | Test patients | Geometry MAE | Geometry + HBGD MAE | Improvement | Improvement % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 45 | 8 | 13 | 9.9698 | 9.9431 | 0.0267 | 0.27% |
| 7 | 45 | 8 | 13 | 9.6609 | 9.3917 | 0.2693 | 2.79% |
| 21 | 45 | 8 | 13 | 8.6975 | 8.2145 | 0.4830 | 5.55% |
| 100 | 45 | 8 | 13 | 8.1968 | 8.7804 | -0.5835 | -7.12% |
| 123 | 45 | 8 | 13 | 9.9111 | 9.7710 | 0.1401 | 1.41% |
| 202 | 45 | 8 | 13 | 9.4247 | 9.3848 | 0.0399 | 0.42% |
| 314 | 45 | 8 | 13 | 9.5480 | 9.4010 | 0.1470 | 1.54% |
| 512 | 45 | 8 | 13 | 9.2235 | 9.2415 | -0.0180 | -0.20% |
| 777 | 45 | 8 | 13 | 7.7541 | 7.7374 | 0.0167 | 0.22% |
| 1001 | 45 | 8 | 13 | 8.1638 | 8.0933 | 0.0706 | 0.86% |

## Robustness summary

- Valid splits: 10
- Splits improved by HBGD: 8/10
- Splits worsened by HBGD: 2/10
- Fraction of splits improved: 80.00%
- Mean split improvement: 0.0592 mm
- Median split improvement: 0.0552 mm
- Mean percentage improvement: 0.57%
- Median percentage improvement: 0.64%

## Patient-level robustness

- Patient evaluations across splits: 130
- Improved evaluations: 80
- Worsened evaluations: 50
- Mean patient improvement: 0.1498 mm
- Median patient improvement: 0.1732 mm

## Interpretation

The transferred HBGD representation showed a directionally favorable effect in at least 70% of valid patient-level splits.

This analysis evaluates stability of the observed directional effect across alternative patient cohorts. It does not constitute a formal hypothesis test and should not be interpreted as clinical validation.

## Scientific conclusion

Across repeated patient-level splits, the Geometry + Transferred HBGD model showed a generally favorable directional pattern relative to Geometry alone. This strengthens the hypothesis that source-derived HBGD information may provide complementary cross-cancer predictive signal. However, the magnitude and consistency of the effect must still be evaluated with larger independent cohorts and formal statistical analysis.
