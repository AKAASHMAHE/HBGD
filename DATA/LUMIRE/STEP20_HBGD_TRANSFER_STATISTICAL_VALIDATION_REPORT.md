# Step 20 — Patient-Level Statistical Validation of Cross-Cancer HBGD Transfer

## Objective

This analysis statistically validates the exact Step 19 held-out comparison between the LUMIERE geometry-only model and the geometry model augmented with the transferred HBGD representation.

No model was retrained during this analysis.

## Test cohort

- Expected test patients: 13
- Test patients analyzed: 13
- Prediction cases analyzed: 75

The analysis uses patient-level aggregation so that multiple prediction cases from the same patient do not artificially inflate the effective sample size.

## Pooled test performance

| Model | MAE (mm) | RMSE (mm) |
|---|---:|---:|
| Geometry | 7.4565 | 11.3638 |
| Geometry + Transferred HBGD | 7.3678 | 11.1678 |

Pooled MAE improvement: 0.0887 mm (1.19%).

## Patient-level results

- Mean improvement: 0.2062 mm
- Median improvement: 0.1613 mm
- Standard deviation: 0.4073 mm
- Patients improved: 9/13
- Patients worsened: 4/13
- Patients unchanged: 0/13

## Statistical validation

### Bootstrap

Mean patient-level improvement: 0.2065 mm

95% bootstrap CI: [-0.0083, 0.4210] mm

### Paired permutation test

Observed mean patient-level improvement: 0.2062 mm

Permutation p-value: 0.097390

### Wilcoxon signed-rank test

Wilcoxon statistic: 24.0000

Wilcoxon p-value: 0.146484

### Paired effect size

Effect size: 0.5062

## Interpretation

The transferred HBGD representation showed a directionally positive patient-level improvement over geometry alone, but the available statistical evidence does not establish significance.

A positive improvement means that the Geometry + Transferred HBGD model has lower patient-level MAE than Geometry alone.

Because only 13 held-out patients are available, statistical power is limited. Even a positive effect should therefore be described as preliminary unless replicated in larger independent cohorts.

## Important limitation

The statistical analysis validates the observed Step 19 performance difference but does not establish that the transferred representation is universally transferable across cancer types. Independent replication and larger longitudinal datasets remain necessary.
