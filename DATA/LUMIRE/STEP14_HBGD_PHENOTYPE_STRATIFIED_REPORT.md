# STEP 14 - HBGD PHENOTYPE-STRATIFIED PREDICTIVE ANALYSIS

## Objective

This analysis evaluates whether HBGD-derived growth information has different predictive value across tumor dynamic phenotypes.

The phenotypes are Growth, Stable, and Shrinkage.

## Phenotype Definitions

| Phenotype | HBGD rate criterion |
|---|---|
| Growth | r > 0.0005/day |
| Stable | -0.0010 <= r <= 0.0005/day |
| Shrinkage | r < -0.0010/day |

## Test Performance by Phenotype

| Phenotype | N | Geometry MAE | Geometry + HBGD MAE | Improvement | Improvement % |
|---|---:|---:|---:|---:|---:|
| Growth | 38 | 0.4608 | 0.4604 | 0.0004 | 0.08% |
| Stable | 26 | 0.1122 | 0.1046 | 0.0076 | 6.75% |
| Shrinkage | 24 | 0.3043 | 0.3825 | -0.0782 | -25.68% |

## Overall Test Performance

Geometry-only MAE: 0.3151 mm

Geometry + HBGD MAE: 0.3340 mm

Geometry-only RMSE: 1.1004 mm

Geometry + HBGD RMSE: 1.1793 mm

## Interpretation

The phenotype with the largest positive HBGD improvement was Stable (0.0076 mm, 6.75%).

The phenotype with the weakest relative result was Shrinkage (-0.0782 mm, -25.68%).

## Scientific Caution

Phenotype-stratified results are descriptive because the held-out test cohort contains only a limited number of cases in each phenotype.

The current analysis does not establish statistical significance of phenotype-specific HBGD benefit.

The observed phenotype should therefore be interpreted as an analysis of where HBGD-derived temporal information may be most useful, rather than as proof of phenotype-specific clinical efficacy.

## Conclusion

Step 14 determines whether the incremental predictive contribution of HBGD is heterogeneous across tumor dynamic phenotypes. A stronger benefit in the Growth phenotype would support the hypothesis that HBGD-derived temporal information is particularly useful for modeling actively progressing tumors, while a weak or negative result would indicate that current HBGD features do not provide phenotype-specific predictive advantage.