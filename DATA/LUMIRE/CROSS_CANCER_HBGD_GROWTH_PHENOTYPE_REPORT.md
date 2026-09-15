# CROSS-CANCER HBGD GROWTH PHENOTYPE ALIGNMENT

## 1. Objective

This experiment evaluates whether growth dynamics represented by HBGD show comparable structure between the existing source-cancer cohort and the LUMIERE glioblastoma cohort.

This step is a phenotype-alignment analysis rather than a predictive modeling experiment.

## 2. Cohorts

| Domain | Phenotype subjects |
|---|---:|
| Source cancer | 25 |
| LUMIERE GBM | 67 |

## 3. Comparable HBGD Features

| Feature | Source mean | GBM mean | Standardized difference |
|---|---:|---:|---:|
| growth_rate_per_day | -0.000893 | -0.003728 | -0.4653 |
| volume_growth_rate_per_day | -0.001758 | -0.011183 | -0.5205 |
| growth_fraction | 0.200000 | 0.471868 | 0.9321 |
| stable_fraction | 0.420000 | 0.201403 | -0.6817 |
| shrinkage_fraction | 0.380000 | 0.326729 | -0.1559 |
| state_persistence | 0.400000 | 0.410026 | 0.0247 |

## 4. Dynamic-State Comparison

| State | Source mean fraction | GBM mean fraction | Absolute difference |
|---|---:|---:|---:|
| growth | 0.2000 | 0.4719 | 0.2719 |
| stable | 0.4200 | 0.2014 | 0.2186 |
| shrinkage | 0.3800 | 0.3267 | 0.0533 |

## 5. Phenotype Distance

The mean standardized phenotype distance across comparable features was 0.4634.

For descriptive interpretation only, the observed distance corresponds to strong phenotype alignment.

These categories are heuristic and should not be interpreted as statistical significance thresholds.

## 6. Interpretation

The source and LUMIERE cohorts show relatively similar standardized HBGD growth-phenotype structure across the measured features. This supports proceeding to a formal representation-transfer experiment.

## 7. Important Scientific Limitation

Distributional similarity does not establish biological equivalence, predictive transferability, or clinical utility.

The source and target cohorts may differ in cancer biology, imaging protocol, segmentation characteristics, follow-up schedule, lesion definition, and cohort composition.

The analysis therefore provides evidence about representation structure rather than proof of cross-cancer biological transfer.

## 8. Next Step

If the growth phenotype demonstrates sufficient common structure, the next experiment should implement a learned cross-cancer representation-transfer or domain-adaptation model.

The transferred representation should then be evaluated on an untouched LUMIERE patient-level test set.

If substantial domain differences are observed, the next model should explicitly learn cancer-specific adaptation rather than directly transferring source-cancer growth parameters.