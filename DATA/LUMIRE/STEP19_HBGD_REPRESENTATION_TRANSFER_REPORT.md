# Step 19 — Cross-Cancer Transfer of HBGD-Derived Growth Representations

## Objective

This experiment evaluates whether a compact growth-dynamics representation learned from the validated source HBGD cohort can transfer to LUMIERE glioblastoma progression prediction.

The experiment does not assume that the numerical growth rate from the source cancer can simply be copied to GBM. Instead, a source HBGD growth-rate mapping is transferred and then optionally calibrated using historical intervals from target training patients only.

## Source cohort

- Source artifact: `17_hbgd_state_parameters.csv`
- Validated source intervals: 49
- Source patients represented: 12
- Source growth-rate model: Random Forest
- Source training MAE: 0.00127081 mm-equivalent rate error
- Source training RMSE: 0.00186938

## Target cohort

- LUMIERE longitudinal trajectory rows: 476
- LUMIERE patients represented in prediction cases: 66
- Train patients: 46
- Validation patients: 7
- Exact held-out Step-13 test patients: 13

## Leakage control

Target prediction inputs use only measurements available at or before the prediction time. The future diameter is used only as the evaluation target.

The target-domain residual calibration uses historical intervals from training patients only. The 13-patient test cohort is not used for fitting or calibration.

## Models

- No-change
- Geometry
- Transferred HBGD
- Domain-adapted HBGD
- Geometry + Transferred HBGD

## Performance

| Split | Model | Cases | Patients | MAE (mm) | RMSE (mm) |
|---|---|---:|---:|---:|---:|
| train | No-change | 221 | 46 | 7.9114 | 11.8838 |
| train | Geometry | 221 | 46 | 5.7790 | 8.4120 |
| train | Transferred HBGD | 221 | 46 | 8.6117 | 12.4602 |
| train | Domain-adapted HBGD | 221 | 46 | 8.7847 | 12.7439 |
| train | Geometry + Transferred HBGD | 221 | 46 | 5.6237 | 8.1523 |
| validation | No-change | 46 | 7 | 8.1358 | 11.7956 |
| validation | Geometry | 46 | 7 | 8.3680 | 11.8468 |
| validation | Transferred HBGD | 46 | 7 | 8.8128 | 12.1235 |
| validation | Domain-adapted HBGD | 46 | 7 | 9.5377 | 13.2041 |
| validation | Geometry + Transferred HBGD | 46 | 7 | 8.3516 | 11.7305 |
| test | No-change | 75 | 13 | 8.9621 | 13.3902 |
| test | Geometry | 75 | 13 | 7.4565 | 11.3638 |
| test | Transferred HBGD | 75 | 13 | 9.1404 | 13.4270 |
| test | Domain-adapted HBGD | 75 | 13 | 9.4760 | 13.1519 |
| test | Geometry + Transferred HBGD | 75 | 13 | 7.3678 | 11.1678 |

## Held-out test interpretation

Geometry-only test MAE: 7.4565 mm

Direct transferred HBGD test MAE: 9.1404 mm (-22.58% vs geometry)

Domain-adapted HBGD test MAE: 9.4760 mm (-27.08% vs geometry)

Geometry + transferred HBGD test MAE: 7.3678 mm (1.19% vs geometry)

## Scientific interpretation rule

Cross-cancer HBGD transfer is considered preliminarily supported only if the transferred representation improves the untouched LUMIERE test cohort relative to the geometry-only baseline. A numerical improvement alone should not be described as statistically significant without patient-level statistical testing.

Because the source cohort contains only 12 patients and 49 validated intervals, results should be considered exploratory until independently replicated in larger longitudinal cohorts.
