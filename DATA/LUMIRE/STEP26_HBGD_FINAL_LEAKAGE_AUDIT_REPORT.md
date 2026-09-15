# Step 26 — Final HBGD Leakage Audit and Evidence Consolidation

## Purpose

Step 26 audits the provenance and structural integrity of the strict temporal Step 25 experiment and consolidates the major cross-cancer HBGD evidence generated in Steps 19–25.

No predictive model is retrained in this step.

## Audit summary

- PASS checks: 13
- FAIL checks: 0
- CHECK items: 1

## Dataset provenance

| Artifact | Role | Count | Patients |
|---|---|---:|---:|
| Source HBGD | Validated source growth intervals | 49 | 12 |
| LUMIERE trajectories | Target longitudinal tumor trajectories | 476 | 67 |
| Step 25 predictions | Strict temporal held-out predictions | 276 | 59 |
| Step 25 performance | Temporal model comparison | 4 | 59 |

## Leakage audit

| Check | Status | Evidence | Interpretation |
|---|---|---|---|
| Source HBGD schema | PASS | 17_hbgd_state_parameters.csv contains patient, interval time, diameter and growth-rate fields. | The validated source interval artifact is suitable as the provenance source for transferred HBGD. |
| Source validated interval count | PASS | 49 valid source intervals from 12 source patients. | The established source cohort contains 49 validated HBGD intervals from 12 patients. |
| Target trajectory schema | PASS | LUMIERE trajectory file contains patient, week, volume and equivalent diameter. | The target trajectory artifact supports temporal forecast auditing. |
| Step 25 prediction schema | PASS | All required temporal prediction fields are present. | Step 25 contains explicit prediction-origin, future target and model-error fields. |
| Future observation occurs after prediction origin | PASS | 276/276 forecasts have future_week > current_week and positive forecast horizon. | No Step 25 forecast reverses temporal order. |
| Future target separated from predictor fields | PASS | Future target column: actual_diameter_mm. Historical predictor fields are stored separately. | The prediction artifact explicitly distinguishes the future target from historical information. |
| Historical HBGD feature availability | PASS | history_n_timepoints, history_n_intervals, history_mean_rate, history_rate_std | HBGD history fields are explicitly represented as historical quantities. |
| Patient identifiers present for every forecast | PASS | 276 forecasts across 59 patients. | Each temporal forecast is attributable to a specific held-out patient. |
| Duplicate patient temporal forecast origins | PASS | 0 duplicate forecast rows detected. | A zero count indicates unique patient/time prediction origins in the final temporal prediction artifact. |
| Prediction error arithmetic integrity | PASS | Geometry and Geometry+HBGD absolute errors were independently reconstructed. | Stored prediction errors agree with actual minus predicted diameter calculations. |
| HBGD improvement arithmetic integrity | PASS | Improvement = Geometry absolute error minus Geometry+HBGD absolute error. | Positive values indicate that HBGD reduced prediction error. |
| Prediction numerical validity | PASS | 0 forecasts contain non-positive targets or non-finite HBGD predictions. | Valid forecasts require positive tumor diameter and finite model predictions. |
| Source-target patient identity separation | CHECK | 11 patient identifiers overlap between source and LUMIERE target cohorts. | Zero overlap supports cohort-level source/target separation. |
| Step 25 performance artifact consistency | PASS | Geometry MAE 7.0506 mm; Geometry+HBGD MAE 7.0272 mm; difference 0.0234 mm. | The stored performance table provides the expected comparison between geometry and transferred HBGD. |
| Source HBGD sample-size limitation | LIMITATION | 49 intervals from 12 source patients. | The source HBGD representation is based on a relatively small longitudinal cohort and therefore limits the strength of cross-cancer generalization claims. |
| LUMIERE week-to-day conversion | ASSUMPTION | Step 25 uses future_days derived from week differences using 7 days per week. | This temporal conversion should be verified against the official LUMIERE dataset documentation before making precise biological time-course claims. |

## Consolidated evidence matrix

| Step | Analysis | Key result | Status | Interpretation |
|---:|---|---|---|---|
| 19 | Representation-level cross-cancer transfer | Geometry + transferred HBGD improved MAE by 0.0887 mm (1.19%) | PRELIMINARY POSITIVE | Source-derived HBGD representation added small complementary information beyond geometry. |
| 20 | Statistical validation of Step 19 | 9/13 patients improved; p=0.097390; bootstrap CI crossed zero | NOT SIGNIFICANT | Directional benefit but insufficient evidence for statistical significance. |
| 22 | Repeated patient-level splits | 8/10 splits improved; mean +0.0592 mm | DIRECTIONALLY POSITIVE | Effect was present across a majority of repeated splits but was small and split-dependent. |
| 23 | Repeated-split statistical aggregation | Bootstrap CI [-0.1097, 0.2065] mm; sign-test p=0.109375 | NOT STATISTICALLY ESTABLISHED | Repeated-split evidence does not establish a non-zero effect. |
| 24 | Responder analysis | 61.54% of patient-seed evaluations favored HBGD; correlations weak | HYPOTHESIS-GENERATING | Benefit appears heterogeneous and no strong responder phenotype was established. |
| 25 | Strict temporal holdout | Geometry MAE 7.0506 mm; Geometry+HBGD MAE 7.0272 mm; +0.0234 mm (0.33%) | WEAK POSITIVE | A small pooled benefit persisted under strict temporal forecasting, but only 28/59 patients improved and median patient benefit was negative. |
| 26 | Final leakage audit | Audit of source provenance, temporal ordering, prediction arithmetic and patient identifiers | AUDIT | Establishes methodological traceability and documents remaining assumptions/limitations. |

## Overall scientific interpretation

The cross-cancer HBGD experiments provide preliminary evidence that a source-derived growth representation can contribute complementary information beyond target tumor geometry. The strongest representation-level experiment produced a 1.19% pooled MAE improvement, while repeated patient-level splits showed improvement in 8 of 10 splits. Under strict temporal holdout, the pooled improvement remained positive but decreased to 0.33%.

However, the available statistical analyses do not establish a non-zero effect. Step 23 produced a split-level bootstrap confidence interval crossing zero, and Step 25 showed that only 28 of 59 temporally held-out patients improved, with a negative median patient-level improvement.

Therefore, the appropriate conclusion is that cross-cancer HBGD transfer is **preliminary and not statistically established**. The findings support continued investigation rather than a claim of clinical validation or universal transferability.

## Major limitations

1. The source HBGD representation is based on only 49 validated intervals from 12 source patients.
2. The LUMIERE target cohort contains limited longitudinal observations per patient.
3. The observed predictive benefit is small.
4. Repeated-split results are heterogeneous.
5. Statistical confidence intervals include zero.
6. The week-to-day temporal conversion should be verified against the official dataset documentation.
7. Independent external validation is still required.

## Recommended final claim

A source-derived HBGD growth representation demonstrated preliminary cross-cancer transferability by providing a small complementary predictive contribution beyond target-domain tumor geometry. This directional effect persisted across repeated patient-level splits and under strict temporal forecasting, but remained heterogeneous and was not statistically established. Larger independent longitudinal cohorts are required to determine whether the observed signal represents a generalizable biological growth-dynamics representation.
