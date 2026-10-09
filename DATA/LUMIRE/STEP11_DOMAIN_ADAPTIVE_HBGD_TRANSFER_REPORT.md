# Step 11 — Leakage-Safe Domain-Adaptive HBGD Transfer

## Objective

Evaluate source-cancer next-timepoint diameter prediction in the LUMIERE GBM cohort, then learn a target-domain residual correction using target training patients only.

## Leakage safeguards

- Every prediction feature is constructed from observations strictly before the target timepoint.
- The target diameter is used only as the supervised label or evaluation outcome.
- Source training and target inference use the same ordered feature schema and day-based time units.
- Target patient IDs are split before evaluation; no patient contributes timepoints to multiple partitions.
- The combined model uses patient-grouped out-of-fold adaptation corrections for its training rows.
- Source generalization is estimated using a patient-grouped source holdout. The separate source training metrics are explicitly labelled as in-sample only.

## Patient split

| Partition | Patients | Prediction cases |
|---|---:|---:|
| Train | 47 | 255 |
| Validation | 7 | 66 |
| Test | 13 | 88 |

## Results

| Model | Validation MAE (mm) | Validation RMSE (mm) | Test MAE (mm) | Test RMSE (mm) |
|---|---:|---:|---:|---:|
| No_Change_Baseline | 7.4820 | 10.7220 | 9.0360 | 13.1131 |
| Direct_HBGD_Historical_Rate | 15.7307 | 49.6088 | 15.8693 | 27.0294 |
| Source_HBGD_Transfer | 8.1664 | 11.0336 | 9.0535 | 12.8447 |
| Target_Morphology_History_Baseline | 7.5587 | 10.2690 | 7.8918 | 11.0956 |
| Domain_Adapted_HBGD | 7.4392 | 9.7713 | 7.9403 | 11.2803 |
| Morphology_Plus_Adapted_HBGD | 7.3653 | 10.0880 | 7.9726 | 11.2015 |

## Key comparisons

- Source-transfer test MAE: 9.0535 mm.
- Domain-adapted HBGD test MAE: 7.9403 mm.
- Relative MAE reduction from source transfer to domain-adapted HBGD: 12.30%.
- Target morphology/history baseline test MAE: 7.8918 mm.
- Morphology + adapted HBGD test MAE: 7.9726 mm.
- Relative MAE reduction from target baseline to combined model: -1.02%. Positive means the combined model is better; negative means it is worse.

## Source cohort holdout

- Source cases: 49 across 12 source patients.
- Patient-grouped source holdout MAE: 7.8891 mm; RMSE: 10.1113 mm.
- In-sample source training MAE/RMSE are provided for debugging only and must not be interpreted as generalization performance.

## Interpretation

This run evaluates transfer of a source-trained prediction model plus a target-domain residual correction. It is not, by itself, proof that a biological growth parameter transfers unchanged between cancer types. Any claimed benefit must be judged against the held-out target baselines and patient-level variability.

## Limitations and checks

- The target cohort has a limited number of held-out patients; confidence intervals or repeated patient-level splits are recommended.
- Equivalent diameter and source observed diameter may be produced by different measurement pipelines.
- The conversion of week to seven days must be confirmed from the target dataset documentation.
- Results are next-observed-timepoint predictions; they do not establish long-horizon forecasting or clinical utility.
- Review step11_target_patient_performance.csv; pooled metrics can conceal patient-level failures.

## Final conclusion

Interpret the regenerated metrics only after confirming input schemas, time units, and the patient split. Claim cross-cancer predictive benefit only if the corrected held-out target results support it. Distinguish predictive-model adaptation from direct biological parameter transfer.
