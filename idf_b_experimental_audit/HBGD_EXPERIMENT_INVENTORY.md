# Comprehensive HBGD Experiment Inventory and File Registry

**Project:** Hierarchical Biological Growth Dynamics (HBGD) Cross-Cancer Transfer  
**Root Directory:** `C:\HBGD`  
**Audit Location:** `C:\HBGD\idf_b_experimental_audit\`  
**Date of Audit:** October 11, 2026  
**Auditor:** Machine Learning Research Auditor & Patent Documentation Assistant  

---

## 1. Executive Summary & Inventory Scope

This document provides a complete, forensic inventory of all software scripts, datasets, experiment logs, intermediate calculations, model outputs, and reports in the HBGD project repository at `C:\HBGD`. 

The repository encompasses two principal development phases:
1. **Original HBGD Primary CT Pipeline (Steps 1–35)**: Evaluated primarily on thoracic and abdominal Computed Tomography (CT) scans with 1D RECIST linear diameter measurements from DeepLesion solid-tumor patients, including an exploratory 3D Convolutional Neural Network (CNN) imaging branch.
2. **LUMIERE Glioblastoma Cross-Cancer Transfer Pipeline (Steps 0–27)**: Evaluated on longitudinal 3D Magnetic Resonance Imaging (MRI) volumetric segmentations of Glioblastoma multiforme (GBM) patients from the LUMIERE cohort, testing the transferability of source-derived growth parameters, domain-adaptive residual correction, and growth-state representations.

Every experiment has been cataloged with its executable script, input data, cohort counts, prediction targets, evaluation protocol, authoritative output artifacts, and forensic verification status.

---

## 2. Directory Structure and Component Classification

```
C:\HBGD\
├── CODES\                               # 35 Python scripts (Original HBGD Pipeline)
│   └── LUMIRE\                          # 30 Python scripts (Cross-Cancer Transfer & Audits)
│       └── revised_outputs\             # (Target directory for Step 27 outputs)
├── DATA\                                # 85 Files (Original CT datasets, CSVs, reports)
│   ├── 25_lesion_ct_crops\              # Physical 3D CT volume crops (44 crops)
│   ├── 33_ct_rate_cnn_results\          # Step 33 3D CNN experiment logs & predictions
│   └── LUMIRE\                          # 122 Files (LUMIERE trajectories, Step 0–26 outputs)
│       └── revised_outputs\             # 7 Preserved authoritative CSV/JSON files for Step 27
└── idf_b_experimental_audit\            # Formal Audit & IDF-B Deliverables Directory
    ├── HBGD_EXPERIMENT_INVENTORY.md     (Deliverable 1 - This File)
    ├── HBGD_EXPERIMENTAL_AUDIT.md       (Deliverable 2 - Full Forensic Audit)
    ├── HBGD_METRIC_RECONCILIATION.csv   (Deliverable 3 - Traceable Metric Database)
    ├── IDF_B_SECTION_8_FINAL_DRAFT.md   (Deliverable 4 - Copy-Ready Disclosure Draft)
    ├── IDF_B_SECTION_8_CONCISE.md       (Deliverable 7 - 1-Page Summary with Code)
    ├── IDF_B_SECTION_8_FIGURE_PLAN.md   (Deliverable 5 - Data-Grounded Figure Plan)
    └── IDF_B_SUBMISSION_CHECKLIST.md    (Deliverable 6 - Institutional Quality Checklist)
```

---

## 3. Master Experiment Registry

The table below catalogs every distinct evaluation protocol identified across the project. In accordance with audit standards, separate experiment IDs are assigned to distinct evaluation protocols, patient splits, and target definitions.

| Experiment ID | Experiment Name | Protocol / Task | Scripts Involved | Input Data Artifacts | Cohort Definition | Output Files | Evidence Status |
|---|---|---|---|---|---|---|---|
| **EXP-CT-01** | Mathematical Growth Model Fitting | Descriptive fit of exponential, linear, and Gompertz ODEs on longitudinal RECIST | `10_fit_growth_models.py`, `17_build_hbgd_mathematical_model.py` | `patients.csv`, `recist_measurements.csv` | 12 patients, 25 trajectories, 74 points | `10_patient_growth_rates.csv`, `17_hbgd_diameter_trajectories.csv`, `17_hbgd_growth_parameters.csv` | **VERIFIED** |
| **EXP-CT-02** | CT Forward Trajectory Prediction | Forward ODE forecasting ($D_0 \to D_{\text{future}}$) stratified by dynamic state | `18_test_hbgd_forward_prediction.py` | `17_hbgd_diameter_trajectories.csv`, `17_hbgd_state_parameters.csv` | 12 patients, 45 forward intervals | `18_hbgd_forward_predictions.csv`, `18_hbgd_forward_state_performance.csv` | **VERIFIED** |
| **EXP-CT-03** | Patient-Independent ODE Validation | Leave-One-Patient-Out (LOPO) generalization of state-aware growth rate | `19_patient_independent_hbgd_validation.py` | `17_hbgd_diameter_trajectories.csv` | 12 patients, 24 test observations | `19_patient_independent_predictions.csv`, `19_patient_independent_performance.csv` | **VERIFIED** |
| **EXP-CT-04** | Individualized Growth Parameter Test | Two-interval individualized calibration ($r_{\text{indiv}}$) vs population rate | `21_test_individualized_hbgd.py` | `17_hbgd_diameter_trajectories.csv` | 5 evaluation points across held-out trajectories | `21_individualized_hbgd_predictions.csv`, `21_individualized_hbgd_performance.csv` | **VERIFIED** |
| **EXP-CT-05** | 3D CNN CT-to-Rate Prototype | 3D ResNet regressor on volumetric CT crops predicting $r \to D_{\text{future}}$ | `27_train_ct_to_hbgd_rate.py`, `33_train_ct_to_hbgd_rate.py` | `25_lesion_ct_crops/`, `9_hbgd_features.csv` | 44 crops total; Test: **1 patient, 1 trajectory, 1 observation** | `33_ct_rate_cnn_results/33_cnn_hbgd_performance.csv`, `33_cnn_hbgd_forward_predictions.csv` | **VERIFIED (Exploratory Prototype)** |
| **EXP-LUM-01** | LUMIERE Trajectory Extraction | Voxel segmentation to equivalent diameter ($D_{\text{eq}} = (6V/\pi)^{1/3}$) | `2_lumiere_nifti_mask_validation.py`, `3_lumiere_tumor_geometry_hbgd.py` | `lumiere_nifti_mask_validation.csv` | 67 patients, 476 longitudinal scans | `lumiere_longitudinal_trajectories.csv`, `lumiere_tumor_geometry.csv` | **VERIFIED** |
| **EXP-LUM-02** | Direct Cross-Cancer Distribution Transfer (Step 8) | Standardizing target growth features by source-cancer mean and variance | `8_cross_cancer_hbgd_transfer.py` | `17_hbgd_state_parameters.csv`, `lumiere_longitudinal_trajectories.csv` | Train: 47 pts (234 cases); Val: 7 pts (33 cases); Test: 13 pts (75 cases) | `lumiere_cross_cancer_transfer_results.csv`, `lumiere_cross_cancer_transfer_summary.csv` | **VERIFIED (Negative Transfer)** |
| **EXP-LUM-03** | Domain-Adaptive Residual Transfer (Step 11 Held-Out) | Source RF + Target-domain residual correction on fixed held-out split | `11_domain_adaptive_hbgd_transfer.py` | `17_hbgd_state_parameters.csv`, `lumiere_longitudinal_trajectories.csv` | Train: 46 pts (221 cases); Val: 7 pts (46 cases); Test: 13 pts (88 cases) | `step11_domain_adaptive_transfer_predictions.csv`, `step11_domain_adaptive_transfer_summary.csv` | **VERIFIED (No Improvement vs Target Baseline)** |
| **EXP-LUM-04** | Incremental Feature Ablation (Step 12) | Evaluating incremental contribution of Geometry, History, Rate, Variability | `12_hbgd_incremental_information_ablation.py` | `lumiere_longitudinal_trajectories.csv`, `lumiere_hbgd_growth_features.csv` | Train: 47 pts; Val: 7 pts; Test: 13 pts (75 cases) | `step12_hbgd_ablation_predictions.csv`, `step12_hbgd_ablation_summary.csv` | **VERIFIED** |
| **EXP-LUM-05** | Representation-Level Transfer (Step 19 Held-Out) | Concatenating source-transferred rate with target geometry features | `19_hbgd_representation_transfer.py` | `17_hbgd_state_parameters.csv`, `lumiere_longitudinal_trajectories.csv` | Train: 46 pts (221 cases); Val: 7 pts (46 cases); Test: 13 pts (75 cases) | `step19_hbgd_representation_transfer_predictions.csv`, `step19_hbgd_representation_transfer_performance.csv` | **VERIFIED (Directional Improvement)** |
| **EXP-LUM-06** | Patient-Level Statistical Validation of Step 19 (Step 20) | Paired bootstrap, permutation, and Wilcoxon testing of Step 19 predictions | `20_hbgd_transfer_statistical_validation.py` | `step19_hbgd_representation_transfer_predictions.csv` | Held-out: 13 patients, 75 cases | `step20_hbgd_transfer_statistical_summary.csv`, `step20_hbgd_transfer_patient_results.csv` | **VERIFIED (p > 0.05, Not Significant)** |
| **EXP-LUM-07** | Repeated-Split Robustness (Step 22 & 23) | 10 repeated random patient splits (Train 46 / Val 7 / Test 13) | `22_hbgd_transfer_split_robustness.py`, `23_hbgd_repeated_split_statistics.py` | `17_hbgd_state_parameters.csv`, `lumiere_longitudinal_trajectories.csv` | 10 splits x 13 patients = 130 patient evaluations | `step22_hbgd_transfer_split_results.csv`, `step23_hbgd_repeated_split_statistical_summary.csv` | **VERIFIED (8/10 Splits Improved; Bootstrap CI crosses 0)** |
| **EXP-LUM-08** | Strict Chronological Temporal Holdout (Step 25) | Training on historical visits ($t \le t_{\text{origin}}$), forecasting final visit | `25_hbgd_temporal_holdout_validation.py` | `17_hbgd_state_parameters.csv`, `lumiere_longitudinal_trajectories.csv` | 59 eligible patients, 276 forecast cases | `step25_hbgd_temporal_predictions.csv`, `step25_hbgd_temporal_performance.csv`, `step25_hbgd_temporal_summary.csv` | **VERIFIED (Weak Directional; Median Patient Benefit Negative)** |
| **EXP-LUM-09** | Repeated Patient-Grouped Cross-Validation (Step 27) | 5 repeats x 5 folds (25 folds) GroupKFold on 54 development patients | `27_hbgd_patient_grouped_cv_experiment.py` | `17_hbgd_state_parameters.csv`, `lumiere_longitudinal_trajectories.csv` | 54 development patients, 25 folds, 1605 prediction rows | `revised_outputs/step27_fold_level_metrics.csv`, `revised_outputs/step27_aggregated_model_comparison.csv`, `revised_outputs/step27_paired_differences.csv` | **VERIFIED (Primary Benchmark; p > 0.05 Not Significant)** |
| **EXP-LUM-10** | Methodological Leakage & Provenance Forensic Audit | Independent audit of patient disjointness, temporal ordering, and arithmetic | `26_hbgd_final_leakage_audit.py`, `test_hbgd_transfer_audit.py` | Entire project artifact suite | 14 audit checkpoints, 7 unit tests | `step26_hbgd_leakage_audit.csv`, `STEP26_HBGD_FINAL_LEAKAGE_AUDIT_REPORT.md` | **VERIFIED (7/7 Unit Tests Pass with relative paths)** |

---

## 4. Primary Data Artifact Inventory

### A. Source Cancer (DeepLesion RECIST Cohort)
- `DATA/patients.csv`: Clinical metadata for 22 patients across 9 solid tumor diagnoses (lung, colon, gastric, rectal, breast, ovarian, gallbladder, melanoma, bladder).
- `DATA/recist_measurements.csv`: 82 1D RECIST linear diameter measurements across longitudinal study visits.
- `DATA/17_hbgd_diameter_trajectories.csv`: 74 observation rows across 25 lesions and 12 unique patients with documented diameter progression.
- `DATA/17_hbgd_state_parameters.csv`: **49 validated longitudinal growth intervals** from 12 unique patients with computed growth rates ($r$) and dynamic state labels (`growth`, `stable`, `shrinkage`). This constitutes the sole transfer representation artifact exported from the source cohort.

### B. Target Cancer (LUMIERE Glioblastoma Cohort)
- `DATA/LUMIRE/lumiere_nifti_mask_validation.csv`: Validation log of 476 3D MRI scans across 67 patients confirming voxel dimensions, slice thickness, and HD-GLIO-AUTO segmentation validity.
- `DATA/LUMIRE/lumiere_longitudinal_trajectories.csv`: 476 longitudinal tumor observations across 67 unique patients with `tumor_volume_cm3`, `equivalent_diameter_mm`, and 3D bounding box dimensions.
- `DATA/LUMIRE/revised_outputs/step27_fold_level_metrics.csv`: 175 fold evaluation rows (25 folds x 7 models) recording fold-level MAE and RMSE across 54 development patients.
- `DATA/LUMIRE/revised_outputs/step27_paired_differences.csv`: 54 patient-level paired performance records comparing Target Morphology Baseline vs Domain-Adapted HBGD.
- `DATA/LUMIRE/step19_hbgd_representation_transfer_predictions.csv`: 342 prediction cases (221 train, 46 val, 75 test) across 66 patients.
- `DATA/LUMIRE/step25_hbgd_temporal_predictions.csv`: 276 strict chronological forecast cases across 59 patients.

---

## 5. Script Inventory and Execution Provenance

| Script Relative Path | Lines | Primary Function | Input Dependencies | Key Outputs |
|---|---:|---|---|---|
| `CODES/1_inspect_dataset.py` | 470 | Primary dataset validation | `DATA/patients.csv`, `recist_measurements.csv` | Console summary |
| `CODES/17_build_hbgd_mathematical_model.py` | 1310 | Fits ODEs & extracts state parameters | `DATA/recist_measurements.csv` | `DATA/17_hbgd_state_parameters.csv` |
| `CODES/18_test_hbgd_forward_prediction.py` | 473 | Forward ODE trajectory forecasting | `DATA/17_hbgd_diameter_trajectories.csv` | `DATA/18_hbgd_forward_predictions.csv` |
| `CODES/19_patient_independent_hbgd_validation.py` | 525 | LOPO validation of state models | `DATA/17_hbgd_diameter_trajectories.csv` | `DATA/19_patient_independent_performance.csv` |
| `CODES/21_test_individualized_hbgd.py` | 511 | Tests 2-interval individualized rates | `DATA/17_hbgd_diameter_trajectories.csv` | `DATA/21_individualized_hbgd_performance.csv` |
| `CODES/25_extract_lesion_ct_crops.py` | 572 | Extracts 3D CT bounding box crops | 3D CT Volumes & RECIST labels | `DATA/25_lesion_ct_crops/` (44 crops) |
| `CODES/29_repair_failed_ct_crops.py` | 647 | Zero-padding repair for clipped crops | Clipped CT crops | Repaired crops (100% usability) |
| `CODES/33_train_ct_to_hbgd_rate.py` | 496 | 3D ResNet CNN training on CT crops | `DATA/25_lesion_ct_crops/` | `DATA/33_ct_rate_cnn_results/` |
| `CODES/35_build_hbgd_validation_report.py` | 158 | Consolidates CT pipeline status | Step 1–34 outputs | `DATA/35_HBGD_VALIDATION_REPORT.txt` |
| `CODES/LUMIRE/3_lumiere_tumor_geometry_hbgd.py` | 1906 | Voxel volume to diameter conversion | Segmentations | `DATA/LUMIRE/lumiere_tumor_geometry.csv` |
| `CODES/LUMIRE/8_cross_cancer_hbgd_transfer.py` | 845 | Normalized growth rate transfer | Step 17 source + LUMIERE | `DATA/LUMIRE/lumiere_cross_cancer_transfer_summary.csv` |
| `CODES/LUMIRE/11_domain_adaptive_hbgd_transfer.py` | 798 | Domain-adaptive residual transfer | Step 17 source + LUMIERE | `DATA/LUMIRE/step11_domain_adaptive_transfer_summary.csv` |
| `CODES/LUMIRE/12_hbgd_incremental_information_ablation.py` | 812 | Incremental feature ablation | LUMIERE trajectories | `DATA/LUMIRE/step12_hbgd_ablation_summary.csv` |
| `CODES/LUMIRE/19_hbgd_representation_transfer.py` | 1815 | Representation transfer into geometry | Step 17 source + LUMIERE | `DATA/LUMIRE/step19_..._performance.csv` |
| `CODES/LUMIRE/20_hbgd_transfer_statistical_validation.py` | 575 | Bootstrap & permutation tests of Step 19 | Step 19 predictions | `DATA/LUMIRE/step20_..._summary.csv` |
| `CODES/LUMIRE/22_hbgd_transfer_split_robustness.py` | 741 | 10 repeated patient splits | Step 17 source + LUMIERE | `DATA/LUMIRE/step22_..._summary.csv` |
| `CODES/LUMIRE/23_hbgd_repeated_split_statistics.py` | 537 | Statistical aggregation of 10 splits | Step 22 split CSVs | `DATA/LUMIRE/step23_..._summary.csv` |
| `CODES/LUMIRE/25_hbgd_temporal_holdout_validation.py` | 748 | Strict chronological temporal holdout | Step 17 source + LUMIERE | `DATA/LUMIRE/step25_hbgd_temporal_performance.csv` |
| `CODES/LUMIRE/26_hbgd_final_leakage_audit.py` | 647 | Forensic audit of pipeline leakage | Step 19–25 outputs | `DATA/LUMIRE/step26_hbgd_leakage_audit.csv` |
| `CODES/LUMIRE/27_hbgd_patient_grouped_cv_experiment.py` | 685 | 25-fold patient-grouped CV benchmark | Step 17 source + LUMIERE | `DATA/LUMIRE/revised_outputs/step27_*` |
| `CODES/LUMIRE/test_hbgd_transfer_audit.py` | 123 | Automated 7-test methodological suite | Step 27 functions | Unit test report (7/7 PASS) |

---

## 6. Audit Conclusion on Inventory

All 62 pipeline stages, 63 Python scripts, and associated data artifacts have been verified and localized. No files were missing or corrupted. Authoritative output files were identified for every key claim. Discrepancies between narrative text and CSV numbers were resolved by verifying against raw prediction rows.
