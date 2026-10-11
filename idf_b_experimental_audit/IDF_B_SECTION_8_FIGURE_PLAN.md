# IDF-B Section 8 Figure Plan and Visual Evidence Specification

**Project:** Hierarchical Biological Growth Dynamics (HBGD) Cross-Cancer Transfer  
**Location:** `C:\HBGD\idf_b_experimental_audit\IDF_B_SECTION_8_FIGURE_PLAN.md`  
**Date of Audit:** October 11, 2026  
**Auditor:** Machine Learning Research Auditor & Patent Documentation Assistant  

---

## 1. Overview and Evidential Grounding Rules

This document establishes the official Figure Plan for Section 8 of the Indian Invention Disclosure (IDF-B). In strict compliance with audit standards and user instructions:
1. **Schematic Classification**: Figures 1, 2, and 6 are classified and explicitly marked as **Schematic Architectural Diagrams**. They illustrate computational workflows and data structures; they do not represent measured empirical performance.
2. **Empirical Data Grounding**: Figures 3, 4, and 5 are classified as **Measured Experimental Data Plots**. They are built exclusively from verified project CSV files with 100% mathematical traceability. No illustrative curves or simulated data are permitted to masquerade as measured findings.
3. **Repository Image Status**: Forensic inspection confirmed that no pre-rendered `.png`, `.jpg`, `.svg`, or `.pdf` files exist in the project repository. All empirical plots specified below are reproducible directly from the preserved CSV prediction artifacts.

---

## 2. Comprehensive Figure Registry

| Figure Number | Figure Title | Figure Classification | Source Experiment ID | Supporting Data / Script Path | Patent Purpose |
|---|---|---|---|---|---|
| **Figure 8.1** | End-to-End HBGD Multi-Modal Progression System Architecture | **Schematic Diagram** | N/A (System Architecture) | Conceptual design from `CODES/LUMIRE/` & `CODES/` | Illustrates complete ingestion, harmonization, ODE parameterization, and transfer engine. |
| **Figure 8.2** | Cross-Cancer Growth State Representation Transfer Dataflow | **Schematic Diagram** | Steps 11, 19, 27 | Architectural workflow of `11_domain_adaptive_hbgd_transfer.py` & `19_hbgd_representation_transfer.py` | Contrasts direct parameter copy against residual adaptation and feature concatenation. |
| **Figure 8.3** | Comparative Forecasting Error across 25 Patient-Grouped Cross-Validation Folds | **Measured Experimental Data** | **EXP-LUM-09** (Step 27) | `DATA/LUMIRE/revised_outputs/step27_fold_level_metrics.csv` & `step27_aggregated_model_comparison.csv` | Empirical boxplot/bar plot comparing Domain-Adapted HBGD vs Target Morphology Baseline vs Direct Transfer. |
| **Figure 8.4** | Patient-Level Error Differences and Representation Transfer Distribution | **Measured Experimental Data** | **EXP-LUM-05** / **EXP-LUM-06** (Steps 19 & 20) | `DATA/LUMIRE/step19_hbgd_representation_transfer_predictions.csv` & `step20_hbgd_transfer_patient_results.csv` | Empirical scatter / paired difference plot showing 9/13 improved patients and confidence intervals. |
| **Figure 8.5** | Strict Chronological Temporal Forecasting Error Trajectories | **Measured Experimental Data** | **EXP-LUM-08** (Step 25) | `DATA/LUMIRE/step25_hbgd_temporal_predictions.csv` & `step25_hbgd_temporal_performance.csv` | Scatter and error distribution across 276 strictly chronological forecasts over 59 patients. |
| **Figure 8.6** | Volumetric CT Sub-Volume Extraction and Bounding-Box Zero-Padding Repair | **Schematic Diagram** | Steps 25, 26, 29 | Pipeline logic in `CODES/25_extract_lesion_ct_crops.py` & `CODES/29_repair_failed_ct_crops.py` | Technical illustration of 3D CT centroid cropping, clipping detection, and zero-padding boundary repair. |

---

## 3. Detailed Specifications for Each Figure

### Figure 8.1 — End-to-End HBGD Multi-Modal Progression System Architecture
- **Classification**: **Schematic Diagram** (Not empirical data)
- **Concept & Layout**:
  Block diagram mapping multi-modal input processing:
  1. *Input Block*: Multi-modal imaging inputs (1D RECIST axial CT diameters from Chest/Abdomen vs 3D multi-sequence Brain MRI segmentations).
  2. *Geometric Harmonization*: Voxel integration and spherical equivalent conversion ($D_{\text{eq}} = (6V/\pi)^{1/3}$).
  3. *Dynamic Parameterization*: Finite difference velocity extraction ($r = \ln(D_2/D_1)/\Delta t$) and operational state classification (`Growth`, `Stable`, `Shrinkage`).
  4. *Transfer Engine*: Dual-branch execution decoupling Source Model Fitting from Target Domain Adaptation.
  5. *Prediction Output*: Forward ODE numerical integration ($D(t) = D_0 e^{rt}$) generating continuous trajectory curves.
- **Suggested Patent Caption**:
  > **Figure 8.1: Schematic Block Diagram of the End-to-End HBGD Multi-Modal Tumor Progression Forecasting System.** The diagram illustrates multi-modal ingestion of linear CT measurements and volumetric MRI segmentations, geometric harmonization via equivalent spherical diameter conversion, dynamic growth velocity parameterization, and domain-adapted representation transfer into forward Ordinary Differential Equation (ODE) trajectory forecasting.

---

### Figure 8.2 — Cross-Cancer Growth State Representation Transfer Dataflow
- **Classification**: **Schematic Diagram** (Not empirical data)
- **Concept & Layout**:
  Process flowchart detailing the two distinct transfer mechanisms:
  - *Branch A (Residual Correction)*: Source Cohort ($N = 12$ patients, 49 intervals) $\to$ Source Random Forest Regressor $\to$ Initial Target Diameter Prediction $\to$ Target Training Residual Regressor ($\Delta y = y - \hat{y}_{\text{source}}$) $\to$ Domain-Adapted Forecast ($\hat{y}_{\text{adapted}} = \hat{y}_{\text{source}} + \Delta \hat{y}$).
  - *Branch B (Feature Representation Transfer)*: Source Cohort $\to$ Scaled Growth Rate Regressor $\to$ Transferred Velocity ($r_{\text{transferred}}$) $\to$ Feature Matrix Concatenation with Target Geometry $\to$ Target Fusion Random Forest Regressor.
- **Suggested Patent Caption**:
  > **Figure 8.2: Schematic Process Flowchart of Cross-Cancer Representation Transfer Mechanisms.** The diagram depicts the computational separation between source-domain feature extraction (49 validated intervals from 12 solid-tumor CT patients) and target-domain glioblastoma progression modeling, contrasting residual error domain adaptation (Branch A) against representation-level feature concatenation (Branch B).

---

### Figure 8.3 — Comparative Forecasting Error across 25 Patient-Grouped Cross-Validation Folds
- **Classification**: **Measured Experimental Data Plot**
- **Authoritative Data Source**:
  - `DATA/LUMIRE/revised_outputs/step27_fold_level_metrics.csv` (175 fold evaluation rows across 25 folds)
  - `DATA/LUMIRE/revised_outputs/step27_aggregated_model_comparison.csv`
- **Plot Structure**:
  - Grouped boxplot or bar chart with error bars showing validation Mean Absolute Error (MAE, mm) across the 25 evaluation folds (5 repeats x 5 folds):
    1. *Domain_Adapted_HBGD*: Mean $= 7.5301\text{ mm}$, Median $= 7.3560\text{ mm}$, $95\%\text{ CI} = [5.7375, 10.1778]\text{ mm}$
    2. *Target_Morphology_History_Baseline*: Mean $= 7.6981\text{ mm}$, Median $= 7.5920\text{ mm}$, $95\%\text{ CI} = [5.9098, 10.3963]\text{ mm}$
    3. *Morphology_Plus_Adapted_HBGD*: Mean $= 7.8140\text{ mm}$, Median $= 7.5793\text{ mm}$
    4. *No_Change_Baseline*: Mean $= 8.2307\text{ mm}$, Median $= 7.9801\text{ mm}$
    5. *Direct_Biological_Parameter_Transfer*: Mean $= 9.7200\text{ mm}$, Median $= 9.5086\text{ mm}$
- **What It Demonstrates**:
  Empirically establishes that Domain-Adapted HBGD achieves a modest directional reduction in validation MAE ($7.5301\text{ mm}$ vs $7.6981\text{ mm}$, $+2.18\%$), while direct uncalibrated biological transfer degrades accuracy ($9.7200\text{ mm}$).
- **Suggested Patent Caption**:
  > **Figure 8.3: Measured Validation MAE across 25 Patient-Grouped Cross-Validation Folds (Experiment EXP-LUM-09, Step 27).** Bar plots with standard error and 95% confidence intervals illustrate fold-level forecasting errors across 54 development glioblastoma patients (1605 prediction rows). Domain-Adapted HBGD (7.5301 mm) exhibits a 2.18% directional error reduction compared with the Target Morphology Baseline (7.6981 mm), whereas Direct Biological Parameter Transfer (9.7200 mm) substantially worsens error compared with the static No-Change baseline (8.2307 mm).

---

### Figure 8.4 — Patient-Level Error Differences and Representation Transfer Distribution
- **Classification**: **Measured Experimental Data Plot**
- **Authoritative Data Source**:
  - `DATA/LUMIRE/step19_hbgd_representation_transfer_predictions.csv` (75 test cases across 13 held-out patients)
  - `DATA/LUMIRE/step20_hbgd_transfer_patient_results.csv`
- **Plot Structure**:
  - Paired waterfall or difference bar chart showing individual patient MAE differences ($\Delta \text{MAE} = \text{MAE}_{\text{Geometry}} - \text{MAE}_{\text{Fusion}}$) across the 13 held-out test patients.
  - Positive bars indicate lower error for the HBGD-augmented model (9 patients: `Patient-010`, `012`, `019`, `022`, `037`, `038`, `059`, `065`, `085`).
  - Negative bars indicate worsened error (4 patients: `Patient-033`, `042`, `064`, `073`).
  - Inset shows bootstrap distribution of mean improvement with 95% confidence interval $[-0.0083, +0.4210]\text{ mm}$.
- **What It Demonstrates**:
  Demonstrates that representation transfer improves forecasting in a majority of individual patients (9 of 13), but the bootstrap confidence interval encompassing zero reflects statistical uncertainty.
- **Suggested Patent Caption**:
  > **Figure 8.4: Measured Patient-Level Error Reductions from Representation-Level Growth Transfer (Experiment EXP-LUM-05/06, Steps 19 & 20).** Paired difference plot across the 13 held-out test glioblastoma patients (75 prediction cases). Nine of 13 patients exhibit positive error reductions when geometry is augmented with transferred HBGD growth representations (mean improvement: +0.2062 mm, median: +0.1613 mm), while four patients worsen. The 95% bootstrap confidence interval ([-0.0083, +0.4210] mm) illustrates directional benefit without statistical significance (permutation p = 0.09739).

---

### Figure 8.5 — Strict Chronological Temporal Forecasting Error Trajectories
- **Classification**: **Measured Experimental Data Plot**
- **Authoritative Data Source**:
  - `DATA/LUMIRE/step25_hbgd_temporal_predictions.csv` (276 forecast cases across 59 patients)
  - `DATA/LUMIRE/step25_hbgd_temporal_performance.csv`
- **Plot Structure**:
  - Scatter plot of Actual Equivalent Diameter ($y$-axis, mm) versus Predicted Equivalent Diameter ($x$-axis, mm) for Geometry alone (M1) and Geometry + Transferred HBGD (M2).
  - Line of identity ($y = x$) indicating perfect forecasting.
  - Marginal boxplot comparing absolute error distributions across forecast horizons.
- **What It Demonstrates**:
  Empirically confirms that under strict forward chronological forecasting ($t_{\text{target}} > t_{\text{origin}}$), Geometry + Transferred HBGD achieves $\text{MAE} = 7.0272\text{ mm}$ vs Geometry $\text{MAE} = 7.0506\text{ mm}$ ($+0.33\%$ improvement), while documenting that patient-level median improvement is slightly negative ($-0.0379\text{ mm}$).
- **Suggested Patent Caption**:
  > **Figure 8.5: Measured Chronological Forward Forecasting Performance under Strict Temporal Holdout (Experiment EXP-LUM-08, Step 25).** Scatter plot of predicted versus observed tumor equivalent diameter across 276 forward forecasts in 59 eligible glioblastoma patients. Training strictly preceded prediction origins. Augmented HBGD forecasting achieved a pooled MAE of 7.0272 mm compared with 7.0506 mm for geometry alone (+0.33% reduction), with 28 of 59 patients improved.

---

### Figure 8.6 — Volumetric CT Sub-Volume Extraction and Bounding-Box Zero-Padding Repair
- **Classification**: **Schematic Diagram** (Not empirical data)
- **Concept & Layout**:
  Three-panel engineering schematic:
  - *Panel A (Centroid Localization)*: Full axial CT slice $\to$ 2D RECIST axial diameter endpoints $\to$ 3D centroid calculation $(x_c, y_c, z_c)$.
  - *Panel B (Boundary Clipping Detection)*: $64 \times 64 \times 64$ voxel bounding box overlay showing anatomical edge truncation (8 of 44 crops clipped at volume margins in Step 26).
  - *Panel C (Zero-Padding Repair)*: Step 29 mathematical repair expanding image grid via symmetric zero-padding, preserving lesion morphology without voxel distortion.
- **Suggested Patent Caption**:
  > **Figure 8.6: Schematic Illustration of 3D CT Centroid Localization, Crop Boundary Clipping, and Zero-Padding Repair.** The diagram details the lesion sub-volume extraction pipeline: calculating 3D lesion centroids from axial RECIST coordinates, detecting peripheral bounding box clipping (Step 26), and applying zero-padded spatial grid expansion (Step 29) to restore 100% usability across volumetric imaging crops.

---

## 4. Evidential Summary on Figure Implementation

By strictly segregating schematic workflow diagrams (Figures 8.1, 8.2, 8.6) from empirical performance plots (Figures 8.3, 8.4, 8.5), the visual evidence presented in Section 8 provides transparent, reproducible support for the patent disclosure without risking misrepresentation of experimental data.
