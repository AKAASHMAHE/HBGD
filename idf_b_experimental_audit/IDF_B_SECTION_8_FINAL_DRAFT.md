# 8. Experimental Validation Results

The experimental objective of this evaluation is to determine whether longitudinal biological growth dynamics learned from a source cancer cohort (thoracic and abdominal Computed Tomography) can be transferred to forecast forward-in-time tumor trajectories in an anatomically and pathologically distinct target cancer cohort (brain Glioblastoma multiforme on Magnetic Resonance Imaging). The evaluation framework rigorously distinguishes between direct uncalibrated biological parameter transfer, domain-adaptive machine-learning residual correction, and growth-representation feature augmentation, benchmarking each approach against established target-domain morphology and static reference models.

---

### A. Experimental Dataset and Setup

The quantitative validation is conducted across two multi-modal clinical imaging datasets:
1. **Source Cohort**: DeepLesion solid tumor cohort comprising thoracic and abdominal CT scans with one-dimensional (1D) RECIST 1.1 linear axial diameter measurements across 12 unique patients, providing 49 validated longitudinal growth intervals from 25 distinct tumor trajectories.
2. **Target Cohort**: The public LUMIERE dataset comprising longitudinal multi-sequence 3D brain Magnetic Resonance Imaging (MRI) scans across 67 unique glioblastoma patients (476 total scan visits). 3D voxel tumor segmentation volumes ($V$) are converted to equivalent spherical diameters using $D_{\text{eq}} = (6V/\pi)^{1/3}$.

The experimental configurations, dataset splits, and evaluation metrics are summarized in Table 8.1.

**Table 8.1 — Experimental Dataset, Cohort Partitioning, and Validation Parameters**

| Parameter | Source Cohort (DeepLesion CT) | Target Cohort (LUMIERE Glioblastoma MRI) |
|---|---|---|
| Primary Pathology | Thoracic & Abdominal Solid Tumors | Glioblastoma Multiforme (GBM) |
| Imaging Modality | Computed Tomography (CT) | 3D Magnetic Resonance Imaging (MRI) |
| Measurement Modality | 1D RECIST Linear Diameter | 3D Voxel Segmentations $\to$ Equivalent Diameter $D_{\text{eq}}$ |
| Total Available Patients | 22 patients | 67 patients |
| Longitudinal Trajectory Patients | 12 unique patients | 67 unique patients (476 longitudinal visits) |
| Validated Growth Intervals | 49 validated intervals (from 25 trajectories) | 342 prediction cases across 66 patients |
| Development Cohort Split | 12 patients (49 training intervals) | 54 patients (321 prediction cases per repeat) |
| Held-Out Test Cohort Split | N/A (Source reference) | 13 patients (75 to 88 test prediction cases) |
| Strict Temporal Holdout Cohort | N/A (Source reference) | 59 eligible patients (276 chronological forecasts) |
| Prediction Target | Future Linear Diameter $D(t)$ (mm) | Future Equivalent Diameter $D_{\text{eq}}(t)$ (mm) |
| Primary Evaluation Metric | Mean Absolute Error (MAE, mm) | Mean Absolute Error (MAE, mm) |
| Secondary Evaluation Metric | Root Mean Squared Error (RMSE, mm) | Root Mean Squared Error (RMSE, mm) |

---

### B. Comparative Prediction Performance

The primary cross-cancer transfer hypothesis was evaluated under a repeated patient-grouped cross-validation protocol (Experiment ID: **EXP-LUM-09**, Step 27). The evaluation utilized 5 repeats of 5-fold cross-validation (25 total folds) across the 54 development patients, ensuring that repeated observations from the same patient never spanned training and validation partitions.

**Table 8.2 — Primary Comparative Prediction Performance (Repeated Patient-Grouped Cross-Validation, EXP-LUM-09)**

| Experiment ID | Method / Architecture | Evaluation Protocol | Validation MAE (mm) | Validation RMSE (mm) | Difference vs Target Baseline (mm) | Relative Reduction (%) | Statistical Finding |
|---|---|---|---:|---:|---:|---:|---|
| **EXP-LUM-09** | Target Morphology Baseline | 25-Fold Patient-Grouped CV | 7.6981 | 10.9457 | Reference | Reference | Target-domain benchmark |
| **EXP-LUM-09** | Domain-Adapted HBGD | 25-Fold Patient-Grouped CV | 7.5301 | 10.7075 | +0.1680 | +2.18% | $t = 1.2166, p = 0.2291$ (Not statistically significant) |
| **EXP-LUM-09** | Morphology + Adapted HBGD | 25-Fold Patient-Grouped CV | 7.8140 | 11.0661 | -0.1159 | -1.51% | No incremental benefit over Domain-Adapted |

*Note: Metrics reflect the arithmetic mean of 25 fold-level validation metrics across 54 development patients (1605 prediction rows). A positive difference indicates lower error for HBGD. Neither the paired t-test ($p = 0.2291$) nor the Wilcoxon signed-rank test ($p = 0.2010$) demonstrates statistical significance at $\alpha = 0.05$.*

To test whether source-derived biological growth parameters can be directly applied without target-domain adaptation, uncalibrated source growth rate priors were evaluated under the exact same 25-fold patient-grouped cross-validation protocol.

**Table 8.3 — Direct Biological Parameter Transfer versus Target and Static Baselines (EXP-LUM-09)**

| Model Architecture | Mathematical Formulation | Fold-Mean MAE (mm) | Fold-Mean RMSE (mm) | Difference vs No-Change (mm) | Relative Error vs No-Change (%) | Finding on Biological Transfer |
|---|---|---:|---:|---:|---:|---|
| **No-Change Baseline** | $D(t) = D_0$ | 8.2307 | 11.8125 | Reference | Reference | Static clinical benchmark |
| **Target Morphology Baseline** | Random Forest on Target Features | 7.6981 | 10.9457 | +0.5326 | +6.47% | Supervised target model |
| **Domain-Adapted HBGD** | Source Model + Residual Correction | 7.5301 | 10.7075 | +0.7006 | +8.51% | Best performing transfer model |
| **Direct Biological Parameter Transfer** | $D(t) = D_0 \cdot \exp(r_{\text{source}} \cdot t)$ | 9.7200 | 13.0660 | -1.4893 | -18.09% | Substantial deterioration versus baselines |

*Note: Direct uncalibrated transfer of the source biological growth rate parameter ($r_{\text{source}} = -0.001137\text{ day}^{-1}$) performs substantially worse than both the target morphology baseline (7.6981 mm) and the static No-Change baseline (8.2307 mm). Predictive benefit is driven strictly by machine-learning domain-adaptive residual calibration, not biological parameter invariance across cancer types.*

In addition to repeated cross-validation, the domain-adaptive residual correction architecture was evaluated on the independent held-out test cohort of 13 patients (88 prediction cases) in Experiment **EXP-LUM-03** (Step 11). Under this evaluation, the Target Morphology Baseline achieved an MAE of **7.8918 mm**, whereas Domain-Adapted HBGD achieved an MAE of **7.9280 mm** (a negative difference of $-0.0362\text{ mm}$, or $-0.46\%$). Therefore, on this single held-out test split, domain-adaptive residual correction alone showed no performance improvement over the target baseline.

---

### C. Additional Validation and Statistical Analysis

#### 1. Representation-Level Growth Transfer (Step 19)
A complementary transfer architecture was evaluated in Experiment **EXP-LUM-05** (Step 19), where a source-trained Random Forest model was used to predict an individualized growth-state velocity parameter ($r_{\text{transferred}}$), which was subsequently incorporated as an extra numerical feature alongside target geometric variables (`previous_diameter_mm`, `current_diameter_mm`, `diameter_change_mm`).

On the held-out test cohort of 13 patients (75 prediction cases):
- **Target Geometry Baseline MAE**: $7.4565\text{ mm}$ (RMSE: $11.3638\text{ mm}$)
- **Geometry + Transferred HBGD MAE**: $\mathbf{7.3678\text{ mm}}$ (RMSE: $\mathbf{11.1678\text{ mm}}$)
- **Absolute MAE Reduction**: $+0.0887\text{ mm}$ (Relative error reduction: $+1.19\%$)
- **Patient-Level Response**: 9 out of 13 patients exhibited lower prediction error with HBGD feature augmentation, while 4 patients exhibited higher error.

Patient-level statistical analysis of this held-out comparison (Experiment **EXP-LUM-06**, Step 20) demonstrated a patient-level mean improvement of $+0.2062\text{ mm}$ and a median improvement of $+0.1613\text{ mm}$. A paired permutation test yielded $p = 0.09739$, a Wilcoxon signed-rank test yielded $p = 0.1465$, and the 95% bootstrap confidence interval was $[-0.0083, +0.4210]\text{ mm}$. Because the confidence interval encompasses zero and $p > 0.05$, the improvement is directionally positive but is not statistically significant.

#### 2. Strict Chronological Temporal Holdout Validation (Step 25)
To assess operational deployment viability, Experiment **EXP-LUM-08** (Step 25) executed a chronological temporal holdout across 59 eligible glioblastoma patients (276 temporal forecast cases). Models were trained strictly on historical observations preceding each prediction origin ($t \le t_{\text{origin}}$) and evaluated on subsequent chronological visits ($t_{\text{target}} > t_{\text{origin}}$).

**Table 8.4 — Strict Chronological Temporal Forecasting Performance (EXP-LUM-08, Step 25)**

| Model Architecture | Forecast Cases ($N$) | Patient Count | Forecasting MAE (mm) | Forecasting RMSE (mm) | Difference vs Geometry (mm) | Relative Reduction (%) |
|---|---:|---:|---:|---:|---:|---:|
| **M0: No-Change Baseline** | 276 | 59 | 7.0034 | 10.5940 | +0.0472 | +0.67% |
| **M1: Target Geometry Baseline** | 276 | 59 | 7.0506 | 10.0760 | Reference | Reference |
| **M2: Geometry + Transferred HBGD** | 276 | 59 | 7.0272 | 9.9688 | +0.0234 | +0.33% |
| **M3: Direct Transferred HBGD** | 276 | 59 | 7.6441 | 10.9683 | -0.5935 | -8.42% |

*Note: In pooled analysis, Geometry + Transferred HBGD yielded a modest $+0.0234\text{ mm}$ ($+0.33\%$) error reduction over Geometry alone. At the patient level, the mean improvement was $+0.0364\text{ mm}$, whereas the median patient-level change was negative at $\mathbf{-0.0379\text{ mm}}$. A total of 28 of 59 patients (47.46%) improved, while 31 patients (52.54%) worsened. Direct unadapted transfer (M3) degraded accuracy by $-8.42\%$. The static No-Change baseline (7.0034 mm) achieved the lowest overall error.*

#### 3. Incremental Feature Ablation (Step 12)
Experiment **EXP-LUM-04** (Step 12) dissected the individual information contributions of geometric, temporal-history, and dynamical growth features on the held-out test cohort ($N = 75$ cases across 13 patients):
- **M0: No-Change Static Baseline**: $\text{MAE} = 9.0360\text{ mm}$
- **M1: Target Geometry Baseline**: $\text{MAE} = 8.0170\text{ mm}$
- **M2: Geometry + Temporal History**: $\text{MAE} = 7.9180\text{ mm}$ ($+1.23\%$ error reduction vs M1)
- **M3: Geometry + HBGD Growth Rate ($r$)**: $\text{MAE} = 7.9392\text{ mm}$ ($+0.97\%$ error reduction vs M1)
- **M4: Geometry + Rate Variability ($\sigma_r$)**: $\text{MAE} = 8.0126\text{ mm}$ ($+0.06\%$ error reduction vs M1)
- **M5: Geometry + All HBGD Features**: $\text{MAE} = 7.9664\text{ mm}$ ($+0.63\%$ error reduction vs M1)

*Finding*: Incorporating temporal history alone (M2: 7.9180 mm) yielded a greater reduction in error than adding the HBGD growth rate alone (M3: 7.9392 mm) or all HBGD features combined (M5: 7.9664 mm). This confirms that a substantial component of the predictive signal arises from longitudinal multi-visit history rather than the exponential growth formulation in isolation.

#### 4. Exploratory 3D CNN Volumetric CT Regressor (Step 33)
An exploratory imaging extension was prototyped in Experiment **EXP-CT-05** (Step 33) to evaluate whether a 3D Convolutional Neural Network (3D ResNet-18) operating directly on $64 \times 64 \times 64$ voxel sub-volume CT crops could predict tumor growth rate ($r$) to parameterize forward trajectory ODE forecasting.

The end-to-end data processing pipeline executed from raw DICOM to 3D crop extraction, bounding box zero-padding repair, 3D CNN rate inference, and forward ODE integration. However, due to severe data scarcity in the primary dataset, the independent test set contained **exactly one usable longitudinal test trajectory** (Patient-30, 1 forward observation, 62-day horizon). On this single observation:
- **3D CNN-HBGD Forecast Error**: $\text{MAE} = 2.9257\text{ mm}$
- **No-Change Baseline Error**: $\text{MAE} = 1.0000\text{ mm}$

Because this evaluation comprises only a single test observation, it cannot be considered generalizable, statistically reliable, or clinically validated evidence. It is documented strictly as an engineering feasibility demonstration of the hybrid imaging-to-ODE pipeline.

---

### D. Experimental Validation Summary

The experimental evaluation provides a rigorous, objective characterization of cross-cancer growth dynamics transfer:

1. **Failure of Direct Parameter Copy**: Direct, uncalibrated transfer of biological growth rates across disparate cancer types degrades forecasting performance relative to target and static baselines (9.7200 mm vs 8.2307 mm in Step 27; 10.1261 mm vs 7.5529 mm in Step 8). Biological growth rates are not directly invariant across cancer types or imaging modalities.
2. **Directional Transfer via Machine Learning**: When source-derived growth parameters are transferred through domain-adaptive residual calibration (Step 27) or representation-level feature concatenation (Step 19), modest directional reductions in forecasting error are observed ($+2.18\%$ in 25-fold cross-validation; $+1.19\%$ on the held-out test cohort).
3. **Statistical Uncertainty**: Across all evaluated protocols, the observed performance improvements did not achieve statistical significance at $\alpha = 0.05$ (Step 27: paired t-test $p = 0.2291$, Wilcoxon $p = 0.2010$; Step 20: permutation $p = 0.09739$, bootstrap 95% CI encompasses zero).
4. **Negative and Mixed Results**: On the independent held-out test cohort in Step 11, domain-adaptive residual correction alone showed no improvement over the target baseline (7.9280 mm vs 7.8918 mm). Under strict chronological temporal forecasting (Step 25), the median patient-level improvement was negative ($-0.0379\text{ mm}$), with only 28 of 59 patients exhibiting lower error.
5. **Scope of Evidence**: The empirical results substantiate the technical feasibility of domain-adapted representation transfer as an algorithmic framework. However, they do not establish clinical superiority, universal biological generalizability, or diagnostic efficacy. Larger multi-center longitudinal imaging cohorts remain necessary for independent clinical validation.
