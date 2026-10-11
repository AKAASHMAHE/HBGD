# 8. Experimental Validation Results (Concise Disclosure Summary)

The experimental validation evaluates cross-cancer growth dynamics transfer from a source solid-tumor cohort (DeepLesion CT; 12 longitudinal patients, 49 validated intervals) to a target glioblastoma cohort (LUMIERE MRI; 67 patients, 476 longitudinal scans; 3D segmentations harmonized to equivalent diameter $D_{\text{eq}} = (6V/\pi)^{1/3}$). The objective is to determine whether biological growth rates derived from thoracic/abdominal CT can be transferred to forecast forward brain tumor trajectories under domain-adaptive residual calibration and representation feature transfer.

---

### (a) Cross-Cancer Transfer Performance

Table 8.1 contrasts domain-adaptive transfer against target-domain morphology and reference baselines across primary evaluation protocols.

**Table 8.1 — Comparative Transfer Performance across Independent Evaluation Protocols**

| Experiment ID | Evaluation Protocol | Model Architecture | Evaluated Cases | MAE (mm) | Difference vs Baseline (mm) | Relative Reduction (%) | Statistical Finding |
|---|---|---|---:|---:|---:|---:|---|
| **EXP-LUM-09** (Step 27) | 25-Fold Patient-Grouped CV | Target Morphology Baseline | 54 pts (1605 rows) | 7.6981 | Reference | Reference | Fold-level mean MAE |
| **EXP-LUM-09** (Step 27) | 25-Fold Patient-Grouped CV | Domain-Adapted HBGD | 54 pts (1605 rows) | 7.5301 | +0.1680 | +2.18% | Paired $t=1.2166, p=0.2291$ (Not statistically significant) |
| **EXP-LUM-09** (Step 27) | 25-Fold Patient-Grouped CV | Direct Biological Transfer | 54 pts (1605 rows) | 9.7200 | -1.4893 (vs No-Change) | -18.09% | Unadapted parameter copy worsens error |
| **EXP-LUM-09** (Step 27) | 25-Fold Patient-Grouped CV | No-Change Baseline | 54 pts (1605 rows) | 8.2307 | Reference | Reference | Static clinical benchmark |
| **EXP-LUM-05** (Step 19) | Held-Out Test Split | Target Geometry Baseline | 13 pts (75 cases) | 7.4565 | Reference | Reference | Target baseline on test split |
| **EXP-LUM-05** (Step 19) | Held-Out Test Split | Geometry + Transferred HBGD | 13 pts (75 cases) | 7.3678 | +0.0887 | +1.19% | 9/13 patients improved; permutation $p=0.09739$ |
| **EXP-LUM-03** (Step 11) | Held-Out Test Split | Target Morphology Baseline | 13 pts (88 cases) | 7.8918 | Reference | Reference | Target baseline on test split |
| **EXP-LUM-03** (Step 11) | Held-Out Test Split | Domain-Adapted HBGD | 13 pts (88 cases) | 7.9280 | -0.0362 | -0.46% | No improvement on held-out test split |

---

### (b) Robustness and Temporal Validation

Table 8.2 reports repeated-split stability and forward-in-time forecasting under strict chronological holdout.

**Table 8.2 — Robustness Across Repeated Splits and Strict Chronological Temporal Holdout**

| Experiment ID | Evaluation Protocol | Metric / Comparison | Evaluated Cohort | Measured Outcome | Baseline Comparator | Evidential Finding |
|---|---|---|---|---|---|---|
| **EXP-LUM-07** (Steps 22/23) | 10 Repeated Patient Splits | Split-Level MAE Reduction | 10 splits (130 pt evals) | Mean: +0.0592 mm (+0.57%); Median: +0.0552 mm | Target Geometry | 8 of 10 splits improved; bootstrap 95% CI: [-0.1097, +0.2065] mm (crosses zero) |
| **EXP-LUM-08** (Step 25) | Strict Chronological Holdout | Pooled Forecasting MAE | 59 pts (276 forecasts) | Geometry + HBGD: 7.0272 mm vs Geometry: 7.0506 mm | Target Geometry | Pooled MAE reduction: +0.0234 mm (+0.33%) |
| **EXP-LUM-08** (Step 25) | Strict Chronological Holdout | Patient-Level Change | 59 pts (276 forecasts) | Mean: +0.0364 mm; Median: **-0.0379 mm** | Target Geometry | 28/59 patients improved (47.5%), 31 worsened; median change is negative |

---

### (c) Validation Summary

1. **Direct Transfer Failure**: Uncalibrated biological growth parameter transfer degrades forecasting performance (9.7200 mm vs 8.2307 mm for No-Change), demonstrating that growth rates are not directly invariant across distinct tumor pathologies.
2. **Directional Transfer via Machine Learning**: Domain-adaptive residual correction and representation-level feature transfer achieve modest directional error reductions (+2.18% in cross-validation, +1.19% in representation transfer, +0.33% in temporal holdout).
3. **Absence of Statistical Significance**: The observed improvements are not statistically significant ($p > 0.05$ across all tests: paired t-test $p = 0.2291$; Wilcoxon $p = 0.2010$; permutation $p = 0.09739$).
4. **Mixed and Negative Evidence**: Domain-adaptive residual correction alone showed no improvement on the held-out test split (7.9280 mm vs 7.8918 mm in Step 11), and temporal forecasting produced a negative median patient-level improvement ($-0.0379\text{ mm}$ in Step 25). The architecture demonstrates algorithmic feasibility, but clinical superiority is not established.

---

### Sample Source Code: Domain-Adaptive Growth Transfer

The following implementation illustrates the core domain-adaptation algorithm using strictly relative paths:

```python
import os, math, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Relative dataset paths (portable within project repository)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_PATH = os.path.join(BASE_DIR, "..", "..", "DATA", "17_hbgd_state_parameters.csv")
TARGET_PATH = os.path.join(BASE_DIR, "..", "..", "DATA", "LUMIRE", "lumiere_longitudinal_trajectories.csv")

def train_domain_adaptive_transfer(source_df, target_train_cases, target_test_cases):
    # 1. Fit source-domain growth model
    features_src = ["diameter_start_mm", "interval_days"]
    X_src = source_df[features_src].values
    y_src = source_df["diameter_end_mm"].values
    source_model = RandomForestRegressor(n_estimators=300, max_depth=5, min_samples_leaf=2, random_state=42)
    source_model.fit(X_src, y_src)
    
    # 2. Compute initial predictions and residuals on target training data
    X_tgt_train = target_train_cases[features_src].values
    y_tgt_train = target_train_cases["actual_diameter_mm"].values
    pred_tgt_train = source_model.predict(X_tgt_train)
    residuals_train = y_tgt_train - pred_tgt_train
    
    # 3. Fit target-domain residual correction model
    adapt_model = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=2, random_state=43)
    adapt_model.fit(X_tgt_train, residuals_train)
    
    # 4. Infer adapted forecast on held-out test cases
    X_tgt_test = target_test_cases[features_src].values
    y_tgt_test = target_test_cases["actual_diameter_mm"].values
    base_pred_test = source_model.predict(X_tgt_test)
    correction_test = adapt_model.predict(X_tgt_test)
    final_adapted_pred = base_pred_test + correction_test
    
    return mean_absolute_error(y_tgt_test, final_adapted_pred)
```
