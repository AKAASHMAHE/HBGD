# Step 21 — Cross-Cancer HBGD Evidence Synthesis

## 1. Purpose

This report consolidates the cross-cancer HBGD experiments performed in Steps 8–20. No additional model fitting is performed in Step 21. The purpose is to distinguish demonstrated findings from preliminary observations and unsupported claims.

## 2. Validated experimental progression

### Step 8 — Simple transfer

Simple source-to-target HBGD distribution normalization did not establish predictive transfer beyond the target geometry baseline.

### Step 9 — Phenotype comparison

Descriptive HBGD-derived phenotype comparison showed similarities between the source cancer and LUMIERE GBM across several temporal features. This was exploratory and did not establish statistical alignment.

### Step 10 — Statistical alignment

The observed standardized phenotype distance was 2.659276 with a permutation p-value of 1.000000. Therefore, the simple phenotype-distance alignment approach did not establish cross-cancer statistical alignment.

### Step 11 — Domain adaptation

Domain adaptation reduced the error of direct HBGD transfer but did not outperform the target-domain geometry baseline. This indicates that simple residual correction is insufficient to solve the domain mismatch.

### Steps 12–18 — Incremental HBGD signal

The ablation and patient-level analyses consistently suggested a modest incremental contribution from HBGD-derived information beyond geometry. The signal was observed across multiple phenotypes, with the stable phenotype showing the strongest preliminary effect in Step 17. However, the small number of held-out patients prevented firm statistical conclusions.

### Step 19 — Representation-level transfer

The corrected representation-level transfer experiment used the validated source HBGD state-parameter artifact containing 49 source intervals from 12 patients. The exact 13-patient LUMIERE test cohort was preserved.

On the held-out test cohort, geometry-only prediction achieved an MAE of 7.4565 mm, while geometry combined with the transferred HBGD representation achieved 7.3678 mm. This corresponds to a 1.19% reduction in MAE.

The transferred HBGD representation alone did not outperform geometry, demonstrating that its observed value is complementary rather than independently sufficient.

### Step 20 — Statistical validation

At the patient level, 9 of 13 held-out patients improved with transferred HBGD information and 4 worsened. Mean improvement was 0.2062 mm and median improvement was 0.1613 mm. The paired effect size was 0.5062.

However, the 95% bootstrap confidence interval was [-0.0083, 0.4210] mm, the paired permutation p-value was 0.097390, and the Wilcoxon p-value was 0.146484. Therefore, the observed improvement is not statistically significant at the conventional 0.05 threshold.

## 3. Overall evidence assessment

| Evidence area | Assessment |
|---|---|
| HBGD mathematical formulation | Established in source pipeline |
| HBGD-derived predictive signal | Preliminary |
| Simple cross-cancer rate transfer | Not supported |
| Simple phenotype alignment | Not supported statistically |
| Domain-adaptive transfer | Not superior to geometry |
| HBGD incremental information | Preliminary positive |
| Representation-level cross-cancer transfer | Preliminary positive |
| Statistical proof of cross-cancer transfer | Not established |

## 4. Strongest current result

The strongest current cross-cancer observation is that a source-derived HBGD growth representation, when combined with target-domain tumor geometry, produced a directional improvement on the untouched LUMIERE test cohort. The effect was observed in 9 of 13 patients and had a moderate paired effect size of 0.5062.

The improvement remains preliminary because the bootstrap confidence interval crosses zero and both paired statistical tests have p-values above 0.05.

## 5. Recommended scientific claim

> A source-derived HBGD growth representation demonstrated preliminary cross-cancer transferability by providing a small incremental predictive contribution beyond target-domain tumor geometry in a held-out LUMIERE glioblastoma cohort. The effect was directionally consistent at the patient level but did not reach statistical significance, indicating that larger independent longitudinal cohorts are required for confirmation.

## 6. Claims that should NOT be made

- HBGD transfer is statistically proven.
- HBGD transfer is clinically validated.
- HBGD universally transfers between cancer types.
- Source growth rates can be directly copied to GBM.
- HBGD outperforms tumor geometry.
- The 1.19% improvement is statistically significant.

## 7. Main limitations

1. The source HBGD cohort contains only 12 patients and 49 validated longitudinal intervals.
2. The LUMIERE held-out test cohort contains only 13 patients.
3. The cross-cancer improvement is modest.
4. Statistical confidence intervals include zero.
5. The source and target cancers differ biologically and imaging-wise.
6. Larger independent longitudinal datasets are needed for replication.

## 8. Final conclusion

The accumulated experiments do not establish HBGD as a statistically validated cross-cancer predictor. However, they provide a coherent preliminary signal that growth-dynamics information derived from HBGD may contain transferable information that complements target-domain tumor geometry. The representation-level experiment provides the clearest current evidence for this hypothesis, while the statistical validation indicates that the effect remains uncertain because of the limited sample size.
