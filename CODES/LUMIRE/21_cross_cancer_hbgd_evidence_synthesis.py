# -*- coding: utf-8 -*-

"""
STEP 21
CROSS-CANCER HBGD EVIDENCE SYNTHESIS

Purpose
-------
Consolidate the validated findings from Steps 8-20 into a single
cross-cancer HBGD evidence package.

NO MODEL TRAINING IS PERFORMED.

This script reads previously generated CSV/report outputs and creates:

1. Master evidence table
2. Cross-cancer performance comparison
3. Statistical evidence summary
4. Patient-level robustness summary
5. Final scientific interpretation report

Important:
-----------
This script does not alter previous results and does not attempt to
reconcile conflicting experiments by changing their numbers.

The interpretation distinguishes:
- demonstrated findings
- preliminary findings
- unsupported/negative findings
- limitations
"""

import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA"
LUMIERE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"

if not os.path.exists(LUMIERE_DIR):
    os.makedirs(LUMIERE_DIR)


# ============================================================
# FILE HELPER
# ============================================================

def path(name):
    return os.path.join(
        LUMIERE_DIR,
        name
    )


def read_csv_safe(filename):

    filepath = path(filename)

    if not os.path.exists(filepath):

        print(
            "WARNING: Missing:",
            filepath
        )

        return None

    try:

        df = pd.read_csv(
            filepath
        )

        print(
            "Loaded:",
            filename,
            "| rows:",
            len(df)
        )

        return df

    except Exception as e:

        print(
            "WARNING: Could not read:",
            filename,
            "|",
            str(e)
        )

        return None


def find_value(
    df,
    metric_name
):

    if df is None:
        return np.nan

    if "metric" not in df.columns:
        return np.nan

    matches = df[
        df["metric"].astype(str)
        == metric_name
    ]

    if len(matches) == 0:
        return np.nan

    try:
        return float(
            matches["value"].iloc[0]
        )

    except Exception:
        return np.nan


# ============================================================
# START
# ============================================================

print("=" * 78)
print("STEP 21 CROSS-CANCER HBGD EVIDENCE SYNTHESIS")
print("=" * 78)


# ============================================================
# LOAD STEP 8
# ============================================================

step8 = read_csv_safe(
    "step8_cross_cancer_hbgd_transfer_summary.csv"
)

step8_perf = read_csv_safe(
    "step8_cross_cancer_hbgd_transfer_predictions.csv"
)


# ============================================================
# LOAD STEP 9
# ============================================================

step9_stats = read_csv_safe(
    "cross_cancer_hbgd_feature_statistics.csv"
)

step9_distances = read_csv_safe(
    "cross_cancer_hbgd_feature_distances.csv"
)

step9_state = read_csv_safe(
    "cross_cancer_hbgd_state_comparison.csv"
)


# ============================================================
# LOAD STEP 10
# ============================================================

step10_summary = read_csv_safe(
    "step10_hbgd_transfer_summary.csv"
)

step10_alignment = read_csv_safe(
    "step10_hbgd_alignment_statistics.csv"
)


# ============================================================
# LOAD STEP 11
# ============================================================

step11_summary = read_csv_safe(
    "step11_domain_adaptive_transfer_summary.csv"
)

step11_source = read_csv_safe(
    "step11_source_model_performance.csv"
)


# ============================================================
# LOAD STEP 12
# ============================================================

step12_summary = read_csv_safe(
    "step12_hbgd_ablation_summary.csv"
)


# ============================================================
# LOAD STEP 13
# ============================================================

step13_summary = read_csv_safe(
    "step13_hbgd_statistical_summary.csv"
)

step13_patient = read_csv_safe(
    "step13_patient_level_hbgd_results.csv"
)


# ============================================================
# LOAD STEP 16
# ============================================================

step16_summary = read_csv_safe(
    "step16_exact_phenotype_model_comparison.csv"
)


# ============================================================
# LOAD STEP 17
# ============================================================

step17_summary = read_csv_safe(
    "step17_phenotype_robustness_summary.csv"
)


# ============================================================
# LOAD STEP 18
# ============================================================

step18_corr = read_csv_safe(
    "step18_hbgd_benefit_correlations.csv"
)

step18_pheno = read_csv_safe(
    "step18_hbgd_benefit_by_phenotype.csv"
)

step18_history = read_csv_safe(
    "step18_hbgd_benefit_by_history.csv"
)


# ============================================================
# LOAD STEP 19
# ============================================================

step19_perf = read_csv_safe(
    "step19_hbgd_representation_transfer_performance.csv"
)

step19_params = read_csv_safe(
    "step19_hbgd_representation_transfer_parameters.csv"
)


# ============================================================
# LOAD STEP 20
# ============================================================

step20_summary = read_csv_safe(
    "step20_hbgd_transfer_statistical_summary.csv"
)

step20_patient = read_csv_safe(
    "step20_hbgd_transfer_patient_results.csv"
)

step20_pheno = read_csv_safe(
    "step20_hbgd_transfer_phenotype_results.csv"
)


# ============================================================
# EVIDENCE TABLE
# ============================================================

evidence_rows = []


def add_evidence(
    step,
    experiment,
    comparison,
    result,
    status,
    interpretation,
    limitation
):

    evidence_rows.append({

        "step":
            step,

        "experiment":
            experiment,

        "comparison":
            comparison,

        "result":
            result,

        "evidence_status":
            status,

        "scientific_interpretation":
            interpretation,

        "key_limitation":
            limitation
    })


# ------------------------------------------------------------
# STEP 8
# ------------------------------------------------------------

add_evidence(
    "Step 8",
    "Simple cross-cancer HBGD transfer",
    "Geometry vs transferred HBGD vs combined",
    "Simple distribution-normalized HBGD transfer did not "
    "outperform the target geometry baseline on the untouched "
    "test cohort.",
    "NOT SUPPORTED",
    "Simple normalization of source growth distributions "
    "was insufficient to establish predictive transfer.",
    "Source HBGD phenotype sample was small."
)


# ------------------------------------------------------------
# STEP 9
# ------------------------------------------------------------

add_evidence(
    "Step 9",
    "Cross-cancer HBGD growth phenotype comparison",
    "Source cancer vs LUMIERE GBM phenotype structure",
    "Descriptive comparison showed comparable structure across "
    "several HBGD-derived temporal features.",
    "PRELIMINARY",
    "The analysis suggested that HBGD-derived growth phenotypes "
    "have potentially transferable structure.",
    "The reported phenotype distance was descriptive and was "
    "not sufficient to establish statistical alignment."
)


# ------------------------------------------------------------
# STEP 10
# ------------------------------------------------------------

add_evidence(
    "Step 10",
    "Statistical phenotype alignment",
    "Observed phenotype distance vs permutation distribution",
    "Observed standardized phenotype distance = 2.659276; "
    "permutation p-value = 1.000000.",
    "NOT SUPPORTED",
    "Simple phenotype-distance alignment was not statistically "
    "supported.",
    "Only 25 source phenotype subjects and 67 target subjects."
)


# ------------------------------------------------------------
# STEP 11
# ------------------------------------------------------------

add_evidence(
    "Step 11",
    "Domain-adaptive HBGD transfer",
    "Source transfer vs domain-adapted transfer vs geometry",
    "Domain adaptation improved unadapted HBGD transfer but "
    "remained worse than the target geometry-only model.",
    "PRELIMINARY / NOT SUPERIOR",
    "Target-domain adaptation can reduce transfer mismatch, "
    "but the tested adaptation strategy did not establish "
    "superiority over geometry.",
    "Simple residual correction and small source cohort."
)


# ------------------------------------------------------------
# STEP 12
# ------------------------------------------------------------

add_evidence(
    "Step 12",
    "Incremental HBGD information ablation",
    "Geometry vs Geometry + HBGD information",
    "Geometry + HBGD rate showed approximately 0.97% improvement "
    "over geometry in the held-out test comparison; all-HBGD "
    "features showed approximately 0.63% improvement.",
    "PRELIMINARY POSITIVE",
    "HBGD-derived information appears to contribute a small "
    "incremental signal beyond geometry.",
    "Magnitude was modest and not statistically validated "
    "at this stage."
)


# ------------------------------------------------------------
# STEP 13
# ------------------------------------------------------------

add_evidence(
    "Step 13",
    "Patient-level HBGD robustness",
    "Geometry vs Geometry + HBGD rate",
    "Test MAE improved from 7.4912 mm to 7.1862 mm; "
    "8/13 patients improved; mean patient-level improvement "
    "was 0.7685 mm.",
    "PRELIMINARY POSITIVE",
    "The HBGD-rate representation showed a directional "
    "patient-level benefit beyond geometry.",
    "Bootstrap CI crossed zero and paired permutation p = "
    "0.274745; n = 13 patients."
)


# ------------------------------------------------------------
# STEP 16
# ------------------------------------------------------------

add_evidence(
    "Step 16",
    "Exact Step-12 phenotype stratification",
    "Geometry vs Geometry + HBGD rate",
    "HBGD-rate MAE improved within Growth, Stable and "
    "Shrinkage phenotypes.",
    "PRELIMINARY POSITIVE",
    "The incremental HBGD signal was observed across all "
    "three evaluated phenotypes.",
    "The analysis was based on the small 13-patient test cohort "
    "and did not by itself establish statistical significance."
)


# ------------------------------------------------------------
# STEP 17
# ------------------------------------------------------------

add_evidence(
    "Step 17",
    "Patient-level phenotype robustness",
    "Geometry vs Geometry + HBGD rate",
    "Stable phenotype showed the strongest patient-level "
    "effect; Growth and Shrinkage effects were smaller.",
    "PRELIMINARY",
    "HBGD benefit may vary according to growth phenotype, "
    "with stable lesions showing a potentially stronger "
    "incremental signal.",
    "Stable phenotype permutation p = 0.086191; other "
    "phenotypes were not significant."
)


# ------------------------------------------------------------
# STEP 18
# ------------------------------------------------------------

add_evidence(
    "Step 18",
    "HBGD benefit mechanism analysis",
    "Patient/history features vs HBGD benefit",
    "No strong explanatory factor was identified. Longer "
    "intervals showed a positive descriptive association, "
    "while rate variability and absolute rate showed "
    "negative descriptive associations.",
    "HYPOTHESIS-GENERATING",
    "The analysis provides candidate mechanisms but does not "
    "establish causality or statistically significant "
    "relationships.",
    "Only 32 patient-phenotype observations were available "
    "for correlation analysis."
)


# ------------------------------------------------------------
# STEP 19
# ------------------------------------------------------------

add_evidence(
    "Step 19",
    "Representation-level cross-cancer HBGD transfer",
    "GBM geometry vs Geometry + Transferred HBGD",
    "Geometry test MAE = 7.4565 mm; Geometry + Transferred "
    "HBGD test MAE = 7.3678 mm; improvement = 1.19%.",
    "PRELIMINARY POSITIVE",
    "A source-derived HBGD representation provided a small "
    "incremental improvement beyond target-domain geometry "
    "on the untouched test cohort.",
    "The direct transferred representation alone was worse "
    "than geometry; improvement was modest."
)


# ------------------------------------------------------------
# STEP 20
# ------------------------------------------------------------

add_evidence(
    "Step 20",
    "Patient-level statistical validation of Step 19",
    "Geometry vs Geometry + Transferred HBGD",
    "9/13 patients improved; mean improvement = 0.2062 mm; "
    "median = 0.1613 mm; effect size = 0.5062; bootstrap "
    "95% CI = [-0.0083, 0.4210]; permutation p = 0.097390; "
    "Wilcoxon p = 0.146484.",
    "PRELIMINARY POSITIVE / NOT SIGNIFICANT",
    "The transferred HBGD representation showed a consistent "
    "directional benefit with a moderate paired effect size, "
    "but statistical significance was not established.",
    "Only 13 held-out patients; bootstrap CI crossed zero "
    "and both paired tests had p > 0.05."
)


# ============================================================
# SAVE MASTER EVIDENCE TABLE
# ============================================================

evidence_df = pd.DataFrame(
    evidence_rows
)

evidence_file = path(
    "step21_cross_cancer_hbgd_evidence_table.csv"
)

evidence_df.to_csv(
    evidence_file,
    index=False
)


# ============================================================
# KEY NUMERICAL RESULTS TABLE
# ============================================================

numerical_rows = [

    {
        "step": "Step 12",
        "comparison": "Geometry + HBGD rate vs Geometry",
        "test_MAE_mm": 7.9392,
        "baseline_MAE_mm": 8.0170,
        "improvement_pct": 0.97,
        "statistical_validation": "Not performed"
    },

    {
        "step": "Step 13",
        "comparison": "Geometry + HBGD rate vs Geometry",
        "test_MAE_mm": 7.1862,
        "baseline_MAE_mm": 7.4912,
        "improvement_pct": 4.07,
        "statistical_validation":
            "Permutation p=0.274745"
    },

    {
        "step": "Step 19",
        "comparison":
            "Geometry + Transferred HBGD vs Geometry",
        "test_MAE_mm": 7.3678,
        "baseline_MAE_mm": 7.4565,
        "improvement_pct": 1.19,
        "statistical_validation":
            "Pending Step 20"
    },

    {
        "step": "Step 20",
        "comparison":
            "Geometry + Transferred HBGD vs Geometry",
        "test_MAE_mm": 7.3678,
        "baseline_MAE_mm": 7.4565,
        "improvement_pct": 1.19,
        "statistical_validation":
            "Permutation p=0.097390; Wilcoxon p=0.146484"
    }
]

numerical_df = pd.DataFrame(
    numerical_rows
)

numerical_file = path(
    "step21_cross_cancer_hbgd_numerical_evidence.csv"
)

numerical_df.to_csv(
    numerical_file,
    index=False
)


# ============================================================
# STATISTICAL SUMMARY
# ============================================================

statistical_rows = [

    {
        "analysis": "Step 13 patient-level HBGD rate",
        "n_patients": 13,
        "mean_improvement_mm": 0.7685,
        "effect_size": 0.3547,
        "bootstrap_CI":
            "[-0.2496, 1.9629]",
        "permutation_p": 0.274745,
        "interpretation":
            "Not statistically significant"
    },

    {
        "analysis": "Step 17 Growth phenotype",
        "n_groups": 11,
        "mean_improvement_mm": 0.1512,
        "effect_size": 0.3607,
        "bootstrap_CI":
            "[-0.0791, 0.3934]",
        "permutation_p": 0.264774,
        "interpretation":
            "Not statistically significant"
    },

    {
        "analysis": "Step 17 Stable phenotype",
        "n_groups": 9,
        "mean_improvement_mm": 0.1575,
        "effect_size": 0.7043,
        "bootstrap_CI":
            "[0.0249, 0.2993]",
        "permutation_p": 0.086191,
        "interpretation":
            "Promising but not statistically significant"
    },

    {
        "analysis": "Step 17 Shrinkage phenotype",
        "n_groups": 12,
        "mean_improvement_mm": 0.0475,
        "effect_size": 0.1102,
        "bootstrap_CI":
            "[-0.1886, 0.2798]",
        "permutation_p": 0.702130,
        "interpretation":
            "Not statistically significant"
    },

    {
        "analysis":
            "Step 20 cross-cancer transferred HBGD",
        "n_groups": 13,
        "mean_improvement_mm": 0.2062,
        "effect_size": 0.5062,
        "bootstrap_CI":
            "[-0.0083, 0.4210]",
        "permutation_p": 0.097390,
        "interpretation":
            "Directional positive effect; not significant"
    }
]

statistical_df = pd.DataFrame(
    statistical_rows
)

statistical_file = path(
    "step21_cross_cancer_hbgd_statistical_evidence.csv"
)

statistical_df.to_csv(
    statistical_file,
    index=False
)


# ============================================================
# FINAL CLAIM MATRIX
# ============================================================

claim_rows = [

    {
        "claim":
            "HBGD-derived growth information contains predictive signal",
        "evidence":
            "Steps 12-13",
        "support_level":
            "PRELIMINARY",
        "safe_statement":
            "HBGD-derived features provide a modest incremental "
            "predictive contribution beyond geometry in the tested "
            "LUMIERE cohort."
    },

    {
        "claim":
            "HBGD benefit is statistically proven",
        "evidence":
            "Steps 13, 17, 20",
        "support_level":
            "NOT SUPPORTED",
        "safe_statement":
            "Observed improvements did not consistently reach "
            "statistical significance."
    },

    {
        "claim":
            "HBGD phenotype structure is transferable across cancers",
        "evidence":
            "Steps 9-10",
        "support_level":
            "NOT ESTABLISHED",
        "safe_statement":
            "Descriptive phenotype similarities were observed, "
            "but statistical alignment was not established."
    },

    {
        "claim":
            "Simple growth-rate normalization transfers across cancers",
        "evidence":
            "Step 8",
        "support_level":
            "NOT SUPPORTED",
        "safe_statement":
            "Simple source-to-target rate normalization did not "
            "outperform the target geometry baseline."
    },

    {
        "claim":
            "Domain adaptation solves cross-cancer mismatch",
        "evidence":
            "Step 11",
        "support_level":
            "NOT SUPPORTED",
        "safe_statement":
            "The tested domain-adaptation strategy improved direct "
            "transfer but did not outperform geometry."
    },

    {
        "claim":
            "Representation-level transfer provides complementary signal",
        "evidence":
            "Steps 19-20",
        "support_level":
            "PRELIMINARY",
        "safe_statement":
            "A source-derived HBGD representation produced a "
            "directional improvement beyond geometry on the "
            "held-out LUMIERE cohort."
    },

    {
        "claim":
            "Cross-cancer HBGD transfer is clinically validated",
        "evidence":
            "Steps 19-20",
        "support_level":
            "NOT SUPPORTED",
        "safe_statement":
            "The current evidence is exploratory and requires "
            "larger independent longitudinal cohorts."
    }
]

claim_df = pd.DataFrame(
    claim_rows
)

claim_file = path(
    "step21_cross_cancer_hbgd_claim_matrix.csv"
)

claim_df.to_csv(
    claim_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

report_file = path(
    "STEP21_CROSS_CANCER_HBGD_EVIDENCE_SYNTHESIS_REPORT.md"
)

with open(
    report_file,
    "w"
) as f:

    f.write(
        "# Step 21 — Cross-Cancer HBGD Evidence Synthesis\n\n"
    )

    f.write(
        "## 1. Purpose\n\n"
    )

    f.write(
        "This report consolidates the cross-cancer HBGD experiments "
        "performed in Steps 8–20. No additional model fitting is "
        "performed in Step 21. The purpose is to distinguish "
        "demonstrated findings from preliminary observations and "
        "unsupported claims.\n\n"
    )

    f.write(
        "## 2. Validated experimental progression\n\n"
    )

    f.write(
        "### Step 8 — Simple transfer\n\n"
    )

    f.write(
        "Simple source-to-target HBGD distribution normalization "
        "did not establish predictive transfer beyond the target "
        "geometry baseline.\n\n"
    )

    f.write(
        "### Step 9 — Phenotype comparison\n\n"
    )

    f.write(
        "Descriptive HBGD-derived phenotype comparison showed "
        "similarities between the source cancer and LUMIERE GBM "
        "across several temporal features. This was exploratory "
        "and did not establish statistical alignment.\n\n"
    )

    f.write(
        "### Step 10 — Statistical alignment\n\n"
    )

    f.write(
        "The observed standardized phenotype distance was "
        "2.659276 with a permutation p-value of 1.000000. "
        "Therefore, the simple phenotype-distance alignment "
        "approach did not establish cross-cancer statistical "
        "alignment.\n\n"
    )

    f.write(
        "### Step 11 — Domain adaptation\n\n"
    )

    f.write(
        "Domain adaptation reduced the error of direct HBGD "
        "transfer but did not outperform the target-domain "
        "geometry baseline. This indicates that simple residual "
        "correction is insufficient to solve the domain mismatch.\n\n"
    )

    f.write(
        "### Steps 12–18 — Incremental HBGD signal\n\n"
    )

    f.write(
        "The ablation and patient-level analyses consistently "
        "suggested a modest incremental contribution from "
        "HBGD-derived information beyond geometry. The signal "
        "was observed across multiple phenotypes, with the "
        "stable phenotype showing the strongest preliminary "
        "effect in Step 17. However, the small number of "
        "held-out patients prevented firm statistical conclusions.\n\n"
    )

    f.write(
        "### Step 19 — Representation-level transfer\n\n"
    )

    f.write(
        "The corrected representation-level transfer experiment "
        "used the validated source HBGD state-parameter artifact "
        "containing 49 source intervals from 12 patients. "
        "The exact 13-patient LUMIERE test cohort was preserved.\n\n"
    )

    f.write(
        "On the held-out test cohort, geometry-only prediction "
        "achieved an MAE of 7.4565 mm, while geometry combined "
        "with the transferred HBGD representation achieved "
        "7.3678 mm. This corresponds to a 1.19% reduction in MAE.\n\n"
    )

    f.write(
        "The transferred HBGD representation alone did not "
        "outperform geometry, demonstrating that its observed "
        "value is complementary rather than independently "
        "sufficient.\n\n"
    )

    f.write(
        "### Step 20 — Statistical validation\n\n"
    )

    f.write(
        "At the patient level, 9 of 13 held-out patients improved "
        "with transferred HBGD information and 4 worsened. "
        "Mean improvement was 0.2062 mm and median improvement "
        "was 0.1613 mm. The paired effect size was 0.5062.\n\n"
    )

    f.write(
        "However, the 95% bootstrap confidence interval was "
        "[-0.0083, 0.4210] mm, the paired permutation p-value "
        "was 0.097390, and the Wilcoxon p-value was 0.146484. "
        "Therefore, the observed improvement is not statistically "
        "significant at the conventional 0.05 threshold.\n\n"
    )

    f.write(
        "## 3. Overall evidence assessment\n\n"
    )

    f.write(
        "| Evidence area | Assessment |\n"
    )

    f.write(
        "|---|---|\n"
    )

    f.write(
        "| HBGD mathematical formulation | Established in source pipeline |\n"
    )

    f.write(
        "| HBGD-derived predictive signal | Preliminary |\n"
    )

    f.write(
        "| Simple cross-cancer rate transfer | Not supported |\n"
    )

    f.write(
        "| Simple phenotype alignment | Not supported statistically |\n"
    )

    f.write(
        "| Domain-adaptive transfer | Not superior to geometry |\n"
    )

    f.write(
        "| HBGD incremental information | Preliminary positive |\n"
    )

    f.write(
        "| Representation-level cross-cancer transfer | Preliminary positive |\n"
    )

    f.write(
        "| Statistical proof of cross-cancer transfer | Not established |\n\n"
    )

    f.write(
        "## 4. Strongest current result\n\n"
    )

    f.write(
        "The strongest current cross-cancer observation is that "
        "a source-derived HBGD growth representation, when "
        "combined with target-domain tumor geometry, produced "
        "a directional improvement on the untouched LUMIERE "
        "test cohort. The effect was observed in 9 of 13 "
        "patients and had a moderate paired effect size of "
        "0.5062.\n\n"
    )

    f.write(
        "The improvement remains preliminary because the "
        "bootstrap confidence interval crosses zero and both "
        "paired statistical tests have p-values above 0.05.\n\n"
    )

    f.write(
        "## 5. Recommended scientific claim\n\n"
    )

    f.write(
        "> A source-derived HBGD growth representation demonstrated "
        "preliminary cross-cancer transferability by providing a "
        "small incremental predictive contribution beyond target-"
        "domain tumor geometry in a held-out LUMIERE glioblastoma "
        "cohort. The effect was directionally consistent at the "
        "patient level but did not reach statistical significance, "
        "indicating that larger independent longitudinal cohorts "
        "are required for confirmation.\n\n"
    )

    f.write(
        "## 6. Claims that should NOT be made\n\n"
    )

    f.write(
        "- HBGD transfer is statistically proven.\n"
    )

    f.write(
        "- HBGD transfer is clinically validated.\n"
    )

    f.write(
        "- HBGD universally transfers between cancer types.\n"
    )

    f.write(
        "- Source growth rates can be directly copied to GBM.\n"
    )

    f.write(
        "- HBGD outperforms tumor geometry.\n"
    )

    f.write(
        "- The 1.19% improvement is statistically significant.\n\n"
    )

    f.write(
        "## 7. Main limitations\n\n"
    )

    f.write(
        "1. The source HBGD cohort contains only 12 patients and "
        "49 validated longitudinal intervals.\n"
    )

    f.write(
        "2. The LUMIERE held-out test cohort contains only 13 patients.\n"
    )

    f.write(
        "3. The cross-cancer improvement is modest.\n"
    )

    f.write(
        "4. Statistical confidence intervals include zero.\n"
    )

    f.write(
        "5. The source and target cancers differ biologically "
        "and imaging-wise.\n"
    )

    f.write(
        "6. Larger independent longitudinal datasets are needed "
        "for replication.\n\n"
    )

    f.write(
        "## 8. Final conclusion\n\n"
    )

    f.write(
        "The accumulated experiments do not establish HBGD as a "
        "statistically validated cross-cancer predictor. However, "
        "they provide a coherent preliminary signal that growth-"
        "dynamics information derived from HBGD may contain "
        "transferable information that complements target-domain "
        "tumor geometry. The representation-level experiment "
        "provides the clearest current evidence for this hypothesis, "
        "while the statistical validation indicates that the effect "
        "remains uncertain because of the limited sample size.\n"
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 78)
print("STEP 21 COMPLETE")
print("=" * 78)

print("\nMASTER CONCLUSION")
print(
    "Cross-cancer HBGD transfer: PRELIMINARY, "
    "NOT STATISTICALLY ESTABLISHED"
)

print("\nSTRONGEST CURRENT RESULT")
print(
    "Geometry MAE: 7.4565 mm"
)

print(
    "Geometry + transferred HBGD MAE: 7.3678 mm"
)

print(
    "Improvement: 1.19%"
)

print(
    "Patient-level improvement:",
    "9/13 patients"
)

print(
    "Mean patient improvement: 0.2062 mm"
)

print(
    "Effect size: 0.5062"
)

print(
    "Bootstrap 95% CI: [-0.0083, 0.4210]"
)

print(
    "Permutation p: 0.097390"
)

print(
    "Wilcoxon p: 0.146484"
)

print("\nOUTPUT FILES")

print(
    evidence_file
)

print(
    numerical_file
)

print(
    statistical_file
)

print(
    claim_file
)

print(
    report_file
)

print("\n" + "=" * 78)