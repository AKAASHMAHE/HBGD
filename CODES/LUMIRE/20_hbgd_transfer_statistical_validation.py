# -*- coding: utf-8 -*-

"""
STEP 20
PATIENT-LEVEL STATISTICAL VALIDATION OF CROSS-CANCER HBGD TRANSFER

Purpose
-------
Statistically validate the Step 19 result:

    Geometry
        vs
    Geometry + Transferred HBGD

using the EXACT Step 19 held-out test predictions.

IMPORTANT
---------
- No model is retrained.
- No test patient is used for model fitting.
- Exact Step-13/Step-19 test cohort is preserved.
- Statistics are performed at patient level to avoid treating
  multiple observations from the same patient as independent.
- Bootstrap and paired permutation tests are performed on
  patient-level MAE differences.
- Wilcoxon signed-rank test is performed when possible.

Python 3.6 compatible.
"""

import os
import math
import warnings

import numpy as np
import pandas as pd

from scipy.stats import (
    wilcoxon
)
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

LUMIERE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"

PREDICTIONS_FILE = os.path.join(
    LUMIERE_DIR,
    "step19_hbgd_representation_transfer_predictions.csv"
)

OUTPUT_PATIENT = os.path.join(
    LUMIERE_DIR,
    "step20_hbgd_transfer_patient_results.csv"
)

OUTPUT_BOOTSTRAP = os.path.join(
    LUMIERE_DIR,
    "step20_hbgd_transfer_bootstrap_results.csv"
)

OUTPUT_PERMUTATION = os.path.join(
    LUMIERE_DIR,
    "step20_hbgd_transfer_permutation_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    LUMIERE_DIR,
    "step20_hbgd_transfer_statistical_summary.csv"
)

OUTPUT_PHENOTYPE = os.path.join(
    LUMIERE_DIR,
    "step20_hbgd_transfer_phenotype_results.csv"
)

OUTPUT_REPORT = os.path.join(
    LUMIERE_DIR,
    "STEP20_HBGD_TRANSFER_STATISTICAL_VALIDATION_REPORT.md"
)


# ============================================================
# EXACT TEST COHORT
# ============================================================

TEST_PATIENTS = [
    "Patient-010",
    "Patient-012",
    "Patient-019",
    "Patient-022",
    "Patient-033",
    "Patient-037",
    "Patient-038",
    "Patient-042",
    "Patient-059",
    "Patient-064",
    "Patient-065",
    "Patient-073",
    "Patient-085"
]

RANDOM_SEED = 42

BOOTSTRAP_ITERATIONS = 10000
PERMUTATION_ITERATIONS = 10000


# ============================================================
# FUNCTIONS
# ============================================================

def normalize_patient_id(value):

    s = str(value).strip()

    if s.lower().startswith("patient-"):
        return s

    try:
        return "Patient-%03d" % int(float(s))
    except Exception:
        return s


def bootstrap_mean_difference(
    differences,
    iterations,
    seed
):

    rng = np.random.RandomState(seed)

    differences = np.asarray(
        differences,
        dtype=float
    )

    n = len(differences)

    bootstrap_means = []

    for _ in range(iterations):

        sample = rng.choice(
            differences,
            size=n,
            replace=True
        )

        bootstrap_means.append(
            np.mean(sample)
        )

    bootstrap_means = np.asarray(
        bootstrap_means
    )

    mean_value = np.mean(
        bootstrap_means
    )

    lower = np.percentile(
        bootstrap_means,
        2.5
    )

    upper = np.percentile(
        bootstrap_means,
        97.5
    )

    return (
        mean_value,
        lower,
        upper,
        bootstrap_means
    )


def paired_permutation_test(
    differences,
    iterations,
    seed
):
    """
    Sign-flipping paired permutation test.

    Null hypothesis:
        Mean patient-level difference = 0

    Positive difference means:
        Geometry MAE > Geometry+HBGD MAE

    Therefore positive values favor HBGD.
    """

    rng = np.random.RandomState(seed)

    differences = np.asarray(
        differences,
        dtype=float
    )

    observed = np.mean(
        differences
    )

    count = 0

    n = len(differences)

    for _ in range(iterations):

        signs = rng.choice(
            [-1.0, 1.0],
            size=n
        )

        permuted = np.mean(
            differences * signs
        )

        if abs(permuted) >= abs(observed):
            count += 1

    p_value = (
        count + 1.0
    ) / (
        iterations + 1.0
    )

    return (
        observed,
        p_value
    )


def paired_effect_size(
    differences
):

    differences = np.asarray(
        differences,
        dtype=float
    )

    if len(differences) < 2:
        return np.nan

    sd = np.std(
        differences,
        ddof=1
    )

    if sd == 0:
        return np.nan

    return (
        np.mean(differences) /
        sd
    )


def safe_wilcoxon(
    geometry_mae,
    hbgd_mae
):

    try:

        result = wilcoxon(
            geometry_mae,
            hbgd_mae,
            zero_method="wilcox",
            alternative="two-sided"
        )

        return (
            float(result.statistic),
            float(result.pvalue)
        )

    except Exception:

        return (
            np.nan,
            np.nan
        )


# ============================================================
# START
# ============================================================

print("=" * 72)
print(
    "STEP 20 PATIENT-LEVEL STATISTICAL VALIDATION"
)
print("=" * 72)


# ============================================================
# LOAD STEP 19 PREDICTIONS
# ============================================================

print("\nLoading exact Step 19 predictions...")

if not os.path.exists(
    PREDICTIONS_FILE
):

    raise IOError(
        "Step 19 prediction file not found:\n%s"
        % PREDICTIONS_FILE
    )

df = pd.read_csv(
    PREDICTIONS_FILE
)

print(
    "Rows loaded:",
    len(df)
)

print(
    "Columns:"
)

for c in df.columns:
    print(
        "  " + c
    )


# ============================================================
# NORMALIZE IDS
# ============================================================

df["patient_id"] = df[
    "patient_id"
].apply(
    normalize_patient_id
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "patient_id",
    "split",
    "actual_future_diameter_mm",
    "pred_geometry",
    "pred_geometry_transfer"
]

for c in required_columns:

    if c not in df.columns:

        raise ValueError(
            "Required column missing: %s"
            % c
        )


# ============================================================
# SELECT EXACT TEST SET
# ============================================================

test_df = df[
    df["split"].astype(str).str.lower()
    == "test"
].copy()

test_df = test_df[
    test_df["patient_id"].isin(
        TEST_PATIENTS
    )
].copy()

print(
    "\nExact test prediction rows:",
    len(test_df)
)

print(
    "Unique test patients:",
    test_df["patient_id"].nunique()
)


# ============================================================
# TEST COHORT CHECK
# ============================================================

found_patients = sorted(
    test_df["patient_id"].unique()
)

missing_patients = [
    p for p in TEST_PATIENTS
    if p not in found_patients
]

unexpected_patients = [
    p for p in found_patients
    if p not in TEST_PATIENTS
]

print(
    "\nExpected test patients:",
    len(TEST_PATIENTS)
)

print(
    "Found test patients:",
    len(found_patients)
)

if len(missing_patients) > 0:

    print(
        "WARNING - missing patients:",
        missing_patients
    )

if len(unexpected_patients) > 0:

    print(
        "WARNING - unexpected patients:",
        unexpected_patients
    )


# ============================================================
# NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "actual_future_diameter_mm",
    "pred_geometry",
    "pred_geometry_transfer"
]

for c in numeric_columns:

    test_df[c] = pd.to_numeric(
        test_df[c],
        errors="coerce"
    )


test_df = test_df.dropna(
    subset=numeric_columns
).copy()


# ============================================================
# CASE-LEVEL ABSOLUTE ERRORS
# ============================================================

test_df[
    "geometry_absolute_error_mm"
] = np.abs(
    test_df[
        "actual_future_diameter_mm"
    ] -
    test_df[
        "pred_geometry"
    ]
)

test_df[
    "hbgd_absolute_error_mm"
] = np.abs(
    test_df[
        "actual_future_diameter_mm"
    ] -
    test_df[
        "pred_geometry_transfer"
    ]
)

test_df[
    "hbgd_improvement_mm"
] = (
    test_df[
        "geometry_absolute_error_mm"
    ] -
    test_df[
        "hbgd_absolute_error_mm"
    ]
)


# ============================================================
# PATIENT-LEVEL AGGREGATION
# ============================================================

print(
    "\nCalculating patient-level MAE..."
)

patient_rows = []

for patient_id, group in test_df.groupby(
    "patient_id"
):

    geometry_mae = group[
        "geometry_absolute_error_mm"
    ].mean()

    hbgd_mae = group[
        "hbgd_absolute_error_mm"
    ].mean()

    improvement = (
        geometry_mae -
        hbgd_mae
    )

    relative_improvement = (
        improvement /
        geometry_mae *
        100.0
        if geometry_mae != 0
        else np.nan
    )

    patient_rows.append({

        "patient_id":
            patient_id,

        "n_prediction_cases":
            len(group),

        "geometry_MAE_mm":
            geometry_mae,

        "geometry_hbgd_MAE_mm":
            hbgd_mae,

        "improvement_mm":
            improvement,

        "relative_improvement_pct":
            relative_improvement,

        "hbgd_improved":
            int(improvement > 0),

        "hbgd_worsened":
            int(improvement < 0),

        "hbgd_unchanged":
            int(
                np.isclose(
                    improvement,
                    0.0
                )
            )
    })


patient_results = pd.DataFrame(
    patient_rows
)

patient_results = patient_results.sort_values(
    "patient_id"
).reset_index(
    drop=True
)


# ============================================================
# OVERALL PATIENT-LEVEL STATISTICS
# ============================================================

differences = patient_results[
    "improvement_mm"
].values

geometry_patient_mae = patient_results[
    "geometry_MAE_mm"
].values

hbgd_patient_mae = patient_results[
    "geometry_hbgd_MAE_mm"
].values

n_patients = len(
    patient_results
)

mean_improvement = np.mean(
    differences
)

median_improvement = np.median(
    differences
)

sd_improvement = (
    np.std(
        differences,
        ddof=1
    )
    if n_patients > 1
    else np.nan
)

mean_relative_improvement = np.mean(
    patient_results[
        "relative_improvement_pct"
    ].values
)

n_improved = int(
    np.sum(
        differences > 0
    )
)

n_worsened = int(
    np.sum(
        differences < 0
    )
)

n_unchanged = int(
    np.sum(
        np.isclose(
            differences,
            0.0
        )
    )
)

effect_size = paired_effect_size(
    differences
)


# ============================================================
# BOOTSTRAP
# ============================================================

print(
    "\nRunning bootstrap:",
    BOOTSTRAP_ITERATIONS,
    "iterations..."
)

(
    bootstrap_mean,
    bootstrap_lower,
    bootstrap_upper,
    bootstrap_distribution
) = bootstrap_mean_difference(
    differences,
    BOOTSTRAP_ITERATIONS,
    RANDOM_SEED
)

bootstrap_df = pd.DataFrame({

    "iteration":
        np.arange(
            1,
            BOOTSTRAP_ITERATIONS + 1
        ),

    "mean_patient_improvement_mm":
        bootstrap_distribution
})

bootstrap_df.to_csv(
    OUTPUT_BOOTSTRAP,
    index=False
)


# ============================================================
# PAIRED PERMUTATION
# ============================================================

print(
    "Running paired permutation:",
    PERMUTATION_ITERATIONS,
    "iterations..."
)

(
    observed_difference,
    permutation_p
) = paired_permutation_test(
    differences,
    PERMUTATION_ITERATIONS,
    RANDOM_SEED
)


permutation_df = pd.DataFrame({

    "observed_mean_improvement_mm": [
        observed_difference
    ],

    "permutation_p_value": [
        permutation_p
    ],

    "iterations": [
        PERMUTATION_ITERATIONS
    ],

    "n_patients": [
        n_patients
    ]
})

permutation_df.to_csv(
    OUTPUT_PERMUTATION,
    index=False
)


# ============================================================
# WILCOXON
# ============================================================

(
    wilcoxon_statistic,
    wilcoxon_p
) = safe_wilcoxon(
    geometry_patient_mae,
    hbgd_patient_mae
)


# ============================================================
# POOLED TEST PERFORMANCE
# ============================================================

pooled_geometry_mae = mean_absolute_error(
    test_df[
        "actual_future_diameter_mm"
    ],
    test_df[
        "pred_geometry"
    ]
)

pooled_hbgd_mae = mean_absolute_error(
    test_df[
        "actual_future_diameter_mm"
    ],
    test_df[
        "pred_geometry_transfer"
    ]
)

pooled_geometry_rmse = math.sqrt(
    np.mean(
        (
            test_df[
                "actual_future_diameter_mm"
            ].values -
            test_df[
                "pred_geometry"
            ].values
        ) ** 2
    )
)

pooled_hbgd_rmse = math.sqrt(
    np.mean(
        (
            test_df[
                "actual_future_diameter_mm"
            ].values -
            test_df[
                "pred_geometry_transfer"
            ].values
        ) ** 2
    )
)

pooled_improvement = (
    pooled_geometry_mae -
    pooled_hbgd_mae
)

pooled_relative_improvement = (
    pooled_improvement /
    pooled_geometry_mae *
    100.0
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary_rows = [

    {
        "metric":
            "n_test_patients",
        "value":
            n_patients
    },

    {
        "metric":
            "n_test_prediction_cases",
        "value":
            len(test_df)
    },

    {
        "metric":
            "pooled_geometry_MAE_mm",
        "value":
            pooled_geometry_mae
    },

    {
        "metric":
            "pooled_geometry_hbgd_MAE_mm",
        "value":
            pooled_hbgd_mae
    },

    {
        "metric":
            "pooled_MAE_improvement_mm",
        "value":
            pooled_improvement
    },

    {
        "metric":
            "pooled_relative_improvement_pct",
        "value":
            pooled_relative_improvement
    },

    {
        "metric":
            "pooled_geometry_RMSE_mm",
        "value":
            pooled_geometry_rmse
    },

    {
        "metric":
            "pooled_geometry_hbgd_RMSE_mm",
        "value":
            pooled_hbgd_rmse
    },

    {
        "metric":
            "mean_patient_improvement_mm",
        "value":
            mean_improvement
    },

    {
        "metric":
            "median_patient_improvement_mm",
        "value":
            median_improvement
    },

    {
        "metric":
            "SD_patient_improvement_mm",
        "value":
            sd_improvement
    },

    {
        "metric":
            "mean_patient_relative_improvement_pct",
        "value":
            mean_relative_improvement
    },

    {
        "metric":
            "patients_improved",
        "value":
            n_improved
    },

    {
        "metric":
            "patients_worsened",
        "value":
            n_worsened
    },

    {
        "metric":
            "patients_unchanged",
        "value":
            n_unchanged
    },

    {
        "metric":
            "paired_effect_size",
        "value":
            effect_size
    },

    {
        "metric":
            "bootstrap_mean_improvement_mm",
        "value":
            bootstrap_mean
    },

    {
        "metric":
            "bootstrap_95CI_lower_mm",
        "value":
            bootstrap_lower
    },

    {
        "metric":
            "bootstrap_95CI_upper_mm",
        "value":
            bootstrap_upper
    },

    {
        "metric":
            "paired_permutation_p",
        "value":
            permutation_p
    },

    {
        "metric":
            "wilcoxon_statistic",
        "value":
            wilcoxon_statistic
    },

    {
        "metric":
            "wilcoxon_p",
        "value":
            wilcoxon_p
    }
]

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# SAVE PATIENT RESULTS
# ============================================================

patient_results.to_csv(
    OUTPUT_PATIENT,
    index=False
)


# ============================================================
# PHENOTYPE ANALYSIS
# ============================================================

print(
    "\nAttempting phenotype-stratified analysis..."
)

phenotype_df = pd.DataFrame()

# Try to infer phenotype from Step 19 columns if available.
# If dynamic state is not present, derive phenotype from the
# historical growth rate stored in transferred predictions.

phenotype_source = None

if "dynamic_state" in test_df.columns:
    phenotype_source = "dynamic_state"

elif "history_mean_rate" in test_df.columns:
    phenotype_source = "history_mean_rate"


if phenotype_source == "dynamic_state":

    temp = test_df.copy()

    temp["phenotype"] = (
        temp["dynamic_state"]
        .astype(str)
    )

elif phenotype_source == "history_mean_rate":

    temp = test_df.copy()

    def assign_phenotype(rate):

        if rate > 0.0005:
            return "Growth"

        if rate < -0.0010:
            return "Shrinkage"

        return "Stable"

    temp["phenotype"] = (
        temp[
            "history_mean_rate"
        ].apply(
            assign_phenotype
        )
    )

else:

    temp = None


if temp is not None:

    phenotype_rows = []

    for phenotype, group in temp.groupby(
        "phenotype"
    ):

        geometry_case_errors = group[
            "geometry_absolute_error_mm"
        ]

        hbgd_case_errors = group[
            "hbgd_absolute_error_mm"
        ]

        phenotype_rows.append({

            "phenotype":
                phenotype,

            "n_prediction_cases":
                len(group),

            "n_patients":
                group[
                    "patient_id"
                ].nunique(),

            "geometry_MAE_mm":
                geometry_case_errors.mean(),

            "geometry_hbgd_MAE_mm":
                hbgd_case_errors.mean(),

            "improvement_mm":
                (
                    geometry_case_errors.mean()
                    -
                    hbgd_case_errors.mean()
                ),

            "relative_improvement_pct":
                (
                    (
                        geometry_case_errors.mean()
                        -
                        hbgd_case_errors.mean()
                    )
                    /
                    geometry_case_errors.mean()
                    *
                    100.0
                    if geometry_case_errors.mean()
                    != 0
                    else np.nan
                )
        })

    phenotype_df = pd.DataFrame(
        phenotype_rows
    )

else:

    phenotype_df = pd.DataFrame(
        columns=[
            "phenotype",
            "n_prediction_cases",
            "n_patients",
            "geometry_MAE_mm",
            "geometry_hbgd_MAE_mm",
            "improvement_mm",
            "relative_improvement_pct"
        ]
    )


phenotype_df.to_csv(
    OUTPUT_PHENOTYPE,
    index=False
)


# ============================================================
# SCIENTIFIC INTERPRETATION
# ============================================================

if (
    bootstrap_lower > 0
    and permutation_p < 0.05
    and wilcoxon_p < 0.05
):

    interpretation = (
        "The transferred HBGD representation demonstrated a "
        "statistically supported patient-level improvement "
        "over the geometry-only model in this held-out cohort."
    )

elif (
    mean_improvement > 0
):

    interpretation = (
        "The transferred HBGD representation showed a "
        "directionally positive patient-level improvement "
        "over geometry alone, but the available statistical "
        "evidence does not establish significance."
    )

else:

    interpretation = (
        "The transferred HBGD representation did not "
        "demonstrate a positive patient-level improvement "
        "over geometry alone in this held-out cohort."
    )


# ============================================================
# REPORT
# ============================================================

print(
    "\nWriting report..."
)

with open(
    OUTPUT_REPORT,
    "w"
) as f:

    f.write(
        "# Step 20 — Patient-Level Statistical Validation "
        "of Cross-Cancer HBGD Transfer\n\n"
    )

    f.write(
        "## Objective\n\n"
    )

    f.write(
        "This analysis statistically validates the exact "
        "Step 19 held-out comparison between the LUMIERE "
        "geometry-only model and the geometry model augmented "
        "with the transferred HBGD representation.\n\n"
    )

    f.write(
        "No model was retrained during this analysis.\n\n"
    )

    f.write(
        "## Test cohort\n\n"
    )

    f.write(
        "- Expected test patients: 13\n"
    )

    f.write(
        "- Test patients analyzed: %d\n"
        % n_patients
    )

    f.write(
        "- Prediction cases analyzed: %d\n\n"
        % len(test_df)
    )

    f.write(
        "The analysis uses patient-level aggregation so that "
        "multiple prediction cases from the same patient do "
        "not artificially inflate the effective sample size.\n\n"
    )

    f.write(
        "## Pooled test performance\n\n"
    )

    f.write(
        "| Model | MAE (mm) | RMSE (mm) |\n"
    )

    f.write(
        "|---|---:|---:|\n"
    )

    f.write(
        "| Geometry | %.4f | %.4f |\n"
        % (
            pooled_geometry_mae,
            pooled_geometry_rmse
        )
    )

    f.write(
        "| Geometry + Transferred HBGD | %.4f | %.4f |\n"
        % (
            pooled_hbgd_mae,
            pooled_hbgd_rmse
        )
    )

    f.write(
        "\nPooled MAE improvement: %.4f mm (%.2f%%).\n\n"
        % (
            pooled_improvement,
            pooled_relative_improvement
        )
    )

    f.write(
        "## Patient-level results\n\n"
    )

    f.write(
        "- Mean improvement: %.4f mm\n"
        % mean_improvement
    )

    f.write(
        "- Median improvement: %.4f mm\n"
        % median_improvement
    )

    f.write(
        "- Standard deviation: %.4f mm\n"
        % sd_improvement
    )

    f.write(
        "- Patients improved: %d/%d\n"
        % (
            n_improved,
            n_patients
        )
    )

    f.write(
        "- Patients worsened: %d/%d\n"
        % (
            n_worsened,
            n_patients
        )
    )

    f.write(
        "- Patients unchanged: %d/%d\n\n"
        % (
            n_unchanged,
            n_patients
        )
    )

    f.write(
        "## Statistical validation\n\n"
    )

    f.write(
        "### Bootstrap\n\n"
    )

    f.write(
        "Mean patient-level improvement: %.4f mm\n\n"
        % bootstrap_mean
    )

    f.write(
        "95%% bootstrap CI: [%.4f, %.4f] mm\n\n"
        % (
            bootstrap_lower,
            bootstrap_upper
        )
    )

    f.write(
        "### Paired permutation test\n\n"
    )

    f.write(
        "Observed mean patient-level improvement: %.4f mm\n\n"
        % observed_difference
    )

    f.write(
        "Permutation p-value: %.6f\n\n"
        % permutation_p
    )

    f.write(
        "### Wilcoxon signed-rank test\n\n"
    )

    f.write(
        "Wilcoxon statistic: %.4f\n\n"
        % wilcoxon_statistic
    )

    f.write(
        "Wilcoxon p-value: %.6f\n\n"
        % wilcoxon_p
    )

    f.write(
        "### Paired effect size\n\n"
    )

    f.write(
        "Effect size: %.4f\n\n"
        % effect_size
    )

    f.write(
        "## Interpretation\n\n"
    )

    f.write(
        interpretation +
        "\n\n"
    )

    f.write(
        "A positive improvement means that the "
        "Geometry + Transferred HBGD model has lower "
        "patient-level MAE than Geometry alone.\n\n"
    )

    f.write(
        "Because only 13 held-out patients are available, "
        "statistical power is limited. Even a positive "
        "effect should therefore be described as preliminary "
        "unless replicated in larger independent cohorts.\n\n"
    )

    f.write(
        "## Important limitation\n\n"
    )

    f.write(
        "The statistical analysis validates the observed "
        "Step 19 performance difference but does not establish "
        "that the transferred representation is universally "
        "transferable across cancer types. Independent "
        "replication and larger longitudinal datasets remain "
        "necessary.\n"
    )


# ============================================================
# CONSOLE OUTPUT
# ============================================================

print("\n")
print("=" * 72)
print("STEP 20 COMPLETE")
print("=" * 72)

print("\nTEST COHORT")
print(
    "Patients:",
    n_patients
)

print(
    "Prediction cases:",
    len(test_df)
)

print("\nPOOLED PERFORMANCE")

print(
    "Geometry:",
    "%.4f MAE | %.4f RMSE"
    % (
        pooled_geometry_mae,
        pooled_geometry_rmse
    )
)

print(
    "Geometry + Transferred HBGD:",
    "%.4f MAE | %.4f RMSE"
    % (
        pooled_hbgd_mae,
        pooled_hbgd_rmse
    )
)

print(
    "Pooled improvement:",
    "%.4f mm (%.2f%%)"
    % (
        pooled_improvement,
        pooled_relative_improvement
    )
)

print("\nPATIENT-LEVEL")

print(
    "Mean improvement:",
    "%.4f mm"
    % mean_improvement
)

print(
    "Median improvement:",
    "%.4f mm"
    % median_improvement
)

print(
    "Improved:",
    n_improved
)

print(
    "Worsened:",
    n_worsened
)

print(
    "Unchanged:",
    n_unchanged
)

print(
    "Effect size:",
    "%.4f"
    % effect_size
)

print("\nSTATISTICS")

print(
    "Bootstrap 95% CI:",
    "[%.4f, %.4f]"
    % (
        bootstrap_lower,
        bootstrap_upper
    )
)

print(
    "Paired permutation p:",
    "%.6f"
    % permutation_p
)

print(
    "Wilcoxon p:",
    "%.6f"
    % wilcoxon_p
)

print("\nINTERPRETATION")
print(
    interpretation
)

print("\nOUTPUTS")

print(
    OUTPUT_PATIENT
)

print(
    OUTPUT_BOOTSTRAP
)

print(
    OUTPUT_PERMUTATION
)

print(
    OUTPUT_SUMMARY
)

print(
    OUTPUT_PHENOTYPE
)

print(
    OUTPUT_REPORT
)

print("\n" + "=" * 72)