# -*- coding: utf-8 -*-

"""
STEP 17 - PATIENT-LEVEL PHENOTYPE ROBUSTNESS

Purpose
-------
Evaluate whether HBGD-derived growth-rate information provides
consistent patient-level benefit within Growth, Stable and
Shrinkage phenotypes.

Uses:
    Step 16 exact Step-12 test-cohort predictions.

No model retraining.
No split reconstruction.
No phenotype leakage.

Python 3.6 compatible.
"""

from __future__ import print_function

import os
import csv
import math
import random

import numpy as np


# ============================================================
# PATHS
# ============================================================

TARGET_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"

INPUT_FILE = os.path.join(
    TARGET_DIR,
    "step16_exact_step12_phenotype_predictions.csv"
)

OUTPUT_PATIENT = os.path.join(
    TARGET_DIR,
    "step17_patient_level_phenotype_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    TARGET_DIR,
    "step17_phenotype_robustness_summary.csv"
)

OUTPUT_BOOTSTRAP = os.path.join(
    TARGET_DIR,
    "step17_phenotype_bootstrap_results.csv"
)

OUTPUT_PERMUTATION = os.path.join(
    TARGET_DIR,
    "step17_phenotype_permutation_results.csv"
)

OUTPUT_REPORT = os.path.join(
    TARGET_DIR,
    "STEP17_PATIENT_LEVEL_PHENOTYPE_ROBUSTNESS_REPORT.md"
)


# ============================================================
# HELPERS
# ============================================================

def read_csv(path):

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def write_csv(path, rows, fields):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def f(x):

    try:
        return float(x)
    except Exception:
        return float("nan")


def finite(x):

    try:
        return math.isfinite(float(x))
    except Exception:
        return False


# ============================================================
# PATIENT-LEVEL MAE
# ============================================================

def patient_model_mae(rows, model):

    errors = []

    for row in rows:

        if row["model"] != model:
            continue

        value = f(
            row["absolute_error_mm"]
        )

        if finite(value):
            errors.append(
                abs(value)
            )

    if not errors:
        return float("nan")

    return float(
        np.mean(errors)
    )


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_ci(
    values,
    n_bootstrap=10000,
    seed=42
):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
            float("nan")
        )

    rng = np.random.RandomState(
        seed
    )

    boot_means = []

    for _ in range(
        n_bootstrap
    ):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        boot_means.append(
            np.mean(sample)
        )

    boot_means = np.asarray(
        boot_means
    )

    return (
        float(np.mean(values)),
        float(np.percentile(
            boot_means,
            2.5
        )),
        float(np.percentile(
            boot_means,
            97.5
        ))
    )


# ============================================================
# PAIRED PERMUTATION TEST
# ============================================================

def paired_permutation_test(
    improvements,
    n_permutations=10000,
    seed=42
):

    values = np.asarray(
        improvements,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return float("nan")

    observed = abs(
        np.mean(values)
    )

    rng = np.random.RandomState(
        seed
    )

    count = 0

    for _ in range(
        n_permutations
    ):

        signs = rng.choice(
            [-1.0, 1.0],
            size=len(values)
        )

        permuted = np.mean(
            values * signs
        )

        if abs(permuted) >= observed:

            count += 1

    return (
        count + 1
    ) / float(
        n_permutations + 1
    )


# ============================================================
# EFFECT SIZE
# ============================================================

def paired_effect_size(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return float("nan")

    sd = np.std(
        values,
        ddof=1
    )

    if sd == 0:
        return float("nan")

    return float(
        np.mean(values) / sd
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "STEP 17 - PATIENT-LEVEL PHENOTYPE ROBUSTNESS"
    )

    print("=" * 70)

    # ========================================================
    # LOAD STEP 16
    # ========================================================

    rows = read_csv(
        INPUT_FILE
    )

    print(
        "\nStep 16 rows:",
        len(rows)
    )

    patients = sorted(
        list(
            set(
                row[
                    "patient_id"
                ]
                for row in rows
            )
        )
    )

    print(
        "Patients:",
        len(patients)
    )

    if len(patients) != 13:

        print(
            "\nWARNING: Expected 13 test patients."
        )

    # ========================================================
    # MODELS
    # ========================================================

    GEOMETRY = (
        "M1_Geometry"
    )

    HBGD_RATE = (
        "M3_Geometry_HBGD_Rate"
    )

    ALL_HBGD = (
        "M5_Geometry_All_HBGD"
    )

    # ========================================================
    # PATIENT-LEVEL RESULTS
    # ========================================================

    patient_results = []

    phenotypes = [
        "Growth",
        "Stable",
        "Shrinkage"
    ]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PATIENT-LEVEL RESULTS"
    )

    print(
        "=" * 70
    )

    for patient in patients:

        patient_rows = [

            row

            for row in rows

            if row[
                "patient_id"
            ] == patient
        ]

        # ----------------------------------------------------
        # Determine phenotype for patient.
        #
        # A patient can contain multiple observed interval
        # phenotypes. We therefore calculate results at the
        # patient-within-phenotype level.
        # ----------------------------------------------------

        patient_phenotypes = sorted(
            list(
                set(
                    row[
                        "phenotype"
                    ]

                    for row in patient_rows

                    if row[
                        "phenotype"
                    ] in phenotypes
                )
            )
        )

        for state in patient_phenotypes:

            state_rows = [

                row

                for row in patient_rows

                if row[
                    "phenotype"
                ] == state
            ]

            geometry_mae = patient_model_mae(
                state_rows,
                GEOMETRY
            )

            hbgd_mae = patient_model_mae(
                state_rows,
                HBGD_RATE
            )

            all_hbgd_mae = patient_model_mae(
                state_rows,
                ALL_HBGD
            )

            if (
                not finite(
                    geometry_mae
                )
                or
                not finite(
                    hbgd_mae
                )
            ):
                continue

            hbgd_improvement = (
                geometry_mae
                -
                hbgd_mae
            )

            if (
                finite(
                    all_hbgd_mae
                )
            ):

                all_hbgd_improvement = (
                    geometry_mae
                    -
                    all_hbgd_mae
                )

            else:

                all_hbgd_improvement = float(
                    "nan"
                )

            if hbgd_improvement > 0:

                status = "Improved"

            elif hbgd_improvement < 0:

                status = "Worsened"

            else:

                status = "Unchanged"

            prediction_cases = len(
                set(
                    (
                        row[
                            "current_week"
                        ],
                        row[
                            "future_week"
                        ]
                    )

                    for row in state_rows
                )
            )

            patient_results.append({

                "patient_id":
                    patient,

                "phenotype":
                    state,

                "n_cases":
                    prediction_cases,

                "geometry_mae_mm":
                    geometry_mae,

                "hbgd_rate_mae_mm":
                    hbgd_mae,

                "all_hbgd_mae_mm":
                    all_hbgd_mae,

                "hbgd_rate_improvement_mm":
                    hbgd_improvement,

                "all_hbgd_improvement_mm":
                    all_hbgd_improvement,

                "status":
                    status
            })

    # ========================================================
    # PRINT PATIENT RESULTS
    # ========================================================

    for state in phenotypes:

        state_results = [

            row

            for row in patient_results

            if row[
                "phenotype"
            ] == state
        ]

        print(
            "\n--- {} ---".format(
                state
            )
        )

        print(
            "Patients:",
            len(state_results)
        )

        for row in state_results:

            print(
                "{} | cases={} | Geometry={:.4f} | HBGD={:.4f} | improvement={:+.4f} | {}".format(
                    row["patient_id"],
                    row["n_cases"],
                    row["geometry_mae_mm"],
                    row["hbgd_rate_mae_mm"],
                    row["hbgd_rate_improvement_mm"],
                    row["status"]
                )
            )

    # ========================================================
    # SAVE PATIENT RESULTS
    # ========================================================

    patient_fields = [

        "patient_id",
        "phenotype",
        "n_cases",
        "geometry_mae_mm",
        "hbgd_rate_mae_mm",
        "all_hbgd_mae_mm",
        "hbgd_rate_improvement_mm",
        "all_hbgd_improvement_mm",
        "status"
    ]

    write_csv(
        OUTPUT_PATIENT,
        patient_results,
        patient_fields
    )

    # ========================================================
    # PHENOTYPE ROBUSTNESS SUMMARY
    # ========================================================

    summaries = []

    bootstrap_rows = []

    permutation_rows = []

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHENOTYPE ROBUSTNESS SUMMARY"
    )

    print(
        "=" * 70
    )

    for state in phenotypes:

        state_results = [

            row

            for row in patient_results

            if row[
                "phenotype"
            ] == state
        ]

        improvements = np.asarray(

            [
                row[
                    "hbgd_rate_improvement_mm"
                ]

                for row in state_results
            ],

            dtype=float
        )

        improvements = improvements[
            np.isfinite(
                improvements
            )
        ]

        n = len(
            improvements
        )

        if n == 0:
            continue

        improved = int(
            np.sum(
                improvements > 0
            )
        )

        worsened = int(
            np.sum(
                improvements < 0
            )
        )

        unchanged = int(
            np.sum(
                improvements == 0
            )
        )

        mean_improvement = float(
            np.mean(
                improvements
            )
        )

        median_improvement = float(
            np.median(
                improvements
            )
        )

        sd_improvement = float(
            np.std(
                improvements,
                ddof=1
            )
        ) if n > 1 else float(
            "nan"
        )

        effect_size = paired_effect_size(
            improvements
        )

        mean_bootstrap, ci_low, ci_high = (
            bootstrap_ci(
                improvements,
                n_bootstrap=10000,
                seed=42
            )
        )

        permutation_p = (
            paired_permutation_test(
                improvements,
                n_permutations=10000,
                seed=42
            )
        )

        # ----------------------------------------------------
        # Geometry and HBGD pooled patient-level MAE
        # ----------------------------------------------------

        geometry_values = np.asarray(

            [
                row[
                    "geometry_mae_mm"
                ]

                for row in state_results

                if finite(
                    row[
                        "geometry_mae_mm"
                    ]
                )
            ],

            dtype=float
        )

        hbgd_values = np.asarray(

            [
                row[
                    "hbgd_rate_mae_mm"
                ]

                for row in state_results

                if finite(
                    row[
                        "hbgd_rate_mae_mm"
                    ]
                )
            ],

            dtype=float
        )

        pooled_geometry = float(
            np.mean(
                geometry_values
            )
        )

        pooled_hbgd = float(
            np.mean(
                hbgd_values
            )
        )

        relative_improvement = (

            mean_improvement
            /
            pooled_geometry
            *
            100.0
        )

        summary = {

            "phenotype":
                state,

            "n_patients":
                n,

            "improved":
                improved,

            "worsened":
                worsened,

            "unchanged":
                unchanged,

            "mean_geometry_mae_mm":
                pooled_geometry,

            "mean_hbgd_rate_mae_mm":
                pooled_hbgd,

            "mean_improvement_mm":
                mean_improvement,

            "median_improvement_mm":
                median_improvement,

            "sd_improvement_mm":
                sd_improvement,

            "relative_improvement_percent":
                relative_improvement,

            "paired_effect_size":
                effect_size,

            "bootstrap_ci_low_mm":
                ci_low,

            "bootstrap_ci_high_mm":
                ci_high,

            "paired_permutation_p":
                permutation_p
        }

        summaries.append(
            summary
        )

        bootstrap_rows.append({

            "phenotype":
                state,

            "n_patients":
                n,

            "bootstrap_mean_improvement_mm":
                mean_bootstrap,

            "bootstrap_95ci_low_mm":
                ci_low,

            "bootstrap_95ci_high_mm":
                ci_high,

            "n_bootstrap":
                10000
        })

        permutation_rows.append({

            "phenotype":
                state,

            "n_patients":
                n,

            "observed_mean_improvement_mm":
                mean_improvement,

            "paired_permutation_p":
                permutation_p,

            "n_permutations":
                10000
        })

        print(
            "\n{}".format(
                state
            )
        )

        print(
            "Patients:",
            n
        )

        print(
            "Improved:",
            improved
        )

        print(
            "Worsened:",
            worsened
        )

        print(
            "Unchanged:",
            unchanged
        )

        print(
            "Mean improvement: {:.4f} mm".format(
                mean_improvement
            )
        )

        print(
            "Median improvement: {:.4f} mm".format(
                median_improvement
            )
        )

        print(
            "Relative improvement: {:.2f}%".format(
                relative_improvement
            )
        )

        print(
            "Effect size: {:.4f}".format(
                effect_size
            )
        )

        print(
            "Bootstrap 95% CI: [{:.4f}, {:.4f}]".format(
                ci_low,
                ci_high
            )
        )

        print(
            "Paired permutation p: {:.6f}".format(
                permutation_p
            )
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_fields = [

        "phenotype",
        "n_patients",
        "improved",
        "worsened",
        "unchanged",
        "mean_geometry_mae_mm",
        "mean_hbgd_rate_mae_mm",
        "mean_improvement_mm",
        "median_improvement_mm",
        "sd_improvement_mm",
        "relative_improvement_percent",
        "paired_effect_size",
        "bootstrap_ci_low_mm",
        "bootstrap_ci_high_mm",
        "paired_permutation_p"
    ]

    write_csv(
        OUTPUT_SUMMARY,
        summaries,
        summary_fields
    )

    write_csv(

        OUTPUT_BOOTSTRAP,

        bootstrap_rows,

        [
            "phenotype",
            "n_patients",
            "bootstrap_mean_improvement_mm",
            "bootstrap_95ci_low_mm",
            "bootstrap_95ci_high_mm",
            "n_bootstrap"
        ]
    )

    write_csv(

        OUTPUT_PERMUTATION,

        permutation_rows,

        [
            "phenotype",
            "n_patients",
            "observed_mean_improvement_mm",
            "paired_permutation_p",
            "n_permutations"
        ]
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    report.append(
        "# STEP 17 - PATIENT-LEVEL PHENOTYPE ROBUSTNESS"
    )

    report.append("")

    report.append(
        "## Objective"
    )

    report.append("")

    report.append(
        "Evaluate whether HBGD-derived growth-rate information provides consistent patient-level predictive benefit within Growth, Stable and Shrinkage phenotypes."
    )

    report.append("")

    report.append(
        "## Data Integrity"
    )

    report.append("")

    report.append(
        "The analysis uses the exact Step-12 predictions recovered in Step 16."
    )

    report.append("")

    report.append(
        "No model was retrained and no patient split was reconstructed."
    )

    report.append("")

    report.append(
        "Phenotype was assigned from observed interval growth dynamics and was used only for post-hoc stratification."
    )

    report.append("")

    report.append(
        "## Results"
    )

    report.append("")

    report.append(
        "| Phenotype | Patients | Improved | Worsened | Mean Improvement (mm) | Relative Improvement | Bootstrap 95% CI | Permutation p |"
    )

    report.append(
        "|---|---:|---:|---:|---:|---:|---|---:|"
    )

    for row in summaries:

        report.append(
            "| {} | {} | {} | {} | {:.4f} | {:.2f}% | [{:.4f}, {:.4f}] | {:.6f} |".format(
                row["phenotype"],
                row["n_patients"],
                row["improved"],
                row["worsened"],
                row["mean_improvement_mm"],
                row["relative_improvement_percent"],
                row["bootstrap_ci_low_mm"],
                row["bootstrap_ci_high_mm"],
                row["paired_permutation_p"]
            )
        )

    report.append("")

    report.append(
        "## Interpretation"
    )

    report.append("")

    report.append(
        "Positive improvement values indicate lower MAE for Geometry + HBGD Rate compared with Geometry alone."
    )

    report.append("")

    report.append(
        "Negative values indicate that adding HBGD rate worsened prediction."
    )

    report.append("")

    report.append(
        "Bootstrap confidence intervals and paired permutation tests are patient-level analyses and should be interpreted cautiously because the test cohort contains only 13 patients."
    )

    report.append("")

    report.append(
        "The phenotype-specific analysis is therefore evidence of robustness or heterogeneity of the observed effect, rather than definitive statistical validation."
    )

    report.append("")

    report.append(
        "## Conclusion"
    )

    report.append("")

    report.append(
        "Step 17 evaluates whether the incremental contribution of HBGD-derived growth-rate information is consistent across individual patients within each observed growth phenotype."
    )

    report.append("")

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(report)
        )

    # ========================================================
    # DONE
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 17 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nOutputs saved to:"
    )

    print(
        TARGET_DIR
    )

    print(
        "\nSummary:"
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        "\nReport:"
    )

    print(
        OUTPUT_REPORT
    )


if __name__ == "__main__":

    main()