# -*- coding: utf-8 -*-

"""
STEP 16
EXACT STEP-12 TEST COHORT FROM STEP-13 PATIENT RESULTS

Step 13 was generated using the established Step-12
patient-level split.

Therefore, Step 13 patient-level results are used ONLY
to recover the exact 13 test patient IDs.

No model is retrained.
No predictions are modified.
Phenotype is used only for post-hoc stratification.
"""

from __future__ import print_function

import os
import csv
import math
import numpy as np


# ============================================================
# PATHS
# ============================================================

TARGET_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"

STEP12_FILE = os.path.join(
    TARGET_DIR,
    "step12_hbgd_ablation_predictions.csv"
)

STEP13_FILE = os.path.join(
    TARGET_DIR,
    "step13_patient_level_hbgd_results.csv"
)

TRAJECTORY_FILE = os.path.join(
    TARGET_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

OUTPUT_COHORT = os.path.join(
    TARGET_DIR,
    "step16_exact_step12_test_cohort.csv"
)

OUTPUT_MATCHED = os.path.join(
    TARGET_DIR,
    "step16_exact_step12_phenotype_predictions.csv"
)

OUTPUT_SUMMARY = os.path.join(
    TARGET_DIR,
    "step16_exact_phenotype_model_comparison.csv"
)

OUTPUT_PATIENT = os.path.join(
    TARGET_DIR,
    "step16_exact_phenotype_patient_results.csv"
)

OUTPUT_REPORT = os.path.join(
    TARGET_DIR,
    "STEP16_EXACT_STEP12_TEST_COHORT_REPORT.md"
)


# ============================================================
# CSV HELPERS
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


def to_float(x):

    try:
        return float(x)
    except Exception:
        return float("nan")


def is_finite(x):

    try:
        return math.isfinite(float(x))
    except Exception:
        return False


# ============================================================
# HBGD RATE
# ============================================================

def calculate_rate(d1, d2, days):

    if (
        not is_finite(d1)
        or not is_finite(d2)
        or d1 <= 0
        or d2 <= 0
        or days <= 0
    ):
        return float("nan")

    return math.log(d2 / d1) / days


# ============================================================
# PHENOTYPE
# ============================================================

def classify_phenotype(r):

    if not is_finite(r):
        return "Unknown"

    if r > 0.0005:
        return "Growth"

    if r < -0.0010:
        return "Shrinkage"

    return "Stable"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 16 - EXACT STEP-12 TEST COHORT")
    print("=" * 70)

    # ========================================================
    # 1. LOAD STEP 13 PATIENT RESULTS
    # ========================================================

    step13 = read_csv(
        STEP13_FILE
    )

    print(
        "\nStep 13 rows:",
        len(step13)
    )

    print(
        "Step 13 columns:"
    )

    for column in step13[0].keys():

        print(
            "  " + column
        )

    # ========================================================
    # 2. EXTRACT EXACT TEST PATIENTS
    # ========================================================

    test_patients = sorted(
        list(
            set(
                [
                    str(
                        row["patient_id"]
                    )
                    for row in step13
                ]
            )
        )
    )

    print(
        "\nRecovered test patients:",
        len(test_patients)
    )

    for patient in test_patients:

        print(
            "  " + patient
        )

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(test_patients) != 13:

        print("\nERROR:")
        print(
            "Expected 13 test patients but found {}.".format(
                len(test_patients)
            )
        )

        print(
            "Stopping. No cohort will be guessed."
        )

        return

    # ========================================================
    # 3. LOAD STEP 12
    # ========================================================

    step12 = read_csv(
        STEP12_FILE
    )

    print(
        "\nStep 12 prediction rows:",
        len(step12)
    )

    # ========================================================
    # 4. FILTER STEP 12 TO EXACT TEST COHORT
    # ========================================================

    test_rows = [

        row

        for row in step12

        if str(
            row["patient_id"]
        ) in test_patients
    ]

    print(
        "Filtered Step 12 rows:",
        len(test_rows)
    )

    # ========================================================
    # 5. UNIQUE PREDICTION CASES
    # ========================================================

    case_keys = sorted(
        list(
            set(
                [
                    (
                        str(
                            row["patient_id"]
                        ),

                        int(
                            round(
                                to_float(
                                    row[
                                        "current_week"
                                    ]
                                )
                            )
                        ),

                        int(
                            round(
                                to_float(
                                    row[
                                        "future_week"
                                    ]
                                )
                            )
                        )
                    )

                    for row in test_rows
                ]
            )
        )
    )

    print(
        "Unique prediction cases:",
        len(case_keys)
    )

    print(
        "Expected prediction cases: 88"
    )

    print(
        "Expected model rows: 440"
    )

    # ========================================================
    # VERIFY
    # ========================================================

    if len(case_keys) != 88:

        print(
            "\nWARNING: Expected 88 cases."
        )

    if len(test_rows) != 440:

        print(
            "\nWARNING: Expected 440 rows."
        )

    if (
        len(case_keys) == 88
        and
        len(test_rows) == 440
    ):

        print(
            "\nSUCCESS:"
        )

        print(
            "Exact Step-12 test cohort recovered."
        )

    # ========================================================
    # SAVE COHORT
    # ========================================================

    cohort_rows = []

    for patient in test_patients:

        patient_rows = [

            row

            for row in test_rows

            if str(
                row["patient_id"]
            ) == patient
        ]

        patient_cases = set(

            (
                row["current_week"],
                row["future_week"]
            )

            for row in patient_rows
        )

        cohort_rows.append({

            "patient_id":
                patient,

            "prediction_cases":
                len(patient_cases),

            "prediction_rows":
                len(patient_rows)
        })

    write_csv(

        OUTPUT_COHORT,

        cohort_rows,

        [
            "patient_id",
            "prediction_cases",
            "prediction_rows"
        ]
    )

    # ========================================================
    # 6. LOAD LONGITUDINAL TRAJECTORIES
    # ========================================================

    trajectories = read_csv(
        TRAJECTORY_FILE
    )

    print(
        "\nTrajectory rows:",
        len(trajectories)
    )

    # ========================================================
    # BUILD INTERVAL LOOKUP
    # ========================================================

    lookup = {}

    patient_data = {}

    for row in trajectories:

        patient = str(
            row.get(
                "patient_id",
                ""
            )
        )

        week = to_float(
            row.get(
                "week",
                ""
            )
        )

        diameter = to_float(
            row.get(
                "equivalent_diameter_mm",
                ""
            )
        )

        if (
            not patient
            or
            not is_finite(week)
            or
            not is_finite(diameter)
            or
            diameter <= 0
        ):

            continue

        if patient not in patient_data:

            patient_data[
                patient
            ] = []

        patient_data[
            patient
        ].append(
            (
                int(round(week)),
                diameter
            )
        )

    # ========================================================
    # SORT
    # ========================================================

    for patient in patient_data:

        patient_data[
            patient
        ].sort(
            key=lambda x: x[0]
        )

    # ========================================================
    # CREATE INTERVAL PHENOTYPES
    # ========================================================

    for patient in patient_data:

        observations = patient_data[
            patient
        ]

        for i in range(
            len(observations) - 1
        ):

            week1, d1 = observations[i]

            week2, d2 = observations[i + 1]

            days = (
                week2 -
                week1
            ) * 7.0

            r = calculate_rate(
                d1,
                d2,
                days
            )

            key = (
                patient,
                week1,
                week2
            )

            lookup[key] = {

                "growth_rate":
                    r,

                "phenotype":
                    classify_phenotype(
                        r
                    )
            }

    # ========================================================
    # 7. MATCH STEP 12 PREDICTIONS
    # ========================================================

    matched = []

    unmatched = []

    for row in test_rows:

        patient = str(
            row["patient_id"]
        )

        current_week = int(
            round(
                to_float(
                    row[
                        "current_week"
                    ]
                )
            )
        )

        future_week = int(
            round(
                to_float(
                    row[
                        "future_week"
                    ]
                )
            )
        )

        key = (
            patient,
            current_week,
            future_week
        )

        info = lookup.get(
            key
        )

        if info is None:

            unmatched.append(
                row
            )

            continue

        new_row = dict(row)

        new_row[
            "observed_hbgd_rate_per_day"
        ] = info[
            "growth_rate"
        ]

        new_row[
            "phenotype"
        ] = info[
            "phenotype"
        ]

        matched.append(
            new_row
        )

    print(
        "\nPhenotype matched rows:",
        len(matched)
    )

    print(
        "Unmatched rows:",
        len(unmatched)
    )

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(unmatched) > 0:

        print(
            "\nWARNING:"
        )

        print(
            "Some Step-12 intervals could not be matched."
        )

    # ========================================================
    # SAVE MATCHED PREDICTIONS
    # ========================================================

    fields = list(
        matched[0].keys()
    )

    write_csv(

        OUTPUT_MATCHED,

        matched,

        fields
    )

    # ========================================================
    # 8. PHENOTYPE DISTRIBUTION
    # ========================================================

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
        "PHENOTYPE DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    for state in phenotypes:

        cases = set(

            (
                row["patient_id"],
                row["current_week"],
                row["future_week"]
            )

            for row in matched

            if row[
                "phenotype"
            ] == state
        )

        print(
            "{}: {} cases".format(
                state,
                len(cases)
            )
        )

    # ========================================================
    # 9. MODEL PERFORMANCE BY PHENOTYPE
    # ========================================================

    models = sorted(
        list(
            set(
                [
                    row["model"]
                    for row in matched
                ]
            )
        )
    )

    summary = []

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHENOTYPE-STRATIFIED PERFORMANCE"
    )

    print(
        "=" * 70
    )

    for state in phenotypes:

        print(
            "\n--- {} ---".format(
                state
            )
        )

        state_rows = [

            row

            for row in matched

            if row[
                "phenotype"
            ] == state
        ]

        state_cases = set(

            (
                row["patient_id"],
                row["current_week"],
                row["future_week"]
            )

            for row in state_rows
        )

        for model in models:

            rows = [

                row

                for row in state_rows

                if row["model"] == model
            ]

            errors = [

                abs(
                    to_float(
                        row[
                            "absolute_error_mm"
                        ]
                    )
                )

                for row in rows

                if is_finite(
                    row[
                        "absolute_error_mm"
                    ]
                )
            ]

            squared_errors = [

                (
                    to_float(
                        row[
                            "predicted_diameter_mm"
                        ]
                    )
                    -
                    to_float(
                        row[
                            "actual_diameter_mm"
                        ]
                    )
                ) ** 2

                for row in rows

                if (
                    is_finite(
                        row[
                            "predicted_diameter_mm"
                        ]
                    )
                    and
                    is_finite(
                        row[
                            "actual_diameter_mm"
                        ]
                    )
                )
            ]

            if not errors:
                continue

            mae = float(
                np.mean(
                    errors
                )
            )

            rmse = math.sqrt(
                np.mean(
                    squared_errors
                )
            )

            print(
                "{:<35} N={}  MAE={:.4f}  RMSE={:.4f}".format(
                    model,
                    len(state_cases),
                    mae,
                    rmse
                )
            )

            summary.append({

                "phenotype":
                    state,

                "n_cases":
                    len(state_cases),

                "model":
                    model,

                "mae_mm":
                    mae,

                "rmse_mm":
                    rmse
            })

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    write_csv(

        OUTPUT_SUMMARY,

        summary,

        [
            "phenotype",
            "n_cases",
            "model",
            "mae_mm",
            "rmse_mm"
        ]
    )

    # ========================================================
    # 10. PATIENT-LEVEL PHENOTYPE RESULTS
    # ========================================================

    patient_results = []

    for state in phenotypes:

        state_rows = [

            row

            for row in matched

            if row[
                "phenotype"
            ] == state
        ]

        patients = sorted(
            list(
                set(
                    [
                        row[
                            "patient_id"
                        ]

                        for row in state_rows
                    ]
                )
            )
        )

        for patient in patients:

            patient_rows = [

                row

                for row in state_rows

                if row[
                    "patient_id"
                ] == patient
            ]

            result = {

                "patient_id":
                    patient,

                "phenotype":
                    state
            }

            model_mae = {}

            for model in models:

                rows = [

                    row

                    for row in patient_rows

                    if row["model"] == model
                ]

                errors = [

                    abs(
                        to_float(
                            row[
                                "absolute_error_mm"
                            ]
                        )
                    )

                    for row in rows

                    if is_finite(
                        row[
                            "absolute_error_mm"
                        ]
                    )
                ]

                if not errors:
                    continue

                mae = float(
                    np.mean(
                        errors
                    )
                )

                model_mae[
                    model
                ] = mae

                result[
                    model +
                    "_MAE_mm"
                ] = mae

            # ------------------------------------------------
            # HBGD rate improvement over geometry
            # ------------------------------------------------

            if (
                "M1_Geometry"
                in model_mae
                and
                "M3_Geometry_HBGD_Rate"
                in model_mae
            ):

                result[
                    "HBGD_Rate_Improvement_mm"
                ] = (

                    model_mae[
                        "M1_Geometry"
                    ]
                    -
                    model_mae[
                        "M3_Geometry_HBGD_Rate"
                    ]
                )

            # ------------------------------------------------
            # All HBGD improvement
            # ------------------------------------------------

            if (
                "M1_Geometry"
                in model_mae
                and
                "M5_Geometry_All_HBGD"
                in model_mae
            ):

                result[
                    "All_HBGD_Improvement_mm"
                ] = (

                    model_mae[
                        "M1_Geometry"
                    ]
                    -
                    model_mae[
                        "M5_Geometry_All_HBGD"
                    ]
                )

            patient_results.append(
                result
            )

    # ========================================================
    # SAVE PATIENT RESULTS
    # ========================================================

    if patient_results:

        patient_fields = list(
            patient_results[0].keys()
        )

        write_csv(

            OUTPUT_PATIENT,

            patient_results,

            patient_fields
        )

    # ========================================================
    # 11. REPORT
    # ========================================================

    report = []

    report.append(
        "# STEP 16 - EXACT STEP-12 TEST COHORT PHENOTYPE ANALYSIS"
    )

    report.append("")

    report.append(
        "## Cohort Recovery"
    )

    report.append("")

    report.append(
        "The exact Step-12 test cohort was recovered from the patient IDs contained in the Step-13 patient-level results."
    )

    report.append("")

    report.append(
        "Step 13 was generated using the established Step-12 patient split. Therefore, these IDs were used only to identify the same held-out cohort."
    )

    report.append("")

    report.append(
        "Test patients: {}".format(
            len(test_patients)
        )
    )

    report.append("")

    report.append(
        "Prediction cases: {}".format(
            len(case_keys)
        )
    )

    report.append("")

    report.append(
        "Prediction rows: {}".format(
            len(test_rows)
        )
    )

    report.append("")

    report.append(
        "Phenotype matched rows: {}".format(
            len(matched)
        )
    )

    report.append("")

    report.append(
        "Unmatched rows: {}".format(
            len(unmatched)
        )
    )

    report.append("")

    report.append(
        "## Test Patients"
    )

    report.append("")

    for patient in test_patients:

        report.append(
            "- {}".format(
                patient
            )
        )

    report.append("")

    report.append(
        "## Phenotype-Stratified Results"
    )

    report.append("")

    report.append(
        "| Phenotype | N Cases | Model | MAE (mm) | RMSE (mm) |"
    )

    report.append(
        "|---|---:|---|---:|---:|"
    )

    for row in summary:

        report.append(
            "| {} | {} | {} | {:.4f} | {:.4f} |".format(
                row["phenotype"],
                row["n_cases"],
                row["model"],
                row["mae_mm"],
                row["rmse_mm"]
            )
        )

    report.append("")

    report.append(
        "## Interpretation"
    )

    report.append("")

    report.append(
        "Phenotype was assigned post-hoc from the observed interval growth rate and was not provided as a predictive model input."
    )

    report.append("")

    report.append(
        "The analysis therefore evaluates whether the already-generated Step-12 models behave differently across observed growth phenotypes."
    )

    report.append("")

    report.append(
        "These results are descriptive and should not be interpreted as statistically significant without an appropriate patient-level statistical analysis."
    )

    report.append("")

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
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
        "STEP 16 COMPLETE"
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
        "\nKey output:"
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