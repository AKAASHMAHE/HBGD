# -*- coding: utf-8 -*-

"""
STEP 15 - EXACT STEP-12 PHENOTYPE STRATIFICATION

Uses the EXISTING Step 12 prediction CSV.

Step 12 CSV format:

    patient_id
    current_week
    future_week
    actual_diameter_mm
    model
    predicted_diameter_mm
    absolute_error_mm

No model retraining is performed.

The script:
    1. Loads Step 12 predictions
    2. Loads LUMIERE longitudinal trajectories
    3. Determines observed phenotype for each prediction interval
    4. Matches phenotype to the exact Step 12 prediction
    5. Compares models within:
         Growth
         Stable
         Shrinkage
    6. Produces case-level and patient-level summaries

Python 3.6 compatible.
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

TRAJECTORY_FILE = os.path.join(
    TARGET_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

OUTPUT_MATCHED = os.path.join(
    TARGET_DIR,
    "step15_step12_predictions_with_phenotype.csv"
)

OUTPUT_SUMMARY = os.path.join(
    TARGET_DIR,
    "step15_phenotype_model_comparison.csv"
)

OUTPUT_PATIENT = os.path.join(
    TARGET_DIR,
    "step15_phenotype_patient_results.csv"
)

OUTPUT_REPORT = os.path.join(
    TARGET_DIR,
    "STEP15_EXACT_STEP12_PHENOTYPE_STRATIFICATION_REPORT.md"
)


# ============================================================
# HBGD THRESHOLDS
# ============================================================

GROWTH_THRESHOLD = 0.0005

SHRINKAGE_THRESHOLD = -0.0010

DAYS_PER_WEEK = 7.0


# ============================================================
# CSV HELPERS
# ============================================================

def read_csv(path):

    if not os.path.exists(path):

        raise IOError(
            "File not found:\n{}".format(path)
        )

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


def write_csv(
    path,
    rows,
    fields
):

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


# ============================================================
# NUMERIC
# ============================================================

def safe_float(value):

    try:

        return float(value)

    except Exception:

        return float("nan")


def finite(value):

    try:

        return math.isfinite(
            float(value)
        )

    except Exception:

        return False


def calculate_rate(
    d1,
    d2,
    days
):

    if (
        not finite(d1)
        or
        not finite(d2)
        or
        d1 <= 0
        or
        d2 <= 0
        or
        days <= 0
    ):

        return float("nan")

    return (
        math.log(
            d2 / d1
        )
        /
        float(days)
    )


def get_state(rate):

    if not finite(rate):

        return "Unknown"

    if rate > GROWTH_THRESHOLD:

        return "Growth"

    elif rate < SHRINKAGE_THRESHOLD:

        return "Shrinkage"

    else:

        return "Stable"


# ============================================================
# LOAD TRAJECTORIES
# ============================================================

def load_trajectories():

    rows = read_csv(
        TRAJECTORY_FILE
    )

    patients = {}

    for row in rows:

        patient_id = str(
            row.get(
                "patient_id",
                ""
            )
        )

        if not patient_id:

            continue

        week = safe_float(
            row.get(
                "week",
                ""
            )
        )

        diameter = safe_float(
            row.get(
                "equivalent_diameter_mm",
                ""
            )
        )

        if (
            not finite(week)
            or
            not finite(diameter)
            or
            diameter <= 0
        ):

            continue

        if patient_id not in patients:

            patients[
                patient_id
            ] = []

        patients[
            patient_id
        ].append({

            "week":
                int(round(week)),

            "diameter":
                diameter
        })

    for patient_id in patients:

        patients[
            patient_id
        ].sort(
            key=lambda x:
                x["week"]
        )

    return patients


# ============================================================
# PHENOTYPE LOOKUP
# ============================================================

def build_phenotype_lookup(
    patients
):

    lookup = {}

    for patient_id in patients:

        observations = patients[
            patient_id
        ]

        for i in range(
            1,
            len(observations)
        ):

            start = observations[
                i - 1
            ]

            end = observations[
                i
            ]

            start_week = start[
                "week"
            ]

            target_week = end[
                "week"
            ]

            delta_days = (
                target_week -
                start_week
            ) * DAYS_PER_WEEK

            rate = calculate_rate(
                start[
                    "diameter"
                ],
                end[
                    "diameter"
                ],
                delta_days
            )

            state = get_state(
                rate
            )

            key = (
                patient_id,
                start_week,
                target_week
            )

            lookup[
                key
            ] = {

                "phenotype":
                    state,

                "observed_rate":
                    rate,

                "start_diameter":
                    start[
                        "diameter"
                    ],

                "target_diameter":
                    end[
                        "diameter"
                    ]
            }

    return lookup


# ============================================================
# NORMALIZE MODEL NAME
# ============================================================

def normalize_model_name(
    value
):

    text = str(
        value
    ).strip().lower()

    text = text.replace(
        "_",
        " "
    )

    text = text.replace(
        "-",
        " "
    )

    # --------------------------------------------------------
    # Detect no-change
    # --------------------------------------------------------

    if (
        "no change" in text
        or
        "nochange" in text
        or
        "baseline" in text
    ):

        return "No-change"

    # --------------------------------------------------------
    # Detect all HBGD
    # --------------------------------------------------------

    if (
        "m5" in text
        or
        "all hbgd" in text
        or
        "full hbgd" in text
        or
        (
            "geometry" in text
            and
            "all" in text
            and
            "hbgd" in text
        )
    ):

        return "Geometry + All HBGD"

    # --------------------------------------------------------
    # Detect HBGD rate
    # --------------------------------------------------------

    if (
        "m3" in text
        or
        "hbgd rate" in text
        or
        (
            "geometry" in text
            and
            "hbgd" in text
            and
            "rate" in text
        )
    ):

        return "Geometry + HBGD Rate"

    # --------------------------------------------------------
    # Detect geometry
    # --------------------------------------------------------

    if (
        text == "geometry"
        or
        "m1" in text
        or
        "geometry only" in text
    ):

        return "Geometry"

    return str(
        value
    )


# ============================================================
# MAE
# ============================================================

def calculate_mae(
    rows
):

    errors = []

    for row in rows:

        value = safe_float(
            row[
                "absolute_error_mm"
            ]
        )

        if finite(value):

            errors.append(
                abs(value)
            )

    if not errors:

        return float("nan")

    return float(
        np.mean(
            np.asarray(
                errors,
                dtype=float
            )
        )
    )


# ============================================================
# RMSE
# ============================================================

def calculate_rmse(
    rows
):

    squared_errors = []

    for row in rows:

        actual = safe_float(
            row[
                "actual_diameter_mm"
            ]
        )

        prediction = safe_float(
            row[
                "predicted_diameter_mm"
            ]
        )

        if (
            finite(actual)
            and
            finite(prediction)
        ):

            squared_errors.append(
                (
                    prediction -
                    actual
                ) ** 2
            )

    if not squared_errors:

        return float("nan")

    return float(
        math.sqrt(
            np.mean(
                np.asarray(
                    squared_errors,
                    dtype=float
                )
            )
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "STEP 15 - EXACT STEP-12 PHENOTYPE STRATIFICATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    step12_rows = read_csv(
        STEP12_FILE
    )

    trajectory_patients = (
        load_trajectories()
    )

    print(
        "\nStep 12 prediction rows:",
        len(step12_rows)
    )

    print(
        "Trajectory patients:",
        len(
            trajectory_patients
        )
    )

    if not step12_rows:

        raise ValueError(
            "Step 12 prediction file is empty."
        )

    # --------------------------------------------------------
    # Show columns
    # --------------------------------------------------------

    columns = list(
        step12_rows[0].keys()
    )

    print(
        "\nStep 12 columns:"
    )

    for column in columns:

        print(
            "  {}".format(
                column
            )
        )

    required_columns = [

        "patient_id",
        "current_week",
        "future_week",
        "actual_diameter_mm",
        "model",
        "predicted_diameter_mm",
        "absolute_error_mm"
    ]

    for column in required_columns:

        if column not in columns:

            raise ValueError(
                "Required Step 12 column missing: {}".format(
                    column
                )
            )

    # --------------------------------------------------------
    # Build phenotype lookup
    # --------------------------------------------------------

    phenotype_lookup = (
        build_phenotype_lookup(
            trajectory_patients
        )
    )

    # --------------------------------------------------------
    # Match exact prediction rows
    # --------------------------------------------------------

    matched_rows = []

    unmatched = []

    for row in step12_rows:

        patient_id = str(
            row[
                "patient_id"
            ]
        )

        current_week = int(
            round(
                safe_float(
                    row[
                        "current_week"
                    ]
                )
            )
        )

        future_week = int(
            round(
                safe_float(
                    row[
                        "future_week"
                    ]
                )
            )
        )

        key = (
            patient_id,
            current_week,
            future_week
        )

        phenotype_info = (
            phenotype_lookup.get(
                key
            )
        )

        if phenotype_info is None:

            unmatched.append(
                (
                    patient_id,
                    current_week,
                    future_week
                )
            )

            continue

        new_row = dict(
            row
        )

        new_row[
            "normalized_model"
        ] = normalize_model_name(
            row[
                "model"
            ]
        )

        new_row[
            "phenotype"
        ] = phenotype_info[
            "phenotype"
        ]

        new_row[
            "observed_interval_hbgd_rate_per_day"
        ] = phenotype_info[
            "observed_rate"
        ]

        matched_rows.append(
            new_row
        )

    print(
        "\nMatched rows:",
        len(matched_rows)
    )

    print(
        "Unmatched rows:",
        len(unmatched)
    )

    # --------------------------------------------------------
    # Show model names
    # --------------------------------------------------------

    print(
        "\nOriginal model names:"
    )

    original_models = sorted(
        list(
            set(
                [
                    str(
                        row[
                            "model"
                        ]
                    )
                    for row in step12_rows
                ]
            )
        )
    )

    for model in original_models:

        print(
            "  {} -> {}".format(
                model,
                normalize_model_name(
                    model
                )
            )
        )

    # --------------------------------------------------------
    # Phenotype counts
    # --------------------------------------------------------

    phenotypes = [

        "Growth",

        "Stable",

        "Shrinkage"
    ]

    print(
        "\nPhenotype distribution:"
    )

    for phenotype in phenotypes:

        count = sum(

            1

            for row in matched_rows

            if row[
                "phenotype"
            ] == phenotype
        )

        print(
            "{}: {}".format(
                phenotype,
                count
            )
        )

    # --------------------------------------------------------
    # Test patients
    #
    # Step 12 contains 440 rows = 88 cases x 5 models.
    #
    # We identify the test cohort using the known Step 12
    # test patients generated by seed 42.
    # --------------------------------------------------------

    all_patient_ids = sorted(
        list(
            set(
                [
                    str(
                        row[
                            "patient_id"
                        ]
                    )

                    for row in matched_rows
                ]
            )
        )
    )

    # --------------------------------------------------------
    # Reconstruct exact seed-42 split
    # --------------------------------------------------------

    import random

    random.seed(
        42
    )

    shuffled = all_patient_ids[:]

    random.shuffle(
        shuffled
    )

    n = len(shuffled)

    n_test = int(
        round(
            n * 0.20
        )
    )

    test_ids = sorted(
        shuffled[
            :n_test
        ]
    )

    print(
        "\nReconstructed Step 12 test patients:",
        len(test_ids)
    )

    print(
        test_ids
    )

    # --------------------------------------------------------
    # Keep ONLY test rows
    # --------------------------------------------------------

    test_rows = [

        row

        for row in matched_rows

        if str(
            row[
                "patient_id"
            ]
        )
        in test_ids
    ]

    print(
        "\nStep 12 test prediction rows:",
        len(test_rows)
    )

    # --------------------------------------------------------
    # Expected: 88 cases x 5 models = 440 rows
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Model names actually available
    # --------------------------------------------------------

    normalized_models = sorted(
        list(
            set(
                [
                    row[
                        "normalized_model"
                    ]

                    for row in test_rows
                ]
            )
        )
    )

    print(
        "\nNormalized models:"
    )

    for model in normalized_models:

        print(
            "  {}".format(
                model
            )
        )

    # --------------------------------------------------------
    # Phenotype/model summary
    # --------------------------------------------------------

    summary_rows = []

    for phenotype in phenotypes:

        phenotype_rows = [

            row

            for row in test_rows

            if row[
                "phenotype"
            ] == phenotype
        ]

        if not phenotype_rows:

            continue

        for model in normalized_models:

            model_rows = [

                row

                for row in phenotype_rows

                if row[
                    "normalized_model"
                ] == model
            ]

            if not model_rows:

                continue

            phenotype_mae = calculate_mae(
                model_rows
            )

            phenotype_rmse = calculate_rmse(
                model_rows
            )

            summary_rows.append({

                "phenotype":
                    phenotype,

                "n_cases":
                    len(
                        model_rows
                    ),

                "model":
                    model,

                "mae_mm":
                    phenotype_mae,

                "rmse_mm":
                    phenotype_rmse
            })

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHENOTYPE-STRATIFIED TEST RESULTS"
    )

    print(
        "=" * 70
    )

    for phenotype in phenotypes:

        rows = [

            row

            for row in summary_rows

            if row[
                "phenotype"
            ] == phenotype
        ]

        if not rows:

            continue

        print(
            "\n{} PHENOTYPE".format(
                phenotype.upper()
            )
        )

        print(
            "N cases:",
            rows[0][
                "n_cases"
            ]
        )

        for row in rows:

            print(
                "{}: MAE = {:.4f} mm, RMSE = {:.4f} mm".format(
                    row["model"],
                    row["mae_mm"],
                    row["rmse_mm"]
                )
            )

    # --------------------------------------------------------
    # Patient-level analysis
    # --------------------------------------------------------

    patient_results = []

    for phenotype in phenotypes:

        phenotype_rows = [

            row

            for row in test_rows

            if row[
                "phenotype"
            ] == phenotype
        ]

        phenotype_patient_ids = sorted(
            list(
                set(
                    [
                        str(
                            row[
                                "patient_id"
                            ]
                        )

                        for row in phenotype_rows
                    ]
                )
            )
        )

        for patient_id in phenotype_patient_ids:

            patient_rows = [

                row

                for row in phenotype_rows

                if str(
                    row[
                        "patient_id"
                    ]
                )
                ==
                patient_id
            ]

            model_values = {}

            for model in normalized_models:

                model_rows = [

                    row

                    for row in patient_rows

                    if row[
                        "normalized_model"
                    ] == model
                ]

                if not model_rows:

                    continue

                model_values[
                    model
                ] = calculate_mae(
                    model_rows
                )

            result = {

                "patient_id":
                    patient_id,

                "phenotype":
                    phenotype,

                "n_cases":
                    len(
                        patient_rows
                    )
            }

            for model in normalized_models:

                if model not in model_values:

                    continue

                safe_name = (
                    model
                    .lower()
                    .replace(
                        " ",
                        "_"
                    )
                    .replace(
                        "+",
                        "plus"
                    )
                    .replace(
                        "-",
                        "_"
                    )
                )

                result[
                    safe_name +
                    "_mae_mm"
                ] = model_values[
                    model
                ]

            # ------------------------------------------------
            # Improvement of HBGD-rate vs geometry
            # ------------------------------------------------

            if (
                "Geometry"
                in model_values
                and
                "Geometry + HBGD Rate"
                in model_values
            ):

                result[
                    "hbgd_rate_improvement_mm"
                ] = (
                    model_values[
                        "Geometry"
                    ]
                    -
                    model_values[
                        "Geometry + HBGD Rate"
                    ]
                )

            # ------------------------------------------------
            # Improvement of All-HBGD vs geometry
            # ------------------------------------------------

            if (
                "Geometry"
                in model_values
                and
                "Geometry + All HBGD"
                in model_values
            ):

                result[
                    "all_hbgd_improvement_mm"
                ] = (
                    model_values[
                        "Geometry"
                    ]
                    -
                    model_values[
                        "Geometry + All HBGD"
                    ]
                )

            patient_results.append(
                result
            )

    # --------------------------------------------------------
    # Print patient-level results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PATIENT-LEVEL PHENOTYPE RESULTS"
    )

    print(
        "=" * 70
    )

    for phenotype in phenotypes:

        rows = [

            row

            for row in patient_results

            if row[
                "phenotype"
            ] == phenotype
        ]

        if not rows:

            continue

        hbgd_improvements = [

            row[
                "hbgd_rate_improvement_mm"
            ]

            for row in rows

            if "hbgd_rate_improvement_mm"
            in row
        ]

        all_hbgd_improvements = [

            row[
                "all_hbgd_improvement_mm"
            ]

            for row in rows

            if "all_hbgd_improvement_mm"
            in row
        ]

        print(
            "\n{}".format(
                phenotype
            )
        )

        print(
            "Patients:",
            len(rows)
        )

        if hbgd_improvements:

            print(
                "Mean HBGD-rate improvement:",
                "{:.4f} mm".format(
                    float(
                        np.mean(
                            hbgd_improvements
                        )
                    )
                )
            )

            print(
                "Patients improved by HBGD rate:",
                sum(
                    1
                    for x in hbgd_improvements
                    if x > 0
                )
            )

            print(
                "Patients worsened by HBGD rate:",
                sum(
                    1
                    for x in hbgd_improvements
                    if x < 0
                )
            )

        if all_hbgd_improvements:

            print(
                "Mean All-HBGD improvement:",
                "{:.4f} mm".format(
                    float(
                        np.mean(
                            all_hbgd_improvements
                        )
                    )
                )
            )

    # --------------------------------------------------------
    # Save matched predictions
    # --------------------------------------------------------

    matched_fields = list(
        matched_rows[0].keys()
    )

    write_csv(

        OUTPUT_MATCHED,

        matched_rows,

        matched_fields
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    write_csv(

        OUTPUT_SUMMARY,

        summary_rows,

        [
            "phenotype",
            "n_cases",
            "model",
            "mae_mm",
            "rmse_mm"
        ]
    )

    # --------------------------------------------------------
    # Save patient table
    # --------------------------------------------------------

    if patient_results:

        patient_fields = list(
            patient_results[0].keys()
        )

        write_csv(

            OUTPUT_PATIENT,

            patient_results,

            patient_fields
        )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    report = []

    report.append(
        "# STEP 15 - EXACT STEP-12 PHENOTYPE STRATIFICATION"
    )

    report.append("")

    report.append(
        "## Objective"
    )

    report.append("")

    report.append(
        "This analysis uses the exact prediction rows generated by Step 12 and stratifies them by observed HBGD tumor dynamic phenotype."
    )

    report.append("")

    report.append(
        "No model was retrained and no prediction was modified."
    )

    report.append("")

    report.append(
        "## Step 12 Prediction Structure"
    )

    report.append("")

    report.append(
        "The Step 12 file stores predictions in long format, with one row per patient-interval-model combination."
    )

    report.append("")

    report.append(
        "Columns: patient_id, current_week, future_week, actual_diameter_mm, model, predicted_diameter_mm, absolute_error_mm."
    )

    report.append("")

    report.append(
        "## Phenotype Definitions"
    )

    report.append("")

    report.append(
        "| Phenotype | HBGD rate |"
    )

    report.append(
        "|---|---|"
    )

    report.append(
        "| Growth | r > {:.4f}/day |".format(
            GROWTH_THRESHOLD
        )
    )

    report.append(
        "| Stable | {:.4f} <= r <= {:.4f}/day |".format(
            SHRINKAGE_THRESHOLD,
            GROWTH_THRESHOLD
        )
    )

    report.append(
        "| Shrinkage | r < {:.4f}/day |".format(
            SHRINKAGE_THRESHOLD
        )
    )

    report.append("")

    report.append(
        "## Test Cohort"
    )

    report.append("")

    report.append(
        "Test patients: {}".format(
            len(test_ids)
        )
    )

    report.append("")

    report.append(
        "Test prediction rows: {}".format(
            len(test_rows)
        )
    )

    report.append("")

    report.append(
        "## Phenotype-Stratified Results"
    )

    report.append("")

    report.append(
        "| Phenotype | N | Model | MAE (mm) | RMSE (mm) |"
    )

    report.append(
        "|---|---:|---|---:|---:|"
    )

    for row in summary_rows:

        report.append(
            "| {} | {} | {} | {:.4f} | {:.4f} |".format(
                row[
                    "phenotype"
                ],
                row[
                    "n_cases"
                ],
                row[
                    "model"
                ],
                row[
                    "mae_mm"
                ],
                row[
                    "rmse_mm"
                ]
            )
        )

    report.append("")

    report.append(
        "## Interpretation"
    )

    report.append("")

    report.append(
        "These results evaluate whether the predictive contribution of HBGD differs across observed tumor dynamic phenotypes."
    )

    report.append("")

    report.append(
        "The phenotype is assigned post-hoc from the observed longitudinal interval and is not used as a predictive feature."
    )

    report.append("")

    report.append(
        "Therefore, phenotype-specific performance should be interpreted as stratified predictive analysis rather than evidence that the phenotype itself was predicted."
    )

    report.append("")

    report.append(
        "No statistical significance claim is made from phenotype-specific MAE differences alone."
    )

    report.append("")

    report.append(
        "## Reproducibility"
    )

    report.append("")

    report.append(
        "This analysis directly uses Step 12 prediction outputs rather than retraining the models. It therefore avoids the numerical inconsistency identified in the earlier Step 14 implementation."
    )

    report.append("")

    report.append(
        "## Conclusion"
    )

    report.append("")

    report.append(
        "Step 15 provides the appropriate phenotype-stratified evaluation of the Step 12 experiment. The results can be used to determine whether HBGD-derived information provides differential predictive value for growth, stable, or shrinkage intervals."
    )

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report)
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 15 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nOutputs:"
    )

    print(
        OUTPUT_MATCHED
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_PATIENT
    )

    print(
        OUTPUT_REPORT
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()