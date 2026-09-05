# -*- coding: utf-8 -*-

"""
4_lumiere_hbgd_forward_prediction.py

STEP 4:
    Within-LUMIERE longitudinal forward prediction.

Goal:
    Test whether individualized HBGD growth rates can predict
    a future tumor measurement.

Prediction:
    D_pred = D_current * exp(r * delta_t)

where:
    r = ln(D_current / D_previous) / delta_t

Baselines:
    1. No-change
    2. Global exponential growth
    3. Individualized HBGD

IMPORTANT:
    - Patient-level split.
    - No patient appears in both train and test.
    - Week difference is converted to days using 7 days/week.
    - This temporal assumption must later be verified against
      official LUMIERE documentation.
    - Python 3.6 compatible.
"""

from __future__ import print_function

import os
import csv
import math
import random


# ============================================================
# PATHS
# ============================================================

LUMIERE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"

TRAJECTORY_CSV = os.path.join(
    LUMIERE_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

HBGD_CSV = os.path.join(
    LUMIERE_DIR,
    "lumiere_hbgd_growth_features.csv"
)

RESULTS_CSV = os.path.join(
    LUMIERE_DIR,
    "lumiere_forward_prediction_results.csv"
)

PATIENT_RESULTS_CSV = os.path.join(
    LUMIERE_DIR,
    "lumiere_forward_prediction_patient_summary.csv"
)

REPORT_PATH = os.path.join(
    LUMIERE_DIR,
    "LUMIERE_FORWARD_PREDICTION_REPORT.md"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TEST_FRACTION = 0.20

DAYS_PER_WEEK = 7.0


# ============================================================
# HELPERS
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


def mean(values):

    if not values:
        return float("nan")

    return (
        sum(values) /
        float(len(values))
    )


def median(values):

    if not values:
        return float("nan")

    values = sorted(values)

    n = len(values)

    middle = n // 2

    if n % 2 == 0:

        return (
            values[middle - 1] +
            values[middle]
        ) / 2.0

    return values[middle]


def mae(actual, predicted):

    errors = []

    for a, p in zip(
        actual,
        predicted
    ):

        if (
            finite(a)
            and
            finite(p)
        ):

            errors.append(
                abs(a - p)
            )

    return mean(errors)


def rmse(actual, predicted):

    errors = []

    for a, p in zip(
        actual,
        predicted
    ):

        if (
            finite(a)
            and
            finite(p)
        ):

            errors.append(
                (a - p) ** 2
            )

    if not errors:
        return float("nan")

    return math.sqrt(
        mean(errors)
    )


def mape(actual, predicted):

    errors = []

    for a, p in zip(
        actual,
        predicted
    ):

        if (
            finite(a)
            and
            finite(p)
            and
            a != 0
        ):

            errors.append(
                abs(
                    (a - p) / a
                ) * 100.0
            )

    return mean(errors)


def median_absolute_error(
    actual,
    predicted
):

    errors = []

    for a, p in zip(
        actual,
        predicted
    ):

        if (
            finite(a)
            and
            finite(p)
        ):

            errors.append(
                abs(a - p)
            )

    return median(errors)


def calculate_r(
    d_previous,
    d_current,
    delta_weeks
):

    if (
        d_previous <= 0
        or
        d_current <= 0
        or
        delta_weeks <= 0
    ):

        return float("nan")

    delta_days = (
        delta_weeks *
        DAYS_PER_WEEK
    )

    return (
        math.log(
            d_current /
            d_previous
        )
        /
        delta_days
    )


def predict_diameter(
    current_diameter,
    r,
    delta_weeks
):

    if (
        current_diameter <= 0
        or
        not finite(r)
        or
        delta_weeks <= 0
    ):

        return float("nan")

    delta_days = (
        delta_weeks *
        DAYS_PER_WEEK
    )

    try:

        return (
            current_diameter *
            math.exp(
                r * delta_days
            )
        )

    except OverflowError:

        return float("nan")


def diameter_to_volume(
    diameter_mm
):

    if (
        not finite(
            diameter_mm
        )
        or
        diameter_mm <= 0
    ):

        return float("nan")

    volume_mm3 = (
        math.pi /
        6.0
    ) * (
        diameter_mm ** 3
    )

    return (
        volume_mm3 /
        1000.0
    )


# ============================================================
# LOAD TRAJECTORIES
# ============================================================

def load_trajectories():

    if not os.path.exists(
        TRAJECTORY_CSV
    ):

        raise IOError(
            "Trajectory CSV not found: {}".format(
                TRAJECTORY_CSV
            )
        )

    patients = {}

    with open(
        TRAJECTORY_CSV,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            patient_id = row[
                "patient_id"
            ]

            observation = {

                "patient_id":
                    patient_id,

                "week":
                    int(row["week"]),

                "diameter":
                    safe_float(
                        row[
                            "equivalent_diameter_mm"
                        ]
                    ),

                "volume":
                    safe_float(
                        row[
                            "tumor_volume_cm3"
                        ]
                    )
            }

            if patient_id not in patients:

                patients[
                    patient_id
                ] = []

            patients[
                patient_id
            ].append(
                observation
            )

    # Sort by time
    for patient_id in patients:

        patients[
            patient_id
        ].sort(
            key=lambda x: x["week"]
        )

    return patients


# ============================================================
# CREATE PATIENT-LEVEL SPLIT
# ============================================================

def split_patients(
    patients
):

    patient_ids = sorted(
        patients.keys()
    )

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        patient_ids
    )

    n_patients = len(
        patient_ids
    )

    n_test = int(
        round(
            n_patients *
            TEST_FRACTION
        )
    )

    if n_test < 1:

        n_test = 1

    if n_test >= n_patients:

        n_test = n_patients - 1

    test_patients = sorted(
        patient_ids[
            :n_test
        ]
    )

    train_patients = sorted(
        patient_ids[
            n_test:
        ]
    )

    return (
        train_patients,
        test_patients
    )


# ============================================================
# GLOBAL EXPONENTIAL RATE
# ============================================================

def calculate_global_rate(
    patients,
    patient_ids
):

    numerator = 0.0

    denominator = 0.0

    interval_count = 0

    for patient_id in patient_ids:

        observations = patients[
            patient_id
        ]

        for i in range(
            len(observations) - 1
        ):

            first = observations[i]

            second = observations[i + 1]

            d1 = first[
                "diameter"
            ]

            d2 = second[
                "diameter"
            ]

            delta_weeks = (
                second["week"] -
                first["week"]
            )

            if (
                d1 <= 0
                or
                d2 <= 0
                or
                delta_weeks <= 0
            ):

                continue

            delta_days = (
                delta_weeks *
                DAYS_PER_WEEK
            )

            numerator += math.log(
                d2 / d1
            )

            denominator += (
                delta_days
            )

            interval_count += 1

    if denominator <= 0:

        return float("nan"), 0

    return (
        numerator /
        denominator,
        interval_count
    )


# ============================================================
# CREATE PREDICTION CASES
# ============================================================

def create_prediction_cases(
    patients,
    patient_ids,
    global_rate
):

    results = []

    """
    For each three-point sequence:

        t0 -> t1 -> t2

    use t0 and t1 to estimate r,
    then predict t2.

    This ensures that t2 is genuinely future
    relative to the HBGD rate calculation.
    """

    for patient_id in patient_ids:

        observations = patients[
            patient_id
        ]

        if len(
            observations
        ) < 3:

            continue

        for i in range(
            1,
            len(observations) - 1
        ):

            previous = observations[
                i - 1
            ]

            current = observations[
                i
            ]

            future = observations[
                i + 1
            ]

            d_previous = previous[
                "diameter"
            ]

            d_current = current[
                "diameter"
            ]

            d_actual = future[
                "diameter"
            ]

            delta_train_weeks = (
                current["week"] -
                previous["week"]
            )

            delta_future_weeks = (
                future["week"] -
                current["week"]
            )

            if (
                d_previous <= 0
                or
                d_current <= 0
                or
                d_actual <= 0
                or
                delta_train_weeks <= 0
                or
                delta_future_weeks <= 0
            ):

                continue

            # ----------------------------------------------
            # Individualized HBGD rate
            # ----------------------------------------------

            individual_rate = calculate_r(
                d_previous,
                d_current,
                delta_train_weeks
            )

            # ----------------------------------------------
            # Predictions
            # ----------------------------------------------

            no_change_prediction = (
                d_current
            )

            global_prediction = predict_diameter(
                d_current,
                global_rate,
                delta_future_weeks
            )

            individual_prediction = predict_diameter(
                d_current,
                individual_rate,
                delta_future_weeks
            )

            # ----------------------------------------------
            # Errors
            # ----------------------------------------------

            no_change_error = abs(
                d_actual -
                no_change_prediction
            )

            global_error = abs(
                d_actual -
                global_prediction
            )

            individual_error = abs(
                d_actual -
                individual_prediction
            )

            # ----------------------------------------------
            # Improvement vs baselines
            # ----------------------------------------------

            improvement_vs_no_change = (
                (
                    no_change_error -
                    individual_error
                )
                /
                no_change_error
                * 100.0
                if no_change_error > 0
                else 0.0
            )

            improvement_vs_global = (
                (
                    global_error -
                    individual_error
                )
                /
                global_error
                * 100.0
                if global_error > 0
                else 0.0
            )

            results.append({

                "patient_id":
                    patient_id,

                "previous_week":
                    previous["week"],

                "current_week":
                    current["week"],

                "future_week":
                    future["week"],

                "delta_train_weeks":
                    delta_train_weeks,

                "delta_future_weeks":
                    delta_future_weeks,

                "previous_diameter_mm":
                    d_previous,

                "current_diameter_mm":
                    d_current,

                "actual_future_diameter_mm":
                    d_actual,

                "individual_r_per_day":
                    individual_rate,

                "global_r_per_day":
                    global_rate,

                "no_change_prediction_mm":
                    no_change_prediction,

                "global_prediction_mm":
                    global_prediction,

                "individual_hbgd_prediction_mm":
                    individual_prediction,

                "no_change_error_mm":
                    no_change_error,

                "global_error_mm":
                    global_error,

                "individual_hbgd_error_mm":
                    individual_error,

                "improvement_vs_no_change_percent":
                    improvement_vs_no_change,

                "improvement_vs_global_percent":
                    improvement_vs_global,

                "actual_future_volume_cm3":
                    future["volume"],

                "predicted_hbgd_volume_cm3":
                    diameter_to_volume(
                        individual_prediction
                    )

            })

    return results


# ============================================================
# PATIENT SUMMARY
# ============================================================

def create_patient_summary(
    results
):

    patients = {}

    for row in results:

        patient_id = row[
            "patient_id"
        ]

        if patient_id not in patients:

            patients[
                patient_id
            ] = []

        patients[
            patient_id
        ].append(
            row
        )

    summaries = []

    for patient_id in sorted(
        patients.keys()
    ):

        rows = patients[
            patient_id
        ]

        actual = [
            r[
                "actual_future_diameter_mm"
            ]
            for r in rows
        ]

        no_change = [
            r[
                "no_change_prediction_mm"
            ]
            for r in rows
        ]

        global_prediction = [
            r[
                "global_prediction_mm"
            ]
            for r in rows
        ]

        hbgd_prediction = [
            r[
                "individual_hbgd_prediction_mm"
            ]
            for r in rows
        ]

        summaries.append({

            "patient_id":
                patient_id,

            "prediction_cases":
                len(rows),

            "no_change_mae_mm":
                mae(
                    actual,
                    no_change
                ),

            "global_exponential_mae_mm":
                mae(
                    actual,
                    global_prediction
                ),

            "individual_hbgd_mae_mm":
                mae(
                    actual,
                    hbgd_prediction
                ),

            "no_change_rmse_mm":
                rmse(
                    actual,
                    no_change
                ),

            "global_exponential_rmse_mm":
                rmse(
                    actual,
                    global_prediction
                ),

            "individual_hbgd_rmse_mm":
                rmse(
                    actual,
                    hbgd_prediction
                )

        })

    return summaries


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    path,
    rows
):

    if not rows:

        return

    columns = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )


# ============================================================
# REPORT
# ============================================================

def create_report(
    patients,
    train_patients,
    test_patients,
    global_rate,
    global_intervals,
    test_results,
    patient_summary
):

    actual = [
        r[
            "actual_future_diameter_mm"
        ]
        for r in test_results
    ]

    no_change = [
        r[
            "no_change_prediction_mm"
        ]
        for r in test_results
    ]

    global_prediction = [
        r[
            "global_prediction_mm"
        ]
        for r in test_results
    ]

    hbgd_prediction = [
        r[
            "individual_hbgd_prediction_mm"
        ]
        for r in test_results
    ]

    no_change_mae = mae(
        actual,
        no_change
    )

    global_mae = mae(
        actual,
        global_prediction
    )

    hbgd_mae = mae(
        actual,
        hbgd_prediction
    )

    no_change_rmse = rmse(
        actual,
        no_change
    )

    global_rmse = rmse(
        actual,
        global_prediction
    )

    hbgd_rmse = rmse(
        actual,
        hbgd_prediction
    )

    no_change_mape = mape(
        actual,
        no_change
    )

    global_mape = mape(
        actual,
        global_prediction
    )

    hbgd_mape = mape(
        actual,
        hbgd_prediction
    )

    no_change_median = median_absolute_error(
        actual,
        no_change
    )

    global_median = median_absolute_error(
        actual,
        global_prediction
    )

    hbgd_median = median_absolute_error(
        actual,
        hbgd_prediction
    )

    if no_change_mae > 0:

        improvement_no_change = (
            (
                no_change_mae -
                hbgd_mae
            )
            /
            no_change_mae
        ) * 100.0

    else:

        improvement_no_change = 0.0

    if global_mae > 0:

        improvement_global = (
            (
                global_mae -
                hbgd_mae
            )
            /
            global_mae
        ) * 100.0

    else:

        improvement_global = 0.0

    # Number of cases where HBGD wins
    hbgd_better_no_change = 0

    hbgd_better_global = 0

    for row in test_results:

        if (
            row[
                "individual_hbgd_error_mm"
            ]
            <
            row[
                "no_change_error_mm"
            ]
        ):

            hbgd_better_no_change += 1

        if (
            row[
                "individual_hbgd_error_mm"
            ]
            <
            row[
                "global_error_mm"
            ]
        ):

            hbgd_better_global += 1

    n_cases = len(
        test_results
    )

    if n_cases > 0:

        win_no_change_pct = (
            hbgd_better_no_change /
            float(n_cases)
        ) * 100.0

        win_global_pct = (
            hbgd_better_global /
            float(n_cases)
        ) * 100.0

    else:

        win_no_change_pct = 0.0
        win_global_pct = 0.0

    report = []

    report.append(
        "# LUMIERE HBGD FORWARD PREDICTION"
    )

    report.append("")

    report.append(
        "## 1. Objective"
    )

    report.append("")

    report.append(
        "This experiment evaluates whether an individualized HBGD growth rate calculated from previous longitudinal tumor observations can predict the next observed tumor diameter."
    )

    report.append("")

    report.append(
        "The evaluation is performed using a patient-level train/test split to prevent observations from the same patient appearing in both sets."
    )

    report.append("")

    report.append(
        "## 2. Dataset"
    )

    report.append("")

    report.append(
        "| Metric | Value |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        "| Total patients | {} |".format(
            len(patients)
        )
    )

    report.append(
        "| Training patients | {} |".format(
            len(train_patients)
        )
    )

    report.append(
        "| Test patients | {} |".format(
            len(test_patients)
        )
    )

    report.append(
        "| Test prediction cases | {} |".format(
            n_cases
        )
    )

    report.append("")

    report.append(
        "## 3. Patient-Level Split"
    )

    report.append("")

    report.append(
        "Random seed: {}".format(
            RANDOM_SEED
        )
    )

    report.append("")

    report.append(
        "### Training patients"
    )

    report.append("")

    report.append(
        ", ".join(
            train_patients
        )
    )

    report.append("")

    report.append(
        "### Test patients"
    )

    report.append("")

    report.append(
        ", ".join(
            test_patients
        )
    )

    report.append("")

    report.append(
        "## 4. Global HBGD Rate"
    )

    report.append("")

    report.append(
        "The global exponential rate is estimated only from training patients."
    )

    report.append("")

    report.append(
        "```text"
    )

    report.append(
        "r_global = sum[ln(D2/D1)] / sum[Δt]"
    )

    report.append(
        "```"
    )

    report.append("")

    report.append(
        "Global rate: {:.8f} per day".format(
            global_rate
        )
    )

    report.append(
        "Training intervals used: {}".format(
            global_intervals
        )
    )

    report.append("")

    report.append(
        "## 5. Forward Prediction Results"
    )

    report.append("")

    report.append(
        "| Model | MAE (mm) | RMSE (mm) | Median AE (mm) | MAPE (%) |"
    )

    report.append(
        "|---|---:|---:|---:|---:|"
    )

    report.append(
        "| No-change | {:.4f} | {:.4f} | {:.4f} | {:.2f} |".format(
            no_change_mae,
            no_change_rmse,
            no_change_median,
            no_change_mape
        )
    )

    report.append(
        "| Global exponential | {:.4f} | {:.4f} | {:.4f} | {:.2f} |".format(
            global_mae,
            global_rmse,
            global_median,
            global_mape
        )
    )

    report.append(
        "| Individualized HBGD | {:.4f} | {:.4f} | {:.4f} | {:.2f} |".format(
            hbgd_mae,
            hbgd_rmse,
            hbgd_median,
            hbgd_mape
        )
    )

    report.append("")

    report.append(
        "## 6. HBGD Improvement"
    )

    report.append("")

    report.append(
        "| Comparison | Improvement from HBGD |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        "| vs No-change | {:.2f}% |".format(
            improvement_no_change
        )
    )

    report.append(
        "| vs Global exponential | {:.2f}% |".format(
            improvement_global
        )
    )

    report.append("")

    report.append(
        "HBGD produced lower absolute error than no-change in {} / {} test cases ({:.2f}%).".format(
            hbgd_better_no_change,
            n_cases,
            win_no_change_pct
        )
    )

    report.append("")

    report.append(
        "HBGD produced lower absolute error than global exponential prediction in {} / {} test cases ({:.2f}%).".format(
            hbgd_better_global,
            n_cases,
            win_global_pct
        )
    )

    report.append("")

    report.append(
        "## 7. Interpretation"
    )

    report.append("")

    if (
        hbgd_mae <
        no_change_mae
        and
        hbgd_mae <
        global_mae
    ):

        report.append(
            "**Individualized HBGD achieved the lowest test-set MAE among the evaluated models.** This provides evidence that patient-specific longitudinal growth information improves one-step-ahead tumor diameter prediction within the LUMIERE dataset."
        )

    elif (
        hbgd_mae <
        no_change_mae
    ):

        report.append(
            "**Individualized HBGD improved over the no-change baseline but did not outperform the global exponential model.** The result supports some predictive value of longitudinal growth information, but does not establish individualized HBGD as the strongest model in this experiment."
        )

    else:

        report.append(
            "**Individualized HBGD did not outperform the no-change baseline on the test set.** The HBGD formulation therefore requires further analysis before claiming predictive superiority."
        )

    report.append("")

    report.append(
        "## 8. Important Limitation"
    )

    report.append("")

    report.append(
        "The temporal conversion currently assumes one LUMIERE week identifier corresponds to seven days. This must be verified against official dataset documentation before interpreting the estimated rates as biologically calibrated growth rates."
    )

    report.append("")

    report.append(
        "The experiment is a one-step-ahead prediction evaluation. It does not yet establish long-horizon forecasting performance."
    )

    report.append("")

    report.append(
        "The individualized rate is estimated from the immediately preceding pair of observations. More robust versions can later use multiple historical observations, regression-based rate estimation, dynamic state information, or learned representations."
    )

    report.append("")

    report.append(
        "## 9. Next Stage"
    )

    report.append("")

    report.append(
        "If within-LUMIERE prediction is successful, the next stage is cross-cancer transfer: learn growth-related representations from the existing HBGD source-cancer dataset and evaluate whether those representations improve prediction on LUMIERE GBM patients."
    )

    report.append("")

    report.append(
        "This transfer experiment must also use patient-level separation and must not use future LUMIERE measurements during model training."
    )

    return "\n".join(
        report
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "STEP 4 - LUMIERE HBGD FORWARD PREDICTION"
    )
    print("=" * 70)

    if not os.path.exists(
        LUMIERE_DIR
    ):

        os.makedirs(
            LUMIERE_DIR
        )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\nLoading longitudinal trajectories..."
    )

    patients = load_trajectories()

    print(
        "Patients loaded:",
        len(patients)
    )

    # --------------------------------------------------------
    # Patient split
    # --------------------------------------------------------

    (
        train_patients,
        test_patients
    ) = split_patients(
        patients
    )

    print(
        "\nTraining patients:",
        len(train_patients)
    )

    print(
        "Test patients:",
        len(test_patients)
    )

    # --------------------------------------------------------
    # Global rate from training only
    # --------------------------------------------------------

    (
        global_rate,
        global_intervals
    ) = calculate_global_rate(
        patients,
        train_patients
    )

    print(
        "\nGlobal training HBGD rate:",
        "{:.8f}".format(
            global_rate
        ),
        "per day"
    )

    print(
        "Training intervals:",
        global_intervals
    )

    # --------------------------------------------------------
    # Create test prediction cases
    # --------------------------------------------------------

    print(
        "\nCreating one-step-ahead test cases..."
    )

    test_results = create_prediction_cases(
        patients,
        test_patients,
        global_rate
    )

    print(
        "Test prediction cases:",
        len(test_results)
    )

    if len(test_results) == 0:

        print(
            "\nERROR: No valid prediction cases."
        )

        return

    # --------------------------------------------------------
    # Patient summaries
    # --------------------------------------------------------

    patient_summary = create_patient_summary(
        test_results
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    save_csv(
        RESULTS_CSV,
        test_results
    )

    save_csv(
        PATIENT_RESULTS_CSV,
        patient_summary
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = create_report(
        patients,
        train_patients,
        test_patients,
        global_rate,
        global_intervals,
        test_results,
        patient_summary
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            report
        )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    actual = [
        r[
            "actual_future_diameter_mm"
        ]
        for r in test_results
    ]

    no_change = [
        r[
            "no_change_prediction_mm"
        ]
        for r in test_results
    ]

    global_prediction = [
        r[
            "global_prediction_mm"
        ]
        for r in test_results
    ]

    hbgd_prediction = [
        r[
            "individual_hbgd_prediction_mm"
        ]
        for r in test_results
    ]

    no_change_mae = mae(
        actual,
        no_change
    )

    global_mae = mae(
        actual,
        global_prediction
    )

    hbgd_mae = mae(
        actual,
        hbgd_prediction
    )

    no_change_rmse = rmse(
        actual,
        no_change
    )

    global_rmse = rmse(
        actual,
        global_prediction
    )

    hbgd_rmse = rmse(
        actual,
        hbgd_prediction
    )

    print("\n")
    print("=" * 70)
    print("STEP 4 COMPLETE")
    print("=" * 70)

    print(
        "\nTest prediction cases:",
        len(test_results)
    )

    print(
        "\nMAE:"
    )

    print(
        "No-change:",
        "{:.4f} mm".format(
            no_change_mae
        )
    )

    print(
        "Global exponential:",
        "{:.4f} mm".format(
            global_mae
        )
    )

    print(
        "Individualized HBGD:",
        "{:.4f} mm".format(
            hbgd_mae
        )
    )

    print(
        "\nRMSE:"
    )

    print(
        "No-change:",
        "{:.4f} mm".format(
            no_change_rmse
        )
    )

    print(
        "Global exponential:",
        "{:.4f} mm".format(
            global_rmse
        )
    )

    print(
        "Individualized HBGD:",
        "{:.4f} mm".format(
            hbgd_rmse
        )
    )

    if (
        hbgd_mae <
        no_change_mae
        and
        hbgd_mae <
        global_mae
    ):

        print(
            "\nRESULT: Individualized HBGD is the best model."
        )

    elif (
        hbgd_mae <
        no_change_mae
    ):

        print(
            "\nRESULT: HBGD beats no-change but not global exponential."
        )

    else:

        print(
            "\nRESULT: HBGD does not beat no-change."
        )

    print(
        "\nOUTPUT:"
    )

    print(
        RESULTS_CSV
    )

    print(
        PATIENT_RESULTS_CSV
    )

    print(
        REPORT_PATH
    )

    print("=" * 70)


if __name__ == "__main__":

    main()