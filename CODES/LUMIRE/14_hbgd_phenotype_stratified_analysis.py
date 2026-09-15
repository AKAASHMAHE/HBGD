# -*- coding: utf-8 -*-

"""
STEP 14 - HBGD PHENOTYPE-STRATIFIED PREDICTIVE ANALYSIS

Purpose
-------
Evaluate whether HBGD-derived growth information provides different
predictive value for different tumor dynamic states:

    Growth
    Stable
    Shrinkage

Primary comparison
------------------
Geometry-only
        VS
Geometry + HBGD rate

The dynamic state is assigned from the CURRENT longitudinal interval:

    Growth:
        r > +0.0005 / day

    Stable:
        -0.0010 <= r <= +0.0005 / day

    Shrinkage:
        r < -0.0010 / day

Important
---------
The state is descriptive of the observed current interval. It is NOT
used as an input feature to the predictive models.

This avoids directly feeding the target phenotype into the model.

Patient-level split:
    47 train
    7 validation
    13 test

Python:
    3.6 compatible
"""

from __future__ import print_function

import os
import csv
import math
import random

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# ============================================================
# PATHS
# ============================================================

TARGET_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"

TRAJECTORY_FILE = os.path.join(
    TARGET_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

OUTPUT_INTERVAL = os.path.join(
    TARGET_DIR,
    "step14_hbgd_phenotype_interval_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    TARGET_DIR,
    "step14_hbgd_phenotype_summary.csv"
)

OUTPUT_PATIENT = os.path.join(
    TARGET_DIR,
    "step14_hbgd_phenotype_patient_results.csv"
)

OUTPUT_REPORT = os.path.join(
    TARGET_DIR,
    "STEP14_HBGD_PHENOTYPE_STRATIFIED_REPORT.md"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

N_ESTIMATORS = 300

MAX_DEPTH = 6

MIN_SAMPLES_LEAF = 2

DAYS_PER_WEEK = 7.0

GROWTH_THRESHOLD = 0.0005

SHRINKAGE_THRESHOLD = -0.0010


# ============================================================
# CSV
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

        return 0.0

    return (
        math.log(
            d2 / d1
        )
        /
        float(days)
    )


def dynamic_state(rate):

    if rate > GROWTH_THRESHOLD:

        return "Growth"

    elif rate < SHRINKAGE_THRESHOLD:

        return "Shrinkage"

    else:

        return "Stable"


# ============================================================
# LOAD PATIENTS
# ============================================================

def load_patients():

    rows = read_csv(
        TRAJECTORY_FILE
    )

    patients = {}

    for row in rows:

        patient_id = str(
            row["patient_id"]
        )

        if patient_id not in patients:

            patients[
                patient_id
            ] = []

        week = int(
            float(
                row["week"]
            )
        )

        diameter = safe_float(
            row[
                "equivalent_diameter_mm"
            ]
        )

        volume = safe_float(
            row[
                "tumor_volume_cm3"
            ]
        )

        if (
            finite(diameter)
            and
            diameter > 0
        ):

            patients[
                patient_id
            ].append({

                "week":
                    week,

                "diameter":
                    diameter,

                "volume":
                    volume
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
# BUILD LONGITUDINAL CASES
# ============================================================

def build_cases(
    patients,
    patient_ids
):

    cases = []

    for patient_id in patient_ids:

        observations = patients[
            patient_id
        ]

        if len(observations) < 2:

            continue

        for i in range(
            1,
            len(observations)
        ):

            previous = observations[
                i - 1
            ]

            current = observations[
                i
            ]

            delta_weeks = (
                current["week"] -
                previous["week"]
            )

            delta_days = (
                delta_weeks *
                DAYS_PER_WEEK
            )

            if delta_days <= 0:

                continue

            previous_d = previous[
                "diameter"
            ]

            current_d = current[
                "diameter"
            ]

            if (
                previous_d <= 0
                or
                current_d <= 0
            ):

                continue

            # ------------------------------------------------
            # Current observed interval HBGD rate
            #
            # Used ONLY to assign phenotype.
            # It is NOT used as model input.
            # ------------------------------------------------

            current_interval_rate = (
                calculate_rate(
                    previous_d,
                    current_d,
                    delta_days
                )
            )

            state = dynamic_state(
                current_interval_rate
            )

            # ------------------------------------------------
            # Previous history
            # ------------------------------------------------

            historical_rates = []

            for j in range(
                1,
                i
            ):

                d1 = observations[
                    j - 1
                ][
                    "diameter"
                ]

                d2 = observations[
                    j
                ][
                    "diameter"
                ]

                days = (
                    observations[j]["week"]
                    -
                    observations[j - 1]["week"]
                ) * DAYS_PER_WEEK

                rate = calculate_rate(
                    d1,
                    d2,
                    days
                )

                if finite(rate):

                    historical_rates.append(
                        rate
                    )

            if historical_rates:

                recent_rate = (
                    historical_rates[-1]
                )

                mean_rate = float(
                    np.mean(
                        historical_rates
                    )
                )

                rate_std = float(
                    np.std(
                        historical_rates
                    )
                )

                min_rate = float(
                    np.min(
                        historical_rates
                    )
                )

                max_rate = float(
                    np.max(
                        historical_rates
                    )
                )

            else:

                recent_rate = 0.0

                mean_rate = 0.0

                rate_std = 0.0

                min_rate = 0.0

                max_rate = 0.0

            if i >= 2:

                previous_previous = (
                    observations[
                        i - 2
                    ]
                )

                previous_previous_d = (
                    previous_previous[
                        "diameter"
                    ]
                )

                previous_days = (
                    previous["week"]
                    -
                    previous_previous["week"]
                ) * DAYS_PER_WEEK

                rate_change = (
                    recent_rate -
                    calculate_rate(
                        previous_previous_d,
                        previous_d,
                        previous_days
                    )
                )

            else:

                rate_change = 0.0

            previous_volume = safe_float(
                previous[
                    "volume"
                ]
            )

            current_volume = safe_float(
                current[
                    "volume"
                ]
            )

            if not finite(
                previous_volume
            ):

                previous_volume = 0.0

            if not finite(
                current_volume
            ):

                current_volume = 0.0

            # ------------------------------------------------
            # Geometry
            # ------------------------------------------------

            geometry = [

                previous_d,

                previous_volume,

                current_d,

                current_d -
                previous_d
            ]

            # ------------------------------------------------
            # HBGD HISTORY FEATURES
            #
            # These describe information available BEFORE
            # the target interval.
            # ------------------------------------------------

            hbgd = [

                recent_rate,

                mean_rate,

                rate_change,

                rate_std,

                min_rate,

                max_rate
            ]

            cases.append({

                "patient_id":
                    patient_id,

                "start_week":
                    previous["week"],

                "target_week":
                    current["week"],

                "actual":
                    current_d,

                "state":
                    state,

                "current_interval_rate":
                    current_interval_rate,

                "geometry":
                    geometry,

                "hbgd":
                    hbgd
            })

    return cases


# ============================================================
# FEATURES
# ============================================================

def make_features(
    cases,
    use_hbgd
):

    X = []

    for case in cases:

        features = []

        features += case[
            "geometry"
        ]

        if use_hbgd:

            features += case[
                "hbgd"
            ]

        X.append(
            features
        )

    return np.asarray(
        X,
        dtype=float
    )


# ============================================================
# MODEL
# ============================================================

def train_model(
    X,
    y,
    seed
):

    model = RandomForestRegressor(

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        min_samples_leaf=MIN_SAMPLES_LEAF,

        random_state=seed,

        n_jobs=-1
    )

    model.fit(
        X,
        y
    )

    return model


# ============================================================
# RMSE
# ============================================================

def rmse(
    actual,
    predicted
):

    return math.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "STEP 14 - HBGD PHENOTYPE-STRATIFIED ANALYSIS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    patients = load_patients()

    patient_ids = sorted(
        patients.keys()
    )

    print(
        "\nPatients:",
        len(patient_ids)
    )

    # --------------------------------------------------------
    # SAME SPLIT AS STEPS 12/13
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    shuffled = patient_ids[:]

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

    remaining = shuffled[
        n_test:
    ]

    n_validation = int(
        round(
            len(remaining) *
            0.125
        )
    )

    if n_validation < 1:

        n_validation = 1

    validation_ids = sorted(
        remaining[
            :n_validation
        ]
    )

    train_ids = sorted(
        remaining[
            n_validation:
        ]
    )

    print(
        "\nPatient split:"
    )

    print(
        "Train:",
        len(train_ids)
    )

    print(
        "Validation:",
        len(validation_ids)
    )

    print(
        "Test:",
        len(test_ids)
    )

    # --------------------------------------------------------
    # Build cases
    # --------------------------------------------------------

    train_cases = build_cases(
        patients,
        train_ids
    )

    validation_cases = build_cases(
        patients,
        validation_ids
    )

    test_cases = build_cases(
        patients,
        test_ids
    )

    print(
        "\nCases:"
    )

    print(
        "Train:",
        len(train_cases)
    )

    print(
        "Validation:",
        len(validation_cases)
    )

    print(
        "Test:",
        len(test_cases)
    )

    # --------------------------------------------------------
    # State counts
    # --------------------------------------------------------

    print(
        "\nTest phenotype distribution:"
    )

    states = [
        "Growth",
        "Stable",
        "Shrinkage"
    ]

    for state in states:

        count = sum(
            1
            for case in test_cases
            if case[
                "state"
            ] == state
        )

        print(
            "{}: {}".format(
                state,
                count
            )
        )

    # --------------------------------------------------------
    # Feature matrices
    # --------------------------------------------------------

    X_train_geometry = make_features(
        train_cases,
        False
    )

    X_train_hbgd = make_features(
        train_cases,
        True
    )

    X_validation_geometry = make_features(
        validation_cases,
        False
    )

    X_validation_hbgd = make_features(
        validation_cases,
        True
    )

    X_test_geometry = make_features(
        test_cases,
        False
    )

    X_test_hbgd = make_features(
        test_cases,
        True
    )

    y_train = np.asarray(
        [
            case["actual"]
            for case in train_cases
        ],
        dtype=float
    )

    y_validation = np.asarray(
        [
            case["actual"]
            for case in validation_cases
        ],
        dtype=float
    )

    y_test = np.asarray(
        [
            case["actual"]
            for case in test_cases
        ],
        dtype=float
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    geometry_model = train_model(
        X_train_geometry,
        y_train,
        RANDOM_SEED
    )

    hbgd_model = train_model(
        X_train_hbgd,
        y_train,
        RANDOM_SEED + 1
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    geometry_validation = (
        geometry_model.predict(
            X_validation_geometry
        )
    )

    hbgd_validation = (
        hbgd_model.predict(
            X_validation_hbgd
        )
    )

    geometry_test = (
        geometry_model.predict(
            X_test_geometry
        )
    )

    hbgd_test = (
        hbgd_model.predict(
            X_test_hbgd
        )
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall_geometry_mae = (
        mean_absolute_error(
            y_test,
            geometry_test
        )
    )

    overall_hbgd_mae = (
        mean_absolute_error(
            y_test,
            hbgd_test
        )
    )

    overall_geometry_rmse = rmse(
        y_test,
        geometry_test
    )

    overall_hbgd_rmse = rmse(
        y_test,
        hbgd_test
    )

    print(
        "\nOverall test:"
    )

    print(
        "Geometry MAE:",
        "{:.4f}".format(
            overall_geometry_mae
        )
    )

    print(
        "Geometry + HBGD MAE:",
        "{:.4f}".format(
            overall_hbgd_mae
        )
    )

    # --------------------------------------------------------
    # Phenotype stratification
    # --------------------------------------------------------

    interval_results = []

    summary_rows = []

    for state in states:

        indices = [

            i

            for i, case in enumerate(
                test_cases
            )

            if case[
                "state"
            ] == state
        ]

        if len(indices) == 0:

            print(
                "\n{}: no cases".format(
                    state
                )
            )

            continue

        actual = np.asarray(
            [
                y_test[i]
                for i in indices
            ],
            dtype=float
        )

        geometry_prediction = np.asarray(
            [
                geometry_test[i]
                for i in indices
            ],
            dtype=float
        )

        hbgd_prediction = np.asarray(
            [
                hbgd_test[i]
                for i in indices
            ],
            dtype=float
        )

        geometry_mae = (
            mean_absolute_error(
                actual,
                geometry_prediction
            )
        )

        hbgd_mae = (
            mean_absolute_error(
                actual,
                hbgd_prediction
            )
        )

        geometry_rmse = rmse(
            actual,
            geometry_prediction
        )

        hbgd_rmse = rmse(
            actual,
            hbgd_prediction
        )

        improvement_mm = (
            geometry_mae -
            hbgd_mae
        )

        if geometry_mae != 0:

            improvement_percent = (
                improvement_mm /
                geometry_mae
            ) * 100.0

        else:

            improvement_percent = 0.0

        print(
            "\n{} phenotype".format(
                state
            )
        )

        print(
            "N:",
            len(indices)
        )

        print(
            "Geometry MAE:",
            "{:.4f} mm".format(
                geometry_mae
            )
        )

        print(
            "HBGD MAE:",
            "{:.4f} mm".format(
                hbgd_mae
            )
        )

        print(
            "Improvement:",
            "{:.4f} mm ({:.2f}%)".format(
                improvement_mm,
                improvement_percent
            )
        )

        summary_rows.append({

            "phenotype":
                state,

            "n_cases":
                len(indices),

            "geometry_mae_mm":
                geometry_mae,

            "hbgd_mae_mm":
                hbgd_mae,

            "geometry_rmse_mm":
                geometry_rmse,

            "hbgd_rmse_mm":
                hbgd_rmse,

            "improvement_mm":
                improvement_mm,

            "improvement_percent":
                improvement_percent
        })

        # ----------------------------------------------------
        # Individual interval records
        # ----------------------------------------------------

        for i in indices:

            case = test_cases[
                i
            ]

            geometry_error = abs(
                y_test[i] -
                geometry_test[i]
            )

            hbgd_error = abs(
                y_test[i] -
                hbgd_test[i]
            )

            interval_results.append({

                "patient_id":
                    case[
                        "patient_id"
                    ],

                "start_week":
                    case[
                        "start_week"
                    ],

                "target_week":
                    case[
                        "target_week"
                    ],

                "phenotype":
                    state,

                "observed_current_interval_rate":
                    case[
                        "current_interval_rate"
                    ],

                "actual_diameter_mm":
                    y_test[i],

                "geometry_prediction_mm":
                    geometry_test[i],

                "hbgd_prediction_mm":
                    hbgd_test[i],

                "geometry_error_mm":
                    geometry_error,

                "hbgd_error_mm":
                    hbgd_error,

                "hbgd_improvement_mm":
                    geometry_error -
                    hbgd_error
            })

    # --------------------------------------------------------
    # Patient-level phenotype analysis
    # --------------------------------------------------------

    patient_results = []

    for patient_id in test_ids:

        patient_cases = [

            (
                i,
                case
            )

            for i, case in enumerate(
                test_cases
            )

            if case[
                "patient_id"
            ] == patient_id
        ]

        if not patient_cases:

            continue

        states_for_patient = [

            case[
                "state"
            ]

            for i, case in patient_cases
        ]

        patient_actual = np.asarray(
            [
                y_test[i]
                for i, case in patient_cases
            ],
            dtype=float
        )

        patient_geometry = np.asarray(
            [
                geometry_test[i]
                for i, case in patient_cases
            ],
            dtype=float
        )

        patient_hbgd = np.asarray(
            [
                hbgd_test[i]
                for i, case in patient_cases
            ],
            dtype=float
        )

        geometry_mae = (
            mean_absolute_error(
                patient_actual,
                patient_geometry
            )
        )

        hbgd_mae = (
            mean_absolute_error(
                patient_actual,
                patient_hbgd
            )
        )

        improvement = (
            geometry_mae -
            hbgd_mae
        )

        # Dominant phenotype
        counts = {}

        for state in states_for_patient:

            counts[state] = (
                counts.get(
                    state,
                    0
                )
                + 1
            )

        dominant_state = max(
            counts,
            key=counts.get
        )

        patient_results.append({

            "patient_id":
                patient_id,

            "n_cases":
                len(patient_cases),

            "dominant_phenotype":
                dominant_state,

            "geometry_mae_mm":
                geometry_mae,

            "hbgd_mae_mm":
                hbgd_mae,

            "improvement_mm":
                improvement
        })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    write_csv(

        OUTPUT_INTERVAL,

        interval_results,

        [
            "patient_id",
            "start_week",
            "target_week",
            "phenotype",
            "observed_current_interval_rate",
            "actual_diameter_mm",
            "geometry_prediction_mm",
            "hbgd_prediction_mm",
            "geometry_error_mm",
            "hbgd_error_mm",
            "hbgd_improvement_mm"
        ]
    )

    write_csv(

        OUTPUT_SUMMARY,

        summary_rows,

        [
            "phenotype",
            "n_cases",
            "geometry_mae_mm",
            "hbgd_mae_mm",
            "geometry_rmse_mm",
            "hbgd_rmse_mm",
            "improvement_mm",
            "improvement_percent"
        ]
    )

    write_csv(

        OUTPUT_PATIENT,

        patient_results,

        [
            "patient_id",
            "n_cases",
            "dominant_phenotype",
            "geometry_mae_mm",
            "hbgd_mae_mm",
            "improvement_mm"
        ]
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = []

    report.append(
        "# STEP 14 - HBGD PHENOTYPE-STRATIFIED PREDICTIVE ANALYSIS"
    )

    report.append("")

    report.append(
        "## Objective"
    )

    report.append("")

    report.append(
        "This analysis evaluates whether HBGD-derived growth information has different predictive value across tumor dynamic phenotypes."
    )

    report.append("")

    report.append(
        "The phenotypes are Growth, Stable, and Shrinkage."
    )

    report.append("")

    report.append(
        "## Phenotype Definitions"
    )

    report.append("")

    report.append(
        "| Phenotype | HBGD rate criterion |"
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
        "## Test Performance by Phenotype"
    )

    report.append("")

    report.append(
        "| Phenotype | N | Geometry MAE | Geometry + HBGD MAE | Improvement | Improvement % |"
    )

    report.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for row in summary_rows:

        report.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.2f}% |".format(
                row["phenotype"],
                row["n_cases"],
                row["geometry_mae_mm"],
                row["hbgd_mae_mm"],
                row["improvement_mm"],
                row["improvement_percent"]
            )
        )

    report.append("")

    report.append(
        "## Overall Test Performance"
    )

    report.append("")

    report.append(
        "Geometry-only MAE: {:.4f} mm".format(
            overall_geometry_mae
        )
    )

    report.append("")

    report.append(
        "Geometry + HBGD MAE: {:.4f} mm".format(
            overall_hbgd_mae
        )
    )

    report.append("")

    report.append(
        "Geometry-only RMSE: {:.4f} mm".format(
            overall_geometry_rmse
        )
    )

    report.append("")

    report.append(
        "Geometry + HBGD RMSE: {:.4f} mm".format(
            overall_hbgd_rmse
        )
    )

    report.append("")

    # --------------------------------------------------------
    # Automatic interpretation
    # --------------------------------------------------------

    if summary_rows:

        best_row = max(
            summary_rows,
            key=lambda row:
                row[
                    "improvement_mm"
                ]
        )

        worst_row = min(
            summary_rows,
            key=lambda row:
                row[
                    "improvement_mm"
                ]
        )

        report.append(
            "## Interpretation"
        )

        report.append("")

        report.append(
            "The phenotype with the largest positive HBGD improvement was {} ({:.4f} mm, {:.2f}%).".format(
                best_row[
                    "phenotype"
                ],
                best_row[
                    "improvement_mm"
                ],
                best_row[
                    "improvement_percent"
                ]
            )
        )

        report.append("")

        report.append(
            "The phenotype with the weakest relative result was {} ({:.4f} mm, {:.2f}%).".format(
                worst_row[
                    "phenotype"
                ],
                worst_row[
                    "improvement_mm"
                ],
                worst_row[
                    "improvement_percent"
                ]
            )
        )

        report.append("")

    report.append(
        "## Scientific Caution"
    )

    report.append("")

    report.append(
        "Phenotype-stratified results are descriptive because the held-out test cohort contains only a limited number of cases in each phenotype."
    )

    report.append("")

    report.append(
        "The current analysis does not establish statistical significance of phenotype-specific HBGD benefit."
    )

    report.append("")

    report.append(
        "The observed phenotype should therefore be interpreted as an analysis of where HBGD-derived temporal information may be most useful, rather than as proof of phenotype-specific clinical efficacy."
    )

    report.append("")

    report.append(
        "## Conclusion"
    )

    report.append("")

    report.append(
        "Step 14 determines whether the incremental predictive contribution of HBGD is heterogeneous across tumor dynamic phenotypes. A stronger benefit in the Growth phenotype would support the hypothesis that HBGD-derived temporal information is particularly useful for modeling actively progressing tumors, while a weak or negative result would indicate that current HBGD features do not provide phenotype-specific predictive advantage."
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
    # Final
    # --------------------------------------------------------

    print("\n")

    print("=" * 70)

    print(
        "STEP 14 COMPLETE"
    )

    print("=" * 70)

    print("\nOUTPUTS:")

    print(
        OUTPUT_INTERVAL
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

    print("=" * 70)


if __name__ == "__main__":

    main()