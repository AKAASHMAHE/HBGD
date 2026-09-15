# -*- coding: utf-8 -*-

"""
STEP 13 - PATIENT-LEVEL HBGD ROBUSTNESS AND STATISTICAL TESTING

Purpose
-------
Determine whether the small HBGD improvement observed in Step 12
is consistent across patients or driven by a small number of cases.

Primary comparison
------------------
Geometry-only
        VS
Geometry + HBGD rate

Patient-level analysis
----------------------
For every held-out test patient:

    patient MAE - Geometry
    patient MAE - Geometry + HBGD
    improvement

Then calculate:

    mean improvement
    median improvement
    number improved
    number worsened
    bootstrap 95% CI
    paired permutation test
    Wilcoxon signed-rank test if scipy is available
    Cohen-style standardized paired effect size

Important
---------
The 13 test patients remain untouched during model fitting.

Python
------
Compatible with Python 3.6.

Output
------
D:\\VIT\\PROJECT\\NLP\\PATENT\\DATA\\LUMIRE\\
"""

from __future__ import print_function

import os
import csv
import math
import random

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# PATHS
# ============================================================

TARGET_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"

TARGET_TRAJECTORY = os.path.join(
    TARGET_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

OUTPUT_PATIENT = os.path.join(
    TARGET_DIR,
    "step13_patient_level_hbgd_results.csv"
)

OUTPUT_BOOTSTRAP = os.path.join(
    TARGET_DIR,
    "step13_hbgd_bootstrap_results.csv"
)

OUTPUT_PERMUTATION = os.path.join(
    TARGET_DIR,
    "step13_hbgd_permutation_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    TARGET_DIR,
    "step13_hbgd_statistical_summary.csv"
)

OUTPUT_REPORT = os.path.join(
    TARGET_DIR,
    "STEP13_HBGD_PATIENT_LEVEL_ROBUSTNESS_REPORT.md"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

N_ESTIMATORS = 300

MAX_DEPTH = 6

MIN_SAMPLES_LEAF = 2

DAYS_PER_WEEK = 7.0

N_BOOTSTRAPS = 5000

N_PERMUTATIONS = 5000


# ============================================================
# CSV HELPERS
# ============================================================

def read_csv(path):

    if not os.path.exists(path):

        raise IOError(
            "File not found:\n{}".format(
                path
            )
        )

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(
            f
        )

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
# NUMERIC HELPERS
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

    values = [
        float(x)
        for x in values
        if finite(x)
    ]

    if not values:

        return float("nan")

    return sum(values) / float(
        len(values)
    )


def std(values):

    values = [
        float(x)
        for x in values
        if finite(x)
    ]

    if len(values) < 2:

        return 0.0

    return float(
        np.std(
            np.asarray(values),
            ddof=1
        )
    )


def median(values):

    values = [
        float(x)
        for x in values
        if finite(x)
    ]

    if not values:

        return float("nan")

    return float(
        np.median(
            np.asarray(values)
        )
    )


# ============================================================
# HBGD
# ============================================================

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


# ============================================================
# LOAD LUMIERE
# ============================================================

def load_patients():

    rows = read_csv(
        TARGET_TRAJECTORY
    )

    patients = {}

    for row in rows:

        patient_id = str(
            row[
                "patient_id"
            ]
        )

        if patient_id not in patients:

            patients[
                patient_id
            ] = []

        week = int(
            float(
                row[
                    "week"
                ]
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

            if not finite(volume):

                volume = 0.0

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
# BUILD CASES
# ============================================================

def build_cases(
    patients,
    patient_ids
):

    cases = []

    for patient_id in patient_ids:

        obs = patients[
            patient_id
        ]

        if len(obs) < 2:

            continue

        for target_index in range(
            1,
            len(obs)
        ):

            current = obs[
                target_index - 1
            ]

            target = obs[
                target_index
            ]

            delta_weeks = (
                target["week"] -
                current["week"]
            )

            delta_days = (
                delta_weeks *
                DAYS_PER_WEEK
            )

            if delta_days <= 0:

                continue

            current_d = current[
                "diameter"
            ]

            target_d = target[
                "diameter"
            ]

            if (
                current_d <= 0
                or
                target_d <= 0
            ):

                continue

            # ------------------------------------------------
            # Historical HBGD information
            # ------------------------------------------------

            if target_index >= 2:

                previous = obs[
                    target_index - 2
                ]

                previous_d = previous[
                    "diameter"
                ]

                previous_days = (
                    current["week"] -
                    previous["week"]
                ) * DAYS_PER_WEEK

                recent_rate = calculate_rate(
                    previous_d,
                    current_d,
                    previous_days
                )

            else:

                previous_d = current_d

                recent_rate = 0.0

            historical_rates = []

            for j in range(
                1,
                target_index
            ):

                d1 = obs[
                    j - 1
                ][
                    "diameter"
                ]

                d2 = obs[
                    j
                ][
                    "diameter"
                ]

                days = (
                    obs[j]["week"] -
                    obs[j - 1]["week"]
                ) * DAYS_PER_WEEK

                r = calculate_rate(
                    d1,
                    d2,
                    days
                )

                if finite(r):

                    historical_rates.append(
                        r
                    )

            if historical_rates:

                mean_rate = mean(
                    historical_rates
                )

                rate_std = std(
                    historical_rates
                )

            else:

                mean_rate = 0.0

                rate_std = 0.0

            rate_change = (
                recent_rate -
                mean_rate
            )

            history_length = (
                target_index + 1
            )

            total_history_weeks = (
                current["week"] -
                obs[0]["week"]
            )

            if total_history_weeks < 0:

                total_history_weeks = 0

            current_volume = current[
                "volume"
            ]

            previous_volume = 0.0

            if target_index >= 2:

                previous_volume = previous[
                    "volume"
                ]

            # ------------------------------------------------
            # EXACT STEP 12 FEATURE STRUCTURE
            # ------------------------------------------------

            geometry = [

                current_d,

                current_volume,

                previous_d,

                current_d -
                previous_d
            ]

            temporal = [

                delta_weeks,

                history_length,

                total_history_weeks
            ]

            hbgd_rate = [

                recent_rate,

                mean_rate,

                rate_change
            ]

            hbgd_variability = [

                rate_std,

                min(
                    historical_rates
                )
                if historical_rates
                else 0.0,

                max(
                    historical_rates
                )
                if historical_rates
                else 0.0
            ]

            cases.append({

                "patient_id":
                    patient_id,

                "current_week":
                    current["week"],

                "future_week":
                    target["week"],

                "actual":
                    target_d,

                "geometry":
                    geometry,

                "temporal":
                    temporal,

                "hbgd_rate":
                    hbgd_rate,

                "hbgd_variability":
                    hbgd_variability
            })

    return cases


# ============================================================
# FEATURES
# ============================================================

def make_features(
    cases,
    include_hbgd
):

    X = []

    for case in cases:

        values = []

        values += case[
            "geometry"
        ]

        if include_hbgd:

            values += case[
                "hbgd_rate"
            ]

        X.append(
            values
        )

    return np.asarray(
        X,
        dtype=float
    )


# ============================================================
# TRAIN
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
# PATIENT MAE
# ============================================================

def patient_mae(
    actual,
    prediction
):

    return mean_absolute_error(
        actual,
        prediction
    )


# ============================================================
# BOOTSTRAP CI
# ============================================================

def bootstrap_mean_difference(
    differences
):

    rng = np.random.RandomState(
        RANDOM_SEED + 100
    )

    differences = np.asarray(
        differences,
        dtype=float
    )

    n = len(
        differences
    )

    bootstrap_means = []

    for i in range(
        N_BOOTSTRAPS
    ):

        indices = rng.randint(
            0,
            n,
            n
        )

        sample = differences[
            indices
        ]

        bootstrap_means.append(
            float(
                np.mean(sample)
            )
        )

    bootstrap_means = np.asarray(
        bootstrap_means
    )

    return {

        "mean":
            float(
                np.mean(
                    bootstrap_means
                )
            ),

        "median":
            float(
                np.median(
                    bootstrap_means
                )
            ),

        "ci_lower":
            float(
                np.percentile(
                    bootstrap_means,
                    2.5
                )
            ),

        "ci_upper":
            float(
                np.percentile(
                    bootstrap_means,
                    97.5
                )
            ),

        "std":
            float(
                np.std(
                    bootstrap_means
                )
            )
    }


# ============================================================
# PAIRED PERMUTATION TEST
# ============================================================

def paired_permutation_test(
    differences
):

    """
    Null hypothesis:
        mean patient-level improvement = 0

    Sign of each patient difference is randomly flipped.

    Positive difference:
        Geometry error > Geometry+HBGD error
        = HBGD improvement.

    Negative difference:
        HBGD worsened performance.
    """

    rng = np.random.RandomState(
        RANDOM_SEED + 200
    )

    differences = np.asarray(
        differences,
        dtype=float
    )

    observed = float(
        np.mean(
            differences
        )
    )

    absolute_observed = abs(
        observed
    )

    null_means = []

    for i in range(
        N_PERMUTATIONS
    ):

        signs = rng.choice(
            [-1.0, 1.0],
            size=len(
                differences
            )
        )

        randomized = (
            differences *
            signs
        )

        null_means.append(
            float(
                np.mean(
                    randomized
                )
            )
        )

    null_means = np.asarray(
        null_means
    )

    p_value = (
        np.sum(
            np.abs(
                null_means
            )
            >=
            absolute_observed
        )
        + 1.0
    ) / (
        float(
            N_PERMUTATIONS
        )
        + 1.0
    )

    return (
        observed,
        null_means,
        p_value
    )


# ============================================================
# WILCOXON
# ============================================================

def try_wilcoxon(
    differences
):

    try:

        from scipy.stats import wilcoxon

        result = wilcoxon(
            differences
        )

        statistic = float(
            result.statistic
        )

        p_value = float(
            result.pvalue
        )

        return (
            statistic,
            p_value
        )

    except Exception:

        return (
            float("nan"),
            float("nan")
        )


# ============================================================
# EFFECT SIZE
# ============================================================

def paired_effect_size(
    differences
):

    d_mean = mean(
        differences
    )

    d_std = std(
        differences
    )

    if d_std == 0:

        return 0.0

    return (
        d_mean /
        d_std
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "STEP 13 - PATIENT-LEVEL HBGD ROBUSTNESS"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    patients = load_patients()

    patient_ids = sorted(
        patients.keys()
    )

    print(
        "\nLUMIERE patients:",
        len(patient_ids)
    )

    # --------------------------------------------------------
    # Same seed/split as Step 12
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

    if n_test < 1:

        n_test = 1

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
    # Cases
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
    # Geometry model
    # --------------------------------------------------------

    X_train_geometry = make_features(
        train_cases,
        False
    )

    X_val_geometry = make_features(
        validation_cases,
        False
    )

    X_test_geometry = make_features(
        test_cases,
        False
    )

    X_train_hbgd = make_features(
        train_cases,
        True
    )

    X_val_hbgd = make_features(
        validation_cases,
        True
    )

    X_test_hbgd = make_features(
        test_cases,
        True
    )

    y_train = np.asarray(
        [
            x["actual"]
            for x in train_cases
        ],
        dtype=float
    )

    y_val = np.asarray(
        [
            x["actual"]
            for x in validation_cases
        ],
        dtype=float
    )

    y_test = np.asarray(
        [
            x["actual"]
            for x in test_cases
        ],
        dtype=float
    )

    # --------------------------------------------------------
    # Train two models
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

    geometry_test_prediction = (
        geometry_model.predict(
            X_test_geometry
        )
    )

    hbgd_test_prediction = (
        hbgd_model.predict(
            X_test_hbgd
        )
    )

    geometry_val_prediction = (
        geometry_model.predict(
            X_val_geometry
        )
    )

    hbgd_val_prediction = (
        hbgd_model.predict(
            X_val_hbgd
        )
    )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    geometry_test_mae = (
        mean_absolute_error(
            y_test,
            geometry_test_prediction
        )
    )

    hbgd_test_mae = (
        mean_absolute_error(
            y_test,
            hbgd_test_prediction
        )
    )

    geometry_val_mae = (
        mean_absolute_error(
            y_val,
            geometry_val_prediction
        )
    )

    hbgd_val_mae = (
        mean_absolute_error(
            y_val,
            hbgd_val_prediction
        )
    )

    print(
        "\nOverall validation:"
    )

    print(
        "Geometry:",
        "{:.4f} mm".format(
            geometry_val_mae
        )
    )

    print(
        "Geometry + HBGD rate:",
        "{:.4f} mm".format(
            hbgd_val_mae
        )
    )

    print(
        "\nOverall test:"
    )

    print(
        "Geometry:",
        "{:.4f} mm".format(
            geometry_test_mae
        )
    )

    print(
        "Geometry + HBGD rate:",
        "{:.4f} mm".format(
            hbgd_test_mae
        )
    )

    # --------------------------------------------------------
    # Patient-level metrics
    # --------------------------------------------------------

    patient_results = []

    unique_test_patients = sorted(
        list(
            set(
                [
                    x[
                        "patient_id"
                    ]
                    for x in test_cases
                ]
            )
        )
    )

    differences = []

    geometry_patient_maes = []

    hbgd_patient_maes = []

    for patient_id in unique_test_patients:

        indices = [

            i

            for i, case in enumerate(
                test_cases
            )

            if case[
                "patient_id"
            ]
            ==
            patient_id
        ]

        actual = np.asarray(
            [
                y_test[i]
                for i in indices
            ],
            dtype=float
        )

        geometry_prediction = np.asarray(
            [
                geometry_test_prediction[i]
                for i in indices
            ],
            dtype=float
        )

        hbgd_prediction = np.asarray(
            [
                hbgd_test_prediction[i]
                for i in indices
            ],
            dtype=float
        )

        geometry_mae_patient = (
            mean_absolute_error(
                actual,
                geometry_prediction
            )
        )

        hbgd_mae_patient = (
            mean_absolute_error(
                actual,
                hbgd_prediction
            )
        )

        improvement = (
            geometry_mae_patient -
            hbgd_mae_patient
        )

        relative_improvement = 0.0

        if geometry_mae_patient != 0:

            relative_improvement = (
                improvement /
                geometry_mae_patient
            ) * 100.0

        if improvement > 0:

            status = "improved"

        elif improvement < 0:

            status = "worsened"

        else:

            status = "unchanged"

        patient_results.append({

            "patient_id":
                patient_id,

            "n_prediction_cases":
                len(indices),

            "geometry_mae_mm":
                geometry_mae_patient,

            "geometry_hbgd_mae_mm":
                hbgd_mae_patient,

            "absolute_improvement_mm":
                improvement,

            "relative_improvement_percent":
                relative_improvement,

            "status":
                status
        })

        differences.append(
            improvement
        )

        geometry_patient_maes.append(
            geometry_mae_patient
        )

        hbgd_patient_maes.append(
            hbgd_mae_patient
        )

    # --------------------------------------------------------
    # Patient statistics
    # --------------------------------------------------------

    differences_array = np.asarray(
        differences,
        dtype=float
    )

    improved = int(
        np.sum(
            differences_array > 0
        )
    )

    worsened = int(
        np.sum(
            differences_array < 0
        )
    )

    unchanged = int(
        np.sum(
            differences_array == 0
        )
    )

    mean_improvement = mean(
        differences_array
    )

    median_improvement = median(
        differences_array
    )

    sd_improvement = std(
        differences_array
    )

    effect_size = paired_effect_size(
        differences_array
    )

    # --------------------------------------------------------
    # Bootstrap
    # --------------------------------------------------------

    bootstrap = (
        bootstrap_mean_difference(
            differences_array
        )
    )

    # --------------------------------------------------------
    # Permutation
    # --------------------------------------------------------

    (
        permutation_observed,
        permutation_distribution,
        permutation_p
    ) = paired_permutation_test(
        differences_array
    )

    # --------------------------------------------------------
    # Wilcoxon
    # --------------------------------------------------------

    (
        wilcoxon_statistic,
        wilcoxon_p
    ) = try_wilcoxon(
        differences_array
    )

    # --------------------------------------------------------
    # Console statistics
    # --------------------------------------------------------

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "PATIENT-LEVEL RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        "Test patients:",
        len(unique_test_patients)
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
        "Mean improvement:",
        "{:.4f} mm".format(
            mean_improvement
        )
    )

    print(
        "Median improvement:",
        "{:.4f} mm".format(
            median_improvement
        )
    )

    print(
        "SD improvement:",
        "{:.4f} mm".format(
            sd_improvement
        )
    )

    print(
        "Paired effect size:",
        "{:.4f}".format(
            effect_size
        )
    )

    print(
        "\nBootstrap 95% CI:"
    )

    print(
        "[{:.4f}, {:.4f}] mm".format(
            bootstrap[
                "ci_lower"
            ],
            bootstrap[
                "ci_upper"
            ]
        )
    )

    print(
        "\nPaired permutation p-value:",
        "{:.6f}".format(
            permutation_p
        )
    )

    if finite(
        wilcoxon_p
    ):

        print(
            "Wilcoxon p-value:",
            "{:.6f}".format(
                wilcoxon_p
            )
        )

    else:

        print(
            "Wilcoxon p-value:",
            "scipy unavailable"
        )

    # --------------------------------------------------------
    # Save patient table
    # --------------------------------------------------------

    write_csv(

        OUTPUT_PATIENT,

        patient_results,

        [
            "patient_id",
            "n_prediction_cases",
            "geometry_mae_mm",
            "geometry_hbgd_mae_mm",
            "absolute_improvement_mm",
            "relative_improvement_percent",
            "status"
        ]
    )

    # --------------------------------------------------------
    # Save bootstrap
    # --------------------------------------------------------

    write_csv(

        OUTPUT_BOOTSTRAP,

        [

            {

                "metric":
                    "patient_level_mae_improvement",

                "observed_mean":
                    mean_improvement,

                "bootstrap_mean":
                    bootstrap[
                        "mean"
                    ],

                "bootstrap_median":
                    bootstrap[
                        "median"
                    ],

                "bootstrap_std":
                    bootstrap[
                        "std"
                    ],

                "ci_lower_2_5":
                    bootstrap[
                        "ci_lower"
                    ],

                "ci_upper_97_5":
                    bootstrap[
                        "ci_upper"
                    ]
            }
        ],

        [
            "metric",
            "observed_mean",
            "bootstrap_mean",
            "bootstrap_median",
            "bootstrap_std",
            "ci_lower_2_5",
            "ci_upper_97_5"
        ]
    )

    # --------------------------------------------------------
    # Save permutation distribution
    # --------------------------------------------------------

    permutation_rows = []

    for i, value in enumerate(
        permutation_distribution
    ):

        permutation_rows.append({

            "permutation":
                i + 1,

            "mean_randomized_improvement_mm":
                value
        })

    write_csv(

        OUTPUT_PERMUTATION,

        permutation_rows,

        [
            "permutation",
            "mean_randomized_improvement_mm"
        ]
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows = [

        {

            "metric":
                "test_patients",

            "value":
                len(
                    unique_test_patients
                )
        },

        {

            "metric":
                "improved_patients",

            "value":
                improved
        },

        {

            "metric":
                "worsened_patients",

            "value":
                worsened
        },

        {

            "metric":
                "unchanged_patients",

            "value":
                unchanged
        },

        {

            "metric":
                "geometry_test_mae_mm",

            "value":
                geometry_test_mae
        },

        {

            "metric":
                "geometry_hbgd_test_mae_mm",

            "value":
                hbgd_test_mae
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
                "patient_improvement_sd_mm",

            "value":
                sd_improvement
        },

        {

            "metric":
                "paired_effect_size",

            "value":
                effect_size
        },

        {

            "metric":
                "bootstrap_ci_lower_mm",

            "value":
                bootstrap[
                    "ci_lower"
                ]
        },

        {

            "metric":
                "bootstrap_ci_upper_mm",

            "value":
                bootstrap[
                    "ci_upper"
                ]
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

    write_csv(

        OUTPUT_SUMMARY,

        summary_rows,

        [
            "metric",
            "value"
        ]
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = []

    report.append(
        "# STEP 13 - PATIENT-LEVEL HBGD ROBUSTNESS AND STATISTICAL TESTING"
    )

    report.append("")

    report.append(
        "## Objective"
    )

    report.append("")

    report.append(
        "Step 13 tests whether the small predictive improvement from adding HBGD rate features is consistent across held-out patients rather than being driven by a small number of prediction cases."
    )

    report.append("")

    report.append(
        "## Primary Comparison"
    )

    report.append("")

    report.append(
        "Geometry-only versus Geometry + HBGD rate."
    )

    report.append("")

    report.append(
        "## Patient Split"
    )

    report.append("")

    report.append(
        "| Partition | Patients |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        "| Train | {} |".format(
            len(train_ids)
        )
    )

    report.append(
        "| Validation | {} |".format(
            len(validation_ids)
        )
    )

    report.append(
        "| Test | {} |".format(
            len(test_ids)
        )
    )

    report.append("")

    report.append(
        "## Overall Test Performance"
    )

    report.append("")

    report.append(
        "| Model | Test MAE |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        "| Geometry-only | {:.4f} mm |".format(
            geometry_test_mae
        )
    )

    report.append(
        "| Geometry + HBGD rate | {:.4f} mm |".format(
            hbgd_test_mae
        )
    )

    report.append("")

    report.append(
        "## Patient-Level Results"
    )

    report.append("")

    report.append(
        "| Metric | Value |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        "| Patients improved | {} |".format(
            improved
        )
    )

    report.append(
        "| Patients worsened | {} |".format(
            worsened
        )
    )

    report.append(
        "| Patients unchanged | {} |".format(
            unchanged
        )
    )

    report.append(
        "| Mean improvement | {:.4f} mm |".format(
            mean_improvement
        )
    )

    report.append(
        "| Median improvement | {:.4f} mm |".format(
            median_improvement
        )
    )

    report.append(
        "| Improvement SD | {:.4f} mm |".format(
            sd_improvement
        )
    )

    report.append(
        "| Paired effect size | {:.4f} |".format(
            effect_size
        )
    )

    report.append("")

    report.append(
        "## Bootstrap Analysis"
    )

    report.append("")

    report.append(
        "The patient-level improvement was resampled 5000 times."
    )

    report.append("")

    report.append(
        "Bootstrap mean: {:.4f} mm".format(
            bootstrap[
                "mean"
            ]
        )
    )

    report.append("")

    report.append(
        "Bootstrap 95% CI: [{:.4f}, {:.4f}] mm".format(
            bootstrap[
                "ci_lower"
            ],
            bootstrap[
                "ci_upper"
            ]
        )
    )

    report.append("")

    report.append(
        "## Paired Permutation Test"
    )

    report.append("")

    report.append(
        "Null hypothesis: the mean patient-level MAE improvement is zero."
    )

    report.append("")

    report.append(
        "Observed mean improvement: {:.4f} mm".format(
            permutation_observed
        )
    )

    report.append("")

    report.append(
        "Permutation p-value: {:.6f}".format(
            permutation_p
        )
    )

    report.append("")

    if finite(
        wilcoxon_p
    ):

        report.append(
            "Wilcoxon signed-rank p-value: {:.6f}".format(
                wilcoxon_p
            )
        )

    else:

        report.append(
            "Wilcoxon signed-rank test was not available because scipy could not be imported."
        )

    report.append("")

    report.append(
        "## Interpretation"
    )

    report.append("")

    if mean_improvement > 0:

        report.append(
            "The mean patient-level improvement favors the geometry + HBGD-rate model."
        )

    elif mean_improvement < 0:

        report.append(
            "The mean patient-level difference favors geometry-only."
        )

    else:

        report.append(
            "The mean patient-level difference is zero."
        )

    report.append("")

    if improved > worsened:

        report.append(
            "More held-out patients improved than worsened when HBGD rate was added."
        )

    elif improved < worsened:

        report.append(
            "More held-out patients worsened than improved when HBGD rate was added."
        )

    else:

        report.append(
            "The number of improved and worsened patients was equal."
        )

    report.append("")

    if (
        bootstrap[
            "ci_lower"
        ] > 0
    ):

        report.append(
            "The bootstrap 95% confidence interval lies above zero, indicating that the observed positive patient-level improvement is relatively robust under bootstrap resampling."
        )

    else:

        report.append(
            "The bootstrap 95% confidence interval includes zero. Therefore, the small observed HBGD improvement cannot be considered statistically robust based on this bootstrap analysis."
        )

    report.append("")

    if permutation_p < 0.05:

        report.append(
            "The paired permutation test rejects the zero-improvement null hypothesis at the 0.05 level."
        )

    else:

        report.append(
            "The paired permutation test does not reject the zero-improvement null hypothesis at the 0.05 level."
        )

    report.append("")

    report.append(
        "## Scientific Interpretation"
    )

    report.append("")

    report.append(
        "This analysis evaluates incremental predictive information, not biological validity of the exponential HBGD model."
    )

    report.append("")

    report.append(
        "Because only 13 test patients are available, statistical power is limited and the result should be considered preliminary even if a nominal statistical threshold is reached."
    )

    report.append("")

    report.append(
        "## Conclusion"
    )

    report.append("")

    report.append(
        "Step 13 determines whether the small HBGD-related improvement observed in Step 12 is reproducible at the patient level. The result should be interpreted using the direction of the effect, the number of improved versus worsened patients, the bootstrap interval, and the paired statistical tests together."
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

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "STEP 13 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nOUTPUTS:"
    )

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
        OUTPUT_REPORT
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()