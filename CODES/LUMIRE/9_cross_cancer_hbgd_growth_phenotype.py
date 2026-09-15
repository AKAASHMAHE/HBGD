
"""
9_cross_cancer_hbgd_growth_phenotype.py

STEP 9
Cross-Cancer HBGD Growth-Phenotype Alignment

SOURCE:
    Existing HBGD source-cancer pipeline

TARGET:
    LUMIERE glioblastoma longitudinal cohort

OBJECTIVE:
    Determine whether HBGD growth dynamics from the source
    cancer and LUMIERE GBM can be represented in a comparable
    growth-phenotype space.

IMPORTANT:
    This is NOT a prediction experiment.

    It evaluates:
        - growth-rate distributions
        - volume-growth distributions
        - temporal variability
        - dynamic states
        - trajectory behavior
        - normalized growth
        - cross-domain feature similarity

    It does NOT claim that source-cancer growth rates are
    biologically identical to GBM growth rates.

OUTPUT:
    D:\\VIT\\PROJECT\\NLP\\PATENT\\DATA\\LUMIRE\\

Python:
    Designed for Python 3.6 compatibility.
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

SOURCE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA"

LUMIERE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"


SOURCE_MATH = os.path.join(
    SOURCE_DIR,
    "17_hbgd_mathematical_features.csv"
)

SOURCE_GROWTH = os.path.join(
    SOURCE_DIR,
    "17_hbgd_growth_parameters.csv"
)

SOURCE_STATE = os.path.join(
    SOURCE_DIR,
    "17_hbgd_state_parameters.csv"
)

SOURCE_VOLUME = os.path.join(
    SOURCE_DIR,
    "17_hbgd_volume_parameters.csv"
)

SOURCE_TRAJECTORY = os.path.join(
    SOURCE_DIR,
    "17_hbgd_diameter_trajectories.csv"
)

TARGET_TRAJECTORY = os.path.join(
    LUMIERE_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

TARGET_HBGD = os.path.join(
    LUMIERE_DIR,
    "lumiere_hbgd_growth_features.csv"
)


SOURCE_PHENOTYPE_CSV = os.path.join(
    LUMIERE_DIR,
    "source_hbgd_growth_phenotypes.csv"
)

TARGET_PHENOTYPE_CSV = os.path.join(
    LUMIERE_DIR,
    "lumiere_hbgd_growth_phenotypes.csv"
)

COMBINED_PHENOTYPE_CSV = os.path.join(
    LUMIERE_DIR,
    "cross_cancer_hbgd_phenotype_vectors.csv"
)

STATISTICS_CSV = os.path.join(
    LUMIERE_DIR,
    "cross_cancer_hbgd_feature_statistics.csv"
)

DISTANCE_CSV = os.path.join(
    LUMIERE_DIR,
    "cross_cancer_hbgd_feature_distances.csv"
)

STATE_CSV = os.path.join(
    LUMIERE_DIR,
    "cross_cancer_hbgd_state_comparison.csv"
)

REPORT_PATH = os.path.join(
    LUMIERE_DIR,
    "CROSS_CANCER_HBGD_GROWTH_PHENOTYPE_REPORT.md"
)


# ============================================================
# SETTINGS
# ============================================================

DAYS_PER_WEEK = 7.0

RANDOM_SEED = 42


# ============================================================
# BASIC HELPERS
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
        float(v)
        for v in values
        if finite(v)
    ]

    if not values:
        return float("nan")

    return sum(values) / float(
        len(values)
    )


def std(values):

    values = [
        float(v)
        for v in values
        if finite(v)
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
        float(v)
        for v in values
        if finite(v)
    ]

    if not values:
        return float("nan")

    return float(
        np.median(
            np.asarray(values)
        )
    )


def minimum(values):

    values = [
        float(v)
        for v in values
        if finite(v)
    ]

    if not values:
        return float("nan")

    return min(values)


def maximum(values):

    values = [
        float(v)
        for v in values
        if finite(v)
    ]

    if not values:
        return float("nan")

    return max(values)


def percentile(values, p):

    values = [
        float(v)
        for v in values
        if finite(v)
    ]

    if not values:
        return float("nan")

    return float(
        np.percentile(
            np.asarray(values),
            p
        )
    )


# ============================================================
# CSV
# ============================================================

def read_csv(path):

    if not os.path.exists(path):

        raise IOError(
            "Required file not found:\n{}".format(
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

            rows.append(
                row
            )

    return rows


def write_csv(
    path,
    rows,
    fieldnames
):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )


# ============================================================
# HBGD FUNCTIONS
# ============================================================

def calculate_rate(
    d1,
    d2,
    delta_days
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
        delta_days <= 0
    ):

        return float("nan")

    return (
        math.log(
            d2 / d1
        )
        /
        float(delta_days)
    )


def classify_state(rate):

    if not finite(rate):

        return "UNKNOWN"

    if rate > 0.0005:

        return "growth"

    if rate < -0.0010:

        return "shrinkage"

    return "stable"


# ============================================================
# SOURCE PHENOTYPE
# ============================================================

def build_source_phenotypes():

    print(
        "\nLoading source HBGD data..."
    )

    growth_rows = read_csv(
        SOURCE_GROWTH
    )

    state_rows = read_csv(
        SOURCE_STATE
    )

    volume_rows = read_csv(
        SOURCE_VOLUME
    )

    math_rows = read_csv(
        SOURCE_MATH
    )

    trajectory_rows = read_csv(
        SOURCE_TRAJECTORY
    )

    # --------------------------------------------------------
    # Growth parameters
    # --------------------------------------------------------

    growth_by_subject = {}

    for row in growth_rows:

        patient = row[
            "patient_id"
        ]

        lesion = row[
            "lesion_alias"
        ]

        key = (
            str(patient),
            str(lesion)
        )

        growth_by_subject[
            key
        ] = {

            "growth_rate":
                safe_float(
                    row[
                        "growth_rate_r_per_day"
                    ]
                ),

            "trajectory_mae":
                safe_float(
                    row[
                        "trajectory_MAE_mm"
                    ]
                ),

            "trajectory_rmse":
                safe_float(
                    row[
                        "trajectory_RMSE_mm"
                    ]
                )
        }

    # --------------------------------------------------------
    # Volume parameters
    # --------------------------------------------------------

    volume_by_subject = {}

    for row in volume_rows:

        patient = row[
            "patient_id"
        ]

        lesion = row[
            "lesion_alias"
        ]

        key = (
            str(patient),
            str(lesion)
        )

        volume_by_subject[
            key
        ] = safe_float(
            row[
                "volume_growth_rate_per_day"
            ]
        )

    # --------------------------------------------------------
    # State behavior
    # --------------------------------------------------------

    state_by_subject = {}

    for row in state_rows:

        patient = row[
            "patient_id"
        ]

        lesion = row[
            "lesion_alias"
        ]

        key = (
            str(patient),
            str(lesion)
        )

        if key not in state_by_subject:

            state_by_subject[
                key
            ] = []

        state = row[
            "dynamic_state"
        ].strip().lower()

        rate = safe_float(
            row[
                "growth_rate"
            ]
        )

        state_by_subject[
            key
        ].append(
            (
                state,
                rate
            )
        )

    # --------------------------------------------------------
    # Build phenotype subjects
    # --------------------------------------------------------

    keys = set(
        growth_by_subject.keys()
    )

    keys.update(
        volume_by_subject.keys()
    )

    keys.update(
        state_by_subject.keys()
    )

    phenotypes = []

    for key in sorted(keys):

        patient = key[0]

        lesion = key[1]

        growth = growth_by_subject.get(
            key,
            {}
        )

        growth_rate = growth.get(
            "growth_rate",
            float("nan")
        )

        volume_rate = volume_by_subject.get(
            key,
            float("nan")
        )

        states = state_by_subject.get(
            key,
            []
        )

        state_values = [
            x[0]
            for x in states
        ]

        state_rates = [
            x[1]
            for x in states
            if finite(x[1])
        ]

        total_states = len(
            state_values
        )

        growth_fraction = 0.0
        stable_fraction = 0.0
        shrink_fraction = 0.0

        if total_states > 0:

            growth_fraction = (
                state_values.count(
                    "growth"
                )
                /
                float(total_states)
            )

            stable_fraction = (
                state_values.count(
                    "stable"
                )
                /
                float(total_states)
            )

            shrink_fraction = (
                state_values.count(
                    "shrinkage"
                )
                /
                float(total_states)
            )

        # State persistence

        transitions = 0

        persistent = 0

        for i in range(
            len(state_values) - 1
        ):

            transitions += 1

            if (
                state_values[i]
                ==
                state_values[i + 1]
            ):

                persistent += 1

        if transitions > 0:

            state_persistence = (
                persistent /
                float(transitions)
            )

        else:

            state_persistence = 0.0

        phenotypes.append({

            "subject_id":
                patient,

            "lesion_id":
                lesion,

            "growth_rate_per_day":
                growth_rate,

            "volume_growth_rate_per_day":
                volume_rate,

            "state_rate_mean":
                mean(
                    state_rates
                ),

            "state_rate_std":
                std(
                    state_rates
                ),

            "growth_fraction":
                growth_fraction,

            "stable_fraction":
                stable_fraction,

            "shrinkage_fraction":
                shrink_fraction,

            "state_persistence":
                state_persistence,

            "state_interval_count":
                total_states,

            "trajectory_mae_mm":
                growth.get(
                    "trajectory_mae",
                    float("nan")
                ),

            "trajectory_rmse_mm":
                growth.get(
                    "trajectory_rmse",
                    float("nan")
                )
        })

    print(
        "Source phenotype subjects:",
        len(phenotypes)
    )

    return phenotypes


# ============================================================
# TARGET LUMIERE PHENOTYPES
# ============================================================

def build_lumiere_phenotypes():

    print(
        "\nLoading LUMIERE trajectories..."
    )

    rows = read_csv(
        TARGET_TRAJECTORY
    )

    patients = {}

    for row in rows:

        patient = row[
            "patient_id"
        ]

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

        if patient not in patients:

            patients[
                patient
            ] = []

        patients[
            patient
        ].append({

            "week":
                week,

            "diameter":
                diameter,

            "volume":
                volume
        })

    for patient in patients:

        patients[
            patient
        ].sort(
            key=lambda x:
                x["week"]
        )

    phenotypes = []

    for patient in sorted(
        patients.keys()
    ):

        obs = patients[
            patient
        ]

        if len(obs) < 2:

            continue

        rates = []

        volume_rates = []

        states = []

        interval_lengths = []

        for i in range(
            len(obs) - 1
        ):

            d1 = obs[i][
                "diameter"
            ]

            d2 = obs[i + 1][
                "diameter"
            ]

            v1 = obs[i][
                "volume"
            ]

            v2 = obs[i + 1][
                "volume"
            ]

            delta_weeks = (
                obs[i + 1]["week"] -
                obs[i]["week"]
            )

            delta_days = (
                delta_weeks *
                DAYS_PER_WEEK
            )

            r = calculate_rate(
                d1,
                d2,
                delta_days
            )

            if finite(r):

                rates.append(
                    r
                )

                states.append(
                    classify_state(r)
                )

            if (
                finite(v1)
                and
                finite(v2)
                and
                v1 > 0
                and
                v2 > 0
                and
                delta_days > 0
            ):

                vr = (
                    math.log(
                        v2 / v1
                    )
                    /
                    delta_days
                )

                if finite(vr):

                    volume_rates.append(
                        vr
                    )

            if delta_days > 0:

                interval_lengths.append(
                    delta_days
                )

        if not rates:

            continue

        # ----------------------------------------------------
        # Overall growth
        # ----------------------------------------------------

        first_d = obs[0][
            "diameter"
        ]

        last_d = obs[-1][
            "diameter"
        ]

        total_days = (
            obs[-1]["week"] -
            obs[0]["week"]
        ) * DAYS_PER_WEEK

        overall_rate = calculate_rate(
            first_d,
            last_d,
            total_days
        )

        # ----------------------------------------------------
        # State fractions
        # ----------------------------------------------------

        n_states = len(
            states
        )

        growth_fraction = (
            states.count("growth") /
            float(n_states)
        )

        stable_fraction = (
            states.count("stable") /
            float(n_states)
        )

        shrinkage_fraction = (
            states.count("shrinkage") /
            float(n_states)
        )

        # ----------------------------------------------------
        # State persistence
        # ----------------------------------------------------

        transitions = 0

        persistent = 0

        for i in range(
            len(states) - 1
        ):

            transitions += 1

            if (
                states[i]
                ==
                states[i + 1]
            ):

                persistent += 1

        if transitions > 0:

            state_persistence = (
                persistent /
                float(transitions)
            )

        else:

            state_persistence = 0.0

        # ----------------------------------------------------
        # Normalized trajectory
        # ----------------------------------------------------

        initial_diameter = obs[0][
            "diameter"
        ]

        normalized_final = (
            last_d /
            initial_diameter
            if initial_diameter > 0
            else float("nan")
        )

        # ----------------------------------------------------
        # Acceleration
        # ----------------------------------------------------

        acceleration = 0.0

        if len(rates) >= 2:

            first_half = rates[
                :len(rates) // 2
            ]

            second_half = rates[
                len(rates) // 2:
            ]

            if (
                first_half
                and
                second_half
            ):

                acceleration = (
                    mean(
                        second_half
                    )
                    -
                    mean(
                        first_half
                    )
                )

        # ----------------------------------------------------
        # Phenotype
        # ----------------------------------------------------

        phenotypes.append({

            "subject_id":
                patient,

            "lesion_id":
                "whole_tumor",

            "growth_rate_per_day":
                mean(rates),

            "volume_growth_rate_per_day":
                mean(
                    volume_rates
                ),

            "overall_growth_rate_per_day":
                overall_rate,

            "growth_rate_std":
                std(rates),

            "growth_rate_median":
                median(rates),

            "growth_rate_p10":
                percentile(
                    rates,
                    10
                ),

            "growth_rate_p90":
                percentile(
                    rates,
                    90
                ),

            "growth_fraction":
                growth_fraction,

            "stable_fraction":
                stable_fraction,

            "shrinkage_fraction":
                shrinkage_fraction,

            "state_persistence":
                state_persistence,

            "state_interval_count":
                n_states,

            "mean_interval_days":
                mean(
                    interval_lengths
                ),

            "total_followup_days":
                total_days,

            "normalized_final_diameter":
                normalized_final,

            "growth_rate_acceleration":
                acceleration,

            "initial_diameter_mm":
                initial_diameter,

            "final_diameter_mm":
                last_d
        })

    print(
        "LUMIERE phenotype patients:",
        len(phenotypes)
    )

    return phenotypes


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

SOURCE_FEATURES = [

    "growth_rate_per_day",

    "volume_growth_rate_per_day",

    "state_rate_mean",

    "state_rate_std",

    "growth_fraction",

    "stable_fraction",

    "shrinkage_fraction",

    "state_persistence"
]


TARGET_FEATURES = [

    "growth_rate_per_day",

    "volume_growth_rate_per_day",

    "overall_growth_rate_per_day",

    "growth_rate_std",

    "growth_rate_median",

    "growth_rate_p10",

    "growth_rate_p90",

    "growth_fraction",

    "stable_fraction",

    "shrinkage_fraction",

    "state_persistence",

    "mean_interval_days",

    "total_followup_days",

    "normalized_final_diameter",

    "growth_rate_acceleration"
]


# ============================================================
# STATISTICS
# ============================================================

def calculate_statistics(
    source_rows,
    target_rows
):

    rows = []

    comparable_features = [

        (
            "growth_rate_per_day",
            "growth_rate_per_day"
        ),

        (
            "volume_growth_rate_per_day",
            "volume_growth_rate_per_day"
        ),

        (
            "growth_fraction",
            "growth_fraction"
        ),

        (
            "stable_fraction",
            "stable_fraction"
        ),

        (
            "shrinkage_fraction",
            "shrinkage_fraction"
        ),

        (
            "state_persistence",
            "state_persistence"
        )
    ]

    for source_feature, target_feature in comparable_features:

        source_values = [

            safe_float(
                row.get(
                    source_feature,
                    float("nan")
                )
            )

            for row in source_rows
        ]

        target_values = [

            safe_float(
                row.get(
                    target_feature,
                    float("nan")
                )
            )

            for row in target_rows
        ]

        source_values = [
            x
            for x in source_values
            if finite(x)
        ]

        target_values = [
            x
            for x in target_values
            if finite(x)
        ]

        source_mean = mean(
            source_values
        )

        target_mean = mean(
            target_values
        )

        source_std = std(
            source_values
        )

        target_std = std(
            target_values
        )

        pooled_std = math.sqrt(
            (
                source_std ** 2 +
                target_std ** 2
            )
            /
            2.0
        )

        if pooled_std > 0:

            standardized_mean_difference = (
                (
                    target_mean -
                    source_mean
                )
                /
                pooled_std
            )

        else:

            standardized_mean_difference = 0.0

        rows.append({

            "feature":
                source_feature,

            "source_n":
                len(source_values),

            "target_n":
                len(target_values),

            "source_mean":
                source_mean,

            "source_std":
                source_std,

            "source_median":
                median(
                    source_values
                ),

            "source_p10":
                percentile(
                    source_values,
                    10
                ),

            "source_p90":
                percentile(
                    source_values,
                    90
                ),

            "target_mean":
                target_mean,

            "target_std":
                target_std,

            "target_median":
                median(
                    target_values
                ),

            "target_p10":
                percentile(
                    target_values,
                    10
                ),

            "target_p90":
                percentile(
                    target_values,
                    90
                ),

            "standardized_mean_difference":
                standardized_mean_difference
        })

    return rows


# ============================================================
# STATE COMPARISON
# ============================================================

def state_comparison(
    source_rows,
    target_rows
):

    source_growth = mean([
        row[
            "growth_fraction"
        ]
        for row in source_rows
    ])

    source_stable = mean([
        row[
            "stable_fraction"
        ]
        for row in source_rows
    ])

    source_shrink = mean([
        row[
            "shrinkage_fraction"
        ]
        for row in source_rows
    ])

    target_growth = mean([
        row[
            "growth_fraction"
        ]
        for row in target_rows
    ])

    target_stable = mean([
        row[
            "stable_fraction"
        ]
        for row in target_rows
    ])

    target_shrink = mean([
        row[
            "shrinkage_fraction"
        ]
        for row in target_rows
    ])

    return [

        {
            "state":
                "growth",

            "source_mean_fraction":
                source_growth,

            "lumiere_mean_fraction":
                target_growth,

            "absolute_difference":
                abs(
                    source_growth -
                    target_growth
                )
        },

        {
            "state":
                "stable",

            "source_mean_fraction":
                source_stable,

            "lumiere_mean_fraction":
                target_stable,

            "absolute_difference":
                abs(
                    source_stable -
                    target_stable
                )
        },

        {
            "state":
                "shrinkage",

            "source_mean_fraction":
                source_shrink,

            "lumiere_mean_fraction":
                target_shrink,

            "absolute_difference":
                abs(
                    source_shrink -
                    target_shrink
                )
        }
    ]


# ============================================================
# PHENOTYPE VECTORS
# ============================================================

def create_comparable_vectors(
    source_rows,
    target_rows,
    statistics
):

    """
    Create domain-standardized vectors.

    Each feature is standardized using the combined
    source + target distribution.

    This is for phenotype-space visualization/alignment,
    not for supervised prediction.
    """

    combined_rows = []

    for row in source_rows:

        output = {

            "domain":
                "source_cancer",

            "subject_id":
                row[
                    "subject_id"
                ],

            "lesion_id":
                row[
                    "lesion_id"
                ]
        }

        for stat in statistics:

            feature = stat[
                "feature"
            ]

            value = safe_float(
                row.get(
                    feature,
                    float("nan")
                )
            )

            if finite(value):

                source_mean = float(
                    stat[
                        "source_mean"
                    ]
                )

                target_mean = float(
                    stat[
                        "target_mean"
                    ]
                )

                source_std = float(
                    stat[
                        "source_std"
                    ]
                )

                target_std = float(
                    stat[
                        "target_std"
                    ]
                )

                pooled_std = math.sqrt(
                    (
                        source_std ** 2 +
                        target_std ** 2
                    )
                    /
                    2.0
                )

                if pooled_std > 0:

                    standardized = (
                        value -
                        (
                            source_mean +
                            target_mean
                        ) / 2.0
                    ) / pooled_std

                else:

                    standardized = 0.0

            else:

                standardized = 0.0

            output[
                "{}_standardized".format(
                    feature
                )
            ] = standardized

        combined_rows.append(
            output
        )

    for row in target_rows:

        output = {

            "domain":
                "lumiere_gbm",

            "subject_id":
                row[
                    "subject_id"
                ],

            "lesion_id":
                row[
                    "lesion_id"
                ]
        }

        for stat in statistics:

            feature = stat[
                "feature"
            ]

            value = safe_float(
                row.get(
                    feature,
                    float("nan")
                )
            )

            if finite(value):

                source_mean = float(
                    stat[
                        "source_mean"
                    ]
                )

                target_mean = float(
                    stat[
                        "target_mean"
                    ]
                )

                source_std = float(
                    stat[
                        "source_std"
                    ]
                )

                target_std = float(
                    stat[
                        "target_std"
                    ]
                )

                pooled_std = math.sqrt(
                    (
                        source_std ** 2 +
                        target_std ** 2
                    )
                    /
                    2.0
                )

                if pooled_std > 0:

                    standardized = (
                        value -
                        (
                            source_mean +
                            target_mean
                        ) / 2.0
                    ) / pooled_std

                else:

                    standardized = 0.0

            else:

                standardized = 0.0

            output[
                "{}_standardized".format(
                    feature
                )
            ] = standardized

        combined_rows.append(
            output
        )

    return combined_rows


# ============================================================
# DISTANCE ANALYSIS
# ============================================================

def calculate_distances(
    statistics
):

    rows = []

    for stat in statistics:

        feature = stat[
            "feature"
        ]

        source_mean = stat[
            "source_mean"
        ]

        target_mean = stat[
            "target_mean"
        ]

        source_std = stat[
            "source_std"
        ]

        target_std = stat[
            "target_std"
        ]

        pooled_std = math.sqrt(
            (
                source_std ** 2 +
                target_std ** 2
            )
            /
            2.0
        )

        if pooled_std > 0:

            mean_distance = (
                abs(
                    target_mean -
                    source_mean
                )
                /
                pooled_std
            )

        else:

            mean_distance = 0.0

        if (
            source_std > 0
            and
            target_std > 0
        ):

            variability_ratio = (
                target_std /
                source_std
            )

        else:

            variability_ratio = float(
                "nan"
            )

        rows.append({

            "feature":
                feature,

            "absolute_mean_difference":
                abs(
                    target_mean -
                    source_mean
                ),

            "pooled_standard_deviation":
                pooled_std,

            "standardized_mean_distance":
                mean_distance,

            "source_std":
                source_std,

            "lumiere_std":
                target_std,

            "lumiere_to_source_std_ratio":
                variability_ratio
        })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "STEP 9 - CROSS-CANCER HBGD GROWTH PHENOTYPE ALIGNMENT"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    source_rows = (
        build_source_phenotypes()
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_rows = (
        build_lumiere_phenotypes()
    )

    if not source_rows:

        print(
            "\nERROR: No source phenotypes."
        )

        return

    if not target_rows:

        print(
            "\nERROR: No LUMIERE phenotypes."
        )

        return

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "\nCalculating feature statistics..."
    )

    statistics = calculate_statistics(
        source_rows,
        target_rows
    )

    distances = calculate_distances(
        statistics
    )

    states = state_comparison(
        source_rows,
        target_rows
    )

    # --------------------------------------------------------
    # Vectors
    # --------------------------------------------------------

    vectors = create_comparable_vectors(
        source_rows,
        target_rows,
        statistics
    )

    # --------------------------------------------------------
    # Save source phenotype
    # --------------------------------------------------------

    source_fields = [

        "subject_id",
        "lesion_id",
        "growth_rate_per_day",
        "volume_growth_rate_per_day",
        "state_rate_mean",
        "state_rate_std",
        "growth_fraction",
        "stable_fraction",
        "shrinkage_fraction",
        "state_persistence",
        "state_interval_count",
        "trajectory_mae_mm",
        "trajectory_rmse_mm"
    ]

    write_csv(
        SOURCE_PHENOTYPE_CSV,
        source_rows,
        source_fields
    )

    # --------------------------------------------------------
    # Save target phenotype
    # --------------------------------------------------------

    target_fields = [

        "subject_id",
        "lesion_id",
        "growth_rate_per_day",
        "volume_growth_rate_per_day",
        "overall_growth_rate_per_day",
        "growth_rate_std",
        "growth_rate_median",
        "growth_rate_p10",
        "growth_rate_p90",
        "growth_fraction",
        "stable_fraction",
        "shrinkage_fraction",
        "state_persistence",
        "state_interval_count",
        "mean_interval_days",
        "total_followup_days",
        "normalized_final_diameter",
        "growth_rate_acceleration",
        "initial_diameter_mm",
        "final_diameter_mm"
    ]

    write_csv(
        TARGET_PHENOTYPE_CSV,
        target_rows,
        target_fields
    )

    # --------------------------------------------------------
    # Save vectors
    # --------------------------------------------------------

    vector_fields = [

        "domain",
        "subject_id",
        "lesion_id"
    ]

    for stat in statistics:

        vector_fields.append(
            "{}_standardized".format(
                stat[
                    "feature"
                ]
            )
        )

    write_csv(
        COMBINED_PHENOTYPE_CSV,
        vectors,
        vector_fields
    )

    # --------------------------------------------------------
    # Save statistics
    # --------------------------------------------------------

    statistics_fields = [

        "feature",
        "source_n",
        "target_n",
        "source_mean",
        "source_std",
        "source_median",
        "source_p10",
        "source_p90",
        "target_mean",
        "target_std",
        "target_median",
        "target_p10",
        "target_p90",
        "standardized_mean_difference"
    ]

    write_csv(
        STATISTICS_CSV,
        statistics,
        statistics_fields
    )

    # --------------------------------------------------------
    # Save distances
    # --------------------------------------------------------

    distance_fields = [

        "feature",
        "absolute_mean_difference",
        "pooled_standard_deviation",
        "standardized_mean_distance",
        "source_std",
        "lumiere_std",
        "lumiere_to_source_std_ratio"
    ]

    write_csv(
        DISTANCE_CSV,
        distances,
        distance_fields
    )

    # --------------------------------------------------------
    # Save states
    # --------------------------------------------------------

    state_fields = [

        "state",
        "source_mean_fraction",
        "lumiere_mean_fraction",
        "absolute_difference"
    ]

    write_csv(
        STATE_CSV,
        states,
        state_fields
    )

    # --------------------------------------------------------
    # Console
    # --------------------------------------------------------

    print(
        "\n"
    )

    print(
        "FEATURE COMPARISON"
    )

    print(
        "-" * 70
    )

    for row in statistics:

        print(
            "\n{}:".format(
                row["feature"]
            )
        )

        print(
            "  Source mean:",
            "{:.6f}".format(
                row[
                    "source_mean"
                ]
            )
        )

        print(
            "  GBM mean:",
            "{:.6f}".format(
                row[
                    "target_mean"
                ]
            )
        )

        print(
            "  Source std:",
            "{:.6f}".format(
                row[
                    "source_std"
                ]
            )
        )

        print(
            "  GBM std:",
            "{:.6f}".format(
                row[
                    "target_std"
                ]
            )
        )

        print(
            "  Standardized difference:",
            "{:.4f}".format(
                row[
                    "standardized_mean_difference"
                ]
            )
        )

    print(
        "\n"
    )

    print(
        "STATE COMPARISON"
    )

    print(
        "-" * 70
    )

    for row in states:

        print(
            "{:<12} Source={:.4f}  GBM={:.4f}  Difference={:.4f}".format(
                row["state"],
                row[
                    "source_mean_fraction"
                ],
                row[
                    "lumiere_mean_fraction"
                ],
                row[
                    "absolute_difference"
                ]
            )
        )

    # --------------------------------------------------------
    # Overall alignment score
    # --------------------------------------------------------

    absolute_distances = [

        abs(
            float(
                row[
                    "standardized_mean_distance"
                ]
            )
        )

        for row in distances

        if finite(
            row[
                "standardized_mean_distance"
            ]
        )
    ]

    overall_distance = mean(
        absolute_distances
    )

    print(
        "\nOverall mean standardized phenotype distance:",
        "{:.4f}".format(
            overall_distance
        )
    )

    # --------------------------------------------------------
    # Interpretation category
    # --------------------------------------------------------

    if overall_distance < 0.5:

        alignment = (
            "strong phenotype alignment"
        )

    elif overall_distance < 1.0:

        alignment = (
            "moderate phenotype alignment"
        )

    elif overall_distance < 2.0:

        alignment = (
            "substantial phenotype difference"
        )

    else:

        alignment = (
            "large phenotype difference"
        )

    print(
        "Preliminary interpretation:",
        alignment
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = []

    report.append(
        "# CROSS-CANCER HBGD GROWTH PHENOTYPE ALIGNMENT"
    )

    report.append("")

    report.append(
        "## 1. Objective"
    )

    report.append("")

    report.append(
        "This experiment evaluates whether growth dynamics represented by HBGD show comparable structure between the existing source-cancer cohort and the LUMIERE glioblastoma cohort."
    )

    report.append("")

    report.append(
        "This step is a phenotype-alignment analysis rather than a predictive modeling experiment."
    )

    report.append("")

    report.append(
        "## 2. Cohorts"
    )

    report.append("")

    report.append(
        "| Domain | Phenotype subjects |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        "| Source cancer | {} |".format(
            len(source_rows)
        )
    )

    report.append(
        "| LUMIERE GBM | {} |".format(
            len(target_rows)
        )
    )

    report.append("")

    report.append(
        "## 3. Comparable HBGD Features"
    )

    report.append("")

    report.append(
        "| Feature | Source mean | GBM mean | Standardized difference |"
    )

    report.append(
        "|---|---:|---:|---:|"
    )

    for row in statistics:

        report.append(
            "| {} | {:.6f} | {:.6f} | {:.4f} |".format(
                row["feature"],
                row["source_mean"],
                row["target_mean"],
                row[
                    "standardized_mean_difference"
                ]
            )
        )

    report.append("")

    report.append(
        "## 4. Dynamic-State Comparison"
    )

    report.append("")

    report.append(
        "| State | Source mean fraction | GBM mean fraction | Absolute difference |"
    )

    report.append(
        "|---|---:|---:|---:|"
    )

    for row in states:

        report.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} |".format(
                row["state"],
                row[
                    "source_mean_fraction"
                ],
                row[
                    "lumiere_mean_fraction"
                ],
                row[
                    "absolute_difference"
                ]
            )
        )

    report.append("")

    report.append(
        "## 5. Phenotype Distance"
    )

    report.append("")

    report.append(
        "The mean standardized phenotype distance across comparable features was {:.4f}.".format(
            overall_distance
        )
    )

    report.append("")

    report.append(
        "For descriptive interpretation only, the observed distance corresponds to {}.".format(
            alignment
        )
    )

    report.append("")

    report.append(
        "These categories are heuristic and should not be interpreted as statistical significance thresholds."
    )

    report.append("")

    report.append(
        "## 6. Interpretation"
    )

    report.append("")

    if overall_distance < 1.0:

        report.append(
            "The source and LUMIERE cohorts show relatively similar standardized HBGD growth-phenotype structure across the measured features. This supports proceeding to a formal representation-transfer experiment."
        )

    elif overall_distance < 2.0:

        report.append(
            "The source and LUMIERE cohorts show related but substantially different growth-phenotype distributions. This suggests that transfer may require domain normalization or adaptation rather than direct feature transfer."
        )

    else:

        report.append(
            "The source and LUMIERE cohorts show large differences in their measured growth-phenotype distributions. Direct HBGD transfer is therefore unlikely to be sufficient and a domain-adaptation strategy would be required."
        )

    report.append("")

    report.append(
        "## 7. Important Scientific Limitation"
    )

    report.append("")

    report.append(
        "Distributional similarity does not establish biological equivalence, predictive transferability, or clinical utility."
    )

    report.append("")

    report.append(
        "The source and target cohorts may differ in cancer biology, imaging protocol, segmentation characteristics, follow-up schedule, lesion definition, and cohort composition."
    )

    report.append("")

    report.append(
        "The analysis therefore provides evidence about representation structure rather than proof of cross-cancer biological transfer."
    )

    report.append("")

    report.append(
        "## 8. Next Step"
    )

    report.append("")

    report.append(
        "If the growth phenotype demonstrates sufficient common structure, the next experiment should implement a learned cross-cancer representation-transfer or domain-adaptation model."
    )

    report.append("")

    report.append(
        "The transferred representation should then be evaluated on an untouched LUMIERE patient-level test set."
    )

    report.append("")

    report.append(
        "If substantial domain differences are observed, the next model should explicitly learn cancer-specific adaptation rather than directly transferring source-cancer growth parameters."
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(
                report
            )
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
        "STEP 9 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nOUTPUT FILES:"
    )

    print(
        SOURCE_PHENOTYPE_CSV
    )

    print(
        TARGET_PHENOTYPE_CSV
    )

    print(
        COMBINED_PHENOTYPE_CSV
    )

    print(
        STATISTICS_CSV
    )

    print(
        DISTANCE_CSV
    )

    print(
        STATE_CSV
    )

    print(
        REPORT_PATH
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()