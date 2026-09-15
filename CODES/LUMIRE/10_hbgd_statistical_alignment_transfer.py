# -*- coding: utf-8 -*-

"""
10_hbgd_statistical_alignment_transfer.py

STEP 10A
    Statistical validation of cross-cancer HBGD phenotype alignment.

STEP 10B
    Cancer-adaptive HBGD phenotype transfer.

SOURCE:
    Existing source-cancer HBGD pipeline.

TARGET:
    LUMIERE glioblastoma cohort.

IMPORTANT:
    This experiment does NOT directly copy source-cancer growth
    rates to GBM.

    It:
        1. Builds comparable phenotype features.
        2. Tests source-vs-GBM alignment using permutation analysis.
        3. Estimates bootstrap confidence intervals.
        4. Learns source HBGD growth-pattern prototypes.
        5. Projects LUMIERE tumors into the source phenotype space.
        6. Generates transfer similarity features.
        7. Tests whether transfer features improve GBM prediction.

Python:
    Compatible with Python 3.6.

All outputs:
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
from sklearn.metrics import mean_squared_error


# ============================================================
# PATHS
# ============================================================

SOURCE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA"

TARGET_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"


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
    TARGET_DIR,
    "lumiere_longitudinal_trajectories.csv"
)


ALIGNMENT_CSV = os.path.join(
    TARGET_DIR,
    "step10_hbgd_alignment_statistics.csv"
)

BOOTSTRAP_CSV = os.path.join(
    TARGET_DIR,
    "step10_hbgd_bootstrap_results.csv"
)

PERMUTATION_CSV = os.path.join(
    TARGET_DIR,
    "step10_hbgd_permutation_results.csv"
)

TRANSFER_PROFILE_CSV = os.path.join(
    TARGET_DIR,
    "step10_hbgd_transfer_profiles.csv"
)

TRANSFER_RESULTS_CSV = os.path.join(
    TARGET_DIR,
    "step10_hbgd_transfer_predictions.csv"
)

TRANSFER_SUMMARY_CSV = os.path.join(
    TARGET_DIR,
    "step10_hbgd_transfer_summary.csv"
)

REPORT_PATH = os.path.join(
    TARGET_DIR,
    "STEP10_HBGD_STATISTICAL_ALIGNMENT_TRANSFER_REPORT.md"
)


# ============================================================
# SETTINGS
# ============================================================

DAYS_PER_WEEK = 7.0

RANDOM_SEED = 42

N_PERMUTATIONS = 1000

N_BOOTSTRAPS = 1000

N_PROTOTYPES = 3

N_ESTIMATORS = 300

MAX_DEPTH = 5

MIN_SAMPLES_LEAF = 2


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

    values = [
        float(v)
        for v in values
        if finite(v)
    ]

    if not values:
        return float("nan")

    return sum(values) / float(len(values))


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


def rmse(actual, predicted):

    values = []

    for a, p in zip(
        actual,
        predicted
    ):

        values.append(
            (
                float(a) -
                float(p)
            ) ** 2
        )

    return math.sqrt(
        mean(values)
    )


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

            rows.append(
                row
            )

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

            writer.writerow(
                row
            )


# ============================================================
# HBGD
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

        return "unknown"

    if rate > 0.0005:

        return "growth"

    if rate < -0.0010:

        return "shrinkage"

    return "stable"


# ============================================================
# SOURCE PHENOTYPES
# ============================================================

def build_source_phenotypes():

    growth_rows = read_csv(
        SOURCE_GROWTH
    )

    state_rows = read_csv(
        SOURCE_STATE
    )

    volume_rows = read_csv(
        SOURCE_VOLUME
    )

    growth_by_subject = {}

    for row in growth_rows:

        key = (
            str(row["patient_id"]),
            str(row["lesion_alias"])
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

            "mae":
                safe_float(
                    row[
                        "trajectory_MAE_mm"
                    ]
                ),

            "rmse":
                safe_float(
                    row[
                        "trajectory_RMSE_mm"
                    ]
                )
        }

    volume_by_subject = {}

    for row in volume_rows:

        key = (
            str(row["patient_id"]),
            str(row["lesion_alias"])
        )

        volume_by_subject[
            key
        ] = safe_float(
            row[
                "volume_growth_rate_per_day"
            ]
        )

    states_by_subject = {}

    for row in state_rows:

        key = (
            str(row["patient_id"]),
            str(row["lesion_alias"])
        )

        if key not in states_by_subject:

            states_by_subject[
                key
            ] = []

        states_by_subject[
            key
        ].append(
            (
                row[
                    "dynamic_state"
                ].strip().lower(),

                safe_float(
                    row[
                        "growth_rate"
                    ]
                )
            )
        )

    keys = set(
        growth_by_subject.keys()
    )

    keys.update(
        volume_by_subject.keys()
    )

    keys.update(
        states_by_subject.keys()
    )

    phenotypes = []

    for key in sorted(keys):

        growth = growth_by_subject.get(
            key,
            {}
        )

        volume_rate = volume_by_subject.get(
            key,
            float("nan")
        )

        state_values = states_by_subject.get(
            key,
            []
        )

        states = [
            x[0]
            for x in state_values
        ]

        state_rates = [
            x[1]
            for x in state_values
            if finite(x[1])
        ]

        n = len(states)

        if n > 0:

            growth_fraction = (
                states.count(
                    "growth"
                )
                /
                float(n)
            )

            stable_fraction = (
                states.count(
                    "stable"
                )
                /
                float(n)
            )

            shrink_fraction = (
                states.count(
                    "shrinkage"
                )
                /
                float(n)
            )

        else:

            growth_fraction = 0.0

            stable_fraction = 0.0

            shrink_fraction = 0.0

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

            persistence = (
                persistent /
                float(transitions)
            )

        else:

            persistence = 0.0

        phenotypes.append({

            "subject_id":
                key[0],

            "lesion_id":
                key[1],

            "growth_rate":
                growth.get(
                    "growth_rate",
                    float("nan")
                ),

            "volume_growth_rate":
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
                persistence,

            "state_count":
                n,

            "trajectory_mae":
                growth.get(
                    "mae",
                    float("nan")
                ),

            "trajectory_rmse":
                growth.get(
                    "rmse",
                    float("nan")
                )
        })

    return phenotypes


# ============================================================
# LUMIERE PHENOTYPES
# ============================================================

def build_lumiere_phenotypes():

    rows = read_csv(
        TARGET_TRAJECTORY
    )

    patients = {}

    for row in rows:

        patient = row[
            "patient_id"
        ]

        if patient not in patients:

            patients[
                patient
            ] = []

        patients[
            patient
        ].append({

            "week":
                int(
                    float(
                        row[
                            "week"
                        ]
                    )
                ),

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

        intervals = []

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

            delta_days = (
                obs[i + 1]["week"] -
                obs[i]["week"]
            ) * DAYS_PER_WEEK

            r = calculate_rate(
                d1,
                d2,
                delta_days
            )

            if finite(r):

                rates.append(r)

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

                intervals.append(
                    delta_days
                )

        if not rates:

            continue

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

        n = len(states)

        growth_fraction = (
            states.count("growth")
            /
            float(n)
        )

        stable_fraction = (
            states.count("stable")
            /
            float(n)
        )

        shrink_fraction = (
            states.count("shrinkage")
            /
            float(n)
        )

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

            persistence = (
                persistent /
                float(transitions)
            )

        else:

            persistence = 0.0

        # Growth acceleration

        acceleration = 0.0

        if len(rates) >= 2:

            midpoint = (
                len(rates) // 2
            )

            first_half = rates[
                :midpoint
            ]

            second_half = rates[
                midpoint:
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

        normalized_final = (

            last_d /
            first_d

            if first_d > 0

            else float("nan")
        )

        phenotypes.append({

            "subject_id":
                patient,

            "lesion_id":
                "whole_tumor",

            "growth_rate":
                mean(rates),

            "volume_growth_rate":
                mean(
                    volume_rates
                ),

            "state_rate_mean":
                mean(rates),

            "state_rate_std":
                std(rates),

            "growth_fraction":
                growth_fraction,

            "stable_fraction":
                stable_fraction,

            "shrinkage_fraction":
                shrink_fraction,

            "state_persistence":
                persistence,

            "state_count":
                n,

            "overall_growth_rate":
                overall_rate,

            "growth_rate_acceleration":
                acceleration,

            "normalized_final":
                normalized_final,

            "mean_interval_days":
                mean(intervals),

            "total_followup_days":
                total_days
        })

    return phenotypes


# ============================================================
# COMPARABLE FEATURES
# ============================================================

FEATURES = [

    "growth_rate",

    "volume_growth_rate",

    "state_rate_mean",

    "state_rate_std",

    "growth_fraction",

    "stable_fraction",

    "shrinkage_fraction",

    "state_persistence"
]


# ============================================================
# CONVERT TO MATRIX
# ============================================================

def matrix_from_rows(
    rows
):

    matrix = []

    for row in rows:

        values = []

        for feature in FEATURES:

            value = safe_float(
                row.get(
                    feature,
                    float("nan")
                )
            )

            if not finite(value):

                value = 0.0

            values.append(
                value
            )

        matrix.append(
            values
        )

    return np.asarray(
        matrix,
        dtype=float
    )


# ============================================================
# STANDARDIZATION
# ============================================================

def fit_standardizer(
    X
):

    means = np.mean(
        X,
        axis=0
    )

    stds = np.std(
        X,
        axis=0,
        ddof=1
    )

    stds[
        stds == 0
    ] = 1.0

    return (
        means,
        stds
    )


def transform(
    X,
    means,
    stds
):

    return (
        X -
        means
    ) / stds


# ============================================================
# ALIGNMENT SCORE
# ============================================================

def alignment_score(
    source_X,
    target_X
):

    source_mean = np.mean(
        source_X,
        axis=0
    )

    target_mean = np.mean(
        target_X,
        axis=0
    )

    distance = np.sqrt(
        np.mean(
            (
                target_mean -
                source_mean
            ) ** 2
        )
    )

    return float(
        distance
    )


# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test(
    source_X,
    target_X,
    n_permutations
):

    combined = np.vstack(
        [
            source_X,
            target_X
        ]
    )

    n_source = source_X.shape[
        0
    ]

    observed = alignment_score(
        source_X,
        target_X
    )

    rng = np.random.RandomState(
        RANDOM_SEED
    )

    permutation_scores = []

    for i in range(
        n_permutations
    ):

        indices = rng.permutation(
            combined.shape[0]
        )

        perm_source = combined[
            indices[
                :n_source
            ]
        ]

        perm_target = combined[
            indices[
                n_source:
            ]
        ]

        score = alignment_score(
            perm_source,
            perm_target
        )

        permutation_scores.append(
            score
        )

    permutation_scores = np.asarray(
        permutation_scores
    )

    # Smaller distance = stronger alignment.
    p_value = (
        np.sum(
            permutation_scores
            <=
            observed
        )
        + 1.0
    ) / (
        float(
            n_permutations
        )
        + 1.0
    )

    return (
        observed,
        permutation_scores,
        p_value
    )


# ============================================================
# BOOTSTRAP
# ============================================================

def bootstrap_alignment(
    source_X,
    target_X,
    n_bootstraps
):

    rng = np.random.RandomState(
        RANDOM_SEED + 100
    )

    scores = []

    for i in range(
        n_bootstraps
    ):

        source_indices = rng.randint(
            0,
            source_X.shape[0],
            source_X.shape[0]
        )

        target_indices = rng.randint(
            0,
            target_X.shape[0],
            target_X.shape[0]
        )

        source_sample = source_X[
            source_indices
        ]

        target_sample = target_X[
            target_indices
        ]

        score = alignment_score(
            source_sample,
            target_sample
        )

        scores.append(
            score
        )

    scores = np.asarray(
        scores
    )

    return {

        "mean":
            float(
                np.mean(scores)
            ),

        "median":
            float(
                np.median(scores)
            ),

        "ci_lower":
            float(
                np.percentile(
                    scores,
                    2.5
                )
            ),

        "ci_upper":
            float(
                np.percentile(
                    scores,
                    97.5
                )
            ),

        "std":
            float(
                np.std(
                    scores
                )
            )
    }


# ============================================================
# PROTOTYPES
# ============================================================

def build_prototypes(
    source_X
):

    """
    Simple deterministic prototype construction.

    Prototype 1:
        mean of source phenotype.

    Prototype 2:
        source subject closest to mean.

    Prototype 3:
        source subject farthest from mean.

    This avoids requiring additional clustering packages
    and remains compatible with Python 3.6 environments.
    """

    center = np.mean(
        source_X,
        axis=0
    )

    distances = np.sqrt(
        np.sum(
            (
                source_X -
                center
            ) ** 2,
            axis=1
        )
    )

    closest = int(
        np.argmin(
            distances
        )
    )

    farthest = int(
        np.argmax(
            distances
        )
    )

    prototypes = np.vstack(
        [
            center,
            source_X[closest],
            source_X[farthest]
        ]
    )

    return prototypes


# ============================================================
# DISTANCE FEATURES
# ============================================================

def prototype_features(
    X,
    prototypes
):

    output = []

    for row in X:

        distances = []

        for prototype in prototypes:

            distance = math.sqrt(
                np.sum(
                    (
                        row -
                        prototype
                    ) ** 2
                )
            )

            distances.append(
                distance
            )

        sorted_distances = sorted(
            distances
        )

        output.append(
            distances +
            [
                sorted_distances[0],
                sorted_distances[1],
                mean(distances)
            ]
        )

    return np.asarray(
        output,
        dtype=float
    )


# ============================================================
# TARGET PREDICTION DATA
# ============================================================

def build_prediction_cases(
    patients,
    patient_ids,
    source_means,
    source_stds,
    prototypes
):

    cases = []

    for patient_id in patient_ids:

        observations = patients[
            patient_id
        ]

        if len(observations) < 3:

            continue

        for target_index in range(
            2,
            len(observations)
        ):

            history = observations[
                :target_index
            ]

            current = history[
                -1
            ]

            previous = history[
                -2
            ]

            target = observations[
                target_index
            ]

            current_d = current[
                "diameter"
            ]

            current_v = current[
                "volume"
            ]

            previous_d = previous[
                "diameter"
            ]

            previous_v = previous[
                "volume"
            ]

            future_weeks = (
                target["week"] -
                current["week"]
            )

            previous_weeks = (
                current["week"] -
                previous["week"]
            )

            if (
                current_d <= 0
                or
                previous_d <= 0
                or
                target["diameter"] <= 0
                or
                future_weeks <= 0
                or
                previous_weeks <= 0
            ):

                continue

            recent_r = calculate_rate(
                previous_d,
                current_d,
                previous_weeks *
                DAYS_PER_WEEK
            )

            first_d = history[0][
                "diameter"
            ]

            total_weeks = (
                current["week"] -
                history[0]["week"]
            )

            overall_r = calculate_rate(
                first_d,
                current_d,
                total_weeks *
                DAYS_PER_WEEK
            )

            rates = []

            for i in range(
                len(history) - 1
            ):

                d1 = history[i][
                    "diameter"
                ]

                d2 = history[i + 1][
                    "diameter"
                ]

                dw = (
                    history[i + 1]["week"] -
                    history[i]["week"]
                )

                r = calculate_rate(
                    d1,
                    d2,
                    dw *
                    DAYS_PER_WEEK
                )

                if finite(r):

                    rates.append(r)

            multi_r = (
                mean(rates)
                if rates
                else 0.0
            )

            rate_std = (
                std(rates)
                if len(rates) > 1
                else 0.0
            )

            state = state_code(
                recent_r
            )

            geometry = [

                current_d,

                current_v,

                previous_d,

                previous_v,

                previous_weeks,

                current_d -
                previous_d,

                current_v -
                previous_v,

                len(history),

                total_weeks
            ]

            # HBGD phenotype

            phenotype = np.asarray(
                [
                    recent_r,
                    multi_r,
                    recent_r,
                    rate_std,
                    1.0 if state > 0 else 0.0,
                    1.0 if state == 0 else 0.0,
                    1.0 if state < 0 else 0.0,
                    0.0
                ],
                dtype=float
            )

            phenotype = (
                phenotype -
                source_means
            ) / source_stds

            distances = prototype_features(
                phenotype.reshape(
                    1,
                    -1
                ),
                prototypes
            )[0]

            cases.append({

                "patient_id":
                    patient_id,

                "target_week":
                    target["week"],

                "future_delta_weeks":
                    future_weeks,

                "actual":
                    target[
                        "diameter"
                    ],

                "geometry":
                    geometry,

                "transfer":
                    distances.tolist()
            })

    return cases


def state_code(
    rate
):

    if not finite(rate):

        return 0.0

    if rate > 0.0005:

        return 1.0

    if rate < -0.0010:

        return -1.0

    return 0.0


# ============================================================
# LOAD LUMIERE PATIENTS
# ============================================================

def load_lumiere_patients():

    rows = read_csv(
        TARGET_TRAJECTORY
    )

    patients = {}

    for row in rows:

        patient = row[
            "patient_id"
        ]

        if patient not in patients:

            patients[
                patient
            ] = []

        patients[
            patient
        ].append({

            "week":
                int(
                    float(
                        row[
                            "week"
                        ]
                    )
                ),

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
        })

    for patient in patients:

        patients[
            patient
        ].sort(
            key=lambda x:
                x["week"]
        )

    return patients


# ============================================================
# MODEL
# ============================================================

def train_rf(
    X,
    y
):

    model = RandomForestRegressor(

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        min_samples_leaf=MIN_SAMPLES_LEAF,

        random_state=RANDOM_SEED,

        n_jobs=-1
    )

    model.fit(
        X,
        y
    )

    return model


def evaluate(
    model,
    X,
    y
):

    prediction = model.predict(
        X
    )

    return (

        prediction,

        mean_absolute_error(
            y,
            prediction
        ),

        rmse(
            y,
            prediction
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "STEP 10 - STATISTICAL ALIGNMENT + HBGD TRANSFER"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Build phenotypes
    # --------------------------------------------------------

    print(
        "\nBuilding source phenotypes..."
    )

    source_rows = (
        build_source_phenotypes()
    )

    print(
        "Source subjects:",
        len(source_rows)
    )

    print(
        "\nBuilding LUMIERE phenotypes..."
    )

    target_rows = (
        build_lumiere_phenotypes()
    )

    print(
        "LUMIERE subjects:",
        len(target_rows)
    )

    # --------------------------------------------------------
    # Matrices
    # --------------------------------------------------------

    source_raw = matrix_from_rows(
        source_rows
    )

    target_raw = matrix_from_rows(
        target_rows
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Standardization is fitted on SOURCE only.
    # --------------------------------------------------------

    source_means, source_stds = (
        fit_standardizer(
            source_raw
        )
    )

    source_X = transform(
        source_raw,
        source_means,
        source_stds
    )

    target_X = transform(
        target_raw,
        source_means,
        source_stds
    )

    # --------------------------------------------------------
    # Step 10A
    # --------------------------------------------------------

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "STEP 10A - STATISTICAL ALIGNMENT"
    )

    print(
        "=" * 70
    )

    observed, permutation_scores, p_value = (
        permutation_test(
            source_X,
            target_X,
            N_PERMUTATIONS
        )
    )

    bootstrap = bootstrap_alignment(
        source_X,
        target_X,
        N_BOOTSTRAPS
    )

    print(
        "\nObserved standardized phenotype distance:",
        "{:.6f}".format(
            observed
        )
    )

    print(
        "Permutation p-value:",
        "{:.6f}".format(
            p_value
        )
    )

    print(
        "Bootstrap mean:",
        "{:.6f}".format(
            bootstrap["mean"]
        )
    )

    print(
        "Bootstrap 95% CI:",
        "[{:.6f}, {:.6f}]".format(
            bootstrap["ci_lower"],
            bootstrap["ci_upper"]
        )
    )

    # --------------------------------------------------------
    # Feature-level statistics
    # --------------------------------------------------------

    alignment_rows = []

    for i, feature in enumerate(
        FEATURES
    ):

        source_values = source_raw[
            :,
            i
        ]

        target_values = target_raw[
            :,
            i
        ]

        source_mean = mean(
            source_values
        )

        target_mean = mean(
            target_values
        )

        pooled_std = math.sqrt(
            (
                std(source_values) ** 2
                +
                std(target_values) ** 2
            )
            /
            2.0
        )

        if pooled_std > 0:

            effect = (
                target_mean -
                source_mean
            ) / pooled_std

        else:

            effect = 0.0

        alignment_rows.append({

            "feature":
                feature,

            "source_mean":
                source_mean,

            "source_std":
                std(
                    source_values
                ),

            "target_mean":
                target_mean,

            "target_std":
                std(
                    target_values
                ),

            "standardized_mean_difference":
                effect
        })

    write_csv(
        ALIGNMENT_CSV,
        alignment_rows,
        [
            "feature",
            "source_mean",
            "source_std",
            "target_mean",
            "target_std",
            "standardized_mean_difference"
        ]
    )

    # --------------------------------------------------------
    # Save bootstrap
    # --------------------------------------------------------

    bootstrap_rows = [

        {
            "metric":
                "alignment_distance",

            "observed":
                observed,

            "bootstrap_mean":
                bootstrap["mean"],

            "bootstrap_median":
                bootstrap["median"],

            "bootstrap_std":
                bootstrap["std"],

            "ci_lower_2_5":
                bootstrap["ci_lower"],

            "ci_upper_97_5":
                bootstrap["ci_upper"]
        }
    ]

    write_csv(
        BOOTSTRAP_CSV,
        bootstrap_rows,
        [
            "metric",
            "observed",
            "bootstrap_mean",
            "bootstrap_median",
            "bootstrap_std",
            "ci_lower_2_5",
            "ci_upper_97_5"
        ]
    )

    # --------------------------------------------------------
    # Save permutation sample
    # --------------------------------------------------------

    permutation_rows = []

    for i, score in enumerate(
        permutation_scores
    ):

        permutation_rows.append({

            "permutation":
                i + 1,

            "alignment_distance":
                float(score)
        })

    write_csv(
        PERMUTATION_CSV,
        permutation_rows,
        [
            "permutation",
            "alignment_distance"
        ]
    )

    # --------------------------------------------------------
    # Step 10B
    # --------------------------------------------------------

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "STEP 10B - CANCER-ADAPTIVE HBGD TRANSFER"
    )

    print(
        "=" * 70
    )

    prototypes = build_prototypes(
        source_X
    )

    # --------------------------------------------------------
    # Save prototype profile
    # --------------------------------------------------------

    profile_rows = []

    for p in range(
        prototypes.shape[0]
    ):

        for i, feature in enumerate(
            FEATURES
        ):

            profile_rows.append({

                "prototype":
                    p + 1,

                "feature":
                    feature,

                "standardized_value":
                    float(
                        prototypes[p, i]
                    )
            })

    write_csv(
        TRANSFER_PROFILE_CSV,
        profile_rows,
        [
            "prototype",
            "feature",
            "standardized_value"
        ]
    )

    # --------------------------------------------------------
    # Patient split
    # --------------------------------------------------------

    patients = (
        load_lumiere_patients()
    )

    patient_ids = sorted(
        patients.keys()
    )

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
    # Build prediction cases
    # --------------------------------------------------------

    train_cases = build_prediction_cases(
        patients,
        train_ids,
        source_means,
        source_stds,
        prototypes
    )

    validation_cases = build_prediction_cases(
        patients,
        validation_ids,
        source_means,
        source_stds,
        prototypes
    )

    test_cases = build_prediction_cases(
        patients,
        test_ids,
        source_means,
        source_stds,
        prototypes
    )

    print(
        "\nPrediction cases:"
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
    # Arrays
    # --------------------------------------------------------

    Xg_train = np.asarray([
        x["geometry"]
        for x in train_cases
    ], dtype=float)

    Xg_val = np.asarray([
        x["geometry"]
        for x in validation_cases
    ], dtype=float)

    Xg_test = np.asarray([
        x["geometry"]
        for x in test_cases
    ], dtype=float)

    Xt_train = np.asarray([
        x["transfer"]
        for x in train_cases
    ], dtype=float)

    Xt_val = np.asarray([
        x["transfer"]
        for x in validation_cases
    ], dtype=float)

    Xt_test = np.asarray([
        x["transfer"]
        for x in test_cases
    ], dtype=float)

    y_train = np.asarray([
        x["actual"]
        for x in train_cases
    ], dtype=float)

    y_val = np.asarray([
        x["actual"]
        for x in validation_cases
    ], dtype=float)

    y_test = np.asarray([
        x["actual"]
        for x in test_cases
    ], dtype=float)

    # --------------------------------------------------------
    # Combined
    # --------------------------------------------------------

    Xc_train = np.concatenate(
        [
            Xg_train,
            Xt_train
        ],
        axis=1
    )

    Xc_val = np.concatenate(
        [
            Xg_val,
            Xt_val
        ],
        axis=1
    )

    Xc_test = np.concatenate(
        [
            Xg_test,
            Xt_test
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Train/evaluate
    # --------------------------------------------------------

    experiments = [

        (
            "Geometry_only",
            Xg_train,
            Xg_val,
            Xg_test
        ),

        (
            "Transferred_HBGD_only",
            Xt_train,
            Xt_val,
            Xt_test
        ),

        (
            "Geometry_plus_transferred_HBGD",
            Xc_train,
            Xc_val,
            Xc_test
        )
    ]

    summary_rows = []

    prediction_rows = []

    for (
        name,
        X_train,
        X_val,
        X_test
    ) in experiments:

        print(
            "\n"
        )

        print(
            "-" * 70
        )

        print(
            "MODEL:",
            name
        )

        model = train_rf(
            X_train,
            y_train
        )

        validation_prediction, validation_mae, validation_rmse = (
            evaluate(
                model,
                X_val,
                y_val
            )
        )

        test_prediction, test_mae, test_rmse = (
            evaluate(
                model,
                X_test,
                y_test
            )
        )

        print(
            "Validation MAE:",
            "{:.4f} mm".format(
                validation_mae
            )
        )

        print(
            "Validation RMSE:",
            "{:.4f} mm".format(
                validation_rmse
            )
        )

        print(
            "Test MAE:",
            "{:.4f} mm".format(
                test_mae
            )
        )

        print(
            "Test RMSE:",
            "{:.4f} mm".format(
                test_rmse
            )
        )

        summary_rows.append({

            "model":
                name,

            "train_patients":
                len(train_ids),

            "validation_patients":
                len(validation_ids),

            "test_patients":
                len(test_ids),

            "train_cases":
                len(train_cases),

            "validation_cases":
                len(validation_cases),

            "test_cases":
                len(test_cases),

            "validation_mae_mm":
                validation_mae,

            "validation_rmse_mm":
                validation_rmse,

            "test_mae_mm":
                test_mae,

            "test_rmse_mm":
                test_rmse
        })

        for case, prediction in zip(
            test_cases,
            test_prediction
        ):

            prediction_rows.append({

                "patient_id":
                    case[
                        "patient_id"
                    ],

                "target_week":
                    case[
                        "target_week"
                    ],

                "future_delta_weeks":
                    case[
                        "future_delta_weeks"
                    ],

                "actual_future_diameter_mm":
                    case[
                        "actual"
                    ],

                "model":
                    name,

                "predicted_future_diameter_mm":
                    float(
                        prediction
                    ),

                "absolute_error_mm":
                    abs(
                        case[
                            "actual"
                        ] -
                        prediction
                    )
            })

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    write_csv(
        TRANSFER_RESULTS_CSV,
        prediction_rows,
        [
            "patient_id",
            "target_week",
            "future_delta_weeks",
            "actual_future_diameter_mm",
            "model",
            "predicted_future_diameter_mm",
            "absolute_error_mm"
        ]
    )

    write_csv(
        TRANSFER_SUMMARY_CSV,
        summary_rows,
        [
            "model",
            "train_patients",
            "validation_patients",
            "test_patients",
            "train_cases",
            "validation_cases",
            "test_cases",
            "validation_mae_mm",
            "validation_rmse_mm",
            "test_mae_mm",
            "test_rmse_mm"
        ]
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    geometry_row = None

    transfer_row = None

    combined_row = None

    for row in summary_rows:

        if row["model"] == "Geometry_only":

            geometry_row = row

        elif row["model"] == "Transferred_HBGD_only":

            transfer_row = row

        elif row["model"] == "Geometry_plus_transferred_HBGD":

            combined_row = row

    report = []

    report.append(
        "# STEP 10 - HBGD STATISTICAL ALIGNMENT AND CROSS-CANCER TRANSFER"
    )

    report.append("")

    report.append(
        "## 1. Objective"
    )

    report.append("")

    report.append(
        "Step 10 evaluates whether the cross-cancer HBGD phenotype similarity observed in Step 9 is statistically robust and whether a source-cancer growth-pattern representation can provide useful transfer information for LUMIERE GBM."
    )

    report.append("")

    report.append(
        "## 2. Statistical Alignment"
    )

    report.append("")

    report.append(
        "| Metric | Value |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        "| Observed standardized phenotype distance | {:.6f} |".format(
            observed
        )
    )

    report.append(
        "| Permutation p-value | {:.6f} |".format(
            p_value
        )
    )

    report.append(
        "| Bootstrap mean distance | {:.6f} |".format(
            bootstrap["mean"]
        )
    )

    report.append(
        "| Bootstrap 95% CI lower | {:.6f} |".format(
            bootstrap["ci_lower"]
        )
    )

    report.append(
        "| Bootstrap 95% CI upper | {:.6f} |".format(
            bootstrap["ci_upper"]
        )
    )

    report.append("")

    report.append(
        "A smaller phenotype distance indicates greater similarity between the source and target domains."
    )

    report.append("")

    report.append(
        "## 3. Feature-Level Comparison"
    )

    report.append("")

    report.append(
        "| Feature | Source mean | GBM mean | Standardized difference |"
    )

    report.append(
        "|---|---:|---:|---:|"
    )

    for row in alignment_rows:

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
        "## 4. Transfer Experiment"
    )

    report.append("")

    report.append(
        "| Model | Validation MAE | Test MAE | Test RMSE |"
    )

    report.append(
        "|---|---:|---:|---:|"
    )

    for row in summary_rows:

        report.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} |".format(
                row["model"],
                row[
                    "validation_mae_mm"
                ],
                row[
                    "test_mae_mm"
                ],
                row[
                    "test_rmse_mm"
                ]
            )
        )

    report.append("")

    report.append(
        "## 5. Transfer Interpretation"
    )

    report.append("")

    if (
        geometry_row is not None
        and
        combined_row is not None
    ):

        geometry_mae = float(
            geometry_row[
                "test_mae_mm"
            ]
        )

        combined_mae = float(
            combined_row[
                "test_mae_mm"
            ]
        )

        if combined_mae < geometry_mae:

            improvement = (
                (
                    geometry_mae -
                    combined_mae
                )
                /
                geometry_mae
            ) * 100.0

            report.append(
                "The geometry + transferred-HBGD model improved test MAE over geometry-only by {:.2f}%. This provides preliminary evidence that the transferred growth phenotype contains predictive information beyond the geometry features used in the baseline.".format(
                    improvement
                )
            )

        else:

            worsening = (
                (
                    combined_mae -
                    geometry_mae
                )
                /
                geometry_mae
            ) * 100.0

            report.append(
                "The geometry + transferred-HBGD model did not improve the held-out test MAE. It was {:.2f}% worse than the geometry-only baseline. Therefore, this experiment does not establish predictive benefit from the transferred representation.".format(
                    worsening
                )
            )

    report.append("")

    report.append(
        "## 6. Statistical Caution"
    )

    report.append("")

    report.append(
        "The source phenotype cohort is small. Therefore, alignment estimates and transfer performance should be considered preliminary."
    )

    report.append("")

    report.append(
        "The permutation analysis tests the observed feature-space distance under random reassignment of domain labels; it does not establish biological equivalence."
    )

    report.append("")

    report.append(
        "The bootstrap interval reflects resampling uncertainty within the available cohorts and should not be interpreted as a clinical confidence interval."
    )

    report.append("")

    report.append(
        "## 7. Temporal Assumption"
    )

    report.append("")

    report.append(
        "LUMIERE week identifiers are converted using seven days per week. This assumption should be verified against the official dataset documentation before treating rates as clinically calibrated temporal measurements."
    )

    report.append("")

    report.append(
        "## 8. Scientific Conclusion"
    )

    report.append("")

    report.append(
        "Step 10 separates phenotype alignment from predictive transfer. Similarity between source and GBM growth phenotypes is evaluated independently from whether transferred information improves future-diameter prediction."
    )

    report.append("")

    report.append(
        "A positive transfer result would support further development of a cancer-adaptive HBGD representation. A negative transfer result would indicate that the current source-to-GBM mapping is insufficient and that more explicit domain adaptation is required."
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report)
        )

    # --------------------------------------------------------
    # Final console
    # --------------------------------------------------------

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "STEP 10 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nSTATISTICAL ALIGNMENT"
    )

    print(
        "Observed distance:",
        "{:.6f}".format(
            observed
        )
    )

    print(
        "Permutation p-value:",
        "{:.6f}".format(
            p_value
        )
    )

    print(
        "Bootstrap 95% CI:",
        "[{:.6f}, {:.6f}]".format(
            bootstrap["ci_lower"],
            bootstrap["ci_upper"]
        )
    )

    print(
        "\nTRANSFER RESULTS"
    )

    print(
        "-" * 70
    )

    for row in summary_rows:

        print(
            "{:<38} Validation MAE={:.4f}  Test MAE={:.4f}  Test RMSE={:.4f}".format(
                row["model"],
                row[
                    "validation_mae_mm"
                ],
                row[
                    "test_mae_mm"
                ],
                row[
                    "test_rmse_mm"
                ]
            )
        )

    print(
        "-" * 70
    )

    print(
        "\nOUTPUT FILES:"
    )

    print(
        ALIGNMENT_CSV
    )

    print(
        BOOTSTRAP_CSV
    )

    print(
        PERMUTATION_CSV
    )

    print(
        TRANSFER_PROFILE_CSV
    )

    print(
        TRANSFER_RESULTS_CSV
    )

    print(
        TRANSFER_SUMMARY_CSV
    )

    print(
        REPORT_PATH
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()