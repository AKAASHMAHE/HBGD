# -*- coding: utf-8 -*-

"""
STEP 11 - DOMAIN-ADAPTIVE HBGD TRANSFER

Goal
----
Test whether an HBGD growth predictor learned from the
source cancer can be adapted to LUMIERE glioblastoma.

This experiment deliberately avoids assuming that source
and GBM growth-rate distributions are identical.

Pipeline
--------
SOURCE CANCER
    longitudinal trajectories
        |
        v
source HBGD prediction model
        |
        v
source prediction applied to GBM
        |
        v
GBM TRAIN patients
        |
        v
learn target-domain residual correction
        |
        v
adapted HBGD prediction
        |
        v
untouched GBM TEST patients

Models
------
1. Geometry-only target model
2. Source HBGD direct transfer
3. Domain-adapted HBGD
4. Geometry + adapted HBGD

Patient split
-------------
47 train
7 validation
13 test

The test patients are never used to fit the target-domain
adaptation model.

Python
------
Compatible with Python 3.6.

Output directory
----------------
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


SOURCE_TRAJECTORY = os.path.join(
    SOURCE_DIR,
    "17_hbgd_diameter_trajectories.csv"
)

TARGET_TRAJECTORY = os.path.join(
    TARGET_DIR,
    "lumiere_longitudinal_trajectories.csv"
)


SOURCE_GROWTH = os.path.join(
    SOURCE_DIR,
    "17_hbgd_growth_parameters.csv"
)


OUTPUT_SUMMARY = os.path.join(
    TARGET_DIR,
    "step11_domain_adaptive_transfer_summary.csv"
)

OUTPUT_PREDICTIONS = os.path.join(
    TARGET_DIR,
    "step11_domain_adaptive_transfer_predictions.csv"
)

OUTPUT_SOURCE_MODEL = os.path.join(
    TARGET_DIR,
    "step11_source_model_performance.csv"
)

OUTPUT_ADAPTATION = os.path.join(
    TARGET_DIR,
    "step11_domain_adaptation_parameters.csv"
)

OUTPUT_REPORT = os.path.join(
    TARGET_DIR,
    "STEP11_DOMAIN_ADAPTIVE_HBGD_TRANSFER_REPORT.md"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

N_ESTIMATORS = 300

MAX_DEPTH = 6

MIN_SAMPLES_LEAF = 2

DAYS_PER_WEEK = 7.0


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


def rmse(
    actual,
    predicted
):

    errors = []

    for a, p in zip(
        actual,
        predicted
    ):

        errors.append(
            (
                float(a) -
                float(p)
            ) ** 2
        )

    return math.sqrt(
        mean(errors)
    )


# ============================================================
# HBGD
# ============================================================

def hbgd_rate(
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


def hbgd_predict(
    current_diameter,
    rate,
    future_days
):

    if (
        not finite(current_diameter)
        or
        not finite(rate)
        or
        current_diameter <= 0
        or
        future_days <= 0
    ):

        return float("nan")

    return (
        current_diameter *
        math.exp(
            rate *
            future_days
        )
    )


# ============================================================
# SOURCE DATA
# ============================================================

def load_source_patients():

    rows = read_csv(
        SOURCE_TRAJECTORY
    )

    patients = {}

    for row in rows:

        patient = str(
            row[
                "patient_id"
            ]
        )

        lesion = str(
            row[
                "lesion_alias"
            ]
        )

        key = (
            patient,
            lesion
        )

        if key not in patients:

            patients[
                key
            ] = []

        time_days = safe_float(
            row[
                "time_days"
            ]
        )

        observed = safe_float(
            row[
                "observed_diameter_mm"
            ]
        )

        if (
            finite(time_days)
            and
            finite(observed)
            and
            observed > 0
        ):

            patients[
                key
            ].append({

                "time":
                    time_days,

                "diameter":
                    observed
            })

    for key in patients:

        patients[
            key
        ].sort(
            key=lambda x:
                x["time"]
        )

    return patients


# ============================================================
# BUILD SOURCE SUPERVISED CASES
# ============================================================

def build_source_cases():

    patients = (
        load_source_patients()
    )

    cases = []

    for key in patients:

        obs = patients[
            key
        ]

        if len(obs) < 2:

            continue

        for i in range(
            1,
            len(obs)
        ):

            previous = obs[
                i - 1
            ]

            current = obs[
                i
            ]

            delta_days = (
                current["time"] -
                previous["time"]
            )

            if delta_days <= 0:

                continue

            rate = hbgd_rate(
                previous[
                    "diameter"
                ],
                current[
                    "diameter"
                ],
                delta_days
            )

            if not finite(rate):

                continue

            # Predict the current observation
            # from the previous observation.

            hbgd_prediction = hbgd_predict(
                previous[
                    "diameter"
                ],
                rate,
                delta_days
            )

            cases.append({

                "patient_id":
                    key[0],

                "lesion_id":
                    key[1],

                "current_diameter":
                    previous[
                        "diameter"
                    ],

                "previous_diameter":
                    (
                        obs[i - 2][
                            "diameter"
                        ]
                        if i >= 2
                        else previous[
                            "diameter"
                        ]
                    ),

                "growth_rate":
                    rate,

                "delta_days":
                    delta_days,

                "hbgd_prediction":
                    hbgd_prediction,

                "actual":
                    current[
                        "diameter"
                    ]
            })

    return cases


# ============================================================
# TARGET DATA
# ============================================================

def load_target_patients():

    rows = read_csv(
        TARGET_TRAJECTORY
    )

    patients = {}

    for row in rows:

        patient = str(
            row[
                "patient_id"
            ]
        )

        if patient not in patients:

            patients[
                patient
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

    return patients


# ============================================================
# TARGET SUPERVISED CASES
# ============================================================

def build_target_cases(
    patients,
    patient_ids
):

    cases = []

    for patient_id in patient_ids:

        if patient_id not in patients:

            continue

        obs = patients[
            patient_id
        ]

        if len(obs) < 2:

            continue

        for i in range(
            1,
            len(obs)
        ):

            previous = obs[
                i - 1
            ]

            current = obs[
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

            current_d = previous[
                "diameter"
            ]

            target_d = current[
                "diameter"
            ]

            if (
                current_d <= 0
                or
                target_d <= 0
            ):

                continue

            rate = hbgd_rate(
                current_d,
                target_d,
                delta_days
            )

            if not finite(rate):

                continue

            direct_hbgd_prediction = hbgd_predict(
                current_d,
                rate,
                delta_days
            )

            previous_d = (
                obs[i - 2][
                    "diameter"
                ]
                if i >= 2
                else current_d
            )

            # Additional history features

            history_length = i + 1

            total_history_weeks = (
                current["week"] -
                obs[0]["week"]
            )

            if total_history_weeks < 0:

                total_history_weeks = 0

            if i >= 2:

                old_rate = hbgd_rate(
                    previous_d,
                    current_d,
                    (
                        previous["week"] -
                        obs[i - 2]["week"]
                    ) *
                    DAYS_PER_WEEK
                )

            else:

                old_rate = rate

            if not finite(old_rate):

                old_rate = rate

            rate_change = (
                rate -
                old_rate
            )

            geometry = [

                current_d,

                current["volume"]
                if finite(
                    current["volume"]
                )
                else 0.0,

                previous_d,

                (
                    current_d -
                    previous_d
                ),

                delta_weeks,

                history_length,

                total_history_weeks,

                rate,

                old_rate,

                rate_change
            ]

            cases.append({

                "patient_id":
                    patient_id,

                "current_week":
                    current["week"],

                "future_week":
                    current["week"],

                "future_delta_weeks":
                    delta_weeks,

                "current_diameter":
                    current_d,

                "actual":
                    target_d,

                "hbgd_rate":
                    rate,

                "direct_hbgd_prediction":
                    direct_hbgd_prediction,

                "geometry":
                    geometry
            })

    return cases


# ============================================================
# IMPORTANT TARGET CASE BUILDER
# ============================================================

def build_prediction_cases(
    patients,
    patient_ids
):

    """
    Each case uses all observations before the target
    and predicts the next observed timepoint.

    This prevents the future target diameter from being
    used as an input feature.
    """

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

            # Recent HBGD rate is calculated ONLY
            # from historical observations.

            if target_index >= 2:

                previous = obs[
                    target_index - 2
                ]

                previous_d = previous[
                    "diameter"
                ]

                previous_delta_days = (
                    current["week"] -
                    previous["week"]
                ) * DAYS_PER_WEEK

                recent_rate = hbgd_rate(
                    previous_d,
                    current_d,
                    previous_delta_days
                )

            else:

                previous_d = current_d

                recent_rate = 0.0

            if not finite(recent_rate):

                recent_rate = 0.0

            # Multi-point historical HBGD rate

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

                r = hbgd_rate(
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

                rate_std = (
                    float(
                        np.std(
                            historical_rates
                        )
                    )
                    if len(
                        historical_rates
                    ) > 1
                    else 0.0
                )

            else:

                mean_rate = 0.0

                rate_std = 0.0

            # Direct HBGD prediction

            direct_prediction = hbgd_predict(
                current_d,
                recent_rate,
                delta_days
            )

            if not finite(
                direct_prediction
            ):

                direct_prediction = (
                    current_d
                )

            history_length = (
                target_index + 1
            )

            total_history_weeks = (
                current["week"] -
                obs[0]["week"]
            )

            current_volume = current[
                "volume"
            ]

            if not finite(
                current_volume
            ):

                current_volume = 0.0

            previous_volume = 0.0

            if target_index >= 2:

                previous_volume = obs[
                    target_index - 2
                ][
                    "volume"
                ]

                if not finite(
                    previous_volume
                ):

                    previous_volume = 0.0

            rate_change = (
                recent_rate -
                mean_rate
            )

            geometry = [

                current_d,

                current_volume,

                previous_d,

                current_d -
                previous_d,

                delta_weeks,

                history_length,

                total_history_weeks,

                recent_rate,

                mean_rate,

                rate_std,

                rate_change
            ]

            cases.append({

                "patient_id":
                    patient_id,

                "current_week":
                    current["week"],

                "future_week":
                    target["week"],

                "future_delta_weeks":
                    delta_weeks,

                "current_diameter":
                    current_d,

                "actual":
                    target_d,

                "recent_rate":
                    recent_rate,

                "mean_rate":
                    mean_rate,

                "rate_std":
                    rate_std,

                "direct_hbgd_prediction":
                    direct_prediction,

                "geometry":
                    geometry
            })

    return cases


# ============================================================
# SOURCE MODEL FEATURES
# ============================================================

def source_feature_matrix(
    cases
):

    X = []

    y = []

    for case in cases:

        X.append([

            case[
                "current_diameter"
            ],

            case[
                "previous_diameter"
            ],

            case[
                "growth_rate"
            ],

            case[
                "delta_days"
            ],

            case[
                "hbgd_prediction"
            ]
        ])

        y.append(
            case[
                "actual"
            ]
        )

    return (
        np.asarray(
            X,
            dtype=float
        ),
        np.asarray(
            y,
            dtype=float
        )
    )


# ============================================================
# TRAIN SOURCE MODEL
# ============================================================

def train_source_model():

    cases = build_source_cases()

    print(
        "Source training cases:",
        len(cases)
    )

    if len(cases) < 5:

        raise RuntimeError(
            "Too few source training cases."
        )

    X, y = source_feature_matrix(
        cases
    )

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

    prediction = model.predict(
        X
    )

    mae = mean_absolute_error(
        y,
        prediction
    )

    model_rmse = rmse(
        y,
        prediction
    )

    return (
        model,
        mae,
        model_rmse
    )


# ============================================================
# APPLY SOURCE MODEL TO TARGET
# ============================================================

def source_prediction_for_target(
    model,
    case
):

    X = np.asarray(
        [[

            case[
                "current_diameter"
            ],

            case[
                "current_diameter"
            ],

            case[
                "recent_rate"
            ],

            case[
                "future_delta_weeks"
            ] *
            DAYS_PER_WEEK,

            case[
                "direct_hbgd_prediction"
            ]
        ]],
        dtype=float
    )

    return float(
        model.predict(
            X
        )[0]
    )


# ============================================================
# DIRECT HBGD BASELINE
# ============================================================

def evaluate_direct_hbgd(
    cases
):

    actual = []

    predicted = []

    for case in cases:

        p = case[
            "direct_hbgd_prediction"
        ]

        if not finite(p):

            p = case[
                "current_diameter"
            ]

        actual.append(
            case[
                "actual"
            ]
        )

        predicted.append(
            p
        )

    return (
        mean_absolute_error(
            actual,
            predicted
        ),

        rmse(
            actual,
            predicted
        )
    )


# ============================================================
# GEOMETRY MODEL
# ============================================================

def train_geometry_model(
    train_cases
):

    X = np.asarray(
        [
            x[
                "geometry"
            ]
            for x in train_cases
        ],
        dtype=float
    )

    y = np.asarray(
        [
            x[
                "actual"
            ]
            for x in train_cases
        ],
        dtype=float
    )

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


# ============================================================
# TARGET DOMAIN ADAPTATION
# ============================================================

def train_domain_adaptation(
    source_model,
    train_cases
):

    """
    Learn target-domain correction to the source model.

    The source model gives:
        source_prediction

    Target training data gives:
        actual - source_prediction

    The residual model learns:
        GBM-specific correction

    Importantly, test patients are NOT included.
    """

    X = []

    residual = []

    for case in train_cases:

        source_prediction = (
            source_prediction_for_target(
                source_model,
                case
            )
        )

        actual = case[
            "actual"
        ]

        # Domain adaptation features

        X.append([

            source_prediction,

            case[
                "current_diameter"
            ],

            case[
                "recent_rate"
            ],

            case[
                "mean_rate"
            ],

            case[
                "rate_std"
            ],

            case[
                "future_delta_weeks"
            ],

            case[
                "current_diameter"
            ] -
            source_prediction
        ])

        residual.append(
            actual -
            source_prediction
        )

    X = np.asarray(
        X,
        dtype=float
    )

    residual = np.asarray(
        residual,
        dtype=float
    )

    model = RandomForestRegressor(

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        min_samples_leaf=MIN_SAMPLES_LEAF,

        random_state=RANDOM_SEED + 1,

        n_jobs=-1
    )

    model.fit(
        X,
        residual
    )

    return model


# ============================================================
# ADAPTED PREDICTION
# ============================================================

def adapted_prediction(
    source_model,
    adaptation_model,
    case
):

    source_prediction = (
        source_prediction_for_target(
            source_model,
            case
        )
    )

    X = np.asarray(
        [[

            source_prediction,

            case[
                "current_diameter"
            ],

            case[
                "recent_rate"
            ],

            case[
                "mean_rate"
            ],

            case[
                "rate_std"
            ],

            case[
                "future_delta_weeks"
            ],

            case[
                "current_diameter"
            ] -
            source_prediction

        ]],
        dtype=float
    )

    correction = float(
        adaptation_model.predict(
            X
        )[0]
    )

    prediction = (
        source_prediction +
        correction
    )

    return (
        source_prediction,
        correction,
        prediction
    )


# ============================================================
# COMBINED MODEL
# ============================================================

def train_combined_model(
    train_cases
):

    X = []

    y = []

    for case in train_cases:

        X.append(
            case[
                "geometry"
            ]
        )

        y.append(
            case[
                "actual"
            ]
        )

    return train_geometry_model(
        train_cases
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "STEP 11 - DOMAIN-ADAPTIVE HBGD TRANSFER"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Source model
    # --------------------------------------------------------

    print(
        "\nTraining source HBGD model..."
    )

    (
        source_model,
        source_mae,
        source_rmse
    ) = train_source_model()

    print(
        "Source model MAE:",
        "{:.4f} mm".format(
            source_mae
        )
    )

    print(
        "Source model RMSE:",
        "{:.4f} mm".format(
            source_rmse
        )
    )

    write_csv(

        OUTPUT_SOURCE_MODEL,

        [{

            "source_model":
                "RandomForest_HBGD",

            "training_mae_mm":
                source_mae,

            "training_rmse_mm":
                source_rmse
        }],

        [
            "source_model",
            "training_mae_mm",
            "training_rmse_mm"
        ]
    )

    # --------------------------------------------------------
    # Target patients
    # --------------------------------------------------------

    print(
        "\nLoading LUMIERE..."
    )

    patients = (
        load_target_patients()
    )

    patient_ids = sorted(
        patients.keys()
    )

    print(
        "LUMIERE patients:",
        len(patient_ids)
    )

    # --------------------------------------------------------
    # Patient split
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
    # Target cases
    # --------------------------------------------------------

    train_cases = (
        build_prediction_cases(
            patients,
            train_ids
        )
    )

    validation_cases = (
        build_prediction_cases(
            patients,
            validation_ids
        )
    )

    test_cases = (
        build_prediction_cases(
            patients,
            test_ids
        )
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
    # Direct HBGD
    # --------------------------------------------------------

    (
        direct_val_mae,
        direct_val_rmse
    ) = evaluate_direct_hbgd(
        validation_cases
    )

    (
        direct_test_mae,
        direct_test_rmse
    ) = evaluate_direct_hbgd(
        test_cases
    )

    # --------------------------------------------------------
    # Geometry model
    # --------------------------------------------------------

    geometry_model = (
        train_geometry_model(
            train_cases
        )
    )

    X_val_geometry = np.asarray(
        [
            x["geometry"]
            for x in validation_cases
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

    X_test_geometry = np.asarray(
        [
            x["geometry"]
            for x in test_cases
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

    geometry_val_prediction = (
        geometry_model.predict(
            X_val_geometry
        )
    )

    geometry_test_prediction = (
        geometry_model.predict(
            X_test_geometry
        )
    )

    geometry_val_mae = (
        mean_absolute_error(
            y_val,
            geometry_val_prediction
        )
    )

    geometry_val_rmse = rmse(
        y_val,
        geometry_val_prediction
    )

    geometry_test_mae = (
        mean_absolute_error(
            y_test,
            geometry_test_prediction
        )
    )

    geometry_test_rmse = rmse(
        y_test,
        geometry_test_prediction
    )

    # --------------------------------------------------------
    # Domain adaptation
    # --------------------------------------------------------

    print(
        "\nTraining GBM domain adaptation..."
    )

    adaptation_model = (
        train_domain_adaptation(
            source_model,
            train_cases
        )
    )

    # --------------------------------------------------------
    # Adaptation performance
    # --------------------------------------------------------

    def get_adapted_predictions(
        cases
    ):

        source_predictions = []

        corrections = []

        adapted_predictions = []

        actual = []

        for case in cases:

            (
                source_prediction,
                correction,
                adapted_prediction_value
            ) = adapted_prediction(
                source_model,
                adaptation_model,
                case
            )

            source_predictions.append(
                source_prediction
            )

            corrections.append(
                correction
            )

            adapted_predictions.append(
                adapted_prediction_value
            )

            actual.append(
                case[
                    "actual"
                ]
            )

        return (
            np.asarray(
                actual,
                dtype=float
            ),

            np.asarray(
                source_predictions,
                dtype=float
            ),

            np.asarray(
                corrections,
                dtype=float
            ),

            np.asarray(
                adapted_predictions,
                dtype=float
            )
        )

    (
        y_val_adapt,
        source_val_prediction,
        correction_val,
        adapted_val_prediction
    ) = get_adapted_predictions(
        validation_cases
    )

    (
        y_test_adapt,
        source_test_prediction,
        correction_test,
        adapted_test_prediction
    ) = get_adapted_predictions(
        test_cases
    )

    source_val_mae = (
        mean_absolute_error(
            y_val_adapt,
            source_val_prediction
        )
    )

    source_val_rmse = rmse(
        y_val_adapt,
        source_val_prediction
    )

    source_test_mae = (
        mean_absolute_error(
            y_test_adapt,
            source_test_prediction
        )
    )

    source_test_rmse = rmse(
        y_test_adapt,
        source_test_prediction
    )

    adapted_val_mae = (
        mean_absolute_error(
            y_val_adapt,
            adapted_val_prediction
        )
    )

    adapted_val_rmse = rmse(
        y_val_adapt,
        adapted_val_prediction
    )

    adapted_test_mae = (
        mean_absolute_error(
            y_test_adapt,
            adapted_test_prediction
        )
    )

    adapted_test_rmse = rmse(
        y_test_adapt,
        adapted_test_prediction
    )

    # --------------------------------------------------------
    # Geometry + adapted HBGD
    # --------------------------------------------------------

    # Train a target model using geometry plus
    # the source-model prediction and correction.

    def combined_features(
        cases
    ):

        output = []

        for case in cases:

            (
                source_prediction,
                correction,
                adapted_prediction_value
            ) = adapted_prediction(
                source_model,
                adaptation_model,
                case
            )

            output.append(
                case[
                    "geometry"
                ]
                +
                [
                    source_prediction,
                    correction,
                    adapted_prediction_value
                ]
            )

        return np.asarray(
            output,
            dtype=float
        )

    X_combined_train = (
        combined_features(
            train_cases
        )
    )

    X_combined_val = (
        combined_features(
            validation_cases
        )
    )

    X_combined_test = (
        combined_features(
            test_cases
        )
    )

    y_train_combined = np.asarray(
        [
            x[
                "actual"
            ]
            for x in train_cases
        ],
        dtype=float
    )

    combined_model = (
        RandomForestRegressor(

            n_estimators=N_ESTIMATORS,

            max_depth=MAX_DEPTH,

            min_samples_leaf=MIN_SAMPLES_LEAF,

            random_state=RANDOM_SEED + 2,

            n_jobs=-1
        )
    )

    combined_model.fit(
        X_combined_train,
        y_train_combined
    )

    combined_val_prediction = (
        combined_model.predict(
            X_combined_val
        )
    )

    combined_test_prediction = (
        combined_model.predict(
            X_combined_test
        )
    )

    combined_val_mae = (
        mean_absolute_error(
            y_val,
            combined_val_prediction
        )
    )

    combined_val_rmse = rmse(
        y_val,
        combined_val_prediction
    )

    combined_test_mae = (
        mean_absolute_error(
            y_test,
            combined_test_prediction
        )
    )

    combined_test_rmse = rmse(
        y_test,
        combined_test_prediction
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary_rows = [

        {

            "model":
                "Direct_HBGD",

            "validation_mae_mm":
                direct_val_mae,

            "validation_rmse_mm":
                direct_val_rmse,

            "test_mae_mm":
                direct_test_mae,

            "test_rmse_mm":
                direct_test_rmse
        },

        {

            "model":
                "Source_HBGD_Transfer",

            "validation_mae_mm":
                source_val_mae,

            "validation_rmse_mm":
                source_val_rmse,

            "test_mae_mm":
                source_test_mae,

            "test_rmse_mm":
                source_test_rmse
        },

        {

            "model":
                "GBM_Geometry_Only",

            "validation_mae_mm":
                geometry_val_mae,

            "validation_rmse_mm":
                geometry_val_rmse,

            "test_mae_mm":
                geometry_test_mae,

            "test_rmse_mm":
                geometry_test_rmse
        },

        {

            "model":
                "Domain_Adapted_HBGD",

            "validation_mae_mm":
                adapted_val_mae,

            "validation_rmse_mm":
                adapted_val_rmse,

            "test_mae_mm":
                adapted_test_mae,

            "test_rmse_mm":
                adapted_test_rmse
        },

        {

            "model":
                "Geometry_Plus_Adapted_HBGD",

            "validation_mae_mm":
                combined_val_mae,

            "validation_rmse_mm":
                combined_val_rmse,

            "test_mae_mm":
                combined_test_mae,

            "test_rmse_mm":
                combined_test_rmse
        }
    ]

    write_csv(

        OUTPUT_SUMMARY,

        summary_rows,

        [
            "model",
            "validation_mae_mm",
            "validation_rmse_mm",
            "test_mae_mm",
            "test_rmse_mm"
        ]
    )

    # --------------------------------------------------------
    # Adaptation parameters
    # --------------------------------------------------------

    adaptation_rows = [

        {

            "parameter":
                "mean_validation_correction_mm",

            "value":
                mean(
                    correction_val
                )
        },

        {

            "parameter":
                "std_validation_correction_mm",

            "value":
                float(
                    np.std(
                        correction_val
                    )
                )
        },

        {

            "parameter":
                "mean_test_correction_mm",

            "value":
                mean(
                    correction_test
                )
        },

        {

            "parameter":
                "std_test_correction_mm",

            "value":
                float(
                    np.std(
                        correction_test
                    )
                )
        }
    ]

    write_csv(

        OUTPUT_ADAPTATION,

        adaptation_rows,

        [
            "parameter",
            "value"
        ]
    )

    # --------------------------------------------------------
    # Prediction file
    # --------------------------------------------------------

    prediction_rows = []

    for i, case in enumerate(
        test_cases
    ):

        prediction_rows.append({

            "patient_id":
                case[
                    "patient_id"
                ],

            "current_week":
                case[
                    "current_week"
                ],

            "future_week":
                case[
                    "future_week"
                ],

            "future_delta_weeks":
                case[
                    "future_delta_weeks"
                ],

            "actual_diameter_mm":
                case[
                    "actual"
                ],

            "direct_hbgd_prediction_mm":
                case[
                    "direct_hbgd_prediction"
                ],

            "source_model_prediction_mm":
                source_test_prediction[i],

            "domain_correction_mm":
                correction_test[i],

            "domain_adapted_prediction_mm":
                adapted_test_prediction[i],

            "geometry_prediction_mm":
                geometry_test_prediction[i],

            "geometry_plus_adapted_prediction_mm":
                combined_test_prediction[i],

            "direct_hbgd_error_mm":
                abs(
                    case[
                        "actual"
                    ] -
                    case[
                        "direct_hbgd_prediction"
                    ]
                ),

            "source_transfer_error_mm":
                abs(
                    case[
                        "actual"
                    ] -
                    source_test_prediction[i]
                ),

            "adapted_hbgd_error_mm":
                abs(
                    case[
                        "actual"
                    ] -
                    adapted_test_prediction[i]
                ),

            "geometry_error_mm":
                abs(
                    case[
                        "actual"
                    ] -
                    geometry_test_prediction[i]
                ),

            "combined_error_mm":
                abs(
                    case[
                        "actual"
                    ] -
                    combined_test_prediction[i]
                )
        })

    write_csv(

        OUTPUT_PREDICTIONS,

        prediction_rows,

        [
            "patient_id",
            "current_week",
            "future_week",
            "future_delta_weeks",
            "actual_diameter_mm",
            "direct_hbgd_prediction_mm",
            "source_model_prediction_mm",
            "domain_correction_mm",
            "domain_adapted_prediction_mm",
            "geometry_prediction_mm",
            "geometry_plus_adapted_prediction_mm",
            "direct_hbgd_error_mm",
            "source_transfer_error_mm",
            "adapted_hbgd_error_mm",
            "geometry_error_mm",
            "combined_error_mm"
        ]
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = []

    report.append(
        "# STEP 11 - DOMAIN-ADAPTIVE HBGD TRANSFER"
    )

    report.append("")

    report.append(
        "## Objective"
    )

    report.append("")

    report.append(
        "Step 11 tests whether a growth-prediction model learned from the source cancer can be adapted to the LUMIERE glioblastoma domain without assuming identical growth-rate distributions."
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
        "The target test patients were not used to fit the GBM-specific domain adaptation model."
    )

    report.append("")

    report.append(
        "## Results"
    )

    report.append("")

    report.append(
        "| Model | Validation MAE (mm) | Test MAE (mm) | Test RMSE (mm) |"
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
        "## Domain Adaptation"
    )

    report.append("")

    report.append(
        "The adaptation model learns a GBM-specific residual correction to the source-cancer prediction using only the GBM training patients."
    )

    report.append("")

    report.append(
        "Mean validation correction: {:.4f} mm".format(
            mean(
                correction_val
            )
        )
    )

    report.append("")

    report.append(
        "Mean test correction: {:.4f} mm".format(
            mean(
                correction_test
            )
        )
    )

    report.append("")

    # --------------------------------------------------------
    # Determine improvement
    # --------------------------------------------------------

    improvement = (
        geometry_test_mae -
        combined_test_mae
    )

    if geometry_test_mae != 0:

        improvement_percent = (
            improvement /
            geometry_test_mae
        ) * 100.0

    else:

        improvement_percent = 0.0

    report.append(
        "## Comparison Against Geometry Baseline"
    )

    report.append("")

    report.append(
        "Geometry-only test MAE: {:.4f} mm".format(
            geometry_test_mae
        )
    )

    report.append("")

    report.append(
        "Geometry + adapted HBGD test MAE: {:.4f} mm".format(
            combined_test_mae
        )
    )

    report.append("")

    report.append(
        "Difference: {:.4f} mm".format(
            improvement
        )
    )

    report.append("")

    report.append(
        "Relative change: {:.2f}%".format(
            improvement_percent
        )
    )

    report.append("")

    if combined_test_mae < geometry_test_mae:

        report.append(
            "The geometry + adapted HBGD model achieved lower held-out test MAE than geometry-only. This provides preliminary evidence that cancer-adapted HBGD information may contain predictive information beyond the geometry baseline."
        )

    else:

        report.append(
            "The geometry + adapted HBGD model did not outperform the geometry-only baseline on the held-out test cohort. Therefore, this experiment does not establish predictive benefit from the current domain-adaptive HBGD representation."
        )

    report.append("")

    report.append(
        "## Scientific Interpretation"
    )

    report.append("")

    report.append(
        "This experiment is intentionally more conservative than direct cross-cancer rate transfer. The source model provides a transferable growth prior, while the GBM training cohort learns a target-specific correction."
    )

    report.append("")

    report.append(
        "A successful result would support the hypothesis that source-cancer growth dynamics can be adapted to another cancer domain. A negative result would indicate that the current HBGD representation or adaptation strategy is insufficient."
    )

    report.append("")

    report.append(
        "## Limitations"
    )

    report.append("")

    report.append(
        "The source cohort is substantially smaller than an ideal multi-cancer training cohort, and the LUMIERE cohort contains only 67 patients with longitudinal phenotype trajectories."
    )

    report.append("")

    report.append(
        "The current experiment uses equivalent tumor diameter derived from segmentation volume for LUMIERE rather than an identical measurement process across cancer datasets."
    )

    report.append("")

    report.append(
        "The seven-days-per-week conversion of LUMIERE week identifiers remains a dataset-temporal assumption and should be verified against the official documentation."
    )

    report.append("")

    report.append(
        "The held-out test cohort contains 13 patients, so performance estimates remain preliminary."
    )

    report.append("")

    report.append(
        "## Final Conclusion"
    )

    report.append("")

    report.append(
        "Step 11 evaluates domain adaptation as a mechanism for cross-cancer HBGD transfer. The primary endpoint is held-out GBM prediction performance relative to a geometry-only baseline."
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
    # Console
    # --------------------------------------------------------

    print(
        "\n"
    )

    print(
        "=" * 70
    )

    print(
        "STEP 11 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nRESULTS"
    )

    print(
        "-" * 70
    )

    for row in summary_rows:

        print(
            "{:<35} Validation MAE={:.4f}  Test MAE={:.4f}  Test RMSE={:.4f}".format(
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
        "\nGeometry-only test MAE:",
        "{:.4f} mm".format(
            geometry_test_mae
        )
    )

    print(
        "Geometry + adapted HBGD test MAE:",
        "{:.4f} mm".format(
            combined_test_mae
        )
    )

    print(
        "Relative change:",
        "{:.2f}%".format(
            improvement_percent
        )
    )

    print(
        "\nMean target-domain correction:",
        "{:.4f} mm".format(
            mean(
                correction_test
            )
        )
    )

    print(
        "\nOUTPUTS:"
    )

    print(
        OUTPUT_SUMMARY
    )

    print(
        OUTPUT_PREDICTIONS
    )

    print(
        OUTPUT_SOURCE_MODEL
    )

    print(
        OUTPUT_ADAPTATION
    )

    print(
        OUTPUT_REPORT
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()