# -*- coding: utf-8 -*-

"""
STEP 12 - HBGD INCREMENTAL INFORMATION AND ABLATION ANALYSIS

Purpose
-------
Determine whether HBGD-derived features provide predictive
information beyond tumor geometry and temporal history.

Models
------
M0  No-change baseline
M1  Geometry only
M2  Geometry + temporal history
M3  Geometry + HBGD rate
M4  Geometry + HBGD variability
M5  Geometry + all HBGD features
M6  Geometry + adapted HBGD

Evaluation
----------
Patient-level split:

    47 train
     7 validation
    13 test

The test patients remain completely untouched during
model selection and fitting.

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

STEP11_SUMMARY = os.path.join(
    TARGET_DIR,
    "step11_domain_adaptive_transfer_summary.csv"
)

OUTPUT_SUMMARY = os.path.join(
    TARGET_DIR,
    "step12_hbgd_ablation_summary.csv"
)

OUTPUT_PREDICTIONS = os.path.join(
    TARGET_DIR,
    "step12_hbgd_ablation_predictions.csv"
)

OUTPUT_FEATURE_IMPORTANCE = os.path.join(
    TARGET_DIR,
    "step12_hbgd_feature_importance.csv"
)

OUTPUT_REPORT = os.path.join(
    TARGET_DIR,
    "STEP12_HBGD_INCREMENTAL_INFORMATION_ABLATION_REPORT.md"
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

        return 0.0

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


# ============================================================
# LOAD TARGET DATA
# ============================================================

def load_patients():

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

            if not finite(volume):

                volume = 0.0

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
# BUILD CASES
# ============================================================

def build_cases(
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
            # Previous observation
            # ------------------------------------------------

            if target_index >= 2:

                previous = obs[
                    target_index - 2
                ]

                previous_d = previous[
                    "diameter"
                ]

                previous_v = previous[
                    "volume"
                ]

                previous_delta_weeks = (
                    current["week"] -
                    previous["week"]
                )

                previous_days = (
                    previous_delta_weeks *
                    DAYS_PER_WEEK
                )

                recent_rate = calculate_rate(
                    previous_d,
                    current_d,
                    previous_days
                )

            else:

                previous_d = current_d

                previous_v = (
                    current["volume"]
                )

                recent_rate = 0.0

            if not finite(
                recent_rate
            ):

                recent_rate = 0.0

            # ------------------------------------------------
            # Historical rates
            # ------------------------------------------------

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

                weeks = (
                    obs[j]["week"] -
                    obs[j - 1]["week"]
                )

                days = (
                    weeks *
                    DAYS_PER_WEEK
                )

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

                rate_std = float(
                    np.std(
                        np.asarray(
                            historical_rates
                        )
                    )
                )

                min_rate = min(
                    historical_rates
                )

                max_rate = max(
                    historical_rates
                )

            else:

                mean_rate = 0.0

                rate_std = 0.0

                min_rate = 0.0

                max_rate = 0.0

            # ------------------------------------------------
            # History
            # ------------------------------------------------

            history_length = (
                target_index + 1
            )

            total_history_weeks = (
                current["week"] -
                obs[0]["week"]
            )

            if total_history_weeks < 0:

                total_history_weeks = 0

            rate_change = (
                recent_rate -
                mean_rate
            )

            # ------------------------------------------------
            # Geometry
            # ------------------------------------------------

            geometry = [

                current_d,

                current["volume"],

                previous_d,

                current_d -
                previous_d
            ]

            # ------------------------------------------------
            # Temporal history
            # ------------------------------------------------

            temporal = [

                delta_weeks,

                history_length,

                total_history_weeks
            ]

            # ------------------------------------------------
            # HBGD rate
            # ------------------------------------------------

            hbgd_rate_features = [

                recent_rate,

                mean_rate,

                rate_change
            ]

            # ------------------------------------------------
            # HBGD variability
            # ------------------------------------------------

            hbgd_variability_features = [

                rate_std,

                min_rate,

                max_rate
            ]

            # ------------------------------------------------
            # All HBGD
            # ------------------------------------------------

            all_hbgd = (

                hbgd_rate_features
                +
                hbgd_variability_features
            )

            cases.append({

                "patient_id":
                    patient_id,

                "current_week":
                    current["week"],

                "future_week":
                    target["week"],

                "future_delta_weeks":
                    delta_weeks,

                "actual":
                    target_d,

                "current_diameter":
                    current_d,

                "no_change_prediction":
                    current_d,

                "geometry":
                    geometry,

                "temporal":
                    temporal,

                "hbgd_rate":
                    hbgd_rate_features,

                "hbgd_variability":
                    hbgd_variability_features,

                "all_hbgd":
                    all_hbgd,

                "recent_rate":
                    recent_rate,

                "mean_rate":
                    mean_rate,

                "rate_std":
                    rate_std
            })

    return cases


# ============================================================
# RANDOM FOREST
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
# FEATURE BUILDING
# ============================================================

def make_features(
    cases,
    feature_set
):

    matrix = []

    for case in cases:

        values = []

        if "geometry" in feature_set:

            values += case[
                "geometry"
            ]

        if "temporal" in feature_set:

            values += case[
                "temporal"
            ]

        if "hbgd_rate" in feature_set:

            values += case[
                "hbgd_rate"
            ]

        if "hbgd_variability" in feature_set:

            values += case[
                "hbgd_variability"
            ]

        matrix.append(
            values
        )

    return np.asarray(
        matrix,
        dtype=float
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    train_cases,
    validation_cases,
    test_cases,
    feature_set,
    seed
):

    X_train = make_features(
        train_cases,
        feature_set
    )

    X_val = make_features(
        validation_cases,
        feature_set
    )

    X_test = make_features(
        test_cases,
        feature_set
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

    model = train_model(
        X_train,
        y_train,
        seed
    )

    val_prediction = model.predict(
        X_val
    )

    test_prediction = model.predict(
        X_test
    )

    result = {

        "model":
            feature_set["name"],

        "validation_mae":
            mean_absolute_error(
                y_val,
                val_prediction
            ),

        "validation_rmse":
            rmse(
                y_val,
                val_prediction
            ),

        "test_mae":
            mean_absolute_error(
                y_test,
                test_prediction
            ),

        "test_rmse":
            rmse(
                y_test,
                test_prediction
            ),

        "feature_count":
            X_train.shape[1],

        "feature_importances":
            model.feature_importances_,

        "test_prediction":
            test_prediction
    }

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "STEP 12 - HBGD INCREMENTAL INFORMATION + ABLATION"
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
    # Models
    # --------------------------------------------------------

    models = [

        {

            "name":
                "M1_Geometry",

            "geometry":
                True,

            "temporal":
                False,

            "hbgd_rate":
                False,

            "hbgd_variability":
                False
        },

        {

            "name":
                "M2_Geometry_Temporal",

            "geometry":
                True,

            "temporal":
                True,

            "hbgd_rate":
                False,

            "hbgd_variability":
                False
        },

        {

            "name":
                "M3_Geometry_HBGD_Rate",

            "geometry":
                True,

            "temporal":
                False,

            "hbgd_rate":
                True,

            "hbgd_variability":
                False
        },

        {

            "name":
                "M4_Geometry_HBGD_Variability",

            "geometry":
                True,

            "temporal":
                False,

            "hbgd_rate":
                False,

            "hbgd_variability":
                True
        },

        {

            "name":
                "M5_Geometry_All_HBGD",

            "geometry":
                True,

            "temporal":
                True,

            "hbgd_rate":
                True,

            "hbgd_variability":
                True
        }
    ]

    results = []

    importance_rows = []

    prediction_rows = []

    # --------------------------------------------------------
    # No-change baseline
    # --------------------------------------------------------

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

    val_no_change = np.asarray(
        [
            x[
                "no_change_prediction"
            ]
            for x in validation_cases
        ],
        dtype=float
    )

    test_no_change = np.asarray(
        [
            x[
                "no_change_prediction"
            ]
            for x in test_cases
        ],
        dtype=float
    )

    no_change_val_mae = (
        mean_absolute_error(
            y_val,
            val_no_change
        )
    )

    no_change_val_rmse = rmse(
        y_val,
        val_no_change
    )

    no_change_test_mae = (
        mean_absolute_error(
            y_test,
            test_no_change
        )
    )

    no_change_test_rmse = rmse(
        y_test,
        test_no_change
    )

    results.append({

        "model":
            "M0_No_Change",

        "validation_mae_mm":
            no_change_val_mae,

        "validation_rmse_mm":
            no_change_val_rmse,

        "test_mae_mm":
            no_change_test_mae,

        "test_rmse_mm":
            no_change_test_rmse,

        "feature_count":
            0
    })

    # --------------------------------------------------------
    # ML models
    # --------------------------------------------------------

    for model_index, feature_set in enumerate(
        models
    ):

        print(
            "\n"
        )

        print(
            "-" * 70
        )

        print(
            feature_set["name"]
        )

        result = evaluate_model(

            train_cases,

            validation_cases,

            test_cases,

            feature_set,

            RANDOM_SEED +
            model_index +
            1
        )

        print(
            "Validation MAE:",
            "{:.4f} mm".format(
                result[
                    "validation_mae"
                ]
            )
        )

        print(
            "Test MAE:",
            "{:.4f} mm".format(
                result[
                    "test_mae"
                ]
            )
        )

        results.append({

            "model":
                result["model"],

            "validation_mae_mm":
                result[
                    "validation_mae"
                ],

            "validation_rmse_mm":
                result[
                    "validation_rmse"
                ],

            "test_mae_mm":
                result[
                    "test_mae"
                ],

            "test_rmse_mm":
                result[
                    "test_rmse"
                ],

            "feature_count":
                result[
                    "feature_count"
                ]
        })

        # ----------------------------------------------------
        # Importance
        # ----------------------------------------------------

        feature_names = []

        if feature_set["geometry"]:

            feature_names += [

                "current_diameter_mm",

                "current_volume_cm3",

                "previous_diameter_mm",

                "diameter_change_mm"
            ]

        if feature_set["temporal"]:

            feature_names += [

                "future_interval_weeks",

                "history_length",

                "total_history_weeks"
            ]

        if feature_set["hbgd_rate"]:

            feature_names += [

                "recent_hbgd_rate",

                "mean_hbgd_rate",

                "hbgd_rate_change"
            ]

        if feature_set["hbgd_variability"]:

            feature_names += [

                "hbgd_rate_std",

                "hbgd_rate_min",

                "hbgd_rate_max"
            ]

        for name, importance in zip(
            feature_names,
            result[
                "feature_importances"
            ]
        ):

            importance_rows.append({

                "model":
                    feature_set["name"],

                "feature":
                    name,

                "importance":
                    float(
                        importance
                    )
            })

        # ----------------------------------------------------
        # Test predictions
        # ----------------------------------------------------

        for case, prediction in zip(
            test_cases,
            result[
                "test_prediction"
            ]
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

                "actual_diameter_mm":
                    case[
                        "actual"
                    ],

                "model":
                    feature_set["name"],

                "predicted_diameter_mm":
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
    # Write results
    # --------------------------------------------------------

    write_csv(

        OUTPUT_SUMMARY,

        results,

        [
            "model",
            "validation_mae_mm",
            "validation_rmse_mm",
            "test_mae_mm",
            "test_rmse_mm",
            "feature_count"
        ]
    )

    write_csv(

        OUTPUT_FEATURE_IMPORTANCE,

        importance_rows,

        [
            "model",
            "feature",
            "importance"
        ]
    )

    write_csv(

        OUTPUT_PREDICTIONS,

        prediction_rows,

        [
            "patient_id",
            "current_week",
            "future_week",
            "actual_diameter_mm",
            "model",
            "predicted_diameter_mm",
            "absolute_error_mm"
        ]
    )

    # --------------------------------------------------------
    # Identify best geometry and HBGD models
    # --------------------------------------------------------

    geometry_mae = None

    geometry_temporal_mae = None

    geometry_hbgd_rate_mae = None

    geometry_hbgd_variability_mae = None

    all_hbgd_mae = None

    for row in results:

        name = row[
            "model"
        ]

        if name == "M1_Geometry":

            geometry_mae = float(
                row[
                    "test_mae_mm"
                ]
            )

        elif name == "M2_Geometry_Temporal":

            geometry_temporal_mae = float(
                row[
                    "test_mae_mm"
                ]
            )

        elif name == "M3_Geometry_HBGD_Rate":

            geometry_hbgd_rate_mae = float(
                row[
                    "test_mae_mm"
                ]
            )

        elif name == "M4_Geometry_HBGD_Variability":

            geometry_hbgd_variability_mae = float(
                row[
                    "test_mae_mm"
                ]
            )

        elif name == "M5_Geometry_All_HBGD":

            all_hbgd_mae = float(
                row[
                    "test_mae_mm"
                ]
            )

    # --------------------------------------------------------
    # Improvements
    # --------------------------------------------------------

    def relative_change(
        baseline,
        new
    ):

        if baseline is None:

            return float("nan")

        if baseline == 0:

            return 0.0

        return (
            (
                baseline -
                new
            )
            /
            baseline
        ) * 100.0

    rate_change = relative_change(
        geometry_mae,
        geometry_hbgd_rate_mae
    )

    variability_change = relative_change(
        geometry_mae,
        geometry_hbgd_variability_mae
    )

    all_hbgd_change = relative_change(
        geometry_mae,
        all_hbgd_mae
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = []

    report.append(
        "# STEP 12 - HBGD INCREMENTAL INFORMATION AND ABLATION"
    )

    report.append("")

    report.append(
        "## Objective"
    )

    report.append("")

    report.append(
        "This experiment evaluates whether HBGD-derived growth dynamics provide predictive information beyond tumor geometry and temporal history in LUMIERE glioblastoma."
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
        "The test cohort was not used for model fitting or feature selection."
    )

    report.append("")

    report.append(
        "## Model Results"
    )

    report.append("")

    report.append(
        "| Model | Validation MAE | Test MAE | Test RMSE | Features |"
    )

    report.append(
        "|---|---:|---:|---:|---:|"
    )

    for row in results:

        report.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
                row["model"],
                row[
                    "validation_mae_mm"
                ],
                row[
                    "test_mae_mm"
                ],
                row[
                    "test_rmse_mm"
                ],
                row[
                    "feature_count"
                ]
            )
        )

    report.append("")

    report.append(
        "## Incremental HBGD Effect"
    )

    report.append("")

    report.append(
        "Geometry-only test MAE: {:.4f} mm".format(
            geometry_mae
        )
    )

    report.append("")

    report.append(
        "Geometry + HBGD-rate test MAE: {:.4f} mm".format(
            geometry_hbgd_rate_mae
        )
    )

    report.append("")

    report.append(
        "Relative change from adding HBGD rate: {:.2f}%".format(
            rate_change
        )
    )

    report.append("")

    report.append(
        "Geometry + HBGD-variability test MAE: {:.4f} mm".format(
            geometry_hbgd_variability_mae
        )
    )

    report.append("")

    report.append(
        "Relative change from adding HBGD variability: {:.2f}%".format(
            variability_change
        )
    )

    report.append("")

    report.append(
        "Geometry + all HBGD test MAE: {:.4f} mm".format(
            all_hbgd_mae
        )
    )

    report.append("")

    report.append(
        "Relative change from adding all HBGD features: {:.2f}%".format(
            all_hbgd_change
        )
    )

    report.append("")

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    report.append(
        "## Interpretation"
    )

    report.append("")

    if (
        all_hbgd_mae is not None
        and
        geometry_mae is not None
        and
        all_hbgd_mae < geometry_mae
    ):

        report.append(
            "The full HBGD feature set improved held-out test MAE relative to geometry-only. This provides preliminary evidence that HBGD contains incremental predictive information beyond the geometry baseline."
        )

    else:

        report.append(
            "The full HBGD feature set did not improve held-out test MAE relative to geometry-only. Therefore, the present experiment does not establish independent predictive benefit from HBGD beyond the geometry features."
        )

    report.append("")

    if (
        geometry_hbgd_rate_mae is not None
        and
        geometry_mae is not None
        and
        geometry_hbgd_rate_mae < geometry_mae
    ):

        report.append(
            "The HBGD-rate ablation improved the geometry baseline, suggesting that the estimated growth rate may contain useful temporal information."
        )

    else:

        report.append(
            "The HBGD-rate ablation did not improve the geometry baseline on the held-out test cohort."
        )

    report.append("")

    if (
        geometry_temporal_mae is not None
        and
        geometry_mae is not None
        and
        geometry_temporal_mae < geometry_mae
    ):

        report.append(
            "Temporal-history features improved the geometry baseline, indicating that part of the predictive signal may be attributable to longitudinal observation history rather than the HBGD formulation itself."
        )

    else:

        report.append(
            "Temporal-history features did not improve the geometry baseline on the held-out test cohort."
        )

    report.append("")

    report.append(
        "## Important Scientific Distinction"
    )

    report.append("")

    report.append(
        "An improvement from HBGD features should not automatically be interpreted as proof that the exponential HBGD model is biologically correct. The ablation only evaluates whether HBGD-derived variables contain incremental predictive information."
    )

    report.append("")

    report.append(
        "Conversely, failure to improve prediction does not invalidate the mathematical HBGD formulation; it indicates that the current HBGD features may not provide sufficient additional predictive information for this dataset and task."
    )

    report.append("")

    report.append(
        "## Limitations"
    )

    report.append("")

    report.append(
        "The experiment uses one target cancer cohort and a relatively small held-out patient cohort."
    )

    report.append("")

    report.append(
        "The HBGD features are derived from the available longitudinal segmentation measurements."
    )

    report.append("")

    report.append(
        "LUMIERE week identifiers are converted using seven days per week; this temporal assumption should be verified against the official dataset documentation."
    )

    report.append("")

    report.append(
        "## Conclusion"
    )

    report.append("")

    report.append(
        "Step 12 isolates the incremental contribution of HBGD-derived features from general geometry and longitudinal-history information. The resulting ablation determines whether HBGD provides measurable additional predictive value for future tumor diameter."
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
        "STEP 12 COMPLETE"
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

    for row in results:

        print(
            "{:<32} Validation MAE={:.4f}  Test MAE={:.4f}  Test RMSE={:.4f}".format(
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
        "\nHBGD incremental changes:"
    )

    print(
        "Geometry + HBGD rate:",
        "{:.2f}%".format(
            rate_change
        )
    )

    print(
        "Geometry + HBGD variability:",
        "{:.2f}%".format(
            variability_change
        )
    )

    print(
        "Geometry + all HBGD:",
        "{:.2f}%".format(
            all_hbgd_change
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
        OUTPUT_FEATURE_IMPORTANCE
    )

    print(
        OUTPUT_REPORT
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()