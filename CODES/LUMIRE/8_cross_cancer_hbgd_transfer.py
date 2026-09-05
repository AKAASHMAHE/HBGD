

"""
8_cross_cancer_hbgd_transfer.py

CROSS-CANCER HBGD TRANSFER EXPERIMENT

Source cancer:
    Existing HBGD pipeline

Target cancer:
    LUMIERE glioblastoma cohort

Objective:
    Test whether source-cancer HBGD growth-pattern information
    can be transferred to LUMIERE and provide predictive value.

IMPORTANT:
    Source growth rates are NOT directly copied to GBM.

    Instead:
        1. Learn source-cancer growth-pattern normalization.
        2. Transform LUMIERE HBGD features into the source
           representation.
        3. Compare GBM geometry-only against
           geometry + transferred HBGD representation.

All outputs:
    D:\\VIT\\PROJECT\\NLP\\PATENT\\DATA\\LUMIRE\
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

TARGET_HBGD = os.path.join(
    TARGET_DIR,
    "lumiere_hbgd_growth_features.csv"
)


RESULTS_CSV = os.path.join(
    TARGET_DIR,
    "lumiere_cross_cancer_transfer_results.csv"
)

SUMMARY_CSV = os.path.join(
    TARGET_DIR,
    "lumiere_cross_cancer_transfer_summary.csv"
)

SOURCE_PROFILE_CSV = os.path.join(
    TARGET_DIR,
    "source_cancer_hbgd_transfer_profile.csv"
)

REPORT_PATH = os.path.join(
    TARGET_DIR,
    "LUMIERE_CROSS_CANCER_HBGD_TRANSFER_REPORT.md"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

DAYS_PER_WEEK = 7.0

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


def rmse(actual, predicted):

    return math.sqrt(
        mean([
            (
                float(a) -
                float(p)
            ) ** 2

            for a, p in zip(
                actual,
                predicted
            )
        ])
    )


# ============================================================
# CSV LOADER
# ============================================================

def read_csv(path):

    if not os.path.exists(path):

        raise IOError(
            "Missing file:\n{}".format(
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


# ============================================================
# SOURCE HBGD PROFILE
# ============================================================

def build_source_profile():

    print(
        "\nLoading source HBGD files..."
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
    # Growth-rate distribution
    # --------------------------------------------------------

    growth_rates = []

    for row in growth_rows:

        r = safe_float(
            row[
                "growth_rate_r_per_day"
            ]
        )

        if finite(r):

            growth_rates.append(
                r
            )

    # --------------------------------------------------------
    # Volume growth
    # --------------------------------------------------------

    volume_rates = []

    for row in volume_rows:

        r = safe_float(
            row[
                "volume_growth_rate_per_day"
            ]
        )

        if finite(r):

            volume_rates.append(
                r
            )

    # --------------------------------------------------------
    # State distribution
    # --------------------------------------------------------

    state_counts = {}

    state_rates = []

    for row in state_rows:

        state = row[
            "dynamic_state"
        ].strip().lower()

        state_counts[
            state
        ] = (
            state_counts.get(
                state,
                0
            ) + 1
        )

        r = safe_float(
            row[
                "growth_rate"
            ]
        )

        if finite(r):

            state_rates.append(
                r
            )

    # --------------------------------------------------------
    # Normalized geometry
    # --------------------------------------------------------

    normalized_diameter = []

    normalized_volume = []

    for row in math_rows:

        d = safe_float(
            row[
                "diameter_normalized"
            ]
        )

        v = safe_float(
            row[
                "volume_normalized"
            ]
        )

        if finite(d):

            normalized_diameter.append(
                d
            )

        if finite(v):

            normalized_volume.append(
                v
            )

    # --------------------------------------------------------
    # Trajectory error
    # --------------------------------------------------------

    trajectory_errors = []

    for row in trajectory_rows:

        e = safe_float(
            row[
                "diameter_error_mm"
            ]
        )

        if finite(e):

            trajectory_errors.append(
                abs(e)
            )

    profile = {

        "growth_rate_mean":
            mean(
                growth_rates
            ),

        "growth_rate_std":
            std(
                growth_rates
            ),

        "growth_rate_median":
            median(
                growth_rates
            ),

        "volume_rate_mean":
            mean(
                volume_rates
            ),

        "volume_rate_std":
            std(
                volume_rates
            ),

        "state_rate_mean":
            mean(
                state_rates
            ),

        "state_rate_std":
            std(
                state_rates
            ),

        "normalized_diameter_mean":
            mean(
                normalized_diameter
            ),

        "normalized_diameter_std":
            std(
                normalized_diameter
            ),

        "normalized_volume_mean":
            mean(
                normalized_volume
            ),

        "normalized_volume_std":
            std(
                normalized_volume
            ),

        "trajectory_error_mean":
            mean(
                trajectory_errors
            ),

        "growth_rate_count":
            len(
                growth_rates
            ),

        "volume_rate_count":
            len(
                volume_rates
            ),

        "state_interval_count":
            len(
                state_rows
            )
    }

    for state in [
        "growth",
        "stable",
        "shrinkage"
    ]:

        profile[
            "state_{}_count".format(
                state
            )
        ] = state_counts.get(
            state,
            0
        )

    print(
        "Source growth-rate count:",
        profile[
            "growth_rate_count"
        ]
    )

    print(
        "Source growth-rate mean:",
        profile[
            "growth_rate_mean"
        ]
    )

    print(
        "Source growth-rate std:",
        profile[
            "growth_rate_std"
        ]
    )

    print(
        "Source volume-rate mean:",
        profile[
            "volume_rate_mean"
        ]
    )

    print(
        "Source state intervals:",
        profile[
            "state_interval_count"
        ]
    )

    return profile


# ============================================================
# TARGET TRAJECTORIES
# ============================================================

def load_target_patients():

    rows = read_csv(
        TARGET_TRAJECTORY
    )

    patients = {}

    for row in rows:

        patient_id = row[
            "patient_id"
        ]

        observation = {

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

    for patient_id in patients:

        patients[
            patient_id
        ].sort(
            key=lambda x:
                x["week"]
        )

    return patients


# ============================================================
# PATIENT SPLIT
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

    n = len(
        patient_ids
    )

    n_test = int(
        round(
            n * 0.20
        )
    )

    test = sorted(
        patient_ids[
            :n_test
        ]
    )

    remaining = sorted(
        patient_ids[
            n_test:
        ]
    )

    n_validation = int(
        round(
            len(remaining) *
            0.125
        )
    )

    validation = sorted(
        remaining[
            :n_validation
        ]
    )

    train = sorted(
        remaining[
            n_validation:
        ]
    )

    return (
        train,
        validation,
        test
    )


# ============================================================
# HBGD
# ============================================================

def calculate_rate(
    d1,
    d2,
    delta_weeks
):

    if (
        d1 <= 0
        or
        d2 <= 0
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
            d2 / d1
        )
        /
        delta_days
    )


def state_code(
    r
):

    if not finite(r):

        return 0.0

    if r > 0.0005:

        return 1.0

    if r < -0.0010:

        return -1.0

    return 0.0


# ============================================================
# BUILD TARGET CASES
# ============================================================

def build_target_cases(
    patients,
    patient_ids,
    source_profile
):

    cases = []

    for patient_id in patient_ids:

        observations = patients[
            patient_id
        ]

        if len(
            observations
        ) < 3:

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

            target = observations[
                target_index
            ]

            current_diameter = current[
                "diameter"
            ]

            current_volume = current[
                "volume"
            ]

            target_diameter = target[
                "diameter"
            ]

            delta_weeks = (
                target["week"] -
                current["week"]
            )

            if (
                current_diameter <= 0
                or
                target_diameter <= 0
                or
                delta_weeks <= 0
            ):

                continue

            # ------------------------------------------------
            # Previous interval HBGD
            # ------------------------------------------------

            previous = history[
                -2
            ]

            previous_diameter = previous[
                "diameter"
            ]

            previous_volume = previous[
                "volume"
            ]

            previous_delta_weeks = (
                current["week"] -
                previous["week"]
            )

            recent_r = calculate_rate(
                previous_diameter,
                current_diameter,
                previous_delta_weeks
            )

            # ------------------------------------------------
            # Overall HBGD
            # ------------------------------------------------

            first = history[
                0
            ]

            total_weeks = (
                current["week"] -
                first["week"]
            )

            overall_r = calculate_rate(
                first["diameter"],
                current_diameter,
                total_weeks
            )

            # ------------------------------------------------
            # Multi-point rate
            # ------------------------------------------------

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
                    dw
                )

                if finite(r):

                    rates.append(
                        r
                    )

            if rates:

                multi_r = mean(
                    rates
                )

                rate_std = std(
                    rates
                )

            else:

                multi_r = 0.0

                rate_std = 0.0

            # ------------------------------------------------
            # Transfer normalization
            #
            # Standardize target growth dynamics against
            # source-cancer HBGD distribution.
            # ------------------------------------------------

            source_mean = (
                source_profile[
                    "growth_rate_mean"
                ]
            )

            source_std = (
                source_profile[
                    "growth_rate_std"
                ]
            )

            if (
                not finite(
                    source_std
                )
                or
                source_std <= 0
            ):

                source_std = 1.0

            recent_transfer = (
                (
                    recent_r -
                    source_mean
                )
                /
                source_std
            )

            overall_transfer = (
                (
                    overall_r -
                    source_mean
                )
                /
                source_std
            )

            multi_transfer = (
                (
                    multi_r -
                    source_mean
                )
                /
                source_std
            )

            # ------------------------------------------------
            # State
            # ------------------------------------------------

            current_state = state_code(
                recent_r
            )

            # ------------------------------------------------
            # Geometry features
            # ------------------------------------------------

            diameter_change = (
                current_diameter -
                previous_diameter
            )

            volume_change = (
                current_volume -
                previous_volume
            )

            diameter_pct = (
                diameter_change /
                previous_diameter *
                100.0
            )

            if previous_volume > 0:

                volume_pct = (
                    volume_change /
                    previous_volume *
                    100.0
                )

            else:

                volume_pct = 0.0

            # ------------------------------------------------
            # Geometry-only
            # ------------------------------------------------

            geometry = [

                current_diameter,

                current_volume,

                previous_diameter,

                previous_volume,

                previous_delta_weeks,

                diameter_change,

                volume_change,

                diameter_pct,

                volume_pct,

                len(history),

                total_weeks
            ]

            # ------------------------------------------------
            # Transfer HBGD
            # ------------------------------------------------

            transferred = [

                recent_transfer,

                multi_transfer,

                overall_transfer,

                rate_std,

                current_state
            ]

            cases.append({

                "patient_id":
                    patient_id,

                "target_week":
                    target["week"],

                "future_delta_weeks":
                    delta_weeks,

                "actual":
                    target_diameter,

                "geometry":
                    geometry,

                "transfer":
                    transferred
            })

    return cases


# ============================================================
# MATRIX
# ============================================================

def matrix(
    cases,
    key
):

    return np.asarray([
        case[key]
        for case in cases
    ], dtype=float)


def targets(
    cases
):

    return np.asarray([
        case[
            "actual"
        ]
        for case in cases
    ], dtype=float)


# ============================================================
# TRAIN
# ============================================================

def train_model(
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


# ============================================================
# IMPUTE
# ============================================================

def fit_medians(
    X
):

    result = []

    for i in range(
        X.shape[1]
    ):

        column = X[
            :,
            i
        ]

        valid = column[
            np.isfinite(column)
        ]

        if len(valid):

            result.append(
                float(
                    np.median(valid)
                )
            )

        else:

            result.append(
                0.0
            )

    return np.asarray(
        result
    )


def apply_medians(
    X,
    medians
):

    X = X.copy()

    for i in range(
        X.shape[1]
    ):

        invalid = ~np.isfinite(
            X[:, i]
        )

        X[
            invalid,
            i
        ] = medians[i]

    return X


# ============================================================
# EVALUATION
# ============================================================

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
        "STEP 8 - CROSS-CANCER HBGD TRANSFER"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Source profile
    # --------------------------------------------------------

    source_profile = (
        build_source_profile()
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    print(
        "\nLoading LUMIERE..."
    )

    patients = (
        load_target_patients()
    )

    print(
        "LUMIERE patients:",
        len(patients)
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_patients,
        validation_patients,
        test_patients
    ) = split_patients(
        patients
    )

    print(
        "\nPATIENT SPLIT"
    )

    print(
        "Train:",
        len(train_patients)
    )

    print(
        "Validation:",
        len(validation_patients)
    )

    print(
        "Test:",
        len(test_patients)
    )

    # --------------------------------------------------------
    # Build cases
    # --------------------------------------------------------

    train_cases = build_target_cases(
        patients,
        train_patients,
        source_profile
    )

    validation_cases = build_target_cases(
        patients,
        validation_patients,
        source_profile
    )

    test_cases = build_target_cases(
        patients,
        test_patients,
        source_profile
    )

    print(
        "\nCASES"
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

    Xg_train = matrix(
        train_cases,
        "geometry"
    )

    Xg_val = matrix(
        validation_cases,
        "geometry"
    )

    Xg_test = matrix(
        test_cases,
        "geometry"
    )

    Xt_train = matrix(
        train_cases,
        "transfer"
    )

    Xt_val = matrix(
        validation_cases,
        "transfer"
    )

    Xt_test = matrix(
        test_cases,
        "transfer"
    )

    y_train = targets(
        train_cases
    )

    y_val = targets(
        validation_cases
    )

    y_test = targets(
        test_cases
    )

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
    # Imputation
    # --------------------------------------------------------

    geometry_medians = fit_medians(
        Xg_train
    )

    transfer_medians = fit_medians(
        Xt_train
    )

    combined_medians = fit_medians(
        Xc_train
    )

    Xg_train = apply_medians(
        Xg_train,
        geometry_medians
    )

    Xg_val = apply_medians(
        Xg_val,
        geometry_medians
    )

    Xg_test = apply_medians(
        Xg_test,
        geometry_medians
    )

    Xt_train = apply_medians(
        Xt_train,
        transfer_medians
    )

    Xt_val = apply_medians(
        Xt_val,
        transfer_medians
    )

    Xt_test = apply_medians(
        Xt_test,
        transfer_medians
    )

    Xc_train = apply_medians(
        Xc_train,
        combined_medians
    )

    Xc_val = apply_medians(
        Xc_val,
        combined_medians
    )

    Xc_test = apply_medians(
        Xc_test,
        combined_medians
    )

    # --------------------------------------------------------
    # Models
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

    result_rows = []

    summary_rows = []

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

        model = train_model(
            X_train,
            y_train
        )

        validation_prediction, validation_mae, validation_rmse = evaluate(
            model,
            X_val,
            y_val
        )

        test_prediction, test_mae, test_rmse = evaluate(
            model,
            X_test,
            y_test
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

        for case, prediction in zip(
            test_cases,
            test_prediction
        ):

            result_rows.append({

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
                        case["actual"] -
                        prediction
                    )
            })

        summary_rows.append({

            "model":
                name,

            "train_patients":
                len(train_patients),

            "validation_patients":
                len(validation_patients),

            "test_patients":
                len(test_patients),

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

    # --------------------------------------------------------
    # Save source profile
    # --------------------------------------------------------

    profile_rows = []

    for key in sorted(
        source_profile.keys()
    ):

        profile_rows.append({

            "parameter":
                key,

            "value":
                source_profile[key]
        })

    with open(
        SOURCE_PROFILE_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "parameter",
                "value"
            ]
        )

        writer.writeheader()

        for row in profile_rows:

            writer.writerow(
                row
            )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "patient_id",
                "target_week",
                "future_delta_weeks",
                "actual_future_diameter_mm",
                "model",
                "predicted_future_diameter_mm",
                "absolute_error_mm"
            ]
        )

        writer.writeheader()

        for row in result_rows:

            writer.writerow(
                row
            )

    with open(
        SUMMARY_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
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

        writer.writeheader()

        for row in summary_rows:

            writer.writerow(
                row
            )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = []

    report.append(
        "# LUMIERE CROSS-CANCER HBGD TRANSFER REPORT"
    )

    report.append("")

    report.append(
        "## Objective"
    )

    report.append("")

    report.append(
        "This experiment evaluates whether growth-related information derived from the existing source-cancer HBGD pipeline can be transferred to the LUMIERE glioblastoma cohort."
    )

    report.append("")

    report.append(
        "The source-cancer absolute growth rate was not directly imposed on LUMIERE. Instead, LUMIERE growth-rate features were standardized relative to the source-cancer HBGD growth-rate distribution."
    )

    report.append("")

    report.append(
        "## Patient Split"
    )

    report.append("")

    report.append(
        "| Split | Patients | Cases |"
    )

    report.append(
        "|---|---:|---:|"
    )

    report.append(
        "| Training | {} | {} |".format(
            len(train_patients),
            len(train_cases)
        )
    )

    report.append(
        "| Validation | {} | {} |".format(
            len(validation_patients),
            len(validation_cases)
        )
    )

    report.append(
        "| Test | {} | {} |".format(
            len(test_patients),
            len(test_cases)
        )
    )

    report.append("")

    report.append(
        "## Source HBGD Profile"
    )

    report.append("")

    report.append(
        "| Parameter | Value |"
    )

    report.append(
        "|---|---:|"
    )

    for key in sorted(
        source_profile.keys()
    ):

        report.append(
            "| {} | {} |".format(
                key,
                source_profile[key]
            )
        )

    report.append("")

    report.append(
        "## Model Comparison"
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
                row["validation_mae_mm"],
                row["test_mae_mm"],
                row["test_rmse_mm"]
            )
        )

    report.append("")

    # --------------------------------------------------------
    # Interpretation
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

    report.append(
        "## Interpretation"
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
                "The geometry + transferred-HBGD model improved test MAE over the geometry-only model by {:.2f}%. This provides preliminary evidence that source-cancer HBGD growth-pattern information may contribute transferable information for GBM progression prediction.".format(
                    improvement
                )
            )

        else:

            report.append(
                "The geometry + transferred-HBGD model did not improve test MAE over the geometry-only model. Therefore, this experiment does not establish incremental predictive benefit from the transferred HBGD representation."
            )

    report.append("")

    report.append(
        "## Scientific Caution"
    )

    report.append("")

    report.append(
        "This experiment is a preliminary cross-cancer transfer test. It does not establish clinical validity, biological equivalence of growth rates between cancer types, or generalization to unseen institutions."
    )

    report.append("")

    report.append(
        "The source-cancer HBGD distribution is used as a transfer reference rather than assuming that its absolute growth rates are directly applicable to GBM."
    )

    report.append("")

    report.append(
        "## Next Step"
    )

    report.append("")

    report.append(
        "If transferred HBGD improves performance, the next stage should investigate a learned domain-adaptation or representation-transfer model and evaluate the effect across repeated patient-level splits."
    )

    report.append("")

    report.append(
        "If transferred HBGD does not improve performance, the result should be reported as evidence that direct source-cancer HBGD distribution transfer is insufficient, motivating a more explicit domain-adaptation strategy."
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
        "STEP 8 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nOUTPUT FILES:"
    )

    print(
        SOURCE_PROFILE_CSV
    )

    print(
        SUMMARY_CSV
    )

    print(
        RESULTS_CSV
    )

    print(
        REPORT_PATH
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()