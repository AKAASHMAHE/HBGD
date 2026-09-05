# -*- coding: utf-8 -*-

"""
7_lumiere_hbgd_feature_ml.py

STEP 7:
    Evaluate whether HBGD-derived temporal features improve
    machine-learning prediction of future tumor diameter.

EXPERIMENT:

    A. No-change baseline
    B. Geometry/history only
    C. HBGD features only
    D. Geometry + HBGD features

MODEL:
    Random Forest Regressor

SPLIT:
    Patient-level
        70% training
        10% validation
        20% test

With 67 patients:
    47 train
     7 validation
    13 test

IMPORTANT:
    - Patient-level separation.
    - Test patients are completely untouched during training/model selection.
    - Future target measurement is never included in features.
    - Feature construction uses only observations before target.
    - Python 3.6 compatible.
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

LUMIERE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"

TRAJECTORY_CSV = os.path.join(
    LUMIERE_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

RESULTS_CSV = os.path.join(
    LUMIERE_DIR,
    "lumiere_hbgd_feature_ml_results.csv"
)

SUMMARY_CSV = os.path.join(
    LUMIERE_DIR,
    "lumiere_hbgd_feature_ml_summary.csv"
)

REPORT_PATH = os.path.join(
    LUMIERE_DIR,
    "LUMIERE_HBGD_FEATURE_ML_REPORT.md"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TEST_FRACTION = 0.20

VALIDATION_FRACTION_OF_REMAINING = 0.125

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
        return math.isfinite(float(value))
    except Exception:
        return False


def mean(values):

    if not values:
        return float("nan")

    return sum(values) / float(len(values))


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


def rmse(actual, predicted):

    if not actual:

        return float("nan")

    errors = []

    for a, p in zip(
        actual,
        predicted
    ):

        errors.append(
            (a - p) ** 2
        )

    return math.sqrt(
        mean(errors)
    )


def mape(actual, predicted):

    values = []

    for a, p in zip(
        actual,
        predicted
    ):

        if a != 0:

            values.append(
                abs(
                    (a - p) / a
                ) * 100.0
            )

    return mean(values)


# ============================================================
# LOAD PATIENT TRAJECTORIES
# ============================================================

def load_patients():

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

                "week":
                    int(
                        row[
                            "week"
                        ]
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
            key=lambda x: x["week"]
        )

    return patients


# ============================================================
# PATIENT-LEVEL SPLIT
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
            n *
            TEST_FRACTION
        )
    )

    if n_test < 1:

        n_test = 1

    remaining = n - n_test

    n_validation = int(
        round(
            remaining *
            VALIDATION_FRACTION_OF_REMAINING
        )
    )

    if n_validation < 1:

        n_validation = 1

    test = sorted(
        patient_ids[
            :n_test
        ]
    )

    validation = sorted(
        patient_ids[
            n_test:
            n_test + n_validation
        ]
    )

    train = sorted(
        patient_ids[
            n_test + n_validation:
        ]
    )

    return (
        train,
        validation,
        test
    )


# ============================================================
# HBGD FUNCTIONS
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


def classify_state(
    rate
):

    if not finite(rate):

        return "UNKNOWN"

    if rate > 0.0005:

        return "GROWTH"

    if rate < -0.0010:

        return "SHRINKAGE"

    return "STABLE"


def calculate_multi_rate(
    history
):

    if len(history) < 2:

        return float("nan")

    first_week = history[
        0
    ]["week"]

    x = []

    y = []

    for obs in history:

        diameter = obs[
            "diameter"
        ]

        if diameter <= 0:

            continue

        elapsed_days = (
            obs["week"] -
            first_week
        ) * DAYS_PER_WEEK

        x.append(
            elapsed_days
        )

        y.append(
            math.log(
                diameter
            )
        )

    if len(x) < 2:

        return float("nan")

    x_mean = mean(x)

    y_mean = mean(y)

    numerator = 0.0

    denominator = 0.0

    for xi, yi in zip(
        x,
        y
    ):

        numerator += (
            (xi - x_mean) *
            (yi - y_mean)
        )

        denominator += (
            (xi - x_mean) ** 2
        )

    if denominator <= 0:

        return float("nan")

    return (
        numerator /
        denominator
    )


def calculate_rate_variability(
    history
):

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

        delta_weeks = (
            history[i + 1]["week"] -
            history[i]["week"]
        )

        r = calculate_rate(
            d1,
            d2,
            delta_weeks
        )

        if finite(r):

            rates.append(
                r
            )

    if len(rates) < 2:

        return 0.0

    average = mean(
        rates
    )

    variance = mean([
        (
            r -
            average
        ) ** 2
        for r in rates
    ])

    return math.sqrt(
        variance
    )


# ============================================================
# BUILD FEATURE VECTOR
# ============================================================

def build_features(
    history
):

    """
    Only history BEFORE the target observation
    is passed here.
    """

    current = history[
        -1
    ]

    current_diameter = current[
        "diameter"
    ]

    current_volume = current[
        "volume"
    ]

    # --------------------------------------------------------
    # Basic geometry
    # --------------------------------------------------------

    features = [

        current_diameter,

        current_volume
    ]

    # --------------------------------------------------------
    # Previous observation
    # --------------------------------------------------------

    if len(history) >= 2:

        previous = history[
            -2
        ]

        previous_diameter = previous[
            "diameter"
        ]

        previous_volume = previous[
            "volume"
        ]

        delta_weeks = (
            current["week"] -
            previous["week"]
        )

        recent_rate = calculate_rate(
            previous_diameter,
            current_diameter,
            delta_weeks
        )

        diameter_change = (
            current_diameter -
            previous_diameter
        )

        volume_change = (
            current_volume -
            previous_volume
        )

        diameter_percent_change = (
            diameter_change /
            previous_diameter *
            100.0
            if previous_diameter > 0
            else 0.0
        )

        volume_percent_change = (
            volume_change /
            previous_volume *
            100.0
            if previous_volume > 0
            else 0.0
        )

    else:

        previous_diameter = (
            current_diameter
        )

        previous_volume = (
            current_volume
        )

        delta_weeks = 0

        recent_rate = 0.0

        diameter_change = 0.0

        volume_change = 0.0

        diameter_percent_change = 0.0

        volume_percent_change = 0.0

    # --------------------------------------------------------
    # Multi-point history
    # --------------------------------------------------------

    multi_rate = calculate_multi_rate(
        history
    )

    rate_variability = (
        calculate_rate_variability(
            history
        )
    )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    current_state = classify_state(
        recent_rate
    )

    # Numerical encoding
    if current_state == "GROWTH":

        state_code = 1.0

    elif current_state == "SHRINKAGE":

        state_code = -1.0

    elif current_state == "STABLE":

        state_code = 0.0

    else:

        state_code = 0.0

    # --------------------------------------------------------
    # Longitudinal information
    # --------------------------------------------------------

    number_of_observations = len(
        history
    )

    first_diameter = history[
        0
    ]["diameter"]

    total_elapsed_weeks = (
        current["week"] -
        history[0]["week"]
    )

    overall_rate = calculate_rate(
        first_diameter,
        current_diameter,
        total_elapsed_weeks
        if total_elapsed_weeks > 0
        else 1
    )

    # --------------------------------------------------------
    # Feature vector
    # --------------------------------------------------------

    feature_vector = [

        current_diameter,

        current_volume,

        previous_diameter,

        previous_volume,

        delta_weeks,

        diameter_change,

        volume_change,

        diameter_percent_change,

        volume_percent_change,

        recent_rate,

        multi_rate,

        rate_variability,

        state_code,

        number_of_observations,

        total_elapsed_weeks,

        overall_rate
    ]

    return feature_vector


# ============================================================
# FEATURE NAMES
# ============================================================

FEATURE_NAMES = [

    "current_diameter_mm",

    "current_volume_cm3",

    "previous_diameter_mm",

    "previous_volume_cm3",

    "recent_interval_weeks",

    "diameter_change_mm",

    "volume_change_cm3",

    "diameter_percent_change",

    "volume_percent_change",

    "recent_hbgd_rate_per_day",

    "multi_point_hbgd_rate_per_day",

    "hbgd_rate_variability",

    "hbgd_state_code",

    "number_of_previous_observations",

    "total_history_weeks",

    "overall_hbgd_rate_per_day"
]


# ============================================================
# FEATURE GROUPS
# ============================================================

GEOMETRY_FEATURES = [

    "current_diameter_mm",

    "current_volume_cm3",

    "previous_diameter_mm",

    "previous_volume_cm3",

    "recent_interval_weeks",

    "diameter_change_mm",

    "volume_change_cm3",

    "diameter_percent_change",

    "volume_percent_change",

    "number_of_previous_observations",

    "total_history_weeks"
]

HBGD_FEATURES = [

    "recent_hbgd_rate_per_day",

    "multi_point_hbgd_rate_per_day",

    "hbgd_rate_variability",

    "hbgd_state_code",

    "overall_hbgd_rate_per_day"
]

COMBINED_FEATURES = (
    GEOMETRY_FEATURES +
    HBGD_FEATURES
)


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(
    patients,
    patient_ids
):

    rows = []

    for patient_id in patient_ids:

        observations = patients[
            patient_id
        ]

        # Need at least 3 observations:
        #
        # previous
        # current
        # future target

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

            target = observations[
                target_index
            ]

            current = history[
                -1
            ]

            delta_future_weeks = (
                target["week"] -
                current["week"]
            )

            if delta_future_weeks <= 0:

                continue

            target_diameter = target[
                "diameter"
            ]

            if (
                not finite(
                    target_diameter
                )
                or
                target_diameter <= 0
            ):

                continue

            features = build_features(
                history
            )

            rows.append({

                "patient_id":
                    patient_id,

                "target_week":
                    target["week"],

                "target_diameter":
                    target_diameter,

                "future_delta_weeks":
                    delta_future_weeks,

                "features":
                    features
            })

    return rows


# ============================================================
# IMPUTE
# ============================================================

def fit_imputation(
    X
):

    X = np.asarray(
        X,
        dtype=float
    )

    medians = []

    for i in range(
        X.shape[1]
    ):

        column = X[:, i]

        valid = column[
            np.isfinite(column)
        ]

        if len(valid) == 0:

            medians.append(
                0.0
            )

        else:

            medians.append(
                float(
                    np.median(valid)
                )
            )

    return np.asarray(
        medians,
        dtype=float
    )


def apply_imputation(
    X,
    medians
):

    X = np.asarray(
        X,
        dtype=float
    )

    output = X.copy()

    for i in range(
        output.shape[1]
    ):

        invalid = ~np.isfinite(
            output[:, i]
        )

        output[
            invalid,
            i
        ] = medians[i]

    return output


# ============================================================
# FEATURE SELECTION
# ============================================================

def feature_indices(
    feature_names
):

    return [
        FEATURE_NAMES.index(
            name
        )
        for name in feature_names
    ]


def select_features(
    X,
    names
):

    indices = feature_indices(
        names
    )

    return X[:, indices]


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):

    model = RandomForestRegressor(

        n_estimators=N_ESTIMATORS,

        max_depth=MAX_DEPTH,

        min_samples_leaf=MIN_SAMPLES_LEAF,

        random_state=RANDOM_SEED,

        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y
):

    prediction = model.predict(
        X
    )

    return {

        "predictions":
            prediction,

        "mae":
            mean_absolute_error(
                y,
                prediction
            ),

        "rmse":
            math.sqrt(
                mean_squared_error(
                    y,
                    prediction
                )
            ),

        "mape":
            mape(
                y,
                prediction
            )
    }


# ============================================================
# BASELINE
# ============================================================

def evaluate_no_change(
    rows
):

    actual = []

    prediction = []

    for row in rows:

        current_diameter = row[
            "features"
        ][0]

        actual.append(
            row[
                "target_diameter"
            ]
        )

        prediction.append(
            current_diameter
        )

    return {

        "predictions":
            prediction,

        "mae":
            mean_absolute_error(
                actual,
                prediction
            ),

        "rmse":
            rmse(
                actual,
                prediction
            ),

        "mape":
            mape(
                actual,
                prediction
            )
    }


# ============================================================
# ADD RESULTS
# ============================================================

def add_predictions(
    output_rows,
    dataset_rows,
    predictions,
    model_name
):

    for row, prediction in zip(
        dataset_rows,
        predictions
    ):

        output_rows.append({

            "patient_id":
                row[
                    "patient_id"
                ],

            "target_week":
                row[
                    "target_week"
                ],

            "future_delta_weeks":
                row[
                    "future_delta_weeks"
                ],

            "actual_future_diameter_mm":
                row[
                    "target_diameter"
                ],

            "model":
                model_name,

            "predicted_future_diameter_mm":
                float(
                    prediction
                ),

            "absolute_error_mm":
                abs(
                    row[
                        "target_diameter"
                    ] -
                    prediction
                )
        })


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "STEP 7 - HBGD FEATURE MACHINE LEARNING"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        "\nLoading trajectories..."
    )

    patients = load_patients()

    print(
        "Patients:",
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
        "Training:",
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
    # Dataset
    # --------------------------------------------------------

    print(
        "\nBuilding training dataset..."
    )

    train_rows = build_dataset(
        patients,
        train_patients
    )

    print(
        "Training cases:",
        len(train_rows)
    )

    print(
        "\nBuilding validation dataset..."
    )

    validation_rows = build_dataset(
        patients,
        validation_patients
    )

    print(
        "Validation cases:",
        len(validation_rows)
    )

    print(
        "\nBuilding test dataset..."
    )

    test_rows = build_dataset(
        patients,
        test_patients
    )

    print(
        "Test cases:",
        len(test_rows)
    )

    if not train_rows:

        print(
            "\nERROR: No training cases."
        )

        return

    if not test_rows:

        print(
            "\nERROR: No test cases."
        )

        return

    # --------------------------------------------------------
    # Arrays
    # --------------------------------------------------------

    X_train = np.asarray([
        row[
            "features"
        ]
        for row in train_rows
    ])

    y_train = np.asarray([
        row[
            "target_diameter"
        ]
        for row in train_rows
    ])

    X_validation = np.asarray([
        row[
            "features"
        ]
        for row in validation_rows
    ])

    y_validation = np.asarray([
        row[
            "target_diameter"
        ]
        for row in validation_rows
    ])

    X_test = np.asarray([
        row[
            "features"
        ]
        for row in test_rows
    ])

    y_test = np.asarray([
        row[
            "target_diameter"
        ]
        for row in test_rows
    ])

    # --------------------------------------------------------
    # Imputation based ONLY on training
    # --------------------------------------------------------

    medians = fit_imputation(
        X_train
    )

    X_train = apply_imputation(
        X_train,
        medians
    )

    X_validation = apply_imputation(
        X_validation,
        medians
    )

    X_test = apply_imputation(
        X_test,
        medians
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    no_change = evaluate_no_change(
        test_rows
    )

    print(
        "\nNo-change baseline:"
    )

    print(
        "MAE:",
        "{:.4f} mm".format(
            no_change["mae"]
        )
    )

    print(
        "RMSE:",
        "{:.4f} mm".format(
            no_change["rmse"]
        )
    )

    # --------------------------------------------------------
    # Feature groups
    # --------------------------------------------------------

    experiments = [

        (
            "Geometry_only",
            GEOMETRY_FEATURES
        ),

        (
            "HBGD_only",
            HBGD_FEATURES
        ),

        (
            "Geometry_plus_HBGD",
            COMBINED_FEATURES
        )
    ]

    result_rows = []

    summary_rows = []

    # Save baseline into summary
    summary_rows.append({

        "model":
            "No_change",

        "feature_group":
            "Baseline",

        "n_train":
            len(train_rows),

        "n_validation":
            len(validation_rows),

        "n_test":
            len(test_rows),

        "test_mae_mm":
            no_change["mae"],

        "test_rmse_mm":
            no_change["rmse"],

        "test_mape_percent":
            no_change["mape"],

        "validation_mae_mm":
            "",

        "validation_rmse_mm":
            ""
    })

    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for model_name, feature_names in experiments:

        print(
            "\n"
        )

        print(
            "-" * 70
        )

        print(
            "MODEL:",
            model_name
        )

        print(
            "FEATURES:",
            len(feature_names)
        )

        # ----------------------------------------------
        # Select
        # ----------------------------------------------

        Xtr = select_features(
            X_train,
            feature_names
        )

        Xv = select_features(
            X_validation,
            feature_names
        )

        Xt = select_features(
            X_test,
            feature_names
        )

        # ----------------------------------------------
        # Train
        # ----------------------------------------------

        model = train_model(
            Xtr,
            y_train
        )

        # ----------------------------------------------
        # Validation
        # ----------------------------------------------

        validation_metrics = evaluate_model(
            model,
            Xv,
            y_validation
        )

        print(
            "Validation MAE:",
            "{:.4f} mm".format(
                validation_metrics["mae"]
            )
        )

        print(
            "Validation RMSE:",
            "{:.4f} mm".format(
                validation_metrics["rmse"]
            )
        )

        # ----------------------------------------------
        # Test
        # ----------------------------------------------

        test_metrics = evaluate_model(
            model,
            Xt,
            y_test
        )

        print(
            "Test MAE:",
            "{:.4f} mm".format(
                test_metrics["mae"]
            )
        )

        print(
            "Test RMSE:",
            "{:.4f} mm".format(
                test_metrics["rmse"]
            )
        )

        print(
            "Test MAPE:",
            "{:.2f}%".format(
                test_metrics["mape"]
            )
        )

        # ----------------------------------------------
        # Improvement vs no-change
        # ----------------------------------------------

        if no_change["mae"] > 0:

            improvement = (
                (
                    no_change["mae"] -
                    test_metrics["mae"]
                )
                /
                no_change["mae"]
            ) * 100.0

        else:

            improvement = 0.0

        print(
            "Improvement vs no-change:",
            "{:.2f}%".format(
                improvement
            )
        )

        # ----------------------------------------------
        # Save predictions
        # ----------------------------------------------

        add_predictions(
            result_rows,
            test_rows,
            test_metrics["predictions"],
            model_name
        )

        summary_rows.append({

            "model":
                model_name,

            "feature_group":
                ",".join(
                    feature_names
                ),

            "n_train":
                len(train_rows),

            "n_validation":
                len(validation_rows),

            "n_test":
                len(test_rows),

            "test_mae_mm":
                test_metrics["mae"],

            "test_rmse_mm":
                test_metrics["rmse"],

            "test_mape_percent":
                test_metrics["mape"],

            "validation_mae_mm":
                validation_metrics["mae"],

            "validation_rmse_mm":
                validation_metrics["rmse"]
        })

    # --------------------------------------------------------
    # Feature importance for combined model
    # --------------------------------------------------------

    print(
        "\nCalculating combined-model feature importance..."
    )

    combined_X_train = select_features(
        X_train,
        COMBINED_FEATURES
    )

    combined_model = train_model(
        combined_X_train,
        y_train
    )

    importances = combined_model.feature_importances_

    importance_rows = []

    for name, importance in zip(
        COMBINED_FEATURES,
        importances
    ):

        importance_rows.append({

            "feature":
                name,

            "importance":
                float(importance)
        })

    importance_rows.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    print(
        "\nTOP FEATURES"
    )

    for row in importance_rows[
        :10
    ]:

        print(
            "{:<40} {:.6f}".format(
                row["feature"],
                row["importance"]
            )
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
                "feature_group",
                "n_train",
                "n_validation",
                "n_test",
                "test_mae_mm",
                "test_rmse_mm",
                "test_mape_percent",
                "validation_mae_mm",
                "validation_rmse_mm"
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
        "# LUMIERE HBGD FEATURE MACHINE-LEARNING ANALYSIS"
    )

    report.append("")

    report.append(
        "## 1. Objective"
    )

    report.append("")

    report.append(
        "This experiment tests whether HBGD-derived temporal features improve prediction of future tumor diameter when used as machine-learning features."
    )

    report.append("")

    report.append(
        "The experiment does not assume that the HBGD exponential equation itself is sufficient to forecast GBM progression."
    )

    report.append("")

    report.append(
        "## 2. Patient-Level Split"
    )

    report.append("")

    report.append(
        "| Split | Patients |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        "| Training | {} |".format(
            len(train_patients)
        )
    )

    report.append(
        "| Validation | {} |".format(
            len(validation_patients)
        )
    )

    report.append(
        "| Test | {} |".format(
            len(test_patients)
        )
    )

    report.append("")

    report.append(
        "The test patients were not used for model training or feature/model selection."
    )

    report.append("")

    report.append(
        "## 3. Prediction Cases"
    )

    report.append("")

    report.append(
        "| Split | Prediction cases |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        "| Training | {} |".format(
            len(train_rows)
        )
    )

    report.append(
        "| Validation | {} |".format(
            len(validation_rows)
        )
    )

    report.append(
        "| Test | {} |".format(
            len(test_rows)
        )
    )

    report.append("")

    report.append(
        "## 4. Models"
    )

    report.append("")

    report.append(
        "### No-change"
    )

    report.append("")

    report.append(
        "Future diameter is assumed to equal the current diameter."
    )

    report.append("")

    report.append(
        "### Geometry-only"
    )

    report.append("")

    report.append(
        "Uses tumor size, recent changes, interval duration, and longitudinal history information without HBGD rate/state features."
    )

    report.append("")

    report.append(
        "### HBGD-only"
    )

    report.append("")

    report.append(
        "Uses recent HBGD rate, multi-point HBGD rate, rate variability, state code, and overall HBGD rate."
    )

    report.append("")

    report.append(
        "### Geometry + HBGD"
    )

    report.append("")

    report.append(
        "Combines tumor geometry/history with HBGD temporal features."
    )

    report.append("")

    report.append(
        "## 5. Test Results"
    )

    report.append("")

    report.append(
        "| Model | MAE (mm) | RMSE (mm) | MAPE (%) |"
    )

    report.append(
        "|---|---:|---:|---:|"
    )

    for row in summary_rows:

        report.append(
            "| {} | {:.4f} | {:.4f} | {:.2f} |".format(
                row["model"],
                float(
                    row["test_mae_mm"]
                ),
                float(
                    row["test_rmse_mm"]
                ),
                float(
                    row["test_mape_percent"]
                )
            )
        )

    report.append("")

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    ml_rows = [
        row
        for row in summary_rows
        if row[
            "model"
        ] != "No_change"
    ]

    ml_rows.sort(
        key=lambda x: float(
            x["test_mae_mm"]
        )
    )

    best = ml_rows[
        0
    ]

    report.append(
        "## 6. Best ML Model"
    )

    report.append("")

    report.append(
        "**{}** achieved the lowest test MAE among the machine-learning models: {:.4f} mm.".format(
            best["model"],
            float(
                best["test_mae_mm"]
            )
        )
    )

    report.append("")

    baseline = float(
        no_change["mae"]
    )

    best_mae = float(
        best["test_mae_mm"]
    )

    if baseline > 0:

        best_improvement = (
            (
                baseline -
                best_mae
            )
            /
            baseline
        ) * 100.0

    else:

        best_improvement = 0.0

    report.append(
        "Improvement of the best ML model versus no-change: {:.2f}%.".format(
            best_improvement
        )
    )

    report.append("")

    # --------------------------------------------------------
    # HBGD contribution
    # --------------------------------------------------------

    geometry_row = None

    hbgd_row = None

    combined_row = None

    for row in summary_rows:

        if row[
            "model"
        ] == "Geometry_only":

            geometry_row = row

        elif row[
            "model"
        ] == "HBGD_only":

            hbgd_row = row

        elif row[
            "model"
        ] == "Geometry_plus_HBGD":

            combined_row = row

    report.append(
        "## 7. HBGD Contribution"
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

            contribution = (
                (
                    geometry_mae -
                    combined_mae
                )
                /
                geometry_mae
            ) * 100.0

            report.append(
                "**Adding HBGD features improved the test MAE over geometry/history-only features by {:.2f}%.** This supports the hypothesis that HBGD temporal information contributes predictive information beyond tumor geometry/history alone.".format(
                    contribution
                )
            )

        else:

            report.append(
                "**Adding HBGD features did not improve test MAE over geometry/history-only features.** Therefore, this experiment does not establish an incremental predictive contribution from HBGD features."
            )

    report.append("")

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    report.append(
        "## 8. Combined Model Feature Importance"
    )

    report.append("")

    report.append(
        "| Feature | Importance |"
    )

    report.append(
        "|---|---:|"
    )

    for row in importance_rows:

        report.append(
            "| {} | {:.6f} |".format(
                row["feature"],
                row["importance"]
            )
        )

    report.append("")

    # --------------------------------------------------------
    # Limitations
    # --------------------------------------------------------

    report.append(
        "## 9. Limitations"
    )

    report.append("")

    report.append(
        "The target is future equivalent spherical tumor diameter derived from segmentation volume."
    )

    report.append("")

    report.append(
        "The current temporal representation assumes seven days per LUMIERE week identifier."
    )

    report.append("")

    report.append(
        "The dataset is limited to the validated LUMIERE longitudinal cohort and the test set contains only 13 patients."
    )

    report.append("")

    report.append(
        "Random Forest performance should therefore be interpreted as an experiment on this cohort rather than proof of general clinical performance."
    )

    report.append("")

    # --------------------------------------------------------
    # Next
    # --------------------------------------------------------

    report.append(
        "## 10. Next Stage"
    )

    report.append("")

    report.append(
        "If the combined HBGD model demonstrates improvement, the next experiment should test whether HBGD representations transfer from the existing source-cancer dataset to LUMIERE GBM."
    )

    report.append("")

    report.append(
        "If HBGD does not improve the combined model, the transfer experiment should be framed as a representation-transfer investigation rather than assuming that HBGD provides predictive benefit."
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

    print("\n")
    print("=" * 70)
    print("STEP 7 COMPLETE")
    print("=" * 70)

    print(
        "\nTraining patients:",
        len(train_patients)
    )

    print(
        "Validation patients:",
        len(validation_patients)
    )

    print(
        "Test patients:",
        len(test_patients)
    )

    print(
        "\nTraining cases:",
        len(train_rows)
    )

    print(
        "Validation cases:",
        len(validation_rows)
    )

    print(
        "Test cases:",
        len(test_rows)
    )

    print(
        "\nMODEL RESULTS"
    )

    print(
        "-" * 70
    )

    for row in summary_rows:

        print(
            "{:<30} MAE={:>10.4f}  RMSE={:>10.4f}".format(
                row["model"],
                float(
                    row["test_mae_mm"]
                ),
                float(
                    row["test_rmse_mm"]
                )
            )
        )

    print(
        "-" * 70
    )

    print(
        "\nOUTPUT:"
    )

    print(
        RESULTS_CSV
    )

    print(
        SUMMARY_CSV
    )

    print(
        REPORT_PATH
    )

    print("=" * 70)


if __name__ == "__main__":

    main()