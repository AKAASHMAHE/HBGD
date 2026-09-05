# -*- coding: utf-8 -*-

"""
5_lumiere_hbgd_robust_prediction.py

STEP 5:
    Diagnose the failure of one-interval HBGD and evaluate
    multi-timepoint growth-rate estimation.

Models:

    1. No-change
    2. Global exponential
    3. One-interval HBGD
    4. Two-interval average HBGD
    5. Multi-point exponential regression
    6. Recent multi-point exponential regression

IMPORTANT:
    - Patient-level train/test split.
    - Same random seed as Step 4.
    - Test patients remain completely unseen.
    - Future observation is never used to estimate its prediction.
    - Week difference is converted to days using 7 days/week.
    - Python 3.6 compatible.

INPUT:
    lumiere_longitudinal_trajectories.csv

OUTPUT:
    lumiere_robust_hbgd_prediction_results.csv
    lumiere_robust_hbgd_patient_summary.csv
    LUMIERE_ROBUST_HBGD_REPORT.md
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

RESULTS_CSV = os.path.join(
    LUMIERE_DIR,
    "lumiere_robust_hbgd_prediction_results.csv"
)

PATIENT_SUMMARY_CSV = os.path.join(
    LUMIERE_DIR,
    "lumiere_robust_hbgd_patient_summary.csv"
)

REPORT_PATH = os.path.join(
    LUMIERE_DIR,
    "LUMIERE_ROBUST_HBGD_REPORT.md"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TEST_FRACTION = 0.20

DAYS_PER_WEEK = 7.0

# Number of previous observations used by
# recent multi-point regression.
RECENT_WINDOW = 4


# ============================================================
# BASIC FUNCTIONS
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

    return sum(values) / float(
        len(values)
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


def calculate_r(
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


def predict(
    diameter,
    rate,
    delta_weeks
):

    if (
        diameter <= 0
        or
        not finite(rate)
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
            diameter *
            math.exp(
                rate *
                delta_days
            )
        )

    except Exception:

        return float("nan")


# ============================================================
# LOAD DATA
# ============================================================

def load_patients():

    if not os.path.exists(
        TRAJECTORY_CSV
    ):

        raise IOError(
            "Missing file: {}".format(
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

    for patient_id in patients:

        patients[
            patient_id
        ].sort(
            key=lambda x: x["week"]
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
            n *
            TEST_FRACTION
        )
    )

    if n_test < 1:

        n_test = 1

    if n_test >= n:

        n_test = n - 1

    test = sorted(
        patient_ids[
            :n_test
        ]
    )

    train = sorted(
        patient_ids[
            n_test:
        ]
    )

    return train, test


# ============================================================
# GLOBAL RATE
# ============================================================

def global_rate(
    patients,
    patient_ids
):

    numerator = 0.0

    denominator = 0.0

    intervals = 0

    for patient_id in patient_ids:

        obs = patients[
            patient_id
        ]

        for i in range(
            len(obs) - 1
        ):

            d1 = obs[i][
                "diameter"
            ]

            d2 = obs[i + 1][
                "diameter"
            ]

            delta_weeks = (
                obs[i + 1]["week"] -
                obs[i]["week"]
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

            intervals += 1

    if denominator <= 0:

        return float("nan"), 0

    return (
        numerator /
        denominator,
        intervals
    )


# ============================================================
# LINEAR REGRESSION
# ============================================================

def linear_regression(
    x,
    y
):

    """
    Ordinary least squares:

        y = a + b*x

    Here:

        y = ln(D)

    and:

        x = elapsed days

    Therefore:

        b = exponential growth rate r
    """

    if len(x) != len(y):

        return float("nan")

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


# ============================================================
# MULTI-POINT RATE
# ============================================================

def multipoint_rate(
    observations
):

    """
    Fit:

        ln(D) = a + r*t

    using all observations supplied.
    """

    if len(
        observations
    ) < 2:

        return float("nan")

    first_week = observations[
        0
    ]["week"]

    x = []

    y = []

    for obs in observations:

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

    return linear_regression(
        x,
        y
    )


# ============================================================
# TWO-INTERVAL AVERAGE RATE
# ============================================================

def two_interval_rate(
    observations
):

    """
    Calculate the rate for each recent interval,
    then average the rates.

    This reduces the influence of one unusually
    large/small interval.
    """

    if len(
        observations
    ) < 3:

        return float("nan")

    rates = []

    start = max(
        0,
        len(observations) - 3
    )

    recent = observations[
        start:
    ]

    for i in range(
        len(recent) - 1
    ):

        d1 = recent[i][
            "diameter"
        ]

        d2 = recent[i + 1][
            "diameter"
        ]

        delta_weeks = (
            recent[i + 1]["week"] -
            recent[i]["week"]
        )

        r = calculate_r(
            d1,
            d2,
            delta_weeks
        )

        if finite(r):

            rates.append(
                r
            )

    if not rates:

        return float("nan")

    return mean(
        rates
    )


# ============================================================
# CREATE TEST CASES
# ============================================================

def create_cases(
    patients,
    test_patients,
    global_r
):

    results = []

    for patient_id in test_patients:

        obs = patients[
            patient_id
        ]

        # Need at least 3 observations
        if len(obs) < 3:

            continue

        # ----------------------------------------------------
        # Predict each observation after the first two
        # ----------------------------------------------------

        for target_index in range(
            2,
            len(obs)
        ):

            target = obs[
                target_index
            ]

            current = obs[
                target_index - 1
            ]

            previous = obs[
                target_index - 2
            ]

            d_previous = previous[
                "diameter"
            ]

            d_current = current[
                "diameter"
            ]

            d_actual = target[
                "diameter"
            ]

            train_delta = (
                current["week"] -
                previous["week"]
            )

            future_delta = (
                target["week"] -
                current["week"]
            )

            if (
                d_previous <= 0
                or
                d_current <= 0
                or
                d_actual <= 0
                or
                train_delta <= 0
                or
                future_delta <= 0
            ):

                continue

            # ------------------------------------------------
            # 1. One interval
            # ------------------------------------------------

            r_one = calculate_r(
                d_previous,
                d_current,
                train_delta
            )

            # ------------------------------------------------
            # 2. Two interval average
            #
            # Uses all observations BEFORE target.
            # ------------------------------------------------

            history = obs[
                :target_index
            ]

            r_two = two_interval_rate(
                history
            )

            # ------------------------------------------------
            # 3. All-history regression
            # ------------------------------------------------

            r_multi = multipoint_rate(
                history
            )

            # ------------------------------------------------
            # 4. Recent-window regression
            # ------------------------------------------------

            recent_history = history[
                -RECENT_WINDOW:
            ]

            r_recent = multipoint_rate(
                recent_history
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            pred_no_change = (
                d_current
            )

            pred_global = predict(
                d_current,
                global_r,
                future_delta
            )

            pred_one = predict(
                d_current,
                r_one,
                future_delta
            )

            pred_two = predict(
                d_current,
                r_two,
                future_delta
            )

            pred_multi = predict(
                d_current,
                r_multi,
                future_delta
            )

            pred_recent = predict(
                d_current,
                r_recent,
                future_delta
            )

            # ------------------------------------------------
            # Errors
            # ------------------------------------------------

            actual = d_actual

            result = {

                "patient_id":
                    patient_id,

                "previous_week":
                    previous["week"],

                "current_week":
                    current["week"],

                "future_week":
                    target["week"],

                "history_points":
                    target_index,

                "train_delta_weeks":
                    train_delta,

                "future_delta_weeks":
                    future_delta,

                "previous_diameter_mm":
                    d_previous,

                "current_diameter_mm":
                    d_current,

                "actual_future_diameter_mm":
                    actual,

                "r_one_interval":
                    r_one,

                "r_two_interval_average":
                    r_two,

                "r_all_history":
                    r_multi,

                "r_recent_history":
                    r_recent,

                "r_global":
                    global_r,

                "no_change_prediction_mm":
                    pred_no_change,

                "global_prediction_mm":
                    pred_global,

                "one_interval_hbgd_prediction_mm":
                    pred_one,

                "two_interval_hbgd_prediction_mm":
                    pred_two,

                "multi_point_hbgd_prediction_mm":
                    pred_multi,

                "recent_window_hbgd_prediction_mm":
                    pred_recent,

                "no_change_error_mm":
                    abs(
                        actual -
                        pred_no_change
                    ),

                "global_error_mm":
                    abs(
                        actual -
                        pred_global
                    ),

                "one_interval_error_mm":
                    abs(
                        actual -
                        pred_one
                    ),

                "two_interval_error_mm":
                    abs(
                        actual -
                        pred_two
                    ),

                "multi_point_error_mm":
                    abs(
                        actual -
                        pred_multi
                    ),

                "recent_window_error_mm":
                    abs(
                        actual -
                        pred_recent
                    )
            }

            results.append(
                result
            )

    return results


# ============================================================
# MODEL METRICS
# ============================================================

def model_metrics(
    results,
    prediction_column
):

    actual = []

    predicted = []

    for row in results:

        a = safe_float(
            row[
                "actual_future_diameter_mm"
            ]
        )

        p = safe_float(
            row[
                prediction_column
            ]
        )

        if (
            finite(a)
            and
            finite(p)
        ):

            actual.append(
                a
            )

            predicted.append(
                p
            )

    return {

        "n":
            len(actual),

        "mae":
            mae(
                actual,
                predicted
            ),

        "rmse":
            rmse(
                actual,
                predicted
            ),

        "mape":
            mape(
                actual,
                predicted
            ),

        "median_ae":
            median([
                abs(a - p)
                for a, p in zip(
                    actual,
                    predicted
                )
            ])

    }


# ============================================================
# GROWTH/STATE ANALYSIS
# ============================================================

def classify_rate(
    rate
):

    if not finite(rate):

        return "UNKNOWN"

    if rate > 0.0005:

        return "GROWTH"

    if rate < -0.0010:

        return "SHRINKAGE"

    return "STABLE"


def state_metrics(
    results,
    prediction_column
):

    groups = {

        "GROWTH": [],
        "STABLE": [],
        "SHRINKAGE": []
    }

    for row in results:

        rate = safe_float(
            row[
                "r_one_interval"
            ]
        )

        state = classify_rate(
            rate
        )

        if state not in groups:

            continue

        actual = safe_float(
            row[
                "actual_future_diameter_mm"
            ]
        )

        prediction = safe_float(
            row[
                prediction_column
            ]
        )

        if (
            finite(actual)
            and
            finite(prediction)
        ):

            groups[
                state
            ].append(
                abs(
                    actual -
                    prediction
                )
            )

    output = {}

    for state in groups:

        values = groups[
            state
        ]

        output[
            state
        ] = {

            "n":
                len(values),

            "mae":
                mean(values)

        }

    return output


# ============================================================
# SAVE
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
# PATIENT SUMMARY
# ============================================================

def create_patient_summary(
    results
):

    patient_ids = sorted(
        list(
            set(
                row[
                    "patient_id"
                ]
                for row in results
            )
        )
    )

    models = {

        "no_change":
            "no_change_prediction_mm",

        "global":
            "global_prediction_mm",

        "one_interval":
            "one_interval_hbgd_prediction_mm",

        "two_interval":
            "two_interval_hbgd_prediction_mm",

        "multi_point":
            "multi_point_hbgd_prediction_mm",

        "recent_window":
            "recent_window_hbgd_prediction_mm"
    }

    summaries = []

    for patient_id in patient_ids:

        rows = [
            row
            for row in results
            if row[
                "patient_id"
            ] == patient_id
        ]

        output = {

            "patient_id":
                patient_id,

            "prediction_cases":
                len(rows)
        }

        for model_name, column in models.items():

            metrics = model_metrics(
                rows,
                column
            )

            output[
                model_name + "_mae_mm"
            ] = metrics[
                "mae"
            ]

            output[
                model_name + "_rmse_mm"
            ] = metrics[
                "rmse"
            ]

        summaries.append(
            output
        )

    return summaries


# ============================================================
# REPORT
# ============================================================

def build_report(
    patients,
    train_patients,
    test_patients,
    global_r,
    global_intervals,
    results
):

    models = [

        (
            "No-change",
            "no_change_prediction_mm"
        ),

        (
            "Global exponential",
            "global_prediction_mm"
        ),

        (
            "One-interval HBGD",
            "one_interval_hbgd_prediction_mm"
        ),

        (
            "Two-interval average HBGD",
            "two_interval_hbgd_prediction_mm"
        ),

        (
            "Multi-point HBGD",
            "multi_point_hbgd_prediction_mm"
        ),

        (
            "Recent-window HBGD",
            "recent_window_hbgd_prediction_mm"
        )
    ]

    metrics = {}

    for name, column in models:

        metrics[
            name
        ] = model_metrics(
            results,
            column
        )

    report = []

    report.append(
        "# LUMIERE ROBUST HBGD PREDICTION"
    )

    report.append("")

    report.append(
        "## 1. Objective"
    )

    report.append("")

    report.append(
        "Step 4 showed that the one-interval individualized HBGD model did not outperform the no-change baseline. This experiment investigates whether the poor performance is caused by unstable single-interval rate estimation."
    )

    report.append("")

    report.append(
        "Multiple historical observations are therefore used to estimate a more robust exponential growth rate."
    )

    report.append("")

    report.append(
        "## 2. Dataset Split"
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
            len(results)
        )
    )

    report.append("")

    report.append(
        "Patient-level splitting is maintained. No test-patient observations are used to estimate the global rate."
    )

    report.append("")

    report.append(
        "## 3. Models"
    )

    report.append("")

    report.append(
        "### No-change"
    )

    report.append("")

    report.append(
        "D_future = D_current"
    )

    report.append("")

    report.append(
        "### Global exponential"
    )

    report.append("")

    report.append(
        "Uses one population-level rate estimated only from training patients."
    )

    report.append("")

    report.append(
        "### One-interval HBGD"
    )

    report.append("")

    report.append(
        "Uses the immediately preceding observation pair."
    )

    report.append("")

    report.append(
        "### Two-interval average HBGD"
    )

    report.append("")

    report.append(
        "Averages the two most recent interval-specific growth rates."
    )

    report.append("")

    report.append(
        "### Multi-point HBGD"
    )

    report.append("")

    report.append(
        "Fits ln(D) = a + r*t using all observations available before the target observation."
    )

    report.append("")

    report.append(
        "### Recent-window HBGD"
    )

    report.append("")

    report.append(
        "Fits the exponential model using the most recent {} historical observations.".format(
            RECENT_WINDOW
        )
    )

    report.append("")

    report.append(
        "## 4. Results"
    )

    report.append("")

    report.append(
        "| Model | MAE (mm) | RMSE (mm) | Median AE (mm) | MAPE (%) | N |"
    )

    report.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for name, column in models:

        m = metrics[
            name
        ]

        report.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.2f} | {} |".format(
                name,
                m["mae"],
                m["rmse"],
                m["median_ae"],
                m["mape"],
                m["n"]
            )
        )

    report.append("")

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    valid_models = []

    for name, column in models:

        m = metrics[
            name
        ]

        if finite(
            m["mae"]
        ):

            valid_models.append(
                (
                    m["mae"],
                    name
                )
            )

    valid_models.sort()

    if valid_models:

        best_mae = valid_models[
            0
        ][0]

        best_model = valid_models[
            0
        ][1]

        report.append(
            "### Best MAE"
        )

        report.append("")

        report.append(
            "**{}** achieved the lowest MAE at {:.4f} mm.".format(
                best_model,
                best_mae
            )
        )

        report.append("")

    # --------------------------------------------------------
    # Improvement
    # --------------------------------------------------------

    no_change_mae = metrics[
        "No-change"
    ]["mae"]

    for name in [
        "One-interval HBGD",
        "Two-interval average HBGD",
        "Multi-point HBGD",
        "Recent-window HBGD"
    ]:

        model_mae = metrics[
            name
        ]["mae"]

        if (
            finite(no_change_mae)
            and
            no_change_mae > 0
            and
            finite(model_mae)
        ):

            improvement = (
                (
                    no_change_mae -
                    model_mae
                )
                /
                no_change_mae
            ) * 100.0

            report.append(
                "{} vs no-change: {:.2f}% improvement.".format(
                    name,
                    improvement
                )
            )

    report.append("")

    # --------------------------------------------------------
    # State-specific analysis
    # --------------------------------------------------------

    report.append(
        "## 5. Growth-State Analysis"
    )

    report.append("")

    state_model_columns = [

        (
            "No-change",
            "no_change_prediction_mm"
        ),

        (
            "One-interval HBGD",
            "one_interval_hbgd_prediction_mm"
        ),

        (
            "Multi-point HBGD",
            "multi_point_hbgd_prediction_mm"
        ),

        (
            "Recent-window HBGD",
            "recent_window_hbgd_prediction_mm"
        )
    ]

    report.append(
        "| State | N | No-change MAE | One-interval MAE | Multi-point MAE | Recent-window MAE |"
    )

    report.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    state_groups = {

        "GROWTH": [],
        "STABLE": [],
        "SHRINKAGE": []
    }

    for row in results:

        state = classify_rate(
            safe_float(
                row[
                    "r_one_interval"
                ]
            )
        )

        if state in state_groups:

            state_groups[
                state
            ].append(
                row
            )

    for state in [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]:

        rows = state_groups[
            state
        ]

        values = []

        for name, column in state_model_columns:

            m = model_metrics(
                rows,
                column
            )

            values.append(
                m["mae"]
            )

        report.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                state,
                len(rows),
                values[0],
                values[1],
                values[2],
                values[3]
            )
        )

    report.append("")

    # --------------------------------------------------------
    # Rate instability
    # --------------------------------------------------------

    report.append(
        "## 6. Rate Stability Diagnosis"
    )

    report.append("")

    one_rates = []

    multi_rates = []

    recent_rates = []

    for row in results:

        for value, container in [

            (
                row[
                    "r_one_interval"
                ],
                one_rates
            ),

            (
                row[
                    "r_all_history"
                ],
                multi_rates
            ),

            (
                row[
                    "r_recent_history"
                ],
                recent_rates
            )

        ]:

            value = safe_float(
                value
            )

            if finite(value):

                container.append(
                    value
                )

    report.append(
        "| Rate estimator | N | Mean r/day | Median r/day |"
    )

    report.append(
        "|---|---:|---:|---:|"
    )

    report.append(
        "| One-interval | {} | {:.8f} | {:.8f} |".format(
            len(one_rates),
            mean(one_rates),
            median(one_rates)
        )
    )

    report.append(
        "| Multi-point | {} | {:.8f} | {:.8f} |".format(
            len(multi_rates),
            mean(multi_rates),
            median(multi_rates)
        )
    )

    report.append(
        "| Recent-window | {} | {:.8f} | {:.8f} |".format(
            len(recent_rates),
            mean(recent_rates),
            median(recent_rates)
        )
    )

    report.append("")

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    report.append(
        "## 7. Interpretation"
    )

    report.append("")

    robust_models = [

        "Two-interval average HBGD",
        "Multi-point HBGD",
        "Recent-window HBGD"
    ]

    robust_best = None

    robust_best_mae = float("inf")

    for name in robust_models:

        value = metrics[
            name
        ]["mae"]

        if (
            finite(value)
            and
            value < robust_best_mae
        ):

            robust_best_mae = value

            robust_best = name

    if robust_best is not None:

        if (
            robust_best_mae <
            no_change_mae
        ):

            report.append(
                "**At least one robust multi-timepoint HBGD estimator outperformed the no-change baseline.** This would indicate that the poor Step 4 result was at least partly associated with unstable single-interval rate estimation."
            )

        else:

            report.append(
                "**The robust multi-timepoint HBGD estimators did not outperform the no-change baseline.** This would suggest that simply stabilizing the exponential growth-rate estimate is insufficient for reliable GBM forward prediction."
            )

    report.append("")

    report.append(
        "This experiment does not establish that HBGD is universally predictive or non-predictive. It evaluates the specific exponential diameter formulation on this LUMIERE test split."
    )

    report.append("")

    # --------------------------------------------------------
    # Temporal limitation
    # --------------------------------------------------------

    report.append(
        "## 8. Temporal Assumption"
    )

    report.append("")

    report.append(
        "The current implementation converts differences in LUMIERE week identifiers to days using 7 days per week. This assumption should be verified against the official LUMIERE documentation."
    )

    report.append("")

    # --------------------------------------------------------
    # Next stage
    # --------------------------------------------------------

    report.append(
        "## 9. Next Decision"
    )

    report.append("")

    report.append(
        "If a robust HBGD estimator improves prediction, it can become the HBGD baseline for the cross-cancer transfer experiment."
    )

    report.append("")

    report.append(
        "If no HBGD estimator improves prediction, the project should not claim that HBGD alone predicts GBM progression. The subsequent transfer experiment should instead test whether learned growth representations or auxiliary HBGD features improve a supervised predictive model."
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
        "STEP 5 - ROBUST LUMIERE HBGD PREDICTION"
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

    train_patients, test_patients = (
        split_patients(
            patients
        )
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
    # Global rate
    # --------------------------------------------------------

    global_r, global_intervals = (
        global_rate(
            patients,
            train_patients
        )
    )

    print(
        "\nGlobal rate:",
        "{:.8f}".format(
            global_r
        )
    )

    print(
        "Training intervals:",
        global_intervals
    )

    # --------------------------------------------------------
    # Test cases
    # --------------------------------------------------------

    print(
        "\nGenerating test cases..."
    )

    results = create_cases(
        patients,
        test_patients,
        global_r
    )

    print(
        "Prediction cases:",
        len(results)
    )

    if not results:

        print(
            "\nERROR: No test cases generated."
        )

        return

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    models = [

        (
            "No-change",
            "no_change_prediction_mm"
        ),

        (
            "Global exponential",
            "global_prediction_mm"
        ),

        (
            "One-interval HBGD",
            "one_interval_hbgd_prediction_mm"
        ),

        (
            "Two-interval average HBGD",
            "two_interval_hbgd_prediction_mm"
        ),

        (
            "Multi-point HBGD",
            "multi_point_hbgd_prediction_mm"
        ),

        (
            "Recent-window HBGD",
            "recent_window_hbgd_prediction_mm"
        )
    ]

    print("\n")
    print(
        "-" * 70
    )

    print(
        "{:<30} {:>12} {:>12}".format(
            "MODEL",
            "MAE",
            "RMSE"
        )
    )

    print(
        "-" * 70
    )

    for name, column in models:

        metrics = model_metrics(
            results,
            column
        )

        print(
            "{:<30} {:>12.4f} {:>12.4f}".format(
                name,
                metrics["mae"],
                metrics["rmse"]
            )
        )

    print(
        "-" * 70
    )

    # --------------------------------------------------------
    # Patient summary
    # --------------------------------------------------------

    patient_summary = (
        create_patient_summary(
            results
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_csv(
        RESULTS_CSV,
        results
    )

    save_csv(
        PATIENT_SUMMARY_CSV,
        patient_summary
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = build_report(
        patients,
        train_patients,
        test_patients,
        global_r,
        global_intervals,
        results
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
    # Final
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 5 COMPLETE")
    print("=" * 70)

    print(
        "\nDetailed results:"
    )

    print(
        RESULTS_CSV
    )

    print(
        "\nPatient summary:"
    )

    print(
        PATIENT_SUMMARY_CSV
    )

    print(
        "\nReport:"
    )

    print(
        REPORT_PATH
    )

    print("=" * 70)


if __name__ == "__main__":

    main()