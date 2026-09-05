# -*- coding: utf-8 -*-

"""
6_lumiere_state_aware_hbgd.py

STEP 6:
    State-aware HBGD analysis for LUMIERE.

Purpose:
    Determine whether HBGD dynamic states contain predictive
    information about subsequent tumor behavior.

States:
    GROWTH
    STABLE
    SHRINKAGE

Thresholds:
    GROWTH    r > +0.0005/day
    STABLE   -0.0010 <= r <= +0.0005/day
    SHRINKAGE r < -0.0010/day

Experiments:
    1. Current-state persistence
    2. State transition matrix
    3. State-specific future diameter change
    4. State-specific prediction error
    5. State-aware prediction
    6. Comparison against no-change baseline
    7. Growth-state transition statistics

Patient-level split:
    Same seed and split as Steps 4 and 5.

Python:
    3.6 compatible.
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
    "lumiere_state_aware_results.csv"
)

TRANSITION_CSV = os.path.join(
    LUMIERE_DIR,
    "lumiere_hbgd_state_transition_matrix.csv"
)

PATIENT_SUMMARY_CSV = os.path.join(
    LUMIERE_DIR,
    "lumiere_state_aware_patient_summary.csv"
)

REPORT_PATH = os.path.join(
    LUMIERE_DIR,
    "LUMIERE_STATE_AWARE_HBGD_REPORT.md"
)


# ============================================================
# PARAMETERS
# ============================================================

RANDOM_SEED = 42

TEST_FRACTION = 0.20

DAYS_PER_WEEK = 7.0

GROWTH_THRESHOLD = 0.0005

STABLE_LOWER = -0.0010

STABLE_UPPER = 0.0005


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


def mae(
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

    return mean(errors)


def rmse(
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
                (a - p) ** 2
            )

    if not errors:

        return float("nan")

    return math.sqrt(
        mean(errors)
    )


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


def classify_state(
    rate
):

    if not finite(rate):

        return "UNKNOWN"

    if rate > GROWTH_THRESHOLD:

        return "GROWTH"

    if rate < STABLE_LOWER:

        return "SHRINKAGE"

    return "STABLE"


# ============================================================
# LOAD TRAJECTORIES
# ============================================================

def load_trajectories():

    patients = {}

    if not os.path.exists(
        TRAJECTORY_CSV
    ):

        raise IOError(
            "Missing trajectory file: {}".format(
                TRAJECTORY_CSV
            )
        )

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
# GLOBAL STATE-SPECIFIC RATES
# ============================================================

def calculate_state_rates(
    patients,
    train_patients
):

    state_numerator = {

        "GROWTH": 0.0,
        "STABLE": 0.0,
        "SHRINKAGE": 0.0
    }

    state_denominator = {

        "GROWTH": 0.0,
        "STABLE": 0.0,
        "SHRINKAGE": 0.0
    }

    state_intervals = {

        "GROWTH": 0,
        "STABLE": 0,
        "SHRINKAGE": 0
    }

    for patient_id in train_patients:

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

            r = calculate_r(
                d1,
                d2,
                delta_weeks
            )

            state = classify_state(
                r
            )

            if state not in state_numerator:

                continue

            delta_days = (
                delta_weeks *
                DAYS_PER_WEEK
            )

            state_numerator[
                state
            ] += math.log(
                d2 / d1
            )

            state_denominator[
                state
            ] += delta_days

            state_intervals[
                state
            ] += 1

    state_rates = {}

    for state in state_numerator:

        denominator = (
            state_denominator[
                state
            ]
        )

        if denominator > 0:

            state_rates[
                state
            ] = (
                state_numerator[
                    state
                ] /
                denominator
            )

        else:

            state_rates[
                state
            ] = float("nan")

    return (
        state_rates,
        state_intervals
    )


# ============================================================
# CREATE STATE-AWARE TEST CASES
# ============================================================

def create_test_cases(
    patients,
    test_patients,
    state_rates
):

    results = []

    for patient_id in test_patients:

        obs = patients[
            patient_id
        ]

        if len(obs) < 3:

            continue

        # ----------------------------------------------------
        # Every future observation after the first two
        # ----------------------------------------------------

        for target_index in range(
            2,
            len(obs)
        ):

            previous = obs[
                target_index - 2
            ]

            current = obs[
                target_index - 1
            ]

            future = obs[
                target_index
            ]

            d_previous = previous[
                "diameter"
            ]

            d_current = current[
                "diameter"
            ]

            d_future = future[
                "diameter"
            ]

            train_delta = (
                current["week"] -
                previous["week"]
            )

            future_delta = (
                future["week"] -
                current["week"]
            )

            if (
                d_previous <= 0
                or
                d_current <= 0
                or
                d_future <= 0
                or
                train_delta <= 0
                or
                future_delta <= 0
            ):

                continue

            # ------------------------------------------------
            # Current HBGD rate/state
            # ------------------------------------------------

            current_r = calculate_r(
                d_previous,
                d_current,
                train_delta
            )

            current_state = classify_state(
                current_r
            )

            # ------------------------------------------------
            # State-aware prediction
            # ------------------------------------------------

            state_rate = state_rates.get(
                current_state,
                float("nan")
            )

            if finite(
                state_rate
            ):

                state_prediction = (
                    d_current *
                    math.exp(
                        state_rate *
                        future_delta *
                        DAYS_PER_WEEK
                    )
                )

            else:

                state_prediction = (
                    d_current
                )

            # ------------------------------------------------
            # No-change
            # ------------------------------------------------

            no_change_prediction = (
                d_current
            )

            # ------------------------------------------------
            # Actual future change
            # ------------------------------------------------

            future_change_mm = (
                d_future -
                d_current
            )

            future_percent_change = (
                (
                    d_future -
                    d_current
                )
                /
                d_current
            ) * 100.0

            # Actual future state
            future_r = calculate_r(
                d_current,
                d_future,
                future_delta
            )

            future_state = classify_state(
                future_r
            )

            # ------------------------------------------------
            # State persistence
            # ------------------------------------------------

            state_persisted = (
                1
                if current_state ==
                future_state
                else 0
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

                "current_diameter_mm":
                    d_current,

                "future_diameter_mm":
                    d_future,

                "current_r_per_day":
                    current_r,

                "current_state":
                    current_state,

                "state_rate_per_day":
                    state_rate,

                "state_prediction_mm":
                    state_prediction,

                "no_change_prediction_mm":
                    no_change_prediction,

                "state_prediction_error_mm":
                    abs(
                        d_future -
                        state_prediction
                    ),

                "no_change_error_mm":
                    abs(
                        d_future -
                        no_change_prediction
                    ),

                "future_r_per_day":
                    future_r,

                "future_state":
                    future_state,

                "state_persisted":
                    state_persisted,

                "future_change_mm":
                    future_change_mm,

                "future_percent_change":
                    future_percent_change
            })

    return results


# ============================================================
# TRANSITION MATRIX
# ============================================================

def transition_matrix(
    results
):

    states = [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]

    counts = {}

    for current in states:

        counts[
            current
        ] = {}

        for future in states:

            counts[
                current
            ][
                future
            ] = 0

    for row in results:

        current = row[
            "current_state"
        ]

        future = row[
            "future_state"
        ]

        if (
            current in counts
            and
            future in counts[current]
        ):

            counts[
                current
            ][
                future
            ] += 1

    return counts


def transition_percentages(
    counts
):

    percentages = {}

    for current in counts:

        total = sum(
            counts[
                current
            ].values()
        )

        percentages[
            current
        ] = {}

        for future in counts[
            current
        ]:

            if total > 0:

                percentages[
                    current
                ][
                    future
                ] = (
                    counts[
                        current
                    ][
                        future
                    ]
                    /
                    float(total)
                ) * 100.0

            else:

                percentages[
                    current
                ][
                    future
                ] = 0.0

    return percentages


# ============================================================
# STATE METRICS
# ============================================================

def calculate_state_metrics(
    results
):

    states = [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]

    output = {}

    for state in states:

        rows = [
            row
            for row in results
            if row[
                "current_state"
            ] == state
        ]

        actual = [
            row[
                "future_diameter_mm"
            ]
            for row in rows
        ]

        state_prediction = [
            row[
                "state_prediction_mm"
            ]
            for row in rows
        ]

        no_change = [
            row[
                "no_change_prediction_mm"
            ]
            for row in rows
        ]

        persistence = [
            row[
                "state_persisted"
            ]
            for row in rows
        ]

        future_changes = [
            row[
                "future_change_mm"
            ]
            for row in rows
        ]

        output[
            state
        ] = {

            "n":
                len(rows),

            "state_mae":
                mae(
                    actual,
                    state_prediction
                ),

            "state_rmse":
                rmse(
                    actual,
                    state_prediction
                ),

            "no_change_mae":
                mae(
                    actual,
                    no_change
                ),

            "no_change_rmse":
                rmse(
                    actual,
                    no_change
                ),

            "mean_future_change_mm":
                mean(
                    future_changes
                ),

            "median_future_change_mm":
                median(
                    future_changes
                ),

            "state_persistence_percent":
                mean(
                    persistence
                ) * 100.0

        }

    return output


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
# PATIENT SUMMARY
# ============================================================

def patient_summary(
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

    summaries = []

    for patient_id in patient_ids:

        rows = [
            row
            for row in results
            if row[
                "patient_id"
            ] == patient_id
        ]

        actual = [
            row[
                "future_diameter_mm"
            ]
            for row in rows
        ]

        state_prediction = [
            row[
                "state_prediction_mm"
            ]
            for row in rows
        ]

        no_change = [
            row[
                "no_change_prediction_mm"
            ]
            for row in rows
        ]

        persistence = [
            row[
                "state_persisted"
            ]
            for row in rows
        ]

        summaries.append({

            "patient_id":
                patient_id,

            "prediction_cases":
                len(rows),

            "state_aware_mae_mm":
                mae(
                    actual,
                    state_prediction
                ),

            "no_change_mae_mm":
                mae(
                    actual,
                    no_change
                ),

            "state_persistence_percent":
                mean(
                    persistence
                ) * 100.0
        })

    return summaries


# ============================================================
# REPORT
# ============================================================

def build_report(
    patients,
    train_patients,
    test_patients,
    state_rates,
    state_intervals,
    results
):

    metrics = calculate_state_metrics(
        results
    )

    counts = transition_matrix(
        results
    )

    percentages = transition_percentages(
        counts
    )

    all_actual = [
        row[
            "future_diameter_mm"
        ]
        for row in results
    ]

    all_state_prediction = [
        row[
            "state_prediction_mm"
        ]
        for row in results
    ]

    all_no_change = [
        row[
            "no_change_prediction_mm"
        ]
        for row in results
    ]

    state_mae = mae(
        all_actual,
        all_state_prediction
    )

    state_rmse = rmse(
        all_actual,
        all_state_prediction
    )

    no_change_mae = mae(
        all_actual,
        all_no_change
    )

    no_change_rmse = rmse(
        all_actual,
        all_no_change
    )

    report = []

    report.append(
        "# LUMIERE STATE-AWARE HBGD ANALYSIS"
    )

    report.append("")

    report.append(
        "## 1. Objective"
    )

    report.append("")

    report.append(
        "This experiment evaluates whether the current HBGD dynamic state contains useful information about subsequent tumor behavior."
    )

    report.append("")

    report.append(
        "The analysis follows the same patient-level test split used in Steps 4 and 5."
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
            len(results)
        )
    )

    report.append("")

    # --------------------------------------------------------
    # State rates
    # --------------------------------------------------------

    report.append(
        "## 3. State-Specific Training Rates"
    )

    report.append("")

    report.append(
        "| State | Training intervals | Estimated r/day |"
    )

    report.append(
        "|---|---:|---:|"
    )

    for state in [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]:

        rate = state_rates[
            state
        ]

        if finite(rate):

            rate_text = "{:.8f}".format(
                rate
            )

        else:

            rate_text = "N/A"

        report.append(
            "| {} | {} | {} |".format(
                state,
                state_intervals[
                    state
                ],
                rate_text
            )
        )

    report.append("")

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    report.append(
        "## 4. Overall Prediction"
    )

    report.append("")

    report.append(
        "| Model | MAE (mm) | RMSE (mm) |"
    )

    report.append(
        "|---|---:|---:|"
    )

    report.append(
        "| No-change | {:.4f} | {:.4f} |".format(
            no_change_mae,
            no_change_rmse
        )
    )

    report.append(
        "| State-aware HBGD | {:.4f} | {:.4f} |".format(
            state_mae,
            state_rmse
        )
    )

    report.append("")

    if no_change_mae > 0:

        improvement = (
            (
                no_change_mae -
                state_mae
            )
            /
            no_change_mae
        ) * 100.0

    else:

        improvement = 0.0

    report.append(
        "State-aware HBGD improvement versus no-change: {:.2f}%.".format(
            improvement
        )
    )

    report.append("")

    # --------------------------------------------------------
    # State-specific
    # --------------------------------------------------------

    report.append(
        "## 5. State-Specific Future Behavior"
    )

    report.append("")

    report.append(
        "| Current state | N | State-aware MAE | No-change MAE | Mean future diameter change | State persistence |"
    )

    report.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for state in [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]:

        m = metrics[
            state
        ]

        report.append(
            "| {} | {} | {:.4f} | {:.4f} | {:.4f} mm | {:.2f}% |".format(
                state,
                m["n"],
                m["state_mae"],
                m["no_change_mae"],
                m["mean_future_change_mm"],
                m["state_persistence_percent"]
            )
        )

    report.append("")

    # --------------------------------------------------------
    # Transition matrix
    # --------------------------------------------------------

    report.append(
        "## 6. HBGD State Transition Matrix"
    )

    report.append("")

    report.append(
        "| Current → Future | Growth | Stable | Shrinkage |"
    )

    report.append(
        "|---|---:|---:|---:|"
    )

    for current in [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]:

        report.append(
            "| {} | {:.2f}% | {:.2f}% | {:.2f}% |".format(
                current,
                percentages[
                    current
                ]["GROWTH"],
                percentages[
                    current
                ]["STABLE"],
                percentages[
                    current
                ]["SHRINKAGE"]
            )
        )

    report.append("")

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    report.append(
        "### Transition counts"
    )

    report.append("")

    report.append(
        "| Current → Future | Growth | Stable | Shrinkage |"
    )

    report.append(
        "|---|---:|---:|---:|"
    )

    for current in [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]:

        report.append(
            "| {} | {} | {} | {} |".format(
                current,
                counts[
                    current
                ]["GROWTH"],
                counts[
                    current
                ]["STABLE"],
                counts[
                    current
                ]["SHRINKAGE"]
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

    if state_mae < no_change_mae:

        report.append(
            "**The state-aware model outperformed the no-change baseline on the held-out test cases.** This indicates that the HBGD state provides predictive information about subsequent tumor diameter."
        )

    else:

        report.append(
            "**The state-aware model did not outperform the no-change baseline overall.** Therefore, HBGD state alone does not provide sufficient predictive accuracy for this test set."
        )

    report.append("")

    # State-specific interpretation

    for state in [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]:

        m = metrics[
            state
        ]

        if m["n"] == 0:

            continue

        if (
            m["state_mae"] <
            m["no_change_mae"]
        ):

            report.append(
                "For **{}**, state-aware prediction outperformed no-change.".format(
                    state
                )
            )

        else:

            report.append(
                "For **{}**, state-aware prediction did not outperform no-change.".format(
                    state
                )
            )

    report.append("")

    report.append(
        "## 8. Scientific Interpretation"
    )

    report.append("")

    report.append(
        "The purpose of this analysis is diagnostic as well as predictive. If HBGD states show meaningful persistence or distinct future behavior even when the direct exponential forecast is inaccurate, the states may remain useful as features in a larger predictive-intelligence model."
    )

    report.append("")

    report.append(
        "Conversely, if states show weak persistence and poor state-aware prediction, HBGD state alone should not be presented as a reliable GBM forecasting mechanism."
    )

    report.append("")

    report.append(
        "## 9. Temporal Limitation"
    )

    report.append("")

    report.append(
        "Time intervals are currently derived from LUMIERE week identifiers and converted using 7 days per week. This temporal interpretation must be confirmed against official dataset documentation."
    )

    report.append("")

    report.append(
        "## 10. Next Decision"
    )

    report.append("")

    report.append(
        "The next stage should combine HBGD-derived features with predictive machine-learning models rather than relying on exponential HBGD alone. Candidate features include current diameter, volume, recent growth rate, growth state, rate variability, interval duration, and longitudinal trend."
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
        "STEP 6 - STATE-AWARE LUMIERE HBGD ANALYSIS"
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

    patients = load_trajectories()

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
    # State rates
    # --------------------------------------------------------

    print(
        "\nCalculating state-specific rates..."
    )

    (
        state_rates,
        state_intervals
    ) = calculate_state_rates(
        patients,
        train_patients
    )

    for state in [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]:

        print(
            "{}: {} intervals, r = {:.8f}".format(
                state,
                state_intervals[
                    state
                ],
                state_rates[
                    state
                ]
            )
        )

    # --------------------------------------------------------
    # Test cases
    # --------------------------------------------------------

    print(
        "\nCreating state-aware test cases..."
    )

    results = create_test_cases(
        patients,
        test_patients,
        state_rates
    )

    print(
        "Prediction cases:",
        len(results)
    )

    if not results:

        print(
            "ERROR: No test cases."
        )

        return

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_state_metrics(
        results
    )

    all_actual = [
        row[
            "future_diameter_mm"
        ]
        for row in results
    ]

    all_state = [
        row[
            "state_prediction_mm"
        ]
        for row in results
    ]

    all_no_change = [
        row[
            "no_change_prediction_mm"
        ]
        for row in results
    ]

    print("\n")
    print(
        "-" * 70
    )

    print(
        "{:<20} {:>10} {:>12} {:>12}".format(
            "STATE",
            "N",
            "STATE MAE",
            "NO-CHANGE MAE"
        )
    )

    print(
        "-" * 70
    )

    for state in [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]:

        m = metrics[
            state
        ]

        print(
            "{:<20} {:>10} {:>12.4f} {:>12.4f}".format(
                state,
                m["n"],
                m["state_mae"],
                m["no_change_mae"]
            )
        )

    print(
        "-" * 70
    )

    print(
        "\nOVERALL"
    )

    print(
        "No-change MAE:",
        "{:.4f} mm".format(
            mae(
                all_actual,
                all_no_change
            )
        )
    )

    print(
        "State-aware MAE:",
        "{:.4f} mm".format(
            mae(
                all_actual,
                all_state
            )
        )
    )

    print(
        "\nSTATE PERSISTENCE:"
    )

    for state in [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]:

        print(
            "{}: {:.2f}%".format(
                state,
                metrics[
                    state
                ][
                    "state_persistence_percent"
                ]
            )
        )

    # --------------------------------------------------------
    # Transition matrix
    # --------------------------------------------------------

    counts = transition_matrix(
        results
    )

    percentages = transition_percentages(
        counts
    )

    print(
        "\nTRANSITION MATRIX (%)"
    )

    print(
        "{:<15} {:>12} {:>12} {:>12}".format(
            "CURRENT",
            "GROWTH",
            "STABLE",
            "SHRINKAGE"
        )
    )

    for state in [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]:

        print(
            "{:<15} {:>12.2f} {:>12.2f} {:>12.2f}".format(
                state,
                percentages[
                    state
                ]["GROWTH"],
                percentages[
                    state
                ]["STABLE"],
                percentages[
                    state
                ]["SHRINKAGE"]
            )
        )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    save_csv(
        RESULTS_CSV,
        results
    )

    # --------------------------------------------------------
    # Save transition matrix
    # --------------------------------------------------------

    transition_rows = []

    for current in [
        "GROWTH",
        "STABLE",
        "SHRINKAGE"
    ]:

        transition_rows.append({

            "current_state":
                current,

            "future_growth_count":
                counts[
                    current
                ]["GROWTH"],

            "future_stable_count":
                counts[
                    current
                ]["STABLE"],

            "future_shrinkage_count":
                counts[
                    current
                ]["SHRINKAGE"],

            "future_growth_percent":
                percentages[
                    current
                ]["GROWTH"],

            "future_stable_percent":
                percentages[
                    current
                ]["STABLE"],

            "future_shrinkage_percent":
                percentages[
                    current
                ]["SHRINKAGE"]
        })

    save_csv(
        TRANSITION_CSV,
        transition_rows
    )

    # --------------------------------------------------------
    # Patient summary
    # --------------------------------------------------------

    summary = patient_summary(
        results
    )

    save_csv(
        PATIENT_SUMMARY_CSV,
        summary
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = build_report(
        patients,
        train_patients,
        test_patients,
        state_rates,
        state_intervals,
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
    print("STEP 6 COMPLETE")
    print("=" * 70)

    print(
        "\nDetailed results:"
    )

    print(
        RESULTS_CSV
    )

    print(
        "\nTransition matrix:"
    )

    print(
        TRANSITION_CSV
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