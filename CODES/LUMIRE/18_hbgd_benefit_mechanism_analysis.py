# -*- coding: utf-8 -*-

"""
STEP 18 - HBGD BENEFIT MECHANISM ANALYSIS

Question
--------
Under what longitudinal / growth conditions does HBGD-derived
growth-rate information improve prediction over geometry alone?

Data source
-----------
Exact Step-12 test cohort recovered in Step 16.

Inputs
-------
1. step16_exact_step12_phenotype_predictions.csv
2. lumiere_longitudinal_trajectories.csv

Analysis
--------
For every held-out patient and phenotype:

    HBGD improvement =
        Geometry MAE - Geometry + HBGD Rate MAE

Investigated explanatory characteristics:

    - baseline/current diameter
    - baseline/current volume
    - observed HBGD growth rate
    - absolute growth rate
    - growth-rate variability
    - number of longitudinal observations
    - history duration
    - interval duration
    - absolute diameter change
    - percentage diameter change
    - phenotype

No model is retrained.
No new train/test split.
No prediction is changed.

Python 3.6 compatible.
"""

from __future__ import print_function

import os
import csv
import math

import numpy as np


# ============================================================
# PATHS
# ============================================================

TARGET_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"

PREDICTION_FILE = os.path.join(
    TARGET_DIR,
    "step16_exact_step12_phenotype_predictions.csv"
)

TRAJECTORY_FILE = os.path.join(
    TARGET_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

OUTPUT_PATIENT = os.path.join(
    TARGET_DIR,
    "step18_hbgd_benefit_patient_features.csv"
)

OUTPUT_CORRELATION = os.path.join(
    TARGET_DIR,
    "step18_hbgd_benefit_correlations.csv"
)

OUTPUT_PHENOTYPE = os.path.join(
    TARGET_DIR,
    "step18_hbgd_benefit_by_phenotype.csv"
)

OUTPUT_HISTORY = os.path.join(
    TARGET_DIR,
    "step18_hbgd_benefit_by_history.csv"
)

OUTPUT_REPORT = os.path.join(
    TARGET_DIR,
    "STEP18_HBGD_BENEFIT_MECHANISM_REPORT.md"
)


# ============================================================
# HELPERS
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


def write_csv(path, rows, fields):

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


def to_float(x):

    try:
        return float(x)
    except Exception:
        return float("nan")


def finite(x):

    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def safe_mean(values):

    values = [

        float(x)

        for x in values

        if finite(x)
    ]

    if not values:
        return float("nan")

    return float(
        np.mean(values)
    )


def safe_std(values):

    values = [

        float(x)

        for x in values

        if finite(x)
    ]

    if len(values) < 2:
        return float("nan")

    return float(
        np.std(
            values,
            ddof=1
        )
    )


# ============================================================
# SPEARMAN-LIKE RANK CORRELATION
#
# Avoid scipy dependency because the environment is
# Python 3.6 32-bit.
# ============================================================

def ranks(values):

    indexed = list(
        enumerate(values)
    )

    indexed.sort(
        key=lambda x: x[1]
    )

    output = [0.0] * len(values)

    i = 0

    while i < len(indexed):

        j = i + 1

        while (
            j < len(indexed)
            and
            indexed[j][1]
            == indexed[i][1]
        ):

            j += 1

        rank = (
            (i + 1)
            +
            j
        ) / 2.0

        for k in range(
            i,
            j
        ):

            output[
                indexed[k][0]
            ] = rank

        i = j

    return output


def pearson(x, y):

    if len(x) < 3:
        return float("nan")

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    if (
        np.std(x) == 0
        or
        np.std(y) == 0
    ):

        return float("nan")

    return float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )


def spearman(x, y):

    return pearson(
        ranks(x),
        ranks(y)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "STEP 18 - HBGD BENEFIT MECHANISM ANALYSIS"
    )

    print("=" * 70)

    # ========================================================
    # LOAD DATA
    # ========================================================

    predictions = read_csv(
        PREDICTION_FILE
    )

    trajectories = read_csv(
        TRAJECTORY_FILE
    )

    print(
        "\nPrediction rows:",
        len(predictions)
    )

    print(
        "Trajectory rows:",
        len(trajectories)
    )

    # ========================================================
    # EXACT TEST PATIENTS
    # ========================================================

    patients = sorted(
        list(
            set(
                row[
                    "patient_id"
                ]

                for row in predictions
            )
        )
    )

    print(
        "Test patients:",
        len(patients)
    )

    if len(patients) != 13:

        print(
            "\nWARNING: Expected 13 test patients."
        )

    # ========================================================
    # BUILD TRAJECTORY DATA
    # ========================================================

    patient_data = {}

    for row in trajectories:

        patient = str(
            row.get(
                "patient_id",
                ""
            )
        )

        if patient not in patients:
            continue

        week = to_float(
            row.get(
                "week",
                ""
            )
        )

        diameter = to_float(
            row.get(
                "equivalent_diameter_mm",
                ""
            )
        )

        volume = to_float(
            row.get(
                "volume_cm3",
                ""
            )
        )

        if (
            not finite(week)
            or
            not finite(diameter)
        ):
            continue

        if patient not in patient_data:

            patient_data[
                patient
            ] = []

        patient_data[
            patient
        ].append({

            "week":
                int(
                    round(
                        week
                    )
                ),

            "diameter":
                diameter,

            "volume":
                volume
        })

    for patient in patient_data:

        patient_data[
            patient
        ].sort(
            key=lambda x:
                x["week"]
        )

    # ========================================================
    # BUILD INTERVAL INFORMATION
    # ========================================================

    interval_lookup = {}

    patient_interval_features = {}

    for patient in patients:

        observations = patient_data.get(
            patient,
            []
        )

        rates = []

        for i in range(
            len(observations) - 1
        ):

            current = observations[i]

            future = observations[
                i + 1
            ]

            current_week = current[
                "week"
            ]

            future_week = future[
                "week"
            ]

            days = (
                future_week
                -
                current_week
            ) * 7.0

            d1 = current[
                "diameter"
            ]

            d2 = future[
                "diameter"
            ]

            if (
                d1 > 0
                and
                d2 > 0
                and
                days > 0
            ):

                r = (
                    math.log(
                        d2 / d1
                    )
                    /
                    days
                )

            else:

                r = float("nan")

            if finite(r):

                rates.append(
                    r
                )

            key = (
                patient,
                current_week,
                future_week
            )

            interval_lookup[
                key
            ] = {

                "rate":
                    r,

                "current_diameter":
                    d1,

                "future_diameter":
                    d2,

                "current_volume":
                    current[
                        "volume"
                    ],

                "future_volume":
                    future[
                        "volume"
                    ],

                "interval_weeks":
                    future_week
                    -
                    current_week
            }

        # ----------------------------------------------------
        # Patient-level temporal characteristics
        # ----------------------------------------------------

        if observations:

            baseline_diameter = (
                observations[0][
                    "diameter"
                ]
            )

            current_diameter = (
                observations[-1][
                    "diameter"
                ]
            )

            baseline_volume = (
                observations[0][
                    "volume"
                ]
            )

            current_volume = (
                observations[-1][
                    "volume"
                ]
            )

            history_weeks = (
                observations[-1][
                    "week"
                ]
                -
                observations[0][
                    "week"
                ]
            )

        else:

            baseline_diameter = float("nan")
            current_diameter = float("nan")
            baseline_volume = float("nan")
            current_volume = float("nan")
            history_weeks = float("nan")

        patient_interval_features[
            patient
        ] = {

            "n_timepoints":
                len(observations),

            "history_weeks":
                history_weeks,

            "baseline_diameter":
                baseline_diameter,

            "current_diameter":
                current_diameter,

            "baseline_volume":
                baseline_volume,

            "current_volume":
                current_volume,

            "mean_rate":
                safe_mean(rates),

            "rate_std":
                safe_std(rates),

            "abs_mean_rate":
                safe_mean(
                    [
                        abs(x)
                        for x in rates
                    ]
                )
        }

    # ========================================================
    # BUILD PATIENT/PHENOTYPE PERFORMANCE
    # ========================================================

    patient_results = []

    phenotypes = [
        "Growth",
        "Stable",
        "Shrinkage"
    ]

    geometry_model = (
        "M1_Geometry"
    )

    hbgd_model = (
        "M3_Geometry_HBGD_Rate"
    )

    for patient in patients:

        for state in phenotypes:

            rows = [

                row

                for row in predictions

                if (
                    row[
                        "patient_id"
                    ] == patient
                    and
                    row[
                        "phenotype"
                    ] == state
                )
            ]

            if not rows:
                continue

            geometry_errors = [

                abs(
                    to_float(
                        row[
                            "absolute_error_mm"
                        ]
                    )
                )

                for row in rows

                if (
                    row[
                        "model"
                    ]
                    ==
                    geometry_model
                    and
                    finite(
                        row[
                            "absolute_error_mm"
                        ]
                    )
                )
            ]

            hbgd_errors = [

                abs(
                    to_float(
                        row[
                            "absolute_error_mm"
                        ]
                    )
                )

                for row in rows

                if (
                    row[
                        "model"
                    ]
                    ==
                    hbgd_model
                    and
                    finite(
                        row[
                            "absolute_error_mm"
                        ]
                    )
                )
            ]

            if (
                not geometry_errors
                or
                not hbgd_errors
            ):
                continue

            geometry_mae = float(
                np.mean(
                    geometry_errors
                )
            )

            hbgd_mae = float(
                np.mean(
                    hbgd_errors
                )
            )

            improvement = (
                geometry_mae
                -
                hbgd_mae
            )

            # ------------------------------------------------
            # Interval-level characteristics for this
            # phenotype
            # ------------------------------------------------

            rates = []

            interval_lengths = []

            diameter_changes = []

            percent_changes = []

            current_diameters = []

            current_volumes = []

            for row in rows:

                if row[
                    "model"
                ] != geometry_model:

                    continue

                key = (

                    patient,

                    int(
                        round(
                            to_float(
                                row[
                                    "current_week"
                                ]
                            )
                        )
                    ),

                    int(
                        round(
                            to_float(
                                row[
                                    "future_week"
                                ]
                            )
                        )
                    )
                )

                info = interval_lookup.get(
                    key
                )

                if info is None:
                    continue

                if finite(
                    info[
                        "rate"
                    ]
                ):

                    rates.append(
                        info[
                            "rate"
                        ]
                    )

                if finite(
                    info[
                        "interval_weeks"
                    ]
                ):

                    interval_lengths.append(
                        info[
                            "interval_weeks"
                        ]
                    )

                d1 = info[
                    "current_diameter"
                ]

                d2 = info[
                    "future_diameter"
                ]

                if (
                    finite(d1)
                    and
                    finite(d2)
                ):

                    diameter_changes.append(
                        d2 - d1
                    )

                    if d1 != 0:

                        percent_changes.append(
                            (
                                d2 - d1
                            )
                            /
                            d1
                            *
                            100.0
                        )

                    current_diameters.append(
                        d1
                    )

                if finite(
                    info[
                        "current_volume"
                    ]
                ):

                    current_volumes.append(
                        info[
                            "current_volume"
                        ]
                    )

            temporal = patient_interval_features.get(
                patient,
                {}
            )

            patient_results.append({

                "patient_id":
                    patient,

                "phenotype":
                    state,

                "n_prediction_cases":
                    len(
                        geometry_errors
                    ),

                "geometry_mae_mm":
                    geometry_mae,

                "hbgd_rate_mae_mm":
                    hbgd_mae,

                "hbgd_improvement_mm":
                    improvement,

                "hbgd_relative_improvement_percent":
                    (
                        improvement
                        /
                        geometry_mae
                        *
                        100.0
                    )
                    if geometry_mae > 0
                    else float("nan"),

                "mean_interval_rate_per_day":
                    safe_mean(rates),

                "mean_absolute_interval_rate_per_day":
                    safe_mean(
                        [
                            abs(x)
                            for x in rates
                        ]
                    ),

                "interval_rate_std_per_day":
                    safe_std(rates),

                "mean_interval_weeks":
                    safe_mean(
                        interval_lengths
                    ),

                "mean_diameter_change_mm":
                    safe_mean(
                        diameter_changes
                    ),

                "mean_absolute_diameter_change_mm":
                    safe_mean(
                        [
                            abs(x)
                            for x in diameter_changes
                        ]
                    ),

                "mean_percent_diameter_change":
                    safe_mean(
                        percent_changes
                    ),

                "mean_current_diameter_mm":
                    safe_mean(
                        current_diameters
                    ),

                "mean_current_volume_cm3":
                    safe_mean(
                        current_volumes
                    ),

                "n_timepoints":
                    temporal.get(
                        "n_timepoints",
                        float("nan")
                    ),

                "history_weeks":
                    temporal.get(
                        "history_weeks",
                        float("nan")
                    ),

                "patient_mean_rate_per_day":
                    temporal.get(
                        "mean_rate",
                        float("nan")
                    ),

                "patient_rate_variability_per_day":
                    temporal.get(
                        "rate_std",
                        float("nan")
                    )
            })

    # ========================================================
    # SAVE PATIENT FEATURES
    # ========================================================

    fields = list(
        patient_results[0].keys()
    )

    write_csv(

        OUTPUT_PATIENT,

        patient_results,

        fields
    )

    # ========================================================
    # CORRELATION ANALYSIS
    # ========================================================

    features = [

        "mean_interval_rate_per_day",

        "mean_absolute_interval_rate_per_day",

        "interval_rate_std_per_day",

        "mean_interval_weeks",

        "mean_diameter_change_mm",

        "mean_absolute_diameter_change_mm",

        "mean_percent_diameter_change",

        "mean_current_diameter_mm",

        "mean_current_volume_cm3",

        "n_timepoints",

        "history_weeks",

        "patient_mean_rate_per_day",

        "patient_rate_variability_per_day"
    ]

    correlations = []

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HBGD BENEFIT CORRELATIONS"
    )

    print(
        "=" * 70
    )

    for feature in features:

        x = []

        y = []

        for row in patient_results:

            xv = to_float(
                row[
                    feature
                ]
            )

            yv = to_float(
                row[
                    "hbgd_improvement_mm"
                ]
            )

            if (
                finite(xv)
                and
                finite(yv)
            ):

                x.append(xv)
                y.append(yv)

        if len(x) < 3:

            continue

        pearson_r = pearson(
            x,
            y
        )

        spearman_r = spearman(
            x,
            y
        )

        correlations.append({

            "feature":
                feature,

            "n":
                len(x),

            "pearson_r":
                pearson_r,

            "spearman_r":
                spearman_r
        })

        print(
            "{:<45} N={} Pearson={:+.4f} Spearman={:+.4f}".format(
                feature,
                len(x),
                pearson_r,
                spearman_r
            )
        )

    write_csv(

        OUTPUT_CORRELATION,

        correlations,

        [
            "feature",
            "n",
            "pearson_r",
            "spearman_r"
        ]
    )

    # ========================================================
    # PHENOTYPE SUMMARY
    # ========================================================

    phenotype_summary = []

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PHENOTYPE BENEFIT SUMMARY"
    )

    print(
        "=" * 70
    )

    for state in phenotypes:

        state_rows = [

            row

            for row in patient_results

            if row[
                "phenotype"
            ] == state
        ]

        improvements = [

            to_float(
                row[
                    "hbgd_improvement_mm"
                ]
            )

            for row in state_rows

            if finite(
                row[
                    "hbgd_improvement_mm"
                ]
            )
        ]

        if not improvements:
            continue

        improved = sum(
            1
            for x in improvements
            if x > 0
        )

        worsened = sum(
            1
            for x in improvements
            if x < 0
        )

        mean_improvement = float(
            np.mean(
                improvements
            )
        )

        median_improvement = float(
            np.median(
                improvements
            )
        )

        phenotype_summary.append({

            "phenotype":
                state,

            "n_patient_phenotype_groups":
                len(
                    improvements
                ),

            "improved":
                improved,

            "worsened":
                worsened,

            "mean_improvement_mm":
                mean_improvement,

            "median_improvement_mm":
                median_improvement,

            "mean_relative_improvement_percent":
                safe_mean(
                    [
                        to_float(
                            row[
                                "hbgd_relative_improvement_percent"
                            ]
                        )

                        for row in state_rows
                    ]
                )
        })

        print(
            "{}: N={} Improved={} Worsened={} Mean={:+.4f} mm Median={:+.4f} mm".format(
                state,
                len(improvements),
                improved,
                worsened,
                mean_improvement,
                median_improvement
            )
        )

    write_csv(

        OUTPUT_PHENOTYPE,

        phenotype_summary,

        [
            "phenotype",
            "n_patient_phenotype_groups",
            "improved",
            "worsened",
            "mean_improvement_mm",
            "median_improvement_mm",
            "mean_relative_improvement_percent"
        ]
    )

    # ========================================================
    # HISTORY STRATIFICATION
    # ========================================================

    history_values = [

        to_float(
            row[
                "n_timepoints"
            ]
        )

        for row in patient_results

        if finite(
            row[
                "n_timepoints"
            ]
        )
    ]

    median_history = (
        float(
            np.median(
                history_values
            )
        )
        if history_values
        else float("nan")
    )

    short_rows = []
    long_rows = []

    for row in patient_results:

        value = to_float(
            row[
                "n_timepoints"
            ]
        )

        if not finite(value):
            continue

        if value <= median_history:

            short_rows.append(
                row
            )

        else:

            long_rows.append(
                row
            )

    history_summary = []

    for label, group in [

        (
            "Shorter_History",
            short_rows
        ),

        (
            "Longer_History",
            long_rows
        )

    ]:

        improvements = [

            to_float(
                row[
                    "hbgd_improvement_mm"
                ]
            )

            for row in group

            if finite(
                row[
                    "hbgd_improvement_mm"
                ]
            )
        ]

        if not improvements:
            continue

        history_summary.append({

            "history_group":
                label,

            "n":
                len(improvements),

            "median_timepoints":
                median_history,

            "mean_improvement_mm":
                float(
                    np.mean(
                        improvements
                    )
                ),

            "median_improvement_mm":
                float(
                    np.median(
                        improvements
                    )
                ),

            "improved":
                sum(
                    1
                    for x in improvements
                    if x > 0
                ),

            "worsened":
                sum(
                    1
                    for x in improvements
                    if x < 0
                )
        })

    write_csv(

        OUTPUT_HISTORY,

        history_summary,

        [
            "history_group",
            "n",
            "median_timepoints",
            "mean_improvement_mm",
            "median_improvement_mm",
            "improved",
            "worsened"
        ]
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "HISTORY ANALYSIS"
    )

    print(
        "=" * 70
    )

    print(
        "Median number of timepoints:",
        median_history
    )

    for row in history_summary:

        print(
            "{}: N={} Mean improvement={:+.4f} mm Improved={} Worsened={}".format(
                row["history_group"],
                row["n"],
                row["mean_improvement_mm"],
                row["improved"],
                row["worsened"]
            )
        )

    # ========================================================
    # FIND STRONGEST ASSOCIATIONS
    # ========================================================

    valid_correlations = [

        row

        for row in correlations

        if finite(
            row["spearman_r"]
        )
    ]

    valid_correlations.sort(

        key=lambda x:
            abs(
                x["spearman_r"]
            ),

        reverse=True
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    report.append(
        "# STEP 18 - HBGD BENEFIT MECHANISM ANALYSIS"
    )

    report.append("")

    report.append(
        "## Objective"
    )

    report.append("")

    report.append(
        "Determine whether HBGD-derived growth-rate information provides greater predictive benefit under particular longitudinal or tumor-growth conditions."
    )

    report.append("")

    report.append(
        "## Experimental Integrity"
    )

    report.append("")

    report.append(
        "The analysis uses the exact Step-12 held-out predictions recovered in Step 16."
    )

    report.append("")

    report.append(
        "No model retraining was performed."
    )

    report.append("")

    report.append(
        "No new train/test split was created."
    )

    report.append("")

    report.append(
        "Phenotype and longitudinal characteristics were analyzed post-hoc and were not supplied as predictive inputs."
    )

    report.append("")

    report.append(
        "## Phenotype-Level Benefit"
    )

    report.append("")

    report.append(
        "| Phenotype | N | Improved | Worsened | Mean Improvement (mm) | Median Improvement (mm) |"
    )

    report.append(
        "|---|---:|---:|---:|---:|---:|"
    )

    for row in phenotype_summary:

        report.append(
            "| {} | {} | {} | {} | {:+.4f} | {:+.4f} |".format(
                row["phenotype"],
                row["n_patient_phenotype_groups"],
                row["improved"],
                row["worsened"],
                row["mean_improvement_mm"],
                row["median_improvement_mm"]
            )
        )

    report.append("")

    report.append(
        "## Feature Associations"
    )

    report.append("")

    report.append(
        "| Feature | N | Pearson r | Spearman r |"
    )

    report.append(
        "|---|---:|---:|---:|"
    )

    for row in correlations:

        report.append(
            "| {} | {} | {:+.4f} | {:+.4f} |".format(
                row["feature"],
                row["n"],
                row["pearson_r"],
                row["spearman_r"]
            )
        )

    report.append("")

    report.append(
        "## Strongest Descriptive Associations"
    )

    report.append("")

    if valid_correlations:

        for row in valid_correlations[:5]:

            report.append(
                "- **{}**: Spearman r = {:+.4f}".format(
                    row["feature"],
                    row["spearman_r"]
                )
            )

    else:

        report.append(
            "No valid correlations were available."
        )

    report.append("")

    report.append(
        "## History Analysis"
    )

    report.append("")

    report.append(
        "| Group | N | Mean Improvement (mm) | Improved | Worsened |"
    )

    report.append(
        "|---|---:|---:|---:|---:|"
    )

    for row in history_summary:

        report.append(
            "| {} | {} | {:+.4f} | {} | {} |".format(
                row["history_group"],
                row["n"],
                row["mean_improvement_mm"],
                row["improved"],
                row["worsened"]
            )
        )

    report.append("")

    report.append(
        "## Interpretation"
    )

    report.append("")

    report.append(
        "Positive HBGD improvement means that adding HBGD-derived growth-rate information reduced prediction MAE relative to geometry alone."
    )

    report.append("")

    report.append(
        "Correlation results are exploratory and descriptive. They do not establish causality or statistical significance."
    )

    report.append("")

    report.append(
        "The test cohort contains only 13 patients, and some patients contribute to multiple phenotype groups. Therefore, feature associations should be considered hypothesis-generating."
    )

    report.append("")

    report.append(
        "## Conclusion"
    )

    report.append("")

    report.append(
        "Step 18 identifies longitudinal and growth characteristics associated with the incremental predictive benefit of HBGD-derived growth-rate information, while preserving the original held-out predictions."
    )

    report.append("")

    with open(
        OUTPUT_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report)
        )

    # ========================================================
    # DONE
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 18 COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nOutputs:"
    )

    print(
        OUTPUT_PATIENT
    )

    print(
        OUTPUT_CORRELATION
    )

    print(
        OUTPUT_PHENOTYPE
    )

    print(
        OUTPUT_HISTORY
    )

    print(
        OUTPUT_REPORT
    )


if __name__ == "__main__":

    main()