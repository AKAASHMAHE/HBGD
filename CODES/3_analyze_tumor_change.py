# 3_analyze_tumor_change.py

import os
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

RECIST_FILE = os.path.join(
    DATA_DIR,
    "recist_measurements.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def parse_date(value):
    return pd.to_datetime(
        str(int(value)),
        format="%Y%m%d",
        errors="coerce"
    )


# ============================================================
# LOAD DATA
# ============================================================

header("LOADING RECIST DATA")

df = pd.read_csv(RECIST_FILE)

df["date"] = df["study_date"].apply(parse_date)

df = df.sort_values(
    [
        "patient_id",
        "date",
        "lesion_label_alias"
    ]
)

print(f"Total RECIST records: {len(df)}")
print(f"Patients: {df['patient_id'].nunique()}")
print(f"Study dates: {df['date'].nunique()}")


# ============================================================
# CHECK FOR MULTIPLE RECORDS FOR SAME
# PATIENT + DATE + LESION
# ============================================================

header("CHECK 1 — MULTIPLE MEASUREMENTS WITHIN SAME TIME POINT")

duplicate_groups = (
    df.groupby(
        [
            "patient_id",
            "date",
            "lesion_label_alias"
        ]
    )
    .size()
    .reset_index(name="count")
)

duplicates = duplicate_groups[
    duplicate_groups["count"] > 1
]

print(
    f"Patient-date-lesion groups: "
    f"{len(duplicate_groups)}"
)

print(
    f"Groups with multiple measurements: "
    f"{len(duplicates)}"
)

if len(duplicates) > 0:

    print("\nThese groups need investigation:")

    print(
        duplicates.to_string(index=False)
    )

    for _, row in duplicates.iterrows():

        p = df[
            (df["patient_id"] == row["patient_id"]) &
            (df["date"] == row["date"]) &
            (
                df["lesion_label_alias"] ==
                row["lesion_label_alias"]
            )
        ]

        print("\n" + "-" * 80)

        print(
            f"Patient {row['patient_id']} | "
            f"Date {row['date'].date()} | "
            f"Lesion {row['lesion_label_alias']}"
        )

        print(
            p[
                [
                    "study_uuid",
                    "uuid",
                    "region",
                    "lesion_label_value",
                    "lesion_label_alias",
                    "recist_measurement_mm",
                    "study_order"
                ]
            ].to_string(index=False)
        )


# ============================================================
# CALCULATE TARGET-LESION SUM
# ============================================================

header("CHECK 2 — RECIST SUM OF LESION DIAMETERS")

"""
For a patient at a given study:

RECIST Sum =
    sum of RECIST lesion measurements

This gives us a simple longitudinal tumor-burden measure.

Example:

Baseline:
    A = 40 mm
    B = 20 mm

Sum = 60 mm

Follow-up:
    A = 35 mm
    B = 18 mm

Sum = 53 mm

Percentage change:

    ((53 - 60) / 60) * 100
        = -11.67%

Negative = decrease
Positive = increase
"""


# IMPORTANT:
# For this preliminary analysis, aggregate all measurements
# belonging to the same patient + study date.

study_summary = (
    df.groupby(
        [
            "patient_id",
            "date",
            "study_uuid",
            "study_order"
        ],
        as_index=False
    )
    .agg(
        number_of_lesions=(
            "recist_measurement_mm",
            "count"
        ),
        total_diameter_mm=(
            "recist_measurement_mm",
            "sum"
        ),
        minimum_diameter_mm=(
            "recist_measurement_mm",
            "min"
        ),
        maximum_diameter_mm=(
            "recist_measurement_mm",
            "max"
        )
    )
)

study_summary = study_summary.sort_values(
    [
        "patient_id",
        "date"
    ]
)


print(
    study_summary.to_string(index=False)
)


# ============================================================
# CALCULATE LONGITUDINAL PERCENTAGE CHANGE
# ============================================================

header("CHECK 3 — LONGITUDINAL TUMOR CHANGE")

study_summary["baseline_sum_mm"] = (
    study_summary
    .groupby("patient_id")["total_diameter_mm"]
    .transform("first")
)

study_summary["percentage_change_from_baseline"] = (
    (
        study_summary["total_diameter_mm"]
        -
        study_summary["baseline_sum_mm"]
    )
    /
    study_summary["baseline_sum_mm"]
) * 100


study_summary["absolute_change_mm"] = (
    study_summary["total_diameter_mm"]
    -
    study_summary["baseline_sum_mm"]
)


print(
    study_summary[
        [
            "patient_id",
            "date",
            "study_order",
            "number_of_lesions",
            "total_diameter_mm",
            "baseline_sum_mm",
            "absolute_change_mm",
            "percentage_change_from_baseline"
        ]
    ].to_string(index=False)
)


# ============================================================
# PATIENT-LEVEL TRAJECTORIES
# ============================================================

header("CHECK 4 — PATIENT TRAJECTORIES")

patient_trajectory_results = []

for patient_id in sorted(
    study_summary["patient_id"].unique()
):

    p = study_summary[
        study_summary["patient_id"] == patient_id
    ].sort_values("date")

    if len(p) < 2:
        continue

    baseline = p.iloc[0]

    final = p.iloc[-1]

    change_mm = (
        final["total_diameter_mm"]
        -
        baseline["total_diameter_mm"]
    )

    change_percent = (
        change_mm /
        baseline["total_diameter_mm"]
    ) * 100

    days = (
        final["date"] -
        baseline["date"]
    ).days

    patient_trajectory_results.append({

        "patient_id": patient_id,

        "baseline_date":
            baseline["date"].date(),

        "final_date":
            final["date"].date(),

        "followup_days":
            days,

        "num_timepoints":
            len(p),

        "baseline_sum_mm":
            baseline["total_diameter_mm"],

        "final_sum_mm":
            final["total_diameter_mm"],

        "change_mm":
            change_mm,

        "change_percent":
            change_percent
    })


trajectory = pd.DataFrame(
    patient_trajectory_results
)


if len(trajectory) > 0:

    print(
        trajectory.to_string(index=False)
    )


# ============================================================
# SIMPLE RESPONSE DIRECTION
# ============================================================

header("CHECK 5 — GROWTH VS SHRINKAGE")

if len(trajectory) > 0:

    trajectory["direction"] = "STABLE"

    trajectory.loc[
        trajectory["change_percent"] > 0,
        "direction"
    ] = "INCREASED"

    trajectory.loc[
        trajectory["change_percent"] < 0,
        "direction"
    ] = "DECREASED"

    print(
        trajectory[
            [
                "patient_id",
                "baseline_sum_mm",
                "final_sum_mm",
                "change_mm",
                "change_percent",
                "direction"
            ]
        ].to_string(index=False)
    )

    print("\nDirection counts:")

    print(
        trajectory["direction"]
        .value_counts()
        .to_string()
    )


# ============================================================
# CHECK FOR EXTREME CHANGES
# ============================================================

header("CHECK 6 — EXTREME LONGITUDINAL CHANGES")

if len(trajectory) > 0:

    extreme = trajectory[
        trajectory["change_percent"].abs() >= 50
    ]

    print(
        f"Patients with >=50% absolute change: "
        f"{len(extreme)}"
    )

    if len(extreme) > 0:

        print(
            extreme[
                [
                    "patient_id",
                    "baseline_sum_mm",
                    "final_sum_mm",
                    "change_percent",
                    "followup_days"
                ]
            ].to_string(index=False)
        )


# ============================================================
# STUDY-TO-STUDY CHANGES
# ============================================================

header("CHECK 7 — STUDY-TO-STUDY CHANGE")

interval_results = []

for patient_id in sorted(
    study_summary["patient_id"].unique()
):

    p = study_summary[
        study_summary["patient_id"] == patient_id
    ].sort_values("date")

    if len(p) < 2:
        continue

    for i in range(1, len(p)):

        previous = p.iloc[i - 1]
        current = p.iloc[i]

        previous_sum = previous["total_diameter_mm"]
        current_sum = current["total_diameter_mm"]

        change_mm = current_sum - previous_sum

        if previous_sum != 0:
            change_percent = (
                change_mm /
                previous_sum
            ) * 100
        else:
            change_percent = None

        days = (
            current["date"] -
            previous["date"]
        ).days

        interval_results.append({

            "patient_id":
                patient_id,

            "previous_date":
                previous["date"].date(),

            "current_date":
                current["date"].date(),

            "interval_days":
                days,

            "previous_sum_mm":
                previous_sum,

            "current_sum_mm":
                current_sum,

            "change_mm":
                change_mm,

            "change_percent":
                change_percent
        })


interval_df = pd.DataFrame(
    interval_results
)

if len(interval_df) > 0:

    print(
        interval_df.to_string(index=False)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

summary_file = os.path.join(
    DATA_DIR,
    "3_tumor_change_summary.csv"
)

trajectory_file = os.path.join(
    DATA_DIR,
    "3_patient_trajectories.csv"
)

study_summary.to_csv(
    summary_file,
    index=False
)

if len(trajectory) > 0:
    trajectory.to_csv(
        trajectory_file,
        index=False
    )


# ============================================================
# FINAL ASSESSMENT
# ============================================================

header("STEP 3 — PRELIMINARY ASSESSMENT")

print("""
We are testing whether the dataset provides a usable numerical
longitudinal tumor-burden signal.

The important evidence is:

    CT scan 1
        ↓
    RECIST measurement
        ↓
    time interval
        ↓
    CT scan 2
        ↓
    RECIST measurement
        ↓
    measurable change

This is NOT yet the HBGD model.

It is only validation that the input signal needed by the model
actually exists in the dataset.

IMPORTANT:
Do not interpret the simple percentage change as official RECIST
Progressive Disease / Partial Response / Stable Disease classification.

That requires proper RECIST 1.1 rules and confirmation of the
underlying lesion correspondence.

Send me the COMPLETE terminal output from:

    python 3_analyze_tumor_change.py

Do not modify it.
""")

print("=" * 100)
print("STEP 3 COMPLETE")
print("=" * 100)