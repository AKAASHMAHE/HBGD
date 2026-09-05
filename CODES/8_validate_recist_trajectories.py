# 8_validate_recist_trajectories.py

import os
import sys
import pandas as pd
import numpy as np


# ============================================================
# TERMINAL ENCODING
# ============================================================

try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )
except Exception:
    pass


# ============================================================
# PATHS
# ============================================================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "DATA"
)

RECIST_FILE = os.path.join(
    DATA_DIR,
    "recist_measurements.csv"
)

GEOMETRY_FILE = os.path.join(
    DATA_DIR,
    "7_recist_geometry.csv"
)


# ============================================================
# HEADER
# ============================================================

def header(text):

    print()
    print("=" * 90)
    print(text)
    print("=" * 90)


# ============================================================
# LOAD RECIST
# ============================================================

header(
    "STEP 8 - LOADING RECIST DATA"
)

if not os.path.exists(
    RECIST_FILE
):

    print(
        "ERROR: recist_measurements.csv not found"
    )

    print(
        RECIST_FILE
    )

    sys.exit(1)


recist = pd.read_csv(
    RECIST_FILE
)


print(
    "RECIST records:",
    len(recist)
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "patient_id",
    "study_date",
    "filename",
    "lesion_label_value",
    "lesion_label_alias",
    "recist_measurement_mm",
    "study_order"
]


header(
    "CHECKING REQUIRED COLUMNS"
)


missing = []

for column in required_columns:

    if column not in recist.columns:

        missing.append(
            column
        )


if len(missing) > 0:

    print(
        "ERROR: Missing columns:"
    )

    for column in missing:

        print(
            "  ",
            column
        )

    sys.exit(1)


print(
    "All required columns are present."
)


# ============================================================
# NORMALIZE DATA
# ============================================================

header(
    "NORMALIZING RECIST DATA"
)


recist["patient_id"] = pd.to_numeric(
    recist["patient_id"],
    errors="coerce"
)

recist["study_date"] = (
    recist["study_date"]
    .astype(str)
)

recist["date"] = pd.to_datetime(
    recist["study_date"],
    format="%Y%m%d",
    errors="coerce"
)

recist["recist_measurement_mm"] = (
    pd.to_numeric(
        recist[
            "recist_measurement_mm"
        ],
        errors="coerce"
    )
)

recist["lesion_label_value"] = (
    pd.to_numeric(
        recist[
            "lesion_label_value"
        ],
        errors="coerce"
    )
)

recist["lesion_label_alias"] = (
    recist[
        "lesion_label_alias"
    ]
    .astype(str)
    .str.strip()
)


print(
    "Valid dates:",
    recist["date"].notna().sum()
)

print(
    "Valid RECIST measurements:",
    recist[
        "recist_measurement_mm"
    ].notna().sum()
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

header(
    "CHECKING FOR DUPLICATE TARGET-LESION RECORDS"
)


duplicate_columns = [
    "patient_id",
    "study_date",
    "lesion_label_alias"
]


duplicates = recist[
    recist.duplicated(
        subset=duplicate_columns,
        keep=False
    )
].copy()


print(
    "Duplicate patient/date/alias records:",
    len(duplicates)
)


if len(duplicates) > 0:

    print()

    print(
        duplicates[
            duplicate_columns
            +
            [
                "lesion_label_value",
                "recist_measurement_mm",
                "study_order"
            ]
        ]
        .sort_values(
            duplicate_columns
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# CREATE TRAJECTORIES
# ============================================================

header(
    "BUILDING RECIST LESION TRAJECTORIES"
)


trajectory_groups = (
    recist
    .groupby(
        [
            "patient_id",
            "lesion_label_alias"
        ],
        dropna=False
    )
)


trajectory_records = []


for (
    patient_id,
    alias
), group in trajectory_groups:

    group = group.sort_values(
        "date"
    ).copy()

    dates = (
        group[
            "date"
        ]
        .dropna()
        .unique()
    )

    measurements = (
        group[
            "recist_measurement_mm"
        ]
        .dropna()
        .astype(float)
        .tolist()
    )

    trajectory_records.append({

        "patient_id":
            patient_id,

        "lesion_alias":
            alias,

        "number_of_records":
            len(group),

        "number_of_dates":
            len(dates),

        "longitudinal":
            len(dates) >= 2,

        "first_date":
            dates[0]
            if len(dates) > 0
            else pd.NaT,

        "last_date":
            dates[-1]
            if len(dates) > 0
            else pd.NaT,

        "first_measurement_mm":
            measurements[0]
            if len(measurements) > 0
            else np.nan,

        "last_measurement_mm":
            measurements[-1]
            if len(measurements) > 0
            else np.nan,

        "number_of_measurements":
            len(measurements)
    })


trajectory_df = pd.DataFrame(
    trajectory_records
)


print(
    "Unique patient-lesion trajectories:",
    len(trajectory_df)
)

print(
    "Longitudinal trajectories:",
    int(
        trajectory_df[
            "longitudinal"
        ].sum()
    )
)


# ============================================================
# DETAILED TRAJECTORY ANALYSIS
# ============================================================

header(
    "DETAILED LONGITUDINAL TRAJECTORIES"
)


longitudinal_df = trajectory_df[
    trajectory_df[
        "longitudinal"
    ]
].copy()


detailed_records = []


for (
    patient_id,
    alias
), group in trajectory_groups:

    group = group.sort_values(
        "date"
    ).copy()

    if (
        group[
            "date"
        ].nunique()
        <
        2
    ):

        continue


    rows = group[
        [
            "patient_id",
            "lesion_label_alias",
            "study_date",
            "date",
            "lesion_label_value",
            "recist_measurement_mm",
            "study_order",
            "filename"
        ]
    ].to_dict(
        "records"
    )


    print()

    print(
        "Patient:",
        int(patient_id)
        if pd.notna(patient_id)
        else patient_id
    )

    print(
        "Lesion alias:",
        alias
    )

    print(
        "Number of observations:",
        len(rows)
    )


    for i, row in enumerate(
        rows,
        start=1
    ):

        print(
            "   ",
            i,
            "|",
            row["study_date"],
            "|",
            row["study_order"],
            "| diameter:",
            row[
                "recist_measurement_mm"
            ],
            "mm",
            "| mask label:",
            row[
                "lesion_label_value"
            ]
        )


    # --------------------------------------------------------
    # PAIRWISE CHANGES
    # --------------------------------------------------------

    for i in range(
        1,
        len(rows)
    ):

        previous = rows[
            i - 1
        ]

        current = rows[
            i
        ]


        previous_date = (
            previous["date"]
        )

        current_date = (
            current["date"]
        )


        interval_days = (
            current_date -
            previous_date
        ).days


        previous_mm = float(
            previous[
                "recist_measurement_mm"
            ]
        )

        current_mm = float(
            current[
                "recist_measurement_mm"
            ]
        )


        diameter_change = (
            current_mm -
            previous_mm
        )


        if previous_mm != 0:

            percent_change = (
                diameter_change /
                previous_mm
            ) * 100.0

        else:

            percent_change = np.nan


        if interval_days > 0:

            rate_mm_per_day = (
                diameter_change /
                interval_days
            )

        else:

            rate_mm_per_day = np.nan


        detailed_records.append({

            "patient_id":
                patient_id,

            "lesion_alias":
                alias,

            "previous_date":
                previous_date.strftime(
                    "%Y%m%d"
                ),

            "current_date":
                current_date.strftime(
                    "%Y%m%d"
                ),

            "interval_days":
                interval_days,

            "previous_mask_label":
                previous[
                    "lesion_label_value"
                ],

            "current_mask_label":
                current[
                    "lesion_label_value"
                ],

            "previous_diameter_mm":
                previous_mm,

            "current_diameter_mm":
                current_mm,

            "diameter_change_mm":
                diameter_change,

            "percent_change":
                percent_change,

            "rate_mm_per_day":
                rate_mm_per_day,

            "previous_study_order":
                previous[
                    "study_order"
                ],

            "current_study_order":
                current[
                    "study_order"
                ]
        })


pairwise_df = pd.DataFrame(
    detailed_records
)


# ============================================================
# PAIRWISE SUMMARY
# ============================================================

header(
    "PAIRWISE RECIST CHANGE SUMMARY"
)


print(
    "Longitudinal pairwise observations:",
    len(pairwise_df)
)


if len(pairwise_df) > 0:

    print()

    print(
        "Mean diameter change:",
        round(
            pairwise_df[
                "diameter_change_mm"
            ].mean(),
            3
        ),
        "mm"
    )

    print(
        "Median diameter change:",
        round(
            pairwise_df[
                "diameter_change_mm"
            ].median(),
            3
        ),
        "mm"
    )

    print(
        "Mean percentage change:",
        round(
            pairwise_df[
                "percent_change"
            ].mean(),
            3
        ),
        "%"
    )

    print(
        "Median percentage change:",
        round(
            pairwise_df[
                "percent_change"
            ].median(),
            3
        ),
        "%"
    )

    print(
        "Mean interval:",
        round(
            pairwise_df[
                "interval_days"
            ].mean(),
            2
        ),
        "days"
    )

    print(
        "Median interval:",
        round(
            pairwise_df[
                "interval_days"
            ].median(),
            2
        ),
        "days"
    )


# ============================================================
# GROWTH / SHRINKAGE / STABLE COUNTS
# ============================================================

header(
    "GROWTH DIRECTION ANALYSIS"
)


if len(pairwise_df) > 0:

    growth_count = int(
        (
            pairwise_df[
                "diameter_change_mm"
            ]
            > 0
        ).sum()
    )

    shrinkage_count = int(
        (
            pairwise_df[
                "diameter_change_mm"
            ]
            < 0
        ).sum()
    )

    stable_count = int(
        (
            pairwise_df[
                "diameter_change_mm"
            ]
            == 0
        ).sum()
    )

    total_pairs = len(
        pairwise_df
    )


    print(
        "Growth:",
        growth_count,
        "/",
        total_pairs
    )

    print(
        "Shrinkage:",
        shrinkage_count,
        "/",
        total_pairs
    )

    print(
        "No diameter change:",
        stable_count,
        "/",
        total_pairs
    )


# ============================================================
# TIME INTERVAL QUALITY
# ============================================================

header(
    "TIME INTERVAL QUALITY"
)


if len(pairwise_df) > 0:

    intervals = pairwise_df[
        "interval_days"
    ]


    print(
        "Minimum interval:",
        intervals.min(),
        "days"
    )

    print(
        "Maximum interval:",
        intervals.max(),
        "days"
    )

    print(
        "Median interval:",
        intervals.median(),
        "days"
    )


    print()

    print(
        "Intervals:"
    )

    print(
        intervals
        .value_counts()
        .sort_index()
        .to_string()
    )


# ============================================================
# MASK LABEL STABILITY
# ============================================================

header(
    "MASK LABEL STABILITY ACROSS TIME"
)


if len(pairwise_df) > 0:

    same_label = int(
        (
            pairwise_df[
                "previous_mask_label"
            ]
            ==
            pairwise_df[
                "current_mask_label"
            ]
        ).sum()
    )

    different_label = int(
        (
            pairwise_df[
                "previous_mask_label"
            ]
            !=
            pairwise_df[
                "current_mask_label"
            ]
        ).sum()
    )

    total_pairs = len(
        pairwise_df
    )


    print(
        "Same numerical mask label:",
        same_label,
        "/",
        total_pairs
    )

    print(
        "Different numerical mask label:",
        different_label,
        "/",
        total_pairs
    )


# ============================================================
# STUDY ORDER VALIDATION
# ============================================================

header(
    "STUDY ORDER VALIDATION"
)


if len(pairwise_df) > 0:

    valid_order = 0

    invalid_order = 0


    order_rank = {

        "baseline": 0,

        "follow-up-1": 1,

        "follow-up-2": 2,

        "follow-up-3": 3
    }


    for _, row in (
        pairwise_df.iterrows()
    ):

        previous_order = (
            str(
                row[
                    "previous_study_order"
                ]
            )
        )

        current_order = (
            str(
                row[
                    "current_study_order"
                ]
            )
        )


        previous_rank = (
            order_rank.get(
                previous_order,
                -1
            )
        )

        current_rank = (
            order_rank.get(
                current_order,
                -1
            )
        )


        if (
            current_rank
            >
            previous_rank
        ):

            valid_order += 1

        else:

            invalid_order += 1


    print(
        "Chronologically valid RECIST transitions:",
        valid_order
    )

    print(
        "Potentially invalid transitions:",
        invalid_order
    )


# ============================================================
# OPTIONAL GEOMETRY INFORMATION
# ============================================================

header(
    "CHECKING RECIST + MASK GEOMETRY"
)


if os.path.exists(
    GEOMETRY_FILE
):

    geometry_df = pd.read_csv(
        GEOMETRY_FILE
    )

    print(
        "Geometry records loaded:",
        len(geometry_df)
    )


    if (
        "volume_cm3"
        in geometry_df.columns
    ):

        linked = int(
            geometry_df[
                "volume_cm3"
            ].notna()
            .sum()
        )

        print(
            "RECIST records with mask volume:",
            linked
        )

        if linked > 0:

            print(
                "Median RECIST-linked mask volume:",
                round(
                    geometry_df[
                        "volume_cm3"
                    ]
                    .dropna()
                    .median(),
                    3
                ),
                "cm3"
            )

else:

    geometry_df = pd.DataFrame()

    print(
        "7_recist_geometry.csv not found."
    )


# ============================================================
# IDENTIFY BEST LONGITUDINAL PATIENTS
# ============================================================

header(
    "LONGITUDINAL PATIENT RANKING"
)


patient_scores = []


if len(pairwise_df) > 0:

    for patient_id, group in (
        pairwise_df.groupby(
            "patient_id"
        )
    ):

        aliases = (
            group[
                "lesion_alias"
            ]
            .nunique()
        )

        transitions = len(
            group
        )

        max_followups = (
            group[
                "current_study_order"
            ]
            .nunique()
        )

        median_interval = (
            group[
                "interval_days"
            ]
            .median()
        )

        patient_scores.append({

            "patient_id":
                patient_id,

            "lesion_aliases":
                aliases,

            "longitudinal_transitions":
                transitions,

            "study_orders_observed":
                max_followups,

            "median_interval_days":
                median_interval
        })


patient_scores_df = pd.DataFrame(
    patient_scores
)


if len(patient_scores_df) > 0:

    patient_scores_df = (
        patient_scores_df
        .sort_values(
            [
                "longitudinal_transitions",
                "lesion_aliases"
            ],
            ascending=False
        )
    )


    print(
        patient_scores_df.to_string(
            index=False
        )
    )


# ============================================================
# FINAL FEASIBILITY
# ============================================================

header(
    "STEP 8 - FINAL FEASIBILITY"
)


unique_trajectories = len(
    trajectory_df
)

longitudinal_trajectories = int(
    trajectory_df[
        "longitudinal"
    ].sum()
)

pairwise_observations = len(
    pairwise_df
)


print(
    "["
    +
    (
        "YES"
        if unique_trajectories > 0
        else "NO"
    )
    +
    "] RECIST target lesions identified"
)


print(
    "["
    +
    (
        "YES"
        if longitudinal_trajectories > 0
        else "NO"
    )
    +
    "] RECIST target lesions observed longitudinally"
)


print(
    "["
    +
    (
        "YES"
        if pairwise_observations > 0
        else "NO"
    )
    +
    "] Pairwise temporal measurements available"
)


print(
    "Unique target-lesion trajectories:",
    unique_trajectories
)

print(
    "Longitudinal trajectories:",
    longitudinal_trajectories
)

print(
    "Pairwise temporal observations:",
    pairwise_observations
)


# ============================================================
# IMPORTANT INTERPRETATION
# ============================================================

print()

print(
    "=" * 90
)

print(
    "IMPORTANT INTERPRETATION"
)

print(
    "=" * 90
)

print("""
This step validates the RECIST target-lesion temporal
structure.

A repeated lesion_label_alias is treated as a candidate
longitudinal identity because the dataset provides that
alias.

However, this step does NOT independently prove biological
identity using image registration.

The purpose is to determine whether the RECIST metadata
provides enough structured longitudinal observations for
the HBGD growth analysis.

DO NOT TRAIN A MODEL YET.
""")


# ============================================================
# SAVE RESULTS
# ============================================================

header(
    "SAVING STEP 8 RESULTS"
)


trajectory_file = os.path.join(
    DATA_DIR,
    "8_recist_trajectory_summary.csv"
)

pairwise_file = os.path.join(
    DATA_DIR,
    "8_recist_pairwise_changes.csv"
)

patient_file = os.path.join(
    DATA_DIR,
    "8_longitudinal_patient_ranking.csv"
)


trajectory_df.to_csv(
    trajectory_file,
    index=False
)

pairwise_df.to_csv(
    pairwise_file,
    index=False
)

patient_scores_df.to_csv(
    patient_file,
    index=False
)


print(
    "Saved:",
    trajectory_file
)

print(
    "Saved:",
    pairwise_file
)

print(
    "Saved:",
    patient_file
)


# ============================================================
# COMPLETE
# ============================================================

header(
    "STEP 8 COMPLETE"
)

print("""
Run:

    python 8_validate_recist_trajectories.py

Then send me the COMPLETE terminal output.

Do not train anything yet.
""")

print(
    "=" * 90
)

print(
    "END OF STEP 8"
)

print(
    "=" * 90
)