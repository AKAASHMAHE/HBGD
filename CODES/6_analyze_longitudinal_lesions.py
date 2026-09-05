import os
import sys
import json
import numpy as np
import pandas as pd


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

SERIES_FILE = os.path.join(
    DATA_DIR,
    "series.json"
)

RECIST_FILE = os.path.join(
    DATA_DIR,
    "recist_measurements.csv"
)


# ============================================================
# HEADER
# ============================================================

def header(title):

    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


# ============================================================
# LOAD DATA
# ============================================================

header("STEP 6 - LOADING LONGITUDINAL DATA")


if not os.path.exists(SERIES_FILE):

    print("ERROR: series.json not found")
    print(SERIES_FILE)
    sys.exit(1)


if not os.path.exists(RECIST_FILE):

    print("ERROR: recist_measurements.csv not found")
    print(RECIST_FILE)
    sys.exit(1)


with open(
    SERIES_FILE,
    "r",
    encoding="utf-8"
) as f:

    series_data = json.load(f)


series_df = pd.DataFrame(
    series_data
)

recist_df = pd.read_csv(
    RECIST_FILE
)


print(
    "Series records:",
    len(series_df)
)

print(
    "RECIST records:",
    len(recist_df)
)


# ============================================================
# NORMALIZE DATES
# ============================================================

header("NORMALIZING STUDY DATES")


series_df["study_date"] = (
    series_df["study_date"]
    .astype(str)
)

series_df["date_parsed"] = pd.to_datetime(
    series_df["study_date"],
    format="%Y%m%d",
    errors="coerce"
)


recist_df["study_date"] = (
    recist_df["study_date"]
    .astype(str)
)

recist_df["date_parsed"] = pd.to_datetime(
    recist_df["study_date"],
    format="%Y%m%d",
    errors="coerce"
)


print(
    "Valid series dates:",
    series_df["date_parsed"].notna().sum()
)

print(
    "Invalid series dates:",
    series_df["date_parsed"].isna().sum()
)


# ============================================================
# PATIENT LONGITUDINAL STRUCTURE
# ============================================================

header("PATIENT LONGITUDINAL STRUCTURE")


patient_summary = []


for patient_id, group in series_df.groupby(
    "patient_id"
):

    group = group.sort_values(
        "date_parsed"
    )

    dates = (
        group["date_parsed"]
        .dropna()
        .unique()
    )

    number_of_scans = len(
        dates
    )

    if number_of_scans >= 2:

        longitudinal = True

    else:

        longitudinal = False


    if len(dates) >= 2:

        intervals = []

        for i in range(
            1,
            len(dates)
        ):

            delta = (
                dates[i] -
                dates[i - 1]
            )

            intervals.append(
                delta.days
            )

        min_interval = min(
            intervals
        )

        max_interval = max(
            intervals
        )

        total_followup_days = (
            dates[-1] -
            dates[0]
        ).days

    else:

        min_interval = np.nan
        max_interval = np.nan
        total_followup_days = 0


    patient_summary.append({

        "patient_id":
            patient_id,

        "number_of_series":
            len(group),

        "number_of_unique_dates":
            number_of_scans,

        "longitudinal":
            longitudinal,

        "first_date":
            dates[0].strftime("%Y%m%d")
            if len(dates) > 0
            else "",

        "last_date":
            dates[-1].strftime("%Y%m%d")
            if len(dates) > 0
            else "",

        "min_interval_days":
            min_interval,

        "max_interval_days":
            max_interval,

        "total_followup_days":
            total_followup_days
    })


patient_df = pd.DataFrame(
    patient_summary
)


print(
    "Unique patients:",
    len(patient_df)
)

print(
    "Patients with >=2 scan dates:",
    int(
        patient_df[
            "longitudinal"
        ].sum()
    )
)

print(
    "Patients with only one scan date:",
    int(
        (
            ~patient_df[
                "longitudinal"
            ]
        ).sum()
    )
)


# ============================================================
# PATIENT DETAILS
# ============================================================

header("PATIENT SCAN TIMELINES")


for _, row in patient_df.iterrows():

    if not row["longitudinal"]:
        continue

    patient_id = row[
        "patient_id"
    ]

    group = series_df[
        series_df[
            "patient_id"
        ] == patient_id
    ].sort_values(
        "date_parsed"
    )

    unique_dates = (
        group[
            "date_parsed"
        ]
        .drop_duplicates()
        .tolist()
    )

    print()
    print(
        "Patient:",
        patient_id
    )

    print(
        "  Number of series:",
        len(group)
    )

    print(
        "  Scan dates:"
    )

    for i, date in enumerate(
        unique_dates,
        start=1
    ):

        print(
            "   ",
            i,
            date.strftime("%Y-%m-%d")
        )

    if len(unique_dates) >= 2:

        print(
            "  Intervals:"
        )

        for i in range(
            1,
            len(unique_dates)
        ):

            days = (
                unique_dates[i]
                -
                unique_dates[i - 1]
            ).days

            print(
                "   ",
                unique_dates[i - 1].strftime(
                    "%Y-%m-%d"
                ),
                "->",
                unique_dates[i].strftime(
                    "%Y-%m-%d"
                ),
                ":",
                days,
                "days"
            )


# ============================================================
# RECIST TEMPORAL STRUCTURE
# ============================================================

header("RECIST TEMPORAL STRUCTURE")


recist_patient_summary = []


for patient_id, group in recist_df.groupby(
    "patient_id"
):

    group = group.sort_values(
        "date_parsed"
    )

    dates = (
        group[
            "date_parsed"
        ]
        .dropna()
        .unique()
    )

    recist_patient_summary.append({

        "patient_id":
            patient_id,

        "recist_records":
            len(group),

        "unique_recist_dates":
            len(dates),

        "longitudinal_recist":
            len(dates) >= 2
    })


recist_patient_df = pd.DataFrame(
    recist_patient_summary
)


print(
    "Patients with RECIST data:",
    len(recist_patient_df)
)

print(
    "Patients with RECIST on >=2 dates:",
    int(
        recist_patient_df[
            "longitudinal_recist"
        ].sum()
    )
)


# ============================================================
# RECIST STUDY ORDER
# ============================================================

header("RECIST STUDY ORDER")


if "study_order" in recist_df.columns:

    study_order_counts = (
        recist_df[
            "study_order"
        ]
        .value_counts()
    )

    for name, count in (
        study_order_counts.items()
    ):

        print(
            f"{name}: {count}"
        )


# ============================================================
# LESION ALIAS ANALYSIS
# ============================================================

header("LESION ALIAS LONGITUDINAL ANALYSIS")


if "lesion_label_alias" in recist_df.columns:

    alias_summary = []

    for (
        patient_id,
        alias
    ), group in recist_df.groupby(
        [
            "patient_id",
            "lesion_label_alias"
        ]
    ):

        group = group.sort_values(
            "date_parsed"
        )

        dates = (
            group[
                "date_parsed"
            ]
            .dropna()
            .unique()
        )

        measurements = (
            group[
                "recist_measurement_mm"
            ]
            .astype(float)
            .tolist()
        )

        alias_summary.append({

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

            "first_measurement_mm":
                measurements[0]
                if len(measurements) > 0
                else np.nan,

            "last_measurement_mm":
                measurements[-1]
                if len(measurements) > 0
                else np.nan
        })


    alias_df = pd.DataFrame(
        alias_summary
    )

    print(
        "Unique patient-lesion aliases:",
        len(alias_df)
    )

    print(
        "Aliases appearing on >=2 dates:",
        int(
            alias_df[
                "longitudinal"
            ].sum()
        )
    )

else:

    print(
        "lesion_label_alias column not found"
    )

    alias_df = pd.DataFrame()


# ============================================================
# RECIST GROWTH CALCULATION
# ============================================================

header("RECIST DIAMETER CHANGE ANALYSIS")


growth_results = []


if not recist_df.empty:

    grouped = recist_df.groupby(
        [
            "patient_id",
            "lesion_label_alias"
        ]
    )

    for (
        patient_id,
        alias
    ), group in grouped:

        group = group.sort_values(
            "date_parsed"
        ).copy()

        if len(group) < 2:
            continue

        first = group.iloc[0]
        last = group.iloc[-1]

        first_date = (
            first["date_parsed"]
        )

        last_date = (
            last["date_parsed"]
        )

        first_mm = float(
            first[
                "recist_measurement_mm"
            ]
        )

        last_mm = float(
            last[
                "recist_measurement_mm"
            ]
        )

        delta_days = (
            last_date -
            first_date
        ).days

        diameter_change = (
            last_mm -
            first_mm
        )

        if first_mm != 0:

            percent_change = (
                diameter_change /
                first_mm
            ) * 100.0

        else:

            percent_change = np.nan


        if delta_days > 0:

            diameter_rate_mm_per_day = (
                diameter_change /
                delta_days
            )

        else:

            diameter_rate_mm_per_day = np.nan


        growth_results.append({

            "patient_id":
                patient_id,

            "lesion_alias":
                alias,

            "first_date":
                first_date.strftime(
                    "%Y%m%d"
                ),

            "last_date":
                last_date.strftime(
                    "%Y%m%d"
                ),

            "time_days":
                delta_days,

            "first_diameter_mm":
                first_mm,

            "last_diameter_mm":
                last_mm,

            "diameter_change_mm":
                diameter_change,

            "percent_change":
                percent_change,

            "diameter_rate_mm_per_day":
                diameter_rate_mm_per_day
        })


growth_df = pd.DataFrame(
    growth_results
)


print(
    "Longitudinal lesion trajectories:",
    len(growth_df)
)


if len(growth_df) > 0:

    print()
    print(
        "Diameter change statistics:"
    )

    print(
        "  Mean change:",
        round(
            growth_df[
                "diameter_change_mm"
            ].mean(),
            3
        ),
        "mm"
    )

    print(
        "  Median change:",
        round(
            growth_df[
                "diameter_change_mm"
            ].median(),
            3
        ),
        "mm"
    )

    print(
        "  Mean percent change:",
        round(
            growth_df[
                "percent_change"
            ].mean(),
            3
        ),
        "%"
    )

    print(
        "  Mean time interval:",
        round(
            growth_df[
                "time_days"
            ].mean(),
            2
        ),
        "days"
    )


# ============================================================
# RECIST MEASUREMENT COUNTS BY PATIENT
# ============================================================

header("RECIST MEASUREMENTS PER PATIENT")


patient_recist_counts = (
    recist_df
    .groupby(
        "patient_id"
    )
    .size()
    .sort_values(
        ascending=False
    )
)


for patient_id, count in (
    patient_recist_counts.items()
):

    print(
        "Patient",
        patient_id,
        ":",
        count,
        "RECIST records"
    )


# ============================================================
# STUDY ORDER CONSISTENCY
# ============================================================

header("STUDY ORDER CONSISTENCY")


if "study_order" in recist_df.columns:

    consistency_rows = []

    for patient_id, group in (
        recist_df.groupby(
            "patient_id"
        )
    ):

        unique_orders = (
            group[
                "study_order"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        unique_dates = (
            group[
                "study_date"
            ]
            .unique()
        )

        consistency_rows.append({

            "patient_id":
                patient_id,

            "number_of_dates":
                len(unique_dates),

            "number_of_study_orders":
                len(unique_orders),

            "study_orders":
                ",".join(
                    unique_orders
                )
        })


    consistency_df = pd.DataFrame(
        consistency_rows
    )

    print(
        "Patients analyzed:",
        len(consistency_df)
    )

    print()

    for _, row in (
        consistency_df.iterrows()
    ):

        print(
            "Patient",
            row["patient_id"],
            ":",
            row["study_orders"]
        )

else:

    consistency_df = pd.DataFrame()


# ============================================================
# FINAL DATASET FEASIBILITY
# ============================================================

header("STEP 6 - LONGITUDINAL FEASIBILITY")


number_of_patients = len(
    patient_df
)

longitudinal_patients = int(
    patient_df[
        "longitudinal"
    ].sum()
)

recist_longitudinal_patients = int(
    recist_patient_df[
        "longitudinal_recist"
    ].sum()
)


has_longitudinal_data = (
    longitudinal_patients > 0
)

has_longitudinal_recist = (
    recist_longitudinal_patients > 0
)

has_trajectory_data = (
    len(growth_df) > 0
)


print(
    "["
    + (
        "YES"
        if number_of_patients > 0
        else "NO"
    )
    + "] Patients identified"
)


print(
    "["
    + (
        "YES"
        if has_longitudinal_data
        else "NO"
    )
    + "] Patients have multiple scan dates"
)


print(
    "["
    + (
        "YES"
        if has_longitudinal_recist
        else "NO"
    )
    + "] Patients have longitudinal RECIST data"
)


print(
    "["
    + (
        "YES"
        if has_trajectory_data
        else "NO"
    )
    + "] At least one RECIST lesion trajectory exists"
)


# ============================================================
# IMPORTANT INTERPRETATION
# ============================================================

print()
print("=" * 90)
print("IMPORTANT INTERPRETATION")
print("=" * 90)

print("""
This step does NOT assume that a numerical mask label
represents the same biological lesion across different scans.

It only determines whether longitudinal information exists
and whether RECIST measurements can be organized across time.

A later step will be required to establish reliable lesion
identity/tracking between scans.
""")


# ============================================================
# SAVE OUTPUTS
# ============================================================

header("SAVING STEP 6 RESULTS")


patient_file = os.path.join(
    DATA_DIR,
    "6_patient_longitudinal_summary.csv"
)

recist_patient_file = os.path.join(
    DATA_DIR,
    "6_recist_patient_summary.csv"
)

growth_file = os.path.join(
    DATA_DIR,
    "6_recist_growth_analysis.csv"
)

alias_file = os.path.join(
    DATA_DIR,
    "6_lesion_alias_summary.csv"
)

consistency_file = os.path.join(
    DATA_DIR,
    "6_study_order_consistency.csv"
)


patient_df.to_csv(
    patient_file,
    index=False
)

recist_patient_df.to_csv(
    recist_patient_file,
    index=False
)

growth_df.to_csv(
    growth_file,
    index=False
)

alias_df.to_csv(
    alias_file,
    index=False
)

consistency_df.to_csv(
    consistency_file,
    index=False
)


print(
    "Saved:",
    patient_file
)

print(
    "Saved:",
    recist_patient_file
)

print(
    "Saved:",
    growth_file
)

print(
    "Saved:",
    alias_file
)

print(
    "Saved:",
    consistency_file
)


# ============================================================
# END
# ============================================================

header("STEP 6 COMPLETE")

print("""
DO NOT TRAIN A MODEL YET.

Send me the COMPLETE terminal output from:

    python 6_analyze_longitudinal_lesions.py

The next decision depends on how many genuine longitudinal
RECIST trajectories are available.
""")

print("=" * 90)
print("END OF STEP 6")
print("=" * 90)