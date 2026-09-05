# 2_longitudinal_patient_analysis.py

import os
import json
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

PATIENTS_FILE = os.path.join(DATA_DIR, "patients.csv")
RECIST_FILE = os.path.join(DATA_DIR, "recist_measurements.csv")
SERIES_FILE = os.path.join(DATA_DIR, "series.json")


# ============================================================
# HELPER
# ============================================================

def header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def parse_date(value):
    """
    Dataset stores dates as YYYYMMDD integers.
    Example: 20230426 -> 2023-04-26
    """
    return pd.to_datetime(
        str(int(value)),
        format="%Y%m%d",
        errors="coerce"
    )


# ============================================================
# LOAD DATA
# ============================================================

header("LOADING DATA")

patients = pd.read_csv(PATIENTS_FILE)
recist = pd.read_csv(RECIST_FILE)

with open(SERIES_FILE, "r", encoding="utf-8") as f:
    series = json.load(f)

series_df = pd.DataFrame(series)

print(f"Patients rows       : {len(patients)}")
print(f"RECIST rows         : {len(recist)}")
print(f"Series rows         : {len(series_df)}")


# ============================================================
# CORRECT DATE PARSING
# ============================================================

patients["date"] = patients["first_study_date"].apply(parse_date)
recist["date"] = recist["study_date"].apply(parse_date)
series_df["date"] = series_df["study_date"].apply(parse_date)

print("\nCorrectly parsed RECIST date range:")
print(f"  {recist['date'].min().date()} -> {recist['date'].max().date()}")


# ============================================================
# PATIENT-LEVEL LONGITUDINAL ANALYSIS
# ============================================================

header("PATIENT-LEVEL LONGITUDINAL ANALYSIS")

patient_results = []

for patient_id in sorted(patients["patient_id"].unique()):

    patient_info = patients[
        patients["patient_id"] == patient_id
    ]

    patient_recist = recist[
        recist["patient_id"] == patient_id
    ].copy()

    patient_series = series_df[
        series_df["patient_id"] == patient_id
    ].copy()

    if len(patient_recist) == 0:
        patient_results.append({
            "patient_id": patient_id,
            "diagnosis": patient_info["diagnosis"].iloc[0],
            "num_recist_rows": 0,
            "num_studies": 0,
            "num_dates": 0,
            "num_series": len(patient_series),
            "first_date": None,
            "last_date": None,
            "followup_days": 0,
            "has_longitudinal_data": False,
            "num_measurements": 0
        })
        continue

    dates = sorted(
        patient_recist["date"].dropna().unique()
    )

    first_date = min(dates)
    last_date = max(dates)

    followup_days = (
        pd.Timestamp(last_date) -
        pd.Timestamp(first_date)
    ).days

    patient_results.append({
        "patient_id": patient_id,
        "diagnosis": patient_info["diagnosis"].iloc[0],
        "num_recist_rows": len(patient_recist),
        "num_studies": patient_recist["study_uuid"].nunique(),
        "num_dates": len(dates),
        "num_series": len(patient_series),
        "first_date": first_date.date(),
        "last_date": last_date.date(),
        "followup_days": followup_days,
        "has_longitudinal_data": len(dates) >= 2,
        "num_measurements": patient_recist["recist_measurement_mm"].notna().sum()
    })


results = pd.DataFrame(patient_results)

print(results.to_string(index=False))


# ============================================================
# LONGITUDINAL PATIENTS
# ============================================================

header("PATIENTS WITH MULTIPLE TIME POINTS")

longitudinal = results[
    results["has_longitudinal_data"] == True
].copy()

print(
    f"Patients with >=2 study dates: "
    f"{len(longitudinal)} / {len(results)}"
)

if len(longitudinal) > 0:

    print("\nLongitudinal patients:")

    print(
        longitudinal[
            [
                "patient_id",
                "diagnosis",
                "num_studies",
                "num_dates",
                "num_series",
                "first_date",
                "last_date",
                "followup_days",
                "num_measurements"
            ]
        ].to_string(index=False)
    )


# ============================================================
# ALL RECIST MEASUREMENTS BY PATIENT AND DATE
# ============================================================

header("RECIST MEASUREMENTS OVER TIME")

for patient_id in sorted(recist["patient_id"].unique()):

    p = recist[
        recist["patient_id"] == patient_id
    ].copy()

    p = p.sort_values(["date", "lesion_label_alias"])

    print("\n" + "-" * 90)

    diagnosis = patients.loc[
        patients["patient_id"] == patient_id,
        "diagnosis"
    ].iloc[0]

    print(
        f"PATIENT {patient_id} | "
        f"DIAGNOSIS: {diagnosis}"
    )

    print(
        p[
            [
                "date",
                "study_order",
                "region",
                "lesion_label_alias",
                "lesion_label_value",
                "recist_measurement_mm",
                "study_uuid"
            ]
        ].to_string(index=False)
    )


# ============================================================
# TIME INTERVAL ANALYSIS
# ============================================================

header("TIME INTERVAL BETWEEN STUDIES")

interval_results = []

for patient_id in sorted(recist["patient_id"].unique()):

    p = recist[
        recist["patient_id"] == patient_id
    ].copy()

    dates = sorted(
        p["date"].dropna().unique()
    )

    if len(dates) < 2:
        continue

    for i in range(1, len(dates)):

        previous_date = pd.Timestamp(dates[i - 1])
        current_date = pd.Timestamp(dates[i])

        interval_days = (
            current_date - previous_date
        ).days

        interval_results.append({
            "patient_id": patient_id,
            "previous_date": previous_date.date(),
            "current_date": current_date.date(),
            "interval_days": interval_days
        })


interval_df = pd.DataFrame(interval_results)

if len(interval_df) > 0:

    print(interval_df.to_string(index=False))

    print("\nInterval statistics:")

    print(
        f"  Minimum interval : "
        f"{interval_df['interval_days'].min()} days"
    )

    print(
        f"  Maximum interval : "
        f"{interval_df['interval_days'].max()} days"
    )

    print(
        f"  Mean interval    : "
        f"{interval_df['interval_days'].mean():.1f} days"
    )


# ============================================================
# LESION TRACKING
# ============================================================

header("LESION TRACKING ACROSS TIME")

lesion_results = []

for patient_id in sorted(recist["patient_id"].unique()):

    p = recist[
        recist["patient_id"] == patient_id
    ].copy()

    for lesion_alias in sorted(
        p["lesion_label_alias"].dropna().unique()
    ):

        lesion = p[
            p["lesion_label_alias"] == lesion_alias
        ].sort_values("date")

        unique_dates = lesion["date"].nunique()

        measurements = (
            lesion["recist_measurement_mm"]
            .dropna()
            .tolist()
        )

        lesion_results.append({
            "patient_id": patient_id,
            "lesion": lesion_alias,
            "num_timepoints": unique_dates,
            "measurements_mm": measurements,
            "has_followup": unique_dates >= 2
        })


lesion_df = pd.DataFrame(lesion_results)

print(
    f"Total patient-lesion combinations: "
    f"{len(lesion_df)}"
)

print(
    f"Lesions observed at >=2 time points: "
    f"{lesion_df['has_followup'].sum()}"
)

if len(lesion_df) > 0:

    print("\nLesion tracking summary:")

    print(
        lesion_df.to_string(index=False)
    )


# ============================================================
# SERIES ↔ RECIST LINKAGE
# ============================================================

header("CT SERIES ↔ RECIST LINKAGE")

print(
    f"Total CT series records: "
    f"{len(series_df)}"
)

print(
    f"Unique series UUIDs: "
    f"{series_df['uuid'].nunique()}"
)

print(
    f"Unique RECIST UUIDs: "
    f"{recist['uuid'].nunique()}"
)


series_uuid_set = set(
    series_df["uuid"].astype(str)
)

recist_uuid_set = set(
    recist["uuid"].astype(str)
)

matched_uuids = series_uuid_set.intersection(
    recist_uuid_set
)

print(
    f"\nRECIST UUIDs matching CT series UUIDs: "
    f"{len(matched_uuids)}"
)

print(
    f"RECIST UUIDs NOT matching CT series UUIDs: "
    f"{len(recist_uuid_set - series_uuid_set)}"
)


# ============================================================
# STUDY UUID LINKAGE
# ============================================================

header("STUDY LINKAGE")

series_studies = set(
    series_df["study_uuid"].astype(str)
)

recist_studies = set(
    recist["study_uuid"].astype(str)
)

matched_studies = series_studies.intersection(
    recist_studies
)

print(
    f"Unique CT study UUIDs: "
    f"{len(series_studies)}"
)

print(
    f"Unique RECIST study UUIDs: "
    f"{len(recist_studies)}"
)

print(
    f"Matching study UUIDs: "
    f"{len(matched_studies)}"
)


# ============================================================
# DIAGNOSIS DISTRIBUTION
# ============================================================

header("DIAGNOSIS DISTRIBUTION")

diagnosis_table = (
    patients
    .groupby("diagnosis")
    .agg(
        patients=("patient_id", "nunique")
    )
    .sort_values("patients", ascending=False)
)

print(diagnosis_table.to_string())


# ============================================================
# FINAL DATASET ASSESSMENT
# ============================================================

header("LONGITUDINAL DATASET ASSESSMENT")

total_patients = len(patients)

recist_patients = recist["patient_id"].nunique()

longitudinal_patients = len(longitudinal)

patients_with_followup_measurements = 0

for patient_id in recist["patient_id"].unique():

    p = recist[
        recist["patient_id"] == patient_id
    ]

    if p["date"].nunique() >= 2:
        patients_with_followup_measurements += 1


print(f"""
TOTAL PATIENTS
--------------
{total_patients}


PATIENTS WITH RECIST DATA
-------------------------
{recist_patients}


PATIENTS WITH MULTIPLE STUDY DATES
----------------------------------
{longitudinal_patients}


PATIENTS WITH POTENTIAL LONGITUDINAL MEASUREMENTS
-------------------------------------------------
{patients_with_followup_measurements}


TOTAL CT SERIES
---------------
{len(series_df)}


TOTAL RECIST RECORDS
--------------------
{len(recist)}


UNIQUE STUDY DATES
------------------
{recist["date"].nunique()}


UNIQUE STUDIES
--------------
{recist["study_uuid"].nunique()}


UNIQUE LESION LABELS
--------------------
{recist["lesion_label_alias"].nunique()}


RECIST ↔ CT SERIES UUID MATCHES
-------------------------------
{len(matched_uuids)}


RECIST ↔ CT STUDY UUID MATCHES
------------------------------
{len(matched_studies)}
""")


# ============================================================
# DECISION FLAGS
# ============================================================

header("PROJECT FEASIBILITY FLAGS")

checks = {
    "Patient IDs available":
        total_patients > 0,

    "RECIST tumor measurements available":
        len(recist) > 0,

    "Multiple CT studies exist":
        recist["study_uuid"].nunique() > total_patients,

    "Multiple scan dates exist":
        recist["date"].nunique() > total_patients,

    "At least one patient has longitudinal data":
        longitudinal_patients > 0,

    "CT series metadata available":
        len(series_df) > 0,

    "RECIST ↔ CT series linkage exists":
        len(matched_uuids) > 0,

    "RECIST ↔ study linkage exists":
        len(matched_studies) > 0
}

for check, result in checks.items():

    status = "YES" if result else "NO"

    print(
        f"[{status:3}] {check}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

output_file = os.path.join(
    DATA_DIR,
    "2_longitudinal_patient_summary.csv"
)

results.to_csv(
    output_file,
    index=False
)

print("\nSummary saved to:")
print(output_file)

print("\n")
print("=" * 90)
print("STEP 2 COMPLETE")
print("=" * 90)

print("""
IMPORTANT:
Do not start model training yet.

This step determines whether the 22-patient dataset actually contains
the longitudinal structure required by our project.

Send me the COMPLETE terminal output from:

    python 2_longitudinal_patient_analysis.py

The next step will be decided from the actual longitudinal patients,
time intervals, lesion tracking, and CT ↔ RECIST linkage.
""")