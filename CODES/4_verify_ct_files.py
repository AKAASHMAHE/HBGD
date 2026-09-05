# 4_verify_ct_files.py

import os
import json
import pandas as pd
import sys
sys.stdout.reconfigure(
    encoding="utf-8",
    errors="replace"
)

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

SERIES_FILE = os.path.join(DATA_DIR, "series.json")
WINDOWS_FILE = os.path.join(DATA_DIR, "windows_mapping.json")
RECIST_FILE = os.path.join(DATA_DIR, "recist_measurements.csv")


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print("\n" + "=" * 100)
    print(title.encode("ascii", "replace").decode())
    print("=" * 100)

def find_nii_files(root):
    """
    Search recursively for NIfTI files.
    """

    nii_files = []

    for current_root, dirs, files in os.walk(root):

        for filename in files:

            if (
                filename.endswith(".nii") or
                filename.endswith(".nii.gz")
            ):
                nii_files.append(
                    os.path.join(
                        current_root,
                        filename
                    )
                )

    return nii_files


# ============================================================
# LOAD METADATA
# ============================================================

header("LOADING DATASET METADATA")

with open(SERIES_FILE, "r", encoding="utf-8") as f:
    series_data = json.load(f)

with open(WINDOWS_FILE, "r", encoding="utf-8") as f:
    windows_mapping = json.load(f)

recist = pd.read_csv(RECIST_FILE)

print(
    f"Series metadata records : {len(series_data)}"
)

print(
    f"Windows mapping records : {len(windows_mapping)}"
)

print(
    f"RECIST records           : {len(recist)}"
)


# ============================================================
# SEARCH FOR ACTUAL CT FILES
# ============================================================

header("SEARCHING FOR ACTUAL CT FILES")

nii_files = find_nii_files(DATA_DIR)

print(
    f"Actual .nii/.nii.gz files found: "
    f"{len(nii_files)}"
)

if len(nii_files) > 0:

    print("\nFirst 20 CT files:")

    for path in nii_files[:20]:

        relative_path = os.path.relpath(
            path,
            DATA_DIR
        )

        size_mb = (
            os.path.getsize(path)
            /
            (1024 * 1024)
        )

        print(
            f"  {relative_path} "
            f"({size_mb:.2f} MB)"
        )

else:

    print("""
NO NIFTI FILES FOUND.

This is important.

The four metadata files may describe CT images that are stored
elsewhere or may have been provided without the actual CT volumes.

Do NOT download or train anything yet.
""")


# ============================================================
# WINDOWS MAPPING ANALYSIS
# ============================================================

header("WINDOWS MAPPING ANALYSIS")

print(
    "Number of mapped filenames: "
    f"{len(windows_mapping)}"
)

print("\nWindow mapping values:")

window_counts = {}

for filename, region in windows_mapping.items():

    window_counts[region] = (
        window_counts.get(region, 0) + 1
    )

for region, count in sorted(
    window_counts.items()
):

    print(
        f"  {region}: {count}"
    )


# ============================================================
# COMPARE WINDOWS MAPPING WITH RECIST FILENAMES
# ============================================================

header("WINDOWS ↔ RECIST FILENAME LINKAGE")

recist_filenames = set(
    recist["filename"]
    .astype(str)
)

mapping_filenames = set(
    windows_mapping.keys()
)

matched = (
    recist_filenames
    .intersection(mapping_filenames)
)

missing_from_mapping = (
    recist_filenames
    -
    mapping_filenames
)

extra_in_mapping = (
    mapping_filenames
    -
    recist_filenames
)

print(
    f"Unique RECIST filenames: "
    f"{len(recist_filenames)}"
)

print(
    f"Windows mapping filenames: "
    f"{len(mapping_filenames)}"
)

print(
    f"RECIST filenames found in mapping: "
    f"{len(matched)}"
)

print(
    f"RECIST filenames missing from mapping: "
    f"{len(missing_from_mapping)}"
)

print(
    f"Mapping filenames without RECIST record: "
    f"{len(extra_in_mapping)}"
)


if len(missing_from_mapping) > 0:

    print("\nRECIST filenames missing from mapping:")

    for filename in sorted(
        missing_from_mapping
    ):

        print(
            f"  {filename}"
        )


# ============================================================
# COMPARE ACTUAL FILES WITH WINDOWS MAPPING
# ============================================================

header("ACTUAL CT FILE ↔ METADATA LINKAGE")

actual_filenames = set()

for path in nii_files:

    filename = os.path.basename(path)

    actual_filenames.add(filename)


print(
    f"Actual CT filenames: "
    f"{len(actual_filenames)}"
)

print(
    f"Metadata filenames: "
    f"{len(mapping_filenames)}"
)

actual_matches = (
    actual_filenames
    .intersection(mapping_filenames)
)

print(
    f"Actual CT files matching metadata: "
    f"{len(actual_matches)}"
)

actual_missing = (
    mapping_filenames
    -
    actual_filenames
)

print(
    f"Metadata files with NO local CT file: "
    f"{len(actual_missing)}"
)


if len(actual_missing) > 0:

    print("\nFirst 30 missing CT files:")

    for filename in sorted(
        actual_missing
    )[:30]:

        print(
            f"  {filename}"
        )


# ============================================================
# VERIFY SERIES METADATA
# ============================================================

header("CT SERIES METADATA CHECK")

series_df = pd.DataFrame(series_data)

required_series_columns = [
    "id",
    "region",
    "uuid",
    "study_uuid",
    "study_date",
    "patient_id",
    "slice_thickness",
    "row_spacing",
    "column_spacing",
    "slices",
    "rows",
    "columns",
    "slice_spacing"
]

print("Expected metadata fields:")

for column in required_series_columns:

    if column in series_df.columns:

        print(
            f"  [YES] {column}"
        )

    else:

        print(
            f"  [NO ] {column}"
        )


# ============================================================
# CT DIMENSION ANALYSIS
# ============================================================

header("CT VOLUME DIMENSIONS")

if len(series_df) > 0:

    print(
        series_df[
            [
                "patient_id",
                "region",
                "study_date",
                "slices",
                "rows",
                "columns",
                "slice_thickness",
                "slice_spacing",
                "row_spacing",
                "column_spacing"
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# UNIQUE PATIENTS WITH CT SERIES
# ============================================================

header("PATIENTS WITH CT SERIES")

patients_with_series = (
    series_df["patient_id"]
    .nunique()
)

print(
    f"Patients represented in CT series: "
    f"{patients_with_series}"
)

print("\nCT series count per patient:")

series_per_patient = (
    series_df
    .groupby("patient_id")
    .size()
    .sort_values(ascending=False)
)

print(
    series_per_patient.to_string()
)


# ============================================================
# CT STUDIES PER PATIENT
# ============================================================

header("STUDIES PER PATIENT")

studies_per_patient = (
    series_df
    .groupby("patient_id")["study_uuid"]
    .nunique()
    .sort_values(ascending=False)
)

print(
    studies_per_patient.to_string()
)


# ============================================================
# LONGITUDINAL CT AVAILABILITY
# ============================================================

header("LONGITUDINAL CT AVAILABILITY")

longitudinal_ct = []

for patient_id in sorted(
    series_df["patient_id"].unique()
):

    p = series_df[
        series_df["patient_id"] == patient_id
    ]

    dates = sorted(
        p["study_date"]
        .astype(str)
        .unique()
    )

    longitudinal_ct.append({

        "patient_id":
            patient_id,

        "number_of_series":
            len(p),

        "number_of_studies":
            p["study_uuid"].nunique(),

        "number_of_dates":
            len(dates),

        "first_date":
            dates[0] if dates else None,

        "last_date":
            dates[-1] if dates else None,

        "has_multiple_dates":
            len(dates) >= 2
    })


longitudinal_ct_df = pd.DataFrame(
    longitudinal_ct
)

print(
    longitudinal_ct_df.to_string(
        index=False
    )
)


# ============================================================
# RECIST + CT PATIENT OVERLAP
# ============================================================

header("RECIST ↔ CT PATIENT OVERLAP")

recist_patients = set(
    recist["patient_id"]
)

ct_patients = set(
    series_df["patient_id"]
)

patient_overlap = (
    recist_patients
    .intersection(ct_patients)
)

print(
    f"Patients with RECIST: "
    f"{len(recist_patients)}"
)

print(
    f"Patients with CT series: "
    f"{len(ct_patients)}"
)

print(
    f"Patients present in BOTH: "
    f"{len(patient_overlap)}"
)

print(
    f"RECIST patients without CT metadata: "
    f"{len(recist_patients - ct_patients)}"
)


# ============================================================
# RECIST STUDY ↔ CT STUDY OVERLAP
# ============================================================

header("RECIST ↔ CT STUDY OVERLAP")

recist_studies = set(
    recist["study_uuid"]
)

ct_studies = set(
    series_df["study_uuid"]
)

study_overlap = (
    recist_studies
    .intersection(ct_studies)
)

print(
    f"RECIST studies: "
    f"{len(recist_studies)}"
)

print(
    f"CT studies: "
    f"{len(ct_studies)}"
)

print(
    f"Studies present in BOTH: "
    f"{len(study_overlap)}"
)

print(
    f"RECIST studies without CT metadata: "
    f"{len(recist_studies - ct_studies)}"
)


# ============================================================
# FINAL FEASIBILITY FLAGS
# ============================================================

header("STEP 4 — CT DATA FEASIBILITY")

checks = {

    "CT series metadata exists":
        len(series_df) > 0,

    "CT patient IDs exist":
        "patient_id" in series_df.columns,

    "CT study IDs exist":
        "study_uuid" in series_df.columns,

    "CT dates exist":
        "study_date" in series_df.columns,

    "CT dimensions exist":
        all(
            c in series_df.columns
            for c in [
                "slices",
                "rows",
                "columns"
            ]
        ),

    "CT spacing information exists":
        all(
            c in series_df.columns
            for c in [
                "slice_spacing",
                "row_spacing",
                "column_spacing"
            ]
        ),

    "RECIST ↔ CT patient linkage exists":
        len(patient_overlap) > 0,

    "RECIST ↔ CT study linkage exists":
        len(study_overlap) > 0,

    "Actual CT files available locally":
        len(nii_files) > 0
}


for check, result in checks.items():

    status = "YES" if result else "NO"

    print(
        f"[{status:3}] {check}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = os.path.join(
    DATA_DIR,
    "4_ct_verification_summary.csv"
)

longitudinal_ct_df.to_csv(
    output_file,
    index=False
)

print("\nSummary saved to:")
print(output_file)


# ============================================================
# FINAL MESSAGE
# ============================================================

header("STEP 4 COMPLETE")

print("""
The critical question now is:

DO WE ACTUALLY HAVE THE CT VOLUMES?

If .nii/.nii.gz files are present:

    CT volume
        ↓
    patient / study UUID
        ↓
    RECIST measurement
        ↓
    longitudinal time point

Then we can move toward extracting imaging-derived features.

If the CT volumes are NOT present:

    We still have a valuable longitudinal RECIST dataset,

but we cannot yet test the image-processing portion of the HBGD
architecture from these files alone.

IMPORTANT:

Send me the COMPLETE output of:

    python 4_verify_ct_files.py

Do not start model training yet.
""")

print("=" * 100)
print("END OF STEP 4")
print("=" * 100)