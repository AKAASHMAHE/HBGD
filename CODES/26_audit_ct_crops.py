import os
import sys
import glob
import numpy as np
import pandas as pd
import nibabel as nib


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

QUALITY_FILE = os.path.join(
    DATA_ROOT,
    "25_ct_crop_quality.csv"
)

CROP_DIR = os.path.join(
    DATA_ROOT,
    "25_lesion_ct_crops"
)

AUDIT_FILE = os.path.join(
    DATA_ROOT,
    "26_ct_crop_audit.csv"
)


# ============================================================
# UTF-8 TERMINAL
# ============================================================

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 95)
    print(title)
    print("=" * 95)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


# ============================================================
# START
# ============================================================

header("STEP 26 - CT CROP FAILURE AND QUALITY AUDIT")

print("Dataset root:")
print(DATA_ROOT)

print()
print("Quality file:")
print(QUALITY_FILE)

print()
print("Crop directory:")
print(CROP_DIR)


# ============================================================
# LOAD QUALITY FILE
# ============================================================

header("LOADING STEP 25 RESULTS")

if not os.path.exists(QUALITY_FILE):
    raise FileNotFoundError(
        "25_ct_crop_quality.csv was not found."
    )

df = pd.read_csv(QUALITY_FILE)

print("Records loaded:", len(df))

required = [
    "patient_id",
    "study_date",
    "lesion_label_alias",
    "lesion_label_value",
    "status",
    "reason"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns: " + str(missing)
    )


# ============================================================
# BASIC STATUS
# ============================================================

header("BASIC EXTRACTION STATUS")

print(
    "Total records:",
    len(df)
)

print(
    "Valid:",
    int(
        (df["status"] == "VALID").sum()
    )
)

print(
    "Failed:",
    int(
        (df["status"] != "VALID").sum()
    )
)


# ============================================================
# FAILURE ANALYSIS
# ============================================================

header("FAILURE REASON ANALYSIS")

failed = df[
    df["status"] != "VALID"
].copy()

if len(failed) == 0:

    print("No failed crops.")

else:

    counts = (
        failed["reason"]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
    )

    for reason, count in counts.items():

        print(
            f"{count:3d} : {reason}"
        )


# ============================================================
# SUCCESSFUL CROP FILE CHECK
# ============================================================

header("VERIFYING SUCCESSFUL CROP FILES")

valid = df[
    df["status"] == "VALID"
].copy()

missing_crop_files = []
load_failures = []
invalid_dimensions = []
invalid_values = []

for i, row in valid.iterrows():

    path = str(
        row.get("crop_file", "")
    )

    if not path or path.lower() == "nan":

        missing_crop_files.append(i)
        continue

    if not os.path.exists(path):

        missing_crop_files.append(i)
        continue

    try:

        img = nib.load(path)

        arr = img.get_fdata(
            dtype=np.float32
        )

        if arr.ndim != 3:

            invalid_dimensions.append(
                (
                    i,
                    arr.shape
                )
            )

            continue

        finite = np.isfinite(arr)

        if not finite.all():

            invalid_values.append(
                (
                    i,
                    "non-finite values"
                )
            )

            continue

        if arr.size == 0:

            invalid_dimensions.append(
                (
                    i,
                    arr.shape
                )
            )

            continue

    except Exception as exc:

        load_failures.append(
            (
                i,
                str(exc)
            )
        )


print(
    "Successful crop records:",
    len(valid)
)

print(
    "Missing crop files:",
    len(missing_crop_files)
)

print(
    "Crop loading failures:",
    len(load_failures)
)

print(
    "Invalid dimensions:",
    len(invalid_dimensions)
)

print(
    "Invalid voxel values:",
    len(invalid_values)
)


# ============================================================
# CROP SHAPE ANALYSIS
# ============================================================

header("CROP SHAPE ANALYSIS")

if len(valid) > 0:

    shape_counts = (
        valid["crop_shape"]
        .fillna("UNKNOWN")
        .astype(str)
        .value_counts()
    )

    print(
        "Unique crop shapes:",
        len(shape_counts)
    )

    print()

    for shape, count in shape_counts.head(20).items():

        print(
            f"{shape:25s} : {count}"
        )


# ============================================================
# CROP FRACTION ANALYSIS
# ============================================================

header("LESION-TO-CROP FRACTION")

if len(valid) > 0:

    fractions = pd.to_numeric(
        valid["crop_fraction"],
        errors="coerce"
    )

    print(
        "Minimum:",
        round(fractions.min(), 6)
    )

    print(
        "Q1:",
        round(fractions.quantile(0.25), 6)
    )

    print(
        "Median:",
        round(fractions.median(), 6)
    )

    print(
        "Q3:",
        round(fractions.quantile(0.75), 6)
    )

    print(
        "Maximum:",
        round(fractions.max(), 6)
    )


# ============================================================
# DUPLICATE PATIENT / DATE / ALIAS CHECK
# ============================================================

header("RECIST LESION IDENTITY AUDIT")

identity_cols = [
    "patient_id",
    "study_date",
    "lesion_label_alias"
]

identity_counts = (
    df
    .groupby(identity_cols)
    .size()
    .reset_index(name="count")
)

duplicates = identity_counts[
    identity_counts["count"] > 1
]

print(
    "Patient/date/alias groups:",
    len(identity_counts)
)

print(
    "Duplicate patient/date/alias groups:",
    len(duplicates)
)

if len(duplicates) > 0:

    print()
    print(
        "DUPLICATE GROUPS:"
    )

    for _, row in duplicates.iterrows():

        print(
            "Patient",
            int(row["patient_id"]),
            "|",
            row["study_date"],
            "| Alias",
            row["lesion_label_alias"],
            "| records:",
            int(row["count"])
        )


# ============================================================
# DUPLICATE LABEL CHECK
# ============================================================

header("LABEL CONSISTENCY AUDIT")

label_counts = (
    df
    .groupby(
        [
            "patient_id",
            "study_date",
            "lesion_label_alias"
        ]
    )["lesion_label_value"]
    .nunique()
    .reset_index(
        name="unique_labels"
    )
)

ambiguous_labels = label_counts[
    label_counts["unique_labels"] > 1
]

print(
    "Patient/date/alias groups:",
    len(label_counts)
)

print(
    "Groups with multiple label values:",
    len(ambiguous_labels)
)

if len(ambiguous_labels) > 0:

    print()

    for _, row in ambiguous_labels.iterrows():

        patient = int(
            row["patient_id"]
        )

        date = row["study_date"]

        alias = row[
            "lesion_label_alias"
        ]

        subset = df[
            (df["patient_id"] == patient) &
            (df["study_date"] == date) &
            (
                df["lesion_label_alias"]
                == alias
            )
        ]

        labels = sorted(
            subset[
                "lesion_label_value"
            ]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        print(
            "Patient",
            patient,
            "|",
            date,
            "| Alias",
            alias,
            "| labels:",
            labels
        )


# ============================================================
# LONGITUDINAL COVERAGE
# ============================================================

header("LONGITUDINAL TRAJECTORY AUDIT")

if len(valid) > 0:

    trajectory = (
        valid
        .groupby(
            [
                "patient_id",
                "lesion_label_alias"
            ]
        )
        .agg(
            observations=(
                "study_date",
                "count"
            ),
            dates=(
                "study_date",
                lambda x:
                sorted(
                    set(
                        x.astype(str)
                    )
                )
            )
        )
        .reset_index()
    )

    print(
        "Unique patient-lesion trajectories:",
        len(trajectory)
    )

    print(
        ">=2 observations:",
        int(
            (
                trajectory["observations"]
                >= 2
            ).sum()
        )
    )

    print(
        ">=3 observations:",
        int(
            (
                trajectory["observations"]
                >= 3
            ).sum()
        )
    )

    print(
        ">=4 observations:",
        int(
            (
                trajectory["observations"]
                >= 4
            ).sum()
        )
    )

    print()

    for _, row in trajectory.iterrows():

        if row["observations"] >= 2:

            print(
                "Patient",
                int(row["patient_id"]),
                "| Alias",
                row["lesion_label_alias"],
                "| n =",
                int(row["observations"]),
                "| dates =",
                ", ".join(row["dates"])
            )


# ============================================================
# PATIENT COVERAGE
# ============================================================

header("PATIENT-LEVEL COVERAGE")

if len(valid) > 0:

    patient_counts = (
        valid
        .groupby("patient_id")
        .size()
    )

    print(
        "Patients represented:",
        len(patient_counts)
    )

    print(
        "Patients with >=2 crops:",
        int(
            (patient_counts >= 2).sum()
        )
    )

    print(
        "Patients with >=3 crops:",
        int(
            (patient_counts >= 3).sum()
        )
    )

    print()

    print(
        "Patient IDs:"
    )

    print(
        sorted(
            patient_counts.index
            .astype(int)
            .tolist()
        )
    )


# ============================================================
# FAILED PATIENT COVERAGE
# ============================================================

header("FAILED RECORDS BY PATIENT")

if len(failed) > 0:

    failed_patient_counts = (
        failed
        .groupby("patient_id")
        .size()
        .sort_values(
            ascending=False
        )
    )

    for patient, count in (
        failed_patient_counts.items()
    ):

        print(
            "Patient",
            int(patient),
            ":",
            int(count),
            "failed"
        )


# ============================================================
# BUILD AUDIT TABLE
# ============================================================

header("CREATING AUDIT TABLE")

audit = df.copy()

audit["crop_file_exists"] = False
audit["crop_loadable"] = False
audit["crop_is_3d"] = False
audit["crop_has_finite_values"] = False

for idx in audit.index:

    if audit.loc[idx, "status"] != "VALID":
        continue

    path = str(
        audit.loc[idx, "crop_file"]
    )

    if not path or path.lower() == "nan":
        continue

    if not os.path.exists(path):
        continue

    audit.loc[
        idx,
        "crop_file_exists"
    ] = True

    try:

        img = nib.load(path)

        arr = img.get_fdata(
            dtype=np.float32
        )

        audit.loc[
            idx,
            "crop_loadable"
        ] = True

        if arr.ndim == 3:

            audit.loc[
                idx,
                "crop_is_3d"
            ] = True

        if np.isfinite(arr).all():

            audit.loc[
                idx,
                "crop_has_finite_values"
            ] = True

    except Exception:
        pass


audit.to_csv(
    AUDIT_FILE,
    index=False
)

print(
    "Saved:",
    AUDIT_FILE
)


# ============================================================
# FINAL DECISION
# ============================================================

header("STEP 26 - FINAL DECISION")

valid_files = audit[
    (audit["status"] == "VALID") &
    (audit["crop_file_exists"]) &
    (audit["crop_loadable"]) &
    (audit["crop_is_3d"]) &
    (audit["crop_has_finite_values"])
]

print(
    "Valid crop files:",
    len(valid_files)
)

print(
    "Total RECIST records:",
    len(audit)
)

print(
    "Failed extraction:",
    int(
        (audit["status"] != "VALID").sum()
    )
)

print()

if len(valid_files) >= 30:
    print(
        "[YES] Enough verified CT crops for an exploratory imaging experiment"
    )
else:
    print(
        "[NO] Too few verified CT crops for imaging experiment"
    )

if len(duplicates) == 0:
    print(
        "[YES] No duplicate patient/date/alias groups"
    )
else:
    print(
        "[WARNING] Duplicate patient/date/alias groups require review"
    )

if len(ambiguous_labels) == 0:
    print(
        "[YES] No ambiguous RECIST lesion labels"
    )
else:
    print(
        "[WARNING] Some RECIST aliases map to multiple labels"
    )

if len(valid) > 0:

    trajectory_check = (
        valid
        .groupby(
            [
                "patient_id",
                "lesion_label_alias"
            ]
        )
        .size()
        .ge(2)
        .sum()
    )

    if trajectory_check >= 8:
        print(
            "[YES] Sufficient longitudinal CT trajectories for exploratory work"
        )
    else:
        print(
            "[NO] Too few longitudinal CT trajectories"
        )

else:

    print(
        "[NO] No longitudinal CT trajectories"
    )


print()
print(
    "IMPORTANT:"
)

print(
    "This audit does not train a model."
)

print(
    "It does not establish biological states."
)

print(
    "It does not establish clinical validity."
)

print(
    "It only determines whether the extracted CT inputs"
)

print(
    "are safe enough for the next imaging experiment."
)


header("STEP 26 COMPLETE")

print()
print(
    "Send the COMPLETE terminal output."
)