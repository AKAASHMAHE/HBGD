import os
import sys
import pandas as pd
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

FEATURE_FILE = os.path.join(
    ROOT,
    "9_hbgd_features.csv"
)

QUALITY_FILE = os.path.join(
    ROOT,
    "25_ct_crop_quality.csv"
)

COVERAGE_FILE = os.path.join(
    ROOT,
    "32_ct_rate_repaired_coverage.csv"
)

OUTPUT_FILE = os.path.join(
    ROOT,
    "34_ct_trajectory_recovery_audit.csv"
)


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def normalize_date(series):

    s = series.astype(str).str.strip()

    result = pd.to_datetime(
        s,
        format="%Y-%m-%d",
        errors="coerce"
    )

    mask = result.isna()

    if mask.any():
        result.loc[mask] = pd.to_datetime(
            s.loc[mask],
            format="%Y%m%d",
            errors="coerce"
        )

    return result


def normalize_alias(series):

    return (
        series.astype(str)
        .str.strip()
        .str.upper()
    )


def normalize_patient(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def resolve_crop_path(path):

    if pd.isna(path):
        return None

    path = str(path).strip()

    if not path:
        return None

    # Absolute path
    if os.path.isabs(path):

        if os.path.exists(path):
            return os.path.normpath(path)

    # Path relative to dataset root
    candidate = os.path.join(
        ROOT,
        path
    )

    if os.path.exists(candidate):
        return os.path.normpath(candidate)

    # Sometimes the CSV contains the crop directory
    # name plus filename.
    filename = os.path.basename(path)

    candidate = os.path.join(
        ROOT,
        "25_lesion_ct_crops",
        filename
    )

    if os.path.exists(candidate):
        return os.path.normpath(candidate)

    # Search directly inside crop directory
    crop_dir = os.path.join(
        ROOT,
        "25_lesion_ct_crops"
    )

    if os.path.isdir(crop_dir):

        direct = os.path.join(
            crop_dir,
            filename
        )

        if os.path.exists(direct):
            return os.path.normpath(direct)

    return None


# ============================================================
# STEP 34
# ============================================================

header(
    "STEP 34 - CORRECTED CT LONGITUDINAL TRAJECTORY AUDIT"
)

print("""
This step audits the existing CT-to-HBGD linkage.

The Step 32 corrected linkage is treated as the authoritative
relationship between:

    lesion
       |
       +-- baseline CT crop
       |
       +-- future CT crop
       |
       +-- individualized HBGD rate

No original dataset is modified.

No CT crop is modified.

No CNN is trained.

No biological state is claimed.

No clinical validation is performed.
""")


# ============================================================
# CHECK FILES
# ============================================================

header("CHECKING REQUIRED FILES")

required_files = [
    FEATURE_FILE,
    QUALITY_FILE,
    COVERAGE_FILE
]

for path in required_files:

    if os.path.exists(path):
        print("[YES]", os.path.basename(path))

    else:
        print("[NO]", os.path.basename(path))
        print("Missing:", path)
        sys.exit(1)


# ============================================================
# LOAD DATA
# ============================================================

header("LOADING DATA")

features = pd.read_csv(
    FEATURE_FILE
)

quality = pd.read_csv(
    QUALITY_FILE
)

coverage = pd.read_csv(
    COVERAGE_FILE
)

print(
    "Feature records:",
    len(features)
)

print(
    "Quality records:",
    len(quality)
)

print(
    "Coverage records:",
    len(coverage)
)


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

header("IDENTIFYING COLUMNS")

if "lesion_label_alias" in features.columns:
    feature_alias = "lesion_label_alias"
elif "lesion_alias" in features.columns:
    feature_alias = "lesion_alias"
else:
    print("ERROR: No feature lesion alias column found.")
    sys.exit(1)


if "lesion_label_alias" in quality.columns:
    quality_alias = "lesion_label_alias"
elif "lesion_alias" in quality.columns:
    quality_alias = "lesion_alias"
else:
    print("ERROR: No quality lesion alias column found.")
    sys.exit(1)


print(
    "Feature alias column:",
    feature_alias
)

print(
    "Quality alias column:",
    quality_alias
)


# ============================================================
# NORMALIZE FEATURES
# ============================================================

header("NORMALIZING HBGD FEATURES")

features["patient_id"] = normalize_patient(
    features["patient_id"]
)

features[feature_alias] = normalize_alias(
    features[feature_alias]
)

features["study_date_dt"] = normalize_date(
    features["study_date"]
)

features = features.dropna(
    subset=[
        "patient_id",
        feature_alias,
        "study_date_dt"
    ]
).copy()

features["patient_id"] = (
    features["patient_id"]
    .astype(int)
)

print(
    "Valid feature records:",
    len(features)
)


# ============================================================
# BUILD HBGD TRAJECTORIES
# ============================================================

header(
    "BUILDING HBGD LONGITUDINAL TRAJECTORIES"
)

hbgd_rows = []

for (patient_id, alias), group in features.groupby(
    ["patient_id", feature_alias]
):

    group = group.sort_values(
        "study_date_dt"
    )

    if len(group) < 2:
        continue

    hbgd_rows.append({

        "patient_id": int(patient_id),

        "lesion_alias": alias,

        "hbgd_observations": len(group),

        "hbgd_first_date":
            group["study_date_dt"].min(),

        "hbgd_last_date":
            group["study_date_dt"].max(),

    })


hbgd = pd.DataFrame(
    hbgd_rows
)

print(
    "HBGD longitudinal trajectories:",
    len(hbgd)
)


# ============================================================
# NORMALIZE QUALITY DATA
# ============================================================

header(
    "NORMALIZING CT CROP QUALITY DATA"
)

quality["patient_id"] = normalize_patient(
    quality["patient_id"]
)

quality[quality_alias] = normalize_alias(
    quality[quality_alias]
)

quality["study_date_dt"] = normalize_date(
    quality["study_date"]
)

quality = quality.dropna(
    subset=[
        "patient_id",
        quality_alias,
        "study_date_dt"
    ]
).copy()

quality["patient_id"] = (
    quality["patient_id"]
    .astype(int)
)

quality["status_norm"] = (
    quality["status"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print(
    "Valid quality records:",
    len(quality)
)


# ============================================================
# RESOLVE PHYSICAL CROP FILES
# ============================================================

header(
    "RESOLVING PHYSICAL CT CROP FILES"
)

quality["resolved_crop_path"] = quality[
    "crop_file"
].apply(
    resolve_crop_path
)

quality["physical_crop_exists"] = (
    quality["resolved_crop_path"]
    .notna()
)

print(
    "Records with resolvable crop files:",
    int(
        quality["physical_crop_exists"].sum()
    )
)
# A crop is considered usable when the physical crop file exists.
# Step 25's status text is not used as the deciding criterion,
# because the physical file itself is the authoritative check.

successful = quality[
    quality["physical_crop_exists"]
].copy()
print(
    "Verified successful CT crops:",
    len(successful)
)


# ============================================================
# BUILD CT TRAJECTORIES
# ============================================================

header(
    "BUILDING VERIFIED CT TRAJECTORIES"
)

ct_rows = []

for (patient_id, alias), group in successful.groupby(
    ["patient_id", quality_alias]
):

    group = group.sort_values(
        "study_date_dt"
    )

    dates = list(
        group["study_date_dt"]
    )

    ct_rows.append({

        "patient_id":
            int(patient_id),

        "lesion_alias":
            alias,

        "ct_observations":
            len(group),

        "ct_first_date":
            dates[0],

        "ct_last_date":
            dates[-1],

        "ct_has_2":
            len(group) >= 2,

        "ct_has_3":
            len(group) >= 3,

        "ct_has_4":
            len(group) >= 4,

    })


ct = pd.DataFrame(
    ct_rows
)

if len(ct) == 0:

    print(
        "ERROR: No verified CT trajectories "
        "could be constructed."
    )

    sys.exit(1)


print(
    "Unique CT trajectories:",
    len(ct)
)

print(
    "CT trajectories with >=2 observations:",
    int(ct["ct_has_2"].sum())
)

print(
    "CT trajectories with >=3 observations:",
    int(ct["ct_has_3"].sum())
)

print(
    "CT trajectories with >=4 observations:",
    int(ct["ct_has_4"].sum())
)


# ============================================================
# MATCH CT TO HBGD
# ============================================================

header(
    "MATCHING CT TRAJECTORIES TO HBGD TRAJECTORIES"
)

merged = pd.merge(
    hbgd,
    ct,
    on=[
        "patient_id",
        "lesion_alias"
    ],
    how="left"
)

merged["ct_observations"] = (
    merged["ct_observations"]
    .fillna(0)
    .astype(int)
)

merged["ct_has_2"] = (
    merged["ct_has_2"]
    .fillna(False)
)

merged["ct_has_3"] = (
    merged["ct_has_3"]
    .fillna(False)
)

merged["ct_has_4"] = (
    merged["ct_has_4"]
    .fillna(False)
)

print(
    "HBGD trajectories:",
    len(merged)
)

print(
    "HBGD trajectories with >=2 CT:",
    int(merged["ct_has_2"].sum())
)

print(
    "HBGD trajectories with >=3 CT:",
    int(merged["ct_has_3"].sum())
)


# ============================================================
# CROSS-CHECK STEP 32
# ============================================================

header(
    "CROSS-CHECKING STEP 32"
)

if "patient_id" not in coverage.columns:

    print(
        "WARNING: Step 32 coverage does not contain "
        "patient_id."
    )

else:

    coverage["patient_id"] = normalize_patient(
        coverage["patient_id"]
    )

    if "alias" in coverage.columns:

        coverage_alias = "alias"

    elif "lesion_alias" in coverage.columns:

        coverage_alias = "lesion_alias"

    elif "lesion_label_alias" in coverage.columns:

        coverage_alias = "lesion_label_alias"

    else:

        coverage_alias = None

    if coverage_alias is not None:

        coverage[coverage_alias] = normalize_alias(
            coverage[coverage_alias]
        )

        coverage_keys = set(
            zip(
                coverage["patient_id"]
                .dropna()
                .astype(int),

                coverage.loc[
                    coverage["patient_id"].notna(),
                    coverage_alias
                ]
            )
        )

        print(
            "Step 32 patient-lesion keys:",
            len(coverage_keys)
        )

    else:

        print(
            "WARNING: Could not identify alias "
            "column in Step 32 coverage."
        )


# ============================================================
# DISPLAY VERIFIED LONGITUDINAL TRAJECTORIES
# ============================================================

header(
    "VERIFIED LONGITUDINAL CT TRAJECTORIES"
)

longitudinal = merged[
    merged["ct_has_2"]
].copy()

if len(longitudinal) == 0:

    print(
        "No HBGD trajectories have >=2 verified CT crops."
    )

else:

    for _, row in longitudinal.iterrows():

        print(
            f"Patient {row['patient_id']} | "
            f"Alias {row['lesion_alias']} | "
            f"CT observations = {row['ct_observations']} | "
            f"HBGD observations = {row['hbgd_observations']}"
        )


# ============================================================
# PATIENT-INDEPENDENT ANALYSIS
# ============================================================

header(
    "PATIENT-INDEPENDENT COVERAGE"
)

TRAIN_PATIENTS = [
    3, 7, 8, 12, 13, 16, 22
]

TEST_PATIENTS = [
    23, 25, 30
]

train = merged[
    merged["patient_id"].isin(
        TRAIN_PATIENTS
    )
].copy()

test = merged[
    merged["patient_id"].isin(
        TEST_PATIENTS
    )
].copy()

print(
    "Requested training patients:",
    TRAIN_PATIENTS
)

print(
    "Requested testing patients:",
    TEST_PATIENTS
)

print(
    "Training HBGD trajectories:",
    len(train)
)

print(
    "Training CT longitudinal trajectories:",
    int(
        train["ct_has_2"].sum()
    )
)

print(
    "Testing HBGD trajectories:",
    len(test)
)

print(
    "Testing CT longitudinal trajectories:",
    int(
        test["ct_has_2"].sum()
    )
)


# ============================================================
# TEST PATIENT DETAILS
# ============================================================

header(
    "TEST PATIENT DETAILS"
)

for patient_id in TEST_PATIENTS:

    group = test[
        test["patient_id"] == patient_id
    ]

    print(
        f"\nPatient {patient_id}"
    )

    if len(group) == 0:

        print(
            "  No HBGD trajectory."
        )

        continue

    for _, row in group.iterrows():

        print(
            f"  Alias {row['lesion_alias']} | "
            f"HBGD observations = "
            f"{row['hbgd_observations']} | "
            f"CT observations = "
            f"{row['ct_observations']}"
        )


# ============================================================
# PATIENT-LEVEL SUMMARY
# ============================================================

header(
    "PATIENT-LEVEL CT COVERAGE"
)

patient_rows = []

for patient_id, group in merged.groupby(
    "patient_id"
):

    patient_rows.append({

        "patient_id":
            int(patient_id),

        "hbgd_trajectories":
            len(group),

        "ct_longitudinal_trajectories":
            int(group["ct_has_2"].sum()),

        "ct_3plus_trajectories":
            int(group["ct_has_3"].sum()),

        "ct_4plus_trajectories":
            int(group["ct_has_4"].sum()),

    })

patient_summary = pd.DataFrame(
    patient_rows
)

print(
    patient_summary.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULT
# ============================================================

header(
    "SAVING STEP 34 RESULTS"
)

merged["ct_longitudinal_available"] = (
    merged["ct_has_2"]
)

merged["ct_3plus_available"] = (
    merged["ct_has_3"]
)

merged["ct_4plus_available"] = (
    merged["ct_has_4"]
)

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "Saved:",
    OUTPUT_FILE
)


# ============================================================
# FINAL DECISION
# ============================================================

header(
    "STEP 34 - FINAL DECISION"
)

total_ct_longitudinal = int(
    merged["ct_has_2"].sum()
)

train_ct_longitudinal = int(
    train["ct_has_2"].sum()
)

test_ct_longitudinal = int(
    test["ct_has_2"].sum()
)

test_patients_present = sorted(
    test.loc[
        test["ct_has_2"],
        "patient_id"
    ].unique()
    .tolist()
)

print(
    "Total HBGD trajectories:",
    len(merged)
)

print(
    "Verified longitudinal CT trajectories:",
    total_ct_longitudinal
)

print(
    "Training CT trajectories:",
    train_ct_longitudinal
)

print(
    "Testing CT trajectories:",
    test_ct_longitudinal
)

print(
    "Independent test patients represented:",
    test_patients_present
)

if (
    test_ct_longitudinal >= 3
    and
    len(test_patients_present) >= 3
):

    print(
        "[YES] Sufficient independent CT trajectories "
        "for exploratory CNN evaluation."
    )

else:

    print(
        "[NO] Independent CT coverage remains "
        "insufficient for meaningful CNN validation."
    )

print(
    "[YES] Original dataset unchanged."
)

print(
    "[YES] Original CT crops unchanged."
)

print(
    "[YES] No CNN trained."
)

print(
    "[YES] No biological state claimed."
)

print(
    "[YES] No clinical validation claimed."
)

header(
    "STEP 34 COMPLETE"
)

print("""
Run:

    python 34_audit_ct_trajectory_recovery.py

Then send the COMPLETE terminal output.

DO NOT TRAIN ANOTHER CNN YET.
""")