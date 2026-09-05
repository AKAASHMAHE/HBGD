import os
import glob
import numpy as np
import pandas as pd


# ============================================================
# STEP 28
# AUDIT CT COVERAGE FOR INDIVIDUALIZED HBGD RATE TARGETS
# ============================================================

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

FEATURE_FILE = os.path.join(ROOT, "9_hbgd_features.csv")
QUALITY_FILE = os.path.join(ROOT, "25_ct_crop_quality.csv")
CROP_DIR = os.path.join(ROOT, "25_lesion_ct_crops")

OUTPUT_FILE = os.path.join(
    ROOT,
    "28_ct_rate_coverage_audit.csv"
)

SUMMARY_FILE = os.path.join(
    ROOT,
    "28_ct_rate_coverage_summary.csv"
)


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 95)
    print(title)
    print("=" * 95)


def normalize_patient(value):
    try:
        return int(float(value))
    except Exception:
        return str(value).strip()


def normalize_alias(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def normalize_date(value):
    if pd.isna(value):
        return pd.NaT

    value = str(value).strip()

    dt = pd.to_datetime(
        value,
        errors="coerce",
        format="%Y-%m-%d"
    )

    if pd.notna(dt):
        return dt

    dt = pd.to_datetime(
        value,
        errors="coerce",
        format="%Y%m%d"
    )

    return dt


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


# ============================================================
# START
# ============================================================

header("STEP 28 - AUDITING CT COVERAGE FOR HBGD RATE TARGETS")

print("Dataset root:")
print(ROOT)

print()
print("Feature file:")
print(FEATURE_FILE)

print()
print("Quality file:")
print(QUALITY_FILE)

print()
print("Crop directory:")
print(CROP_DIR)


# ============================================================
# CHECK FILES
# ============================================================

header("CHECKING REQUIRED FILES")

for path in [FEATURE_FILE, QUALITY_FILE]:

    if os.path.exists(path):
        print("[YES]", os.path.basename(path))
    else:
        print("[NO]", os.path.basename(path))
        print("ERROR: Required file missing:")
        print(path)
        raise SystemExit(1)

if os.path.isdir(CROP_DIR):
    print("[YES] Crop directory exists")
else:
    print("[NO] Crop directory exists")


# ============================================================
# LOAD FEATURES
# ============================================================

header("LOADING HBGD FEATURE DATA")

features = pd.read_csv(FEATURE_FILE)

print("Feature records:", len(features))

required_features = [
    "patient_id",
    "study_date",
    "lesion_label_alias",
    "diameter_mm"
]

missing = [
    c for c in required_features
    if c not in features.columns
]

if missing:
    print("ERROR: Missing feature columns:")
    print(missing)
    raise SystemExit(1)

print("Required feature columns present.")


# ============================================================
# NORMALIZE FEATURES
# ============================================================

header("NORMALIZING FEATURE IDENTIFIERS")

features["patient_norm"] = features[
    "patient_id"
].apply(normalize_patient)

features["alias_norm"] = features[
    "lesion_label_alias"
].apply(normalize_alias)

features["date_norm"] = features[
    "study_date"
].apply(normalize_date)

features["diameter_num"] = features[
    "diameter_mm"
].apply(safe_float)

features = features[
    features["date_norm"].notna()
].copy()

features = features[
    features["diameter_num"].notna()
].copy()

print("Valid feature records:", len(features))


# ============================================================
# BUILD INDIVIDUALIZED RATE TARGETS
# ============================================================

header("BUILDING INDIVIDUALIZED RATE TARGETS")

features = features.sort_values(
    [
        "patient_norm",
        "alias_norm",
        "date_norm"
    ]
).reset_index(drop=True)

rate_records = []

for (patient, alias), group in features.groupby(
    ["patient_norm", "alias_norm"],
    sort=False
):

    group = group.sort_values(
        "date_norm"
    ).reset_index(drop=True)

    if len(group) < 2:
        continue

    first = group.iloc[0]
    second = group.iloc[1]

    d0 = float(first["diameter_num"])
    d1 = float(second["diameter_num"])

    dt = (
        second["date_norm"]
        -
        first["date_norm"]
    ).days

    if dt <= 0:
        continue

    if d0 <= 0 or d1 <= 0:
        continue

    rate = np.log(d1 / d0) / dt

    rate_records.append({
        "patient_id": patient,
        "lesion_alias": alias,
        "baseline_date": first["date_norm"],
        "second_date": second["date_norm"],
        "baseline_diameter_mm": d0,
        "second_diameter_mm": d1,
        "interval_days": dt,
        "individualized_rate": rate
    })


rates = pd.DataFrame(rate_records)

print(
    "Eligible longitudinal trajectories:",
    len(rates)
)

print(
    "Unique patients:",
    rates["patient_id"].nunique()
)


# ============================================================
# LOAD QUALITY DATA
# ============================================================

header("LOADING CT CROP QUALITY DATA")

quality = pd.read_csv(QUALITY_FILE)

print("Quality records:", len(quality))

print()
print("Quality columns detected:")
print(list(quality.columns))


# ============================================================
# FIND ACTUAL ALIAS COLUMN
# ============================================================

if "lesion_label_alias" in quality.columns:

    quality_alias_column = "lesion_label_alias"

elif "lesion_alias" in quality.columns:

    quality_alias_column = "lesion_alias"

else:

    print()
    print("ERROR: No lesion alias column found.")
    print("Expected one of:")
    print("  lesion_label_alias")
    print("  lesion_alias")
    raise SystemExit(1)


print()
print(
    "Using quality alias column:",
    quality_alias_column
)


# ============================================================
# REQUIRED QUALITY COLUMNS
# ============================================================

required_quality = [
    "patient_id",
    "study_date",
    quality_alias_column,
    "status"
]

missing = [
    c for c in required_quality
    if c not in quality.columns
]

if missing:
    print("ERROR: Missing quality columns:")
    print(missing)
    raise SystemExit(1)


# ============================================================
# NORMALIZE QUALITY
# ============================================================

header("NORMALIZING CT CROP RECORDS")

quality["patient_norm"] = quality[
    "patient_id"
].apply(normalize_patient)

quality["alias_norm"] = quality[
    quality_alias_column
].apply(normalize_alias)

quality["date_norm"] = quality[
    "study_date"
].apply(normalize_date)

quality["status_norm"] = (
    quality["status"]
    .astype(str)
    .str.strip()
    .str.lower()
)

print(
    "Valid quality dates:",
    quality["date_norm"].notna().sum()
)


# ============================================================
# CROP FILE COLUMN
# ============================================================

header("IDENTIFYING CROP FILE COLUMN")

crop_columns = [
    "crop_file",
    "crop_path",
    "crop_filename",
    "output_file",
    "file_path",
    "filepath"
]

crop_column = None

for col in crop_columns:
    if col in quality.columns:
        crop_column = col
        break

if crop_column:
    print("Crop path column:", crop_column)
else:
    print("WARNING: Crop path column not found.")


# ============================================================
# INDEX PHYSICAL CROP FILES
# ============================================================

header("INDEXING PHYSICAL CT CROP FILES")

physical_files = []

if os.path.isdir(CROP_DIR):

    for pattern in [
        "*.nii",
        "*.nii.gz",
        "*.npy"
    ]:

        physical_files.extend(
            glob.glob(
                os.path.join(
                    CROP_DIR,
                    "**",
                    pattern
                ),
                recursive=True
            )
        )

print(
    "Physical crop files found:",
    len(physical_files)
)

physical_lookup = {}

for path in physical_files:

    physical_lookup[
        os.path.basename(path).lower()
    ] = path


# ============================================================
# RESOLVE CROP FILE
# ============================================================

def resolve_crop_file(row):

    if crop_column is None:
        return None

    value = row.get(
        crop_column,
        ""
    )

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    # Absolute path
    if os.path.isabs(value):
        if os.path.isfile(value):
            return value

    # Relative to root
    candidate = os.path.join(
        ROOT,
        value
    )

    if os.path.isfile(candidate):
        return candidate

    # Relative to crop directory
    candidate = os.path.join(
        CROP_DIR,
        value
    )

    if os.path.isfile(candidate):
        return candidate

    # Basename
    basename = os.path.basename(
        value
    ).lower()

    return physical_lookup.get(
        basename
    )


# ============================================================
# BUILD QUALITY LOOKUP
# ============================================================

header("BUILDING CT CROP LOOKUP")

quality_lookup = {}

for _, row in quality.iterrows():

    key = (
        row["patient_norm"],
        row["alias_norm"],
        row["date_norm"]
    )

    quality_lookup.setdefault(
        key,
        []
    ).append(row)


print(
    "Patient/date/alias groups:",
    len(quality_lookup)
)


# ============================================================
# AUDIT RATE TARGETS
# ============================================================

header("AUDITING RATE TARGET -> BASELINE CT COVERAGE")

audit_records = []

for _, rate in rates.iterrows():

    patient = rate["patient_id"]
    alias = rate["lesion_alias"]
    baseline_date = rate["baseline_date"]

    key = (
        patient,
        alias,
        baseline_date
    )

    candidates = quality_lookup.get(
        key,
        []
    )

    result = {
        "patient_id": patient,
        "lesion_alias": alias,
        "baseline_date":
            baseline_date.strftime("%Y-%m-%d"),
        "second_date":
            rate["second_date"].strftime("%Y-%m-%d"),
        "interval_days":
            rate["interval_days"],
        "baseline_diameter_mm":
            rate["baseline_diameter_mm"],
        "second_diameter_mm":
            rate["second_diameter_mm"],
        "individualized_rate":
            rate["individualized_rate"],
        "quality_records_found":
            len(candidates),
        "crop_available":
            False,
        "crop_file": "",
        "status": "",
        "coverage_reason": ""
    }

    # No record
    if len(candidates) == 0:

        result["coverage_reason"] = (
            "No crop quality record for baseline lesion"
        )

        audit_records.append(result)
        continue

    # Multiple records
    if len(candidates) > 1:

        valid = []

        for candidate in candidates:

            status = str(
                candidate["status"]
            ).strip().lower()

            crop_path = resolve_crop_file(
                candidate
            )

            if (
                status == "valid"
                and
                crop_path is not None
            ):

                valid.append(
                    (
                        candidate,
                        crop_path
                    )
                )

        if len(valid) == 1:

            candidate, crop_path = valid[0]

            result["crop_available"] = True
            result["crop_file"] = crop_path
            result["status"] = "valid"
            result["coverage_reason"] = (
                "Duplicate records but one valid physical crop"
            )

        elif len(valid) > 1:

            result["coverage_reason"] = (
                "Multiple valid crop records - ambiguous"
            )

        else:

            result["coverage_reason"] = (
                "Duplicate records but no uniquely valid crop"
            )

        audit_records.append(result)
        continue

    # Single record
    candidate = candidates[0]

    status = str(
        candidate["status"]
    ).strip().lower()

    result["status"] = status

    crop_path = resolve_crop_file(
        candidate
    )

    if status != "valid":

        result["coverage_reason"] = (
            "Baseline crop extraction failed"
        )

    elif crop_path is None:

        result["coverage_reason"] = (
            "Crop marked valid but physical file missing"
        )

    else:

        result["crop_available"] = True
        result["crop_file"] = crop_path
        result["coverage_reason"] = (
            "Valid baseline CT crop available"
        )

    audit_records.append(result)


audit = pd.DataFrame(
    audit_records
)


# ============================================================
# OVERALL RESULT
# ============================================================

header("RATE TARGET COVERAGE RESULTS")

total = len(audit)

available = int(
    audit["crop_available"].sum()
)

missing = total - available

print(
    "Total eligible rate targets:",
    total
)

print(
    "Baseline CT crops available:",
    available
)

print(
    "Baseline CT crops unavailable:",
    missing
)

if total > 0:

    print(
        "CT coverage:",
        round(
            available / total * 100,
            2
        ),
        "%"
    )


# ============================================================
# TRAJECTORY DETAIL
# ============================================================

header("TRAJECTORY-BY-TRAJECTORY COVERAGE")

for _, row in audit.iterrows():

    marker = (
        "[YES]"
        if row["crop_available"]
        else "[NO] "
    )

    print(
        f"{marker} "
        f"Patient {row['patient_id']} | "
        f"Alias {row['lesion_alias']} | "
        f"Baseline {row['baseline_date']} | "
        f"Rate {row['individualized_rate']:.6f} | "
        f"{row['coverage_reason']}"
    )


# ============================================================
# MISSING REASONS
# ============================================================

header("MISSING CT COVERAGE REASONS")

missing_rows = audit[
    ~audit["crop_available"]
]

if len(missing_rows) == 0:

    print("No missing baseline CT crops.")

else:

    counts = (
        missing_rows[
            "coverage_reason"
        ]
        .value_counts()
    )

    for reason, count in counts.items():

        print(
            f"{count:3d} : {reason}"
        )


# ============================================================
# PATIENT SUMMARY
# ============================================================

header("PATIENT-LEVEL CT RATE COVERAGE")

patient_summary = (
    audit
    .groupby("patient_id")
    .agg(
        rate_targets=(
            "individualized_rate",
            "count"
        ),
        ct_available=(
            "crop_available",
            "sum"
        )
    )
    .reset_index()
)

patient_summary["ct_missing"] = (
    patient_summary["rate_targets"]
    -
    patient_summary["ct_available"]
)

patient_summary["coverage_percent"] = (
    patient_summary["ct_available"]
    /
    patient_summary["rate_targets"]
    *
    100
)

print(
    patient_summary.to_string(
        index=False
    )
)


# ============================================================
# STEP 27 PATIENT SPLIT
# ============================================================

header("PATIENT-INDEPENDENT SPLIT COVERAGE")

training_patients = [
    3, 7, 8, 12, 13, 16, 22
]

testing_patients = [
    23, 25, 30
]

train = audit[
    audit["patient_id"].isin(
        training_patients
    )
]

test = audit[
    audit["patient_id"].isin(
        testing_patients
    )
]

print(
    "Training patients:",
    training_patients
)

print(
    "Testing patients:",
    testing_patients
)

print()
print(
    "Training rate targets:",
    len(train)
)

print(
    "Training CT-linked samples:",
    int(
        train["crop_available"].sum()
    )
)

print()
print(
    "Testing rate targets:",
    len(test)
)

print(
    "Testing CT-linked samples:",
    int(
        test["crop_available"].sum()
    )
)


# ============================================================
# TEST PATIENT DETAIL
# ============================================================

header("TEST PATIENT CT COVERAGE")

for patient in testing_patients:

    subset = test[
        test["patient_id"] == patient
    ]

    print()
    print(
        "Patient",
        patient
    )

    if len(subset) == 0:

        print(
            "  No eligible rate target."
        )
        continue

    for _, row in subset.iterrows():

        marker = (
            "YES"
            if row["crop_available"]
            else "NO"
        )

        print(
            f"  Alias {row['lesion_alias']} | "
            f"Baseline {row['baseline_date']} | "
            f"CT crop: {marker} | "
            f"{row['coverage_reason']}"
        )


# ============================================================
# FINAL DECISION
# ============================================================

header("STEP 28 - CNN DATA SUFFICIENCY DECISION")

train_count = int(
    train["crop_available"].sum()
)

test_count = int(
    test["crop_available"].sum()
)

print(
    "Total CT-linked rate targets:",
    available
)

print(
    "Training CT-linked samples:",
    train_count
)

print(
    "Testing CT-linked samples:",
    test_count
)

print()

if train_count >= 6:
    print(
        "[YES] At least 6 training samples available"
    )
else:
    print(
        "[NO] Fewer than 6 training samples"
    )

if test_count >= 3:
    print(
        "[YES] At least 3 independent test samples available"
    )
else:
    print(
        "[NO] Fewer than 3 independent test samples"
    )

print()

if train_count >= 6 and test_count >= 3:

    print(
        "[YES] CT-to-HBGD exploratory experiment "
        "has minimum sample coverage."
    )

else:

    print(
        "[NO] CT-to-HBGD CNN experiment is "
        "not sufficiently supported by current coverage."
    )

    print(
        "Do NOT change the patient split merely "
        "to increase the test count."
    )


# ============================================================
# SAVE
# ============================================================

header("SAVING STEP 28 RESULTS")

audit.to_csv(
    OUTPUT_FILE,
    index=False
)

patient_summary.to_csv(
    SUMMARY_FILE,
    index=False
)

print(
    "Saved:",
    OUTPUT_FILE
)

print(
    "Saved:",
    SUMMARY_FILE
)


# ============================================================
# COMPLETE
# ============================================================

header("STEP 28 COMPLETE")

print(
    "No dataset files were modified."
)

print(
    "No CNN was trained."
)

print(
    "No biological state was claimed."
)

print(
    "This audit only evaluates whether existing "
    "CT crops cover the individualized HBGD rate targets."
)

print()
print(
    "Send the COMPLETE terminal output."
)

print()
print(
    "Run:"
)

print(
    "python 28_audit_ct_rate_coverage.py"
)

print()
print("=" * 95)
print("END OF STEP 28")
print("=" * 95)