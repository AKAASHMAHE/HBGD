import os
import glob
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

FEATURE_FILE = os.path.join(ROOT, "9_hbgd_features.csv")
QUALITY_FILE = os.path.join(ROOT, "25_ct_crop_quality.csv")
CROP_DIR = os.path.join(ROOT, "25_lesion_ct_crops")

OUTPUT_FILE = os.path.join(ROOT, "30_ct_rate_coverage.csv")


# ============================================================
# DISPLAY HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 95)
    print(title)
    print("=" * 95)


def subheader(title):
    print()
    print("-" * 95)
    print(title)
    print("-" * 95)


# ============================================================
# START
# ============================================================

header("STEP 30 - AUDITING CT COVERAGE FOR HBGD RATE TARGETS")

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

required = [
    ("9_hbgd_features.csv", FEATURE_FILE),
    ("25_ct_crop_quality.csv", QUALITY_FILE),
]

for name, path in required:
    if os.path.exists(path):
        print("[YES]", name)
    else:
        print("[NO] ", name)
        raise FileNotFoundError(path)

if os.path.isdir(CROP_DIR):
    print("[YES] Crop directory exists")
else:
    print("[NO] Crop directory exists")
    raise FileNotFoundError(CROP_DIR)


# ============================================================
# LOAD FEATURE DATA
# ============================================================

header("LOADING HBGD FEATURE DATA")

features = pd.read_csv(FEATURE_FILE)

print("Feature records:", len(features))

required_feature_columns = [
    "patient_id",
    "study_date",
    "lesion_label_alias",
    "diameter_mm",
]

missing = [
    c for c in required_feature_columns
    if c not in features.columns
]

if missing:
    print("ERROR: Missing feature columns:")
    print(missing)
    raise SystemExit(1)

print("Required feature columns present.")


# ============================================================
# NORMALIZE FEATURE IDENTIFIERS
# ============================================================

header("NORMALIZING FEATURE IDENTIFIERS")

features["patient_id"] = pd.to_numeric(
    features["patient_id"],
    errors="coerce"
)

features["study_date"] = pd.to_datetime(
    features["study_date"],
    errors="coerce"
)

features["lesion_label_alias"] = (
    features["lesion_label_alias"]
    .astype(str)
    .str.strip()
    .str.upper()
)

features["diameter_mm"] = pd.to_numeric(
    features["diameter_mm"],
    errors="coerce"
)

features = features.dropna(
    subset=[
        "patient_id",
        "study_date",
        "lesion_label_alias",
        "diameter_mm",
    ]
).copy()

features["patient_id"] = features["patient_id"].astype(int)

features = features.sort_values(
    [
        "patient_id",
        "lesion_label_alias",
        "study_date",
    ]
).reset_index(drop=True)

print("Valid feature records:", len(features))


# ============================================================
# BUILD TRAJECTORIES
# ============================================================

header("BUILDING LONGITUDINAL RATE TARGETS")

trajectories = []

group_columns = [
    "patient_id",
    "lesion_label_alias",
]

for (patient_id, alias), group in features.groupby(
    group_columns,
    sort=True
):

    group = group.sort_values("study_date").copy()

    if len(group) < 2:
        continue

    first = group.iloc[0]
    second = group.iloc[1]

    d0 = float(first["diameter_mm"])
    d1 = float(second["diameter_mm"])

    t0 = first["study_date"]
    t1 = second["study_date"]

    delta_days = (t1 - t0).days

    if delta_days <= 0:
        continue

    if d0 <= 0 or d1 <= 0:
        continue

    # Individualized exponential growth rate
    r = np.log(d1 / d0) / delta_days

    trajectories.append({
        "patient_id": int(patient_id),
        "lesion_alias": alias,
        "baseline_date": t0,
        "second_date": t1,
        "baseline_diameter_mm": d0,
        "second_diameter_mm": d1,
        "delta_days": delta_days,
        "individualized_rate": r,
        "trajectory_observations": len(group),
    })


targets = pd.DataFrame(trajectories)

print("Eligible longitudinal trajectories:", len(targets))

if len(targets) == 0:
    print("ERROR: No longitudinal trajectories found.")
    raise SystemExit(1)

print(
    "Unique patients:",
    targets["patient_id"].nunique()
)


# ============================================================
# LOAD CT CROP QUALITY
# ============================================================

header("LOADING CT CROP QUALITY DATA")

quality = pd.read_csv(QUALITY_FILE)

print("Quality records:", len(quality))

alias_column = None

for candidate in [
    "lesion_label_alias",
    "lesion_alias",
]:
    if candidate in quality.columns:
        alias_column = candidate
        break

if alias_column is None:
    print("ERROR: No lesion alias column found.")
    print("Available columns:")
    print(list(quality.columns))
    raise SystemExit(1)

print("Alias column:", alias_column)

required_quality_columns = [
    "patient_id",
    "study_date",
    "status",
    "crop_file",
]

missing = [
    c for c in required_quality_columns
    if c not in quality.columns
]

if missing:
    print("ERROR: Missing quality columns:")
    print(missing)
    raise SystemExit(1)


# ============================================================
# NORMALIZE QUALITY DATA
# ============================================================

header("NORMALIZING CT CROP DATA")

quality["patient_id"] = pd.to_numeric(
    quality["patient_id"],
    errors="coerce"
)

quality["study_date"] = pd.to_datetime(
    quality["study_date"],
    errors="coerce"
)

quality["lesion_alias"] = (
    quality[alias_column]
    .astype(str)
    .str.strip()
    .str.upper()
)

quality["status"] = (
    quality["status"]
    .astype(str)
    .str.strip()
    .str.lower()
)

quality = quality.dropna(
    subset=[
        "patient_id",
        "study_date",
        "lesion_alias",
    ]
).copy()

quality["patient_id"] = quality["patient_id"].astype(int)

print("Valid quality records:", len(quality))


# ============================================================
# VERIFY PHYSICAL CROP FILES
# ============================================================

header("VERIFYING PHYSICAL CT CROP FILES")

quality["physical_crop_exists"] = False

for idx, row in quality.iterrows():

    crop_file = str(row["crop_file"]).strip()

    if crop_file == "" or crop_file.lower() == "nan":
        continue

    # Direct path
    if os.path.isabs(crop_file):
        exists = os.path.exists(crop_file)

    else:
        # Relative to crop directory
        path1 = os.path.join(CROP_DIR, crop_file)

        # Relative to dataset root
        path2 = os.path.join(ROOT, crop_file)

        exists = (
            os.path.exists(path1)
            or os.path.exists(path2)
        )

    quality.at[idx, "physical_crop_exists"] = exists


valid_physical = quality[
    quality["physical_crop_exists"]
].copy()

print(
    "Records with physically existing crops:",
    len(valid_physical)
)


# ============================================================
# BUILD CROP LOOKUP
# ============================================================

header("BUILDING CT CROP LOOKUP")

crop_lookup = {}

for _, row in valid_physical.iterrows():

    key = (
        int(row["patient_id"]),
        row["lesion_alias"],
        row["study_date"],
    )

    crop_lookup[key] = {
        "status": row["status"],
        "crop_file": row["crop_file"],
    }

print("Unique CT crop keys:", len(crop_lookup))


# ============================================================
# MATCH BASELINE CT TO RATE TARGET
# ============================================================

header("MATCHING BASELINE CT CROPS TO INDIVIDUALIZED RATES")

audit_rows = []

for _, target in targets.iterrows():

    patient_id = int(target["patient_id"])
    alias = target["lesion_alias"]
    baseline_date = target["baseline_date"]

    key = (
        patient_id,
        alias,
        baseline_date,
    )

    baseline_crop_exists = key in crop_lookup

    baseline_crop_valid = False
    baseline_crop_file = ""

    if baseline_crop_exists:

        crop_info = crop_lookup[key]

        baseline_crop_file = str(
            crop_info["crop_file"]
        )

        status = str(
            crop_info["status"]
        ).lower()

        baseline_crop_valid = (
            status == "success"
            and baseline_crop_file != ""
        )

    # --------------------------------------------------------
    # Find all CT observations for this trajectory
    # --------------------------------------------------------

    trajectory_quality = valid_physical[
        (valid_physical["patient_id"] == patient_id)
        &
        (valid_physical["lesion_alias"] == alias)
    ].copy()

    trajectory_quality = trajectory_quality.sort_values(
        "study_date"
    )

    ct_dates = trajectory_quality[
        "study_date"
    ].tolist()

    ct_observation_count = len(ct_dates)

    future_ct_count = sum(
        d > baseline_date
        for d in ct_dates
    )

    future_ct_exists = future_ct_count > 0

    # --------------------------------------------------------
    # Determine usability
    # --------------------------------------------------------

    usable_for_ct_rate = (
        baseline_crop_valid
        and future_ct_exists
    )

    audit_rows.append({

        "patient_id": patient_id,
        "lesion_alias": alias,

        "baseline_date": baseline_date,
        "second_date": target["second_date"],

        "baseline_diameter_mm":
            target["baseline_diameter_mm"],

        "second_diameter_mm":
            target["second_diameter_mm"],

        "delta_days":
            target["delta_days"],

        "individualized_rate":
            target["individualized_rate"],

        "trajectory_observations":
            target["trajectory_observations"],

        "baseline_ct_crop_exists":
            baseline_crop_exists,

        "baseline_ct_crop_valid":
            baseline_crop_valid,

        "baseline_crop_file":
            baseline_crop_file,

        "ct_observation_count":
            ct_observation_count,

        "future_ct_count":
            future_ct_count,

        "future_ct_exists":
            future_ct_exists,

        "usable_for_ct_rate":
            usable_for_ct_rate,
    })


audit = pd.DataFrame(audit_rows)


# ============================================================
# COVERAGE SUMMARY
# ============================================================

header("CT-TO-RATE COVERAGE SUMMARY")

total_targets = len(audit)

baseline_exists_count = int(
    audit["baseline_ct_crop_exists"].sum()
)

baseline_valid_count = int(
    audit["baseline_ct_crop_valid"].sum()
)

future_count = int(
    audit["future_ct_exists"].sum()
)

usable_count = int(
    audit["usable_for_ct_rate"].sum()
)

print("Individualized rate targets:", total_targets)

print(
    "Baseline CT crop exists:",
    baseline_exists_count
)

print(
    "Baseline CT crop valid:",
    baseline_valid_count
)

print(
    "Trajectories with future CT:",
    future_count
)

print(
    "Usable CT-to-rate trajectories:",
    usable_count
)

if total_targets > 0:
    print(
        "Usable coverage:",
        round(
            100.0 * usable_count / total_targets,
            2
        ),
        "%"
    )


# ============================================================
# PATIENT COVERAGE
# ============================================================

header("PATIENT-LEVEL CT COVERAGE")

patient_summary = []

for patient_id, group in audit.groupby(
    "patient_id",
    sort=True
):

    patient_summary.append({

        "patient_id": int(patient_id),

        "rate_targets":
            len(group),

        "baseline_ct_targets":
            int(
                group[
                    "baseline_ct_crop_valid"
                ].sum()
            ),

        "future_ct_targets":
            int(
                group[
                    "future_ct_exists"
                ].sum()
            ),

        "usable_ct_rate_targets":
            int(
                group[
                    "usable_for_ct_rate"
                ].sum()
            ),
    })


patient_summary = pd.DataFrame(
    patient_summary
)

print(
    patient_summary.to_string(
        index=False
    )
)


# ============================================================
# PATIENT-INDEPENDENT SPLIT
# ============================================================

header("PATIENT-INDEPENDENT CT-TO-RATE COVERAGE")

# Same split used in Steps 19-21 and Step 27
training_patients = [
    3, 7, 8, 12, 13, 16, 22
]

testing_patients = [
    23, 25, 30
]

print(
    "Training patients:",
    training_patients
)

print(
    "Testing patients:",
    testing_patients
)

train_usable = audit[
    audit["patient_id"].isin(training_patients)
    &
    audit["usable_for_ct_rate"]
]

test_usable = audit[
    audit["patient_id"].isin(testing_patients)
    &
    audit["usable_for_ct_rate"]
]

print()
print(
    "Training CT-to-rate samples:",
    len(train_usable)
)

print(
    "Testing CT-to-rate samples:",
    len(test_usable)
)

print(
    "Training patients actually represented:",
    sorted(
        train_usable["patient_id"].unique().tolist()
    )
)

print(
    "Testing patients actually represented:",
    sorted(
        test_usable["patient_id"].unique().tolist()
    )
)


# ============================================================
# DETAILED TEST COVERAGE
# ============================================================

header("TEST PATIENT COVERAGE")

for patient_id in testing_patients:

    group = audit[
        audit["patient_id"] == patient_id
    ]

    usable = group[
        group["usable_for_ct_rate"]
    ]

    print()
    print(
        "Patient",
        patient_id,
        "| rate targets:",
        len(group),
        "| usable CT-to-rate:",
        len(usable)
    )

    if len(usable) > 0:

        for _, row in usable.iterrows():

            print(
                "  Alias",
                row["lesion_alias"],
                "| baseline:",
                row["baseline_date"].strftime(
                    "%Y-%m-%d"
                ),
                "| rate:",
                round(
                    row["individualized_rate"],
                    6
                )
            )

    else:

        print(
            "  NO usable CT-to-rate trajectory"
        )


# ============================================================
# SAVE AUDIT
# ============================================================

header("SAVING STEP 30 RESULTS")

audit.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Saved:")
print(OUTPUT_FILE)


# ============================================================
# FINAL DECISION
# ============================================================

header("STEP 30 - FINAL DECISION")

print(
    "Total individualized rate targets:",
    total_targets
)

print(
    "Usable CT-to-rate trajectories:",
    usable_count
)

print(
    "Training CT-to-rate samples:",
    len(train_usable)
)

print(
    "Testing CT-to-rate samples:",
    len(test_usable)
)

print()

if len(test_usable) >= 3:

    print(
        "[YES] At least 3 independent test samples available"
    )

else:

    print(
        "[NO] Fewer than 3 independent test samples available"
    )


if (
    len(train_usable) >= 6
    and
    len(test_usable) >= 3
):

    print(
        "[YES] Coverage is sufficient for an exploratory "
        "patient-independent CT-to-rate experiment"
    )

else:

    print(
        "[NO] Current CT coverage is insufficient for a "
        "meaningful patient-independent CT-to-rate experiment"
    )


print()
print(
    "IMPORTANT:"
)

print(
    "This step only audits existing CT crop coverage."
)

print(
    "No crops are modified."
)

print(
    "No neural network is trained."
)

print(
    "No biological state is claimed."
)

print(
    "No clinical validation is performed."
)

print()
print("=" * 95)
print("STEP 30 COMPLETE")
print("=" * 95)