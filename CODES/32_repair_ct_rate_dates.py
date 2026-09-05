import os
import re
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

FEATURE_FILE = os.path.join(
    ROOT, "9_hbgd_features.csv"
)

QUALITY_FILE = os.path.join(
    ROOT, "25_ct_crop_quality.csv"
)

OUTPUT_FILE = os.path.join(
    ROOT, "32_ct_rate_repaired_coverage.csv"
)


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def parse_yyyymmdd(value):
    """
    Safely convert YYYYMMDD into a real calendar date.
    """
    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()

    # Remove decimal representation such as 20210526.0
    if text.endswith(".0"):
        text = text[:-2]

    # Keep only digits
    digits = re.sub(r"\D", "", text)

    if len(digits) == 8:
        try:
            return pd.to_datetime(
                digits,
                format="%Y%m%d",
                errors="coerce"
            )
        except Exception:
            return pd.NaT

    return pd.NaT


def extract_date_from_crop_path(path):
    """
    Extract YYYYMMDD from crop filename.

    Example:
    patient_007_20210526_lesion_A_label_5.nii.gz

    -> 2021-05-26
    """

    if pd.isna(path):
        return pd.NaT

    text = os.path.basename(str(path))

    match = re.search(
        r"_(\d{8})_lesion_",
        text
    )

    if match:
        return parse_yyyymmdd(
            match.group(1)
        )

    # fallback: any 8-digit date
    matches = re.findall(
        r"(?<!\d)(20\d{6})(?!\d)",
        text
    )

    if matches:
        return parse_yyyymmdd(
            matches[0]
        )

    return pd.NaT


# ============================================================
# START
# ============================================================

header(
    "STEP 32 - REPAIRING CT CROP DATE LINKAGE"
)

print("Dataset root:")
print(ROOT)

print()
print("Feature file:")
print(FEATURE_FILE)

print()
print("Quality file:")
print(QUALITY_FILE)

print()
print("IMPORTANT:")
print("Original dataset will NOT be modified.")
print("Original CT crops will NOT be modified.")
print("This step only creates a corrected linkage table.")


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

print(
    "Feature records:",
    len(features)
)

print(
    "Quality records:",
    len(quality)
)


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

header("IDENTIFYING REQUIRED COLUMNS")

feature_alias_col = None
quality_alias_col = None

for col in [
    "lesion_label_alias",
    "lesion_alias"
]:
    if col in features.columns:
        feature_alias_col = col
        break

for col in [
    "lesion_label_alias",
    "lesion_alias"
]:
    if col in quality.columns:
        quality_alias_col = col
        break

print(
    "Feature alias column:",
    feature_alias_col
)

print(
    "Quality alias column:",
    quality_alias_col
)

if feature_alias_col is None:
    raise SystemExit(
        "ERROR: Feature lesion alias column not found."
    )

if quality_alias_col is None:
    raise SystemExit(
        "ERROR: Quality lesion alias column not found."
    )


# ============================================================
# NORMALIZE HBGD FEATURES
# ============================================================

header(
    "NORMALIZING HBGD FEATURE DATA"
)

features["patient_id"] = pd.to_numeric(
    features["patient_id"],
    errors="coerce"
)

features["study_date"] = pd.to_datetime(
    features["study_date"],
    errors="coerce"
)

features["alias"] = (
    features[feature_alias_col]
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
        "diameter_mm"
    ]
).copy()

features["patient_id"] = (
    features["patient_id"].astype(int)
)

features = features.sort_values(
    [
        "patient_id",
        "alias",
        "study_date"
    ]
)

print(
    "Valid feature records:",
    len(features)
)


# ============================================================
# BUILD HBGD RATE TARGETS
# ============================================================

header(
    "BUILDING INDIVIDUALIZED HBGD RATE TARGETS"
)

targets = []

for (patient_id, alias), group in features.groupby(
    ["patient_id", "alias"]
):

    group = group.sort_values(
        "study_date"
    ).reset_index(drop=True)

    if len(group) < 2:
        continue

    first = group.iloc[0]
    second = group.iloc[1]

    d0 = float(first["diameter_mm"])
    d1 = float(second["diameter_mm"])

    t0 = first["study_date"]
    t1 = second["study_date"]

    delta_days = (
        t1 - t0
    ).days

    if delta_days <= 0:
        continue

    if d0 <= 0 or d1 <= 0:
        continue

    rate = np.log(
        d1 / d0
    ) / delta_days

    targets.append({

        "patient_id":
            int(patient_id),

        "alias":
            alias,

        "baseline_date":
            t0,

        "second_date":
            t1,

        "d0":
            d0,

        "d1":
            d1,

        "delta_days":
            delta_days,

        "individualized_rate":
            rate,

        "trajectory_observations":
            len(group)
    })


targets = pd.DataFrame(
    targets
)

print(
    "Individualized rate targets:",
    len(targets)
)

print(
    "Patients:",
    targets["patient_id"].nunique()
)


# ============================================================
# NORMALIZE CT QUALITY DATA
# ============================================================

header(
    "NORMALIZING CT CROP DATA"
)

quality["patient_id"] = pd.to_numeric(
    quality["patient_id"],
    errors="coerce"
)

quality["alias"] = (
    quality[quality_alias_col]
    .astype(str)
    .str.strip()
    .str.upper()
)

quality["status_normalized"] = (
    quality["status"]
    .astype(str)
    .str.strip()
    .str.lower()
)

quality["crop_file"] = (
    quality["crop_file"]
    .astype(str)
    .str.strip()
)

quality["original_study_date"] = (
    quality["study_date"]
)

# IMPORTANT:
# Do NOT trust the previous pandas conversion.
# Extract the real date from the YYYYMMDD identifier.
quality["corrected_study_date"] = (
    quality["study_date"]
    .apply(parse_yyyymmdd)
)

# If study_date was already YYYY-MM-DD,
# preserve it when the direct parser succeeds/fails.
fallback_dates = pd.to_datetime(
    quality["study_date"],
    errors="coerce"
)

quality["corrected_study_date"] = (
    quality["corrected_study_date"]
    .fillna(fallback_dates)
)

# Crop filename is the strongest source.
filename_dates = (
    quality["crop_file"]
    .apply(extract_date_from_crop_path)
)

quality["corrected_study_date"] = (
    filename_dates
    .fillna(
        quality["corrected_study_date"]
    )
)

quality = quality.dropna(
    subset=[
        "patient_id",
        "corrected_study_date"
    ]
).copy()

quality["patient_id"] = (
    quality["patient_id"].astype(int)
)

print(
    "Valid quality records:",
    len(quality)
)


# ============================================================
# DATE REPAIR CHECK
# ============================================================

header(
    "DATE REPAIR VERIFICATION"
)

print(
    quality[
        [
            "patient_id",
            "original_study_date",
            "corrected_study_date",
            "alias"
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# SUCCESSFUL CT CROPS
# ============================================================

header(
    "FILTERING SUCCESSFUL CT CROPS"
)

successful = quality[
    quality["status_normalized"].isin(
        [
            "success",
            "successful",
            "valid"
        ]
    )
].copy()

successful = successful[
    successful["crop_file"].apply(
        lambda p: os.path.exists(str(p))
    )
].copy()

print(
    "Physically existing successful crops:",
    len(successful)
)


# ============================================================
# BUILD CT LOOKUP
# ============================================================

header(
    "BUILDING CORRECTED CT LOOKUP"
)

successful["ct_key"] = list(
    zip(
        successful["patient_id"],
        successful["alias"],
        successful["corrected_study_date"]
    )
)

# Remove duplicate keys safely.
successful_unique = (
    successful
    .drop_duplicates(
        subset=["ct_key"],
        keep="first"
    )
    .copy()
)

print(
    "Unique CT keys:",
    len(successful_unique)
)


# ============================================================
# MATCH RATE TARGETS
# ============================================================

header(
    "MATCHING HBGD RATE TARGETS TO CT CROPS"
)

rows = []

for _, target in targets.iterrows():

    patient_id = int(
        target["patient_id"]
    )

    alias = target["alias"]

    baseline_date = (
        target["baseline_date"]
    )

    second_date = (
        target["second_date"]
    )

    baseline_match = successful_unique[
        (successful_unique["patient_id"] == patient_id)
        &
        (successful_unique["alias"] == alias)
        &
        (
            successful_unique["corrected_study_date"]
            == baseline_date
        )
    ]

    second_match = successful_unique[
        (successful_unique["patient_id"] == patient_id)
        &
        (successful_unique["alias"] == alias)
        &
        (
            successful_unique["corrected_study_date"]
            == second_date
        )
    ]

    baseline_exists = (
        len(baseline_match) > 0
    )

    second_exists = (
        len(second_match) > 0
    )

    baseline_path = None

    if baseline_exists:
        baseline_path = (
            baseline_match.iloc[0]["crop_file"]
        )

    second_path = None

    if second_exists:
        second_path = (
            second_match.iloc[0]["crop_file"]
        )

    rows.append({

        "patient_id":
            patient_id,

        "alias":
            alias,

        "baseline_date":
            baseline_date,

        "second_date":
            second_date,

        "individualized_rate":
            target["individualized_rate"],

        "delta_days":
            target["delta_days"],

        "baseline_ct_exists":
            baseline_exists,

        "second_ct_exists":
            second_exists,

        "usable_ct_rate":
            baseline_exists and second_exists,

        "baseline_ct_file":
            baseline_path,

        "second_ct_file":
            second_path
    })


coverage = pd.DataFrame(
    rows
)


# ============================================================
# COVERAGE SUMMARY
# ============================================================

header(
    "CORRECTED CT-TO-RATE COVERAGE SUMMARY"
)

total_targets = len(
    coverage
)

baseline_count = int(
    coverage[
        "baseline_ct_exists"
    ].sum()
)

second_count = int(
    coverage[
        "second_ct_exists"
    ].sum()
)

usable_count = int(
    coverage[
        "usable_ct_rate"
    ].sum()
)

print(
    "Individualized rate targets:",
    total_targets
)

print(
    "Baseline CT crops:",
    baseline_count
)

print(
    "Second-observation CT crops:",
    second_count
)

print(
    "Usable CT-to-rate trajectories:",
    usable_count
)

if total_targets > 0:

    print(
        "Usable coverage:",
        round(
            100 * usable_count / total_targets,
            2
        ),
        "%"
    )


# ============================================================
# SHOW MATCHED TRAJECTORIES
# ============================================================

header(
    "MATCHED CT-TO-RATE TRAJECTORIES"
)

matched = coverage[
    coverage["usable_ct_rate"]
].copy()

if len(matched) > 0:

    print(
        matched[
            [
                "patient_id",
                "alias",
                "baseline_date",
                "second_date",
                "individualized_rate"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "NO fully matched trajectories."
    )


# ============================================================
# PATIENT-INDEPENDENT SPLIT
# ============================================================

header(
    "PATIENT-INDEPENDENT COVERAGE"
)

training_patients = [
    3, 7, 8, 12, 13, 16, 22
]

testing_patients = [
    23, 25, 30
]

train = matched[
    matched["patient_id"].isin(
        training_patients
    )
]

test = matched[
    matched["patient_id"].isin(
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
    "Training CT-to-rate trajectories:",
    len(train)
)

print(
    "Testing CT-to-rate trajectories:",
    len(test)
)

print(
    "Training patients actually represented:",
    sorted(
        train["patient_id"].unique().tolist()
    )
)

print(
    "Testing patients actually represented:",
    sorted(
        test["patient_id"].unique().tolist()
    )
)


# ============================================================
# TEST PATIENT DETAILS
# ============================================================

header(
    "TEST PATIENT DETAILS"
)

for patient_id in testing_patients:

    patient_rows = coverage[
        coverage["patient_id"] == patient_id
    ]

    usable = patient_rows[
        patient_rows["usable_ct_rate"]
    ]

    print()
    print(
        "Patient",
        patient_id,
        "| rate targets:",
        len(patient_rows),
        "| usable:",
        len(usable)
    )

    if len(usable) > 0:

        for _, row in usable.iterrows():

            print(
                "  Alias",
                row["alias"],
                "|",
                row["baseline_date"].strftime("%Y-%m-%d"),
                "->",
                row["second_date"].strftime("%Y-%m-%d")
            )

    else:

        print(
            "  NO usable CT-to-rate trajectory"
        )


# ============================================================
# SAVE
# ============================================================

header(
    "SAVING STEP 32 RESULTS"
)

coverage.to_csv(
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
    "STEP 32 - FINAL DECISION"
)

print(
    "Total rate targets:",
    total_targets
)

print(
    "Baseline CT matches:",
    baseline_count
)

print(
    "Second CT matches:",
    second_count
)

print(
    "Usable CT-to-rate trajectories:",
    usable_count
)

print(
    "Training trajectories:",
    len(train)
)

print(
    "Testing trajectories:",
    len(test)
)

print()

if baseline_count > 0:
    print(
        "[YES] Corrected date parsing recovers baseline CT matches."
    )
else:
    print(
        "[NO] No baseline CT matches recovered."
    )

if usable_count > 0:
    print(
        "[YES] CT-to-HBGD rate linkage is possible."
    )
else:
    print(
        "[NO] No complete CT-to-rate trajectories."
    )

if len(test) >= 3:
    print(
        "[YES] At least 3 independent test trajectories available."
    )
else:
    print(
        "[NO] Fewer than 3 independent test trajectories."
    )

print()
print(
    "NO ORIGINAL DATA WAS MODIFIED."
)

print(
    "NO CT CROPS WERE MODIFIED."
)

print(
    "NO CNN WAS TRAINED."
)

print(
    "NO BIOLOGICAL STATE WAS CLAIMED."
)

print()
print("=" * 100)
print("STEP 32 COMPLETE")
print("=" * 100)