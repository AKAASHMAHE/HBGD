import os
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

FEATURE_FILE = os.path.join(ROOT, "9_hbgd_features.csv")
QUALITY_FILE = os.path.join(ROOT, "25_ct_crop_quality.csv")

OUTPUT_FILE = os.path.join(
    ROOT,
    "31_ct_rate_linkage_reconciliation.csv"
)


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def normalize_date(value):
    return pd.to_datetime(
        value,
        errors="coerce"
    )


# ============================================================
# START
# ============================================================

header("STEP 31 - CT TO HBGD RATE LINKAGE RECONCILIATION")

print("Root:")
print(ROOT)

print()
print("Feature file:")
print(FEATURE_FILE)

print()
print("Quality file:")
print(QUALITY_FILE)


# ============================================================
# LOAD FILES
# ============================================================

header("LOADING DATA")

features = pd.read_csv(FEATURE_FILE)
quality = pd.read_csv(QUALITY_FILE)

print("Feature records:", len(features))
print("Quality records:", len(quality))


# ============================================================
# IDENTIFY ALIAS COLUMNS
# ============================================================

header("IDENTIFYING IDENTIFIER COLUMNS")

feature_alias = None
quality_alias = None

for c in ["lesion_label_alias", "lesion_alias"]:
    if c in features.columns:
        feature_alias = c
        break

for c in ["lesion_label_alias", "lesion_alias"]:
    if c in quality.columns:
        quality_alias = c
        break

print("Feature alias column:", feature_alias)
print("Quality alias column:", quality_alias)

if feature_alias is None or quality_alias is None:
    raise SystemExit(
        "ERROR: Could not identify lesion alias columns."
    )


# ============================================================
# NORMALIZE FEATURES
# ============================================================

header("NORMALIZING HBGD FEATURES")

features["patient_id"] = pd.to_numeric(
    features["patient_id"],
    errors="coerce"
)

features["study_date_raw"] = (
    features["study_date"].astype(str)
)

features["study_date"] = pd.to_datetime(
    features["study_date"],
    errors="coerce"
)

features["alias"] = (
    features[feature_alias]
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
# BUILD RATE TARGETS
# ============================================================

header("BUILDING INDIVIDUALIZED RATE TARGETS")

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

    r = np.log(
        d1 / d0
    ) / delta_days

    targets.append({

        "patient_id": int(patient_id),

        "alias": alias,

        "baseline_date": t0,

        "second_date": t1,

        "d0": d0,

        "d1": d1,

        "delta_days": delta_days,

        "rate": r,

        "trajectory_length": len(group)
    })


targets = pd.DataFrame(targets)

print(
    "Rate targets:",
    len(targets)
)

print(
    "Patients:",
    targets["patient_id"].nunique()
)


# ============================================================
# NORMALIZE QUALITY
# ============================================================

header("NORMALIZING CT CROP QUALITY DATA")

quality["patient_id"] = pd.to_numeric(
    quality["patient_id"],
    errors="coerce"
)

quality["study_date_raw"] = (
    quality["study_date"].astype(str)
)

quality["study_date"] = pd.to_datetime(
    quality["study_date"],
    errors="coerce"
)

quality["alias"] = (
    quality[quality_alias]
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

quality = quality.dropna(
    subset=[
        "patient_id",
        "study_date"
    ]
).copy()

quality["patient_id"] = (
    quality["patient_id"].astype(int)
)


# ============================================================
# SHOW ACTUAL SUCCESSFUL RECORDS
# ============================================================

header("ACTUAL SUCCESSFUL CT CROP RECORDS")

successful = quality[
    quality["status_normalized"].isin(
        ["success", "successful", "valid"]
    )
].copy()

print(
    "Successful records:",
    len(successful)
)

print()

display_columns = [
    "patient_id",
    "study_date_raw",
    "study_date",
    "alias",
    "crop_file"
]

display_columns = [
    c for c in display_columns
    if c in successful.columns
]

print(
    successful[
        display_columns
    ].head(50).to_string(index=False)
)


# ============================================================
# EXACT KEY MATCH
# ============================================================

header("EXACT PATIENT + ALIAS + DATE MATCH")

quality_keys = set()

for _, row in successful.iterrows():

    key = (
        int(row["patient_id"]),
        row["alias"],
        row["study_date"]
    )

    quality_keys.add(key)


exact_matches = 0

for _, row in targets.iterrows():

    key = (
        int(row["patient_id"]),
        row["alias"],
        row["baseline_date"]
    )

    if key in quality_keys:
        exact_matches += 1


print(
    "Rate targets:",
    len(targets)
)

print(
    "Exact baseline CT matches:",
    exact_matches
)


# ============================================================
# PATIENT + ALIAS MATCH WITHOUT DATE
# ============================================================

header("PATIENT + ALIAS MATCH WITHOUT DATE")

trajectory_matches = 0

for _, row in targets.iterrows():

    matches = successful[
        (successful["patient_id"] == row["patient_id"])
        &
        (successful["alias"] == row["alias"])
    ]

    if len(matches) > 0:
        trajectory_matches += 1

print(
    "Rate trajectories with any CT crop:",
    trajectory_matches
)


# ============================================================
# SHOW WHY DATE MATCH FAILS
# ============================================================

header("DATE MATCH DIAGNOSTIC")

shown = 0

for _, row in targets.iterrows():

    matches = successful[
        (successful["patient_id"] == row["patient_id"])
        &
        (successful["alias"] == row["alias"])
    ].sort_values("study_date")

    if len(matches) == 0:
        continue

    print()
    print(
        "Patient:",
        row["patient_id"],
        "| Alias:",
        row["alias"]
    )

    print(
        "HBGD baseline:",
        row["baseline_date"].strftime("%Y-%m-%d")
    )

    print(
        "HBGD second:",
        row["second_date"].strftime("%Y-%m-%d")
    )

    print(
        "CT crop dates:",
        [
            d.strftime("%Y-%m-%d")
            for d in matches["study_date"]
        ]
    )

    shown += 1

    if shown >= 15:
        break


# ============================================================
# FUZZY CHRONOLOGICAL MATCH
# ============================================================

header("CHRONOLOGICAL MATCH TEST")

print(
    "This test does NOT modify data."
)

print(
    "It checks whether the CT crop dates correspond "
    "to the first/second HBGD observations."
)

chrono_matches = 0

reconciliation_rows = []

for _, target in targets.iterrows():

    patient_id = target["patient_id"]
    alias = target["alias"]

    matches = successful[
        (successful["patient_id"] == patient_id)
        &
        (successful["alias"] == alias)
    ].sort_values("study_date")

    ct_dates = matches[
        "study_date"
    ].tolist()

    baseline_date = target["baseline_date"]
    second_date = target["second_date"]

    baseline_exact = (
        baseline_date in ct_dates
    )

    second_exact = (
        second_date in ct_dates
    )

    # --------------------------------------------------------
    # Closest CT dates
    # --------------------------------------------------------

    closest_baseline = None
    closest_second = None

    if len(ct_dates) > 0:

        closest_baseline = min(
            ct_dates,
            key=lambda x: abs(
                (x - baseline_date).days
            )
        )

        closest_second = min(
            ct_dates,
            key=lambda x: abs(
                (x - second_date).days
            )
        )

    if (
        baseline_exact
        and
        second_exact
    ):
        chrono_matches += 1

    reconciliation_rows.append({

        "patient_id":
            patient_id,

        "alias":
            alias,

        "hbgd_baseline_date":
            baseline_date,

        "hbgd_second_date":
            second_date,

        "hbgd_rate":
            target["rate"],

        "ct_observation_count":
            len(ct_dates),

        "baseline_ct_exact":
            baseline_exact,

        "second_ct_exact":
            second_exact,

        "closest_ct_to_baseline":
            closest_baseline,

        "closest_ct_to_second":
            closest_second,

        "baseline_day_difference":
            (
                None
                if closest_baseline is None
                else
                abs(
                    (
                        closest_baseline
                        -
                        baseline_date
                    ).days
                )
            ),

        "second_day_difference":
            (
                None
                if closest_second is None
                else
                abs(
                    (
                        closest_second
                        -
                        second_date
                    ).days
                )
            )
    })


reconciliation = pd.DataFrame(
    reconciliation_rows
)

print(
    "Trajectories with exact CT baseline + second observation:",
    chrono_matches
)


# ============================================================
# PATIENT INDEPENDENT COVERAGE
# ============================================================

header("PATIENT-INDEPENDENT COVERAGE")

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

train = reconciliation[
    reconciliation["patient_id"].isin(
        training_patients
    )
]

test = reconciliation[
    reconciliation["patient_id"].isin(
        testing_patients
    )
]

train_exact = train[
    (train["baseline_ct_exact"])
    &
    (train["second_ct_exact"])
]

test_exact = test[
    (test["baseline_ct_exact"])
    &
    (test["second_ct_exact"])
]

print()
print(
    "Training rate trajectories:",
    len(train)
)

print(
    "Training exact CT trajectories:",
    len(train_exact)
)

print(
    "Testing rate trajectories:",
    len(test)
)

print(
    "Testing exact CT trajectories:",
    len(test_exact)
)


# ============================================================
# SAVE
# ============================================================

header("SAVING STEP 31 RESULTS")

reconciliation.to_csv(
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

header("STEP 31 - FINAL DECISION")

print(
    "Rate targets:",
    len(targets)
)

print(
    "Successful CT crop records:",
    len(successful)
)

print(
    "Exact baseline CT matches:",
    exact_matches
)

print(
    "Rate trajectories with any CT:",
    trajectory_matches
)

print(
    "Exact baseline + second CT trajectories:",
    chrono_matches
)

print(
    "Patient-independent training trajectories:",
    len(train_exact)
)

print(
    "Patient-independent testing trajectories:",
    len(test_exact)
)

print()

if exact_matches > 0:

    print(
        "[YES] CT crops can be linked to HBGD rate targets."
    )

else:

    print(
        "[NO] Exact date linkage still fails."
    )


if trajectory_matches > 0:

    print(
        "[YES] CT crops exist for some HBGD trajectories."
    )

else:

    print(
        "[NO] No CT crops share patient + lesion identity "
        "with the rate trajectories."
    )


if len(test_exact) >= 3:

    print(
        "[YES] Sufficient independent test trajectories "
        "for exploratory CT-to-rate testing."
    )

else:

    print(
        "[NO] Fewer than 3 independent test trajectories."
    )

print()
print(
    "NO DATASET FILES WERE MODIFIED."
)

print(
    "NO CNN WAS TRAINED."
)

print(
    "NO BIOLOGICAL STATES WERE CLAIMED."
)

print()
print("=" * 100)
print("STEP 31 COMPLETE")
print("=" * 100)