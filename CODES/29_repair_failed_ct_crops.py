import os
import glob
import shutil
import numpy as np
import pandas as pd
import nibabel as nib


# ============================================================
# STEP 29
# REPAIR FAILED CT CROP EXTRACTION
# ============================================================

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

QUALITY_FILE = os.path.join(
    ROOT,
    "25_ct_crop_quality.csv"
)

MAPPING_FILE = os.path.join(
    ROOT,
    "24_recist_lesion_mapping.csv"
)

SERIES_FILE = os.path.join(
    ROOT,
    "series.json"
)

ORIGINAL_CROP_DIR = os.path.join(
    ROOT,
    "25_lesion_ct_crops"
)

REPAIR_DIR = os.path.join(
    ROOT,
    "29_repaired_ct_crops"
)

AUDIT_FILE = os.path.join(
    ROOT,
    "29_ct_crop_repair_audit.csv"
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

    result = pd.to_datetime(
        value,
        errors="coerce",
        format="%Y-%m-%d"
    )

    if pd.notna(result):
        return result

    result = pd.to_datetime(
        value,
        errors="coerce",
        format="%Y%m%d"
    )

    return result


def safe_int(value):
    try:
        return int(float(value))
    except Exception:
        return None


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def get_label_value(row):
    possible = [
        "lesion_label_value",
        "label_value",
        "mask_label",
        "label"
    ]

    for col in possible:
        if col in row.index:
            value = safe_int(row[col])
            if value is not None:
                return value

    return None


# ============================================================
# START
# ============================================================

header("STEP 29 - REPAIRING FAILED CT CROPS")

print("Dataset root:")
print(ROOT)

print()
print("Quality file:")
print(QUALITY_FILE)

print()
print("Mapping file:")
print(MAPPING_FILE)

print()
print("Original crop directory:")
print(ORIGINAL_CROP_DIR)

print()
print("Repair output directory:")
print(REPAIR_DIR)


# ============================================================
# CHECK FILES
# ============================================================

header("CHECKING REQUIRED FILES")

required = [
    QUALITY_FILE,
    MAPPING_FILE
]

for path in required:

    if os.path.isfile(path):
        print("[YES]", os.path.basename(path))
    else:
        print("[NO]", os.path.basename(path))
        print("ERROR: Required file not found:")
        print(path)
        raise SystemExit(1)

if os.path.isdir(ORIGINAL_CROP_DIR):
    print("[YES] Original crop directory exists")
else:
    print("[NO] Original crop directory exists")
    print("ERROR: Original crop directory missing.")
    raise SystemExit(1)


# ============================================================
# CREATE REPAIR DIRECTORY
# ============================================================

header("CREATING SAFE REPAIR DIRECTORY")

os.makedirs(
    REPAIR_DIR,
    exist_ok=True
)

print(
    "Repair directory ready:"
)

print(
    REPAIR_DIR
)

print()
print(
    "IMPORTANT: Original crops will NOT be modified."
)


# ============================================================
# LOAD QUALITY FILE
# ============================================================

header("LOADING STEP 25 QUALITY DATA")

quality = pd.read_csv(
    QUALITY_FILE
)

print(
    "Quality records:",
    len(quality)
)

print(
    "Columns:"
)

print(
    list(quality.columns)
)


# ============================================================
# NORMALIZE QUALITY
# ============================================================

header("NORMALIZING QUALITY DATA")

if "lesion_label_alias" in quality.columns:
    alias_column = "lesion_label_alias"
elif "lesion_alias" in quality.columns:
    alias_column = "lesion_alias"
else:
    print(
        "ERROR: No lesion alias column found."
    )
    raise SystemExit(1)

quality["patient_norm"] = quality[
    "patient_id"
].apply(normalize_patient)

quality["alias_norm"] = quality[
    alias_column
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
    "Alias column:",
    alias_column
)


# ============================================================
# FIND FAILED RECORDS
# ============================================================

header("IDENTIFYING FAILED CROPS")

failed = quality[
    quality["status_norm"] != "valid"
].copy()

print(
    "Total records:",
    len(quality)
)

print(
    "Failed records:",
    len(failed)
)

print(
    "Successful records:",
    len(quality) - len(failed)
)


if len(failed) == 0:

    print()
    print(
        "No failed crops found."
    )

    print(
        "Nothing to repair."
    )

    raise SystemExit(0)


# ============================================================
# LOAD MAPPING
# ============================================================

header("LOADING RECIST-TO-MASK MAPPING")

mapping = pd.read_csv(
    MAPPING_FILE
)

print(
    "Mapping records:",
    len(mapping)
)

print(
    "Mapping columns:"
)

print(
    list(mapping.columns)
)


# ============================================================
# NORMALIZE MAPPING
# ============================================================

header("NORMALIZING MAPPING DATA")

mapping["patient_norm"] = mapping[
    "patient_id"
].apply(normalize_patient)

mapping["alias_norm"] = mapping[
    "lesion_label_alias"
].apply(normalize_alias)

mapping["date_norm"] = mapping[
    "study_date"
].apply(normalize_date)

if "lesion_label_value" in mapping.columns:

    mapping["label_norm"] = mapping[
        "lesion_label_value"
    ].apply(safe_int)

else:

    mapping["label_norm"] = np.nan


# ============================================================
# BUILD MAPPING LOOKUP
# ============================================================

mapping_lookup = {}

for _, row in mapping.iterrows():

    key = (
        row["patient_norm"],
        row["alias_norm"],
        row["date_norm"]
    )

    mapping_lookup.setdefault(
        key,
        []
    ).append(row)


# ============================================================
# FIND CT AND MASK FILES
# ============================================================

header("SEARCHING FOR CT AND MASK FILES")

all_nii = glob.glob(
    os.path.join(
        ROOT,
        "**",
        "*.nii.gz"
    ),
    recursive=True
)

all_nii += glob.glob(
    os.path.join(
        ROOT,
        "**",
        "*.nii"
    ),
    recursive=True
)

print(
    "NIfTI files discovered:",
    len(all_nii)
)


# ============================================================
# CLASSIFY NIFTI FILES
# ============================================================

ct_files = []
mask_files = []

for path in all_nii:

    lower = os.path.basename(path).lower()

    # Ignore generated crop files
    if (
        "25_lesion_ct_crops"
        in path
        or
        "29_repaired_ct_crops"
        in path
    ):
        continue

    # Common mask naming patterns
    if any(
        token in lower
        for token in [
            "mask",
            "seg",
            "label",
            "lesion"
        ]
    ):
        mask_files.append(path)
    else:
        ct_files.append(path)


print(
    "Potential CT files:",
    len(ct_files)
)

print(
    "Potential mask files:",
    len(mask_files)
)


# ============================================================
# INDEX USING FILE NAMES
# ============================================================

def basename_key(path):
    return os.path.basename(path).lower()


ct_by_name = {
    basename_key(p): p
    for p in ct_files
}

mask_by_name = {
    basename_key(p): p
    for p in mask_files
}


# ============================================================
# BUILD SERIES LOOKUP
# ============================================================

header("BUILDING CT / MASK SERIES LOOKUP")

series_lookup = {}

for _, row in mapping.iterrows():

    patient = row["patient_norm"]
    date = row["date_norm"]

    key = (
        patient,
        date
    )

    ct_value = ""

    mask_value = ""

    for col in [
        "ct_file",
        "image_file",
        "ct_path",
        "image_path"
    ]:

        if col in row.index:

            value = row[col]

            if not pd.isna(value):

                value = str(value).strip()

                if value:
                    ct_value = value
                    break

    for col in [
        "mask_file",
        "mask_path",
        "segmentation_file",
        "segmentation_path"
    ]:

        if col in row.index:

            value = row[col]

            if not pd.isna(value):

                value = str(value).strip()

                if value:
                    mask_value = value
                    break

    if key not in series_lookup:

        series_lookup[key] = {
            "ct": ct_value,
            "mask": mask_value
        }


# ============================================================
# FALLBACK SEARCH FUNCTION
# ============================================================

def find_file_from_value(value):

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    if os.path.isfile(value):
        return value

    candidate = os.path.join(
        ROOT,
        value
    )

    if os.path.isfile(candidate):
        return candidate

    basename = os.path.basename(
        value
    ).lower()

    for path in all_nii:

        if os.path.basename(path).lower() == basename:

            return path

    return None


# ============================================================
# MATCH CT / MASK USING MAPPING
# ============================================================

def find_ct_mask(patient, alias, date):

    candidates = mapping_lookup.get(
        (
            patient,
            alias,
            date
        ),
        []
    )

    if len(candidates) == 0:

        return None, None, None

    candidate = candidates[0]

    # Try explicit CT file
    ct_path = None

    for col in [
        "ct_file",
        "image_file",
        "ct_path",
        "image_path"
    ]:

        if col in candidate.index:

            value = candidate[col]

            if not pd.isna(value):

                ct_path = find_file_from_value(
                    value
                )

                if ct_path:
                    break

    # Try explicit mask file
    mask_path = None

    for col in [
        "mask_file",
        "mask_path",
        "segmentation_file",
        "segmentation_path"
    ]:

        if col in candidate.index:

            value = candidate[col]

            if not pd.isna(value):

                mask_path = find_file_from_value(
                    value
                )

                if mask_path:
                    break

    return (
        ct_path,
        mask_path,
        candidate
    )


# ============================================================
# CROP PARAMETERS
# ============================================================

header("DEFINING SAFE REPAIR CROP")

# Margin around lesion bounding box.
# This is deliberately moderate.
MARGIN = 32

print(
    "Crop margin:",
    MARGIN,
    "voxels"
)

print()
print(
    "The repair uses clipped intersection with "
    "the actual CT volume."
)


# ============================================================
# EXTRACT REPAIRED CROP
# ============================================================

def extract_repaired_crop(
    ct_path,
    mask_path,
    label_value
):

    ct_img = nib.load(
        ct_path
    )

    mask_img = nib.load(
        mask_path
    )

    ct = ct_img.get_fdata(
        dtype=np.float32
    )

    mask = mask_img.get_fdata()

    # --------------------------------------------------------
    # Shape check
    # --------------------------------------------------------

    if ct.ndim != 3:
        return None, "CT is not 3D"

    if mask.ndim != 3:
        return None, "Mask is not 3D"

    if ct.shape != mask.shape:

        return (
            None,
            "CT/mask shapes do not match"
        )

    # --------------------------------------------------------
    # Label check
    # --------------------------------------------------------

    lesion = (
        np.isclose(
            mask,
            label_value
        )
    )

    voxel_count = int(
        lesion.sum()
    )

    if voxel_count == 0:

        return (
            None,
            "Requested label not present in mask"
        )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    coords = np.argwhere(
        lesion
    )

    min_xyz = coords.min(
        axis=0
    )

    max_xyz = coords.max(
        axis=0
    )

    # --------------------------------------------------------
    # Expand bounding box
    # --------------------------------------------------------

    start = min_xyz - MARGIN
    end = max_xyz + MARGIN + 1

    # --------------------------------------------------------
    # CLIP TO CT VOLUME
    # --------------------------------------------------------

    start_clipped = np.maximum(
        start,
        0
    )

    end_clipped = np.minimum(
        end,
        np.array(ct.shape)
    )

    # --------------------------------------------------------
    # Verify intersection
    # --------------------------------------------------------

    if np.any(
        start_clipped >= end_clipped
    ):

        return (
            None,
            "No valid intersection with CT volume"
        )

    start_clipped = start_clipped.astype(
        int
    )

    end_clipped = end_clipped.astype(
        int
    )

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    crop = ct[
        start_clipped[0]:end_clipped[0],
        start_clipped[1]:end_clipped[1],
        start_clipped[2]:end_clipped[2]
    ]

    crop_mask = mask[
        start_clipped[0]:end_clipped[0],
        start_clipped[1]:end_clipped[1],
        start_clipped[2]:end_clipped[2]
    ]

    if crop.size == 0:

        return (
            None,
            "Empty crop"
        )

    # --------------------------------------------------------
    # Verify lesion survives crop
    # --------------------------------------------------------

    lesion_inside = np.isclose(
        crop_mask,
        label_value
    )

    lesion_voxels_inside = int(
        lesion_inside.sum()
    )

    if lesion_voxels_inside == 0:

        return (
            None,
            "Lesion disappeared after clipping"
        )

    # --------------------------------------------------------
    # Verify finite values
    # --------------------------------------------------------

    if not np.isfinite(crop).all():

        return (
            None,
            "Crop contains non-finite values"
        )

    # --------------------------------------------------------
    # Calculate fraction
    # --------------------------------------------------------

    fraction = (
        lesion_voxels_inside
        /
        crop.size
    )

    metadata = {
        "original_shape":
            str(tuple(ct.shape)),

        "crop_shape":
            str(tuple(crop.shape)),

        "original_lesion_voxels":
            voxel_count,

        "lesion_voxels_inside":
            lesion_voxels_inside,

        "crop_voxels":
            int(crop.size),

        "crop_fraction":
            float(fraction),

        "bbox_min_original":
            str(tuple(min_xyz)),

        "bbox_max_original":
            str(tuple(max_xyz)),

        "crop_start":
            str(tuple(start_clipped)),

        "crop_end":
            str(tuple(end_clipped))
    }

    return (
        (
            crop,
            crop_mask,
            metadata
        ),
        "success"
    )


# ============================================================
# PROCESS FAILED CROPS
# ============================================================

header("REPAIRING FAILED CROP RECORDS")

audit_records = []

success_count = 0
failure_count = 0

for index, (_, row) in enumerate(
    failed.iterrows(),
    start=1
):

    patient = row["patient_norm"]
    alias = row["alias_norm"]
    date = row["date_norm"]

    print(
        f"[{index}/{len(failed)}] "
        f"Patient {patient} | "
        f"{date.strftime('%Y-%m-%d') if pd.notna(date) else 'INVALID DATE'} | "
        f"Alias {alias}"
    )

    record = {
        "patient_id": patient,
        "study_date":
            date.strftime("%Y-%m-%d")
            if pd.notna(date)
            else "",
        "lesion_alias": alias,
        "original_status":
            row["status"],
        "original_reason":
            row.get(
                "reason",
                ""
            ),
        "label_value": None,
        "ct_file": "",
        "mask_file": "",
        "repair_status": "failed",
        "repair_reason": "",
        "repaired_crop_file": "",
        "original_shape": "",
        "crop_shape": "",
        "original_lesion_voxels": 0,
        "lesion_voxels_inside": 0,
        "crop_voxels": 0,
        "crop_fraction": np.nan
    }

    # --------------------------------------------------------
    # Find mapping
    # --------------------------------------------------------

    ct_path, mask_path, mapping_row = find_ct_mask(
        patient,
        alias,
        date
    )

    if mapping_row is None:

        record["repair_reason"] = (
            "No mapping record found"
        )

        audit_records.append(record)

        failure_count += 1

        continue

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label_value = None

    if "lesion_label_value" in mapping_row.index:

        label_value = safe_int(
            mapping_row["lesion_label_value"]
        )

    if label_value is None:

        label_value = get_label_value(
            row
        )

    record["label_value"] = label_value

    if label_value is None:

        record["repair_reason"] = (
            "No valid lesion label value"
        )

        audit_records.append(record)

        failure_count += 1

        continue

    # --------------------------------------------------------
    # CT
    # --------------------------------------------------------

    if ct_path is None:

        record["repair_reason"] = (
            "CT file could not be resolved"
        )

        audit_records.append(record)

        failure_count += 1

        continue

    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    if mask_path is None:

        record["repair_reason"] = (
            "Mask file could not be resolved"
        )

        audit_records.append(record)

        failure_count += 1

        continue

    record["ct_file"] = ct_path
    record["mask_file"] = mask_path

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    try:

        result, reason = extract_repaired_crop(
            ct_path,
            mask_path,
            label_value
        )

    except Exception as exc:

        result = None

        reason = (
            "Exception: "
            +
            str(exc)
        )

    if result is None:

        record["repair_reason"] = reason

        audit_records.append(record)

        failure_count += 1

        continue

    crop, crop_mask, metadata = result

    # --------------------------------------------------------
    # Save repaired crop
    # --------------------------------------------------------

    filename = (
        f"patient_{patient}_"
        f"{date.strftime('%Y%m%d')}_"
        f"lesion_{alias}_"
        f"label_{label_value}_"
        f"repaired.nii.gz"
    )

    output_path = os.path.join(
        REPAIR_DIR,
        filename
    )

    # Use original CT affine/header
    ct_img = nib.load(
        ct_path
    )

    repaired_img = nib.Nifti1Image(
        crop.astype(np.float32),
        ct_img.affine,
        ct_img.header
    )

    nib.save(
        repaired_img,
        output_path
    )

    # --------------------------------------------------------
    # Update record
    # --------------------------------------------------------

    record["repair_status"] = "success"
    record["repair_reason"] = (
        "Successfully repaired using clipped bounding box"
    )
    record["repaired_crop_file"] = (
        output_path
    )

    record["original_shape"] = (
        metadata["original_shape"]
    )

    record["crop_shape"] = (
        metadata["crop_shape"]
    )

    record["original_lesion_voxels"] = (
        metadata["original_lesion_voxels"]
    )

    record["lesion_voxels_inside"] = (
        metadata["lesion_voxels_inside"]
    )

    record["crop_voxels"] = (
        metadata["crop_voxels"]
    )

    record["crop_fraction"] = (
        metadata["crop_fraction"]
    )

    audit_records.append(record)

    success_count += 1


# ============================================================
# SAVE AUDIT
# ============================================================

header("SAVING STEP 29 RESULTS")

audit = pd.DataFrame(
    audit_records
)

audit.to_csv(
    AUDIT_FILE,
    index=False
)

print(
    "Saved:",
    AUDIT_FILE
)


# ============================================================
# SUMMARY
# ============================================================

header("STEP 29 - REPAIR SUMMARY")

print(
    "Failed crops examined:",
    len(failed)
)

print(
    "Successfully repaired:",
    success_count
)

print(
    "Still failed:",
    failure_count
)

if len(failed) > 0:

    print(
        "Repair success rate:",
        round(
            success_count /
            len(failed) *
            100,
            2
        ),
        "%"
    )


# ============================================================
# REPAIRED BY PATIENT
# ============================================================

header("REPAIRED CROPS BY PATIENT")

if success_count > 0:

    repaired = audit[
        audit["repair_status"] == "success"
    ]

    patient_counts = (
        repaired
        .groupby("patient_id")
        .size()
        .reset_index(
            name="repaired_crops"
        )
    )

    print(
        patient_counts.to_string(
            index=False
        )
    )

else:

    print(
        "No crops were repaired."
    )


# ============================================================
# CRITICAL TEST PATIENTS
# ============================================================

header("CRITICAL PATIENTS FOR STEP 27")

critical_patients = [
    23,
    25,
    30
]

for patient in critical_patients:

    subset = audit[
        audit["patient_id"] == patient
    ]

    print()
    print(
        "Patient",
        patient
    )

    if len(subset) == 0:

        print(
            "  No failed crop records."
        )

        continue

    for _, row in subset.iterrows():

        print(
            f"  Alias {row['lesion_alias']} | "
            f"{row['study_date']} | "
            f"{row['repair_status']} | "
            f"{row['repair_reason']}"
        )


# ============================================================
# FINAL DECISION
# ============================================================

header("STEP 29 - FINAL DECISION")

print(
    "Original successful crops were NOT modified."
)

print(
    "Repaired crops are stored separately."
)

print()

if success_count > 0:

    print(
        "[YES] At least one failed crop was repaired."
    )

else:

    print(
        "[NO] No failed crop could be repaired."
    )

print()

if success_count > 0:

    print(
        "DO NOT TRAIN THE CNN YET."
    )

    print(
        "Run the next coverage audit using the repaired "
        "crop directory before training."
    )

else:

    print(
        "The current crop extraction limitations remain."
    )

print()
print(
    "No biological interpretation is made."
)

print(
    "No clinical validation is claimed."
)

print()
print("=" * 95)
print("STEP 29 COMPLETE")
print("=" * 95)