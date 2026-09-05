import os
import json
import glob
import numpy as np
import pandas as pd
import nibabel as nib


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

RECIST_FILE = os.path.join(BASE_DIR, "recist_measurements.csv")
SERIES_FILE = os.path.join(BASE_DIR, "series.json")

OUTPUT_MAPPING = os.path.join(
    BASE_DIR, "24_recist_lesion_mapping.csv"
)

OUTPUT_CROPS = os.path.join(
    BASE_DIR, "24_verified_lesion_crops.csv"
)

CROP_DIR = os.path.join(
    BASE_DIR, "24_lesion_crops"
)


# ============================================================
# HELPERS
# ============================================================

def header(title):
    print()
    print("=" * 95)
    print(title)
    print("=" * 95)


def normalize_filename(value):
    """
    Normalize a filename for reliable comparison.
    """
    if pd.isna(value):
        return ""

    value = str(value).strip()
    value = value.replace("\\", "/")

    return os.path.basename(value)


def find_nifti_files():
    """
    Find CT and mask NIfTI files.
    """

    all_files = glob.glob(
        os.path.join(BASE_DIR, "**", "*.nii.gz"),
        recursive=True
    )

    ct_files = []
    mask_files = []

    for path in all_files:

        normalized = path.replace("\\", "/").lower()

        if "/masks/" in normalized:
            mask_files.append(path)

        elif "/images/" in normalized:
            ct_files.append(path)

    return ct_files, mask_files


def create_filename_index(files):
    """
    Map basename -> actual path.
    """

    index = {}

    for path in files:
        name = os.path.basename(path)

        if name not in index:
            index[name] = path

    return index


def load_nifti(path):
    """
    Load NIfTI safely.
    """

    img = nib.load(path)

    data = np.asanyarray(img.dataobj)

    return img, data


def get_label_bbox(mask_data, label_value):
    """
    Find 3D bounding box of a specific segmentation label.

    Returns:
        xmin, xmax,
        ymin, ymax,
        zmin, zmax,
        voxel_count
    """

    binary = mask_data == label_value

    coordinates = np.argwhere(binary)

    if coordinates.size == 0:
        return None

    zmin = int(coordinates[:, 0].min())
    zmax = int(coordinates[:, 0].max())

    ymin = int(coordinates[:, 1].min())
    ymax = int(coordinates[:, 1].max())

    xmin = int(coordinates[:, 2].min())
    xmax = int(coordinates[:, 2].max())

    voxel_count = int(coordinates.shape[0])

    return (
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        voxel_count
    )


def expand_bbox(
    bbox,
    shape,
    margin_xy=20,
    margin_z=5
):
    """
    Expand lesion bounding box safely.
    """

    xmin, xmax, ymin, ymax, zmin, zmax, _ = bbox

    max_z = shape[0] - 1
    max_y = shape[1] - 1
    max_x = shape[2] - 1

    xmin = max(0, xmin - margin_xy)
    xmax = min(max_x, xmax + margin_xy)

    ymin = max(0, ymin - margin_xy)
    ymax = min(max_y, ymax + margin_xy)

    zmin = max(0, zmin - margin_z)
    zmax = min(max_z, zmax + margin_z)

    return (
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax
    )


# ============================================================
# START
# ============================================================

header("STEP 24 - RECIST TO CT LESION MAPPING")


# ============================================================
# LOAD RECIST
# ============================================================

header("LOADING RECIST DATA")

if not os.path.exists(RECIST_FILE):
    raise FileNotFoundError(
        "recist_measurements.csv not found:\n"
        + RECIST_FILE
    )

recist = pd.read_csv(RECIST_FILE)

print("RECIST records:", len(recist))

required_recist = [
    "patient_id",
    "study_date",
    "filename",
    "lesion_label_value",
    "lesion_label_alias",
    "recist_measurement_mm"
]

missing = [
    c for c in required_recist
    if c not in recist.columns
]

if missing:
    raise ValueError(
        "Missing RECIST columns:\n"
        + str(missing)
    )

recist["filename_norm"] = recist["filename"].apply(
    normalize_filename
)

recist["label_value_numeric"] = pd.to_numeric(
    recist["lesion_label_value"],
    errors="coerce"
)

print("RECIST records with valid label values:",
      recist["label_value_numeric"].notna().sum())


# ============================================================
# LOAD SERIES
# ============================================================

header("LOADING SERIES METADATA")

if not os.path.exists(SERIES_FILE):
    raise FileNotFoundError(
        "series.json not found:\n"
        + SERIES_FILE
    )

with open(SERIES_FILE, "r", encoding="utf-8") as f:
    series_data = json.load(f)

series_df = pd.DataFrame(series_data)

print("Series records:", len(series_df))


# ============================================================
# FIND FILES
# ============================================================

header("SEARCHING FOR CT AND MASK FILES")

ct_files, mask_files = find_nifti_files()

print("CT image files:", len(ct_files))
print("Mask files:", len(mask_files))

ct_index = create_filename_index(ct_files)
mask_index = create_filename_index(mask_files)

print("Indexed CT files:", len(ct_index))
print("Indexed masks:", len(mask_index))


# ============================================================
# SERIES LOOKUP
# ============================================================

header("BUILDING SERIES LOOKUP")

series_lookup = {}

for _, row in series_df.iterrows():

    uuid = str(row.get("uuid", "")).strip()

    if uuid:
        series_lookup[uuid] = row.to_dict()

print("Series lookup records:", len(series_lookup))


# ============================================================
# RECIST -> MASK -> CT
# ============================================================

header("MAPPING RECIST LESIONS TO CT AND MASKS")

mapping_rows = []
crop_rows = []

successful = 0
missing_ct = 0
missing_mask = 0
label_missing = 0
shape_mismatch = 0
errors = 0


for i, row in recist.iterrows():

    filename = row["filename_norm"]
    label_value = row["label_value_numeric"]

    patient_id = row["patient_id"]
    study_date = row["study_date"]
    alias = row["lesion_label_alias"]
    recist_mm = row["recist_measurement_mm"]

    result = {
        "patient_id": patient_id,
        "study_date": study_date,
        "filename": filename,
        "lesion_label_alias": alias,
        "lesion_label_value": label_value,
        "recist_measurement_mm": recist_mm,

        "ct_found": False,
        "mask_found": False,
        "label_found": False,
        "shape_match": False,

        "voxel_count": np.nan,

        "bbox_xmin": np.nan,
        "bbox_xmax": np.nan,
        "bbox_ymin": np.nan,
        "bbox_ymax": np.nan,
        "bbox_zmin": np.nan,
        "bbox_zmax": np.nan,

        "crop_xmin": np.nan,
        "crop_xmax": np.nan,
        "crop_ymin": np.nan,
        "crop_ymax": np.nan,
        "crop_zmin": np.nan,
        "crop_zmax": np.nan,

        "crop_saved": False,
        "status": ""
    }

    # --------------------------------------------------------
    # CT
    # --------------------------------------------------------

    ct_path = ct_index.get(filename)

    if ct_path is None:

        result["status"] = "CT_NOT_FOUND"

        missing_ct += 1

        mapping_rows.append(result)

        continue

    result["ct_found"] = True


    # --------------------------------------------------------
    # MASK
    # --------------------------------------------------------

    mask_path = mask_index.get(filename)

    if mask_path is None:

        result["status"] = "MASK_NOT_FOUND"

        missing_mask += 1

        mapping_rows.append(result)

        continue

    result["mask_found"] = True


    # --------------------------------------------------------
    # LOAD MASK
    # --------------------------------------------------------

    try:

        mask_img = nib.load(mask_path)

        mask_data = np.asanyarray(
            mask_img.dataobj
        )

    except Exception as e:

        result["status"] = (
            "MASK_LOAD_ERROR: " + str(e)
        )

        errors += 1

        mapping_rows.append(result)

        continue


    # --------------------------------------------------------
    # LOAD CT HEADER
    # --------------------------------------------------------

    try:

        ct_img = nib.load(ct_path)

        ct_shape = ct_img.shape
        mask_shape = mask_data.shape

    except Exception as e:

        result["status"] = (
            "CT_LOAD_ERROR: " + str(e)
        )

        errors += 1

        mapping_rows.append(result)

        continue


    # --------------------------------------------------------
    # SHAPE CHECK
    # --------------------------------------------------------

    if tuple(ct_shape) != tuple(mask_shape):

        result["status"] = "SHAPE_MISMATCH"

        shape_mismatch += 1

        mapping_rows.append(result)

        continue

    result["shape_match"] = True


    # --------------------------------------------------------
    # LABEL CHECK
    # --------------------------------------------------------

    if pd.isna(label_value):

        result["status"] = "INVALID_LABEL"

        label_missing += 1

        mapping_rows.append(result)

        continue

    label_value = int(label_value)

    bbox = get_label_bbox(
        mask_data,
        label_value
    )

    if bbox is None:

        result["status"] = "LABEL_NOT_FOUND_IN_MASK"

        label_missing += 1

        mapping_rows.append(result)

        continue


    # --------------------------------------------------------
    # BOUNDING BOX
    # --------------------------------------------------------

    (
        xmin,
        xmax,
        ymin,
        ymax,
        zmin,
        zmax,
        voxel_count
    ) = bbox

    result["label_found"] = True
    result["voxel_count"] = voxel_count

    result["bbox_xmin"] = xmin
    result["bbox_xmax"] = xmax
    result["bbox_ymin"] = ymin
    result["bbox_ymax"] = ymax
    result["bbox_zmin"] = zmin
    result["bbox_zmax"] = zmax


    # --------------------------------------------------------
    # EXPANDED CROP
    # --------------------------------------------------------

    crop_bbox = expand_bbox(
        bbox,
        mask_data.shape,
        margin_xy=20,
        margin_z=5
    )

    (
        cxmin,
        cxmax,
        cymin,
        cymax,
        czmin,
        czmax
    ) = crop_bbox

    result["crop_xmin"] = cxmin
    result["crop_xmax"] = cxmax
    result["crop_ymin"] = cymin
    result["crop_ymax"] = cymax
    result["crop_zmin"] = czmin
    result["crop_zmax"] = czmax


    # --------------------------------------------------------
    # SAVE METADATA ONLY
    # --------------------------------------------------------

    # We deliberately do NOT save all CT crops yet.
    # First we establish that the mapping is correct.

    successful += 1

    result["status"] = "VALID"

    mapping_rows.append(result)


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if successful <= 20 or successful % 10 == 0:

        print(
            f"[{i + 1}/{len(recist)}] "
            f"Patient {patient_id} | "
            f"{study_date} | "
            f"Alias {alias} | "
            f"Label {label_value} | "
            f"voxels {voxel_count}"
        )


# ============================================================
# SAVE MAPPING
# ============================================================

mapping_df = pd.DataFrame(mapping_rows)

mapping_df.to_csv(
    OUTPUT_MAPPING,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

header("RECIST TO MASK TO CT MAPPING SUMMARY")

print("RECIST records:", len(recist))
print("Valid mappings:", successful)
print("Missing CT:", missing_ct)
print("Missing mask:", missing_mask)
print("Missing segmentation label:", label_missing)
print("Shape mismatches:", shape_mismatch)
print("Other errors:", errors)


# ============================================================
# STATUS DISTRIBUTION
# ============================================================

header("MAPPING STATUS")

print(
    mapping_df["status"]
    .value_counts()
    .to_string()
)


# ============================================================
# VALID MAPPING STATISTICS
# ============================================================

valid_df = mapping_df[
    mapping_df["status"] == "VALID"
].copy()

header("VALID LESION GEOMETRY")

if len(valid_df) > 0:

    print(
        "Valid RECIST lesions:",
        len(valid_df)
    )

    print(
        "Unique patients:",
        valid_df["patient_id"].nunique()
    )

    print(
        "Unique lesion aliases:",
        valid_df[
            ["patient_id", "lesion_label_alias"]
        ].drop_duplicates().shape[0]
    )

    print(
        "Minimum lesion voxels:",
        int(valid_df["voxel_count"].min())
    )

    print(
        "Maximum lesion voxels:",
        int(valid_df["voxel_count"].max())
    )

    print(
        "Median lesion voxels:",
        float(valid_df["voxel_count"].median())
    )

else:

    print("NO VALID LESION MAPPINGS FOUND.")


# ============================================================
# RECIST VS MASK CONSISTENCY
# ============================================================

header("RECIST LABEL CONSISTENCY")

if len(valid_df) > 0:

    print(
        "RECIST records with segmentation label:",
        len(valid_df)
    )

    print(
        "RECIST records with CT:",
        int(valid_df["ct_found"].sum())
    )

    print(
        "RECIST records with mask:",
        int(valid_df["mask_found"].sum())
    )

    print(
        "RECIST records with matching shape:",
        int(valid_df["shape_match"].sum())
    )

    print(
        "RECIST records with label present:",
        int(valid_df["label_found"].sum())
    )


# ============================================================
# SAVE A SMALL VERIFIED SAMPLE TABLE
# ============================================================

header("VERIFIED SAMPLE")

if len(valid_df) > 0:

    sample_columns = [
        "patient_id",
        "study_date",
        "lesion_label_alias",
        "lesion_label_value",
        "recist_measurement_mm",
        "voxel_count",
        "bbox_xmin",
        "bbox_xmax",
        "bbox_ymin",
        "bbox_ymax",
        "bbox_zmin",
        "bbox_zmax",
        "status"
    ]

    print(
        valid_df[
            sample_columns
        ].head(20).to_string(index=False)
    )


# ============================================================
# FINAL DECISION
# ============================================================

header("STEP 24 - FINAL FEASIBILITY")

all_valid = (
    len(valid_df) == len(recist)
)

if len(valid_df) > 0:
    print("[YES] RECIST lesions can be linked to segmentation labels")
else:
    print("[NO] No RECIST lesions linked to segmentation labels")

if missing_ct == 0:
    print("[YES] All RECIST CT filenames found")
else:
    print("[NO] Some RECIST CT filenames are missing")

if missing_mask == 0:
    print("[YES] All RECIST masks found")
else:
    print("[NO] Some RECIST masks are missing")

if shape_mismatch == 0:
    print("[YES] CT and mask shapes match")
else:
    print("[NO] CT/mask shape mismatches exist")

if label_missing == 0:
    print("[YES] All RECIST segmentation labels found")
else:
    print("[NO] Some RECIST labels are absent from masks")


# ============================================================
# IMPORTANT INTERPRETATION
# ============================================================

header("IMPORTANT INTERPRETATION")

print(
    """
This step verifies the physical connection:

    RECIST measurement
          |
          v
    segmentation label
          |
          v
    3D lesion
          |
          v
    CT series

No neural network is trained.

No biological state is claimed.

No clinical validation is performed.

The purpose is to establish whether the dataset can safely
provide lesion-specific CT inputs for the future imaging
encoder.
"""
)


# ============================================================
# SAVE
# ============================================================

header("SAVING STEP 24 RESULTS")

print("Saved:")
print(OUTPUT_MAPPING)

print()
print("=" * 95)
print("STEP 24 COMPLETE")
print("=" * 95)