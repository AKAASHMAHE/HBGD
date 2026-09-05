import os
import sys
import json
import glob
import traceback

import numpy as np
import pandas as pd
import nibabel as nib


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

MAPPING_FILE = os.path.join(
    DATA_ROOT,
    "24_recist_lesion_mapping.csv"
)

SERIES_FILE = os.path.join(
    DATA_ROOT,
    "series.json"
)

OUTPUT_DIR = os.path.join(
    DATA_ROOT,
    "25_lesion_ct_crops"
)

OUTPUT_CSV = os.path.join(
    DATA_ROOT,
    "25_ct_crop_quality.csv"
)

# Extra margin around segmentation bounding box.
# This gives the CNN some surrounding tissue/context.
MARGIN_XY = 16
MARGIN_Z = 8

# Minimum number of voxels required.
MIN_VOXELS = 10

# Maximum crop dimension safeguard.
MAX_CROP_XY = 160
MAX_CROP_Z = 96


# ============================================================
# TERMINAL UTF-8 SAFETY
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


def find_files(root, extensions):
    results = []

    for ext in extensions:
        results.extend(
            glob.glob(
                os.path.join(root, "**", ext),
                recursive=True
            )
        )

    return sorted(set(results))


def normalize_filename(path):
    return os.path.basename(path).lower()


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def clamp(v, low, high):
    return max(low, min(v, high))


def normalize_ct(volume):
    """
    Robust CT normalization.

    Uses percentile clipping rather than assuming a particular
    scanner intensity range.
    """

    arr = np.asarray(volume, dtype=np.float32)

    finite = np.isfinite(arr)

    if not np.any(finite):
        return None

    values = arr[finite]

    p1 = np.percentile(values, 1)
    p99 = np.percentile(values, 99)

    if p99 <= p1:
        return np.zeros_like(arr, dtype=np.float32)

    arr = np.clip(arr, p1, p99)

    arr = (arr - p1) / (p99 - p1)

    return arr.astype(np.float32)


def pad_to_shape(arr, target_shape):
    """
    Pad a 3D crop to target_shape.
    """

    output = np.zeros(
        target_shape,
        dtype=np.float32
    )

    slices = []

    for current, target in zip(arr.shape, target_shape):
        length = min(current, target)

        start_src = max((current - length) // 2, 0)
        start_dst = max((target - length) // 2, 0)

        slices.append(
            (
                start_src,
                start_src + length,
                start_dst,
                start_dst + length
            )
        )

    output[
        slices[0][2]:slices[0][3],
        slices[1][2]:slices[1][3],
        slices[2][2]:slices[2][3]
    ] = arr[
        slices[0][0]:slices[0][1],
        slices[1][0]:slices[1][1],
        slices[2][0]:slices[2][1]
    ]

    return output


# ============================================================
# START
# ============================================================

header("STEP 25 - LESION-SPECIFIC CT CROP EXTRACTION")

print("Dataset root:")
print(DATA_ROOT)

print()
print("Mapping file:")
print(MAPPING_FILE)

print()
print("Output directory:")
print(OUTPUT_DIR)


# ============================================================
# LOAD MAPPING
# ============================================================

header("LOADING VERIFIED RECIST-LESION MAPPING")

if not os.path.exists(MAPPING_FILE):
    raise FileNotFoundError(
        "Missing 24_recist_lesion_mapping.csv"
    )

mapping = pd.read_csv(MAPPING_FILE)

print("Mapping records:", len(mapping))

required_columns = [
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

missing = [
    c for c in required_columns
    if c not in mapping.columns
]

if missing:
    raise ValueError(
        "Missing required columns: " + str(missing)
    )

mapping = mapping[
    mapping["status"].astype(str).str.upper() == "VALID"
].copy()

print("Valid mapping records:", len(mapping))


# ============================================================
# LOAD SERIES METADATA
# ============================================================

header("LOADING SERIES METADATA")

if not os.path.exists(SERIES_FILE):
    raise FileNotFoundError(
        "Missing series.json"
    )

with open(
    SERIES_FILE,
    "r",
    encoding="utf-8"
) as f:
    series_records = json.load(f)

print("Series metadata records:", len(series_records))


# ============================================================
# FIND CT FILES
# ============================================================

header("SEARCHING FOR CT FILES")

ct_files = find_files(
    DATA_ROOT,
    ["*.nii.gz", "*.nii"]
)

# Exclude generated crop directory.
ct_files = [
    p for p in ct_files
    if OUTPUT_DIR not in os.path.abspath(p)
]

print("NIfTI files found:", len(ct_files))


# ============================================================
# INDEX CT FILES
# ============================================================

header("INDEXING CT FILES")

ct_index = {}

for path in ct_files:

    base = normalize_filename(path)

    # Only index image files.
    # Mapping file contains the actual CT filename.
    ct_index[base] = path

print("Indexed NIfTI files:", len(ct_index))


# ============================================================
# BUILD SERIES LOOKUP
# ============================================================

header("BUILDING SERIES LOOKUP")

series_lookup = {}

for record in series_records:

    patient_id = safe_int(
        record.get("patient_id")
    )

    study_date = str(
        record.get("study_date", "")
    ).strip()

    uuid = str(
        record.get("uuid", "")
    ).strip()

    if patient_id is None:
        continue

    key = (
        patient_id,
        study_date,
        uuid.lower()
    )

    series_lookup[key] = record

print(
    "Series lookup records:",
    len(series_lookup)
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PROCESS LESIONS
# ============================================================

header("EXTRACTING LESION-SPECIFIC CT CROPS")

results = []

success_count = 0
failure_count = 0

total = len(mapping)

for index, row in mapping.reset_index(drop=True).iterrows():

    patient_id = safe_int(
        row["patient_id"]
    )

    study_date = str(
        row["study_date"]
    ).strip()

    alias = str(
        row["lesion_label_alias"]
    ).strip()

    label_value = safe_int(
        row["lesion_label_value"]
    )

    recist_mm = float(
        row["recist_measurement_mm"]
    )

    voxel_count = safe_int(
        row["voxel_count"]
    )

    xmin = safe_int(row["bbox_xmin"])
    xmax = safe_int(row["bbox_xmax"])

    ymin = safe_int(row["bbox_ymin"])
    ymax = safe_int(row["bbox_ymax"])

    zmin = safe_int(row["bbox_zmin"])
    zmax = safe_int(row["bbox_zmax"])

    print(
        f"[{index + 1}/{total}] "
        f"Patient {patient_id} | "
        f"{study_date} | "
        f"Alias {alias} | "
        f"Label {label_value}"
    )

    result = {
        "patient_id": patient_id,
        "study_date": study_date,
        "lesion_label_alias": alias,
        "lesion_label_value": label_value,
        "recist_measurement_mm": recist_mm,
        "voxel_count": voxel_count,
        "status": "FAILED",
        "reason": "",
        "ct_file": "",
        "crop_file": "",
        "original_shape": "",
        "crop_shape": "",
        "crop_voxels": 0,
        "crop_fraction": np.nan,
        "bbox_width": np.nan,
        "bbox_height": np.nan,
        "bbox_depth": np.nan
    }

    try:

        # ----------------------------------------------------
        # VALIDATE BBOX
        # ----------------------------------------------------

        if any(
            v is None
            for v in [
                xmin, xmax,
                ymin, ymax,
                zmin, zmax
            ]
        ):
            result["reason"] = "Invalid bounding box"
            results.append(result)
            failure_count += 1
            continue

        if voxel_count is None or voxel_count < MIN_VOXELS:
            result["reason"] = "Too few lesion voxels"
            results.append(result)
            failure_count += 1
            continue

        # ----------------------------------------------------
        # FIND CT FILE
        # ----------------------------------------------------

        # The mapping file may contain filename information.
        possible_names = []

        for column in [
            "filename",
            "ct_filename",
            "image_filename"
        ]:
            if column in row.index:
                value = str(row[column]).strip()

                if value and value.lower() != "nan":
                    possible_names.append(
                        value.lower()
                    )

        ct_path = None

        for name in possible_names:

            name = os.path.basename(name).lower()

            if name in ct_index:
                ct_path = ct_index[name]
                break

        # If the mapping does not contain filename,
        # identify by patient/date and inspect metadata.
        if ct_path is None:

            candidates = []

            for record in series_records:

                rp = safe_int(
                    record.get("patient_id")
                )

                rd = str(
                    record.get("study_date", "")
                ).strip()

                uuid = str(
                    record.get("uuid", "")
                ).strip().lower()

                if rp == patient_id and rd == study_date:

                    for path in ct_files:

                        base = os.path.basename(
                            path
                        ).lower()

                        if uuid and uuid in base:
                            candidates.append(path)

            if len(candidates) == 1:
                ct_path = candidates[0]

            elif len(candidates) > 1:

                # Try to find a file whose patient prefix
                # matches the patient ID.
                prefix = (
                    f"{patient_id:03d}_"
                ).lower()

                prefix_candidates = [
                    p for p in candidates
                    if os.path.basename(p).lower().startswith(prefix)
                ]

                if len(prefix_candidates) == 1:
                    ct_path = prefix_candidates[0]

        if ct_path is None:

            result["reason"] = "CT file not uniquely found"

            results.append(result)

            failure_count += 1

            continue

        result["ct_file"] = ct_path

        # ----------------------------------------------------
        # LOAD CT
        # ----------------------------------------------------

        img = nib.load(ct_path)

        volume = img.get_fdata(
            dtype=np.float32
        )

        if volume.ndim != 3:

            result["reason"] = (
                f"CT is not 3D: ndim={volume.ndim}"
            )

            results.append(result)

            failure_count += 1

            continue

        original_shape = volume.shape

        result["original_shape"] = str(
            original_shape
        )

        # ----------------------------------------------------
        # CHECK BBOX AGAINST CT
        # ----------------------------------------------------

        sx, sy, sz = volume.shape

        if (
            xmax < 0 or
            xmin >= sx or
            ymax < 0 or
            ymin >= sy or
            zmax < 0 or
            zmin >= sz
        ):

            result["reason"] = (
                "Bounding box outside CT volume"
            )

            results.append(result)

            failure_count += 1

            continue

        # Clamp bbox.
        xmin = clamp(xmin, 0, sx - 1)
        xmax = clamp(xmax, 0, sx - 1)

        ymin = clamp(ymin, 0, sy - 1)
        ymax = clamp(ymax, 0, sy - 1)

        zmin = clamp(zmin, 0, sz - 1)
        zmax = clamp(zmax, 0, sz - 1)

        # ----------------------------------------------------
        # BBOX DIMENSIONS
        # ----------------------------------------------------

        width = xmax - xmin + 1
        height = ymax - ymin + 1
        depth = zmax - zmin + 1

        result["bbox_width"] = width
        result["bbox_height"] = height
        result["bbox_depth"] = depth

        # ----------------------------------------------------
        # EXPAND BBOX
        # ----------------------------------------------------

        cxmin = clamp(
            xmin - MARGIN_XY,
            0,
            sx - 1
        )

        cxmax = clamp(
            xmax + MARGIN_XY,
            0,
            sx - 1
        )

        cymin = clamp(
            ymin - MARGIN_XY,
            0,
            sy - 1
        )

        cymax = clamp(
            ymax + MARGIN_XY,
            0,
            sy - 1
        )

        czmin = clamp(
            zmin - MARGIN_Z,
            0,
            sz - 1
        )

        czmax = clamp(
            zmax + MARGIN_Z,
            0,
            sz - 1
        )

        crop = volume[
            cxmin:cxmax + 1,
            cymin:cymax + 1,
            czmin:czmax + 1
        ]

        if crop.size == 0:

            result["reason"] = "Empty crop"

            results.append(result)

            failure_count += 1

            continue

        # ----------------------------------------------------
        # NORMALIZE CT
        # ----------------------------------------------------

        crop = normalize_ct(crop)

        if crop is None:

            result["reason"] = (
                "CT normalization failed"
            )

            results.append(result)

            failure_count += 1

            continue

        # ----------------------------------------------------
        # CROP SIZE SAFETY CHECK
        # ----------------------------------------------------

        if (
            crop.shape[0] > MAX_CROP_XY or
            crop.shape[1] > MAX_CROP_XY or
            crop.shape[2] > MAX_CROP_Z
        ):

            # Center crop around lesion bbox.
            center_x = (
                (xmin + xmax) // 2
            )

            center_y = (
                (ymin + ymax) // 2
            )

            center_z = (
                (zmin + zmax) // 2
            )

            target_x = min(
                MAX_CROP_XY,
                sx
            )

            target_y = min(
                MAX_CROP_XY,
                sy
            )

            target_z = min(
                MAX_CROP_Z,
                sz
            )

            fxmin = clamp(
                center_x - target_x // 2,
                0,
                sx - target_x
            )

            fymin = clamp(
                center_y - target_y // 2,
                0,
                sy - target_y
            )

            fzmin = clamp(
                center_z - target_z // 2,
                0,
                sz - target_z
            )

            crop = volume[
                fxmin:fxmin + target_x,
                fymin:fymin + target_y,
                fzmin:fzmin + target_z
            ]

            crop = normalize_ct(crop)

        # ----------------------------------------------------
        # SAVE CROP
        # ----------------------------------------------------

        safe_date = (
            study_date.replace("-", "")
        )

        filename = (
            f"patient_{patient_id:03d}_"
            f"{safe_date}_"
            f"lesion_{alias}_"
            f"label_{label_value}.nii.gz"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        crop_img = nib.Nifti1Image(
            crop.astype(np.float32),
            img.affine
        )

        nib.save(
            crop_img,
            output_path
        )

        # ----------------------------------------------------
        # QUALITY STATISTICS
        # ----------------------------------------------------

        result["crop_file"] = output_path

        result["crop_shape"] = str(
            crop.shape
        )

        result["crop_voxels"] = int(
            crop.size
        )

        result["crop_fraction"] = (
            float(voxel_count) /
            float(crop.size)
        )

        result["status"] = "VALID"
        result["reason"] = "OK"

        results.append(result)

        success_count += 1

    except Exception as exc:

        result["reason"] = (
            f"{type(exc).__name__}: {exc}"
        )

        results.append(result)

        failure_count += 1

        print(
            "    ERROR:",
            result["reason"]
        )


# ============================================================
# SAVE RESULTS
# ============================================================

header("SAVING STEP 25 RESULTS")

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("Saved:")
print(OUTPUT_CSV)


# ============================================================
# SUMMARY
# ============================================================

header("STEP 25 - CROP EXTRACTION SUMMARY")

print("Total RECIST lesions:", total)
print("Successful CT crops:", success_count)
print("Failed crops:", failure_count)

if total > 0:

    success_rate = (
        100.0 *
        success_count /
        total
    )

    print(
        "Success rate:",
        round(success_rate, 2),
        "%"
    )


# ============================================================
# QUALITY SUMMARY
# ============================================================

header("CROP QUALITY")

valid_df = results_df[
    results_df["status"] == "VALID"
].copy()

if len(valid_df) > 0:

    print(
        "Valid crop records:",
        len(valid_df)
    )

    print(
        "Median crop voxel count:",
        int(
            valid_df["crop_voxels"].median()
        )
    )

    print(
        "Median lesion/crop voxel fraction:",
        round(
            valid_df["crop_fraction"].median(),
            6
        )
    )

    print()
    print("Crop shapes:")

    shape_counts = (
        valid_df["crop_shape"]
        .value_counts()
        .head(15)
    )

    for shape, count in shape_counts.items():

        print(
            " ",
            shape,
            ":",
            count
        )

    print()
    print("Bounding-box dimensions:")

    print(
        "  Width median:",
        round(
            valid_df["bbox_width"].median(),
            2
        )
    )

    print(
        "  Height median:",
        round(
            valid_df["bbox_height"].median(),
            2
        )
    )

    print(
        "  Depth median:",
        round(
            valid_df["bbox_depth"].median(),
            2
        )
    )

else:

    print(
        "NO VALID CROPS WERE CREATED."
    )


# ============================================================
# LONGITUDINAL COVERAGE
# ============================================================

header("LONGITUDINAL CT CROP COVERAGE")

if len(valid_df) > 0:

    grouped = (
        valid_df
        .groupby(
            [
                "patient_id",
                "lesion_label_alias"
            ]
        )
        .size()
        .reset_index(
            name="observations"
        )
    )

    longitudinal = grouped[
        grouped["observations"] >= 2
    ]

    print(
        "Unique patient-lesion trajectories:",
        len(grouped)
    )

    print(
        "Trajectories with >=2 CT crops:",
        len(longitudinal)
    )

    print(
        "Trajectories with >=3 CT crops:",
        int(
            (grouped["observations"] >= 3).sum()
        )
    )

    print()
    print(
        "Patients with CT crops:",
        valid_df["patient_id"].nunique()
    )

    print(
        "Patients with >=2 observations:",
        int(
            valid_df.groupby("patient_id")
            .size()
            .ge(2)
            .sum()
        )
    )


# ============================================================
# FINAL FEASIBILITY
# ============================================================

header("STEP 25 - FINAL FEASIBILITY")

all_valid = (
    success_count == total
    and total > 0
)

if all_valid:
    print("[YES] All RECIST lesions have CT crops")
else:
    print("[NO] Some RECIST lesions failed crop extraction")

if len(valid_df) > 0:
    print("[YES] Lesion-specific CT crops created")
else:
    print("[NO] No lesion-specific CT crops created")

if len(valid_df) > 0:

    longitudinal_count = (
        valid_df
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

    if longitudinal_count > 0:
        print(
            "[YES] Longitudinal CT lesion trajectories available"
        )
    else:
        print(
            "[NO] No longitudinal CT lesion trajectories"
        )

else:
    print(
        "[NO] Longitudinal CT lesion trajectories"
    )


print()
print("IMPORTANT:")
print()
print(
    "This step only verifies that lesion-specific CT inputs"
)
print(
    "can be extracted from the verified RECIST-to-mask mapping."
)
print()
print(
    "No CNN is trained."
)
print(
    "No biological state is claimed."
)
print(
    "No clinical validation is performed."
)


# ============================================================
# COMPLETE
# ============================================================

header("STEP 25 COMPLETE")

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Quality CSV:")
print(OUTPUT_CSV)

print()
print(
    "DO NOT TRAIN THE CNN YET."
)

print(
    "Send the COMPLETE terminal output."
)