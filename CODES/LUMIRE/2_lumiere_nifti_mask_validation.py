# -*- coding: utf-8 -*-

"""
2_lumiere_nifti_mask_validation.py

LUMIERE NIfTI + segmentation validation.

IMPORTANT:
- Does NOT extract the complete 30+ GB ZIP.
- Uses the exact paths stored inside the ZIP.
- Selectively extracts only candidate patient/timepoint files.
- Python 3.6 compatible.
- All outputs are saved to DATA/LUMIRE.
"""

from __future__ import print_function

import os
import re
import csv
import shutil
import zipfile
import math

import nibabel as nib
import numpy as np


# ============================================================
# PATHS
# ============================================================

ZIP_PATH = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE.zip"

OUTPUT_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "LUMIERE_NIFTI_MASK_VALIDATION.md"
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "lumiere_nifti_mask_validation.csv"
)

TRAJECTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "lumiere_validated_longitudinal_patients.csv"
)

TEMP_DIR = os.path.join(
    OUTPUT_DIR,
    "_TEMP_NIFTI_VALIDATION"
)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_STRUCTURAL_TIMEPOINTS = 3


# Preferred reference image.
REFERENCE_MODALITIES = [
    "T1.nii.gz",
    "CT1.nii.gz",
    "T2.nii.gz",
    "FLAIR.nii.gz"
]


# Preferred HD-GLIO-AUTO segmentation.
HDGLIO_SEGMENTATIONS = [
    "segmentation_T1_origspace.nii.gz",
    "segmentation_CT1_origspace.nii.gz",
    "segmentation_FLAIR_origspace.nii.gz",
    "segmentation_T2_origspace.nii.gz",
    "segmentation.nii.gz"
]


# Fallback DeepBraTumIA segmentation.
DEEP_SEGMENTATIONS = [
    "t1_seg_mask.nii.gz",
    "ct1_seg_mask.nii.gz",
    "t2_seg_mask.nii.gz",
    "flair_seg_mask.nii.gz"
]


# ============================================================
# REGEX
# ============================================================

PATIENT_RE = re.compile(
    r"(?:^|/)Patient-([^/]+)(?:/|$)",
    re.IGNORECASE
)

WEEK_RE = re.compile(
    r"(?:^|/)week-([0-9]+)(?:/|$)",
    re.IGNORECASE
)


# ============================================================
# HELPERS
# ============================================================

def normalize_path(path):
    return path.replace("\\", "/")


def parse_patient_week(path):

    path = normalize_path(path)

    patient_match = PATIENT_RE.search(path)
    week_match = WEEK_RE.search(path)

    if patient_match is None:
        return None, None

    if week_match is None:
        return None, None

    patient_id = "Patient-" + patient_match.group(1)

    week_id = week_match.group(1).zfill(3)

    return patient_id, week_id


def choose_exact_filename(file_records, candidates):

    """
    file_records contains dictionaries:

        {
            "zip_path": "...",
            "filename": "..."
        }

    Return the actual ZIP path.
    """

    lookup = {}

    for record in file_records:

        filename = record["filename"]

        lookup[
            filename.lower()
        ] = record["zip_path"]

    for candidate in candidates:

        candidate_lower = candidate.lower()

        if candidate_lower in lookup:

            return lookup[candidate_lower]

    return None


def get_filename_from_zip_path(zip_path):

    return normalize_path(zip_path).split("/")[-1]


def get_voxel_spacing(header):

    zooms = header.get_zooms()

    if len(zooms) < 3:
        return None

    return (
        float(zooms[0]),
        float(zooms[1]),
        float(zooms[2])
    )


def calculate_voxel_volume(spacing):

    if spacing is None:
        return float("nan")

    return abs(
        spacing[0] *
        spacing[1] *
        spacing[2]
    )


def check_affine_match(
    affine_a,
    affine_b,
    tolerance=1e-3
):

    try:

        return bool(
            np.allclose(
                affine_a,
                affine_b,
                atol=tolerance,
                rtol=tolerance
            )
        )

    except Exception:

        return False


def check_spacing_match(
    spacing_a,
    spacing_b,
    tolerance=1e-3
):

    if spacing_a is None or spacing_b is None:
        return False

    try:

        return bool(
            np.allclose(
                spacing_a,
                spacing_b,
                atol=tolerance,
                rtol=tolerance
            )
        )

    except Exception:

        return False


def calculate_bbox(mask):

    coordinates = np.argwhere(
        mask != 0
    )

    if coordinates.size == 0:
        return None

    minimum = coordinates.min(
        axis=0
    )

    maximum = coordinates.max(
        axis=0
    )

    return (
        int(minimum[0]),
        int(minimum[1]),
        int(minimum[2]),
        int(maximum[0]),
        int(maximum[1]),
        int(maximum[2])
    )


def calculate_bbox_mm(mask, spacing):

    bbox = calculate_bbox(mask)

    if bbox is None:
        return None

    x_min = bbox[0]
    y_min = bbox[1]
    z_min = bbox[2]

    x_max = bbox[3]
    y_max = bbox[4]
    z_max = bbox[5]

    return (
        (x_max - x_min + 1) * spacing[0],
        (y_max - y_min + 1) * spacing[1],
        (z_max - z_min + 1) * spacing[2]
    )


def safe_string(value):

    if value is None:
        return ""

    return str(value)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LUMIERE NIFTI + SEGMENTATION VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Check ZIP
    # --------------------------------------------------------

    if not os.path.exists(ZIP_PATH):

        print("\nERROR: ZIP not found:")
        print(ZIP_PATH)

        return

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    if not os.path.exists(OUTPUT_DIR):

        os.makedirs(OUTPUT_DIR)

    # --------------------------------------------------------
    # Temporary directory
    # --------------------------------------------------------

    if os.path.exists(TEMP_DIR):

        try:
            shutil.rmtree(TEMP_DIR)
        except Exception:
            pass

    os.makedirs(TEMP_DIR)

    # ========================================================
    # OPEN ZIP
    # ========================================================

    print("\nOpening ZIP metadata...")

    archive = zipfile.ZipFile(
        str(ZIP_PATH),
        "r"
    )

    infos = archive.infolist()

    print(
        "Total ZIP entries: {:,}".format(
            len(infos)
        )
    )

    # ========================================================
    # BUILD EXACT FILE INDEX
    # ========================================================

    print("\nBuilding exact patient-timepoint index...")

    structure = {}

    file_counter = 0

    for info in infos:

        raw_path = info.filename

        if raw_path.endswith("/"):
            continue

        file_counter += 1

        zip_path = normalize_path(
            raw_path
        )

        patient_id, week_id = parse_patient_week(
            zip_path
        )

        if patient_id is None:
            continue

        if week_id is None:
            continue

        if patient_id not in structure:

            structure[patient_id] = {}

        if week_id not in structure[patient_id]:

            structure[
                patient_id
            ][week_id] = []

        structure[
            patient_id
        ][week_id].append({

            "zip_path": zip_path,

            "filename":
                get_filename_from_zip_path(
                    zip_path
                )

        })

    patient_ids = sorted(
        structure.keys()
    )

    print(
        "Files indexed: {:,}".format(
            file_counter
        )
    )

    print(
        "Patients found: {:,}".format(
            len(patient_ids)
        )
    )

    # ========================================================
    # STRUCTURAL ANALYSIS
    # ========================================================

    print(
        "\nDetermining structurally usable timepoints..."
    )

    candidate_patients = []

    candidate_weeks = {}

    for patient_id in patient_ids:

        usable = []

        weeks = sorted(
            structure[
                patient_id
            ].keys(),
            key=lambda x: int(x)
        )

        for week_id in weeks:

            records = structure[
                patient_id
            ][week_id]

            image_path = choose_exact_filename(
                records,
                REFERENCE_MODALITIES
            )

            segmentation_path = choose_exact_filename(
                records,
                HDGLIO_SEGMENTATIONS
            )

            segmentation_source = (
                "HD-GLIO-AUTO"
            )

            if segmentation_path is None:

                segmentation_path = choose_exact_filename(
                    records,
                    DEEP_SEGMENTATIONS
                )

                segmentation_source = (
                    "DeepBraTumIA"
                )

            if (
                image_path is not None
                and segmentation_path is not None
            ):

                usable.append({
                    "week": week_id,
                    "image_path": image_path,
                    "segmentation_path":
                        segmentation_path,
                    "segmentation_source":
                        segmentation_source
                })

        candidate_weeks[
            patient_id
        ] = usable

        if len(usable) >= MIN_STRUCTURAL_TIMEPOINTS:

            candidate_patients.append(
                patient_id
            )

    print(
        "Patients with >= {} structurally usable timepoints: {}".format(
            MIN_STRUCTURAL_TIMEPOINTS,
            len(candidate_patients)
        )
    )

    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    if len(candidate_patients) == 0:

        print("\nWARNING:")
        print(
            "No candidate patients were found."
        )

        print(
            "\nExample indexed patient/week structures:"
        )

        shown = 0

        for patient_id in patient_ids:

            print(
                "\n{}".format(
                    patient_id
                )
            )

            for week_id in sorted(
                structure[
                    patient_id
                ].keys(),
                key=lambda x: int(x)
            )[:3]:

                records = structure[
                    patient_id
                ][week_id]

                print(
                    "  week-{}: {} files".format(
                        week_id,
                        len(records)
                    )
                )

                for record in records[:5]:

                    print(
                        "     {}".format(
                            record["zip_path"]
                        )
                    )

            shown += 1

            if shown >= 3:
                break

        archive.close()

        shutil.rmtree(
            TEMP_DIR,
            ignore_errors=True
        )

        return

    # ========================================================
    # VALIDATION
    # ========================================================

    results = []

    validated_trajectories = {}

    processed = 0
    valid = 0
    check_required = 0
    errors = 0

    # ========================================================
    # PATIENT LOOP
    # ========================================================

    for patient_number, patient_id in enumerate(
        candidate_patients
    ):

        print("\n")
        print("=" * 70)

        print(
            "PATIENT {}/{}: {}".format(
                patient_number + 1,
                len(candidate_patients),
                patient_id
            )
        )

        print("=" * 70)

        patient_valid_weeks = []

        usable_entries = candidate_weeks[
            patient_id
        ]

        usable_entries = sorted(
            usable_entries,
            key=lambda x: int(x["week"])
        )

        # ----------------------------------------------------
        # Timepoint loop
        # ----------------------------------------------------

        for entry in usable_entries:

            week_id = entry["week"]

            image_zip_path = entry[
                "image_path"
            ]

            segmentation_zip_path = entry[
                "segmentation_path"
            ]

            image_filename = get_filename_from_zip_path(
                image_zip_path
            )

            segmentation_filename = get_filename_from_zip_path(
                segmentation_zip_path
            )

            processed += 1

            print(
                "\n[{}/{}] {} / week-{}".format(
                    processed,
                    len(candidate_patients) * 3,
                    patient_id,
                    week_id
                )
            )

            print(
                "Image:",
                image_filename
            )

            print(
                "Segmentation:",
                segmentation_filename
            )

            # ------------------------------------------------
            # Temporary directory
            # ------------------------------------------------

            current_dir = os.path.join(
                TEMP_DIR,
                "{}_{}".format(
                    patient_id,
                    week_id
                )
            )

            if os.path.exists(current_dir):

                shutil.rmtree(
                    current_dir
                )

            os.makedirs(
                current_dir
            )

            image_local = os.path.join(
                current_dir,
                image_filename
            )

            segmentation_local = os.path.join(
                current_dir,
                segmentation_filename
            )

            row = {

                "patient_id":
                    patient_id,

                "week":
                    week_id,

                "image_zip_path":
                    image_zip_path,

                "segmentation_zip_path":
                    segmentation_zip_path,

                "image_file":
                    image_filename,

                "segmentation_file":
                    segmentation_filename,

                "segmentation_source":
                    entry["segmentation_source"],

                "image_load_success":
                    False,

                "segmentation_load_success":
                    False,

                "image_shape":
                    "",

                "segmentation_shape":
                    "",

                "shape_match":
                    False,

                "image_spacing":
                    "",

                "segmentation_spacing":
                    "",

                "spacing_match":
                    False,

                "affine_match":
                    False,

                "nonzero_mask_voxels":
                    0,

                "tumor_volume_mm3":
                    "",

                "tumor_volume_cm3":
                    "",

                "bbox_x_mm":
                    "",

                "bbox_y_mm":
                    "",

                "bbox_z_mm":
                    "",

                "mask_binary":
                    False,

                "validation_status":
                    "ERROR",

                "error":
                    ""

            }

            try:

                # ==================================================
                # EXTRACT IMAGE USING EXACT ZIP PATH
                # ==================================================

                print(
                    "Extracting image..."
                )

                with archive.open(
                    image_zip_path,
                    "r"
                ) as source:

                    with open(
                        image_local,
                        "wb"
                    ) as target:

                        shutil.copyfileobj(
                            source,
                            target,
                            1024 * 1024
                        )

                # ==================================================
                # EXTRACT SEGMENTATION USING EXACT ZIP PATH
                # ==================================================

                print(
                    "Extracting segmentation..."
                )

                with archive.open(
                    segmentation_zip_path,
                    "r"
                ) as source:

                    with open(
                        segmentation_local,
                        "wb"
                    ) as target:

                        shutil.copyfileobj(
                            source,
                            target,
                            1024 * 1024
                        )

                # ==================================================
                # LOAD IMAGE
                # ==================================================

                print(
                    "Loading NIfTI files..."
                )

                image = nib.load(
                    image_local
                )

                segmentation = nib.load(
                    segmentation_local
                )

                row[
                    "image_load_success"
                ] = True

                row[
                    "segmentation_load_success"
                ] = True

                # ==================================================
                # SHAPE
                # ==================================================

                image_shape = tuple(
                    image.shape[:3]
                )

                segmentation_shape = tuple(
                    segmentation.shape[:3]
                )

                row[
                    "image_shape"
                ] = str(image_shape)

                row[
                    "segmentation_shape"
                ] = str(segmentation_shape)

                row[
                    "shape_match"
                ] = (
                    image_shape ==
                    segmentation_shape
                )

                # ==================================================
                # SPACING
                # ==================================================

                image_spacing = (
                    get_voxel_spacing(
                        image.header
                    )
                )

                segmentation_spacing = (
                    get_voxel_spacing(
                        segmentation.header
                    )
                )

                row[
                    "image_spacing"
                ] = str(
                    image_spacing
                )

                row[
                    "segmentation_spacing"
                ] = str(
                    segmentation_spacing
                )

                row[
                    "spacing_match"
                ] = check_spacing_match(
                    image_spacing,
                    segmentation_spacing
                )

                # ==================================================
                # AFFINE
                # ==================================================

                row[
                    "affine_match"
                ] = check_affine_match(
                    image.affine,
                    segmentation.affine
                )

                # ==================================================
                # MASK DATA
                # ==================================================

                print(
                    "Reading segmentation..."
                )

                mask_data = segmentation.get_fdata()

                mask_data = np.nan_to_num(
                    mask_data
                )

                nonzero_voxels = int(
                    np.count_nonzero(
                        mask_data
                    )
                )

                row[
                    "nonzero_mask_voxels"
                ] = nonzero_voxels

                # ==================================================
                # VOLUME
                # ==================================================

                voxel_volume = (
                    calculate_voxel_volume(
                        segmentation_spacing
                    )
                )

                tumor_volume_mm3 = (
                    nonzero_voxels *
                    voxel_volume
                )

                tumor_volume_cm3 = (
                    tumor_volume_mm3 /
                    1000.0
                )

                row[
                    "tumor_volume_mm3"
                ] = tumor_volume_mm3

                row[
                    "tumor_volume_cm3"
                ] = tumor_volume_cm3

                # ==================================================
                # BOUNDING BOX
                # ==================================================

                bbox_mm = (
                    calculate_bbox_mm(
                        mask_data,
                        segmentation_spacing
                    )
                )

                if bbox_mm is not None:

                    row[
                        "bbox_x_mm"
                    ] = bbox_mm[0]

                    row[
                        "bbox_y_mm"
                    ] = bbox_mm[1]

                    row[
                        "bbox_z_mm"
                    ] = bbox_mm[2]

                # ==================================================
                # BINARY CHECK
                # ==================================================

                unique_values = np.unique(
                    mask_data
                )

                row[
                    "mask_binary"
                ] = bool(
                    np.all(
                        np.isin(
                            unique_values,
                            [0, 1]
                        )
                    )
                )

                # ==================================================
                # VALIDATION DECISION
                # ==================================================

                if (
                    row["image_load_success"]
                    and
                    row["segmentation_load_success"]
                    and
                    row["shape_match"]
                    and
                    row["spacing_match"]
                    and
                    row["affine_match"]
                    and
                    nonzero_voxels > 0
                    and
                    tumor_volume_mm3 > 0
                ):

                    row[
                        "validation_status"
                    ] = "VALID"

                    valid += 1

                    patient_valid_weeks.append(
                        week_id
                    )

                else:

                    row[
                        "validation_status"
                    ] = "CHECK_REQUIRED"

                    check_required += 1

            except Exception as error:

                row[
                    "validation_status"
                ] = "ERROR"

                row[
                    "error"
                ] = str(error)

                errors += 1

                print(
                    "\nERROR:",
                    str(error)
                )

            results.append(
                row
            )

            # ==================================================
            # CONSOLE RESULT
            # ==================================================

            print(
                "Status:",
                row["validation_status"]
            )

            if row["tumor_volume_cm3"] != "":

                try:

                    print(
                        "Tumor volume: {:.3f} cm3".format(
                            float(
                                row[
                                    "tumor_volume_cm3"
                                ]
                            )
                        )
                    )

                except Exception:
                    pass

            # ==================================================
            # CLEANUP
            # ==================================================

            try:

                if os.path.exists(
                    current_dir
                ):

                    shutil.rmtree(
                        current_dir
                    )

            except Exception:

                pass

        # ----------------------------------------------------
        # Save validated trajectory
        # ----------------------------------------------------

        if len(patient_valid_weeks) >= 3:

            validated_trajectories[
                patient_id
            ] = sorted(
                patient_valid_weeks,
                key=lambda x: int(x)
            )

    # ========================================================
    # CLOSE ZIP
    # ========================================================

    archive.close()

    # ========================================================
    # CLEAN TEMP
    # ========================================================

    try:

        if os.path.exists(
            TEMP_DIR
        ):

            shutil.rmtree(
                TEMP_DIR
            )

    except Exception:

        pass

    # ========================================================
    # SAVE CSV
    # ========================================================

    columns = [

        "patient_id",
        "week",

        "image_zip_path",
        "segmentation_zip_path",

        "image_file",
        "segmentation_file",
        "segmentation_source",

        "image_load_success",
        "segmentation_load_success",

        "image_shape",
        "segmentation_shape",
        "shape_match",

        "image_spacing",
        "segmentation_spacing",
        "spacing_match",

        "affine_match",

        "nonzero_mask_voxels",

        "tumor_volume_mm3",
        "tumor_volume_cm3",

        "bbox_x_mm",
        "bbox_y_mm",
        "bbox_z_mm",

        "mask_binary",

        "validation_status",
        "error"
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=columns
        )

        writer.writeheader()

        for row in results:

            writer.writerow(
                row
            )

    # ========================================================
    # SAVE TRAJECTORIES
    # ========================================================

    with open(
        TRAJECTORY_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.writer(
            csv_file
        )

        writer.writerow([
            "patient_id",
            "validated_timepoints",
            "number_validated_timepoints",
            "min_week",
            "max_week"
        ])

        for patient_id in sorted(
            validated_trajectories.keys()
        ):

            weeks = (
                validated_trajectories[
                    patient_id
                ]
            )

            writer.writerow([

                patient_id,

                ",".join(
                    weeks
                ),

                len(
                    weeks
                ),

                min(
                    weeks,
                    key=lambda x: int(x)
                ),

                max(
                    weeks,
                    key=lambda x: int(x)
                )
            ])

    # ========================================================
    # CREATE REPORT
    # ========================================================

    report = []

    report.append(
        "# LUMIERE NIFTI + SEGMENTATION VALIDATION"
    )

    report.append("")

    report.append(
        "Selective NIfTI validation using exact paths stored inside the ZIP archive."
    )

    report.append("")

    report.append(
        "## 1. Dataset Summary"
    )

    report.append("")

    report.append(
        "| Metric | Value |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        "| Patients in ZIP | {} |".format(
            len(patient_ids)
        )
    )

    report.append(
        "| Candidate patients | {} |".format(
            len(candidate_patients)
        )
    )

    report.append(
        "| Patient-timepoints processed | {} |".format(
            processed
        )
    )

    report.append(
        "| VALID timepoints | {} |".format(
            valid
        )
    )

    report.append(
        "| CHECK_REQUIRED timepoints | {} |".format(
            check_required
        )
    )

    report.append(
        "| ERROR timepoints | {} |".format(
            errors
        )
    )

    report.append("")

    # ========================================================
    # VALIDATION RATE
    # ========================================================

    if processed > 0:

        rate = (
            float(valid) /
            float(processed)
        ) * 100.0

    else:

        rate = 0.0

    report.append(
        "Validation success rate: **{:.2f}%**".format(
            rate
        )
    )

    report.append("")

    # ========================================================
    # VALIDATED PATIENTS
    # ========================================================

    report.append(
        "## 2. Validated Longitudinal Patients"
    )

    report.append("")

    report.append(
        "Patients with at least 3 VALID timepoints: **{}**".format(
            len(
                validated_trajectories
            )
        )
    )

    report.append("")

    if validated_trajectories:

        for patient_id in sorted(
            validated_trajectories.keys()
        ):

            weeks = (
                validated_trajectories[
                    patient_id
                ]
            )

            report.append(
                "- **{}**: {} validated timepoints — {}".format(
                    patient_id,
                    len(weeks),
                    ", ".join(weeks)
                )
            )

    else:

        report.append(
            "No patient currently has >=3 fully validated timepoints."
        )

    report.append("")

    # ========================================================
    # VOLUME SUMMARY
    # ========================================================

    volumes = []

    for row in results:

        if (
            row["validation_status"] ==
            "VALID"
        ):

            try:

                volume = float(
                    row[
                        "tumor_volume_cm3"
                    ]
                )

                if (
                    math.isfinite(volume)
                    and volume > 0
                ):

                    volumes.append(
                        volume
                    )

            except Exception:

                pass

    report.append(
        "## 3. Tumor Volume Summary"
    )

    report.append("")

    if volumes:

        report.append(
            "| Metric | Value |"
        )

        report.append(
            "|---|---:|"
        )

        report.append(
            "| Valid volume measurements | {} |".format(
                len(volumes)
            )
        )

        report.append(
            "| Minimum volume | {:.3f} cm3 |".format(
                min(volumes)
            )
        )

        report.append(
            "| Maximum volume | {:.3f} cm3 |".format(
                max(volumes)
            )
        )

        report.append(
            "| Mean volume | {:.3f} cm3 |".format(
                sum(volumes) /
                len(volumes)
            )
        )

    else:

        report.append(
            "No valid tumor-volume measurements."
        )

    report.append("")

    # ========================================================
    # FAILED / CHECK CASES
    # ========================================================

    report.append(
        "## 4. Cases Requiring Checking"
    )

    report.append("")

    failed = [
        row
        for row in results
        if row["validation_status"]
        != "VALID"
    ]

    if failed:

        for row in failed:

            report.append(
                "- `{}` / `week-{}` — `{}`".format(
                    row["patient_id"],
                    row["week"],
                    row["validation_status"]
                )
            )

            if row["error"]:

                report.append(
                    "  - {}".format(
                        row["error"]
                    )
                )

            if not row["shape_match"]:

                report.append(
                    "  - Shape mismatch"
                )

            if not row["spacing_match"]:

                report.append(
                    "  - Voxel spacing mismatch"
                )

            if not row["affine_match"]:

                report.append(
                    "  - Affine mismatch"
                )

            if (
                row["nonzero_mask_voxels"] == 0
            ):

                report.append(
                    "  - Empty segmentation"
                )

    else:

        report.append(
            "None."
        )

    report.append("")

    # ========================================================
    # SCIENTIFIC NOTE
    # ========================================================

    report.append(
        "## 5. Scientific Note"
    )

    report.append("")

    report.append(
        "This validation establishes file-level and geometric consistency only. It does not establish clinical correctness of the tumor segmentation."
    )

    report.append("")

    report.append(
        "Validated timepoints are candidates for downstream tumor geometry and HBGD growth-rate analysis."
    )

    report.append("")

    report.append(
        "Longitudinal modelling should use only patient-timepoints that pass this validation stage."
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as report_file:

        report_file.write(
            "\n".join(report)
        )

    # ========================================================
    # FINAL CONSOLE
    # ========================================================

    print("\n")
    print("=" * 70)
    print("NIFTI VALIDATION COMPLETE")
    print("=" * 70)

    print(
        "\nPatients in ZIP:",
        len(patient_ids)
    )

    print(
        "Candidate patients:",
        len(candidate_patients)
    )

    print(
        "Patient-timepoints processed:",
        processed
    )

    print(
        "VALID:",
        valid
    )

    print(
        "CHECK_REQUIRED:",
        check_required
    )

    print(
        "ERROR:",
        errors
    )

    print(
        "Validated longitudinal patients:",
        len(validated_trajectories)
    )

    print("\nOUTPUTS:")

    print(
        REPORT_PATH
    )

    print(
        CSV_PATH
    )

    print(
        TRAJECTORY_PATH
    )

    print(
        "\nComplete ZIP was NOT extracted."
    )

    print(
        "Temporary files were removed."
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()