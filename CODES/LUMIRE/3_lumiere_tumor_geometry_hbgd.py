# -*- coding: utf-8 -*-

"""
3_lumiere_tumor_geometry_hbgd.py

STEP 3:
    Extract tumor geometry and HBGD growth features
    from validated LUMIERE longitudinal timepoints.

IMPORTANT:
    - Does NOT extract the complete ZIP.
    - Uses only validated timepoints from Step 2.
    - Extracts one segmentation at a time.
    - Uses memory-efficient integer mask loading.
    - Does NOT use get_fdata().
    - Python 3.6 compatible.

Outputs:
    lumiere_tumor_geometry.csv
    lumiere_hbgd_growth_features.csv
    lumiere_longitudinal_trajectories.csv
    LUMIERE_TUMOR_GEOMETRY_HBGD_REPORT.md
"""

from __future__ import print_function

import os
import csv
import re
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

VALIDATION_CSV = os.path.join(
    OUTPUT_DIR,
    "lumiere_nifti_mask_validation.csv"
)

GEOMETRY_CSV = os.path.join(
    OUTPUT_DIR,
    "lumiere_tumor_geometry.csv"
)

HBGD_CSV = os.path.join(
    OUTPUT_DIR,
    "lumiere_hbgd_growth_features.csv"
)

TRAJECTORY_CSV = os.path.join(
    OUTPUT_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "LUMIERE_TUMOR_GEOMETRY_HBGD_REPORT.md"
)

TEMP_DIR = os.path.join(
    OUTPUT_DIR,
    "_TEMP_GEOMETRY"
)


# ============================================================
# HBGD PARAMETERS
# ============================================================

# Existing HBGD diameter-state thresholds
GROWTH_THRESHOLD = 0.0005

STABLE_LOWER = -0.0010

STABLE_UPPER = 0.0005


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):

    try:
        return float(value)
    except Exception:
        return float("nan")


def normalize_path(path):

    return path.replace("\\", "/")


def get_filename(path):

    return normalize_path(path).split("/")[-1]


def calculate_equivalent_diameter(volume_mm3):

    """
    Sphere-equivalent diameter:

        V = pi/6 * D^3

    Therefore:

        D = (6V/pi)^(1/3)
    """

    if volume_mm3 <= 0:
        return float("nan")

    return (
        (6.0 * volume_mm3 / math.pi)
        ** (1.0 / 3.0)
    )


def calculate_volume_from_voxels(
    voxel_count,
    voxel_spacing
):

    voxel_volume = (
        abs(
            voxel_spacing[0] *
            voxel_spacing[1] *
            voxel_spacing[2]
        )
    )

    return voxel_count * voxel_volume


def calculate_bbox(mask):

    """
    Returns voxel-space bounding box.
    """

    coords = np.argwhere(
        mask != 0
    )

    if coords.size == 0:
        return None

    minimum = coords.min(
        axis=0
    )

    maximum = coords.max(
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


def calculate_bbox_dimensions(
    mask,
    spacing
):

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


def calculate_centroid_voxel(mask):

    coords = np.argwhere(
        mask != 0
    )

    if coords.size == 0:
        return None

    centroid = coords.mean(
        axis=0
    )

    return (
        float(centroid[0]),
        float(centroid[1]),
        float(centroid[2])
    )


def voxel_to_world(
    affine,
    voxel_coordinate
):

    if voxel_coordinate is None:
        return None

    vector = np.array([
        voxel_coordinate[0],
        voxel_coordinate[1],
        voxel_coordinate[2],
        1.0
    ])

    world = np.dot(
        affine,
        vector
    )

    return (
        float(world[0]),
        float(world[1]),
        float(world[2])
    )


def calculate_growth_rate(
    d1,
    d2,
    delta_weeks
):

    """
    HBGD diameter growth rate:

        r = ln(D2 / D1) / delta_t

    Time unit:
        days

    Since LUMIERE provides week identifiers,
    delta_t = delta_week * 7 days.

    This assumes week identifiers represent
    elapsed time. The assumption is reported
    explicitly.
    """

    if d1 <= 0 or d2 <= 0:
        return float("nan")

    if delta_weeks <= 0:
        return float("nan")

    delta_days = (
        float(delta_weeks) * 7.0
    )

    return (
        math.log(
            float(d2) / float(d1)
        ) /
        delta_days
    )


def calculate_volume_growth_rate(
    v1,
    v2,
    delta_weeks
):

    if v1 <= 0 or v2 <= 0:
        return float("nan")

    if delta_weeks <= 0:
        return float("nan")

    delta_days = (
        float(delta_weeks) * 7.0
    )

    return (
        math.log(
            float(v2) / float(v1)
        ) /
        delta_days
    )


def classify_growth_state(rate):

    if not math.isfinite(rate):
        return "UNKNOWN"

    if rate > GROWTH_THRESHOLD:

        return "GROWTH"

    if rate < STABLE_LOWER:

        return "SHRINKAGE"

    return "STABLE"


def format_number(value):

    try:

        if not math.isfinite(
            float(value)
        ):
            return ""

        return "{:.6f}".format(
            float(value)
        )

    except Exception:

        return ""


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LUMIERE TUMOR GEOMETRY + HBGD FEATURE EXTRACTION")
    print("=" * 70)

    # ========================================================
    # CHECK FILES
    # ========================================================

    if not os.path.exists(ZIP_PATH):

        print(
            "\nERROR: ZIP not found:"
        )

        print(
            ZIP_PATH
        )

        return

    if not os.path.exists(
        VALIDATION_CSV
    ):

        print(
            "\nERROR: Validation CSV not found:"
        )

        print(
            VALIDATION_CSV
        )

        print(
            "\nRun Step 2 first."
        )

        return

    if not os.path.exists(
        OUTPUT_DIR
    ):

        os.makedirs(
            OUTPUT_DIR
        )

    # ========================================================
    # READ VALIDATION RESULTS
    # ========================================================

    print(
        "\nReading Step 2 validation results..."
    )

    validated_rows = []

    with open(
        VALIDATION_CSV,
        "r",
        encoding="utf-8"
    ) as csv_file:

        reader = csv.DictReader(
            csv_file
        )

        for row in reader:

            if (
                row.get(
                    "validation_status",
                    ""
                )
                == "VALID"
            ):

                validated_rows.append(
                    row
                )

    print(
        "VALID timepoints:",
        len(validated_rows)
    )

    if len(validated_rows) == 0:

        print(
            "\nERROR: No VALID timepoints found."
        )

        return

    # ========================================================
    # TEMP DIRECTORY
    # ========================================================

    if os.path.exists(
        TEMP_DIR
    ):

        shutil.rmtree(
            TEMP_DIR
        )

    os.makedirs(
        TEMP_DIR
    )

    # ========================================================
    # OPEN ZIP
    # ========================================================

    print(
        "\nOpening ZIP..."
    )

    archive = zipfile.ZipFile(
        str(ZIP_PATH),
        "r"
    )

    # ========================================================
    # GEOMETRY RESULTS
    # ========================================================

    geometry_results = []

    successful = 0
    failed = 0

    # ========================================================
    # PROCESS VALIDATED TIMEPOINTS
    # ========================================================

    for index, row in enumerate(
        validated_rows
    ):

        patient_id = row[
            "patient_id"
        ]

        week = row[
            "week"
        ]

        segmentation_zip_path = normalize_path(
            row[
                "segmentation_zip_path"
            ]
        )

        segmentation_filename = get_filename(
            segmentation_zip_path
        )

        print(
            "\n[{}/{}] {} / week-{}".format(
                index + 1,
                len(validated_rows),
                patient_id,
                week
            )
        )

        print(
            "Segmentation:",
            segmentation_filename
        )

        # ----------------------------------------------------
        # Temporary directory
        # ----------------------------------------------------

        current_dir = os.path.join(
            TEMP_DIR,
            "{}_{}".format(
                patient_id,
                week
            )
        )

        if os.path.exists(
            current_dir
        ):

            shutil.rmtree(
                current_dir
            )

        os.makedirs(
            current_dir
        )

        local_segmentation = os.path.join(
            current_dir,
            segmentation_filename
        )

        result = {

            "patient_id":
                patient_id,

            "week":
                week,

            "segmentation_source":
                row.get(
                    "segmentation_source",
                    ""
                ),

            "segmentation_file":
                segmentation_filename,

            "voxel_spacing_x_mm":
                "",

            "voxel_spacing_y_mm":
                "",

            "voxel_spacing_z_mm":
                "",

            "voxel_volume_mm3":
                "",

            "mask_voxels":
                "",

            "tumor_volume_mm3":
                "",

            "tumor_volume_cm3":
                "",

            "equivalent_diameter_mm":
                "",

            "bbox_x_mm":
                "",

            "bbox_y_mm":
                "",

            "bbox_z_mm":
                "",

            "centroid_voxel_x":
                "",

            "centroid_voxel_y":
                "",

            "centroid_voxel_z":
                "",

            "centroid_world_x_mm":
                "",

            "centroid_world_y_mm":
                "",

            "centroid_world_z_mm":
                "",

            "status":
                "ERROR",

            "error":
                ""

        }

        try:

            # ------------------------------------------------
            # Extract segmentation
            # ------------------------------------------------

            with archive.open(
                segmentation_zip_path,
                "r"
            ) as source:

                with open(
                    local_segmentation,
                    "wb"
                ) as target:

                    shutil.copyfileobj(
                        source,
                        target,
                        1024 * 1024
                    )

            # ------------------------------------------------
            # Load NIfTI
            # ------------------------------------------------

            segmentation = nib.load(
                local_segmentation
            )

            # ------------------------------------------------
            # Spacing
            # ------------------------------------------------

            spacing = (
                float(
                    segmentation.header.get_zooms()[0]
                ),
                float(
                    segmentation.header.get_zooms()[1]
                ),
                float(
                    segmentation.header.get_zooms()[2]
                )
            )

            result[
                "voxel_spacing_x_mm"
            ] = spacing[0]

            result[
                "voxel_spacing_y_mm"
            ] = spacing[1]

            result[
                "voxel_spacing_z_mm"
            ] = spacing[2]

            voxel_volume = (
                calculate_volume_from_voxels(
                    1,
                    spacing
                )
            )

            result[
                "voxel_volume_mm3"
            ] = voxel_volume

            # ------------------------------------------------
            # MEMORY-EFFICIENT MASK ACCESS
            # ------------------------------------------------

            """
            IMPORTANT:

            Do NOT use:

                segmentation.get_fdata()

            because that converts the mask to float64
            and can consume hundreds of MB.

            Instead, load the underlying mask array.
            """

            mask = np.asanyarray(
                segmentation.dataobj
            )

            # Remove singleton dimensions if present.
            mask = np.squeeze(
                mask
            )

            if mask.ndim != 3:

                raise ValueError(
                    "Segmentation is not 3D: {}".format(
                        mask.shape
                    )
                )

            # ------------------------------------------------
            # Non-zero voxels
            # ------------------------------------------------

            nonzero = (
                mask != 0
            )

            voxel_count = int(
                np.count_nonzero(
                    nonzero
                )
            )

            result[
                "mask_voxels"
            ] = voxel_count

            if voxel_count <= 0:

                raise ValueError(
                    "Segmentation contains zero non-zero voxels."
                )

            # ------------------------------------------------
            # Tumor volume
            # ------------------------------------------------

            tumor_volume_mm3 = (
                voxel_count *
                voxel_volume
            )

            tumor_volume_cm3 = (
                tumor_volume_mm3 /
                1000.0
            )

            result[
                "tumor_volume_mm3"
            ] = tumor_volume_mm3

            result[
                "tumor_volume_cm3"
            ] = tumor_volume_cm3

            # ------------------------------------------------
            # Equivalent diameter
            # ------------------------------------------------

            equivalent_diameter = (
                calculate_equivalent_diameter(
                    tumor_volume_mm3
                )
            )

            result[
                "equivalent_diameter_mm"
            ] = equivalent_diameter

            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            bbox_dimensions = (
                calculate_bbox_dimensions(
                    mask,
                    spacing
                )
            )

            if bbox_dimensions is not None:

                result[
                    "bbox_x_mm"
                ] = bbox_dimensions[0]

                result[
                    "bbox_y_mm"
                ] = bbox_dimensions[1]

                result[
                    "bbox_z_mm"
                ] = bbox_dimensions[2]

            # ------------------------------------------------
            # Centroid
            # ------------------------------------------------

            centroid_voxel = (
                calculate_centroid_voxel(
                    mask
                )
            )

            if centroid_voxel is not None:

                result[
                    "centroid_voxel_x"
                ] = centroid_voxel[0]

                result[
                    "centroid_voxel_y"
                ] = centroid_voxel[1]

                result[
                    "centroid_voxel_z"
                ] = centroid_voxel[2]

                centroid_world = (
                    voxel_to_world(
                        segmentation.affine,
                        centroid_voxel
                    )
                )

                if centroid_world is not None:

                    result[
                        "centroid_world_x_mm"
                    ] = centroid_world[0]

                    result[
                        "centroid_world_y_mm"
                    ] = centroid_world[1]

                    result[
                        "centroid_world_z_mm"
                    ] = centroid_world[2]

            result[
                "status"
            ] = "SUCCESS"

            successful += 1

            print(
                "Volume: {:.3f} cm3".format(
                    tumor_volume_cm3
                )
            )

            print(
                "Equivalent diameter: {:.3f} mm".format(
                    equivalent_diameter
                )
            )

        except Exception as error:

            failed += 1

            result[
                "error"
            ] = str(error)

            print(
                "ERROR:",
                str(error)
            )

        geometry_results.append(
            result
        )

        # ----------------------------------------------------
        # Free memory
        # ----------------------------------------------------

        try:

            del mask

        except Exception:
            pass

        try:

            del segmentation

        except Exception:
            pass

        # ----------------------------------------------------
        # Remove temporary files
        # ----------------------------------------------------

        try:

            if os.path.exists(
                current_dir
            ):

                shutil.rmtree(
                    current_dir
                )

        except Exception:
            pass

    # ========================================================
    # CLOSE ZIP
    # ========================================================

    archive.close()

    # ========================================================
    # SAVE GEOMETRY CSV
    # ========================================================

    geometry_columns = [

        "patient_id",
        "week",

        "segmentation_source",
        "segmentation_file",

        "voxel_spacing_x_mm",
        "voxel_spacing_y_mm",
        "voxel_spacing_z_mm",

        "voxel_volume_mm3",

        "mask_voxels",

        "tumor_volume_mm3",
        "tumor_volume_cm3",

        "equivalent_diameter_mm",

        "bbox_x_mm",
        "bbox_y_mm",
        "bbox_z_mm",

        "centroid_voxel_x",
        "centroid_voxel_y",
        "centroid_voxel_z",

        "centroid_world_x_mm",
        "centroid_world_y_mm",
        "centroid_world_z_mm",

        "status",
        "error"
    ]

    with open(
        GEOMETRY_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=geometry_columns
        )

        writer.writeheader()

        for result in geometry_results:

            writer.writerow(
                result
            )

    # ========================================================
    # BUILD PATIENT TRAJECTORIES
    # ========================================================

    print(
        "\nBuilding longitudinal HBGD trajectories..."
    )

    patient_data = {}

    for row in geometry_results:

        if row[
            "status"
        ] != "SUCCESS":

            continue

        patient_id = row[
            "patient_id"
        ]

        if patient_id not in patient_data:

            patient_data[
                patient_id
            ] = []

        patient_data[
            patient_id
        ].append(
            row
        )

    # ========================================================
    # HBGD GROWTH FEATURES
    # ========================================================

    hbgd_results = []

    trajectory_results = []

    patients_with_growth = 0

    total_intervals = 0

    growth_states = {
        "GROWTH": 0,
        "STABLE": 0,
        "SHRINKAGE": 0,
        "UNKNOWN": 0
    }

    # --------------------------------------------------------
    # Patient loop
    # --------------------------------------------------------

    for patient_id in sorted(
        patient_data.keys()
    ):

        observations = sorted(
            patient_data[
                patient_id
            ],
            key=lambda x: int(
                x["week"]
            )
        )

        # ----------------------------------------------------
        # Save trajectory observations
        # ----------------------------------------------------

        for observation in observations:

            trajectory_results.append({

                "patient_id":
                    patient_id,

                "week":
                    observation["week"],

                "tumor_volume_cm3":
                    observation[
                        "tumor_volume_cm3"
                    ],

                "equivalent_diameter_mm":
                    observation[
                        "equivalent_diameter_mm"
                    ],

                "bbox_x_mm":
                    observation[
                        "bbox_x_mm"
                    ],

                "bbox_y_mm":
                    observation[
                        "bbox_y_mm"
                    ],

                "bbox_z_mm":
                    observation[
                        "bbox_z_mm"
                    ]

            })

        # ----------------------------------------------------
        # Growth intervals
        # ----------------------------------------------------

        if len(observations) < 2:

            continue

        patients_with_growth += 1

        for i in range(
            len(observations) - 1
        ):

            first = observations[i]

            second = observations[i + 1]

            week1 = int(
                first["week"]
            )

            week2 = int(
                second["week"]
            )

            delta_weeks = (
                week2 - week1
            )

            if delta_weeks <= 0:

                continue

            volume1 = float(
                first[
                    "tumor_volume_mm3"
                ]
            )

            volume2 = float(
                second[
                    "tumor_volume_mm3"
                ]
            )

            diameter1 = float(
                first[
                    "equivalent_diameter_mm"
                ]
            )

            diameter2 = float(
                second[
                    "equivalent_diameter_mm"
                ]
            )

            delta_days = (
                delta_weeks * 7.0
            )

            # ------------------------------------------------
            # HBGD rates
            # ------------------------------------------------

            r_diameter = (
                calculate_growth_rate(
                    diameter1,
                    diameter2,
                    delta_weeks
                )
            )

            r_volume = (
                calculate_volume_growth_rate(
                    volume1,
                    volume2,
                    delta_weeks
                )
            )

            # ------------------------------------------------
            # Expected theoretical relationship
            #
            # V proportional to D^3
            #
            # Therefore approximately:
            #
            # r_volume = 3 * r_diameter
            # ------------------------------------------------

            ratio_difference = float(
                "nan"
            )

            if (
                math.isfinite(
                    r_diameter
                )
                and
                math.isfinite(
                    r_volume
                )
            ):

                if abs(
                    r_diameter
                ) > 1e-12:

                    ratio_difference = (
                        r_volume /
                        r_diameter
                    )

            # ------------------------------------------------
            # State
            # ------------------------------------------------

            state = classify_growth_state(
                r_diameter
            )

            growth_states[
                state
            ] += 1

            total_intervals += 1

            # ------------------------------------------------
            # Percentage changes
            # ------------------------------------------------

            diameter_percent_change = (
                (
                    diameter2 -
                    diameter1
                )
                /
                diameter1
            ) * 100.0

            volume_percent_change = (
                (
                    volume2 -
                    volume1
                )
                /
                volume1
            ) * 100.0

            # ------------------------------------------------
            # HBGD row
            # ------------------------------------------------

            hbgd_results.append({

                "patient_id":
                    patient_id,

                "week_start":
                    week1,

                "week_end":
                    week2,

                "delta_weeks":
                    delta_weeks,

                "delta_days":
                    delta_days,

                "volume_start_mm3":
                    volume1,

                "volume_end_mm3":
                    volume2,

                "volume_start_cm3":
                    volume1 / 1000.0,

                "volume_end_cm3":
                    volume2 / 1000.0,

                "diameter_start_mm":
                    diameter1,

                "diameter_end_mm":
                    diameter2,

                "diameter_change_mm":
                    diameter2 - diameter1,

                "diameter_percent_change":
                    diameter_percent_change,

                "volume_percent_change":
                    volume_percent_change,

                "r_diameter_per_day":
                    r_diameter,

                "r_volume_per_day":
                    r_volume,

                "r_volume_div_r_diameter":
                    ratio_difference,

                "growth_state":
                    state

            })

    # ========================================================
    # SAVE HBGD CSV
    # ========================================================

    hbgd_columns = [

        "patient_id",

        "week_start",
        "week_end",

        "delta_weeks",
        "delta_days",

        "volume_start_mm3",
        "volume_end_mm3",

        "volume_start_cm3",
        "volume_end_cm3",

        "diameter_start_mm",
        "diameter_end_mm",

        "diameter_change_mm",

        "diameter_percent_change",

        "volume_percent_change",

        "r_diameter_per_day",
        "r_volume_per_day",

        "r_volume_div_r_diameter",

        "growth_state"
    ]

    with open(
        HBGD_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=hbgd_columns
        )

        writer.writeheader()

        for row in hbgd_results:

            writer.writerow(
                row
            )

    # ========================================================
    # SAVE LONGITUDINAL TRAJECTORIES
    # ========================================================

    trajectory_columns = [

        "patient_id",
        "week",

        "tumor_volume_cm3",

        "equivalent_diameter_mm",

        "bbox_x_mm",
        "bbox_y_mm",
        "bbox_z_mm"
    ]

    with open(
        TRAJECTORY_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=trajectory_columns
        )

        writer.writeheader()

        for row in trajectory_results:

            writer.writerow(
                row
            )

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    report.append(
        "# LUMIERE TUMOR GEOMETRY + HBGD FEATURE EXTRACTION"
    )

    report.append("")

    report.append(
        "This stage converts validated LUMIERE tumor segmentation masks into tumor geometry and longitudinal HBGD growth features."
    )

    report.append("")

    report.append(
        "## 1. Input"
    )

    report.append("")

    report.append(
        "| Metric | Value |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        "| Step 2 VALID timepoints | {} |".format(
            len(validated_rows)
        )
    )

    report.append(
        "| Geometry extraction successes | {} |".format(
            successful
        )
    )

    report.append(
        "| Geometry extraction failures | {} |".format(
            failed
        )
    )

    report.append("")

    # ========================================================
    # VOLUME SUMMARY
    # ========================================================

    volumes = []

    diameters = []

    for row in geometry_results:

        if row[
            "status"
        ] != "SUCCESS":

            continue

        try:

            volume = float(
                row[
                    "tumor_volume_cm3"
                ]
            )

            diameter = float(
                row[
                    "equivalent_diameter_mm"
                ]
            )

            if (
                math.isfinite(volume)
                and volume > 0
            ):

                volumes.append(
                    volume
                )

            if (
                math.isfinite(diameter)
                and diameter > 0
            ):

                diameters.append(
                    diameter
                )

        except Exception:

            pass

    report.append(
        "## 2. Tumor Geometry"
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
            "| Valid tumor volumes | {} |".format(
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

    if diameters:

        report.append(
            "| Minimum equivalent diameter | {:.3f} mm |".format(
                min(diameters)
            )
        )

        report.append(
            "| Maximum equivalent diameter | {:.3f} mm |".format(
                max(diameters)
            )
        )

        report.append(
            "| Mean equivalent diameter | {:.3f} mm |".format(
                sum(diameters) /
                len(diameters)
            )
        )

    report.append("")

    # ========================================================
    # LONGITUDINAL
    # ========================================================

    report.append(
        "## 3. Longitudinal HBGD Features"
    )

    report.append("")

    report.append(
        "| Metric | Value |"
    )

    report.append(
        "|---|---:|"
    )

    report.append(
        "| Patients with >=2 successful timepoints | {} |".format(
            patients_with_growth
        )
    )

    report.append(
        "| Longitudinal intervals | {} |".format(
            total_intervals
        )
    )

    report.append("")

    # ========================================================
    # GROWTH STATES
    # ========================================================

    report.append(
        "## 4. HBGD Growth-State Distribution"
    )

    report.append("")

    report.append(
        "| State | Intervals | Percentage |"
    )

    report.append(
        "|---|---:|---:|"
    )

    for state in [
        "GROWTH",
        "STABLE",
        "SHRINKAGE",
        "UNKNOWN"
    ]:

        count = growth_states[
            state
        ]

        if total_intervals > 0:

            percentage = (
                float(count) /
                float(total_intervals)
            ) * 100.0

        else:

            percentage = 0.0

        report.append(
            "| {} | {} | {:.2f}% |".format(
                state,
                count,
                percentage
            )
        )

    report.append("")

    # ========================================================
    # HBGD FORMULA
    # ========================================================

    report.append(
        "## 5. HBGD Mathematical Formulation"
    )

    report.append("")

    report.append(
        "For equivalent tumor diameter D:"
    )

    report.append("")

    report.append(
        "```text"
    )

    report.append(
        "r_D = ln(D2 / D1) / Δt"
    )

    report.append(
        "```"
    )

    report.append("")

    report.append(
        "where Δt is represented in days using the LUMIERE week identifier multiplied by 7."
    )

    report.append("")

    report.append(
        "Tumor volume is calculated directly from segmentation voxel count:"
    )

    report.append("")

    report.append(
        "```text"
    )

    report.append(
        "V = N_voxels × voxel_volume"
    )

    report.append(
        "```"
    )

    report.append("")

    report.append(
        "Equivalent spherical diameter is calculated using:"
    )

    report.append("")

    report.append(
        "```text"
    )

    report.append(
        "D = (6V / π)^(1/3)"
    )

    report.append(
        "```"
    )

    report.append("")

    report.append(
        "Because V is proportional to D³ under the equivalent-sphere representation, the theoretical relationship is approximately r_V = 3r_D."
    )

    report.append("")

    # ========================================================
    # IMPORTANT LIMITATION
    # ========================================================

    report.append(
        "## 6. Important Temporal Assumption"
    )

    report.append("")

    report.append(
        "This analysis currently uses the difference between LUMIERE week identifiers as the longitudinal time interval and converts weeks to days using 7 days per week."
    )

    report.append("")

    report.append(
        "This should be confirmed against the official LUMIERE dataset documentation before the resulting growth rates are used as final biological estimates."
    )

    report.append("")

    # ========================================================
    # NEXT STEP
    # ========================================================

    report.append(
        "## 7. Next Step"
    )

    report.append("")

    report.append(
        "The resulting longitudinal HBGD features can now be used to evaluate within-LUMIERE forward prediction before performing the cross-cancer transfer experiment."
    )

    report.append("")

    report.append(
        "The next stage should perform patient-level train/validation/test splitting and predict future tumor diameter or volume from previous longitudinal observations."
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
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STEP 3 COMPLETE")
    print("=" * 70)

    print(
        "\nGeometry successes:",
        successful
    )

    print(
        "Geometry failures:",
        failed
    )

    print(
        "Patients with longitudinal data:",
        patients_with_growth
    )

    print(
        "HBGD intervals:",
        total_intervals
    )

    print("\nGROWTH STATES:")

    print(
        "Growth:",
        growth_states["GROWTH"]
    )

    print(
        "Stable:",
        growth_states["STABLE"]
    )

    print(
        "Shrinkage:",
        growth_states["SHRINKAGE"]
    )

    print("\nOUTPUT FILES:")

    print(
        GEOMETRY_CSV
    )

    print(
        HBGD_CSV
    )

    print(
        TRAJECTORY_CSV
    )

    print(
        REPORT_PATH
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