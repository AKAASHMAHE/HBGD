import os
import sys
import json
import numpy as np
import pandas as pd
import nibabel as nib


# ============================================================
# WINDOWS TERMINAL FIX
# ============================================================

try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )
except Exception:
    pass


# ============================================================
# PATHS
# ============================================================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "DATA"
)

SERIES_FILE = os.path.join(
    DATA_DIR,
    "series.json"
)

RECIST_FILE = os.path.join(
    DATA_DIR,
    "recist_measurements.csv"
)


# ============================================================
# PRINT HEADER
# ============================================================

def header(text):
    print()
    print("=" * 90)
    print(text)
    print("=" * 90)


# ============================================================
# FIND CT AND MASK FILES
# ============================================================

def find_nifti_files():

    image_files = []
    mask_files = []

    for root, dirs, files in os.walk(DATA_DIR):

        for filename in files:

            if not (
                filename.endswith(".nii") or
                filename.endswith(".nii.gz")
            ):
                continue

            full_path = os.path.join(
                root,
                filename
            )

            # Get the folder immediately containing the file.
            parent_folder = os.path.basename(
                os.path.dirname(full_path)
            ).lower()

            if parent_folder == "images":

                image_files.append(
                    full_path
                )

            elif parent_folder == "masks":

                mask_files.append(
                    full_path
                )

    return image_files, mask_files


# ============================================================
# LOAD METADATA
# ============================================================

header("LOADING DATA")

if not os.path.exists(SERIES_FILE):

    print(
        "ERROR: series.json was not found:"
    )

    print(
        SERIES_FILE
    )

    sys.exit(1)


if not os.path.exists(RECIST_FILE):

    print(
        "ERROR: recist_measurements.csv was not found:"
    )

    print(
        RECIST_FILE
    )

    sys.exit(1)


with open(
    SERIES_FILE,
    "r",
    encoding="utf-8"
) as file:

    series_data = json.load(
        file
    )


recist = pd.read_csv(
    RECIST_FILE
)

series_df = pd.DataFrame(
    series_data
)


print(
    "Series metadata records:",
    len(series_df)
)

print(
    "RECIST records:",
    len(recist)
)


# ============================================================
# FIND FILES
# ============================================================

header("SEARCHING FOR CT AND MASK FILES")

image_files, mask_files = (
    find_nifti_files()
)


print(
    "CT image files:",
    len(image_files)
)

print(
    "Mask files:",
    len(mask_files)
)


# ============================================================
# SHOW FILE EXAMPLES
# ============================================================

header("FILE EXAMPLES")

print()
print("CT IMAGE EXAMPLES:")

for path in image_files[:5]:

    print(
        " ",
        os.path.relpath(
            path,
            DATA_DIR
        )
    )


print()
print("MASK EXAMPLES:")

for path in mask_files[:5]:

    print(
        " ",
        os.path.relpath(
            path,
            DATA_DIR
        )
    )


# ============================================================
# CREATE LOOKUP DICTIONARIES
# ============================================================

image_dict = {}

for path in image_files:

    filename = os.path.basename(
        path
    )

    image_dict[filename] = path


mask_dict = {}

for path in mask_files:

    filename = os.path.basename(
        path
    )

    mask_dict[filename] = path


image_names = set(
    image_dict.keys()
)

mask_names = set(
    mask_dict.keys()
)


# ============================================================
# IMAGE / MASK MATCHING
# ============================================================

header("CT IMAGE - MASK MATCHING")

matched_names = (
    image_names &
    mask_names
)

images_without_masks = (
    image_names -
    mask_names
)

masks_without_images = (
    mask_names -
    image_names
)


print(
    "Unique CT images:",
    len(image_names)
)

print(
    "Unique masks:",
    len(mask_names)
)

print(
    "Matching image/mask pairs:",
    len(matched_names)
)

print(
    "Images without masks:",
    len(images_without_masks)
)

print(
    "Masks without images:",
    len(masks_without_images)
)


# ============================================================
# RECIST FILE MATCHING
# ============================================================

header("RECIST - CT FILE MATCHING")

recist_filenames = set(
    recist[
        "filename"
    ].astype(str)
)

recist_ct_matches = (
    recist_filenames &
    image_names
)

recist_ct_missing = (
    recist_filenames -
    image_names
)


print(
    "Unique RECIST filenames:",
    len(recist_filenames)
)

print(
    "RECIST filenames matching CT:",
    len(recist_ct_matches)
)

print(
    "RECIST filenames without CT:",
    len(recist_ct_missing)
)


# ============================================================
# VERIFY CT + MASK PAIRS
# ============================================================

header("VERIFYING CT AND MASK PAIRS")

verification_results = []

failed_files = []

sorted_matches = sorted(
    matched_names
)

total_pairs = len(
    sorted_matches
)


for number, filename in enumerate(
    sorted_matches,
    start=1
):

    print()
    print(
        "[" +
        str(number) +
        "/" +
        str(total_pairs) +
        "]"
    )

    print(
        filename
    )

    image_path = image_dict[
        filename
    ]

    mask_path = mask_dict[
        filename
    ]

    try:

        # ----------------------------------------------------
        # LOAD CT
        # ----------------------------------------------------

        ct_img = nib.load(
            image_path
        )

        # ----------------------------------------------------
        # LOAD MASK
        # ----------------------------------------------------

        mask_img = nib.load(
            mask_path
        )

        # ----------------------------------------------------
        # SHAPE
        # ----------------------------------------------------

        ct_shape = ct_img.shape

        mask_shape = mask_img.shape

        shape_match = (
            ct_shape ==
            mask_shape
        )

        # ----------------------------------------------------
        # SPACING
        # ----------------------------------------------------

        ct_spacing = (
            ct_img.header.get_zooms()
        )

        mask_spacing = (
            mask_img.header.get_zooms()
        )

        spacing_match = np.allclose(
            ct_spacing[:3],
            mask_spacing[:3],
            atol=1e-5
        )

        # ----------------------------------------------------
        # AFFINE
        # ----------------------------------------------------

        affine_match = np.allclose(
            ct_img.affine,
            mask_img.affine,
            atol=1e-4
        )

        # ----------------------------------------------------
        # MASK DATA
        # ----------------------------------------------------

        mask_data = (
            mask_img.get_fdata()
        )

        unique_values = np.unique(
            mask_data
        )

        foreground_values = (
            unique_values[
                unique_values != 0
            ]
        )

        number_of_labels = len(
            foreground_values
        )

        lesion_voxels = int(
            np.sum(
                mask_data > 0
            )
        )

        # ----------------------------------------------------
        # VOXEL VOLUME
        # ----------------------------------------------------

        voxel_volume_mm3 = (
            ct_spacing[0]
            *
            ct_spacing[1]
            *
            ct_spacing[2]
        )

        lesion_volume_mm3 = (
            lesion_voxels
            *
            voxel_volume_mm3
        )

        lesion_volume_cm3 = (
            lesion_volume_mm3 /
            1000.0
        )

        # ----------------------------------------------------
        # CT STATISTICS
        # ----------------------------------------------------

        ct_data = ct_img.get_fdata(
            dtype=np.float32
        )

        finite_values = (
            ct_data[
                np.isfinite(ct_data)
            ]
        )

        if len(finite_values) > 0:

            ct_min = float(
                np.min(finite_values)
            )

            ct_max = float(
                np.max(finite_values)
            )

            ct_mean = float(
                np.mean(finite_values)
            )

        else:

            ct_min = np.nan
            ct_max = np.nan
            ct_mean = np.nan

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            "  CT shape:",
            ct_shape
        )

        print(
            "  Mask shape:",
            mask_shape
        )

        print(
            "  Shape match:",
            shape_match
        )

        print(
            "  CT spacing:",
            ct_spacing[:3]
        )

        print(
            "  Mask spacing:",
            mask_spacing[:3]
        )

        print(
            "  Spacing match:",
            spacing_match
        )

        print(
            "  Affine match:",
            affine_match
        )

        print(
            "  Mask labels:",
            foreground_values.tolist()
        )

        print(
            "  Number of labels:",
            number_of_labels
        )

        print(
            "  Lesion voxels:",
            lesion_voxels
        )

        print(
            "  Lesion volume:",
            round(
                lesion_volume_cm3,
                3
            ),
            "cm3"
        )

        print(
            "  CT range:",
            round(ct_min, 2),
            "to",
            round(ct_max, 2)
        )

        verification_results.append({

            "filename":
                filename,

            "ct_shape":
                str(ct_shape),

            "mask_shape":
                str(mask_shape),

            "shape_match":
                shape_match,

            "ct_spacing_x":
                ct_spacing[0],

            "ct_spacing_y":
                ct_spacing[1],

            "ct_spacing_z":
                ct_spacing[2],

            "spacing_match":
                spacing_match,

            "affine_match":
                affine_match,

            "number_of_labels":
                number_of_labels,

            "foreground_labels":
                ",".join(
                    str(int(x))
                    for x in foreground_values
                ),

            "lesion_voxels":
                lesion_voxels,

            "lesion_volume_cm3":
                lesion_volume_cm3,

            "ct_min":
                ct_min,

            "ct_max":
                ct_max,

            "ct_mean":
                ct_mean,

            "has_lesion":
                lesion_voxels > 0
        })

    except Exception as error:

        print(
            "  ERROR:",
            str(error)
        )

        failed_files.append({

            "filename":
                filename,

            "error":
                str(error)
        })


# ============================================================
# VERIFICATION SUMMARY
# ============================================================

header("CT + MASK VERIFICATION SUMMARY")

verification_df = pd.DataFrame(
    verification_results
)

print(
    "Successfully loaded pairs:",
    len(verification_results)
)

print(
    "Failed pairs:",
    len(failed_files)
)


if len(verification_df) > 0:

    shape_errors = int(
        (
            ~verification_df[
                "shape_match"
            ]
        ).sum()
    )

    spacing_errors = int(
        (
            ~verification_df[
                "spacing_match"
            ]
        ).sum()
    )

    affine_errors = int(
        (
            ~verification_df[
                "affine_match"
            ]
        ).sum()
    )

    lesion_volumes = int(
        verification_df[
            "has_lesion"
        ].sum()
    )

    print(
        "Shape mismatches:",
        shape_errors
    )

    print(
        "Spacing mismatches:",
        spacing_errors
    )

    print(
        "Affine mismatches:",
        affine_errors
    )

    print(
        "Volumes containing lesions:",
        lesion_volumes
    )


# ============================================================
# RECIST LABEL CHECK
# ============================================================

header("RECIST LABEL - MASK VERIFICATION")

recist_results = []


for _, row in recist.iterrows():

    filename = str(
        row["filename"]
    )

    label_value = int(
        row["lesion_label_value"]
    )

    mask_path = mask_dict.get(
        filename
    )

    mask_found = (
        mask_path is not None
    )

    label_found = False

    if mask_found:

        try:

            mask_img = nib.load(
                mask_path
            )

            mask_data = (
                mask_img.get_fdata()
            )

            label_found = bool(
                np.any(
                    mask_data ==
                    label_value
                )
            )

        except Exception:

            label_found = False


    recist_results.append({

        "patient_id":
            row["patient_id"],

        "study_date":
            row["study_date"],

        "filename":
            filename,

        "lesion_label_value":
            label_value,

        "lesion_label_alias":
            row["lesion_label_alias"],

        "recist_measurement_mm":
            row["recist_measurement_mm"],

        "mask_found":
            mask_found,

        "label_found":
            label_found
    })


recist_check_df = pd.DataFrame(
    recist_results
)


masks_found_count = int(
    recist_check_df[
        "mask_found"
    ].sum()
)

labels_found_count = int(
    recist_check_df[
        "label_found"
    ].sum()
)


print(
    "RECIST records checked:",
    len(recist_check_df)
)

print(
    "Masks found:",
    masks_found_count
)

print(
    "RECIST labels found in masks:",
    labels_found_count
)


# ============================================================
# RECIST LESION VOLUMES
# ============================================================

header("RECIST LESION VOLUME EXTRACTION")

measurement_results = []


for _, row in recist.iterrows():

    filename = str(
        row["filename"]
    )

    label_value = int(
        row["lesion_label_value"]
    )

    mask_path = mask_dict.get(
        filename
    )

    if mask_path is None:

        continue


    try:

        mask_img = nib.load(
            mask_path
        )

        mask_data = (
            mask_img.get_fdata()
        )

        spacing = (
            mask_img.header.get_zooms()
        )

        voxel_volume_mm3 = (
            spacing[0]
            *
            spacing[1]
            *
            spacing[2]
        )

        lesion_voxels = int(
            np.sum(
                mask_data ==
                label_value
            )
        )

        lesion_volume_mm3 = (
            lesion_voxels
            *
            voxel_volume_mm3
        )

        lesion_volume_cm3 = (
            lesion_volume_mm3 /
            1000.0
        )


        measurement_results.append({

            "patient_id":
                row["patient_id"],

            "study_date":
                row["study_date"],

            "study_order":
                row["study_order"],

            "region":
                row["region"],

            "filename":
                filename,

            "lesion_label_alias":
                row["lesion_label_alias"],

            "lesion_label_value":
                label_value,

            "recist_measurement_mm":
                row["recist_measurement_mm"],

            "lesion_voxels":
                lesion_voxels,

            "lesion_volume_cm3":
                lesion_volume_cm3,

            "label_exists":
                lesion_voxels > 0
        })


    except Exception as error:

        print(
            "Error processing:",
            filename
        )

        print(
            str(error)
        )


measurement_df = pd.DataFrame(
    measurement_results
)


print(
    "RECIST records linked to masks:",
    len(measurement_df)
)


# ============================================================
# SAVE RESULTS
# ============================================================

header("SAVING RESULTS")

verification_file = os.path.join(
    DATA_DIR,
    "5_ct_mask_verification.csv"
)

recist_file = os.path.join(
    DATA_DIR,
    "5_recist_mask_verification.csv"
)

measurement_file = os.path.join(
    DATA_DIR,
    "5_recist_mask_measurements.csv"
)


verification_df.to_csv(
    verification_file,
    index=False
)

recist_check_df.to_csv(
    recist_file,
    index=False
)

measurement_df.to_csv(
    measurement_file,
    index=False
)


print(
    "Saved:",
    verification_file
)

print(
    "Saved:",
    recist_file
)

print(
    "Saved:",
    measurement_file
)


# ============================================================
# FINAL FEASIBILITY
# ============================================================

header("STEP 5 - FINAL FEASIBILITY")

all_shapes_match = False
all_spacing_match = False
all_affines_match = False
any_lesions = False
all_recist_labels_found = False


if len(verification_df) > 0:

    all_shapes_match = bool(
        verification_df[
            "shape_match"
        ].all()
    )

    all_spacing_match = bool(
        verification_df[
            "spacing_match"
        ].all()
    )

    all_affines_match = bool(
        verification_df[
            "affine_match"
        ].all()
    )

    any_lesions = bool(
        verification_df[
            "has_lesion"
        ].any()
    )


if len(recist_check_df) > 0:

    all_recist_labels_found = bool(
        recist_check_df[
            "label_found"
        ].all()
    )


checks = [

    (
        "CT/mask pairs found",
        len(matched_names) > 0
    ),

    (
        "All CT images have masks",
        len(images_without_masks) == 0
    ),

    (
        "All masks have CT images",
        len(masks_without_images) == 0
    ),

    (
        "All CT/mask shapes match",
        all_shapes_match
    ),

    (
        "All CT/mask spacing matches",
        all_spacing_match
    ),

    (
        "All CT/mask affines match",
        all_affines_match
    ),

    (
        "At least one mask contains lesions",
        any_lesions
    ),

    (
        "All RECIST labels found in masks",
        all_recist_labels_found
    )
]


for check_name, result in checks:

    if result:

        status = "YES"

    else:

        status = "NO"


    print(
        "[" +
        status +
        "] " +
        check_name
    )


# ============================================================
# FINISH
# ============================================================

header("STEP 5 COMPLETE")

print("""
IMPORTANT:

Do not train a model yet.

This step verifies:

    CT volume
        +
    segmentation mask
        +
    lesion labels
        +
    RECIST measurements
        +
    voxel spacing

Run:

    python 5_verify_ct_masks.py

Then send me the COMPLETE terminal output.
""")


print("=" * 90)
print("END OF STEP 5")
print("=" * 90)