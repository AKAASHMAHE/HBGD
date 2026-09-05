# 7_analyze_lesion_correspondence.py

import os
import sys
import json
import numpy as np
import pandas as pd
import nibabel as nib

from scipy import ndimage


# ============================================================
# TERMINAL ENCODING
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
# HEADER
# ============================================================

def header(text):

    print()
    print("=" * 90)
    print(text)
    print("=" * 90)


# ============================================================
# FIND MASK FILES
# ============================================================

def find_mask_files():

    mask_files = {}

    for root, dirs, files in os.walk(DATA_DIR):

        for filename in files:

            if not (
                filename.endswith(".nii")
                or
                filename.endswith(".nii.gz")
            ):
                continue

            full_path = os.path.join(
                root,
                filename
            )

            parent = os.path.basename(
                os.path.dirname(full_path)
            ).lower()

            if parent == "masks":

                mask_files[filename] = full_path

    return mask_files


# ============================================================
# EXTRACT LABEL GEOMETRY
# ============================================================

def extract_label_geometry(
    mask_path
):

    img = nib.load(
        mask_path
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # We load the mask only.
    # CT volume is never loaded.
    # --------------------------------------------------------

    mask = np.asanyarray(
        img.dataobj
    )

    mask = np.asarray(
        mask
    )

    affine = img.affine

    spacing = np.asarray(
        img.header.get_zooms()[:3],
        dtype=float
    )

    voxel_volume_mm3 = (
        spacing[0]
        *
        spacing[1]
        *
        spacing[2]
    )

    # --------------------------------------------------------
    # FIND NON-ZERO LABELS
    # --------------------------------------------------------

    labels = np.unique(
        mask
    )

    labels = labels[
        labels != 0
    ]

    results = []

    # --------------------------------------------------------
    # PROCESS EACH LABEL
    # --------------------------------------------------------

    for label in labels:

        binary = (
            mask == label
        )

        voxel_count = int(
            binary.sum()
        )

        if voxel_count == 0:
            continue

        # ----------------------------------------------------
        # CENTER OF MASS
        # ----------------------------------------------------

        centroid_voxel = (
            ndimage.center_of_mass(
                binary
            )
        )

        centroid_voxel = np.asarray(
            centroid_voxel,
            dtype=float
        )

        centroid_world_h = (
            affine
            @
            np.append(
                centroid_voxel,
                1.0
            )
        )

        centroid_world = (
            centroid_world_h[:3]
        )

        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        object_slices = (
            ndimage.find_objects(
                binary
            )
        )

        if (
            object_slices is not None
            and
            len(object_slices) > 0
            and
            object_slices[0] is not None
        ):

            slices = object_slices[0]

            min_x = slices[0].start
            max_x = slices[0].stop - 1

            min_y = slices[1].start
            max_y = slices[1].stop - 1

            min_z = slices[2].start
            max_z = slices[2].stop - 1

        else:

            min_x = 0
            max_x = 0
            min_y = 0
            max_y = 0
            min_z = 0
            max_z = 0

        # ----------------------------------------------------
        # VOLUME
        # ----------------------------------------------------

        volume_mm3 = (
            voxel_count
            *
            voxel_volume_mm3
        )

        volume_cm3 = (
            volume_mm3
            /
            1000.0
        )

        results.append({

            "mask_label":
                int(label),

            "voxel_count":
                voxel_count,

            "volume_cm3":
                float(volume_cm3),

            "centroid_x":
                float(
                    centroid_world[0]
                ),

            "centroid_y":
                float(
                    centroid_world[1]
                ),

            "centroid_z":
                float(
                    centroid_world[2]
                ),

            "min_x":
                int(min_x),

            "min_y":
                int(min_y),

            "min_z":
                int(min_z),

            "max_x":
                int(max_x),

            "max_y":
                int(max_y),

            "max_z":
                int(max_z)
        })

    return results


# ============================================================
# CENTROID DISTANCE
# ============================================================

def centroid_distance(
    lesion_a,
    lesion_b
):

    point_a = np.array([
        lesion_a["centroid_x"],
        lesion_a["centroid_y"],
        lesion_a["centroid_z"]
    ])

    point_b = np.array([
        lesion_b["centroid_x"],
        lesion_b["centroid_y"],
        lesion_b["centroid_z"]
    ])

    return float(
        np.linalg.norm(
            point_a -
            point_b
        )
    )


# ============================================================
# LOAD METADATA
# ============================================================

header(
    "STEP 7 - LOADING DATA"
)

if not os.path.exists(
    SERIES_FILE
):

    print(
        "ERROR: series.json not found:"
    )

    print(
        SERIES_FILE
    )

    sys.exit(1)


if not os.path.exists(
    RECIST_FILE
):

    print(
        "ERROR: recist_measurements.csv not found:"
    )

    print(
        RECIST_FILE
    )

    sys.exit(1)


with open(
    SERIES_FILE,
    "r",
    encoding="utf-8"
) as f:

    series_data = json.load(
        f
    )


series_df = pd.DataFrame(
    series_data
)

recist_df = pd.read_csv(
    RECIST_FILE
)


# ============================================================
# DATE CONVERSION
# ============================================================

series_df["date_parsed"] = (
    pd.to_datetime(
        series_df[
            "study_date"
        ].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )
)

recist_df["date_parsed"] = (
    pd.to_datetime(
        recist_df[
            "study_date"
        ].astype(str),
        format="%Y%m%d",
        errors="coerce"
    )
)


print(
    "Series records:",
    len(series_df)
)

print(
    "RECIST records:",
    len(recist_df)
)


# ============================================================
# FIND MASK FILES
# ============================================================

header(
    "FINDING SEGMENTATION MASKS"
)

mask_files = find_mask_files()

print(
    "Mask files found:",
    len(mask_files)
)


# ============================================================
# CONNECT SERIES TO MASKS
# ============================================================

header(
    "CONNECTING SERIES TO MASKS"
)

series_lookup = []

mask_names = set(
    mask_files.keys()
)


for _, row in series_df.iterrows():

    uuid = str(
        row["uuid"]
    )

    matching_filename = None

    for filename in mask_names:

        if uuid in filename:

            matching_filename = (
                filename
            )

            break

    if matching_filename is not None:

        series_lookup.append({

            "patient_id":
                int(
                    row["patient_id"]
                ),

            "study_date":
                row["date_parsed"],

            "study_date_string":
                str(
                    row["study_date"]
                ),

            "region":
                row["region"],

            "uuid":
                uuid,

            "filename":
                matching_filename
        })


print(
    "Series connected to masks:",
    len(series_lookup)
)


# ============================================================
# EXTRACT 3D GEOMETRY
# ============================================================

header(
    "EXTRACTING 3D LESION GEOMETRY"
)

print(
    "Only segmentation masks are processed."
)

print(
    "CT volumes are NOT loaded."
)

print(
    "Using scipy.ndimage for efficient geometry extraction."
)

print()


lesion_records = []

failed = 0

total = len(
    series_lookup
)


for index, info in enumerate(
    series_lookup,
    start=1
):

    filename = info[
        "filename"
    ]

    mask_path = mask_files[
        filename
    ]

    print(
        "["
        + str(index)
        + "/"
        + str(total)
        + "] Patient "
        + str(
            info["patient_id"]
        )
        + " | "
        + info[
            "study_date_string"
        ]
        + " | processing..."
    )

    try:

        lesions = (
            extract_label_geometry(
                mask_path
            )
        )

        print(
            "    Lesions:",
            len(lesions)
        )

        for lesion in lesions:

            lesion_records.append({

                "patient_id":
                    info[
                        "patient_id"
                    ],

                "study_date":
                    info[
                        "study_date"
                    ],

                "study_date_string":
                    info[
                        "study_date_string"
                    ],

                "region":
                    info[
                        "region"
                    ],

                "uuid":
                    info[
                        "uuid"
                    ],

                "filename":
                    filename,

                "mask_label":
                    lesion[
                        "mask_label"
                    ],

                "voxel_count":
                    lesion[
                        "voxel_count"
                    ],

                "volume_cm3":
                    lesion[
                        "volume_cm3"
                    ],

                "centroid_x":
                    lesion[
                        "centroid_x"
                    ],

                "centroid_y":
                    lesion[
                        "centroid_y"
                    ],

                "centroid_z":
                    lesion[
                        "centroid_z"
                    ],

                "min_x":
                    lesion[
                        "min_x"
                    ],

                "min_y":
                    lesion[
                        "min_y"
                    ],

                "min_z":
                    lesion[
                        "min_z"
                    ],

                "max_x":
                    lesion[
                        "max_x"
                    ],

                "max_y":
                    lesion[
                        "max_y"
                    ],

                "max_z":
                    lesion[
                        "max_z"
                    ]
            })

    except Exception as error:

        failed += 1

        print(
            "    ERROR:",
            str(error)
        )


lesion_df = pd.DataFrame(
    lesion_records
)


# ============================================================
# GEOMETRY SUMMARY
# ============================================================

header(
    "3D LESION GEOMETRY SUMMARY"
)

print(
    "Series processed:",
    total - failed
)

print(
    "Series failed:",
    failed
)

print(
    "Individual segmented lesions:",
    len(lesion_df)
)


if len(lesion_df) > 0:

    print()

    print(
        "Minimum lesion volume:",
        round(
            lesion_df[
                "volume_cm3"
            ].min(),
            3
        ),
        "cm3"
    )

    print(
        "Maximum lesion volume:",
        round(
            lesion_df[
                "volume_cm3"
            ].max(),
            3
        ),
        "cm3"
    )

    print(
        "Mean lesion volume:",
        round(
            lesion_df[
                "volume_cm3"
            ].mean(),
            3
        ),
        "cm3"
    )

    print(
        "Median lesion volume:",
        round(
            lesion_df[
                "volume_cm3"
            ].median(),
            3
        ),
        "cm3"
    )


# ============================================================
# LONGITUDINAL PATIENTS
# ============================================================

header(
    "IDENTIFYING LONGITUDINAL PATIENTS"
)

patient_date_counts = (
    lesion_df
    .groupby(
        "patient_id"
    )[
        "study_date"
    ]
    .nunique()
)


longitudinal_patients = (
    patient_date_counts[
        patient_date_counts >= 2
    ]
)


print(
    "Patients with >=2 lesion-mask dates:",
    len(longitudinal_patients)
)


# ============================================================
# CENTROID CORRESPONDENCE
# ============================================================

header(
    "CALCULATING LONGITUDINAL CORRESPONDENCE"
)

correspondence_results = []


for patient_id in (
    longitudinal_patients.index
):

    patient_data = lesion_df[
        lesion_df[
            "patient_id"
        ] == patient_id
    ].copy()

    dates = sorted(
        patient_data[
            "study_date"
        ]
        .dropna()
        .unique()
    )

    if len(dates) < 2:

        continue


    print()

    print(
        "Patient:",
        patient_id
    )


    for i in range(
        1,
        len(dates)
    ):

        previous_date = dates[
            i - 1
        ]

        current_date = dates[
            i
        ]

        previous = patient_data[
            patient_data[
                "study_date"
            ] == previous_date
        ]

        current = patient_data[
            patient_data[
                "study_date"
            ] == current_date
        ]


        interval_days = (
            current_date -
            previous_date
        ).days


        print(
            "  ",
            previous_date.strftime(
                "%Y-%m-%d"
            ),
            "->",
            current_date.strftime(
                "%Y-%m-%d"
            ),
            ":",
            interval_days,
            "days"
        )

        print(
            "     Previous lesions:",
            len(previous)
        )

        print(
            "     Current lesions:",
            len(current)
        )


        # ----------------------------------------------------
        # MATCH EACH PREVIOUS LESION WITH CLOSEST CURRENT
        # ----------------------------------------------------

        for _, previous_lesion in (
            previous.iterrows()
        ):

            best_distance = np.inf

            best_current = None


            for _, current_lesion in (
                current.iterrows()
            ):

                distance = (
                    centroid_distance(
                        previous_lesion,
                        current_lesion
                    )
                )

                if distance < best_distance:

                    best_distance = (
                        distance
                    )

                    best_current = (
                        current_lesion
                    )


            if best_current is None:

                continue


            previous_volume = float(
                previous_lesion[
                    "volume_cm3"
                ]
            )

            current_volume = float(
                best_current[
                    "volume_cm3"
                ]
            )


            if previous_volume > 0:

                volume_change = (
                    (
                        current_volume -
                        previous_volume
                    )
                    /
                    previous_volume
                ) * 100.0

            else:

                volume_change = np.nan


            correspondence_results.append({

                "patient_id":
                    patient_id,

                "previous_date":
                    previous_date.strftime(
                        "%Y%m%d"
                    ),

                "current_date":
                    current_date.strftime(
                        "%Y%m%d"
                    ),

                "interval_days":
                    interval_days,

                "previous_label":
                    previous_lesion[
                        "mask_label"
                    ],

                "current_label":
                    best_current[
                        "mask_label"
                    ],

                "centroid_distance_mm":
                    best_distance,

                "previous_volume_cm3":
                    previous_volume,

                "current_volume_cm3":
                    current_volume,

                "volume_change_percent":
                    volume_change
            })


correspondence_df = pd.DataFrame(
    correspondence_results
)


# ============================================================
# CORRESPONDENCE SUMMARY
# ============================================================

header(
    "CORRESPONDENCE RESULTS"
)

print(
    "Candidate correspondences:",
    len(correspondence_df)
)


if len(correspondence_df) > 0:

    distances = (
        correspondence_df[
            "centroid_distance_mm"
        ]
    )

    print()

    print(
        "Minimum centroid distance:",
        round(
            distances.min(),
            2
        ),
        "mm"
    )

    print(
        "Median centroid distance:",
        round(
            distances.median(),
            2
        ),
        "mm"
    )

    print(
        "Mean centroid distance:",
        round(
            distances.mean(),
            2
        ),
        "mm"
    )

    print(
        "Maximum centroid distance:",
        round(
            distances.max(),
            2
        ),
        "mm"
    )


# ============================================================
# DISTANCE THRESHOLDS
# ============================================================

header(
    "SPATIAL CORRESPONDENCE THRESHOLDS"
)

if len(correspondence_df) > 0:

    total_candidates = len(
        correspondence_df
    )

    for threshold in [
        5,
        10,
        20,
        30,
        50
    ]:

        count = int(
            (
                correspondence_df[
                    "centroid_distance_mm"
                ]
                <= threshold
            ).sum()
        )

        percentage = (
            count /
            total_candidates
        ) * 100.0

        print(
            "Within",
            threshold,
            "mm:",
            count,
            "/",
            total_candidates,
            "(",
            round(
                percentage,
                2
            ),
            "%)"
        )


# ============================================================
# VOLUME CHANGE
# ============================================================

header(
    "LONGITUDINAL VOLUME CHANGE"
)

if len(correspondence_df) > 0:

    changes = (
        correspondence_df[
            "volume_change_percent"
        ]
        .dropna()
    )

    if len(changes) > 0:

        print(
            "Mean volume change:",
            round(
                changes.mean(),
                2
            ),
            "%"
        )

        print(
            "Median volume change:",
            round(
                changes.median(),
                2
            ),
            "%"
        )

        print(
            "Minimum volume change:",
            round(
                changes.min(),
                2
            ),
            "%"
        )

        print(
            "Maximum volume change:",
            round(
                changes.max(),
                2
            ),
            "%"
        )


# ============================================================
# RECIST + MASK GEOMETRY
# ============================================================

header(
    "LINKING RECIST TO MASK GEOMETRY"
)

recist_temp = recist_df.copy()

recist_temp[
    "study_date_string"
] = (
    recist_temp[
        "study_date"
    ]
    .astype(str)
)


if len(lesion_df) > 0:

    recist_geometry = (
        recist_temp.merge(
            lesion_df,
            left_on=[
                "patient_id",
                "study_date_string",
                "filename",
                "lesion_label_value"
            ],
            right_on=[
                "patient_id",
                "study_date_string",
                "filename",
                "mask_label"
            ],
            how="left"
        )
    )

else:

    recist_geometry = pd.DataFrame()


print(
    "RECIST records:",
    len(recist_temp)
)


if len(recist_geometry) > 0:

    linked_count = int(
        recist_geometry[
            "volume_cm3"
        ]
        .notna()
        .sum()
    )

else:

    linked_count = 0


print(
    "RECIST records linked to geometry:",
    linked_count
)


# ============================================================
# SAVE RESULTS
# ============================================================

header(
    "SAVING STEP 7 RESULTS"
)

geometry_file = os.path.join(
    DATA_DIR,
    "7_lesion_geometry.csv"
)

correspondence_file = os.path.join(
    DATA_DIR,
    "7_lesion_correspondence.csv"
)

recist_geometry_file = os.path.join(
    DATA_DIR,
    "7_recist_geometry.csv"
)


lesion_df.to_csv(
    geometry_file,
    index=False
)

correspondence_df.to_csv(
    correspondence_file,
    index=False
)

recist_geometry.to_csv(
    recist_geometry_file,
    index=False
)


print(
    "Saved:",
    geometry_file
)

print(
    "Saved:",
    correspondence_file
)

print(
    "Saved:",
    recist_geometry_file
)


# ============================================================
# FINAL ASSESSMENT
# ============================================================

header(
    "STEP 7 - FINAL ASSESSMENT"
)

has_geometry = (
    len(lesion_df) > 0
)

has_correspondence = (
    len(correspondence_df) > 0
)


if has_correspondence:

    within_20 = int(
        (
            correspondence_df[
                "centroid_distance_mm"
            ]
            <= 20
        ).sum()
    )

    total = len(
        correspondence_df
    )

    fraction = (
        within_20 /
        total
    ) * 100.0

else:

    within_20 = 0

    total = 0

    fraction = 0.0


print(
    "["
    +
    (
        "YES"
        if has_geometry
        else "NO"
    )
    +
    "] 3D lesion geometry extracted"
)


print(
    "["
    +
    (
        "YES"
        if has_correspondence
        else "NO"
    )
    +
    "] Longitudinal lesion candidates found"
)


print(
    "Candidate correspondences:",
    total
)

print(
    "Candidates within 20 mm:",
    within_20
)

print(
    "Percentage within 20 mm:",
    round(
        fraction,
        2
    ),
    "%"
)


# ============================================================
# IMPORTANT
# ============================================================

print()

print(
    "=" * 90
)

print(
    "IMPORTANT"
)

print(
    "=" * 90
)

print("""
The centroid matching is ONLY a candidate correspondence
method.

A small centroid distance does NOT prove that two regions
are the same biological lesion.

We will use these results to decide whether stronger
registration / spatial-overlap matching is required.

DO NOT TRAIN THE HBGD MODEL YET.
""")


# ============================================================
# COMPLETE
# ============================================================

header(
    "STEP 7 COMPLETE"
)

print("""
Run:

    python 7_analyze_lesion_correspondence.py

The program should continuously show:

    [1/58]
    [2/58]
    [3/58]

Send me the COMPLETE terminal output after it finishes.
""")

print("=" * 90)
print("END OF STEP 7")
print("=" * 90)