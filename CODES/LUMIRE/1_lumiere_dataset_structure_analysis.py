

"""
1_lumiere_dataset_structure_analysis.py

Purpose:
    Analyze the internal structure of the LUMIERE dataset ZIP
    without extracting or reading NIfTI voxel data.

Python:
    Compatible with Python 3.6+

Dataset:
    D:\\VIT\\PROJECT\\NLP\\PATENT\\DATA\\LUMIRE.zip
"""

from __future__ import print_function

import os
import re
import csv
import zipfile
import statistics
from collections import defaultdict, Counter


# ============================================================
# CONFIGURATION
# ============================================================

ZIP_PATH = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE.zip"

OUTPUT_DIR = os.path.dirname(ZIP_PATH)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "LUMIERE_DATASET_STRUCTURE_ANALYSIS.md"
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "lumiere_patient_timepoint_summary.csv"
)

PATIENT_LIST_PATH = os.path.join(
    OUTPUT_DIR,
    "complete_longitudinal_patients.txt"
)


# ============================================================
# EXPECTED FILES
# ============================================================

MODALITIES = [
    "FLAIR.nii.gz",
    "CT1.nii.gz",
    "T1.nii.gz",
    "T2.nii.gz"
]

HDGLIO_SEGMENTATIONS = [
    "segmentation_T1_origspace.nii.gz",
    "segmentation_CT1_origspace.nii.gz",
    "segmentation_FLAIR_origspace.nii.gz",
    "segmentation_T2_origspace.nii.gz",
    "segmentation.nii.gz"
]

DEEPBRATUMIA_SEGMENTATIONS = [
    "t2_seg_mask.nii.gz",
    "ct1_seg_mask.nii.gz",
    "flair_seg_mask.nii.gz",
    "t1_seg_mask.nii.gz"
]

ALL_SEGMENTATIONS = (
    HDGLIO_SEGMENTATIONS +
    DEEPBRATUMIA_SEGMENTATIONS
)


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

def safe_percentage(part, total):
    if total == 0:
        return 0.0
    return (float(part) / float(total)) * 100.0


def median_or_zero(values):
    if not values:
        return 0
    return statistics.median(values)


def mean_or_zero(values):
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def get_extension(filename):
    lower = filename.lower()

    if lower.endswith(".nii.gz"):
        return ".nii.gz"

    _, ext = os.path.splitext(lower)

    if ext:
        return ext

    return "[no extension]"


def is_file_entry(info):
    return not info.filename.endswith("/")


def normalize_path(path):
    return path.replace("\\", "/")


def parse_patient_week(path):
    """
    Extract Patient-XXX and week-XXX from ZIP path.
    """

    path = normalize_path(path)

    patient_match = PATIENT_RE.search(path)
    week_match = WEEK_RE.search(path)

    if patient_match is None or week_match is None:
        return None, None

    patient_id = "Patient-" + patient_match.group(1)
    week_id = week_match.group(1).zfill(3)

    return patient_id, week_id


def write_line(lines, text=""):
    lines.append(text)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LUMIERE DATASET STRUCTURE ANALYSIS")
    print("=" * 70)

    if not os.path.exists(ZIP_PATH):
        print("\nERROR:")
        print("ZIP file not found:")
        print(ZIP_PATH)
        return

    file_size_gb = os.path.getsize(ZIP_PATH) / (
        1024.0 * 1024.0 * 1024.0
    )

    print("\nZIP:")
    print(ZIP_PATH)

    print("\nZIP SIZE:")
    print("{:.2f} GB".format(file_size_gb))

    print("\nOpening ZIP metadata...")

    # IMPORTANT:
    # Convert WindowsPath/string safely for Python 3.6 compatibility.
    with zipfile.ZipFile(str(ZIP_PATH), "r") as archive:

        infos = archive.infolist()

        print("ZIP metadata loaded.")
        print("Total entries:", len(infos))

        # ====================================================
        # BASIC COUNTERS
        # ====================================================

        total_entries = len(infos)

        total_files = 0
        total_directories = 0

        extension_counter = Counter()

        top_level_counter = Counter()

        patient_data = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "files": set(),
                    "modalities": set(),
                    "hdglio": set(),
                    "deepbratumia": set(),
                    "other_segmentations": set(),
                    "mat": set(),
                    "tfm": set(),
                    "json": set()
                }
            )
        )

        unexpected_paths = []

        # ====================================================
        # SCAN ZIP METADATA
        # ====================================================

        for index, info in enumerate(infos):

            path = normalize_path(info.filename)

            # ------------------------------
            # Directory / file
            # ------------------------------

            if path.endswith("/"):
                total_directories += 1
                continue

            total_files += 1

            # ------------------------------
            # Extension
            # ------------------------------

            extension = get_extension(path)
            extension_counter[extension] += 1

            # ------------------------------
            # Top-level directory
            # ------------------------------

            parts = path.split("/")

            if len(parts) > 0:
                top_level_counter[parts[0]] += 1

            # ------------------------------
            # Patient / week
            # ------------------------------

            patient_id, week_id = parse_patient_week(path)

            if patient_id is None or week_id is None:
                unexpected_paths.append(path)
                continue

            filename = parts[-1]

            data = patient_data[patient_id][week_id]

            data["files"].add(filename)

            lower_filename = filename.lower()

            # ------------------------------
            # Modalities
            # ------------------------------

            for modality in MODALITIES:

                if lower_filename == modality.lower():
                    data["modalities"].add(modality)

            # ------------------------------
            # HD-GLIO-AUTO
            # ------------------------------

            for seg in HDGLIO_SEGMENTATIONS:

                if lower_filename == seg.lower():
                    data["hdglio"].add(seg)

            # ------------------------------
            # DeepBraTumIA
            # ------------------------------

            for seg in DEEPBRATUMIA_SEGMENTATIONS:

                if lower_filename == seg.lower():
                    data["deepbratumia"].add(seg)

            # ------------------------------
            # Other segmentation
            # ------------------------------

            if (
                "segmentation" in lower_filename
                and lower_filename not in [
                    x.lower()
                    for x in ALL_SEGMENTATIONS
                ]
            ):
                data["other_segmentations"].add(filename)

            # ------------------------------
            # Metadata files
            # ------------------------------

            if lower_filename.endswith(".mat"):
                data["mat"].add(filename)

            elif lower_filename.endswith(".tfm"):
                data["tfm"].add(filename)

            elif lower_filename.endswith(".json"):
                data["json"].add(filename)

            # Progress indicator
            if (index + 1) % 5000 == 0:
                print(
                    "Processed {:,} / {:,} entries".format(
                        index + 1,
                        total_entries
                    )
                )

    # ========================================================
    # PATIENT STATISTICS
    # ========================================================

    patient_ids = sorted(patient_data.keys())

    total_patients = len(patient_ids)

    all_observed_weeks = []

    patient_week_counts = {}

    for patient_id in patient_ids:

        weeks = sorted(
            patient_data[patient_id].keys(),
            key=lambda x: int(x)
        )

        patient_week_counts[patient_id] = len(weeks)

        all_observed_weeks.extend(weeks)

    # ========================================================
    # TIMEPOINT DISTRIBUTION
    # ========================================================

    exactly_1 = []
    exactly_2 = []
    exactly_3 = []
    exactly_4 = []
    five_plus = []

    at_least_2 = []
    at_least_3 = []
    at_least_4 = []
    at_least_5 = []

    for patient_id in patient_ids:

        count = patient_week_counts[patient_id]

        if count == 1:
            exactly_1.append(patient_id)

        elif count == 2:
            exactly_2.append(patient_id)

        elif count == 3:
            exactly_3.append(patient_id)

        elif count == 4:
            exactly_4.append(patient_id)

        elif count >= 5:
            five_plus.append(patient_id)

        if count >= 2:
            at_least_2.append(patient_id)

        if count >= 3:
            at_least_3.append(patient_id)

        if count >= 4:
            at_least_4.append(patient_id)

        if count >= 5:
            at_least_5.append(patient_id)

    # ========================================================
    # MODALITY / SEGMENTATION STATISTICS
    # ========================================================

    total_patient_timepoints = 0

    modality_timepoints = Counter()
    modality_patients = defaultdict(set)
    modality_complete_patients = defaultdict(set)

    hdglio_timepoints = 0
    deepbratumia_timepoints = 0
    any_segmentation_timepoints = 0

    mat_timepoints = 0
    tfm_timepoints = 0
    json_timepoints = 0

    complete_multimodal_timepoints = 0
    at_least_two_modality_timepoints = 0
    at_least_three_modality_timepoints = 0

    structurally_usable_timepoints = 0
    images_without_segmentation = []
    segmentation_without_images = []

    patient_summary_rows = []

    # ========================================================
    # ANALYZE EVERY PATIENT
    # ========================================================

    for patient_id in patient_ids:

        weeks = sorted(
            patient_data[patient_id].keys(),
            key=lambda x: int(x)
        )

        patient_usable_weeks = []

        patient_complete_weeks = []

        patient_hdglio_weeks = 0
        patient_deepbratumia_weeks = 0

        patient_modality_counts = {
            modality: 0
            for modality in MODALITIES
        }

        patient_intervals = []

        # ----------------------------------------------------
        # Weeks
        # ----------------------------------------------------

        for week_id in weeks:

            data = patient_data[patient_id][week_id]

            total_patient_timepoints += 1

            modalities = data["modalities"]

            has_image = len(modalities) > 0

            has_hdglio = len(data["hdglio"]) > 0
            has_deep = len(data["deepbratumia"]) > 0
            has_other_seg = len(data["other_segmentations"]) > 0

            has_segmentation = (
                has_hdglio or
                has_deep or
                has_other_seg
            )

            # ------------------------------------------------
            # Modality statistics
            # ------------------------------------------------

            for modality in MODALITIES:

                if modality in modalities:

                    modality_timepoints[modality] += 1
                    modality_patients[modality].add(patient_id)
                    patient_modality_counts[modality] += 1

            modality_count = len(modalities)

            if modality_count >= 2:
                at_least_two_modality_timepoints += 1

            if modality_count >= 3:
                at_least_three_modality_timepoints += 1

            if modality_count == 4:
                complete_multimodal_timepoints += 1
                patient_complete_weeks.append(week_id)

            # ------------------------------------------------
            # Segmentation
            # ------------------------------------------------

            if has_hdglio:

                hdglio_timepoints += 1
                patient_hdglio_weeks += 1

            if has_deep:

                deepbratumia_timepoints += 1
                patient_deepbratumia_weeks += 1

            if has_segmentation:

                any_segmentation_timepoints += 1

            # ------------------------------------------------
            # MAT / TFM / JSON
            # ------------------------------------------------

            if len(data["mat"]) > 0:
                mat_timepoints += 1

            if len(data["tfm"]) > 0:
                tfm_timepoints += 1

            if len(data["json"]) > 0:
                json_timepoints += 1

            # ------------------------------------------------
            # Structural usability
            # ------------------------------------------------

            if has_image and has_segmentation:

                structurally_usable_timepoints += 1
                patient_usable_weeks.append(week_id)

            elif has_image and not has_segmentation:

                images_without_segmentation.append(
                    "{} / week-{}".format(
                        patient_id,
                        week_id
                    )
                )

            elif has_segmentation and not has_image:

                segmentation_without_images.append(
                    "{} / week-{}".format(
                        patient_id,
                        week_id
                    )
                )

        # ----------------------------------------------------
        # Consecutive intervals
        # ----------------------------------------------------

        for i in range(len(patient_usable_weeks) - 1):

            week_a = int(patient_usable_weeks[i])
            week_b = int(patient_usable_weeks[i + 1])

            interval = week_b - week_a

            if interval > 0:
                patient_intervals.append(interval)

        # ----------------------------------------------------
        # Patient modality completeness
        # ----------------------------------------------------

        for modality in MODALITIES:

            if len(weeks) > 0:

                if patient_modality_counts[modality] == len(weeks):
                    modality_complete_patients[modality].add(
                        patient_id
                    )

        # ----------------------------------------------------
        # CSV row
        # ----------------------------------------------------

        row = {
            "patient_id": patient_id,
            "total_weeks": len(weeks),
            "min_week": weeks[0] if weeks else "",
            "max_week": weeks[-1] if weeks else "",
            "structurally_usable_weeks": len(
                patient_usable_weeks
            ),
            "weeks_with_FLAIR": patient_modality_counts[
                "FLAIR.nii.gz"
            ],
            "weeks_with_CT1": patient_modality_counts[
                "CT1.nii.gz"
            ],
            "weeks_with_T1": patient_modality_counts[
                "T1.nii.gz"
            ],
            "weeks_with_T2": patient_modality_counts[
                "T2.nii.gz"
            ],
            "weeks_with_HDGLIO_segmentation":
                patient_hdglio_weeks,
            "weeks_with_DeepBraTumIA_segmentation":
                patient_deepbratumia_weeks,
            "complete_multimodal_weeks":
                len(patient_complete_weeks),
            "usable_longitudinal_intervals":
                len(patient_intervals)
        }

        patient_summary_rows.append(row)

    # ========================================================
    # LONGITUDINAL INTERVAL STATISTICS
    # ========================================================

    all_intervals = []

    for patient_id in patient_ids:

        usable_weeks = []

        for week_id in patient_data[patient_id].keys():

            data = patient_data[patient_id][week_id]

            has_image = len(data["modalities"]) > 0

            has_seg = (
                len(data["hdglio"]) > 0 or
                len(data["deepbratumia"]) > 0 or
                len(data["other_segmentations"]) > 0
            )

            if has_image and has_seg:
                usable_weeks.append(week_id)

        usable_weeks = sorted(
            usable_weeks,
            key=lambda x: int(x)
        )

        for i in range(len(usable_weeks) - 1):

            a = int(usable_weeks[i])
            b = int(usable_weeks[i + 1])

            diff = b - a

            if diff > 0:
                all_intervals.append(diff)

    # ========================================================
    # PATIENT STATISTICS
    # ========================================================

    timepoint_counts = list(
        patient_week_counts.values()
    )

    min_timepoints = (
        min(timepoint_counts)
        if timepoint_counts
        else 0
    )

    max_timepoints = (
        max(timepoint_counts)
        if timepoint_counts
        else 0
    )

    mean_timepoints = mean_or_zero(
        timepoint_counts
    )

    median_timepoints = median_or_zero(
        timepoint_counts
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    write_line(report, "# LUMIERE DATASET STRUCTURE ANALYSIS")
    write_line(report)
    write_line(
        report,
        "Metadata-only structural analysis of the LUMIERE ZIP archive."
    )
    write_line(report)

    write_line(report, "## 1. Dataset Information")
    write_line(report)
    write_line(
        report,
        "| Property | Value |"
    )
    write_line(
        report,
        "|---|---:|"
    )
    write_line(
        report,
        "| ZIP path | `{}` |".format(ZIP_PATH)
    )
    write_line(
        report,
        "| ZIP size | {:.2f} GB |".format(file_size_gb)
    )
    write_line(
        report,
        "| Total ZIP entries | {:,} |".format(
            total_entries
        )
    )
    write_line(
        report,
        "| Total files | {:,} |".format(
            total_files
        )
    )
    write_line(
        report,
        "| Total directories | {:,} |".format(
            total_directories
        )
    )
    write_line(report)

    # --------------------------------------------------------
    # Patient summary
    # --------------------------------------------------------

    write_line(report, "## 2. Patient Summary")
    write_line(report)

    write_line(
        report,
        "| Metric | Value |"
    )
    write_line(
        report,
        "|---|---:|"
    )

    write_line(
        report,
        "| Total patients | {:,} |".format(
            total_patients
        )
    )

    if patient_ids:

        write_line(
            report,
            "| First patient ID | {} |".format(
                patient_ids[0]
            )
        )

        write_line(
            report,
            "| Last patient ID | {} |".format(
                patient_ids[-1]
            )
        )

    write_line(
        report,
        "| Minimum timepoints/patient | {} |".format(
            min_timepoints
        )
    )

    write_line(
        report,
        "| Maximum timepoints/patient | {} |".format(
            max_timepoints
        )
    )

    write_line(
        report,
        "| Mean timepoints/patient | {:.2f} |".format(
            mean_timepoints
        )
    )

    write_line(
        report,
        "| Median timepoints/patient | {:.1f} |".format(
            median_timepoints
        )
    )

    write_line(report)

    # --------------------------------------------------------
    # Timepoint distribution
    # --------------------------------------------------------

    write_line(
        report,
        "## 3. Timepoint Distribution"
    )
    write_line(report)

    write_line(
        report,
        "| Category | Patients | Percentage |"
    )
    write_line(
        report,
        "|---|---:|---:|"
    )

    distribution = [
        ("Exactly 1", exactly_1),
        ("Exactly 2", exactly_2),
        ("Exactly 3", exactly_3),
        ("Exactly 4", exactly_4),
        ("5 or more", five_plus),
        ("At least 2", at_least_2),
        ("At least 3", at_least_3),
        ("At least 4", at_least_4),
        ("At least 5", at_least_5)
    ]

    for label, patients in distribution:

        write_line(
            report,
            "| {} | {:,} | {:.2f}% |".format(
                label,
                len(patients),
                safe_percentage(
                    len(patients),
                    total_patients
                )
            )
        )

    write_line(report)

    # --------------------------------------------------------
    # Modalities
    # --------------------------------------------------------

    write_line(
        report,
        "## 4. Imaging Modality Availability"
    )
    write_line(report)

    write_line(
        report,
        "| Modality | Patient-timepoints | % of timepoints | Patients | Patients with modality at every week |"
    )
    write_line(
        report,
        "|---|---:|---:|---:|---:|"
    )

    for modality in MODALITIES:

        write_line(
            report,
            "| {} | {:,} | {:.2f}% | {:,} | {:,} |".format(
                modality,
                modality_timepoints[modality],
                safe_percentage(
                    modality_timepoints[modality],
                    total_patient_timepoints
                ),
                len(modality_patients[modality]),
                len(modality_complete_patients[modality])
            )
        )

    write_line(report)

    write_line(
        report,
        "**Interpretation:** modality availability is based only on file existence in the ZIP."
    )
    write_line(report)

    # --------------------------------------------------------
    # Segmentation
    # --------------------------------------------------------

    write_line(
        report,
        "## 5. Segmentation Availability"
    )
    write_line(report)

    write_line(
        report,
        "| Segmentation category | Patient-timepoints | Percentage |"
    )
    write_line(
        report,
        "|---|---:|---:|"
    )

    write_line(
        report,
        "| HD-GLIO-AUTO | {:,} | {:.2f}% |".format(
            hdglio_timepoints,
            safe_percentage(
                hdglio_timepoints,
                total_patient_timepoints
            )
        )
    )

    write_line(
        report,
        "| DeepBraTumIA | {:,} | {:.2f}% |".format(
            deepbratumia_timepoints,
            safe_percentage(
                deepbratumia_timepoints,
                total_patient_timepoints
            )
        )
    )

    write_line(
        report,
        "| Any segmentation | {:,} | {:.2f}% |".format(
            any_segmentation_timepoints,
            safe_percentage(
                any_segmentation_timepoints,
                total_patient_timepoints
            )
        )
    )

    write_line(report)

    write_line(
        report,
        "### Individual HD-GLIO-AUTO files"
    )
    write_line(report)

    for seg in HDGLIO_SEGMENTATIONS:

        count = 0

        for patient_id in patient_ids:

            for week_id in patient_data[patient_id]:

                if seg in patient_data[patient_id][week_id]["hdglio"]:
                    count += 1

        write_line(
            report,
            "- `{}`: {:,} patient-timepoints".format(
                seg,
                count
            )
        )

    write_line(report)

    write_line(
        report,
        "### Individual DeepBraTumIA files"
    )
    write_line(report)

    for seg in DEEPBRATUMIA_SEGMENTATIONS:

        count = 0

        for patient_id in patient_ids:

            for week_id in patient_data[patient_id]:

                if seg in patient_data[patient_id][week_id]["deepbratumia"]:
                    count += 1

        write_line(
            report,
            "- `{}`: {:,} patient-timepoints".format(
                seg,
                count
            )
        )

    write_line(report)

    # --------------------------------------------------------
    # Structural usability
    # --------------------------------------------------------

    write_line(
        report,
        "## 6. Structural Timepoint Usability"
    )
    write_line(report)

    write_line(
        report,
        "| Category | Count | Percentage |"
    )
    write_line(
        report,
        "|---|---:|---:|"
    )

    write_line(
        report,
        "| Total patient-timepoints | {:,} | 100% |".format(
            total_patient_timepoints
        )
    )

    write_line(
        report,
        "| Structurally usable timepoints | {:,} | {:.2f}% |".format(
            structurally_usable_timepoints,
            safe_percentage(
                structurally_usable_timepoints,
                total_patient_timepoints
            )
        )
    )

    write_line(
        report,
        "| Images but no segmentation | {:,} | {:.2f}% |".format(
            len(images_without_segmentation),
            safe_percentage(
                len(images_without_segmentation),
                total_patient_timepoints
            )
        )
    )

    write_line(
        report,
        "| Segmentation but no images | {:,} | {:.2f}% |".format(
            len(segmentation_without_images),
            safe_percentage(
                len(segmentation_without_images),
                total_patient_timepoints
            )
        )
    )

    write_line(report)

    write_line(
        report,
        "**Important:** A structurally usable timepoint means that at least one image file and at least one segmentation file are present. NIfTI voxel content, tumor volume, mask quality, shape, affine, spacing, and biological validity have NOT been checked."
    )

    write_line(report)

    # --------------------------------------------------------
    # Complete multimodal
    # --------------------------------------------------------

    write_line(
        report,
        "## 7. Multimodal Completeness"
    )
    write_line(report)

    write_line(
        report,
        "| Category | Patient-timepoints | Percentage |"
    )
    write_line(
        report,
        "|---|---:|---:|"
    )

    write_line(
        report,
        "| At least 1 modality | {:,} | {:.2f}% |".format(
            sum(
                1 for patient_id in patient_ids
                for week_id in patient_data[patient_id]
                if len(
                    patient_data[patient_id][week_id]["modalities"]
                ) >= 1
            ),
            safe_percentage(
                sum(
                    1 for patient_id in patient_ids
                    for week_id in patient_data[patient_id]
                    if len(
                        patient_data[patient_id][week_id]["modalities"]
                    ) >= 1
                ),
                total_patient_timepoints
            )
        )
    )

    write_line(
        report,
        "| At least 2 modalities | {:,} | {:.2f}% |".format(
            at_least_two_modality_timepoints,
            safe_percentage(
                at_least_two_modality_timepoints,
                total_patient_timepoints
            )
        )
    )

    write_line(
        report,
        "| At least 3 modalities | {:,} | {:.2f}% |".format(
            at_least_three_modality_timepoints,
            safe_percentage(
                at_least_three_modality_timepoints,
                total_patient_timepoints
            )
        )
    )

    write_line(
        report,
        "| All 4 modalities | {:,} | {:.2f}% |".format(
            complete_multimodal_timepoints,
            safe_percentage(
                complete_multimodal_timepoints,
                total_patient_timepoints
            )
        )
    )

    write_line(report)

    # --------------------------------------------------------
    # Longitudinal trajectories
    # --------------------------------------------------------

    write_line(
        report,
        "## 8. Candidate Longitudinal Trajectories"
    )
    write_line(report)

    write_line(
        report,
        "| Criterion | Patients | Percentage |"
    )
    write_line(
        report,
        "|---|---:|---:|"
    )

    usable_patient_counts = {}

    for threshold in [2, 3, 4, 5]:

        patients = []

        for patient_id in patient_ids:

            usable_count = 0

            for week_id in patient_data[patient_id]:

                data = patient_data[patient_id][week_id]

                has_image = len(data["modalities"]) > 0

                has_seg = (
                    len(data["hdglio"]) > 0 or
                    len(data["deepbratumia"]) > 0 or
                    len(data["other_segmentations"]) > 0
                )

                if has_image and has_seg:
                    usable_count += 1

            if usable_count >= threshold:
                patients.append(patient_id)

        usable_patient_counts[threshold] = patients

        write_line(
            report,
            "| At least {} structurally usable timepoints | {:,} | {:.2f}% |".format(
                threshold,
                len(patients),
                safe_percentage(
                    len(patients),
                    total_patients
                )
            )
        )

    write_line(report)

    # --------------------------------------------------------
    # Interval analysis
    # --------------------------------------------------------

    write_line(
        report,
        "## 9. Longitudinal Interval Analysis"
    )
    write_line(report)

    write_line(
        report,
        "| Metric | Value |"
    )
    write_line(
        report,
        "|---|---:|"
    )

    write_line(
        report,
        "| Total candidate longitudinal intervals | {:,} |".format(
            len(all_intervals)
        )
    )

    if all_intervals:

        write_line(
            report,
            "| Minimum interval | {} weeks |".format(
                min(all_intervals)
            )
        )

        write_line(
            report,
            "| Maximum interval | {} weeks |".format(
                max(all_intervals)
            )
        )

        write_line(
            report,
            "| Mean interval | {:.2f} weeks |".format(
                mean_or_zero(all_intervals)
            )
        )

        write_line(
            report,
            "| Median interval | {:.1f} weeks |".format(
                median_or_zero(all_intervals)
            )
        )

    else:

        write_line(
            report,
            "| Interval statistics | No intervals found |"
        )

    write_line(report)

    # --------------------------------------------------------
    # MAT / TFM / JSON
    # --------------------------------------------------------

    write_line(
        report,
        "## 10. Metadata File Availability"
    )
    write_line(report)

    write_line(
        report,
        "| File type | Patient-timepoints containing file |"
    )
    write_line(
        report,
        "|---|---:|"
    )

    write_line(
        report,
        "| `.mat` | {:,} |".format(
            mat_timepoints
        )
    )

    write_line(
        report,
        "| `.tfm` | {:,} |".format(
            tfm_timepoints
        )
    )

    write_line(
        report,
        "| `.json` | {:,} |".format(
            json_timepoints
        )
    )

    write_line(report)

    # --------------------------------------------------------
    # Extension summary
    # --------------------------------------------------------

    write_line(
        report,
        "## 11. File Extension Summary"
    )
    write_line(report)

    write_line(
        report,
        "| Extension | Count |"
    )
    write_line(
        report,
        "|---|---:|"
    )

    for extension, count in sorted(
        extension_counter.items(),
        key=lambda x: (-x[1], x[0])
    ):

        write_line(
            report,
            "| `{}` | {:,} |".format(
                extension,
                count
            )
        )

    write_line(report)

    # --------------------------------------------------------
    # Directory summary
    # --------------------------------------------------------

    write_line(
        report,
        "## 12. Directory Structure"
    )
    write_line(report)

    write_line(
        report,
        "### Top-level directories"
    )
    write_line(report)

    for directory, count in sorted(
        top_level_counter.items()
    ):

        write_line(
            report,
            "- `{}`: {:,} file entries".format(
                directory,
                count
            )
        )

    write_line(report)

    write_line(
        report,
        "Expected structure:"
    )

    write_line(
        report,
        "```text"
    )

    write_line(
        report,
        "Imaging/"
    )

    write_line(
        report,
        "    Patient-XXX/"
    )

    write_line(
        report,
        "        week-XXX/"
    )

    write_line(
        report,
        "            imaging / segmentation / metadata"
    )

    write_line(
        report,
        "```"
    )

    write_line(report)

    # --------------------------------------------------------
    # Missing data
    # --------------------------------------------------------

    write_line(
        report,
        "## 13. Missing Data Analysis"
    )
    write_line(report)

    write_line(
        report,
        "### Patient-timepoints with images but no segmentation"
    )
    write_line(report)

    if images_without_segmentation:

        for item in images_without_segmentation[:200]:

            write_line(
                report,
                "- {}".format(item)
            )

        if len(images_without_segmentation) > 200:

            write_line(
                report,
                "- ... {} additional entries".format(
                    len(images_without_segmentation) - 200
                )
            )

    else:

        write_line(
            report,
            "None detected."
        )

    write_line(report)

    write_line(
        report,
        "### Patient-timepoints with segmentation but no image"
    )
    write_line(report)

    if segmentation_without_images:

        for item in segmentation_without_images[:200]:

            write_line(
                report,
                "- {}".format(item)
            )

        if len(segmentation_without_images) > 200:

            write_line(
                report,
                "- ... {} additional entries".format(
                    len(segmentation_without_images) - 200
                )
            )

    else:

        write_line(
            report,
            "None detected."
        )

    write_line(report)

    # --------------------------------------------------------
    # Important interpretation
    # --------------------------------------------------------

    write_line(
        report,
        "## 14. Scientific Interpretation"
    )
    write_line(report)

    write_line(
        report,
        "The ZIP inspection confirms the presence of longitudinal patient/week organization and multiple imaging and segmentation resources."
    )

    write_line(report)

    write_line(
        report,
        "However, this analysis is metadata-only. It does not validate image contents, segmentation quality, tumor voxel presence, image orientation, voxel spacing, affine consistency, registration accuracy, or biological longitudinal consistency."
    )

    write_line(report)

    write_line(
        report,
        "Therefore, the patient counts reported as structurally usable or candidate longitudinal trajectories must be treated as preliminary until the corresponding NIfTI files are selectively extracted and validated."
    )

    write_line(report)

    write_line(
        report,
        "For future HBGD modelling, week numbers should not automatically be converted to biological elapsed time unless the LUMIERE dataset documentation confirms that the week identifier represents a reliable temporal scale."
    )

    write_line(report)

    # --------------------------------------------------------
    # Next step
    # --------------------------------------------------------

    write_line(
        report,
        "## 15. Recommended Next Step"
    )
    write_line(report)

    write_line(
        report,
        "The next analysis should selectively extract only the structurally promising longitudinal patient-timepoints and validate the NIfTI images and tumor masks by checking voxel dimensions, affine/orientation, voxel spacing, non-zero mask voxels, tumor volume, and longitudinal consistency."
    )

    write_line(report)

    write_line(
        report,
        "After this validation, the dataset can be used to calculate tumor geometry and HBGD growth parameters for the cross-cancer transfer experiment."
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
    # SAVE CSV
    # ========================================================

    csv_columns = [
        "patient_id",
        "total_weeks",
        "min_week",
        "max_week",
        "structurally_usable_weeks",
        "weeks_with_FLAIR",
        "weeks_with_CT1",
        "weeks_with_T1",
        "weeks_with_T2",
        "weeks_with_HDGLIO_segmentation",
        "weeks_with_DeepBraTumIA_segmentation",
        "complete_multimodal_weeks",
        "usable_longitudinal_intervals"
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_columns
        )

        writer.writeheader()

        for row in patient_summary_rows:
            writer.writerow(row)

    # ========================================================
    # SAVE COMPLETE LONGITUDINAL PATIENT LIST
    # ========================================================

    with open(
        PATIENT_LIST_PATH,
        "w",
        encoding="utf-8"
    ) as patient_file:

        for patient_id in usable_patient_counts[3]:

            patient_file.write(
                patient_id + "\n"
            )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nPatients:", total_patients)

    print(
        "Patient-timepoints:",
        total_patient_timepoints
    )

    print(
        "Structurally usable timepoints:",
        structurally_usable_timepoints
    )

    print("\nLONGITUDINAL PATIENT COUNTS")

    print(
        ">= 2 usable timepoints:",
        len(usable_patient_counts[2])
    )

    print(
        ">= 3 usable timepoints:",
        len(usable_patient_counts[3])
    )

    print(
        ">= 4 usable timepoints:",
        len(usable_patient_counts[4])
    )

    print(
        ">= 5 usable timepoints:",
        len(usable_patient_counts[5])
    )

    print("\nMODALITIES")

    for modality in MODALITIES:

        print(
            "{}: {}".format(
                modality,
                modality_timepoints[modality]
            )
        )

    print("\nSEGMENTATIONS")

    print(
        "HD-GLIO-AUTO:",
        hdglio_timepoints
    )

    print(
        "DeepBraTumIA:",
        deepbratumia_timepoints
    )

    print("\nOUTPUT FILES")

    print(REPORT_PATH)
    print(CSV_PATH)
    print(PATIENT_LIST_PATH)

    print("\nNo extraction performed.")
    print("No NIfTI voxel data was read.")

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()