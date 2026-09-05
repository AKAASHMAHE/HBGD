
"""
0_lumiere_zip_inspection.py

LUMIERE DATASET - ZIP-ONLY INSPECTION

Purpose:
    Inspect ONLY LUMIRE.zip without extracting the dataset.

Expected project structure:

    PROJECT/
    ├── CODES/
    │   └── LUMIRE/
    │       └── 0_lumiere_zip_inspection.py
    │
    └── DATA/
        └── LUMIRE.zip

All inspection reports are saved to:

    DATA/LUMIRE/

IMPORTANT:
    - This script does NOT extract LUMIRE.zip.
    - This script does NOT modify the ZIP.
    - This script does NOT inspect any other ZIP file.
    - Only LUMIRE.zip is inspected.
"""

from pathlib import Path
from collections import Counter
import zipfile
import json
import re


# ============================================================
# PROJECT PATHS
# ============================================================

# Current file:
# PROJECT/CODES/LUMIRE/0_lumiere_zip_inspection.py

SCRIPT_DIR = Path(__file__).resolve().parent

# PROJECT/
PROJECT_DIR = SCRIPT_DIR.parent.parent

# DATA/
DATA_DIR = PROJECT_DIR / "DATA"

# Exact ZIP requested by user
ZIP_PATH = DATA_DIR / "LUMIRE.zip"

# All reports must go here
REPORT_DIR = DATA_DIR / "LUMIRE"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# REPORT FILES
# ============================================================

JSON_REPORT = REPORT_DIR / "0_lumiere_zip_inspection_report.json"
TXT_REPORT = REPORT_DIR / "0_lumiere_zip_inspection_report.txt"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_path(path):
    return path.replace("\\", "/")


def get_extension(filename):
    """
    Detect file extensions including .nii.gz.
    """

    name = filename.lower()

    if name.endswith(".nii.gz"):
        return ".nii.gz"

    if name.endswith(".tar.gz"):
        return ".tar.gz"

    return Path(name).suffix.lower()


def is_medical_image(filename):
    """
    Detect common medical imaging formats.
    """

    name = filename.lower()

    extensions = [
        ".nii",
        ".nii.gz",
        ".mha",
        ".mhd",
        ".nrrd",
        ".dcm",
        ".dicom",
        ".npy",
        ".npz",
    ]

    return any(name.endswith(ext) for ext in extensions)


def is_segmentation_or_mask(filename):
    """
    Detect filenames that may represent segmentations/masks.
    """

    name = filename.lower()

    keywords = [
        "seg",
        "mask",
        "label",
        "annotation",
        "annot",
        "tumor",
        "lesion",
        "groundtruth",
        "ground_truth",
        "rano",
    ]

    return any(keyword in name for keyword in keywords)


def detect_time_information(filename):
    """
    Detect possible timepoint/week information from paths.

    This does NOT assume that the detected number is the actual
    LUMIERE timepoint. It is only a filename/path inspection.
    """

    name = filename.lower()

    patterns = [
        r"timepoint[_\- ]?(\d+)",
        r"time[_\- ]?(\d+)",
        r"tp[_\- ]?(\d+)",
        r"visit[_\- ]?(\d+)",
        r"week[_\- ]?(\d+)",
        r"wk[_\- ]?(\d+)",
        r"followup[_\- ]?(\d+)",
        r"follow[_\- ]?up[_\- ]?(\d+)",
    ]

    for pattern in patterns:

        match = re.search(pattern, name)

        if match:
            return match.group(1)

    return None


def detect_possible_patient(filename):
    """
    Detect possible patient identifiers.

    This is only a heuristic.
    """

    parts = normalize_path(filename).split("/")

    patterns = [
        r"^patient[_\-]?\d+$",
        r"^subject[_\-]?\d+$",
        r"^case[_\-]?\d+$",
        r"^study[_\-]?\d+$",
        r"^sub[_\-]?\d+$",
        r"^patient\d+$",
        r"^case\d+$",
        r"^subject\d+$",
    ]

    for part in parts:

        stem = Path(part).stem.lower()

        for pattern in patterns:

            if re.match(pattern, stem):
                return part

    return None


# ============================================================
# MAIN
# ============================================================

def inspect_lumiere_zip():

    print("\n" + "=" * 100)
    print("LUMIERE ZIP INSPECTION")
    print("=" * 100)

    # --------------------------------------------------------
    # VERIFY ZIP
    # --------------------------------------------------------

    print(f"\nExpected ZIP:")
    print(ZIP_PATH)

    if not ZIP_PATH.exists():

        print("\nERROR:")
        print("LUMIRE.zip was not found.")

        print("\nExpected location:")
        print(ZIP_PATH)

        print("\nPlease place the ZIP exactly here:")
        print("DATA/LUMIRE.zip")

        return

    if not ZIP_PATH.is_file():

        print("\nERROR:")
        print("DATA/LUMIRE.zip exists but is not a normal file.")

        return

    # --------------------------------------------------------
    # ZIP SIZE
    # --------------------------------------------------------

    zip_size_gb = ZIP_PATH.stat().st_size / (1024 ** 3)

    print(f"\nZIP SIZE: {zip_size_gb:.2f} GB")

    # --------------------------------------------------------
    # REPORT OBJECT
    # --------------------------------------------------------

    report = {

        "dataset": "LUMIERE",

        "zip_file": str(ZIP_PATH),

        "zip_size_gb": round(zip_size_gb, 4),

        "extracted": False,

        "inspection_only": True,

        "total_entries": 0,

        "total_files": 0,

        "total_directories": 0,

        "file_extensions": {},

        "top_level_directories": {},

        "medical_image_count": 0,

        "segmentation_mask_count": 0,

        "metadata_count": 0,

        "medical_images_sample": [],

        "segmentation_samples": [],

        "metadata_samples": [],

        "possible_patient_ids": {},

        "possible_timepoints": {},

        "sample_files": [],

        "directory_samples": [],

        "keyword_samples": {},
    }


    # ========================================================
    # OPEN ZIP WITHOUT EXTRACTION
    # ========================================================

    try:

        with zipfile.ZipFile(str(ZIP_PATH), "r") as archive:

# ------------------------------------------------
# ZIP ACCESS CHECK
# ------------------------------------------------

            print("\nZIP opened successfully.")
            print("Skipping full ZIP integrity test because the dataset is 30+ GB.")
            print("Only ZIP metadata and file structure will be inspected.")

            # ------------------------------------------------
            # GET ALL ENTRIES
            # ------------------------------------------------

            entries = archive.infolist()

            report["total_entries"] = len(entries)

            files = []
            directories = []

            for info in entries:

                if info.is_dir():

                    directories.append(
                        normalize_path(info.filename)
                    )

                else:

                    files.append(info)


            report["total_files"] = len(files)

            report["total_directories"] = len(directories)


            # =================================================
            # BASIC INFORMATION
            # =================================================

            print("\n" + "-" * 100)
            print("BASIC ZIP INFORMATION")
            print("-" * 100)

            print(f"Total entries : {len(entries):,}")
            print(f"Files         : {len(files):,}")
            print(f"Directories   : {len(directories):,}")


            # =================================================
            # FILE EXTENSIONS
            # =================================================

            extension_counter = Counter()

            for info in files:

                extension_counter[
                    get_extension(info.filename)
                ] += 1


            report["file_extensions"] = dict(
                extension_counter.most_common()
            )


            print("\n" + "-" * 100)
            print("FILE EXTENSIONS")
            print("-" * 100)

            for extension, count in extension_counter.most_common():

                display_extension = (
                    extension
                    if extension
                    else "[NO EXTENSION]"
                )

                print(
                    f"{display_extension:20s} : {count:,}"
                )


            # =================================================
            # TOP LEVEL DIRECTORIES
            # =================================================

            top_level_counter = Counter()

            for info in entries:

                path = normalize_path(info.filename)

                parts = path.split("/")

                if parts and parts[0]:

                    top_level_counter[parts[0]] += 1


            report["top_level_directories"] = dict(
                top_level_counter.most_common()
            )


            print("\n" + "-" * 100)
            print("TOP-LEVEL ZIP STRUCTURE")
            print("-" * 100)

            for name, count in top_level_counter.most_common():

                print(
                    f"{name:60s} : {count:,} entries"
                )


            # =================================================
            # MEDICAL IMAGES
            # =================================================

            medical_images = []

            for info in files:

                if is_medical_image(info.filename):

                    medical_images.append({
                        "path": normalize_path(info.filename),
                        "size_mb": round(
                            info.file_size / (1024 ** 2),
                            3
                        )
                    })


            report["medical_image_count"] = len(
                medical_images
            )

            report["medical_images_sample"] = (
                medical_images[:1000]
            )


            print("\n" + "-" * 100)
            print("MEDICAL IMAGE FILES")
            print("-" * 100)

            print(
                f"Detected medical image files: "
                f"{len(medical_images):,}"
            )

            for item in medical_images[:100]:

                print(item["path"])

            if len(medical_images) > 100:

                print(
                    f"... and "
                    f"{len(medical_images) - 100:,} more"
                )


            # =================================================
            # SEGMENTATION / MASK FILES
            # =================================================

            segmentation_files = []

            for info in files:

                if is_segmentation_or_mask(
                    info.filename
                ):

                    segmentation_files.append({
                        "path": normalize_path(
                            info.filename
                        ),
                        "size_mb": round(
                            info.file_size / (1024 ** 2),
                            3
                        )
                    })


            report["segmentation_mask_count"] = len(
                segmentation_files
            )

            report["segmentation_samples"] = (
                segmentation_files[:1000]
            )


            print("\n" + "-" * 100)
            print("POSSIBLE SEGMENTATION / MASK FILES")
            print("-" * 100)

            print(
                f"Detected: "
                f"{len(segmentation_files):,}"
            )

            for item in segmentation_files[:200]:

                print(item["path"])

            if len(segmentation_files) > 200:

                print(
                    f"... and "
                    f"{len(segmentation_files) - 200:,} more"
                )


            # =================================================
            # METADATA FILES
            # =================================================

            metadata_extensions = {
                ".csv",
                ".tsv",
                ".json",
                ".xlsx",
                ".xls",
                ".txt",
                ".xml",
                ".yaml",
                ".yml",
            }

            metadata_files = []

            for info in files:

                extension = get_extension(
                    info.filename
                )

                if extension in metadata_extensions:

                    metadata_files.append({
                        "path": normalize_path(
                            info.filename
                        ),
                        "size_kb": round(
                            info.file_size / 1024,
                            3
                        )
                    })


            report["metadata_count"] = len(
                metadata_files
            )

            report["metadata_samples"] = (
                metadata_files[:1000]
            )


            print("\n" + "-" * 100)
            print("METADATA FILES")
            print("-" * 100)

            print(
                f"Detected metadata files: "
                f"{len(metadata_files):,}"
            )

            for item in metadata_files[:200]:

                print(item["path"])


            # =================================================
            # POSSIBLE PATIENT IDS
            # =================================================

            patient_counter = Counter()

            for info in files:

                patient = detect_possible_patient(
                    info.filename
                )

                if patient:

                    patient_counter[patient] += 1


            report["possible_patient_ids"] = dict(
                patient_counter.most_common()
            )


            print("\n" + "-" * 100)
            print("POSSIBLE PATIENT IDENTIFIERS")
            print("-" * 100)

            print(
                f"Possible patient identifiers detected: "
                f"{len(patient_counter):,}"
            )

            for patient, count in patient_counter.most_common(
                200
            ):

                print(
                    f"{patient:40s} : "
                    f"{count:,} files"
                )


            # =================================================
            # POSSIBLE TIMEPOINTS
            # =================================================

            timepoint_counter = Counter()

            for info in files:

                timepoint = detect_time_information(
                    info.filename
                )

                if timepoint is not None:

                    timepoint_counter[
                        timepoint
                    ] += 1


            report["possible_timepoints"] = dict(
                timepoint_counter.most_common()
            )


            print("\n" + "-" * 100)
            print("POSSIBLE TIMEPOINT INFORMATION")
            print("-" * 100)

            if timepoint_counter:

                for timepoint, count in sorted(
                    timepoint_counter.items(),
                    key=lambda x: int(x[0])
                ):

                    print(
                        f"Timepoint {timepoint:10s} : "
                        f"{count:,} files"
                    )

            else:

                print(
                    "No obvious timepoint pattern "
                    "detected in filenames/paths."
                )

                print(
                    "This does NOT mean time information "
                    "is absent."
                )

                print(
                    "It may be stored in metadata."
                )


            # =================================================
            # SAMPLE FILES
            # =================================================

            print("\n" + "-" * 100)
            print("FIRST 500 FILES IN ZIP")
            print("-" * 100)

            for info in files[:500]:

                path = normalize_path(
                    info.filename
                )

                report["sample_files"].append({
                    "path": path,
                    "size_bytes": info.file_size,
                    "extension": get_extension(path)
                })

                print(path)


            # =================================================
            # DIRECTORY SAMPLES
            # =================================================

            unique_directories = set()

            for info in files:

                parts = normalize_path(
                    info.filename
                ).split("/")

                current = ""

                for part in parts[:-1]:

                    if current:

                        current += "/" + part

                    else:

                        current = part

                    unique_directories.add(
                        current
                    )


            report["directory_samples"] = sorted(
                unique_directories
            )[:2000]


            # =================================================
            # KEYWORD INSPECTION
            # =================================================

            keywords = [
                "patient",
                "subject",
                "case",
                "time",
                "week",
                "rano",
                "seg",
                "mask",
                "label",
                "tumor",
                "lesion",
                "t1",
                "t1c",
                "t2",
                "flair",
                "dwi",
                "adc",
                "metadata",
                "clinical",
                "scan",
                "mri",
            ]


            keyword_samples = {}

            for keyword in keywords:

                hits = []

                for info in files:

                    path = normalize_path(
                        info.filename
                    )

                    if keyword in path.lower():

                        hits.append(path)

                        if len(hits) >= 100:

                            break

                keyword_samples[keyword] = hits


            report["keyword_samples"] = keyword_samples


    except zipfile.BadZipFile:

        print("\nERROR:")
        print("LUMIRE.zip is not a valid ZIP archive.")

        return

    except PermissionError:

        print("\nERROR:")
        print("Permission denied while reading LUMIRE.zip.")

        return

    except Exception as error:

        print("\nERROR:")
        print(type(error).__name__)
        print(str(error))

        raise


    # ========================================================
    # SAVE JSON REPORT
    # ========================================================

    with open(
        JSON_REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )


    # ========================================================
    # SAVE HUMAN-READABLE TXT REPORT
    # ========================================================

    with open(
        TXT_REPORT,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "LUMIERE DATASET - ZIP INSPECTION REPORT\n"
        )

        file.write("=" * 100 + "\n\n")

        file.write(
            f"ZIP FILE: {ZIP_PATH}\n"
        )

        file.write(
            f"ZIP SIZE: {zip_size_gb:.2f} GB\n"
        )

        file.write(
            "EXTRACTED: NO\n"
        )

        file.write(
            "INSPECTION TYPE: ZIP CONTENTS ONLY\n\n"
        )


        file.write(
            "BASIC INFORMATION\n"
        )

        file.write("-" * 100 + "\n")

        file.write(
            f"Total entries : "
            f"{report['total_entries']:,}\n"
        )

        file.write(
            f"Total files   : "
            f"{report['total_files']:,}\n"
        )

        file.write(
            f"Directories   : "
            f"{report['total_directories']:,}\n"
        )


        file.write(
            "\nFILE EXTENSIONS\n"
        )

        file.write("-" * 100 + "\n")

        for extension, count in (
            extension_counter.most_common()
        ):

            file.write(
                f"{extension or '[NO EXTENSION]'} : "
                f"{count:,}\n"
            )


        file.write(
            "\nTOP-LEVEL DIRECTORIES\n"
        )

        file.write("-" * 100 + "\n")

        for name, count in (
            top_level_counter.most_common()
        ):

            file.write(
                f"{name} : {count:,} entries\n"
            )


        file.write(
            "\nMEDICAL IMAGE SUMMARY\n"
        )

        file.write("-" * 100 + "\n")

        file.write(
            f"Total detected: "
            f"{len(medical_images):,}\n\n"
        )

        for item in medical_images[:1000]:

            file.write(
                f"{item['path']} | "
                f"{item['size_mb']:.3f} MB\n"
            )


        file.write(
            "\nSEGMENTATION / MASK SUMMARY\n"
        )

        file.write("-" * 100 + "\n")

        file.write(
            f"Total detected: "
            f"{len(segmentation_files):,}\n\n"
        )

        for item in segmentation_files[:1000]:

            file.write(
                f"{item['path']} | "
                f"{item['size_mb']:.3f} MB\n"
            )


        file.write(
            "\nMETADATA FILES\n"
        )

        file.write("-" * 100 + "\n")

        file.write(
            f"Total detected: "
            f"{len(metadata_files):,}\n\n"
        )

        for item in metadata_files[:1000]:

            file.write(
                f"{item['path']} | "
                f"{item['size_kb']:.3f} KB\n"
            )


        file.write(
            "\nPOSSIBLE PATIENT IDENTIFIERS\n"
        )

        file.write("-" * 100 + "\n")

        for patient, count in (
            patient_counter.most_common()
        ):

            file.write(
                f"{patient} : {count:,} files\n"
            )


        file.write(
            "\nPOSSIBLE TIMEPOINTS\n"
        )

        file.write("-" * 100 + "\n")

        for timepoint, count in (
            timepoint_counter.most_common()
        ):

            file.write(
                f"Timepoint {timepoint} : "
                f"{count:,} files\n"
            )


        file.write(
            "\nDIRECTORY STRUCTURE\n"
        )

        file.write("-" * 100 + "\n")

        for directory in sorted(
            unique_directories
        )[:2000]:

            file.write(
                f"{directory}\n"
            )


        file.write(
            "\nFIRST 500 FILES\n"
        )

        file.write("-" * 100 + "\n")

        for info in files[:500]:

            file.write(
                f"{normalize_path(info.filename)}\n"
            )


        file.write(
            "\nKEYWORD SAMPLES\n"
        )

        file.write("-" * 100 + "\n")

        for keyword, hits in (
            keyword_samples.items()
        ):

            file.write(
                f"\n[{keyword}]\n"
            )

            for hit in hits:

                file.write(
                    f"  {hit}\n"
                )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 100)
    print("LUMIERE INSPECTION COMPLETE")
    print("=" * 100)

    print("\nONLY THIS ZIP WAS INSPECTED:")
    print(ZIP_PATH)

    print("\nNO EXTRACTION WAS PERFORMED.")

    print("\nREPORTS SAVED TO:")

    print(JSON_REPORT)
    print(TXT_REPORT)

    print("\nNext step:")
    print(
        "Send me the TXT report or JSON report, "
        "and we will design 1_*.py based on the ACTUAL dataset structure."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    inspect_lumiere_zip()
