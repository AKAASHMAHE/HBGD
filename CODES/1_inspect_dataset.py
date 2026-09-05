# 1_inspect_dataset.py

import os
import json
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

FILES = {
    "patients": "patients.csv",
    "series": "series.json",
    "windows": "windows_mapping.json",
    "recist": "recist_measurements.csv"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def inspect_dataframe(name, path):
    print_header(f"{name.upper()} : {os.path.basename(path)}")

    try:
        df = pd.read_csv(path)

        print(f"Rows       : {len(df):,}")
        print(f"Columns    : {len(df.columns)}")
        print(f"File size  : {os.path.getsize(path) / 1024:.2f} KB")

        print("\nColumns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2}. {col}")

        print("\nData types:")
        print(df.dtypes.to_string())

        print("\nMissing values:")
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) == 0:
            print("  No missing values")
        else:
            for col, count in missing.items():
                percentage = count / len(df) * 100
                print(f"  {col}: {count:,} ({percentage:.2f}%)")

        print("\nFirst 5 rows:")
        print(df.head().to_string(index=False))

        print("\nUnique values per column:")
        for col in df.columns:
            try:
                unique_count = df[col].nunique(dropna=True)
                print(f"  {col}: {unique_count:,}")
            except Exception:
                pass

        return df

    except Exception as e:
        print(f"ERROR reading {path}")
        print(e)
        return None


def inspect_json(name, path):
    print_header(f"{name.upper()} : {os.path.basename(path)}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"File size: {os.path.getsize(path) / 1024:.2f} KB")
        print(f"Top-level Python type: {type(data).__name__}")

        if isinstance(data, dict):
            print(f"Number of top-level keys: {len(data):,}")

            print("\nFirst 20 keys:")
            for key in list(data.keys())[:20]:
                print(f"  {key}")

            print("\nStructure preview:")

            for key, value in list(data.items())[:3]:
                print(f"\nKEY: {key}")
                print(f"TYPE: {type(value).__name__}")

                if isinstance(value, dict):
                    print("SUB-KEYS:")
                    for subkey in list(value.keys())[:15]:
                        print(f"  {subkey}")

                elif isinstance(value, list):
                    print(f"LIST LENGTH: {len(value):,}")
                    if len(value) > 0:
                        print("FIRST ITEM:")
                        print(str(value[0])[:1000])

                else:
                    print("VALUE:")
                    print(str(value)[:1000])

        elif isinstance(data, list):
            print(f"Number of items: {len(data):,}")

            if len(data) > 0:
                print("\nFirst item:")
                print(str(data[0])[:2000])

        return data

    except Exception as e:
        print(f"ERROR reading {path}")
        print(e)
        return None


# ============================================================
# START
# ============================================================

print("\n")
print("#" * 80)
print("# DATASET STRUCTURE VERIFICATION - STEP 1")
print("#" * 80)

print(f"\nDataset directory:")
print(DATA_DIR)

print("\nChecking required files...")

missing_files = []

for key, filename in FILES.items():
    path = os.path.join(DATA_DIR, filename)

    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  [FOUND] {filename} ({size_kb:.2f} KB)")
    else:
        print(f"  [MISSING] {filename}")
        missing_files.append(filename)


if missing_files:
    print("\nERROR: The following files are missing:")
    for filename in missing_files:
        print(f"  - {filename}")

    print("\nPut all four files in the same folder as this Python script.")
    raise SystemExit(1)


# ============================================================
# READ CSV FILES
# ============================================================

patients_path = os.path.join(DATA_DIR, FILES["patients"])
recist_path = os.path.join(DATA_DIR, FILES["recist"])

patients_df = inspect_dataframe("PATIENTS", patients_path)
recist_df = inspect_dataframe("RECIST MEASUREMENTS", recist_path)


# ============================================================
# READ JSON FILES
# ============================================================

series_path = os.path.join(DATA_DIR, FILES["series"])
windows_path = os.path.join(DATA_DIR, FILES["windows"])

series_data = inspect_json("SERIES", series_path)
windows_data = inspect_json("WINDOWS MAPPING", windows_path)


# ============================================================
# BASIC PROJECT RELEVANCE CHECK
# ============================================================

print_header("INITIAL PROJECT RELEVANCE CHECK")

print("""
Our project requires longitudinal cancer imaging information.

The most important things we need to determine are:

1. PATIENT IDENTIFICATION
   Can records belonging to the same patient be connected?

2. MULTIPLE SCANS
   Does a patient have more than one CT scan / imaging series?

3. TIME
   Are scan dates or acquisition timestamps available?

4. TUMOR MEASUREMENTS
   Does RECIST contain target lesion measurements?

5. LONGITUDINAL CHANGE
   Can measurements from different time points be compared?

6. IMAGE/SERIES LINK
   Can RECIST measurements be connected to actual CT series?

7. PATIENT OUTCOME
   Is there enough information to determine response/progression?
""")


# ============================================================
# PATIENT CSV ANALYSIS
# ============================================================

if patients_df is not None:

    print_header("PATIENT FILE - POSSIBLE IDENTIFIERS")

    possible_id_columns = []

    for col in patients_df.columns:
        col_lower = col.lower()

        if any(term in col_lower for term in [
            "patient",
            "subject",
            "case",
            "study",
            "id"
        ]):
            possible_id_columns.append(col)

    if possible_id_columns:
        print("Possible identifier columns:")
        for col in possible_id_columns:
            print(f"  - {col}")
    else:
        print("No obvious patient identifier column found.")


# ============================================================
# RECIST ANALYSIS
# ============================================================

if recist_df is not None:

    print_header("RECIST FILE - IMPORTANT COLUMNS")

    important_terms = [
        "patient",
        "subject",
        "case",
        "study",
        "series",
        "date",
        "time",
        "lesion",
        "target",
        "diameter",
        "size",
        "measurement",
        "recist",
        "response",
        "progress",
        "baseline"
    ]

    found_columns = []

    for col in recist_df.columns:
        col_lower = col.lower()

        if any(term in col_lower for term in important_terms):
            found_columns.append(col)

    if found_columns:
        print("Potentially relevant RECIST columns:")

        for col in found_columns:
            print(f"  - {col}")
    else:
        print("No obvious RECIST-related columns detected.")


# ============================================================
# DATE/TIME DETECTION
# ============================================================

print_header("DATE / TIME DETECTION")

all_dfs = {
    "patients.csv": patients_df,
    "recist_measurements.csv": recist_df
}

for filename, df in all_dfs.items():

    if df is None:
        continue

    print(f"\n{filename}")

    date_columns = []

    for col in df.columns:
        col_lower = col.lower()

        if any(term in col_lower for term in [
            "date",
            "time",
            "timestamp",
            "acquisition",
            "study_date",
            "scan_date"
        ]):
            date_columns.append(col)

    if date_columns:
        print("  Possible date/time columns:")

        for col in date_columns:
            print(f"    - {col}")

            try:
                converted = pd.to_datetime(
                    df[col],
                    errors="coerce"
                )

                valid = converted.notna().sum()

                print(
                    f"      Valid date/time values: "
                    f"{valid:,}/{len(df):,}"
                )

                if valid > 0:
                    print(
                        f"      Earliest: {converted.min()}"
                    )
                    print(
                        f"      Latest  : {converted.max()}"
                    )

            except Exception as e:
                print(f"      Could not parse: {e}")

    else:
        print("  No obvious date/time column found.")


# ============================================================
# DUPLICATE / UNIQUE PATIENT ANALYSIS
# ============================================================

print_header("PATIENT COUNT ANALYSIS")

if patients_df is not None:

    for col in patients_df.columns:

        col_lower = col.lower()

        if any(term in col_lower for term in [
            "patient_id",
            "patientid",
            "patient",
            "subject_id",
            "subjectid"
        ]):

            try:
                unique_count = patients_df[col].nunique()

                print(
                    f"{col}: "
                    f"{unique_count:,} unique values "
                    f"out of {len(patients_df):,} rows"
                )

            except Exception:
                pass


# ============================================================
# RECIST PATIENT COUNT
# ============================================================

if recist_df is not None:

    print_header("RECIST PATIENT COUNT")

    for col in recist_df.columns:

        col_lower = col.lower()

        if any(term in col_lower for term in [
            "patient_id",
            "patientid",
            "patient",
            "subject_id",
            "subjectid"
        ]):

            try:
                unique_count = recist_df[col].nunique()

                print(
                    f"{col}: "
                    f"{unique_count:,} unique patients/IDs "
                    f"out of {len(recist_df):,} rows"
                )

            except Exception:
                pass


# ============================================================
# FINAL STATUS
# ============================================================

print_header("STEP 1 COMPLETE")

print("""
DO NOT START MODEL TRAINING YET.

This script only determines what information is actually present.

The next verification steps will depend on the output of this script.

IMPORTANT:

Send me the COMPLETE TERMINAL OUTPUT after running:

    python 1_inspect_dataset.py

Do not modify the output.

From that output we will determine whether these four files contain:

    Patient
       ↓
    CT Series
       ↓
    Scan Date / Time
       ↓
    Tumor / RECIST Measurement
       ↓
    Follow-up Measurement
       ↓
    Tumor Change / Response

If this chain exists, the dataset may be directly useful for our
longitudinal tumor-growth / response project.
""")

print("=" * 80)