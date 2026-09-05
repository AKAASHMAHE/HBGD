# 9_extract_hbgd_features.py

import os
import sys
import numpy as np
import pandas as pd


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

RECIST_GEOMETRY_FILE = os.path.join(
    DATA_DIR,
    "7_recist_geometry.csv"
)

PAIRWISE_FILE = os.path.join(
    DATA_DIR,
    "8_recist_pairwise_changes.csv"
)

TRAJECTORY_FILE = os.path.join(
    DATA_DIR,
    "8_recist_trajectory_summary.csv"
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
# LOAD FILE
# ============================================================

def load_csv(path, name):

    if not os.path.exists(path):

        print(
            "ERROR: Required file not found:"
        )

        print(path)

        sys.exit(1)

    df = pd.read_csv(path)

    print(
        name,
        "records:",
        len(df)
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

header(
    "STEP 9 - LOADING HBGD DATA"
)

geometry = load_csv(
    RECIST_GEOMETRY_FILE,
    "RECIST + geometry"
)

pairwise = load_csv(
    PAIRWISE_FILE,
    "Pairwise temporal"
)

trajectory = load_csv(
    TRAJECTORY_FILE,
    "Trajectory"
)


# ============================================================
# CHECK GEOMETRY COLUMNS
# ============================================================

header(
    "CHECKING REQUIRED GEOMETRY VARIABLES"
)

required_geometry = [
    "patient_id",
    "study_date_string",
    "lesion_label_alias",
    "lesion_label_value",
    "recist_measurement_mm",
    "volume_cm3"
]

missing = [
    column
    for column in required_geometry
    if column not in geometry.columns
]

if len(missing) > 0:

    print(
        "ERROR: Missing geometry columns:"
    )

    for column in missing:
        print(
            "   ",
            column
        )

    sys.exit(1)

print(
    "All required geometry variables are present."
)


# ============================================================
# NORMALIZE GEOMETRY
# ============================================================

header(
    "NORMALIZING GEOMETRY DATA"
)

geometry["patient_id"] = pd.to_numeric(
    geometry["patient_id"],
    errors="coerce"
)

geometry["study_date"] = pd.to_datetime(
    geometry["study_date_string"].astype(str),
    format="%Y%m%d",
    errors="coerce"
)

geometry["recist_measurement_mm"] = pd.to_numeric(
    geometry["recist_measurement_mm"],
    errors="coerce"
)

geometry["volume_cm3"] = pd.to_numeric(
    geometry["volume_cm3"],
    errors="coerce"
)


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

before = len(geometry)

geometry = geometry[
    geometry["recist_measurement_mm"].notna()
    &
    geometry["volume_cm3"].notna()
    &
    geometry["study_date"].notna()
].copy()

after = len(geometry)

print(
    "Valid RECIST geometry records:",
    after
)

print(
    "Removed invalid records:",
    before - after
)


# ============================================================
# FEATURE 1
# RECIST DIAMETER
# ============================================================

header(
    "FEATURE 1 - RECIST DIAMETER"
)

geometry["diameter_mm"] = (
    geometry[
        "recist_measurement_mm"
    ]
)

print(
    "Minimum:",
    round(
        geometry["diameter_mm"].min(),
        3
    ),
    "mm"
)

print(
    "Maximum:",
    round(
        geometry["diameter_mm"].max(),
        3
    ),
    "mm"
)

print(
    "Mean:",
    round(
        geometry["diameter_mm"].mean(),
        3
    ),
    "mm"
)

print(
    "Median:",
    round(
        geometry["diameter_mm"].median(),
        3
    ),
    "mm"
)


# ============================================================
# FEATURE 2
# SEGMENTED VOLUME
# ============================================================

header(
    "FEATURE 2 - SEGMENTED 3D VOLUME"
)

geometry["volume_cm3"] = (
    geometry["volume_cm3"]
)

print(
    "Minimum:",
    round(
        geometry["volume_cm3"].min(),
        3
    ),
    "cm3"
)

print(
    "Maximum:",
    round(
        geometry["volume_cm3"].max(),
        3
    ),
    "cm3"
)

print(
    "Mean:",
    round(
        geometry["volume_cm3"].mean(),
        3
    ),
    "cm3"
)

print(
    "Median:",
    round(
        geometry["volume_cm3"].median(),
        3
    ),
    "cm3"
)


# ============================================================
# FEATURE 3
# SPHERE-EQUIVALENT DIAMETER
# ============================================================

header(
    "FEATURE 3 - SPHERE-EQUIVALENT DIAMETER"
)

# Volume of sphere:
#
# V = (pi / 6) * d^3
#
# Therefore:
#
# d = (6V/pi)^(1/3)
#
# V must be in mm^3.

geometry["volume_mm3"] = (
    geometry["volume_cm3"]
    * 1000.0
)

geometry[
    "equivalent_diameter_mm"
] = (
    (
        6.0
        *
        geometry["volume_mm3"]
        /
        np.pi
    )
    ** (1.0 / 3.0)
)


print(
    "Minimum:",
    round(
        geometry[
            "equivalent_diameter_mm"
        ].min(),
        3
    ),
    "mm"
)

print(
    "Maximum:",
    round(
        geometry[
            "equivalent_diameter_mm"
        ].max(),
        3
    ),
    "mm"
)

print(
    "Mean:",
    round(
        geometry[
            "equivalent_diameter_mm"
        ].mean(),
        3
    ),
    "mm"
)

print(
    "Median:",
    round(
        geometry[
            "equivalent_diameter_mm"
        ].median(),
        3
    ),
    "mm"
)


# ============================================================
# FEATURE 4
# RECIST / VOLUME CONSISTENCY
# ============================================================

header(
    "FEATURE 4 - RECIST VS 3D GEOMETRY"
)

geometry[
    "equivalent_to_recist_ratio"
] = (
    geometry[
        "equivalent_diameter_mm"
    ]
    /
    geometry[
        "diameter_mm"
    ]
)


geometry[
    "equivalent_minus_recist_mm"
] = (
    geometry[
        "equivalent_diameter_mm"
    ]
    -
    geometry[
        "diameter_mm"
    ]
)


ratio = geometry[
    "equivalent_to_recist_ratio"
].replace(
    [np.inf, -np.inf],
    np.nan
).dropna()


difference = geometry[
    "equivalent_minus_recist_mm"
].dropna()


print(
    "Median equivalent/RECIST ratio:",
    round(
        ratio.median(),
        3
    )
)

print(
    "Mean equivalent/RECIST ratio:",
    round(
        ratio.mean(),
        3
    )
)

print(
    "Median diameter difference:",
    round(
        difference.median(),
        3
    ),
    "mm"
)

print(
    "Mean absolute diameter difference:",
    round(
        difference.abs().mean(),
        3
    ),
    "mm"
)


# ============================================================
# FEATURE 5
# BASELINE NORMALIZATION
# ============================================================

header(
    "FEATURE 5 - BASELINE NORMALIZATION"
)

geometry = geometry.sort_values(
    [
        "patient_id",
        "lesion_label_alias",
        "study_date"
    ]
).copy()


group_columns = [
    "patient_id",
    "lesion_label_alias"
]


geometry[
    "baseline_diameter_mm"
] = (
    geometry
    .groupby(group_columns)[
        "diameter_mm"
    ]
    .transform("first")
)


geometry[
    "baseline_volume_cm3"
] = (
    geometry
    .groupby(group_columns)[
        "volume_cm3"
    ]
    .transform("first")
)


geometry[
    "diameter_normalized"
] = (
    geometry[
        "diameter_mm"
    ]
    /
    geometry[
        "baseline_diameter_mm"
    ]
)


geometry[
    "volume_normalized"
] = (
    geometry[
        "volume_cm3"
    ]
    /
    geometry[
        "baseline_volume_cm3"
    ]
)


geometry[
    "diameter_percent_from_baseline"
] = (
    geometry[
        "diameter_normalized"
    ]
    -
    1.0
) * 100.0


geometry[
    "volume_percent_from_baseline"
] = (
    geometry[
        "volume_normalized"
    ]
    -
    1.0
) * 100.0


print(
    "Baseline-normalized features created."
)


# ============================================================
# FEATURE 6
# TIME FROM BASELINE
# ============================================================

header(
    "FEATURE 6 - TIME FROM BASELINE"
)

geometry[
    "baseline_date"
] = (
    geometry
    .groupby(group_columns)[
        "study_date"
    ]
    .transform("first")
)


geometry[
    "days_from_baseline"
] = (
    geometry[
        "study_date"
    ]
    -
    geometry[
        "baseline_date"
    ]
).dt.days


geometry[
    "months_from_baseline"
] = (
    geometry[
        "days_from_baseline"
    ]
    /
    30.4375
)


print(
    "Minimum time:",
    geometry[
        "days_from_baseline"
    ].min(),
    "days"
)

print(
    "Maximum time:",
    geometry[
        "days_from_baseline"
    ].max(),
    "days"
)


# ============================================================
# FEATURE 7
# PAIRWISE GROWTH RATE
# ============================================================

header(
    "FEATURE 7 - PAIRWISE GROWTH RATE"
)

pairwise["diameter_change_mm"] = pd.to_numeric(
    pairwise["diameter_change_mm"],
    errors="coerce"
)

pairwise["percent_change"] = pd.to_numeric(
    pairwise["percent_change"],
    errors="coerce"
)

pairwise["interval_days"] = pd.to_numeric(
    pairwise["interval_days"],
    errors="coerce"
)


pairwise[
    "diameter_growth_rate_mm_per_day"
] = (
    pairwise[
        "diameter_change_mm"
    ]
    /
    pairwise[
        "interval_days"
    ]
)


pairwise[
    "diameter_growth_rate_mm_per_month"
] = (
    pairwise[
        "diameter_growth_rate_mm_per_day"
    ]
    *
    30.4375
)


pairwise[
    "relative_growth_rate_per_day"
] = (
    (
        pairwise[
            "percent_change"
        ]
        /
        100.0
    )
    /
    pairwise[
        "interval_days"
    ]
)


pairwise[
    "relative_growth_rate_per_month"
] = (
    pairwise[
        "relative_growth_rate_per_day"
    ]
    *
    30.4375
)


print(
    "Mean diameter growth rate:",
    round(
        pairwise[
            "diameter_growth_rate_mm_per_day"
        ].mean(),
        5
    ),
    "mm/day"
)

print(
    "Median diameter growth rate:",
    round(
        pairwise[
            "diameter_growth_rate_mm_per_day"
        ].median(),
        5
    ),
    "mm/day"
)

print(
    "Mean relative growth rate:",
    round(
        pairwise[
            "relative_growth_rate_per_day"
        ].mean(),
        6
    ),
    "per day"
)


# ============================================================
# FEATURE 8
# EXPONENTIAL GROWTH RATE
# ============================================================

header(
    "FEATURE 8 - LOGARITHMIC GROWTH RATE"
)

valid_pairwise = pairwise[
    (
        pairwise[
            "previous_diameter_mm"
        ]
        > 0
    )
    &
    (
        pairwise[
            "current_diameter_mm"
        ]
        > 0
    )
    &
    (
        pairwise[
            "interval_days"
        ]
        > 0
    )
].copy()


valid_pairwise[
    "log_diameter_growth_rate_per_day"
] = (
    np.log(
        valid_pairwise[
            "current_diameter_mm"
        ]
        /
        valid_pairwise[
            "previous_diameter_mm"
        ]
    )
    /
    valid_pairwise[
        "interval_days"
    ]
)


valid_pairwise[
    "doubling_time_days"
] = np.where(
    valid_pairwise[
        "log_diameter_growth_rate_per_day"
    ] > 0,
    np.log(2.0)
    /
    valid_pairwise[
        "log_diameter_growth_rate_per_day"
    ],
    np.nan
)


print(
    "Valid logarithmic transitions:",
    len(valid_pairwise)
)


positive_growth = (
    valid_pairwise[
        "log_diameter_growth_rate_per_day"
    ]
    > 0
)


print(
    "Positive exponential-growth transitions:",
    int(
        positive_growth.sum()
    )
)


if positive_growth.sum() > 0:

    doubling = valid_pairwise[
        "doubling_time_days"
    ].dropna()

    print(
        "Median positive-growth doubling time:",
        round(
            doubling.median(),
            2
        ),
        "days"
    )


# ============================================================
# FEATURE 9
# VOLUME GROWTH APPROXIMATION
# ============================================================

header(
    "FEATURE 9 - VOLUME-BASED GROWTH"
)

# If diameter changes approximately as d,
# spherical volume scales approximately as d^3.
#
# This is NOT a claim that tumors are spheres.
# It is only a derived mathematical feature.

pairwise[
    "previous_diameter_mm"
] = pd.to_numeric(
    pairwise[
        "previous_diameter_mm"
    ],
    errors="coerce"
)

pairwise[
    "current_diameter_mm"
] = pd.to_numeric(
    pairwise[
        "current_diameter_mm"
    ],
    errors="coerce"
)


pairwise[
    "approx_volume_ratio_from_diameter"
] = (
    pairwise[
        "current_diameter_mm"
    ]
    /
    pairwise[
        "previous_diameter_mm"
    ]
) ** 3


pairwise[
    "approx_volume_change_percent"
] = (
    pairwise[
        "approx_volume_ratio_from_diameter"
    ]
    -
    1.0
) * 100.0


print(
    "Mean estimated volume change:",
    round(
        pairwise[
            "approx_volume_change_percent"
        ].mean(),
        3
    ),
    "%"
)

print(
    "Median estimated volume change:",
    round(
        pairwise[
            "approx_volume_change_percent"
        ].median(),
        3
    ),
    "%"
)


# ============================================================
# FEATURE 10
# HBGD CORE FEATURE TABLE
# ============================================================

header(
    "CREATING HBGD CORE FEATURE TABLE"
)

feature_columns = [

    "patient_id",

    "study_date",

    "study_date_string",

    "lesion_label_alias",

    "lesion_label_value",

    "diameter_mm",

    "volume_cm3",

    "volume_mm3",

    "equivalent_diameter_mm",

    "equivalent_to_recist_ratio",

    "equivalent_minus_recist_mm",

    "baseline_diameter_mm",

    "baseline_volume_cm3",

    "diameter_normalized",

    "volume_normalized",

    "diameter_percent_from_baseline",

    "volume_percent_from_baseline",

    "days_from_baseline",

    "months_from_baseline"
]


hbgd_features = geometry[
    feature_columns
].copy()


print(
    "HBGD feature records:",
    len(hbgd_features)
)

print(
    "HBGD features:",
    len(
        feature_columns
    )
)


# ============================================================
# LONGITUDINAL FEATURE TABLE
# ============================================================

header(
    "CREATING LONGITUDINAL HBGD TABLE"
)

longitudinal_features = (
    pairwise.copy()
)


# ============================================================
# FEATURE QUALITY CHECK
# ============================================================

header(
    "FEATURE QUALITY CHECK"
)

for column in feature_columns:

    missing_count = int(
        hbgd_features[
            column
        ].isna().sum()
    )

    missing_percent = (
        missing_count
        /
        len(hbgd_features)
        *
        100.0
    )

    print(
        column,
        ":",
        missing_count,
        "missing",
        "(",
        round(
            missing_percent,
            2
        ),
        "%)"
    )


# ============================================================
# HBGD VARIABLE AVAILABILITY
# ============================================================

header(
    "HBGD VARIABLE AVAILABILITY"
)

availability = {

    "Tumor diameter":
        hbgd_features[
            "diameter_mm"
        ].notna().sum(),

    "Tumor volume":
        hbgd_features[
            "volume_cm3"
        ].notna().sum(),

    "Equivalent diameter":
        hbgd_features[
            "equivalent_diameter_mm"
        ].notna().sum(),

    "Time from baseline":
        hbgd_features[
            "days_from_baseline"
        ].notna().sum(),

    "Baseline diameter":
        hbgd_features[
            "baseline_diameter_mm"
        ].notna().sum(),

    "Baseline volume":
        hbgd_features[
            "baseline_volume_cm3"
        ].notna().sum()
}


for name, count in availability.items():

    print(
        name,
        ":",
        count,
        "/",
        len(hbgd_features)
    )


# ============================================================
# FINAL HBGD FEASIBILITY
# ============================================================

header(
    "STEP 9 - HBGD FEATURE FEASIBILITY"
)


diameter_ok = (
    hbgd_features[
        "diameter_mm"
    ].notna().all()
)

volume_ok = (
    hbgd_features[
        "volume_cm3"
    ].notna().all()
)

time_ok = (
    hbgd_features[
        "days_from_baseline"
    ].notna().all()
)

growth_ok = (
    len(valid_pairwise)
    > 0
)


print(
    "["
    +
    (
        "YES"
        if diameter_ok
        else "NO"
    )
    +
    "] RECIST diameter available"
)


print(
    "["
    +
    (
        "YES"
        if volume_ok
        else "NO"
    )
    +
    "] 3D lesion volume available"
)


print(
    "["
    +
    (
        "YES"
        if time_ok
        else "NO"
    )
    +
    "] Time-from-baseline available"
)


print(
    "["
    +
    (
        "YES"
        if growth_ok
        else "NO"
    )
    +
    "] Longitudinal growth-rate features available"
)


# ============================================================
# IMPORTANT INTERPRETATION
# ============================================================

print()

print(
    "=" * 90
)

print(
    "IMPORTANT INTERPRETATION"
)

print(
    "=" * 90
)

print("""
This step extracts quantitative variables that can potentially
serve as inputs or observables for the HBGD architecture.

The following are directly measured or derived from the dataset:

    RECIST diameter
    segmented 3D volume
    time from baseline
    baseline-normalized diameter
    baseline-normalized volume
    diameter change
    relative growth rate
    logarithmic growth rate
    approximate volume change

The sphere-equivalent diameter and diameter-derived volume
features are mathematical transformations, not additional
biological measurements.

This step does NOT establish biological causality and does
NOT validate the HBGD model.

DO NOT TRAIN A MODEL YET.
""")


# ============================================================
# SAVE OUTPUTS
# ============================================================

header(
    "SAVING STEP 9 RESULTS"
)


feature_file = os.path.join(
    DATA_DIR,
    "9_hbgd_features.csv"
)

longitudinal_file = os.path.join(
    DATA_DIR,
    "9_hbgd_longitudinal_features.csv"
)

growth_file = os.path.join(
    DATA_DIR,
    "9_hbgd_log_growth_features.csv"
)


hbgd_features.to_csv(
    feature_file,
    index=False
)

longitudinal_features.to_csv(
    longitudinal_file,
    index=False
)

valid_pairwise.to_csv(
    growth_file,
    index=False
)


print(
    "Saved:",
    feature_file
)

print(
    "Saved:",
    longitudinal_file
)

print(
    "Saved:",
    growth_file
)


# ============================================================
# COMPLETE
# ============================================================

header(
    "STEP 9 COMPLETE"
)

print("""
Run:

    python 9_extract_hbgd_features.py

Then send me the COMPLETE terminal output.

Do not train the model yet.
""")

print(
    "=" * 90
)

print(
    "END OF STEP 9"
)

print(
    "=" * 90
)