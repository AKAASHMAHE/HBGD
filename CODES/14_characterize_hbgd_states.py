# 14_characterize_hbgd_states.py

import os
import sys
import warnings

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from scipy.stats import spearmanr


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

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "DATA"
)

FEATURE_FILE = os.path.join(
    DATA_DIR,
    "9_hbgd_features.csv"
)

PAIRWISE_FILE = os.path.join(
    DATA_DIR,
    "8_recist_pairwise_changes.csv"
)


# ============================================================
# HEADER
# ============================================================

def header(text):

    print()
    print("=" * 95)
    print(text)
    print("=" * 95)


# ============================================================
# DATE CONVERSION
# ============================================================

def convert_yyyymmdd(value):

    if pd.isna(value):
        return pd.NaT

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    if len(text) != 8:
        return pd.NaT

    return pd.to_datetime(
        text,
        format="%Y%m%d",
        errors="coerce"
    )


# ============================================================
# LOAD DATA
# ============================================================

header(
    "STEP 14 - LOADING HBGD STATE DATA"
)


if not os.path.exists(FEATURE_FILE):

    print(
        "ERROR: Missing file:"
    )

    print(
        FEATURE_FILE
    )

    sys.exit(1)


if not os.path.exists(PAIRWISE_FILE):

    print(
        "ERROR: Missing file:"
    )

    print(
        PAIRWISE_FILE
    )

    sys.exit(1)


features = pd.read_csv(
    FEATURE_FILE
)

pairwise = pd.read_csv(
    PAIRWISE_FILE
)


print(
    "Feature records:",
    len(features)
)

print(
    "Pairwise records:",
    len(pairwise)
)


# ============================================================
# CHECK COLUMNS
# ============================================================

header(
    "CHECKING REQUIRED VARIABLES"
)


required_feature_columns = [

    "patient_id",
    "study_date",
    "lesion_label_alias",
    "diameter_mm",
    "volume_cm3"
]


required_pairwise_columns = [

    "patient_id",
    "lesion_alias",
    "previous_date",
    "current_date",
    "previous_diameter_mm",
    "current_diameter_mm",
    "interval_days"
]


missing_features = [

    c
    for c in required_feature_columns
    if c not in features.columns
]


missing_pairwise = [

    c
    for c in required_pairwise_columns
    if c not in pairwise.columns
]


if missing_features:

    print(
        "ERROR: Missing feature columns:"
    )

    for c in missing_features:
        print(
            "   ",
            c
        )

    sys.exit(1)


if missing_pairwise:

    print(
        "ERROR: Missing pairwise columns:"
    )

    for c in missing_pairwise:
        print(
            "   ",
            c
        )

    sys.exit(1)


print(
    "All required variables are present."
)


# ============================================================
# NORMALIZE FEATURES
# ============================================================

header(
    "NORMALIZING FEATURE DATA"
)


features["patient_id"] = pd.to_numeric(
    features["patient_id"],
    errors="coerce"
)


features["study_date"] = pd.to_datetime(
    features["study_date"].astype(str).str.strip(),
    format="%Y-%m-%d",
    errors="coerce"
)


features["diameter_mm"] = pd.to_numeric(
    features["diameter_mm"],
    errors="coerce"
)


features["volume_cm3"] = pd.to_numeric(
    features["volume_cm3"],
    errors="coerce"
)


features["lesion_label_alias"] = (
    features[
        "lesion_label_alias"
    ]
    .astype(str)
    .str.strip()
)


# ============================================================
# NORMALIZE PAIRWISE DATA
# ============================================================

pairwise["patient_id"] = pd.to_numeric(
    pairwise["patient_id"],
    errors="coerce"
)


pairwise["previous_date"] = (
    pairwise[
        "previous_date"
    ]
    .apply(
        convert_yyyymmdd
    )
)


pairwise["current_date"] = (
    pairwise[
        "current_date"
    ]
    .apply(
        convert_yyyymmdd
    )
)


pairwise["previous_diameter_mm"] = pd.to_numeric(
    pairwise["previous_diameter_mm"],
    errors="coerce"
)


pairwise["current_diameter_mm"] = pd.to_numeric(
    pairwise["current_diameter_mm"],
    errors="coerce"
)


pairwise["interval_days"] = pd.to_numeric(
    pairwise["interval_days"],
    errors="coerce"
)


pairwise["lesion_alias"] = (
    pairwise[
        "lesion_alias"
    ]
    .astype(str)
    .str.strip()
)


# ============================================================
# CLEAN
# ============================================================

features = features.dropna(
    subset=[
        "patient_id",
        "study_date",
        "lesion_label_alias",
        "diameter_mm",
        "volume_cm3"
    ]
).copy()


pairwise = pairwise.dropna(
    subset=[
        "patient_id",
        "previous_date",
        "current_date",
        "lesion_alias",
        "previous_diameter_mm",
        "current_diameter_mm",
        "interval_days"
    ]
).copy()


features = features[
    features[
        "diameter_mm"
    ] > 0
]


features = features[
    features[
        "volume_cm3"
    ] > 0
]


pairwise = pairwise[
    pairwise[
        "previous_diameter_mm"
    ] > 0
]


pairwise = pairwise[
    pairwise[
        "current_diameter_mm"
    ] > 0
]


pairwise = pairwise[
    pairwise[
        "interval_days"
    ] > 0
]


# ============================================================
# BUILD FEATURE LOOKUP
# ============================================================

header(
    "BUILDING LONGITUDINAL GEOMETRY TABLE"
)


feature_groups = {}


for (
    patient_id,
    alias
), group in features.groupby(
    [
        "patient_id",
        "lesion_label_alias"
    ]
):

    group = group.sort_values(
        "study_date"
    ).copy()

    feature_groups[
        (
            int(patient_id),
            alias
        )
    ] = group


records = []


matched = 0


for _, row in pairwise.iterrows():

    patient_id = int(
        row[
            "patient_id"
        ]
    )

    alias = str(
        row[
            "lesion_alias"
        ]
    ).strip()


    key = (
        patient_id,
        alias
    )


    if key not in feature_groups:
        continue


    group = feature_groups[
        key
    ]


    previous_date = row[
        "previous_date"
    ]

    current_date = row[
        "current_date"
    ]


    previous_rows = group[
        group[
            "study_date"
        ]
        ==
        previous_date
    ]


    current_rows = group[
        group[
            "study_date"
        ]
        ==
        current_date
    ]


    if len(previous_rows) == 0:
        continue


    if len(current_rows) == 0:
        continue


    previous = previous_rows.iloc[0]
    current = current_rows.iloc[0]


    records.append({

        "patient_id":
            patient_id,

        "lesion_alias":
            alias,

        "previous_date":
            previous_date,

        "current_date":
            current_date,

        "interval_days":
            (
                current_date
                -
                previous_date
            ).days,

        "previous_diameter_mm":
            previous[
                "diameter_mm"
            ],

        "current_diameter_mm":
            current[
                "diameter_mm"
            ],

        "previous_volume_cm3":
            previous[
                "volume_cm3"
            ],

        "current_volume_cm3":
            current[
                "volume_cm3"
            ]
    })


    matched += 1


temporal = pd.DataFrame(
    records
)


print(
    "Matched longitudinal transitions:",
    matched
)


if len(temporal) == 0:

    print(
        "ERROR: No longitudinal transitions found."
    )

    sys.exit(1)


# ============================================================
# DERIVED DYNAMIC VARIABLES
# ============================================================

header(
    "CALCULATING DYNAMIC STATE VARIABLES"
)


temporal[
    "diameter_change_mm"
] = (
    temporal[
        "current_diameter_mm"
    ]
    -
    temporal[
        "previous_diameter_mm"
    ]
)


temporal[
    "diameter_percent_change"
] = (
    temporal[
        "diameter_change_mm"
    ]
    /
    temporal[
        "previous_diameter_mm"
    ]
    *
    100.0
)


temporal[
    "diameter_rate_mm_day"
] = (
    temporal[
        "diameter_change_mm"
    ]
    /
    temporal[
        "interval_days"
    ]
)


temporal[
    "log_growth_rate"
] = (
    np.log(
        temporal[
            "current_diameter_mm"
        ]
        /
        temporal[
            "previous_diameter_mm"
        ]
    )
    /
    temporal[
        "interval_days"
    ]
)


temporal[
    "volume_change_cm3"
] = (
    temporal[
        "current_volume_cm3"
    ]
    -
    temporal[
        "previous_volume_cm3"
    ]
)


temporal[
    "volume_percent_change"
] = (
    temporal[
        "volume_change_cm3"
    ]
    /
    temporal[
        "previous_volume_cm3"
    ]
    *
    100.0
)


temporal[
    "volume_rate_cm3_day"
] = (
    temporal[
        "volume_change_cm3"
    ]
    /
    temporal[
        "interval_days"
    ]
)


# ============================================================
# CLASSIFY DYNAMIC STATES
# ============================================================

header(
    "CLASSIFYING OBSERVED DYNAMIC STATES"
)


# ------------------------------------------------------------
# State definition
#
# A small change around zero is considered stable.
#
# Threshold:
# absolute diameter change <= 1 mm
#
# This is a descriptive threshold for this feasibility study,
# NOT a clinical RECIST response criterion.
# ------------------------------------------------------------

STABLE_THRESHOLD_MM = 1.0


def classify_state(change):

    if change > STABLE_THRESHOLD_MM:

        return "growth"

    elif change < -STABLE_THRESHOLD_MM:

        return "shrinkage"

    else:

        return "stable"


temporal[
    "dynamic_state"
] = temporal[
    "diameter_change_mm"
].apply(
    classify_state
)


print(
    "State threshold:",
    STABLE_THRESHOLD_MM,
    "mm"
)


state_counts = (
    temporal[
        "dynamic_state"
    ]
    .value_counts()
)


print()

print(
    state_counts.to_string()
)


# ============================================================
# VOLUME STATE
# ============================================================

def classify_volume_state(change):

    if change > 5.0:

        return "volume_growth"

    elif change < -5.0:

        return "volume_shrinkage"

    else:

        return "volume_stable"


temporal[
    "volume_state"
] = temporal[
    "volume_percent_change"
].apply(
    classify_volume_state
)


print()

print(
    "Volume state distribution:"
)


print(
    temporal[
        "volume_state"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# DIAMETER VS VOLUME STATE AGREEMENT
# ============================================================

header(
    "DIAMETER VS VOLUME DYNAMIC STATE AGREEMENT"
)


state_table = pd.crosstab(
    temporal[
        "dynamic_state"
    ],
    temporal[
        "volume_state"
    ]
)


print(
    state_table.to_string()
)


# ============================================================
# STATE TRANSITIONS
# ============================================================

header(
    "DYNAMIC STATE TRANSITIONS"
)


# ------------------------------------------------------------
# Sort trajectories chronologically
# ------------------------------------------------------------

temporal = temporal.sort_values(
    [
        "patient_id",
        "lesion_alias",
        "previous_date"
    ]
).reset_index(
    drop=True
)


temporal[
    "previous_dynamic_state"
] = (
    temporal
    .groupby(
        [
            "patient_id",
            "lesion_alias"
        ]
    )[
        "dynamic_state"
    ]
    .shift(1)
)


transition_records = []


for _, row in temporal.dropna(
    subset=[
        "previous_dynamic_state"
    ]
).iterrows():

    transition_records.append({

        "patient_id":
            row[
                "patient_id"
            ],

        "lesion_alias":
            row[
                "lesion_alias"
            ],

        "previous_date":
            row[
                "previous_date"
            ],

        "current_date":
            row[
                "current_date"
            ],

        "previous_state":
            row[
                "previous_dynamic_state"
            ],

        "current_state":
            row[
                "dynamic_state"
            ]
    })


state_transitions = pd.DataFrame(
    transition_records
)


print(
    "State transitions available:",
    len(state_transitions)
)


if len(state_transitions) > 0:

    transition_matrix = pd.crosstab(
        state_transitions[
            "previous_state"
        ],
        state_transitions[
            "current_state"
        ]
    )

    print()

    print(
        transition_matrix.to_string()
    )


# ============================================================
# LESION-LEVEL BEHAVIOR
# ============================================================

header(
    "LESION-LEVEL DYNAMIC BEHAVIOR"
)


lesion_summary = (
    temporal
    .groupby(
        [
            "patient_id",
            "lesion_alias"
        ]
    )
    .agg(

        transitions=(
            "dynamic_state",
            "count"
        ),

        mean_diameter_rate=(
            "diameter_rate_mm_day",
            "mean"
        ),

        median_diameter_rate=(
            "diameter_rate_mm_day",
            "median"
        ),

        diameter_rate_std=(
            "diameter_rate_mm_day",
            "std"
        ),

        mean_volume_percent_change=(
            "volume_percent_change",
            "mean"
        ),

        growth_transitions=(
            "dynamic_state",
            lambda x:
                (x == "growth").sum()
        ),

        shrinkage_transitions=(
            "dynamic_state",
            lambda x:
                (x == "shrinkage").sum()
        ),

        stable_transitions=(
            "dynamic_state",
            lambda x:
                (x == "stable").sum()
        )
    )
    .reset_index()
)


def behavior_type(row):

    growth = row[
        "growth_transitions"
    ]

    shrinkage = row[
        "shrinkage_transitions"
    ]

    stable = row[
        "stable_transitions"
    ]


    if growth > 0 and shrinkage > 0:

        return "mixed"


    if growth > 0 and shrinkage == 0:

        return "growth_dominant"


    if shrinkage > 0 and growth == 0:

        return "shrinkage_dominant"


    if stable > 0:

        return "stable_dominant"


    return "unknown"


lesion_summary[
    "behavior_type"
] = lesion_summary.apply(
    behavior_type,
    axis=1
)


print(
    lesion_summary.to_string(
        index=False
    )
)


# ============================================================
# BEHAVIOR DISTRIBUTION
# ============================================================

header(
    "LESION BEHAVIOR DISTRIBUTION"
)


print(
    lesion_summary[
        "behavior_type"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# WITHIN-LESION RATE VARIABILITY
# ============================================================

header(
    "WITHIN-LESION RATE VARIABILITY"
)


multi_transition = lesion_summary[
    lesion_summary[
        "transitions"
    ] >= 2
].copy()


print(
    "Lesions with >=2 transitions:",
    len(multi_transition)
)


if len(multi_transition) > 0:

    print(
        "Median rate SD:",
        multi_transition[
            "diameter_rate_std"
        ].median()
    )

    print(
        "Mean rate SD:",
        multi_transition[
            "diameter_rate_std"
        ].mean()
    )


# ============================================================
# DIAMETER-VOLUME RELATIONSHIP
# ============================================================

header(
    "DIAMETER-VOLUME RELATIONSHIP"
)


valid_geometry = features.dropna(
    subset=[
        "diameter_mm",
        "volume_cm3"
    ]
).copy()


if len(valid_geometry) >= 3:

    pearson_r, pearson_p = pearsonr(
        valid_geometry[
            "diameter_mm"
        ],
        valid_geometry[
            "volume_cm3"
        ]
    )


    spearman_r, spearman_p = spearmanr(
        valid_geometry[
            "diameter_mm"
        ],
        valid_geometry[
            "volume_cm3"
        ]
    )


    print(
        "Observations:",
        len(valid_geometry)
    )

    print(
        "Pearson r:",
        round(
            pearson_r,
            5
        )
    )

    print(
        "Pearson p:",
        pearson_p
    )

    print(
        "Spearman rho:",
        round(
            spearman_r,
            5
        )
    )

    print(
        "Spearman p:",
        spearman_p
    )


# ============================================================
# CHANGE CORRELATION
# ============================================================

header(
    "DIAMETER CHANGE VS VOLUME CHANGE"
)


valid_change = temporal.dropna(
    subset=[
        "diameter_change_mm",
        "volume_percent_change"
    ]
).copy()


if len(valid_change) >= 3:

    pearson_r, pearson_p = pearsonr(
        valid_change[
            "diameter_change_mm"
        ],
        valid_change[
            "volume_percent_change"
        ]
    )


    spearman_r, spearman_p = spearmanr(
        valid_change[
            "diameter_change_mm"
        ],
        valid_change[
            "volume_percent_change"
        ]
    )


    print(
        "Transitions:",
        len(valid_change)
    )

    print(
        "Pearson r:",
        round(
            pearson_r,
            5
        )
    )

    print(
        "Pearson p:",
        pearson_p
    )

    print(
        "Spearman rho:",
        round(
            spearman_r,
            5
        )
    )

    print(
        "Spearman p:",
        spearman_p
    )


# ============================================================
# STATE PERSISTENCE
# ============================================================

header(
    "DYNAMIC STATE PERSISTENCE"
)


if len(state_transitions) > 0:

    same_state = (
        state_transitions[
            "previous_state"
        ]
        ==
        state_transitions[
            "current_state"
        ]
    )


    persistence_rate = (
        same_state.mean()
        *
        100
    )


    print(
        "Transitions evaluated:",
        len(state_transitions)
    )


    print(
        "Same-state transitions:",
        same_state.sum()
    )


    print(
        "State persistence:",
        round(
            persistence_rate,
            2
        ),
        "%"
    )


# ============================================================
# PATIENT-LEVEL STATES
# ============================================================

header(
    "PATIENT-LEVEL DYNAMIC STATES"
)


patient_state_summary = (
    temporal
    .groupby(
        "patient_id"
    )
    .agg(

        transitions=(
            "dynamic_state",
            "count"
        ),

        growth=(
            "dynamic_state",
            lambda x:
                (x == "growth").sum()
        ),

        shrinkage=(
            "dynamic_state",
            lambda x:
                (x == "shrinkage").sum()
        ),

        stable=(
            "dynamic_state",
            lambda x:
                (x == "stable").sum()
        ),

        mean_rate=(
            "diameter_rate_mm_day",
            "mean"
        ),

        mean_volume_change=(
            "volume_percent_change",
            "mean"
        )
    )
    .reset_index()
)


print(
    patient_state_summary.to_string(
        index=False
    )
)


# ============================================================
# STATE SEPARABILITY
# ============================================================

header(
    "STATE SEPARABILITY"
)


state_rate_summary = (
    temporal
    .groupby(
        "dynamic_state"
    )
    .agg(

        observations=(
            "diameter_rate_mm_day",
            "count"
        ),

        mean_rate=(
            "diameter_rate_mm_day",
            "mean"
        ),

        median_rate=(
            "diameter_rate_mm_day",
            "median"
        ),

        mean_volume_change=(
            "volume_percent_change",
            "mean"
        ),

        median_volume_change=(
            "volume_percent_change",
            "median"
        )
    )
    .reset_index()
)


print(
    state_rate_summary.to_string(
        index=False
    )
)


# ============================================================
# HBGD STATE INTERPRETATION
# ============================================================

header(
    "STEP 14 - HBGD STATE INTERPRETATION"
)


print("""
This analysis does not train a predictive model.

It asks whether the dataset contains distinguishable
dynamic behaviors that could eventually be represented as
latent HBGD states.

Candidate observable states are:

    growth
    shrinkage
    stable

The analysis also examines:

    volume behavior
    diameter-volume agreement
    state persistence
    state switching
    lesion-level consistency
    patient-level heterogeneity

These are observable response states.

They should NOT automatically be interpreted as biological
states such as proliferation, hypoxia, necrosis, or immune
activity.

Those biological mechanisms require additional evidence or
measurements.

The purpose of this step is to determine whether an
observable dynamical state representation is justified by
the available CT/RECIST data.
""")


# ============================================================
# FEASIBILITY DECISION
# ============================================================

header(
    "STEP 14 - FINAL FEASIBILITY"
)


num_states = (
    temporal[
        "dynamic_state"
    ]
    .nunique()
)


num_behaviors = (
    lesion_summary[
        "behavior_type"
    ]
    .nunique()
)


if num_states >= 2:

    print(
        "[YES] Multiple observable dynamic states exist"
    )

else:

    print(
        "[NO] Only one observable dynamic state exists"
    )


if num_behaviors >= 2:

    print(
        "[YES] Multiple lesion behavior patterns exist"
    )

else:

    print(
        "[NO] Lesion behavior is effectively uniform"
    )


if len(state_transitions) > 0:

    print(
        "[YES] Dynamic state transitions can be analyzed"
    )

else:

    print(
        "[NO] Insufficient repeated transitions for state analysis"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

header(
    "SAVING STEP 14 RESULTS"
)


temporal_file = os.path.join(
    DATA_DIR,
    "14_hbgd_dynamic_states.csv"
)


lesion_file = os.path.join(
    DATA_DIR,
    "14_hbgd_lesion_behaviors.csv"
)


transition_file = os.path.join(
    DATA_DIR,
    "14_hbgd_state_transitions.csv"
)


patient_file = os.path.join(
    DATA_DIR,
    "14_hbgd_patient_states.csv"
)


state_summary_file = os.path.join(
    DATA_DIR,
    "14_hbgd_state_summary.csv"
)


temporal.to_csv(
    temporal_file,
    index=False
)


lesion_summary.to_csv(
    lesion_file,
    index=False
)


state_transitions.to_csv(
    transition_file,
    index=False
)


patient_state_summary.to_csv(
    patient_file,
    index=False
)


state_rate_summary.to_csv(
    state_summary_file,
    index=False
)


print(
    "Saved:",
    temporal_file
)

print(
    "Saved:",
    lesion_file
)

print(
    "Saved:",
    transition_file
)

print(
    "Saved:",
    patient_file
)

print(
    "Saved:",
    state_summary_file
)


# ============================================================
# COMPLETE
# ============================================================

header(
    "STEP 14 COMPLETE"
)


print("""
Run:

    python 14_characterize_hbgd_states.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A NEURAL NETWORK YET.
DO NOT CLAIM BIOLOGICAL STATES YET.
""")

print(
    "=" * 95
)

print(
    "END OF STEP 14"
)

print(
    "=" * 95
)