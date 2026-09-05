# 13_test_hbgd_feature_value.py

import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


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

    """
    Converts dates such as:

        20210615
        20230117

    into pandas timestamps:

        2021-06-15
        2023-01-17
    """

    if pd.isna(value):

        return pd.NaT

    text = str(value).strip()

    # Remove decimal representation if present
    if text.endswith(".0"):

        text = text[:-2]

    # Remove spaces
    text = text.strip()

    if len(text) != 8:

        return pd.NaT

    try:

        return pd.to_datetime(
            text,
            format="%Y%m%d",
            errors="coerce"
        )

    except Exception:

        return pd.NaT


# ============================================================
# LOAD DATA
# ============================================================

header(
    "STEP 13 - LOADING HBGD FEATURE DATA"
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
# REQUIRED COLUMNS
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
# NORMALIZE DATA
# ============================================================

header(
    "NORMALIZING DATA"
)


features["patient_id"] = pd.to_numeric(
    features["patient_id"],
    errors="coerce"
)


pairwise["patient_id"] = pd.to_numeric(
    pairwise["patient_id"],
    errors="coerce"
)


# ------------------------------------------------------------
# FEATURE DATES
# ------------------------------------------------------------

features[
    "study_date"
] = pd.to_datetime(
    features[
        "study_date"
    ].astype(str).str.strip(),
    format="%Y-%m-%d",
    errors="coerce"
)


# ------------------------------------------------------------
# PAIRWISE DATES
# ------------------------------------------------------------

pairwise[
    "previous_date"
] = pairwise[
    "previous_date"
].apply(
    convert_yyyymmdd
)


pairwise[
    "current_date"
] = pairwise[
    "current_date"
].apply(
    convert_yyyymmdd
)


# ------------------------------------------------------------
# NUMERIC VARIABLES
# ------------------------------------------------------------

features[
    "diameter_mm"
] = pd.to_numeric(
    features[
        "diameter_mm"
    ],
    errors="coerce"
)


features[
    "volume_cm3"
] = pd.to_numeric(
    features[
        "volume_cm3"
    ],
    errors="coerce"
)


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
    "interval_days"
] = pd.to_numeric(
    pairwise[
        "interval_days"
    ],
    errors="coerce"
)


# ------------------------------------------------------------
# ALIASES
# ------------------------------------------------------------

features[
    "lesion_label_alias"
] = (
    features[
        "lesion_label_alias"
    ]
    .astype(str)
    .str.strip()
)


pairwise[
    "lesion_alias"
] = (
    pairwise[
        "lesion_alias"
    ]
    .astype(str)
    .str.strip()
)


# ============================================================
# DATE VALIDATION
# ============================================================

header(
    "DATE CONVERSION CHECK"
)


print(
    "Feature valid dates:",
    features[
        "study_date"
    ].notna().sum(),
    "/",
    len(features)
)


print(
    "Pairwise valid previous dates:",
    pairwise[
        "previous_date"
    ].notna().sum(),
    "/",
    len(pairwise)
)


print(
    "Pairwise valid current dates:",
    pairwise[
        "current_date"
    ].notna().sum(),
    "/",
    len(pairwise)
)


print()

print(
    "Example pairwise date conversion:"
)


for _, row in pairwise.head(
    5
).iterrows():

    print(
        " ",
        row[
            "previous_date"
        ].date(),
        "->",
        row[
            "current_date"
        ].date()
    )


# ============================================================
# BUILD FEATURE LOOKUP
# ============================================================

header(
    "BUILDING FEATURE LOOKUP"
)


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
# GROUP FEATURES BY PATIENT + LESION
# ============================================================

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


print(
    "Patient-lesion feature groups:",
    len(feature_groups)
)


# ============================================================
# MATCH TRANSITIONS
# ============================================================

header(
    "MATCHING LONGITUDINAL TRANSITIONS"
)


records = []

matched = 0

unmatched = 0


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

        unmatched += 1

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


    # --------------------------------------------------------
    # EXACT DATE MATCH
    # --------------------------------------------------------

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

        unmatched += 1

        continue


    if len(current_rows) == 0:

        unmatched += 1

        continue


    previous_row = (
        previous_rows
        .iloc[0]
    )


    current_row = (
        current_rows
        .iloc[0]
    )


    actual_interval = (
        current_date
        -
        previous_date
    ).days


    if actual_interval <= 0:

        unmatched += 1

        continue


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
            actual_interval,

        "previous_diameter_mm":
            previous_row[
                "diameter_mm"
            ],

        "current_diameter_mm":
            current_row[
                "diameter_mm"
            ],

        "previous_volume_cm3":
            previous_row[
                "volume_cm3"
            ],

        "current_volume_cm3":
            current_row[
                "volume_cm3"
            ]
    })


    matched += 1


temporal = pd.DataFrame(
    records
)


print(
    "Matched transitions:",
    matched
)


print(
    "Unmatched transitions:",
    unmatched
)


if len(temporal) == 0:

    print()

    print(
        "ERROR: Still unable to match transitions."
    )

    sys.exit(1)


# ============================================================
# DERIVED VARIABLES
# ============================================================

header(
    "CREATING HBGD DERIVED VARIABLES"
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
    100
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
    100
)


# ============================================================
# SUMMARY
# ============================================================

print()

print(
    "Valid temporal records:",
    len(temporal)
)


print(
    "Unique patients:",
    temporal[
        "patient_id"
    ].nunique()
)


print(
    "Unique lesion trajectories:",
    temporal[
        [
            "patient_id",
            "lesion_alias"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


# ============================================================
# PATIENT-LEVEL HOLDOUT
# ============================================================

header(
    "CREATING PATIENT-LEVEL HOLDOUT"
)


patients = sorted(
    temporal[
        "patient_id"
    ].unique()
)


n_test = max(
    1,
    int(
        round(
            len(patients)
            *
            0.25
        )
    )
)


test_patients = set(
    patients[
        -n_test:
    ]
)


train_patients = set(
    patients[
        :-n_test
    ]
)


train = temporal[
    temporal[
        "patient_id"
    ].isin(
        train_patients
    )
].copy()


test = temporal[
    temporal[
        "patient_id"
    ].isin(
        test_patients
    )
].copy()


print(
    "Training patients:",
    len(train_patients)
)


print(
    "Testing patients:",
    len(test_patients)
)


print(
    "Training transitions:",
    len(train)
)


print(
    "Testing transitions:",
    len(test)
)


print(
    "Test patient IDs:",
    sorted(
        test_patients
    )
)


if len(train) == 0 or len(test) == 0:

    print(
        "ERROR: Invalid train/test split."
    )

    sys.exit(1)


# ============================================================
# MODEL A - NO CHANGE
# ============================================================

header(
    "MODEL A - NO CHANGE"
)


test[
    "no_change_prediction"
] = test[
    "previous_diameter_mm"
]


# ============================================================
# MODEL B - DIAMETER ONLY
# ============================================================

header(
    "MODEL B - DIAMETER ONLY"
)


model_diameter = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


model_diameter.fit(
    train[
        [
            "previous_diameter_mm"
        ]
    ],
    train[
        "current_diameter_mm"
    ]
)


test[
    "diameter_only_prediction"
] = model_diameter.predict(
    test[
        [
            "previous_diameter_mm"
        ]
    ]
)


# ============================================================
# MODEL C - DIAMETER + TIME
# ============================================================

header(
    "MODEL C - DIAMETER + TIME"
)


model_dt = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


model_dt.fit(
    train[
        [
            "previous_diameter_mm",
            "interval_days"
        ]
    ],
    train[
        "current_diameter_mm"
    ]
)


test[
    "diameter_time_prediction"
] = model_dt.predict(
    test[
        [
            "previous_diameter_mm",
            "interval_days"
        ]
    ]
)


# ============================================================
# MODEL D - DIAMETER + VOLUME + TIME
# ============================================================

header(
    "MODEL D - DIAMETER + VOLUME + TIME"
)


model_dvt = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


model_dvt.fit(
    train[
        [
            "previous_diameter_mm",
            "previous_volume_cm3",
            "interval_days"
        ]
    ],
    train[
        "current_diameter_mm"
    ]
)


test[
    "diameter_volume_time_prediction"
] = model_dvt.predict(
    test[
        [
            "previous_diameter_mm",
            "previous_volume_cm3",
            "interval_days"
        ]
    ]
)


# ============================================================
# MODEL E - FULL GEOMETRIC MODEL
# ============================================================

header(
    "MODEL E - FULL GEOMETRIC MODEL"
)


train[
    "diameter_volume_interaction"
] = (
    train[
        "previous_diameter_mm"
    ]
    *
    train[
        "previous_volume_cm3"
    ]
)


test[
    "diameter_volume_interaction"
] = (
    test[
        "previous_diameter_mm"
    ]
    *
    test[
        "previous_volume_cm3"
    ]
)


full_features = [

    "previous_diameter_mm",

    "previous_volume_cm3",

    "interval_days",

    "diameter_volume_interaction"
]


model_full = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


model_full.fit(
    train[
        full_features
    ],
    train[
        "current_diameter_mm"
    ]
)


test[
    "full_feature_prediction"
] = model_full.predict(
    test[
        full_features
    ]
)


# ============================================================
# PERFORMANCE
# ============================================================

header(
    "MODEL PERFORMANCE"
)


prediction_columns = {

    "no_change":
        "no_change_prediction",

    "diameter_only":
        "diameter_only_prediction",

    "diameter_plus_time":
        "diameter_time_prediction",

    "diameter_volume_time":
        "diameter_volume_time_prediction",

    "full_feature_model":
        "full_feature_prediction"
}


performance_records = []


observed = test[
    "current_diameter_mm"
]


for model_name, column in prediction_columns.items():

    prediction = test[
        column
    ]


    mae = mean_absolute_error(
        observed,
        prediction
    )


    rmse = np.sqrt(
        mean_squared_error(
            observed,
            prediction
        )
    )


    ss_res = np.sum(
        (
            observed
            -
            prediction
        ) ** 2
    )


    ss_tot = np.sum(
        (
            observed
            -
            observed.mean()
        ) ** 2
    )


    if ss_tot > 0:

        r2 = (
            1
            -
            ss_res / ss_tot
        )

    else:

        r2 = np.nan


    performance_records.append({

        "model":
            model_name,

        "observations":
            len(test),

        "MAE_mm":
            mae,

        "RMSE_mm":
            rmse,

        "R2":
            r2
    })


performance = pd.DataFrame(
    performance_records
)


print(
    performance.to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

header(
    "BEST MODEL"
)


best = performance.loc[
    performance[
        "MAE_mm"
    ].idxmin()
]


print(
    "Best model:",
    best[
        "model"
    ]
)


print(
    "Best MAE:",
    round(
        best[
            "MAE_mm"
        ],
        3
    ),
    "mm"
)


print(
    "Best RMSE:",
    round(
        best[
            "RMSE_mm"
        ],
        3
    ),
    "mm"
)


# ============================================================
# VOLUME CONTRIBUTION
# ============================================================

header(
    "VOLUME CONTRIBUTION"
)


dt_mae = performance[
    performance[
        "model"
    ]
    ==
    "diameter_plus_time"
][
    "MAE_mm"
].iloc[0]


dvt_mae = performance[
    performance[
        "model"
    ]
    ==
    "diameter_volume_time"
][
    "MAE_mm"
].iloc[0]


if dt_mae > 0:

    volume_improvement = (
        (
            dt_mae
            -
            dvt_mae
        )
        /
        dt_mae
    ) * 100

else:

    volume_improvement = np.nan


print(
    "Diameter + time MAE:",
    round(
        dt_mae,
        3
    ),
    "mm"
)


print(
    "Diameter + volume + time MAE:",
    round(
        dvt_mae,
        3
    ),
    "mm"
)


print(
    "Additional improvement from volume:",
    round(
        volume_improvement,
        2
    ),
    "%"
)


# ============================================================
# PATIENT LEVEL
# ============================================================

header(
    "PATIENT-LEVEL PERFORMANCE"
)


patient_results = []


for patient_id, group in test.groupby(
    "patient_id"
):

    result = {

        "patient_id":
            patient_id,

        "observations":
            len(group)
    }


    for model_name, column in prediction_columns.items():

        result[
            model_name
            +
            "_MAE_mm"
        ] = mean_absolute_error(
            group[
                "current_diameter_mm"
            ],
            group[
                column
            ]
        )


    patient_results.append(
        result
    )


patient_performance = pd.DataFrame(
    patient_results
)


print(
    patient_performance.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

header(
    "SAVING STEP 13 RESULTS"
)


prediction_file = os.path.join(
    DATA_DIR,
    "13_hbgd_feature_predictions.csv"
)


performance_file = os.path.join(
    DATA_DIR,
    "13_hbgd_feature_performance.csv"
)


patient_file = os.path.join(
    DATA_DIR,
    "13_hbgd_feature_patient_performance.csv"
)


test.to_csv(
    prediction_file,
    index=False
)


performance.to_csv(
    performance_file,
    index=False
)


patient_performance.to_csv(
    patient_file,
    index=False
)


print(
    "Saved:",
    prediction_file
)


print(
    "Saved:",
    performance_file
)


print(
    "Saved:",
    patient_file
)


# ============================================================
# INTERPRETATION
# ============================================================

header(
    "STEP 13 - INTERPRETATION"
)


print("""
This experiment evaluates whether 3D lesion volume provides
predictive information beyond RECIST diameter and time.

The patient-level holdout prevents observations from the
same patient appearing in both training and testing data.

The models are:

    A. No-change baseline

    B. Diameter only

    C. Diameter + time

    D. Diameter + volume + time

    E. Full geometric model

The result is a feasibility assessment.

It does NOT establish clinical validity or biological causality.
""")


# ============================================================
# FINAL FEASIBILITY
# ============================================================

header(
    "STEP 13 - FINAL FEASIBILITY"
)


if dvt_mae < dt_mae:

    print(
        "[YES] Volume improves over diameter + time"
    )

else:

    print(
        "[NO] Volume does not improve over diameter + time"
    )


no_change_mae = performance[
    performance[
        "model"
    ]
    ==
    "no_change"
][
    "MAE_mm"
].iloc[0]


full_mae = performance[
    performance[
        "model"
    ]
    ==
    "full_feature_model"
][
    "MAE_mm"
].iloc[0]


if full_mae < no_change_mae:

    print(
        "[YES] Full feature model beats no-change baseline"
    )

else:

    print(
        "[NO] Full feature model does not beat no-change baseline"
    )


# ============================================================
# COMPLETE
# ============================================================

header(
    "STEP 13 COMPLETE"
)


print("""
Run:

    python 13_test_hbgd_feature_value.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A NEURAL NETWORK YET.
""")


print(
    "=" * 95
)

print(
    "END OF STEP 13"
)

print(
    "=" * 95
)