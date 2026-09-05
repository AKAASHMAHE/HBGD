# 12_test_dynamic_growth_models.py

import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from scipy.optimize import least_squares


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

PAIRWISE_FILE = os.path.join(
    DATA_DIR,
    "8_recist_pairwise_changes.csv"
)

FEATURE_FILE = os.path.join(
    DATA_DIR,
    "9_hbgd_features.csv"
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
# LOAD DATA
# ============================================================

header(
    "STEP 12 - LOADING LONGITUDINAL DATA"
)


if not os.path.exists(PAIRWISE_FILE):

    print("ERROR: File not found:")
    print(PAIRWISE_FILE)
    sys.exit(1)


if not os.path.exists(FEATURE_FILE):

    print("ERROR: File not found:")
    print(FEATURE_FILE)
    sys.exit(1)


pairwise = pd.read_csv(
    PAIRWISE_FILE
)

features = pd.read_csv(
    FEATURE_FILE
)


print(
    "Pairwise records:",
    len(pairwise)
)

print(
    "Feature records:",
    len(features)
)


# ============================================================
# NORMALIZE
# ============================================================

header(
    "NORMALIZING DATA"
)


numeric_columns = [

    "patient_id",
    "interval_days",
    "previous_diameter_mm",
    "current_diameter_mm",
    "diameter_change_mm"
]


for column in numeric_columns:

    if column in pairwise.columns:

        pairwise[column] = pd.to_numeric(
            pairwise[column],
            errors="coerce"
        )


pairwise = pairwise[
    pairwise["interval_days"] > 0
].copy()


pairwise = pairwise[
    pairwise["previous_diameter_mm"] > 0
].copy()


pairwise = pairwise[
    pairwise["current_diameter_mm"] > 0
].copy()


pairwise = pairwise.reset_index(
    drop=True
)


print(
    "Valid transitions:",
    len(pairwise)
)


# ============================================================
# BUILD FULL LESION TRAJECTORIES
# ============================================================

header(
    "BUILDING FULL LESION TRAJECTORIES"
)


required_feature_columns = [

    "patient_id",
    "study_date",
    "lesion_label_alias",
    "diameter_mm",
    "volume_cm3"
]


missing = [

    c
    for c in required_feature_columns
    if c not in features.columns
]


if len(missing) > 0:

    print(
        "ERROR: Missing feature columns:"
    )

    for c in missing:
        print(
            "   ",
            c
        )

    sys.exit(1)


features["patient_id"] = pd.to_numeric(
    features["patient_id"],
    errors="coerce"
)


features["study_date"] = pd.to_datetime(
    features["study_date"],
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


features = features.dropna(
    subset=[
        "patient_id",
        "study_date",
        "diameter_mm"
    ]
).copy()


features = features.sort_values(
    [
        "patient_id",
        "lesion_label_alias",
        "study_date"
    ]
)


trajectory_groups = []

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

    if len(group) >= 2:

        trajectory_groups.append(
            group
        )


print(
    "Longitudinal lesion trajectories:",
    len(trajectory_groups)
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

header(
    "DEFINING DYNAMIC MODELS"
)


# ------------------------------------------------------------
# MODEL 0
# NO CHANGE
# ------------------------------------------------------------

def predict_no_change(
    d0,
    dt
):

    return d0


# ------------------------------------------------------------
# MODEL 1
# CONSTANT EXPONENTIAL
# ------------------------------------------------------------

def predict_exponential(
    d0,
    rate,
    dt
):

    value = (
        d0
        *
        np.exp(
            rate * dt
        )
    )

    return max(
        value,
        0.001
    )


# ------------------------------------------------------------
# MODEL 2
# LINEAR
# ------------------------------------------------------------

def predict_linear(
    d0,
    rate,
    dt
):

    value = (
        d0
        +
        rate * dt
    )

    return max(
        value,
        0.001
    )


# ------------------------------------------------------------
# MODEL 3
# PIECEWISE RATE
# ------------------------------------------------------------
#
# Estimate rate from earlier observations and allow the
# rate to update when new observations arrive.
#
# This is the model we actually want to test against
# constant exponential behavior.
# ============================================================


# ============================================================
# CREATE LESION-LEVEL EVALUATION
# ============================================================

header(
    "CREATING LESION-LEVEL TRAIN / TEST SPLITS"
)


# We use:
#
# earlier transition(s) -> estimate parameters
#
# later transition(s) -> evaluate
#
# No transition from a lesion is allowed to train the model
# used to predict that SAME transition.
#
# This is still a small feasibility experiment, not a clinical
# validation protocol.


evaluation_records = []


for group in trajectory_groups:

    group = group.sort_values(
        "study_date"
    ).copy()


    if len(group) < 3:

        continue


    patient_id = group[
        "patient_id"
    ].iloc[0]

    alias = group[
        "lesion_label_alias"
    ].iloc[0]


    dates = group[
        "study_date"
    ].tolist()

    diameters = group[
        "diameter_mm"
    ].astype(float).tolist()


    # --------------------------------------------------------
    # USE FIRST TWO OBSERVATIONS AS INITIAL DYNAMIC ESTIMATE
    # --------------------------------------------------------

    initial_d0 = diameters[0]
    initial_d1 = diameters[1]


    dt_initial = (
        dates[1]
        -
        dates[0]
    ).days


    if dt_initial <= 0:
        continue


    # Linear rate
    linear_rate = (
        initial_d1
        -
        initial_d0
    ) / dt_initial


    # Exponential rate
    exponential_rate = (
        np.log(
            initial_d1
            /
            initial_d0
        )
        /
        dt_initial
    )


    # --------------------------------------------------------
    # PREDICT EVERY LATER OBSERVATION
    # --------------------------------------------------------

    for j in range(
        2,
        len(group)
    ):

        dt_test = (
            dates[j]
            -
            dates[j - 1]
        ).days


        if dt_test <= 0:
            continue


        observed = diameters[j]

        previous = diameters[j - 1]


        # ====================================================
        # MODEL A - NO CHANGE
        # ====================================================

        no_change_prediction = (
            predict_no_change(
                previous,
                dt_test
            )
        )


        # ====================================================
        # MODEL B - CONSTANT EXPONENTIAL
        # ====================================================

        exponential_prediction = (
            predict_exponential(
                previous,
                exponential_rate,
                dt_test
            )
        )


        # ====================================================
        # MODEL C - CONSTANT LINEAR
        # ====================================================

        linear_prediction = (
            predict_linear(
                previous,
                linear_rate,
                dt_test
            )
        )


        # ====================================================
        # MODEL D - ADAPTIVE / PIECEWISE RATE
        # ====================================================
        #
        # Estimate a new rate using ALL observations available
        # before the target observation.
        #
        # The test observation itself is never used.
        #
        # This lets the model adapt when the tumor changes
        # behavior.
        # ====================================================

        historical_dates = dates[
            :j
        ]

        historical_diameters = diameters[
            :j
        ]


        historical_rates = []


        for k in range(
            1,
            len(
                historical_diameters
            )
        ):

            dt_hist = (
                historical_dates[k]
                -
                historical_dates[k - 1]
            ).days


            if (
                dt_hist <= 0
                or
                historical_diameters[
                    k - 1
                ] <= 0
                or
                historical_diameters[
                    k
                ] <= 0
            ):

                continue


            rate = (
                np.log(
                    historical_diameters[k]
                    /
                    historical_diameters[k - 1]
                )
                /
                dt_hist
            )


            historical_rates.append(
                rate
            )


        if len(
            historical_rates
        ) > 0:

            adaptive_rate = np.mean(
                historical_rates
            )

        else:

            adaptive_rate = 0.0


        adaptive_prediction = (
            predict_exponential(
                previous,
                adaptive_rate,
                dt_test
            )
        )


        evaluation_records.append({

            "patient_id":
                patient_id,

            "lesion_alias":
                alias,

            "target_date":
                dates[j],

            "previous_diameter_mm":
                previous,

            "observed_diameter_mm":
                observed,

            "interval_days":
                dt_test,

            "no_change_prediction":
                no_change_prediction,

            "linear_prediction":
                linear_prediction,

            "constant_exponential_prediction":
                exponential_prediction,

            "adaptive_exponential_prediction":
                adaptive_prediction,

            "initial_exponential_rate":
                exponential_rate,

            "adaptive_exponential_rate":
                adaptive_rate
        })


evaluation = pd.DataFrame(
    evaluation_records
)


print(
    "Independent later observations:",
    len(evaluation)
)


# ============================================================
# CALCULATE ERRORS
# ============================================================

header(
    "CALCULATING MODEL ERRORS"
)


if len(evaluation) == 0:

    print(
        "Not enough trajectories for evaluation."
    )

    sys.exit(0)


models = {

    "no_change":
        "no_change_prediction",

    "linear":
        "linear_prediction",

    "constant_exponential":
        "constant_exponential_prediction",

    "adaptive_exponential":
        "adaptive_exponential_prediction"
}


for model_name, prediction_column in models.items():

    evaluation[
        model_name + "_error_mm"
    ] = (
        evaluation[
            prediction_column
        ]
        -
        evaluation[
            "observed_diameter_mm"
        ]
    )

    evaluation[
        model_name + "_absolute_error_mm"
    ] = (
        evaluation[
            model_name + "_error_mm"
        ].abs()
    )

    evaluation[
        model_name + "_squared_error"
    ] = (
        evaluation[
            model_name + "_error_mm"
        ]
        ** 2
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

header(
    "MODEL PERFORMANCE"
)


performance_records = []


observed = evaluation[
    "observed_diameter_mm"
]


for model_name, prediction_column in models.items():

    prediction = evaluation[
        prediction_column
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


    try:

        r2 = r2_score(
            observed,
            prediction
        )

    except Exception:

        r2 = np.nan


    performance_records.append({

        "model":
            model_name,

        "observations":
            len(evaluation),

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


best_model_row = performance.loc[
    performance[
        "MAE_mm"
    ].idxmin()
]


print(
    "Best model by MAE:",
    best_model_row[
        "model"
    ]
)


print(
    "Best MAE:",
    round(
        best_model_row[
            "MAE_mm"
        ],
        3
    ),
    "mm"
)


print(
    "Best RMSE:",
    round(
        best_model_row[
            "RMSE_mm"
        ],
        3
    ),
    "mm"
)


# ============================================================
# PAIRWISE MODEL COMPARISON
# ============================================================

header(
    "MODEL ERROR COMPARISON"
)


comparison_columns = [

    "no_change_absolute_error_mm",

    "linear_absolute_error_mm",

    "constant_exponential_absolute_error_mm",

    "adaptive_exponential_absolute_error_mm"
]


comparison = evaluation[
    comparison_columns
].copy()


comparison.columns = [

    "No-change",

    "Linear",

    "Constant-exponential",

    "Adaptive-exponential"
]


print(
    comparison.mean().round(3).to_string()
)


# ============================================================
# ADAPTIVE MODEL IMPROVEMENT
# ============================================================

header(
    "ADAPTIVE DYNAMICS IMPROVEMENT"
)


baseline_mae = (
    performance[
        performance[
            "model"
        ]
        ==
        "no_change"
    ][
        "MAE_mm"
    ].iloc[0]
)


constant_mae = (
    performance[
        performance[
            "model"
        ]
        ==
        "constant_exponential"
    ][
        "MAE_mm"
    ].iloc[0]
)


adaptive_mae = (
    performance[
        performance[
            "model"
        ]
        ==
        "adaptive_exponential"
    ][
        "MAE_mm"
    ].iloc[0]
)


print(
    "No-change MAE:",
    round(
        baseline_mae,
        3
    ),
    "mm"
)


print(
    "Constant exponential MAE:",
    round(
        constant_mae,
        3
    ),
    "mm"
)


print(
    "Adaptive exponential MAE:",
    round(
        adaptive_mae,
        3
    ),
    "mm"
)


if baseline_mae > 0:

    adaptive_vs_baseline = (
        (
            baseline_mae
            -
            adaptive_mae
        )
        /
        baseline_mae
    ) * 100.0

else:

    adaptive_vs_baseline = np.nan


if constant_mae > 0:

    adaptive_vs_constant = (
        (
            constant_mae
            -
            adaptive_mae
        )
        /
        constant_mae
    ) * 100.0

else:

    adaptive_vs_constant = np.nan


print(
    "Adaptive vs no-change improvement:",
    round(
        adaptive_vs_baseline,
        2
    ),
    "%"
)


print(
    "Adaptive vs constant-exponential improvement:",
    round(
        adaptive_vs_constant,
        2
    ),
    "%"
)


# ============================================================
# ERROR BY PATIENT
# ============================================================

header(
    "MODEL PERFORMANCE BY PATIENT"
)


patient_model_records = []


for patient_id, group in evaluation.groupby(
    "patient_id"
):

    record = {

        "patient_id":
            patient_id,

        "observations":
            len(group)
    }


    for model_name, prediction_column in models.items():

        error = (
            group[
                prediction_column
            ]
            -
            group[
                "observed_diameter_mm"
            ]
        ).abs()


        record[
            model_name
            +
            "_MAE_mm"
        ] = error.mean()


    patient_model_records.append(
        record
    )


patient_model_df = pd.DataFrame(
    patient_model_records
)


print(
    patient_model_df.to_string(
        index=False
    )
)


# ============================================================
# DETERMINE WHETHER ADAPTIVE MODEL WINS
# ============================================================

header(
    "ADAPTIVE MODEL WIN COUNT"
)


win_count = 0

loss_count = 0

tie_count = 0


for _, row in evaluation.iterrows():

    errors = {

        "no_change":
            row[
                "no_change_absolute_error_mm"
            ],

        "constant_exponential":
            row[
                "constant_exponential_absolute_error_mm"
            ],

        "adaptive_exponential":
            row[
                "adaptive_exponential_absolute_error_mm"
            ]
    }


    minimum = min(
        errors.values()
    )


    winners = [

        name
        for name, value in errors.items()
        if abs(
            value - minimum
        ) < 1e-12

    ]


    if "adaptive_exponential" in winners:

        if len(winners) == 1:

            win_count += 1

        else:

            tie_count += 1

    else:

        loss_count += 1


print(
    "Adaptive model uniquely best:",
    win_count
)

print(
    "Adaptive model tied:",
    tie_count
)

print(
    "Adaptive model not best:",
    loss_count
)


# ============================================================
# VOLUME INFORMATION
# ============================================================

header(
    "VOLUME INFORMATION CHECK"
)


if "volume_cm3" in features.columns:

    volume_values = pd.to_numeric(
        features[
            "volume_cm3"
        ],
        errors="coerce"
    ).dropna()


    print(
        "Valid volume observations:",
        len(volume_values)
    )

    print(
        "Median volume:",
        round(
            volume_values.median(),
            3
        ),
        "cm3"
    )

    print(
        "Mean volume:",
        round(
            volume_values.mean(),
            3
        ),
        "cm3"
    )


# ============================================================
# IMPORTANT INTERPRETATION
# ============================================================

header(
    "STEP 12 - INTERPRETATION"
)


print("""
This experiment compares four increasingly dynamic
descriptions of tumor trajectory:

    1. No-change baseline
    2. Linear growth
    3. Constant exponential growth
    4. Adaptive exponential growth

The adaptive model is allowed to update its estimated
growth rate when additional historical observations
become available.

This is intended to test whether tumor behavior is better
represented by a changing dynamical state rather than a
single fixed growth rate.

This is still a small feasibility experiment.

It is NOT clinical validation.

It does NOT establish that the adaptive model is biologically
correct.

The important decision is whether allowing the growth rate
to change improves prediction compared with the simple
baselines.
""")


# ============================================================
# FINAL FEASIBILITY
# ============================================================

header(
    "STEP 12 - FINAL FEASIBILITY"
)


if len(evaluation) > 0:

    print(
        "[YES] Independent later observations available"
    )

else:

    print(
        "[NO] Independent later observations available"
    )


if adaptive_mae < baseline_mae:

    print(
        "[YES] Adaptive model beats no-change baseline"
    )

else:

    print(
        "[NO] Adaptive model does not beat no-change baseline"
    )


if adaptive_mae < constant_mae:

    print(
        "[YES] Adaptive model beats constant exponential"
    )

else:

    print(
        "[NO] Adaptive model does not beat constant exponential"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

header(
    "SAVING STEP 12 RESULTS"
)


evaluation_file = os.path.join(
    DATA_DIR,
    "12_dynamic_model_predictions.csv"
)

performance_file = os.path.join(
    DATA_DIR,
    "12_dynamic_model_performance.csv"
)

patient_file = os.path.join(
    DATA_DIR,
    "12_dynamic_model_patient_performance.csv"
)


evaluation.to_csv(
    evaluation_file,
    index=False
)

performance.to_csv(
    performance_file,
    index=False
)

patient_model_df.to_csv(
    patient_file,
    index=False
)


print(
    "Saved:",
    evaluation_file
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
# COMPLETE
# ============================================================

header(
    "STEP 12 COMPLETE"
)

print("""
Run:

    python 12_test_dynamic_growth_models.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A NEURAL NETWORK YET.
""")

print(
    "=" * 95
)

print(
    "END OF STEP 12"
)

print(
    "=" * 95
)