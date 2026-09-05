# 10_fit_growth_models.py

import os
import sys
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


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


warnings.filterwarnings(
    "ignore"
)


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
    "STEP 10 - LOADING LONGITUDINAL RECIST DATA"
)


if not os.path.exists(
    PAIRWISE_FILE
):

    print(
        "ERROR: Required file not found:"
    )

    print(
        PAIRWISE_FILE
    )

    sys.exit(1)


df = pd.read_csv(
    PAIRWISE_FILE
)


print(
    "Pairwise records loaded:",
    len(df)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [

    "patient_id",

    "lesion_alias",

    "previous_date",

    "current_date",

    "interval_days",

    "previous_diameter_mm",

    "current_diameter_mm"
]


missing = [

    column
    for column in required_columns
    if column not in df.columns

]


if len(missing) > 0:

    print(
        "ERROR: Missing columns:"
    )

    for column in missing:

        print(
            "   ",
            column
        )

    sys.exit(1)


print(
    "All required columns are present."
)


# ============================================================
# NORMALIZE
# ============================================================

header(
    "NORMALIZING LONGITUDINAL MEASUREMENTS"
)


numeric_columns = [

    "patient_id",

    "interval_days",

    "previous_diameter_mm",

    "current_diameter_mm"

]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


df = df[
    df[
        "interval_days"
    ] > 0
].copy()


df = df[
    df[
        "previous_diameter_mm"
    ] > 0
].copy()


df = df[
    df[
        "current_diameter_mm"
    ] > 0
].copy()


df = df.reset_index(
    drop=True
)


print(
    "Valid longitudinal transitions:",
    len(df)
)


# ============================================================
# IMPORTANT MODELING DECISION
# ============================================================

print()
print(
    "Each record represents:"
)

print(
    "D(t0) -> D(t1) over Δt"
)

print()
print(
    "The models are evaluated on the observed"
)

print(
    "RECIST diameter change."
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

header(
    "DEFINING GROWTH MODELS"
)


# ------------------------------------------------------------
# MODEL A
# LINEAR
# ------------------------------------------------------------

def linear_model(
    t,
    d0,
    rate
):

    return (
        d0
        +
        rate * t
    )


# ------------------------------------------------------------
# MODEL B
# EXPONENTIAL
# ------------------------------------------------------------

def exponential_model(
    t,
    d0,
    rate
):

    return (
        d0
        *
        np.exp(
            rate * t
        )
    )


# ------------------------------------------------------------
# MODEL C
# LOGISTIC
# ------------------------------------------------------------

def logistic_model(
    t,
    d0,
    rate,
    carrying_capacity
):

    denominator = (
        1.0
        +
        (
            (
                carrying_capacity
                -
                d0
            )
            /
            d0
        )
        *
        np.exp(
            -rate * t
        )
    )

    return (
        carrying_capacity
        /
        denominator
    )


# ============================================================
# FIT EACH TRANSITION
# ============================================================

header(
    "FITTING GROWTH MODELS"
)


results = []


for index, row in df.iterrows():

    d0 = float(
        row[
            "previous_diameter_mm"
        ]
    )

    observed = float(
        row[
            "current_diameter_mm"
        ]
    )

    delta_t = float(
        row[
            "interval_days"
        ]
    )


    # ========================================================
    # LINEAR MODEL
    # ========================================================

    linear_rate = (
        observed - d0
    ) / delta_t


    linear_prediction = (
        linear_model(
            delta_t,
            d0,
            linear_rate
        )
    )


    # ========================================================
    # EXPONENTIAL MODEL
    # ========================================================

    exponential_rate = (
        np.log(
            observed / d0
        )
        /
        delta_t
    )


    exponential_prediction = (
        exponential_model(
            delta_t,
            d0,
            exponential_rate
        )
    )


    # ========================================================
    # LOGISTIC MODEL
    # ========================================================
    #
    # IMPORTANT:
    #
    # A single transition does not contain enough information
    # to independently estimate all three logistic parameters.
    #
    # Therefore we use a bounded carrying capacity estimate
    # based on the observed diameter.
    #
    # This is a diagnostic approximation only.
    #
    # We will NOT claim this is a fitted biological carrying
    # capacity.
    # ========================================================

    carrying_capacity = max(
        d0,
        observed
    ) * 2.0


    try:

        logistic_prediction = (
            logistic_model(
                delta_t,
                d0,
                max(
                    exponential_rate,
                    -0.1
                ),
                carrying_capacity
            )
        )

    except Exception:

        logistic_prediction = np.nan


    # ========================================================
    # STORE
    # ========================================================

    results.append({

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

        "interval_days":
            delta_t,

        "observed_diameter_mm":
            observed,

        "initial_diameter_mm":
            d0,

        "linear_rate_mm_per_day":
            linear_rate,

        "linear_prediction_mm":
            linear_prediction,

        "exponential_rate_per_day":
            exponential_rate,

        "exponential_prediction_mm":
            exponential_prediction,

        "logistic_prediction_mm":
            logistic_prediction,

        "carrying_capacity_assumption_mm":
            carrying_capacity
    })


results_df = pd.DataFrame(
    results
)


# ============================================================
# MODEL ERROR
# ============================================================

header(
    "CALCULATING MODEL ERRORS"
)


results_df[
    "linear_error_mm"
] = (
    results_df[
        "linear_prediction_mm"
    ]
    -
    results_df[
        "observed_diameter_mm"
    ]
)


results_df[
    "exponential_error_mm"
] = (
    results_df[
        "exponential_prediction_mm"
    ]
    -
    results_df[
        "observed_diameter_mm"
    ]
)


results_df[
    "logistic_error_mm"
] = (
    results_df[
        "logistic_prediction_mm"
    ]
    -
    results_df[
        "observed_diameter_mm"
    ]
)


# ============================================================
# NOTE ABOUT THIS ERROR
# ============================================================

print(
    "Because each transition's rate is calculated directly"
)

print(
    "from its two endpoints, linear and exponential models"
)

print(
    "will reproduce the endpoint almost exactly."
)

print()

print(
    "Therefore this is NOT a meaningful generalization test."
)

print(
    "The more important analysis below evaluates"
)

print(
    "patient-level consistency of the inferred growth rate."
)


# ============================================================
# GROWTH RATE STATISTICS
# ============================================================

header(
    "GROWTH RATE DISTRIBUTION"
)


linear_rates = (
    results_df[
        "linear_rate_mm_per_day"
    ]
)

exponential_rates = (
    results_df[
        "exponential_rate_per_day"
    ]
)


print(
    "Linear rate:"
)

print(
    "  Mean:",
    round(
        linear_rates.mean(),
        6
    ),
    "mm/day"
)

print(
    "  Median:",
    round(
        linear_rates.median(),
        6
    ),
    "mm/day"
)

print(
    "  Minimum:",
    round(
        linear_rates.min(),
        6
    ),
    "mm/day"
)

print(
    "  Maximum:",
    round(
        linear_rates.max(),
        6
    ),
    "mm/day"
)


print()

print(
    "Logarithmic exponential rate:"
)

print(
    "  Mean:",
    round(
        exponential_rates.mean(),
        6
    ),
    "per day"
)

print(
    "  Median:",
    round(
        exponential_rates.median(),
        6
    ),
    "per day"
)

print(
    "  Minimum:",
    round(
        exponential_rates.min(),
        6
    ),
    "per day"
)

print(
    "  Maximum:",
    round(
        exponential_rates.max(),
        6
    ),
    "per day"
)


# ============================================================
# GROWTH / SHRINKAGE
# ============================================================

header(
    "GROWTH VS SHRINKAGE"
)


growth = int(
    (
        results_df[
            "observed_diameter_mm"
        ]
        >
        results_df[
            "initial_diameter_mm"
        ]
    ).sum()
)


shrinkage = int(
    (
        results_df[
            "observed_diameter_mm"
        ]
        <
        results_df[
            "initial_diameter_mm"
        ]
    ).sum()
)


stable = int(
    (
        results_df[
            "observed_diameter_mm"
        ]
        ==
        results_df[
            "initial_diameter_mm"
        ]
    ).sum()
)


total = len(
    results_df
)


print(
    "Growth:",
    growth,
    "/",
    total,
    "(",
    round(
        growth / total * 100,
        2
    ),
    "%)"
)


print(
    "Shrinkage:",
    shrinkage,
    "/",
    total,
    "(",
    round(
        shrinkage / total * 100,
        2
    ),
    "%)"
)


print(
    "Stable:",
    stable,
    "/",
    total,
    "(",
    round(
        stable / total * 100,
        2
    ),
    "%)"
)


# ============================================================
# PATIENT-LEVEL GROWTH CONSISTENCY
# ============================================================

header(
    "PATIENT-LEVEL GROWTH CONSISTENCY"
)


patient_results = []


for patient_id, group in (
    results_df.groupby(
        "patient_id"
    )
):

    rates = group[
        "exponential_rate_per_day"
    ]


    positive = int(
        (
            rates > 0
        ).sum()
    )

    negative = int(
        (
            rates < 0
        ).sum()
    )

    zero = int(
        (
            rates == 0
        ).sum()
    )


    patient_results.append({

        "patient_id":
            patient_id,

        "transitions":
            len(group),

        "positive_growth":
            positive,

        "negative_growth":
            negative,

        "stable":
            zero,

        "mean_log_growth_rate":
            rates.mean(),

        "median_log_growth_rate":
            rates.median(),

        "growth_rate_std":
            rates.std()
    })


patient_df = pd.DataFrame(
    patient_results
)


patient_df = patient_df.sort_values(
    "patient_id"
)


print(
    patient_df.to_string(
        index=False
    )
)


# ============================================================
# RATE VARIABILITY
# ============================================================

header(
    "GROWTH RATE VARIABILITY"
)


if len(patient_df) > 0:

    print(
        "Mean patient median growth rate:",
        round(
            patient_df[
                "median_log_growth_rate"
            ].mean(),
            6
        )
    )

    print(
        "Median patient median growth rate:",
        round(
            patient_df[
                "median_log_growth_rate"
            ].median(),
            6
        )
    )

    print(
        "Patient-to-patient standard deviation:",
        round(
            patient_df[
                "median_log_growth_rate"
            ].std(),
            6
        )
    )


# ============================================================
# INTRA-PATIENT CONSISTENCY
# ============================================================

header(
    "INTRA-PATIENT GROWTH CONSISTENCY"
)


consistent_growth = 0

mixed_behavior = 0

consistent_shrinkage = 0


for _, row in patient_df.iterrows():

    positive = row[
        "positive_growth"
    ]

    negative = row[
        "negative_growth"
    ]

    if (
        positive > 0
        and
        negative == 0
    ):

        consistent_growth += 1

    elif (
        negative > 0
        and
        positive == 0
    ):

        consistent_shrinkage += 1

    elif (
        positive > 0
        and
        negative > 0
    ):

        mixed_behavior += 1


print(
    "Patients showing only growth:",
    consistent_growth
)

print(
    "Patients showing only shrinkage:",
    consistent_shrinkage
)

print(
    "Patients showing mixed behavior:",
    mixed_behavior
)


# ============================================================
# HOLD-OUT STYLE TEST
# ============================================================
#
# IMPORTANT:
#
# We now test whether the FIRST observed transition can
# estimate a growth rate that predicts a LATER transition
# for the SAME lesion.
#
# This is much more meaningful than fitting each transition
# independently.
# ============================================================

header(
    "LONGITUDINAL RATE GENERALIZATION TEST"
)


prediction_records = []


group_columns = [
    "patient_id",
    "lesion_alias"
]


for (
    patient_id,
    alias
), group in (
    df.groupby(
        group_columns
    )
):

    group = group.sort_values(
        "previous_date"
    ).copy()


    if len(group) < 2:

        continue


    # --------------------------------------------------------
    # FIRST TRANSITION
    # --------------------------------------------------------

    first = group.iloc[0]

    training_d0 = float(
        first[
            "previous_diameter_mm"
        ]
    )

    training_d1 = float(
        first[
            "current_diameter_mm"
        ]
    )

    training_dt = float(
        first[
            "interval_days"
        ]
    )


    if (
        training_d0 <= 0
        or
        training_d1 <= 0
        or
        training_dt <= 0
    ):

        continue


    # --------------------------------------------------------
    # INFER RATE FROM FIRST TRANSITION
    # --------------------------------------------------------

    inferred_rate = (
        np.log(
            training_d1 /
            training_d0
        )
        /
        training_dt
    )


    # --------------------------------------------------------
    # TEST ON LATER TRANSITIONS
    # --------------------------------------------------------

    for j in range(
        1,
        len(group)
    ):

        test = group.iloc[j]


        test_d0 = float(
            test[
                "previous_diameter_mm"
            ]
        )

        test_dt = float(
            test[
                "interval_days"
            ]
        )

        observed_d1 = float(
            test[
                "current_diameter_mm"
            ]
        )


        predicted_d1 = (
            test_d0
            *
            np.exp(
                inferred_rate
                *
                test_dt
            )
        )


        error = (
            predicted_d1
            -
            observed_d1
        )


        prediction_records.append({

            "patient_id":
                patient_id,

            "lesion_alias":
                alias,

            "training_rate_per_day":
                inferred_rate,

            "test_interval_days":
                test_dt,

            "test_initial_diameter_mm":
                test_d0,

            "observed_diameter_mm":
                observed_d1,

            "predicted_diameter_mm":
                predicted_d1,

            "prediction_error_mm":
                error,

            "absolute_error_mm":
                abs(error)
        })


holdout_df = pd.DataFrame(
    prediction_records
)


print(
    "Independent later transitions tested:",
    len(holdout_df)
)


if len(holdout_df) > 0:

    mae = mean_absolute_error(
        holdout_df[
            "observed_diameter_mm"
        ],
        holdout_df[
            "predicted_diameter_mm"
        ]
    )

    rmse = np.sqrt(
        mean_squared_error(
            holdout_df[
                "observed_diameter_mm"
            ],
            holdout_df[
                "predicted_diameter_mm"
            ]
        )
    )


    observed_values = (
        holdout_df[
            "observed_diameter_mm"
        ]
    )

    predicted_values = (
        holdout_df[
            "predicted_diameter_mm"
        ]
    )


    if (
        len(
            observed_values
        ) >= 2
    ):

        r2 = r2_score(
            observed_values,
            predicted_values
        )

    else:

        r2 = np.nan


    print(
        "Holdout MAE:",
        round(
            mae,
            3
        ),
        "mm"
    )

    print(
        "Holdout RMSE:",
        round(
            rmse,
            3
        ),
        "mm"
    )

    print(
        "Holdout R2:",
        round(
            r2,
            4
        )
        if not np.isnan(r2)
        else "N/A"
    )


# ============================================================
# PREDICTION ERROR BY LESION
# ============================================================

header(
    "HOLDOUT ERROR BY LESION"
)


if len(holdout_df) > 0:

    lesion_error = (
        holdout_df
        .groupby(
            [
                "patient_id",
                "lesion_alias"
            ]
        )
        .agg(
            transitions=(
                "absolute_error_mm",
                "count"
            ),
            mean_absolute_error_mm=(
                "absolute_error_mm",
                "mean"
            )
        )
        .reset_index()
    )


    print(
        lesion_error.to_string(
            index=False
        )
    )


# ============================================================
# MODEL INTERPRETATION
# ============================================================

header(
    "STEP 10 - INTERPRETATION"
)


print("""
The per-transition linear and exponential equations can
reproduce an individual endpoint because their rate is
calculated directly from that endpoint.

Therefore, those per-transition fits are descriptive rather
than predictive.

The important result is the longitudinal generalization test:

    first transition
          ↓
    infer growth rate
          ↓
    predict later transition

This tests whether a lesion's observed growth tendency is
stable enough to support a dynamical growth model.

A good holdout result would support continued development
of the HBGD mathematical framework.

A poor holdout result would mean that a simple constant-growth
model is insufficient and that treatment response, nonlinear
dynamics, or additional biological variables need to be
introduced.

This is still a feasibility experiment.

It is NOT clinical validation.
""")


# ============================================================
# FINAL FEASIBILITY
# ============================================================

header(
    "STEP 10 - FINAL FEASIBILITY"
)


print(
    "["
    +
    (
        "YES"
        if len(df) > 0
        else "NO"
    )
    +
    "] Longitudinal RECIST transitions available"
)


print(
    "["
    +
    (
        "YES"
        if len(patient_df) > 0
        else "NO"
    )
    +
    "] Patient-level growth rates calculable"
)


print(
    "["
    +
    (
        "YES"
        if len(holdout_df) > 0
        else "NO"
    )
    +
    "] Later-transition generalization test available"
)


if len(holdout_df) > 0:

    print(
        "Holdout observations:",
        len(holdout_df)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

header(
    "SAVING STEP 10 RESULTS"
)


transition_file = os.path.join(
    DATA_DIR,
    "10_growth_model_transitions.csv"
)

patient_file = os.path.join(
    DATA_DIR,
    "10_patient_growth_rates.csv"
)

holdout_file = os.path.join(
    DATA_DIR,
    "10_growth_model_holdout.csv"
)


results_df.to_csv(
    transition_file,
    index=False
)

patient_df.to_csv(
    patient_file,
    index=False
)

holdout_df.to_csv(
    holdout_file,
    index=False
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
    holdout_file
)


# ============================================================
# COMPLETE
# ============================================================

header(
    "STEP 10 COMPLETE"
)

print("""
Run:

    python 10_fit_growth_models.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A NEURAL NETWORK YET.
""")

print(
    "=" * 95
)

print(
    "END OF STEP 10"
)

print(
    "=" * 95
)