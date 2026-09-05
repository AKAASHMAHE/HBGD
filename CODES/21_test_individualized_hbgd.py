import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ============================================================
# PATHS
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

FEATURE_FILE = os.path.join(
    DATA_DIR,
    "9_hbgd_features.csv"
)

PREDICTION_FILE = os.path.join(
    DATA_DIR,
    "21_individualized_hbgd_predictions.csv"
)

PERFORMANCE_FILE = os.path.join(
    DATA_DIR,
    "21_individualized_hbgd_performance.csv"
)

PATIENT_FILE = os.path.join(
    DATA_DIR,
    "21_individualized_hbgd_patient_performance.csv"
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
# LOAD
# ============================================================

header("STEP 21 - INDIVIDUALIZED HBGD PARAMETER MODEL")

if not os.path.exists(FEATURE_FILE):
    print("ERROR: Missing:")
    print(FEATURE_FILE)
    sys.exit(1)

df = pd.read_csv(FEATURE_FILE)

print("Feature records:", len(df))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

header("CHECKING REQUIRED VARIABLES")

required = [
    "patient_id",
    "lesion_label_alias",
    "diameter_mm",
    "days_from_baseline"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    print("ERROR: Missing columns:")
    for c in missing:
        print("   ", c)
    sys.exit(1)

print("All required variables are present.")


# ============================================================
# OPTIONAL FEATURES
# ============================================================

optional_features = [
    "volume_cm3",
    "equivalent_diameter_mm",
    "equivalent_to_recist_ratio",
    "diameter_normalized",
    "volume_normalized",
    "diameter_percent_from_baseline",
    "volume_percent_from_baseline"
]

available_optional = [
    c
    for c in optional_features
    if c in df.columns
]

print()
print("Available optional features:")

if available_optional:

    for c in available_optional:
        print("   ", c)

else:

    print("   None")


# ============================================================
# NORMALIZATION
# ============================================================

header("NORMALIZING DATA")

numeric_columns = [
    "patient_id",
    "diameter_mm",
    "days_from_baseline"
] + available_optional

for c in numeric_columns:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "patient_id",
        "lesion_label_alias",
        "diameter_mm",
        "days_from_baseline"
    ]
).copy()


df = df[
    (df["diameter_mm"] > 0) &
    (df["days_from_baseline"] >= 0)
].copy()


df = df.sort_values(
    [
        "patient_id",
        "lesion_label_alias",
        "days_from_baseline"
    ]
).reset_index(drop=True)


print("Valid records:", len(df))


# ============================================================
# BUILD TRAJECTORIES
# ============================================================

header("BUILDING LONGITUDINAL TRAJECTORIES")

trajectories = []

for (
    patient_id,
    alias
), group in df.groupby(
    [
        "patient_id",
        "lesion_label_alias"
    ]
):

    group = group.sort_values(
        "days_from_baseline"
    ).reset_index(drop=True)

    if len(group) < 3:
        continue

    trajectories.append({
        "patient_id": patient_id,
        "lesion_alias": alias,
        "data": group
    })


print(
    "Eligible trajectories:",
    len(trajectories)
)


patients = sorted(
    set(
        item["patient_id"]
        for item in trajectories
    )
)


print(
    "Eligible patients:",
    len(patients)
)

print(
    "Patient IDs:",
    patients
)


# ============================================================
# SPLIT
# ============================================================

header("CREATING PATIENT-LEVEL HOLDOUT")

# Same deterministic holdout as Step 20
# so the experiments remain directly comparable.

test_patients = patients[-3:]

train_patients = [
    p
    for p in patients
    if p not in test_patients
]


print(
    "Training patients:",
    train_patients
)

print(
    "Testing patients:",
    test_patients
)


# ============================================================
# INITIAL TRANSITION TABLE
# ============================================================

header("CREATING INITIAL DYNAMICAL FEATURES")

initial_records = []


for item in trajectories:

    group = item["data"]

    t = group[
        "days_from_baseline"
    ].values.astype(float)

    D = group[
        "diameter_mm"
    ].values.astype(float)


    dt = t[1] - t[0]


    if (
        dt <= 0
        or D[0] <= 0
        or D[1] <= 0
    ):
        continue


    rate = (
        np.log(
            D[1] / D[0]
        )
        /
        dt
    )


    record = {

        "patient_id":
            item["patient_id"],

        "lesion_alias":
            item["lesion_alias"],

        "initial_diameter_mm":
            D[0],

        "initial_time_days":
            t[0],

        "initial_interval_days":
            dt,

        "observed_initial_rate":
            rate
    }


    # Add baseline geometry when available
    for c in available_optional:

        record[
            c
        ] = group.loc[
            0,
            c
        ]


    initial_records.append(
        record
    )


initial = pd.DataFrame(
    initial_records
)


print(
    "Initial transition records:",
    len(initial)
)


# ============================================================
# TRAINING DATA
# ============================================================

train_initial = initial[
    initial["patient_id"].isin(
        train_patients
    )
].copy()


test_initial = initial[
    initial["patient_id"].isin(
        test_patients
    )
].copy()


print(
    "Training initial transitions:",
    len(train_initial)
)

print(
    "Testing initial transitions:",
    len(test_initial)
)


# ============================================================
# MODEL 1: GLOBAL RATE
# ============================================================

header("MODEL 1 - GLOBAL GROWTH RATE")

global_rate = np.median(
    train_initial[
        "observed_initial_rate"
    ]
)

print(
    "Training global rate:",
    round(
        global_rate,
        6
    )
)


# ============================================================
# MODEL 2: STATE RATE
# ============================================================

header("MODEL 2 - STATE-SPECIFIC RATE")

# Use fixed symmetric thresholds around zero.
#
# This avoids the Step 20 problem where the training
# quantile caused the growth-state median to become zero.

GROWTH_THRESHOLD = 0.001

SHRINKAGE_THRESHOLD = -0.001


def state_from_rate(rate):

    if rate >= GROWTH_THRESHOLD:
        return "growth"

    if rate <= SHRINKAGE_THRESHOLD:
        return "shrinkage"

    return "stable"


train_initial[
    "state"
] = train_initial[
    "observed_initial_rate"
].apply(
    state_from_rate
)


state_rates = {}

for state in [
    "growth",
    "stable",
    "shrinkage"
]:

    values = train_initial[
        train_initial["state"] == state
    ][
        "observed_initial_rate"
    ].values


    if len(values) > 0:

        state_rates[state] = np.median(
            values
        )

    else:

        state_rates[state] = global_rate


    print(
        state,
        "n =",
        len(values),
        "| rate =",
        round(
            state_rates[state],
            6
        )
    )


# ============================================================
# MODEL 3: INDIVIDUALIZED RATE
# ============================================================

header("MODEL 3 - INDIVIDUALIZED GROWTH RATE")

print("""
The individualized model estimates the growth rate
from the first observed transition of the held-out lesion:

    r_individual =
        log(D1 / D0) / Δt

This is deliberately NOT a neural network.

It tests whether an individualized dynamical parameter
contains useful predictive information.

If individualized r substantially outperforms fixed
population rates, that supports making r(x) the target
of a future neural imaging encoder.
""")


# ============================================================
# PREDICTIONS
# ============================================================

header("FORWARD PREDICTION ON UNSEEN PATIENTS")

prediction_records = []


for item in trajectories:

    patient_id = item["patient_id"]

    if patient_id not in test_patients:
        continue


    group = item["data"]

    t = group[
        "days_from_baseline"
    ].values.astype(float)

    D = group[
        "diameter_mm"
    ].values.astype(float)


    D0 = D[0]

    t0 = t[0]


    dt_initial = t[1] - t[0]


    if (
        dt_initial <= 0
        or D0 <= 0
        or D[1] <= 0
    ):
        continue


    # Individual initial rate
    individual_rate = (
        np.log(
            D[1] / D0
        )
        /
        dt_initial
    )


    state = state_from_rate(
        individual_rate
    )


    state_rate = state_rates[
        state
    ]


    for i in range(
        2,
        len(group)
    ):

        future_t = t[i]

        observed = D[i]

        delta_t = (
            future_t
            -
            t0
        )


        # ----------------------------------------------------
        # MODEL A: NO CHANGE
        # ----------------------------------------------------

        pred_no_change = D0


        # ----------------------------------------------------
        # MODEL B: GLOBAL EXPONENTIAL
        # ----------------------------------------------------

        pred_global = (
            D0
            *
            np.exp(
                global_rate
                *
                delta_t
            )
        )


        # ----------------------------------------------------
        # MODEL C: STATE-SPECIFIC
        # ----------------------------------------------------

        pred_state = (
            D0
            *
            np.exp(
                state_rate
                *
                delta_t
            )
        )


        # ----------------------------------------------------
        # MODEL D: INDIVIDUALIZED
        # ----------------------------------------------------

        pred_individual = (
            D0
            *
            np.exp(
                individual_rate
                *
                delta_t
            )
        )


        prediction_records.append({

            "patient_id":
                patient_id,

            "lesion_alias":
                item["lesion_alias"],

            "prediction_day":
                future_t,

            "days_ahead":
                delta_t,

            "baseline_diameter_mm":
                D0,

            "observed_diameter_mm":
                observed,

            "individual_rate":
                individual_rate,

            "state":
                state,

            "global_rate":
                global_rate,

            "state_rate":
                state_rate,

            "pred_no_change":
                pred_no_change,

            "pred_global":
                pred_global,

            "pred_state":
                pred_state,

            "pred_individual":
                pred_individual,

            "error_no_change":
                observed
                -
                pred_no_change,

            "error_global":
                observed
                -
                pred_global,

            "error_state":
                observed
                -
                pred_state,

            "error_individual":
                observed
                -
                pred_individual
        })


predictions = pd.DataFrame(
    prediction_records
)


print(
    "Independent future observations:",
    len(predictions)
)


if len(predictions) == 0:

    print(
        "ERROR: No predictions generated."
    )

    sys.exit(1)


# ============================================================
# PERFORMANCE
# ============================================================

header("MODEL PERFORMANCE")

models = {

    "no_change":
        "error_no_change",

    "global_exponential":
        "error_global",

    "state_specific":
        "error_state",

    "individualized":
        "error_individual"
}


performance_rows = []


for model_name, error_column in models.items():

    errors = predictions[
        error_column
    ].values


    mae = np.mean(
        np.abs(errors)
    )


    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )


    performance_rows.append({

        "model":
            model_name,

        "observations":
            len(errors),

        "MAE_mm":
            mae,

        "RMSE_mm":
            rmse
    })


performance = pd.DataFrame(
    performance_rows
)


print(
    performance.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# IMPROVEMENTS
# ============================================================

header("MODEL IMPROVEMENT")

baseline_mae = performance.loc[
    performance["model"]
    ==
    "no_change",
    "MAE_mm"
].iloc[0]


state_mae = performance.loc[
    performance["model"]
    ==
    "state_specific",
    "MAE_mm"
].iloc[0]


individual_mae = performance.loc[
    performance["model"]
    ==
    "individualized",
    "MAE_mm"
].iloc[0]


print(
    "No-change MAE:",
    round(
        baseline_mae,
        4
    )
)

print(
    "State-specific MAE:",
    round(
        state_mae,
        4
    )
)

print(
    "Individualized MAE:",
    round(
        individual_mae,
        4
    )
)


if baseline_mae > 0:

    state_improvement = (
        (
            baseline_mae
            -
            state_mae
        )
        /
        baseline_mae
        *
        100
    )


    individual_improvement = (
        (
            baseline_mae
            -
            individual_mae
        )
        /
        baseline_mae
        *
        100
    )

else:

    state_improvement = np.nan
    individual_improvement = np.nan


print(
    "State-specific improvement:",
    round(
        state_improvement,
        2
    ),
    "%"
)

print(
    "Individualized improvement:",
    round(
        individual_improvement,
        2
    ),
    "%"
)


# ============================================================
# PATIENT PERFORMANCE
# ============================================================

header("PERFORMANCE BY UNSEEN PATIENT")

patient_rows = []


for patient_id, group in predictions.groupby(
    "patient_id"
):

    row = {
        "patient_id":
            patient_id,

        "observations":
            len(group)
    }


    for model_name, error_column in models.items():

        row[
            model_name + "_MAE_mm"
        ] = np.mean(
            np.abs(
                group[
                    error_column
                ]
            )
        )


    patient_rows.append(
        row
    )


patient_performance = pd.DataFrame(
    patient_rows
)


print(
    patient_performance.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# MODEL WIN COUNT
# ============================================================

header("PATIENT-LEVEL MODEL WIN COUNT")

win_counts = {
    model: 0
    for model in models
}


for _, row in patient_performance.iterrows():

    values = {
        model:
            row[
                model + "_MAE_mm"
            ]
        for model in models
    }


    best_model = min(
        values,
        key=values.get
    )


    win_counts[
        best_model
    ] += 1


for model, count in win_counts.items():

    print(
        model,
        ":",
        count
    )


# ============================================================
# INDIVIDUALIZED RATE DISTRIBUTION
# ============================================================

header("INDIVIDUALIZED RATE DISTRIBUTION")

individual_rates = (
    predictions[
        [
            "patient_id",
            "lesion_alias",
            "individual_rate"
        ]
    ]
    .drop_duplicates()
)


print(
    "Trajectories:",
    len(individual_rates)
)

print(
    "Mean r:",
    round(
        individual_rates[
            "individual_rate"
        ].mean(),
        6
    )
)

print(
    "Median r:",
    round(
        individual_rates[
            "individual_rate"
        ].median(),
        6
    )
)

print(
    "Minimum r:",
    round(
        individual_rates[
            "individual_rate"
        ].min(),
        6
    )
)

print(
    "Maximum r:",
    round(
        individual_rates[
            "individual_rate"
        ].max(),
        6
    )
)


# ============================================================
# INTERPRETATION
# ============================================================

header("STEP 21 - INTERPRETATION")

print("""
This experiment asks an important architectural question:

Is a single population-level growth parameter sufficient,
or does the model need an individualized dynamical parameter?

Four approaches are compared:

    A. No-change baseline

    B. Global exponential rate

    C. State-specific population rate

    D. Individualized lesion-specific rate

The individualized rate is estimated only from the first
transition of each held-out lesion.

Its future observations are then predicted.

This experiment does NOT use a neural network.

The purpose is to determine whether individualized
dynamical parameters contain predictive information.

If individualized r substantially improves prediction,
the next architecture can use an imaging encoder to learn:

    CT → latent representation → individualized r

while the differential equation remains:

    dD/dt = r(x,z) D
""")


# ============================================================
# FINAL DECISION
# ============================================================

header("STEP 21 - FINAL FEASIBILITY")

if individual_mae < baseline_mae:

    print(
        "[YES] Individualized growth rate beats "
        "the no-change baseline"
    )

else:

    print(
        "[NO] Individualized growth rate does not "
        "beat the no-change baseline"
    )


if individual_mae < state_mae:

    print(
        "[YES] Individualized rate beats "
        "state-specific population rate"
    )

else:

    print(
        "[NO] Individualized rate does not beat "
        "state-specific population rate"
    )


print(
    "[YES] Patient-independent evaluation performed"
)

print(
    "[YES] No neural network used"
)

print(
    "[YES] Dynamical parameter identified explicitly"
)


# ============================================================
# SAVE
# ============================================================

header("SAVING STEP 21 RESULTS")

predictions.to_csv(
    PREDICTION_FILE,
    index=False
)

performance.to_csv(
    PERFORMANCE_FILE,
    index=False
)

patient_performance.to_csv(
    PATIENT_FILE,
    index=False
)


print(
    "Saved:",
    PREDICTION_FILE
)

print(
    "Saved:",
    PERFORMANCE_FILE
)

print(
    "Saved:",
    PATIENT_FILE
)


# ============================================================
# COMPLETE
# ============================================================

header("STEP 21 COMPLETE")

print("""
Run:

    python 21_test_individualized_hbgd.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A NEURAL NETWORK YET.
""")

print("=" * 95)
print("END OF STEP 21")
print("=" * 95)