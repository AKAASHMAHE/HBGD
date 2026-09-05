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

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "20_hbgd_prototype_predictions.csv"
)

PERFORMANCE_FILE = os.path.join(
    DATA_DIR,
    "20_hbgd_prototype_performance.csv"
)

PARAMETER_FILE = os.path.join(
    DATA_DIR,
    "20_hbgd_prototype_parameters.csv"
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

header("STEP 20 - BUILDING HBGD PROTOTYPE")

if not os.path.exists(FEATURE_FILE):
    print("ERROR: Required file not found:")
    print(FEATURE_FILE)
    sys.exit(1)

df = pd.read_csv(FEATURE_FILE)

print("Feature records:", len(df))


# ============================================================
# REQUIRED VARIABLES
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
# NORMALIZATION
# ============================================================

header("NORMALIZING DATA")

df["patient_id"] = pd.to_numeric(
    df["patient_id"],
    errors="coerce"
)

df["diameter_mm"] = pd.to_numeric(
    df["diameter_mm"],
    errors="coerce"
)

df["days_from_baseline"] = pd.to_numeric(
    df["days_from_baseline"],
    errors="coerce"
)

df = df.dropna(
    subset=required
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

header("BUILDING LESION TRAJECTORIES")

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
        x["patient_id"]
        for x in trajectories
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
# STATE ASSIGNMENT
# ============================================================

header("DEFINING OBSERVABLE DYNAMICAL STATES")

print("""
The initial lesion transition is used to estimate:

    r = log(D1 / D0) / Δt

The rate determines the observable state:

    growth
    stable
    shrinkage

The state is an observable mathematical category.
It is NOT interpreted as a biological state.
""")


def estimate_rate(D0, D1, dt):

    if (
        D0 <= 0 or
        D1 <= 0 or
        dt <= 0
    ):
        return np.nan

    return np.log(
        D1 / D0
    ) / dt


def state_from_rate(
    rate,
    growth_threshold,
    shrink_threshold
):

    if rate >= growth_threshold:
        return "growth"

    if rate <= shrink_threshold:
        return "shrinkage"

    return "stable"


# ============================================================
# CONSTRUCT FIRST-TRANSITION DATA
# ============================================================

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

    rate = estimate_rate(
        D[0],
        D[1],
        dt
    )

    if not np.isfinite(rate):
        continue

    initial_records.append({
        "patient_id":
            item["patient_id"],

        "lesion_alias":
            item["lesion_alias"],

        "initial_rate":
            rate
    })


initial_df = pd.DataFrame(
    initial_records
)

print(
    "Initial transitions:",
    len(initial_df)
)


# ============================================================
# PATIENT-LEVEL SPLIT
# ============================================================

header("CREATING PATIENT-INDEPENDENT TRAINING / TEST SET")

# Fixed deterministic split for reproducibility.
#
# The final three patients are held out.
# No observations from these patients are used to
# estimate HBGD state parameters.

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


train_initial = initial_df[
    initial_df["patient_id"].isin(
        train_patients
    )
].copy()

test_initial = initial_df[
    initial_df["patient_id"].isin(
        test_patients
    )
].copy()


# ============================================================
# LEARN STATE THRESHOLDS FROM TRAINING DATA
# ============================================================

header("LEARNING STATE THRESHOLDS")

train_rates = train_initial[
    "initial_rate"
].values


if len(train_rates) < 3:

    print(
        "ERROR: Too few training rates."
    )

    sys.exit(1)


# Use lower/upper terciles as exploratory
# state boundaries.

growth_threshold = np.quantile(
    train_rates,
    2.0 / 3.0
)

shrink_threshold = np.quantile(
    train_rates,
    1.0 / 3.0
)


print(
    "Shrinkage threshold:",
    round(
        shrink_threshold,
        6
    )
)

print(
    "Growth threshold:",
    round(
        growth_threshold,
        6
    )
)


# ============================================================
# ASSIGN TRAINING STATES
# ============================================================

train_initial[
    "state"
] = train_initial[
    "initial_rate"
].apply(
    lambda x:
        state_from_rate(
            x,
            growth_threshold,
            shrink_threshold
        )
)


print()
print(
    "Training state distribution:"
)

print(
    train_initial[
        "state"
    ].value_counts()
)


# ============================================================
# LEARN STATE PARAMETERS
# ============================================================

header("LEARNING HBGD STATE PARAMETERS")

state_parameters = {}

global_rate = np.median(
    train_initial[
        "initial_rate"
    ]
)


for state in [
    "growth",
    "stable",
    "shrinkage"
]:

    values = train_initial[
        train_initial["state"] == state
    ]["initial_rate"].values


    if len(values) > 0:

        parameter = np.median(
            values
        )

    else:

        parameter = global_rate


    state_parameters[
        state
    ] = parameter


    print(
        state,
        "n =",
        len(values),
        "| r =",
        round(
            parameter,
            6
        )
    )


print()
print(
    "Global fallback rate:",
    round(
        global_rate,
        6
    )
)


# ============================================================
# HBGD EQUATION
# ============================================================

header("HBGD DYNAMICAL EQUATION")

print("""
For a lesion with baseline diameter D0:

    dD/dt = r_z * D

Solution:

    D(t) = D0 * exp(r_z * t)

where r_z is selected according to the
observable initial dynamical state.
""")


# ============================================================
# TEST PREDICTION
# ============================================================

header("PATIENT-INDEPENDENT HBGD PREDICTION")

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


    initial_rate = estimate_rate(
        D[0],
        D[1],
        dt_initial
    )


    if not np.isfinite(initial_rate):
        continue


    state = state_from_rate(
        initial_rate,
        growth_threshold,
        shrink_threshold
    )


    r_hbgd = state_parameters[
        state
    ]


    for i in range(
        2,
        len(group)
    ):

        future_time = t[i]

        observed = D[i]

        delta_t = (
            future_time
            -
            t0
        )


        predicted = (
            D0
            *
            np.exp(
                r_hbgd
                *
                delta_t
            )
        )


        error = (
            observed
            -
            predicted
        )


        prediction_records.append({

            "patient_id":
                patient_id,

            "lesion_alias":
                item["lesion_alias"],

            "prediction_day":
                future_time,

            "days_ahead":
                delta_t,

            "baseline_diameter_mm":
                D0,

            "observed_diameter_mm":
                observed,

            "initial_rate":
                initial_rate,

            "state":
                state,

            "hbgd_rate":
                r_hbgd,

            "predicted_diameter_mm":
                predicted,

            "error_mm":
                error,

            "absolute_error_mm":
                abs(error)
        })


predictions = pd.DataFrame(
    prediction_records
)


print(
    "Independent prediction observations:",
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

header("HBGD PROTOTYPE PERFORMANCE")

hbgd_mae = predictions[
    "absolute_error_mm"
].mean()

hbgd_rmse = np.sqrt(
    np.mean(
        predictions[
            "error_mm"
        ] ** 2
    )
)


# No-change

baseline_error = (
    predictions[
        "observed_diameter_mm"
    ]
    -
    predictions[
        "baseline_diameter_mm"
    ]
)


baseline_mae = np.mean(
    np.abs(
        baseline_error
    )
)

baseline_rmse = np.sqrt(
    np.mean(
        baseline_error ** 2
    )
)


print(
    "HBGD MAE:",
    round(
        hbgd_mae,
        4
    ),
    "mm"
)

print(
    "HBGD RMSE:",
    round(
        hbgd_rmse,
        4
    ),
    "mm"
)

print()

print(
    "No-change MAE:",
    round(
        baseline_mae,
        4
    ),
    "mm"
)

print(
    "No-change RMSE:",
    round(
        baseline_rmse,
        4
    ),
    "mm"
)


if baseline_mae > 0:

    improvement = (
        (
            baseline_mae
            -
            hbgd_mae
        )
        /
        baseline_mae
        *
        100
    )

else:

    improvement = np.nan


print()

print(
    "HBGD improvement:",
    round(
        improvement,
        2
    ),
    "%"
)


# ============================================================
# PERFORMANCE BY STATE
# ============================================================

header("PERFORMANCE BY OBSERVABLE STATE")

state_performance = (
    predictions
    .groupby("state")
    .agg(
        observations=(
            "absolute_error_mm",
            "count"
        ),
        MAE_mm=(
            "absolute_error_mm",
            "mean"
        ),
        RMSE_mm=(
            "error_mm",
            lambda x:
                np.sqrt(
                    np.mean(
                        x ** 2
                    )
                )
        )
    )
    .reset_index()
)


print(
    state_performance.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# PERFORMANCE BY PATIENT
# ============================================================

header("PERFORMANCE BY HELD-OUT PATIENT")

patient_performance = (
    predictions
    .groupby(
        "patient_id"
    )
    .agg(
        observations=(
            "absolute_error_mm",
            "count"
        ),
        HBGD_MAE_mm=(
            "absolute_error_mm",
            "mean"
        ),
        HBGD_RMSE_mm=(
            "error_mm",
            lambda x:
                np.sqrt(
                    np.mean(
                        x ** 2
                    )
                )
        )
    )
    .reset_index()
)


print(
    patient_performance.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# SAVE PARAMETERS
# ============================================================

parameter_rows = [
    {
        "parameter":
            "growth_threshold",

        "value":
            growth_threshold
    },
    {
        "parameter":
            "shrinkage_threshold",

        "value":
            shrink_threshold
    },
    {
        "parameter":
            "growth_rate",

        "value":
            state_parameters[
                "growth"
            ]
    },
    {
        "parameter":
            "stable_rate",

        "value":
            state_parameters[
                "stable"
            ]
    },
    {
        "parameter":
            "shrinkage_rate",

        "value":
            state_parameters[
                "shrinkage"
            ]
    }
]


parameters = pd.DataFrame(
    parameter_rows
)


# ============================================================
# SAVE PERFORMANCE
# ============================================================

performance = pd.DataFrame([

    {
        "model":
            "no_change",

        "observations":
            len(predictions),

        "MAE_mm":
            baseline_mae,

        "RMSE_mm":
            baseline_rmse
    },

    {
        "model":
            "HBGD_state_aware",

        "observations":
            len(predictions),

        "MAE_mm":
            hbgd_mae,

        "RMSE_mm":
            hbgd_rmse
    }
])


# ============================================================
# INTERPRETATION
# ============================================================

header("STEP 20 - INTERPRETATION")

print("""
This is the first compact HBGD prototype.

The prototype contains:

    1. Observable dynamical state
    2. State-dependent growth parameter
    3. Differential equation
    4. Forward integration
    5. Patient-independent prediction

The model is:

    dD/dt = r_z D

with:

    z = observable initial dynamical state

and:

    r_z = learned state-specific parameter.

No neural network is used yet.

This is intentional.

The purpose is to establish whether the
mathematical HBGD core itself works before
adding an imaging encoder.
""")


# ============================================================
# FINAL DECISION
# ============================================================

header("STEP 20 - FINAL FEASIBILITY")

if hbgd_mae < baseline_mae:

    print(
        "[YES] HBGD prototype beats the no-change baseline"
    )

else:

    print(
        "[NO] HBGD prototype does not beat the no-change baseline"
    )


if hbgd_mae < 5:

    print(
        "[YES] Exploratory prediction error is below 5 mm"
    )

else:

    print(
        "[NO] Prediction error remains above 5 mm"
    )


print(
    "[YES] Patient-independent test performed"
)

print(
    "[YES] State-dependent dynamical equation implemented"
)

print(
    "[YES] Neural component intentionally deferred"
)


# ============================================================
# SAVE
# ============================================================

header("SAVING STEP 20 RESULTS")

predictions.to_csv(
    OUTPUT_FILE,
    index=False
)

performance.to_csv(
    PERFORMANCE_FILE,
    index=False
)

parameters.to_csv(
    PARAMETER_FILE,
    index=False
)


print(
    "Saved:",
    OUTPUT_FILE
)

print(
    "Saved:",
    PERFORMANCE_FILE
)

print(
    "Saved:",
    PARAMETER_FILE
)


# ============================================================
# COMPLETE
# ============================================================

header("STEP 20 COMPLETE")

print("""
Run:

    python 20_build_hbgd_prototype.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A CNN YET.
DO NOT CLAIM BIOLOGICAL STATES.
DO NOT CLAIM CLINICAL VALIDATION.
""")

print("=" * 95)
print("END OF STEP 20")
print("=" * 95)