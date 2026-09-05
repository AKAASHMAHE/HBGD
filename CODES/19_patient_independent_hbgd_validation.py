# 19_patient_independent_hbgd_validation.py

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
# PATH
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

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

header("STEP 19 - PATIENT-INDEPENDENT HBGD VALIDATION")

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
# NORMALIZE
# ============================================================

header("NORMALIZING DATA")

for c in [
    "patient_id",
    "diameter_mm",
    "days_from_baseline"
]:
    df[c] = pd.to_numeric(
        df[c],
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
# BUILD ELIGIBLE TRAJECTORIES
# ============================================================

header("BUILDING ELIGIBLE LESION TRAJECTORIES")

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
    "Trajectories with >=3 observations:",
    len(trajectories)
)


eligible_patients = sorted(
    set(
        item["patient_id"]
        for item in trajectories
    )
)


print(
    "Eligible patients:",
    len(eligible_patients)
)

print(
    "Patient IDs:",
    eligible_patients
)


if len(eligible_patients) < 3:

    print(
        "ERROR: Too few eligible patients."
    )

    sys.exit(1)


# ============================================================
# DEFINE MODELS
# ============================================================

header("DEFINING VALIDATION MODELS")

print("""
MODEL A
-------
No-change:

    D(t) = D0


MODEL B
-------
Patient-independent exponential:

    D(t) = D0 * exp(r_global * t)

where r_global is learned only from training patients.


MODEL C
-------
HBGD state-aware model:

    dD/dt = r_z * D

where r_z is learned only from training patients
for:

    growth
    stable
    shrinkage


For every held-out patient, ONLY the first transition
of each lesion is used to estimate its individual
initial state/rate.

Later observations are prediction targets.
""")


# ============================================================
# STATE FUNCTION
# ============================================================

def classify_state(rate):

    if rate > 0.001:
        return "growth"

    if rate < -0.001:
        return "shrinkage"

    return "stable"


# ============================================================
# STORAGE
# ============================================================

all_predictions = []
patient_results = []


# ============================================================
# LEAVE-ONE-PATIENT-OUT
# ============================================================

header(
    "LEAVE-ONE-PATIENT-OUT VALIDATION"
)


for test_patient in eligible_patients:

    print()
    print(
        "-" * 95
    )

    print(
        "Testing patient:",
        test_patient
    )


    train_patients = [
        p
        for p in eligible_patients
        if p != test_patient
    ]


    train_trajectories = [
        item
        for item in trajectories
        if item["patient_id"]
        in train_patients
    ]


    test_trajectories = [
        item
        for item in trajectories
        if item["patient_id"]
        == test_patient
    ]


    # ========================================================
    # LEARN GLOBAL RATE FROM TRAINING PATIENTS
    # ========================================================

    training_rates = []

    state_rates = {
        "growth": [],
        "stable": [],
        "shrinkage": []
    }


    for item in train_trajectories:

        group = item["data"]

        t = group[
            "days_from_baseline"
        ].values.astype(float)

        D = group[
            "diameter_mm"
        ].values.astype(float)


        t0 = t[0]
        D0 = D[0]

        t1 = t[1]
        D1 = D[1]


        dt = t1 - t0


        if (
            dt <= 0
            or D0 <= 0
            or D1 <= 0
        ):
            continue


        rate = (
            np.log(D1 / D0)
            /
            dt
        )


        training_rates.append(rate)


        state = classify_state(rate)

        state_rates[state].append(rate)


    if len(training_rates) == 0:
        continue


    global_rate = np.median(
        training_rates
    )


    state_parameters = {}


    for state in [
        "growth",
        "stable",
        "shrinkage"
    ]:

        if len(
            state_rates[state]
        ) > 0:

            state_parameters[state] = np.median(
                state_rates[state]
            )

        else:

            # fallback to global rate
            state_parameters[state] = global_rate


    # ========================================================
    # TEST EACH LESION
    # ========================================================

    patient_errors = {
        "no_change": [],
        "global_exponential": [],
        "hbgd_state": []
    }


    for item in test_trajectories:

        group = item["data"]

        t = group[
            "days_from_baseline"
        ].values.astype(float)

        D = group[
            "diameter_mm"
        ].values.astype(float)


        # ----------------------------------------------------
        # Initial observation
        # ----------------------------------------------------

        t0 = t[0]
        D0 = D[0]


        # ----------------------------------------------------
        # Initial transition used ONLY to characterize
        # the held-out lesion.
        #
        # It is NOT used to learn population parameters.
        # ----------------------------------------------------

        t1 = t[1]
        D1 = D[1]

        dt01 = t1 - t0


        if (
            dt01 <= 0
            or D0 <= 0
            or D1 <= 0
        ):
            continue


        initial_rate = (
            np.log(D1 / D0)
            /
            dt01
        )


        initial_state = classify_state(
            initial_rate
        )


        hbgd_rate = state_parameters[
            initial_state
        ]


        # ----------------------------------------------------
        # Predict every later observation
        # ----------------------------------------------------

        for i in range(
            2,
            len(group)
        ):

            current_t = t[i]

            observed = D[i]

            delta_t = (
                current_t
                -
                t0
            )


            # ================================================
            # MODEL A
            # ================================================

            pred_no_change = D0


            # ================================================
            # MODEL B
            # ================================================

            pred_global = (
                D0
                *
                np.exp(
                    global_rate
                    *
                    delta_t
                )
            )


            # ================================================
            # MODEL C
            # ================================================

            pred_hbgd = (
                D0
                *
                np.exp(
                    hbgd_rate
                    *
                    delta_t
                )
            )


            error_no_change = (
                observed
                -
                pred_no_change
            )


            error_global = (
                observed
                -
                pred_global
            )


            error_hbgd = (
                observed
                -
                pred_hbgd
            )


            patient_errors[
                "no_change"
            ].append(
                abs(error_no_change)
            )


            patient_errors[
                "global_exponential"
            ].append(
                abs(error_global)
            )


            patient_errors[
                "hbgd_state"
            ].append(
                abs(error_hbgd)
            )


            all_predictions.append({

                "test_patient":
                    test_patient,

                "lesion_alias":
                    item["lesion_alias"],

                "prediction_day":
                    current_t,

                "days_ahead":
                    delta_t,

                "observed_diameter_mm":
                    observed,

                "baseline_diameter_mm":
                    D0,

                "initial_rate":
                    initial_rate,

                "initial_state":
                    initial_state,

                "global_training_rate":
                    global_rate,

                "hbgd_state_rate":
                    hbgd_rate,

                "pred_no_change":
                    pred_no_change,

                "pred_global_exponential":
                    pred_global,

                "pred_hbgd_state":
                    pred_hbgd,

                "error_no_change_mm":
                    error_no_change,

                "error_global_exponential_mm":
                    error_global,

                "error_hbgd_state_mm":
                    error_hbgd
            })


    # ========================================================
    # PATIENT SUMMARY
    # ========================================================

    for model_name in patient_errors:

        if len(
            patient_errors[model_name]
        ) > 0:

            patient_errors[model_name] = np.array(
                patient_errors[model_name]
            )

        else:

            patient_errors[model_name] = np.array([])


    patient_results.append({

        "patient_id":
            test_patient,

        "observations":
            len(
                patient_errors[
                    "no_change"
                ]
            ),

        "no_change_MAE_mm":
            (
                np.mean(
                    patient_errors[
                        "no_change"
                    ]
                )
                if len(
                    patient_errors[
                        "no_change"
                    ]
                )
                else np.nan
            ),

        "global_exponential_MAE_mm":
            (
                np.mean(
                    patient_errors[
                        "global_exponential"
                    ]
                )
                if len(
                    patient_errors[
                        "global_exponential"
                    ]
                )
                else np.nan
            ),

        "hbgd_state_MAE_mm":
            (
                np.mean(
                    patient_errors[
                        "hbgd_state"
                    ]
                )
                if len(
                    patient_errors[
                        "hbgd_state"
                    ]
                )
                else np.nan
            )
    })


    print(
        "Global training rate:",
        round(
            global_rate,
            6
        )
    )


    print(
        "State parameters:"
    )

    for state in [
        "growth",
        "stable",
        "shrinkage"
    ]:

        print(
            "   ",
            state,
            ":",
            round(
                state_parameters[state],
                6
            )
        )


# ============================================================
# RESULTS
# ============================================================

predictions = pd.DataFrame(
    all_predictions
)

patient_performance = pd.DataFrame(
    patient_results
)


header(
    "PATIENT-INDEPENDENT MODEL PERFORMANCE"
)


if len(predictions) == 0:

    print(
        "ERROR: No independent predictions generated."
    )

    sys.exit(1)


performance_rows = []


for model_name, column in [
    (
        "no_change",
        "error_no_change_mm"
    ),
    (
        "global_exponential",
        "error_global_exponential_mm"
    ),
    (
        "hbgd_state",
        "error_hbgd_state_mm"
    )
]:

    errors = predictions[
        column
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
# IMPROVEMENT
# ============================================================

header(
    "MODEL COMPARISON"
)


baseline_mae = performance.loc[
    performance["model"]
    ==
    "no_change",
    "MAE_mm"
].iloc[0]


global_mae = performance.loc[
    performance["model"]
    ==
    "global_exponential",
    "MAE_mm"
].iloc[0]


hbgd_mae = performance.loc[
    performance["model"]
    ==
    "hbgd_state",
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
    "Global exponential MAE:",
    round(
        global_mae,
        4
    )
)


print(
    "HBGD state-aware MAE:",
    round(
        hbgd_mae,
        4
    )
)


if baseline_mae > 0:

    global_improvement = (
        (
            baseline_mae
            -
            global_mae
        )
        /
        baseline_mae
        *
        100
    )


    hbgd_improvement = (
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

    global_improvement = np.nan
    hbgd_improvement = np.nan


print(
    "Global exponential improvement:",
    round(
        global_improvement,
        2
    ),
    "%"
)


print(
    "HBGD state-aware improvement:",
    round(
        hbgd_improvement,
        2
    ),
    "%"
)


# ============================================================
# PATIENT PERFORMANCE
# ============================================================

header(
    "PERFORMANCE BY UNSEEN PATIENT"
)


print(
    patient_performance.round(
        3
    ).to_string(
        index=False
    )
)


# ============================================================
# WHICH MODEL WINS?
# ============================================================

header(
    "PATIENT-LEVEL MODEL WIN COUNT"
)


wins = {
    "no_change": 0,
    "global_exponential": 0,
    "hbgd_state": 0
}


for _, row in patient_performance.iterrows():

    values = {

        "no_change":
            row["no_change_MAE_mm"],

        "global_exponential":
            row[
                "global_exponential_MAE_mm"
            ],

        "hbgd_state":
            row[
                "hbgd_state_MAE_mm"
            ]
    }


    valid_values = {
        k: v
        for k, v in values.items()
        if pd.notna(v)
    }


    if len(valid_values) == 0:
        continue


    best = min(
        valid_values,
        key=valid_values.get
    )


    wins[best] += 1


for model_name, count in wins.items():

    print(
        model_name,
        ":",
        count
    )


# ============================================================
# STATE DISTRIBUTION IN TEST PREDICTIONS
# ============================================================

header(
    "HELD-OUT INITIAL STATE DISTRIBUTION"
)


print(
    predictions[
        "initial_state"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# INTERPRETATION
# ============================================================

header(
    "STEP 19 - INTERPRETATION"
)


print("""
This experiment evaluates HBGD on unseen patients.

For each held-out patient:

    training patients
           ↓
    estimate population parameters
           ↓
    unseen patient
           ↓
    observe first lesion transition
           ↓
    determine initial observable state
           ↓
    select HBGD state parameter
           ↓
    predict later observations

The held-out patient's later observations are never
used to estimate the population growth parameters.

Three models are compared:

    A. No-change baseline

    B. Global exponential model

    C. HBGD state-aware model

This is a substantially stronger feasibility experiment
than fitting all observations together.

However, the number of independent patients remains small.

Therefore the result should be interpreted as preliminary
evidence for or against the mathematical architecture,
not clinical validation.
""")


# ============================================================
# FINAL DECISION
# ============================================================

header(
    "STEP 19 - FINAL FEASIBILITY"
)


if hbgd_mae < baseline_mae:

    print(
        "[YES] HBGD state-aware model beats"
        " the no-change baseline"
    )

else:

    print(
        "[NO] HBGD state-aware model does not beat"
        " the no-change baseline"
    )


if hbgd_mae < global_mae:

    print(
        "[YES] HBGD state-aware model beats"
        " global exponential model"
    )

else:

    print(
        "[NO] HBGD state-aware model does not beat"
        " global exponential model"
    )


if len(
    predictions
) >= 10:

    print(
        "[YES] Independent forward observations"
        " available for exploratory validation"
    )

else:

    print(
        "[WARNING] Very small independent"
        " evaluation set"
    )


# ============================================================
# SAVE
# ============================================================

header(
    "SAVING STEP 19 RESULTS"
)


performance_file = os.path.join(
    DATA_DIR,
    "19_patient_independent_performance.csv"
)

patient_file = os.path.join(
    DATA_DIR,
    "19_patient_independent_patient_performance.csv"
)

prediction_file = os.path.join(
    DATA_DIR,
    "19_patient_independent_predictions.csv"
)


performance.to_csv(
    performance_file,
    index=False
)

patient_performance.to_csv(
    patient_file,
    index=False
)

predictions.to_csv(
    prediction_file,
    index=False
)


print(
    "Saved:",
    performance_file
)

print(
    "Saved:",
    patient_file
)

print(
    "Saved:",
    prediction_file
)


# ============================================================
# COMPLETE
# ============================================================

header(
    "STEP 19 COMPLETE"
)

print("""
Run:

    python 19_patient_independent_hbgd_validation.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A NEURAL NETWORK YET.
""")

print("=" * 95)
print("END OF STEP 19")
print("=" * 95)