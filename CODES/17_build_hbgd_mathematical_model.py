# 17_build_hbgd_mathematical_model.py

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# ============================================================
# TERMINAL
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
    "9_hbgd_log_growth_features.csv"
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
    "STEP 17 - LOADING HBGD MATHEMATICAL DATA"
)


if not os.path.exists(FEATURE_FILE):

    print("ERROR: Missing:")
    print(FEATURE_FILE)
    sys.exit(1)


if not os.path.exists(PAIRWISE_FILE):

    print("ERROR: Missing:")
    print(PAIRWISE_FILE)
    sys.exit(1)


features = pd.read_csv(
    FEATURE_FILE
)

growth_data = pd.read_csv(
    PAIRWISE_FILE
)


print(
    "Feature records:",
    len(features)
)

print(
    "Growth records:",
    len(growth_data)
)


# ============================================================
# REQUIRED VARIABLES
# ============================================================

header(
    "CHECKING REQUIRED VARIABLES"
)


required_features = [
    "patient_id",
    "lesion_label_alias",
    "diameter_mm",
    "volume_cm3",
    "days_from_baseline"
]


missing = [
    c
    for c in required_features
    if c not in features.columns
]


if missing:

    print(
        "ERROR: Missing columns:"
    )

    for c in missing:
        print(
            "   ",
            c
        )

    sys.exit(1)


print(
    "All required variables are present."
)


# ============================================================
# NORMALIZE
# ============================================================

header(
    "NORMALIZING DATA"
)


numeric_columns = [
    "patient_id",
    "diameter_mm",
    "volume_cm3",
    "days_from_baseline"
]


for column in numeric_columns:

    features[column] = pd.to_numeric(
        features[column],
        errors="coerce"
    )


features = features.dropna(
    subset=[
        "patient_id",
        "diameter_mm",
        "volume_cm3",
        "days_from_baseline"
    ]
).copy()


features = features[
    features["diameter_mm"] > 0
].copy()


features = features[
    features["volume_cm3"] > 0
].copy()


features = features.sort_values(
    [
        "patient_id",
        "lesion_label_alias",
        "days_from_baseline"
    ]
).reset_index(
    drop=True
)


print(
    "Valid feature records:",
    len(features)
)


# ============================================================
# MODEL 1
# EXPONENTIAL DIAMETER MODEL
# ============================================================

header(
    "MODEL 1 - EXPONENTIAL DIAMETER DYNAMICS"
)


print("""
Initial HBGD equation:

    dD/dt = rD

whose analytical solution is:

    D(t) = D0 * exp(r*t)

where:

    D  = tumor diameter
    D0 = baseline diameter
    r  = estimated growth-rate parameter
    t  = time from baseline
""")


def exponential_model(
    t,
    D0,
    r
):

    return D0 * np.exp(
        r * t
    )


trajectory_records = []


for (
    patient_id,
    lesion_alias
), group in features.groupby(
    [
        "patient_id",
        "lesion_label_alias"
    ]
):

    group = group.sort_values(
        "days_from_baseline"
    )


    if len(group) < 2:

        continue


    t = group[
        "days_from_baseline"
    ].values.astype(float)


    D = group[
        "diameter_mm"
    ].values.astype(float)


    D0 = D[0]


    if D0 <= 0:

        continue


    # --------------------------------------------------------
    # Estimate r directly from log(D / D0)
    # --------------------------------------------------------

    valid = (
        D > 0
    )


    if valid.sum() < 2:

        continue


    t_valid = t[valid]
    D_valid = D[valid]


    log_ratio = np.log(
        D_valid / D0
    )


    denominator = np.sum(
        t_valid ** 2
    )


    if denominator <= 0:

        continue


    r = np.sum(
        t_valid * log_ratio
    ) / denominator


    predicted = exponential_model(
        t,
        D0,
        r
    )


    errors = (
        D
        -
        predicted
    )


    mae = np.mean(
        np.abs(errors)
    )


    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )


    for i in range(
        len(group)
    ):

        trajectory_records.append({

            "patient_id":
                patient_id,

            "lesion_alias":
                lesion_alias,

            "time_days":
                t[i],

            "observed_diameter_mm":
                D[i],

            "predicted_diameter_mm":
                predicted[i],

            "diameter_error_mm":
                errors[i],

            "growth_rate_r_per_day":
                r,

            "trajectory_MAE_mm":
                mae,

            "trajectory_RMSE_mm":
                rmse
        })


trajectory_table = pd.DataFrame(
    trajectory_records
)


print(
    "Longitudinal trajectories fitted:",
    trajectory_table[
        [
            "patient_id",
            "lesion_alias"
        ]
    ]
    .drop_duplicates()
    .shape[0]
)


# ============================================================
# GROWTH RATE SUMMARY
# ============================================================

header(
    "ESTIMATED PATIENT-LESION GROWTH PARAMETERS"
)


parameter_table = (
    trajectory_table[
        [
            "patient_id",
            "lesion_alias",
            "growth_rate_r_per_day",
            "trajectory_MAE_mm",
            "trajectory_RMSE_mm"
        ]
    ]
    .drop_duplicates(
        subset=[
            "patient_id",
            "lesion_alias"
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    parameter_table.to_string(
        index=False
    )
)


# ============================================================
# MODEL 2
# VOLUME DYNAMICS
# ============================================================

header(
    "MODEL 2 - VOLUME DYNAMICS"
)


print("""
For a simple geometric extension:

    dV/dt = rv * V

with:

    V(t) = V0 * exp(rv*t)

The volume growth parameter is estimated
independently from observed segmented volume.
""")


volume_records = []


for (
    patient_id,
    lesion_alias
), group in features.groupby(
    [
        "patient_id",
        "lesion_label_alias"
    ]
):

    group = group.sort_values(
        "days_from_baseline"
    )


    if len(group) < 2:

        continue


    t = group[
        "days_from_baseline"
    ].values.astype(float)


    V = group[
        "volume_cm3"
    ].values.astype(float)


    V0 = V[0]


    if V0 <= 0:

        continue


    valid = (
        V > 0
    )


    if valid.sum() < 2:

        continue


    tv = t[valid]
    Vv = V[valid]


    log_ratio = np.log(
        Vv / V0
    )


    denominator = np.sum(
        tv ** 2
    )


    if denominator <= 0:

        continue


    rv = np.sum(
        tv * log_ratio
    ) / denominator


    predicted_volume = (
        V0
        *
        np.exp(
            rv * t
        )
    )


    volume_error = (
        V
        -
        predicted_volume
    )


    volume_mae = np.mean(
        np.abs(
            volume_error
        )
    )


    volume_rmse = np.sqrt(
        np.mean(
            volume_error ** 2
        )
    )


    volume_records.append({

        "patient_id":
            patient_id,

        "lesion_alias":
            lesion_alias,

        "volume_growth_rate_per_day":
            rv,

        "volume_MAE_cm3":
            volume_mae,

        "volume_RMSE_cm3":
            volume_rmse
    })


volume_parameter_table = pd.DataFrame(
    volume_records
)


print(
    "Volume trajectories fitted:",
    len(
        volume_parameter_table
    )
)


# ============================================================
# DIAMETER-VOLUME RELATION
# ============================================================

header(
    "MODEL 3 - DIAMETER/VOLUME GEOMETRIC RELATION"
)


print("""
For approximately spherical geometry:

    V = (pi / 6) * D^3

Therefore:

    D_volume = (6V/pi)^(1/3)

This is a geometric transformation,
not a biological law.
""")


features[
    "sphere_equivalent_diameter_mm"
] = (
    (
        6.0
        *
        features["volume_cm3"]
        /
        np.pi
    )
    ** (
        1.0 / 3.0
    )
    *
    10.0
)


features[
    "geometric_diameter_difference_mm"
] = (
    features[
        "sphere_equivalent_diameter_mm"
    ]
    -
    features[
        "diameter_mm"
    ]
)


print(
    "Median sphere-equivalent diameter:",
    round(
        features[
            "sphere_equivalent_diameter_mm"
        ].median(),
        3
    ),
    "mm"
)


print(
    "Median diameter difference:",
    round(
        features[
            "geometric_diameter_difference_mm"
        ].median(),
        3
    ),
    "mm"
)


# ============================================================
# MODEL 4
# STATE-AWARE PARAMETER
# ============================================================

header(
    "MODEL 4 - STATE-AWARE HBGD PARAMETER"
)


print("""
The next HBGD extension introduces an observable state z:

    z ∈ {growth, stable, shrinkage}

The dynamical equation becomes:

    dD/dt = r_z * D

where r_z is allowed to depend on the
observed dynamical state.

This step only estimates the state-specific
parameters from the available observations.

It does NOT claim that these states represent
specific biological mechanisms.
""")


state_parameter_records = []


# ------------------------------------------------------------
# Infer state from sequential diameter changes
# ------------------------------------------------------------

for (
    patient_id,
    lesion_alias
), group in features.groupby(
    [
        "patient_id",
        "lesion_label_alias"
    ]
):

    group = group.sort_values(
        "days_from_baseline"
    ).reset_index(
        drop=True
    )


    if len(group) < 2:

        continue


    for i in range(
        1,
        len(group)
    ):

        d0 = float(
            group.loc[
                i - 1,
                "diameter_mm"
            ]
        )


        d1 = float(
            group.loc[
                i,
                "diameter_mm"
            ]
        )


        t0 = float(
            group.loc[
                i - 1,
                "days_from_baseline"
            ]
        )


        t1 = float(
            group.loc[
                i,
                "days_from_baseline"
            ]
        )


        dt = t1 - t0


        if (
            d0 <= 0
            or
            d1 <= 0
            or
            dt <= 0
        ):

            continue


        r = (
            np.log(
                d1 / d0
            )
            /
            dt
        )


        # ----------------------------------------------------
        # Observable state thresholds
        # ----------------------------------------------------

        if r > 0.001:

            state = "growth"

        elif r < -0.001:

            state = "shrinkage"

        else:

            state = "stable"


        state_parameter_records.append({

            "patient_id":
                patient_id,

            "lesion_alias":
                lesion_alias,

            "time_start_days":
                t0,

            "time_end_days":
                t1,

            "diameter_start_mm":
                d0,

            "diameter_end_mm":
                d1,

            "growth_rate":
                r,

            "dynamic_state":
                state
        })


state_parameter_table = pd.DataFrame(
    state_parameter_records
)


print(
    "State-labelled transitions:",
    len(
        state_parameter_table
    )
)


# ============================================================
# STATE PARAMETER SUMMARY
# ============================================================

header(
    "STATE-SPECIFIC DYNAMICAL PARAMETERS"
)


if len(state_parameter_table) > 0:

    state_summary = (
        state_parameter_table
        .groupby(
            "dynamic_state"
        )[
            "growth_rate"
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
                "min",
                "max"
            ]
        )
        .reindex(
            [
                "growth",
                "stable",
                "shrinkage"
            ]
        )
    )


    print(
        state_summary.round(
            6
        ).to_string()
    )

else:

    state_summary = pd.DataFrame()

    print(
        "No state-labelled transitions available."
    )


# ============================================================
# HBGD CORE EQUATION
# ============================================================

header(
    "HBGD CORE MATHEMATICAL FORMULATION"
)


print("""
The prototype HBGD dynamical formulation is:

    dD/dt = r(z, x, theta) * D

where:

    D     = observable tumor diameter
    z     = observable dynamical state
    x     = additional measurable features
    theta = patient/lesion parameters

For the current feasibility dataset:

    z = {growth, stable, shrinkage}

and the simplest approximation is:

    dD/dt = r_z * D

The model can later be extended to:

    dD/dt = f(D, V, t, z, theta)

without changing the overall architecture.
""")


# ============================================================
# GLOBAL FIT QUALITY
# ============================================================

header(
    "GLOBAL EXPONENTIAL FIT"
)


if len(trajectory_table) > 0:

    global_mae = np.mean(
        np.abs(
            trajectory_table[
                "diameter_error_mm"
            ]
        )
    )


    global_rmse = np.sqrt(
        np.mean(
            trajectory_table[
                "diameter_error_mm"
            ]
            ** 2
        )
    )


    print(
        "Global MAE:",
        round(
            global_mae,
            4
        ),
        "mm"
    )


    print(
        "Global RMSE:",
        round(
            global_rmse,
            4
        ),
        "mm"
    )

else:

    global_mae = np.nan
    global_rmse = np.nan

    print(
        "No fitted trajectories."
    )


# ============================================================
# PARAMETER VARIABILITY
# ============================================================

header(
    "PATIENT/LESION PARAMETER VARIABILITY"
)


if len(parameter_table) > 0:

    r_values = parameter_table[
        "growth_rate_r_per_day"
    ]


    print(
        "Number of estimated r parameters:",
        len(r_values)
    )


    print(
        "Mean r:",
        round(
            r_values.mean(),
            6
        )
    )


    print(
        "Median r:",
        round(
            r_values.median(),
            6
        )
    )


    print(
        "Minimum r:",
        round(
            r_values.min(),
            6
        )
    )


    print(
        "Maximum r:",
        round(
            r_values.max(),
            6
        )
    )


    print(
        "Standard deviation:",
        round(
            r_values.std(),
            6
        )
    )


# ============================================================
# INTERPRETATION
# ============================================================

header(
    "STEP 17 - INTERPRETATION"
)


print("""
This step converts the previously verified measurements
into the first mathematical HBGD prototype.

The prototype contains:

    1. Tumor diameter D
    2. Tumor volume V
    3. Time t
    4. Growth parameter r
    5. Observable dynamical state z

The initial dynamical equation is:

    dD/dt = rD

The state-aware extension is:

    dD/dt = r_z D

This is an interpretable mathematical framework.

It is NOT yet a biological mechanistic model.

It is NOT clinical validation.

It is NOT a universal cancer model.

The purpose of this step is to establish that the
observed dataset can be represented using the proposed
HBGD dynamical variables and equations.
""")


# ============================================================
# FEASIBILITY
# ============================================================

header(
    "STEP 17 - FINAL FEASIBILITY"
)


if len(trajectory_table) > 0:

    print(
        "[YES] Longitudinal diameter trajectories"
        " can be represented mathematically"
    )

else:

    print(
        "[NO] No longitudinal diameter trajectories"
    )


if len(parameter_table) > 0:

    print(
        "[YES] Patient/lesion-specific growth parameters"
        " can be estimated"
    )

else:

    print(
        "[NO] Growth parameters could not be estimated"
    )


if len(state_parameter_table) > 0:

    print(
        "[YES] Observable dynamic states can be"
        " incorporated into the equation"
    )

else:

    print(
        "[NO] Observable dynamic states unavailable"
    )


if (
    len(volume_parameter_table)
    > 0
):

    print(
        "[YES] Volume dynamics can be represented"
    )

else:

    print(
        "[NO] Volume dynamics unavailable"
    )


# ============================================================
# SAVE
# ============================================================

header(
    "SAVING STEP 17 RESULTS"
)


trajectory_file = os.path.join(
    DATA_DIR,
    "17_hbgd_diameter_trajectories.csv"
)


parameter_file = os.path.join(
    DATA_DIR,
    "17_hbgd_growth_parameters.csv"
)


volume_file = os.path.join(
    DATA_DIR,
    "17_hbgd_volume_parameters.csv"
)


state_file = os.path.join(
    DATA_DIR,
    "17_hbgd_state_parameters.csv"
)


feature_file = os.path.join(
    DATA_DIR,
    "17_hbgd_mathematical_features.csv"
)


trajectory_table.to_csv(
    trajectory_file,
    index=False
)


parameter_table.to_csv(
    parameter_file,
    index=False
)


volume_parameter_table.to_csv(
    volume_file,
    index=False
)


state_parameter_table.to_csv(
    state_file,
    index=False
)


features.to_csv(
    feature_file,
    index=False
)


print(
    "Saved:",
    trajectory_file
)

print(
    "Saved:",
    parameter_file
)

print(
    "Saved:",
    volume_file
)

print(
    "Saved:",
    state_file
)

print(
    "Saved:",
    feature_file
)


# ============================================================
# COMPLETE
# ============================================================

header(
    "STEP 17 COMPLETE"
)


print("""
Run:

    python 17_build_hbgd_mathematical_model.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A NEURAL NETWORK YET.
""")


print(
    "=" * 95
)

print(
    "END OF STEP 17"
)

print(
    "=" * 95
)