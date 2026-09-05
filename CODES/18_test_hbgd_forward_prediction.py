# 18_test_hbgd_forward_prediction.py

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
# LOAD
# ============================================================

header("STEP 18 - HBGD FORWARD PREDICTION")

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
    "volume_cm3",
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

for c in [
    "patient_id",
    "diameter_mm",
    "volume_cm3",
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
    (df["volume_cm3"] > 0) &
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

    if len(group) < 2:
        continue

    trajectories.append(
        (
            patient_id,
            alias,
            group
        )
    )

print(
    "Trajectories with >=2 observations:",
    len(trajectories)
)


# ============================================================
# FORWARD PREDICTION
# ============================================================

header(
    "FORWARD PREDICTION FROM FIRST OBSERVATION"
)

print("""
For each lesion:

    Observation 1
          ↓
      D0, V0
          ↓
   estimate growth rate
          ↓
     HBGD equation
          ↓
    predict future D(t)
          ↓
 compare with observed D(t)

The first observation is NEVER used as a
future prediction target.

Only information available at the initial
time point is used to estimate the initial
growth parameter.
""")


prediction_records = []


for patient_id, alias, group in trajectories:

    t = group[
        "days_from_baseline"
    ].values.astype(float)

    diameter = group[
        "diameter_mm"
    ].values.astype(float)

    volume = group[
        "volume_cm3"
    ].values.astype(float)


    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    t0 = t[0]

    D0 = diameter[0]

    V0 = volume[0]


    # --------------------------------------------------------
    # Need at least 3 observations to estimate growth from
    # the first transition and predict a later observation.
    # --------------------------------------------------------

    if len(group) < 3:
        continue


    t1 = t[1]
    D1 = diameter[1]

    dt01 = t1 - t0

    if dt01 <= 0:
        continue


    # --------------------------------------------------------
    # Estimate initial exponential growth rate
    #
    # D(t) = D0 * exp(r*t)
    #
    # r = ln(D1/D0) / dt
    # --------------------------------------------------------

    if D0 <= 0 or D1 <= 0:
        continue


    r_initial = (
        np.log(D1 / D0)
        /
        dt01
    )


    # --------------------------------------------------------
    # Estimate volume growth rate from first transition
    # --------------------------------------------------------

    if V0 > 0 and V0 != V0 or volume[1] <= 0:
        rv_initial = np.nan

    else:

        rv_initial = (
            np.log(
                volume[1] / V0
            )
            /
            dt01
        )


    # --------------------------------------------------------
    # Predict all later observations
    # --------------------------------------------------------

    for i in range(
        1,
        len(group)
    ):

        current_t = t[i]

        observed_D = diameter[i]

        observed_V = volume[i]


        delta_t = (
            current_t
            -
            t0
        )


        predicted_D = (
            D0
            *
            np.exp(
                r_initial
                *
                delta_t
            )
        )


        if np.isfinite(
            rv_initial
        ):

            predicted_V = (
                V0
                *
                np.exp(
                    rv_initial
                    *
                    delta_t
                )
            )

        else:

            predicted_V = np.nan


        diameter_error = (
            observed_D
            -
            predicted_D
        )


        if predicted_D != 0:

            diameter_percent_error = (
                diameter_error
                /
                predicted_D
                *
                100.0
            )

        else:

            diameter_percent_error = np.nan


        if np.isfinite(
            predicted_V
        ):

            volume_error = (
                observed_V
                -
                predicted_V
            )

        else:

            volume_error = np.nan


        prediction_records.append({

            "patient_id":
                patient_id,

            "lesion_alias":
                alias,

            "baseline_day":
                t0,

            "prediction_day":
                current_t,

            "days_ahead":
                delta_t,

            "baseline_diameter_mm":
                D0,

            "observed_diameter_mm":
                observed_D,

            "predicted_diameter_mm":
                predicted_D,

            "diameter_error_mm":
                diameter_error,

            "absolute_diameter_error_mm":
                abs(diameter_error),

            "diameter_percent_error":
                diameter_percent_error,

            "initial_growth_rate_per_day":
                r_initial,

            "baseline_volume_cm3":
                V0,

            "observed_volume_cm3":
                observed_V,

            "predicted_volume_cm3":
                predicted_V,

            "volume_error_cm3":
                volume_error
        })


predictions = pd.DataFrame(
    prediction_records
)


print(
    "Forward prediction observations:",
    len(predictions)
)


if len(predictions) == 0:

    print()
    print(
        "ERROR: No forward prediction observations."
    )

    print(
        "At least three observations per lesion"
        " are required for this experiment."
    )

    sys.exit(1)


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

header("HBGD FORWARD PREDICTION PERFORMANCE")


mae = predictions[
    "absolute_diameter_error_mm"
].mean()


rmse = np.sqrt(
    np.mean(
        predictions[
            "diameter_error_mm"
        ] ** 2
    )
)


median_error = predictions[
    "absolute_diameter_error_mm"
].median()


print(
    "Forward observations:",
    len(predictions)
)

print(
    "MAE:",
    round(mae, 4),
    "mm"
)

print(
    "RMSE:",
    round(rmse, 4),
    "mm"
)

print(
    "Median absolute error:",
    round(
        median_error,
        4
    ),
    "mm"
)


# ============================================================
# NO-CHANGE BASELINE
# ============================================================

header("COMPARISON - NO-CHANGE BASELINE")


predictions[
    "baseline_prediction_mm"
] = predictions[
    "baseline_diameter_mm"
]


predictions[
    "baseline_error_mm"
] = (
    predictions[
        "observed_diameter_mm"
    ]
    -
    predictions[
        "baseline_prediction_mm"
    ]
)


baseline_mae = np.mean(
    np.abs(
        predictions[
            "baseline_error_mm"
        ]
    )
)


baseline_rmse = np.sqrt(
    np.mean(
        predictions[
            "baseline_error_mm"
        ] ** 2
    )
)


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
            mae
        )
        /
        baseline_mae
        *
        100.0
    )

else:

    improvement = np.nan


print(
    "HBGD MAE improvement:",
    round(
        improvement,
        2
    ),
    "%"
)


# ============================================================
# PERFORMANCE BY FORECAST HORIZON
# ============================================================

header(
    "ERROR BY FORECAST HORIZON"
)


horizon_table = (
    predictions
    .groupby(
        "days_ahead"
    )[
        "absolute_diameter_error_mm"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "max"
        ]
    )
    .reset_index()
)


print(
    horizon_table.round(
        3
    ).to_string(
        index=False
    )
)


# ============================================================
# PERFORMANCE BY LESION
# ============================================================

header(
    "ERROR BY LESION TRAJECTORY"
)


lesion_table = (
    predictions
    .groupby(
        [
            "patient_id",
            "lesion_alias"
        ]
    )
    .agg(
        observations=(
            "observed_diameter_mm",
            "count"
        ),
        MAE_mm=(
            "absolute_diameter_error_mm",
            "mean"
        ),
        RMSE_mm=(
            "diameter_error_mm",
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
    lesion_table.round(
        3
    ).to_string(
        index=False
    )
)


# ============================================================
# GROWTH VS SHRINKAGE
# ============================================================

header(
    "FORWARD PREDICTION BY INITIAL DYNAMICAL DIRECTION"
)


def direction(rate):

    if rate > 0.001:
        return "growth"

    if rate < -0.001:
        return "shrinkage"

    return "stable"


predictions[
    "initial_state"
] = predictions[
    "initial_growth_rate_per_day"
].apply(
    direction
)


direction_table = (
    predictions
    .groupby(
        "initial_state"
    )
    .agg(
        observations=(
            "observed_diameter_mm",
            "count"
        ),
        MAE_mm=(
            "absolute_diameter_error_mm",
            "mean"
        ),
        mean_growth_rate=(
            "initial_growth_rate_per_day",
            "mean"
        )
    )
    .reset_index()
)


print(
    direction_table.round(
        5
    ).to_string(
        index=False
    )
)


# ============================================================
# PARAMETER STABILITY
# ============================================================

header(
    "INITIAL GROWTH PARAMETER DISTRIBUTION"
)


rates = predictions[
    [
        "patient_id",
        "lesion_alias",
        "initial_growth_rate_per_day"
    ]
].drop_duplicates()


print(
    "Trajectories:",
    len(rates)
)

print(
    "Mean initial r:",
    round(
        rates[
            "initial_growth_rate_per_day"
        ].mean(),
        6
    )
)

print(
    "Median initial r:",
    round(
        rates[
            "initial_growth_rate_per_day"
        ].median(),
        6
    )
)

print(
    "Minimum initial r:",
    round(
        rates[
            "initial_growth_rate_per_day"
        ].min(),
        6
    )
)

print(
    "Maximum initial r:",
    round(
        rates[
            "initial_growth_rate_per_day"
        ].max(),
        6
    )
)


# ============================================================
# INTERPRETATION
# ============================================================

header("STEP 18 - INTERPRETATION")

print("""
This is the first genuine forward-prediction experiment.

For every lesion trajectory:

    first observation
          ↓
       D0, V0
          ↓
    estimate initial r
          ↓
    HBGD equation
          ↓
    predict future diameter
          ↓
    compare with later observation

Unlike Step 17, future observations are NOT used
to estimate the initial growth parameter.

Therefore this is a substantially more meaningful
test of the mathematical HBGD formulation.

However, this remains an exploratory feasibility
experiment because the dataset is small.

It does NOT establish clinical validity.
It does NOT establish biological causality.
""")


# ============================================================
# FEASIBILITY
# ============================================================

header("STEP 18 - FINAL FEASIBILITY")

print(
    "[YES] Future observations available for"
    " forward prediction"
)

if mae < baseline_mae:

    print(
        "[YES] HBGD exponential prediction beats"
        " the no-change baseline"
    )

else:

    print(
        "[NO] HBGD exponential prediction does NOT"
        " beat the no-change baseline"
    )


if len(predictions) >= 10:

    print(
        "[YES] Sufficient forward observations for"
        " exploratory evaluation"
    )

else:

    print(
        "[WARNING] Very small number of forward"
        " observations"
    )


# ============================================================
# SAVE
# ============================================================

header("SAVING STEP 18 RESULTS")


prediction_file = os.path.join(
    DATA_DIR,
    "18_hbgd_forward_predictions.csv"
)

horizon_file = os.path.join(
    DATA_DIR,
    "18_hbgd_forecast_horizon_analysis.csv"
)

lesion_file = os.path.join(
    DATA_DIR,
    "18_hbgd_forward_lesion_performance.csv"
)

direction_file = os.path.join(
    DATA_DIR,
    "18_hbgd_forward_state_performance.csv"
)


predictions.to_csv(
    prediction_file,
    index=False
)

horizon_table.to_csv(
    horizon_file,
    index=False
)

lesion_table.to_csv(
    lesion_file,
    index=False
)

direction_table.to_csv(
    direction_file,
    index=False
)


print(
    "Saved:",
    prediction_file
)

print(
    "Saved:",
    horizon_file
)

print(
    "Saved:",
    lesion_file
)

print(
    "Saved:",
    direction_file
)


# ============================================================
# COMPLETE
# ============================================================

header("STEP 18 COMPLETE")

print("""
Run:

    python 18_test_hbgd_forward_prediction.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A NEURAL NETWORK YET.
""")

print("=" * 95)
print("END OF STEP 18")
print("=" * 95)