import os
import sys
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    "22_rate_predictions.csv"
)

PERFORMANCE_FILE = os.path.join(
    DATA_DIR,
    "22_rate_performance.csv"
)

PATIENT_FILE = os.path.join(
    DATA_DIR,
    "22_rate_patient_performance.csv"
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

header("STEP 22 - IMAGING FEATURES TO INDIVIDUALIZED HBGD RATE")

if not os.path.exists(FEATURE_FILE):
    print("ERROR: File not found:")
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
    print("ERROR: Missing required columns:")
    for c in missing:
        print("   ", c)
    sys.exit(1)

print("All required variables are present.")


# ============================================================
# AVAILABLE IMAGING / GEOMETRIC VARIABLES
# ============================================================

header("IDENTIFYING AVAILABLE IMAGING FEATURES")

candidate_features = [
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
    "volume_percent_from_baseline"
]

available_features = [
    c for c in candidate_features
    if c in df.columns
]

print("Available candidate features:")

for c in available_features:
    print("   ", c)

if len(available_features) == 0:
    print("ERROR: No usable imaging/geometry features found.")
    sys.exit(1)


# ============================================================
# NORMALIZE DATA
# ============================================================

header("NORMALIZING DATA")

numeric_columns = [
    "patient_id",
    "diameter_mm",
    "days_from_baseline"
] + available_features

numeric_columns = list(
    dict.fromkeys(numeric_columns)
)

for c in numeric_columns:

    df[c] = pd.to_numeric(
        df[c],
        errors="coerce"
    )


df["lesion_label_alias"] = (
    df["lesion_label_alias"]
    .astype(str)
    .str.strip()
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
# BUILD LESION TRAJECTORIES
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
# PATIENT-LEVEL HOLDOUT
# ============================================================

header("CREATING PATIENT-LEVEL HOLDOUT")

# Same deterministic split used by Steps 20 and 21.
# This keeps comparisons consistent.

test_patients = patients[-3:]

train_patients = [
    p for p in patients
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
# CREATE RATE TARGETS
# ============================================================

header("CREATING INDIVIDUALIZED RATE TARGET")

records = []

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


    # Initial individualized dynamical parameter
    r = (
        np.log(
            D[1] / D[0]
        )
        /
        dt
    )


    row = {

        "patient_id":
            item["patient_id"],

        "lesion_alias":
            item["lesion_alias"],

        "target_r":
            r,

        "initial_diameter_mm":
            D[0],

        "initial_time_days":
            t[0],

        "initial_interval_days":
            dt
    }


    # IMPORTANT:
    # Only the FIRST observation is used as input.
    # The future observations remain prediction targets.

    first_row = group.iloc[0]

    for feature in available_features:

        row[
            feature
        ] = first_row[
            feature
        ]


    records.append(row)


rate_df = pd.DataFrame(records)


print(
    "Rate-target records:",
    len(rate_df)
)


# ============================================================
# REMOVE FEATURES THAT ARE NOT AVAILABLE
# ============================================================

usable_features = []

for feature in available_features:

    if feature not in rate_df.columns:
        continue

    if rate_df[
        feature
    ].notna().sum() == 0:
        continue

    usable_features.append(
        feature
    )


print()
print(
    "Usable predictor features:"
)

for feature in usable_features:
    print(
        "   ",
        feature
    )


# ============================================================
# IMPUTE FEATURE VALUES
# ============================================================

header("PREPARING IMAGING FEATURES")

X = rate_df[
    usable_features
].copy()

y = rate_df[
    "target_r"
].copy()


# Median imputation using TRAINING PATIENTS only
train_mask = rate_df[
    "patient_id"
].isin(
    train_patients
)

train_feature_medians = {}

for feature in usable_features:

    median_value = X.loc[
        train_mask,
        feature
    ].median()

    if pd.isna(median_value):
        median_value = 0.0

    train_feature_medians[
        feature
    ] = median_value

    X[feature] = X[
        feature
    ].fillna(
        median_value
    )


rate_df[
    usable_features
] = X


print(
    "Feature matrix shape:",
    X.shape
)


# ============================================================
# TRAIN / TEST
# ============================================================

train_mask = rate_df[
    "patient_id"
].isin(
    train_patients
)

test_mask = rate_df[
    "patient_id"
].isin(
    test_patients
)


X_train = rate_df.loc[
    train_mask,
    usable_features
]

y_train = rate_df.loc[
    train_mask,
    "target_r"
]

X_test = rate_df.loc[
    test_mask,
    usable_features
]

y_test = rate_df.loc[
    test_mask,
    "target_r"
]


print(
    "Training rate records:",
    len(X_train)
)

print(
    "Testing rate records:",
    len(X_test)
)


# ============================================================
# MODEL A - GLOBAL RATE
# ============================================================

header("MODEL A - GLOBAL RATE")

global_rate = np.median(
    y_train
)

print(
    "Training global rate:",
    round(
        global_rate,
        6
    )
)


# ============================================================
# MODEL B - RIDGE RATE MODEL
# ============================================================

header("MODEL B - RIDGE IMAGING-TO-RATE MODEL")

ridge_model = Pipeline(
    [
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=1.0
            )
        )
    ]
)


ridge_model.fit(
    X_train,
    y_train
)


ridge_predictions = ridge_model.predict(
    X_test
)


print(
    "Ridge model trained."
)


# ============================================================
# MODEL C - RANDOM FOREST RATE MODEL
# ============================================================

header("MODEL C - RANDOM FOREST IMAGING-TO-RATE MODEL")

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=3,
    min_samples_leaf=2,
    random_state=42
)


rf_model.fit(
    X_train,
    y_train
)


rf_predictions = rf_model.predict(
    X_test
)


print(
    "Random forest model trained."
)


# ============================================================
# RATE PREDICTION PERFORMANCE
# ============================================================

header("RATE PREDICTION PERFORMANCE")

rate_results = []

models = {

    "global_rate":
        np.full(
            len(y_test),
            global_rate
        ),

    "ridge_imaging":
        ridge_predictions,

    "random_forest_imaging":
        rf_predictions
}


for model_name, predictions in models.items():

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    try:

        r2 = r2_score(
            y_test,
            predictions
        )

    except Exception:

        r2 = np.nan


    rate_results.append({

        "model":
            model_name,

        "observations":
            len(y_test),

        "MAE_rate":
            mae,

        "RMSE_rate":
            rmse,

        "R2":
            r2
    })


rate_performance = pd.DataFrame(
    rate_results
)


print(
    rate_performance.round(
        6
    ).to_string(
        index=False
    )
)


# ============================================================
# FORWARD DIAMETER PREDICTION
# ============================================================

header("FORWARD HBGD PREDICTION USING PREDICTED r")

print("""
For each held-out lesion:

    baseline CT-derived features
              |
              v
       predicted r
              |
              v
    D(t) = D0 * exp(r*t)
              |
              v
       future diameter

Future observations are NOT used as inputs.
""")


prediction_records = []


# Map predicted r back to test rate records
test_rate = rate_df[
    test_mask
].copy().reset_index(drop=True)


test_rate[
    "ridge_r"
] = ridge_predictions

test_rate[
    "random_forest_r"
] = rf_predictions

test_rate[
    "global_r"
] = global_rate


# Each trajectory is used to evaluate future observations
for _, rate_row in test_rate.iterrows():

    patient_id = rate_row[
        "patient_id"
    ]

    alias = rate_row[
        "lesion_alias"
    ]


    matching = [
        item for item in trajectories
        if (
            item["patient_id"] == patient_id
            and
            item["lesion_alias"] == alias
        )
    ]


    if len(matching) == 0:
        continue


    group = matching[0]["data"]


    t = group[
        "days_from_baseline"
    ].values.astype(float)

    D = group[
        "diameter_mm"
    ].values.astype(float)


    D0 = D[0]
    t0 = t[0]


    if D0 <= 0:
        continue


    for i in range(
        2,
        len(group)
    ):

        delta_t = (
            t[i]
            -
            t0
        )

        observed = D[i]


        pred_global = (
            D0
            *
            np.exp(
                rate_row["global_r"]
                *
                delta_t
            )
        )


        pred_ridge = (
            D0
            *
            np.exp(
                rate_row["ridge_r"]
                *
                delta_t
            )
        )


        pred_rf = (
            D0
            *
            np.exp(
                rate_row["random_forest_r"]
                *
                delta_t
            )
        )


        prediction_records.append({

            "patient_id":
                patient_id,

            "lesion_alias":
                alias,

            "days_ahead":
                delta_t,

            "baseline_diameter_mm":
                D0,

            "observed_diameter_mm":
                observed,

            "true_r":
                rate_row["target_r"],

            "global_r":
                rate_row["global_r"],

            "ridge_predicted_r":
                rate_row["ridge_r"],

            "random_forest_predicted_r":
                rate_row["random_forest_r"],

            "pred_global":
                pred_global,

            "pred_ridge":
                pred_ridge,

            "pred_random_forest":
                pred_rf,

            "error_global":
                observed - pred_global,

            "error_ridge":
                observed - pred_ridge,

            "error_random_forest":
                observed - pred_rf
        })


predictions = pd.DataFrame(
    prediction_records
)


print(
    "Future prediction observations:",
    len(predictions)
)


if len(predictions) == 0:
    print(
        "ERROR: No future observations available."
    )
    sys.exit(1)


# ============================================================
# FORWARD PERFORMANCE
# ============================================================

header("FORWARD PREDICTION PERFORMANCE")

forward_models = {

    "global_exponential":
        "error_global",

    "ridge_imaging_rate":
        "error_ridge",

    "random_forest_imaging_rate":
        "error_random_forest"
}


forward_rows = []


for model_name, error_column in forward_models.items():

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


    forward_rows.append({

        "model":
            model_name,

        "observations":
            len(errors),

        "MAE_mm":
            mae,

        "RMSE_mm":
            rmse
    })


forward_performance = pd.DataFrame(
    forward_rows
)


print(
    forward_performance.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# INDIVIDUALIZED RATE REFERENCE
# ============================================================

header("REFERENCE - DIRECT INDIVIDUALIZED RATE")

print("""
This is NOT a deployable model.

It is an upper-reference experiment where the
held-out lesion's first transition is directly
used to calculate:

    r = log(D1/D0) / Δt

It tells us how much predictive information exists
when individualized early dynamics are known.
""")


direct_rates = []

for item in trajectories:

    if item["patient_id"] not in test_patients:
        continue

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


    direct_r = (
        np.log(
            D[1] / D[0]
        )
        /
        dt
    )


    for i in range(
        2,
        len(group)
    ):

        delta_t = (
            t[i] - t[0]
        )

        pred = (
            D[0]
            *
            np.exp(
                direct_r
                *
                delta_t
            )
        )

        direct_rates.append(
            {
                "patient_id":
                    item["patient_id"],

                "lesion_alias":
                    item["lesion_alias"],

                "observed":
                    D[i],

                "predicted":
                    pred
            }
        )


direct_df = pd.DataFrame(
    direct_rates
)


if len(direct_df) > 0:

    direct_mae = mean_absolute_error(
        direct_df["observed"],
        direct_df["predicted"]
    )

    direct_rmse = np.sqrt(
        mean_squared_error(
            direct_df["observed"],
            direct_df["predicted"]
        )
    )

else:

    direct_mae = np.nan
    direct_rmse = np.nan


print(
    "Direct individualized-rate MAE:",
    round(
        direct_mae,
        4
    )
)

print(
    "Direct individualized-rate RMSE:",
    round(
        direct_rmse,
        4
    )
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


    for model_name, error_column in forward_models.items():

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
# FEATURE IMPORTANCE
# ============================================================

header("RANDOM FOREST FEATURE IMPORTANCE")

importance = pd.DataFrame(
    {
        "feature":
            usable_features,

        "importance":
            rf_model.feature_importances_
    }
).sort_values(
    "importance",
    ascending=False
)


print(
    importance.round(
        6
    ).to_string(
        index=False
    )
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

header("STEP 22 - INTERPRETATION")

print("""
This experiment tests the critical bridge:

    CT / lesion geometry
            |
            v
    imaging-derived features
            |
            v
      predicted growth rate
            |
            v
       HBGD equation
            |
            v
    future tumor diameter

Two feature-to-rate models are evaluated:

    1. Ridge regression
    2. Random forest regression

The global exponential model is used as a
population-level baseline.

The direct individualized-rate model is included
only as a reference showing the predictive value
available when the early lesion transition is known.

IMPORTANT:

The current features are derived from the segmented
lesion geometry and RECIST measurements.

They are NOT deep CT image embeddings.

Therefore this experiment does NOT yet validate
a CNN or 3D imaging encoder.

It tests whether measurable lesion characteristics
contain enough signal to estimate individualized
dynamical parameters.
""")


# ============================================================
# FINAL DECISION
# ============================================================

header("STEP 22 - FINAL FEASIBILITY")

global_mae = forward_performance.loc[
    forward_performance["model"]
    ==
    "global_exponential",
    "MAE_mm"
].iloc[0]


ridge_mae = forward_performance.loc[
    forward_performance["model"]
    ==
    "ridge_imaging_rate",
    "MAE_mm"
].iloc[0]


rf_mae = forward_performance.loc[
    forward_performance["model"]
    ==
    "random_forest_imaging_rate",
    "MAE_mm"
].iloc[0]


if (
    ridge_mae < global_mae
    or
    rf_mae < global_mae
):

    print(
        "[YES] At least one imaging-feature model "
        "beats the global exponential baseline"
    )

else:

    print(
        "[NO] Imaging-feature models do not beat "
        "the global exponential baseline"
    )


if (
    ridge_mae < direct_mae
    or
    rf_mae < direct_mae
):

    print(
        "[YES] Feature model approaches or exceeds "
        "the direct individualized-rate reference"
    )

else:

    print(
        "[NO] Feature models remain below the "
        "direct individualized-rate reference"
    )


print(
    "[YES] Patient-independent holdout used"
)

print(
    "[YES] Future observations kept as prediction targets"
)

print(
    "[YES] No neural network used"
)


# ============================================================
# SAVE
# ============================================================

header("SAVING STEP 22 RESULTS")

predictions.to_csv(
    PREDICTION_FILE,
    index=False
)

forward_performance.to_csv(
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

header("STEP 22 COMPLETE")

print("""
Run:

    python 22_predict_individualized_hbgd_rate.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A CNN YET.
""")

print("=" * 95)
print("END OF STEP 22")
print("=" * 95)