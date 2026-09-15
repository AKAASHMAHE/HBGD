# -*- coding: utf-8 -*-

"""
STEP 19
CROSS-CANCER TRANSFER OF HBGD-DERIVED GROWTH REPRESENTATIONS

Purpose
-------
Evaluate whether a growth-dynamics representation learned from the
validated source HBGD cohort can transfer to LUMIERE GBM progression
prediction.

IMPORTANT
---------
1. Source HBGD intervals come ONLY from:
      17_hbgd_state_parameters.csv

   This is the validated 49-interval source cohort identified during
   the Step 19 source audit.

2. LUMIERE uses the exact 13-patient held-out test cohort established
   by Step 13.

3. Target prediction features use only information available before
   the future observation.

4. No future diameter/volume is used as an input feature.

5. Test patients are never used for fitting, calibration, or model
   selection.

Python:
    C:\\Users\\Akaash MK\\AppData\\Local\\Programs\\Python\\Python36-32\\python.exe
"""

import os
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA"
LUMIERE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"

if not os.path.exists(LUMIERE_DIR):
    os.makedirs(LUMIERE_DIR)


SOURCE_STATE = os.path.join(
    BASE_DIR,
    "17_hbgd_state_parameters.csv"
)

TARGET_TRAJECTORIES = os.path.join(
    LUMIERE_DIR,
    "476_lumiere_longitudinal_trajectories.csv"
)

# Fallback names in case Step 3 used a different trajectory filename.
TARGET_TRAJECTORY_ALTERNATIVES = [
    os.path.join(
        LUMIERE_DIR,
        "3_lumiere_longitudinal_trajectories.csv"
    ),
    os.path.join(
        LUMIERE_DIR,
        "lumiere_longitudinal_trajectories.csv"
    ),
    os.path.join(
        LUMIERE_DIR,
        "step3_lumiere_longitudinal_trajectories.csv"
    )
]

STEP13_PATIENT_RESULTS = os.path.join(
    LUMIERE_DIR,
    "step13_patient_level_hbgd_results.csv"
)


# ============================================================
# EXACT STEP-13 TEST COHORT
# ============================================================

TEST_PATIENTS = [
    "Patient-010",
    "Patient-012",
    "Patient-019",
    "Patient-022",
    "Patient-033",
    "Patient-037",
    "Patient-038",
    "Patient-042",
    "Patient-059",
    "Patient-064",
    "Patient-065",
    "Patient-073",
    "Patient-085"
]

RANDOM_SEED = 42


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def normalize_patient_id(x):
    """
    Make source and target patient IDs comparable internally.
    """
    s = str(x).strip()

    if s.lower().startswith("patient-"):
        return s

    try:
        return "Patient-%03d" % int(float(s))
    except Exception:
        return s


def find_target_trajectory_file():
    if os.path.exists(TARGET_TRAJECTORIES):
        return TARGET_TRAJECTORIES

    for path in TARGET_TRAJECTORY_ALTERNATIVES:
        if os.path.exists(path):
            return path

    # Last-resort discovery
    candidates = []

    for name in os.listdir(LUMIERE_DIR):
        low = name.lower()

        if (
            low.endswith(".csv")
            and "trajectory" in low
            and "lumiere" in low
        ):
            candidates.append(os.path.join(LUMIERE_DIR, name))

    if len(candidates) == 1:
        return candidates[0]

    raise IOError(
        "Could not find LUMIERE longitudinal trajectory CSV.\n"
        "Expected one of:\n%s"
        % "\n".join(
            [TARGET_TRAJECTORIES] +
            TARGET_TRAJECTORY_ALTERNATIVES
        )
    )


def safe_numeric(df, columns):
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

    return df


# ============================================================
# LOAD SOURCE HBGD
# ============================================================

print("=" * 72)
print("STEP 19 CROSS-CANCER HBGD REPRESENTATION TRANSFER")
print("=" * 72)

print("\nLoading validated source HBGD state parameters...")

if not os.path.exists(SOURCE_STATE):
    raise IOError(
        "Source HBGD file not found:\n%s"
        % SOURCE_STATE
    )

source = pd.read_csv(SOURCE_STATE)

print("Source rows:", len(source))
print(
    "Source unique patients:",
    source["patient_id"].nunique()
)

required_source = [
    "patient_id",
    "time_start_days",
    "time_end_days",
    "diameter_start_mm",
    "diameter_end_mm",
    "growth_rate",
    "dynamic_state"
]

for c in required_source:
    if c not in source.columns:
        raise ValueError(
            "Missing source column: %s" % c
        )

source["patient_id"] = source["patient_id"].apply(
    normalize_patient_id
)

source = safe_numeric(
    source,
    [
        "time_start_days",
        "time_end_days",
        "diameter_start_mm",
        "diameter_end_mm",
        "growth_rate"
    ]
)

source["interval_days"] = (
    source["time_end_days"] -
    source["time_start_days"]
)

source = source[
    (source["interval_days"] > 0) &
    source["diameter_start_mm"].notna() &
    source["diameter_end_mm"].notna() &
    source["growth_rate"].notna()
].copy()

print(
    "Validated source intervals used:",
    len(source)
)

print(
    "Validated source patients used:",
    source["patient_id"].nunique()
)


# ============================================================
# SOURCE HBGD REPRESENTATION
# ============================================================

print("\nConstructing source HBGD representation...")

source["log_diameter_start"] = np.log(
    source["diameter_start_mm"].clip(lower=0.001)
)

source["log_interval_days"] = np.log(
    source["interval_days"].clip(lower=1.0)
)

source["diameter_change_mm"] = (
    source["diameter_end_mm"] -
    source["diameter_start_mm"]
)

source["abs_growth_rate"] = np.abs(
    source["growth_rate"]
)

SOURCE_FEATURES = [
    "diameter_start_mm",
    "log_diameter_start",
    "interval_days",
    "log_interval_days"
]

X_source = source[SOURCE_FEATURES].values
y_source = source["growth_rate"].values


# ============================================================
# SOURCE STANDARDIZATION
# ============================================================

source_scaler = StandardScaler()

X_source_scaled = source_scaler.fit_transform(
    X_source
)


# ============================================================
# SOURCE MODEL
# ============================================================

print("\nTraining source HBGD growth-rate model...")

source_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=2,
    random_state=RANDOM_SEED
)

source_model.fit(
    X_source_scaled,
    y_source
)

source_pred = source_model.predict(
    X_source_scaled
)

source_train_mae = mean_absolute_error(
    y_source,
    source_pred
)

source_train_rmse = rmse(
    y_source,
    source_pred
)

print(
    "Source model training MAE:",
    "%.8f" % source_train_mae
)

print(
    "Source model training RMSE:",
    "%.8f" % source_train_rmse
)


# ============================================================
# LOAD LUMIERE TRAJECTORIES
# ============================================================

trajectory_file = find_target_trajectory_file()

print("\nLoading LUMIERE trajectories:")
print(trajectory_file)

target = pd.read_csv(
    trajectory_file
)

print(
    "Target trajectory rows:",
    len(target)
)

print(
    "Target patients:",
    target["patient_id"].nunique()
)

print("\nTarget columns:")

for c in target.columns:
    print("  " + c)


# ============================================================
# TARGET COLUMN NORMALIZATION
# ============================================================

target["patient_id"] = target["patient_id"].apply(
    normalize_patient_id
)

# Possible Step-3 naming variants
diameter_candidates = [
    "equivalent_diameter_mm",
    "diameter_mm",
    "observed_diameter_mm"
]

volume_candidates = [
    "volume_cm3",
    "current_volume_cm3"
]

time_candidates = [
    "time_days",
    "days_from_baseline",
    "study_time_days",
    "week"
]

diameter_col = None

for c in diameter_candidates:
    if c in target.columns:
        diameter_col = c
        break

if diameter_col is None:
    raise ValueError(
        "Could not identify target diameter column."
    )

time_col = None

for c in time_candidates:
    if c in target.columns:
        time_col = c
        break

if time_col is None:
    raise ValueError(
        "Could not identify target time column."
    )

volume_col = None

for c in volume_candidates:
    if c in target.columns:
        volume_col = c
        break

target = safe_numeric(
    target,
    [diameter_col, time_col] +
    ([volume_col] if volume_col else [])
)
# LUMIERE uses week as the longitudinal time index.
# Convert weeks to elapsed days for HBGD rate calculations.
if time_col == "week":
    target["time_days"] = (
        target["week"] * 7.0
    )
    time_col = "time_days"
target = target[
    target[diameter_col].notna() &
    target[time_col].notna()
].copy()


# ============================================================
# SORT LONGITUDINAL DATA
# ============================================================

target = target.sort_values(
    [
        "patient_id",
        time_col
    ]
).reset_index(drop=True)


# ============================================================
# BUILD PRE-PREDICTION TARGET CASES
# ============================================================

print("\nConstructing leakage-controlled LUMIERE prediction cases...")

cases = []

for patient_id, group in target.groupby(
    "patient_id"
):

    group = group.sort_values(
        time_col
    ).reset_index(drop=True)

    if len(group) < 2:
        continue

    history_rates = []

    for i in range(
        1,
        len(group)
    ):

        previous_row = group.iloc[i - 1]
        current_row = group.iloc[i]

        previous_diameter = float(
            previous_row[diameter_col]
        )

        current_diameter = float(
            current_row[diameter_col]
        )

        previous_time = float(
            previous_row[time_col]
        )

        current_time = float(
            current_row[time_col]
        )

        interval_days = (
            current_time -
            previous_time
        )

        if (
            interval_days <= 0
            or previous_diameter <= 0
            or current_diameter <= 0
        ):
            continue

        # Historical rate becomes available ONLY after
        # observing the current timepoint.
        historical_rate = (
            np.log(
                current_diameter /
                previous_diameter
            ) /
            interval_days
        )

        history_rates.append(
            historical_rate
        )

        # Need a future observation for evaluation.
        if i + 1 >= len(group):
            continue

        future_row = group.iloc[i + 1]

        future_time = float(
            future_row[time_col]
        )

        future_diameter = float(
            future_row[diameter_col]
        )

        future_days = (
            future_time -
            current_time
        )

        if (
            future_days <= 0
            or future_diameter <= 0
        ):
            continue

        # Current volume is optional.
        current_volume = np.nan

        if volume_col is not None:
            try:
                current_volume = float(
                    current_row[volume_col]
                )
            except Exception:
                current_volume = np.nan

        previous_volume = np.nan

        if volume_col is not None:
            try:
                previous_volume = float(
                    previous_row[volume_col]
                )
            except Exception:
                previous_volume = np.nan

        # ----------------------------------------------------
        # HISTORY-ONLY FEATURES
        # ----------------------------------------------------

        mean_rate = np.mean(
            history_rates
        )

        abs_mean_rate = np.mean(
            np.abs(history_rates)
        )

        rate_std = (
            np.std(history_rates)
            if len(history_rates) >= 2
            else 0.0
        )

        cases.append({
            "patient_id": patient_id,
            "current_time": current_time,
            "future_time": future_time,

            "current_diameter_mm":
                current_diameter,

            "previous_diameter_mm":
                previous_diameter,

            "current_volume_cm3":
                current_volume,

            "previous_volume_cm3":
                previous_volume,

            "diameter_change_mm":
                current_diameter -
                previous_diameter,

            "interval_days":
                interval_days,

            "future_days":
                future_days,

            "actual_future_diameter_mm":
                future_diameter,

            "history_mean_rate":
                mean_rate,

            "history_abs_mean_rate":
                abs_mean_rate,

            "history_rate_std":
                rate_std,

            "history_n_intervals":
                len(history_rates)
        })


cases = pd.DataFrame(cases)

print(
    "Prediction cases created:",
    len(cases)
)

print(
    "Prediction patients:",
    cases["patient_id"].nunique()
)


# ============================================================
# VERIFY EXACT TEST COHORT
# ============================================================

available_test_patients = [
    p for p in TEST_PATIENTS
    if p in set(
        cases["patient_id"].unique()
    )
]

missing_test = [
    p for p in TEST_PATIENTS
    if p not in available_test_patients
]

print("\nExact Step-13 test cohort:")
print(
    "Expected:",
    len(TEST_PATIENTS)
)

print(
    "Available:",
    len(available_test_patients)
)

if len(missing_test) > 0:
    print(
        "WARNING - missing test patients:",
        missing_test
    )


# ============================================================
# TARGET PATIENT SPLIT
# ============================================================

all_patients = sorted(
    cases["patient_id"].unique()
)

test_set = set(
    TEST_PATIENTS
)

remaining_patients = [
    p for p in all_patients
    if p not in test_set
]

rng = np.random.RandomState(
    RANDOM_SEED
)

remaining_shuffled = list(
    remaining_patients
)

rng.shuffle(
    remaining_shuffled
)

n_validation = int(
    round(
        0.125 *
        len(remaining_shuffled)
    )
)

if n_validation < 1:
    n_validation = 1

validation_patients = set(
    remaining_shuffled[
        :n_validation
    ]
)

train_patients = set(
    remaining_shuffled[
        n_validation:
    ]
)

test_patients = set(
    available_test_patients
)

print("\nTarget patient split:")

print(
    "Train:",
    len(train_patients)
)

print(
    "Validation:",
    len(validation_patients)
)

print(
    "Test:",
    len(test_patients)
)


# ============================================================
# ASSIGN SPLIT
# ============================================================

def get_split(pid):

    if pid in train_patients:
        return "train"

    if pid in validation_patients:
        return "validation"

    if pid in test_patients:
        return "test"

    return "excluded"


cases["split"] = cases[
    "patient_id"
].apply(get_split)

train_cases = cases[
    cases["split"] == "train"
].copy()

validation_cases = cases[
    cases["split"] == "validation"
].copy()

test_cases = cases[
    cases["split"] == "test"
].copy()

print("\nPrediction cases by split:")

print(
    "Train:",
    len(train_cases)
)

print(
    "Validation:",
    len(validation_cases)
)

print(
    "Test:",
    len(test_cases)
)


# ============================================================
# TRANSFER SOURCE MODEL TO TARGET
# ============================================================

TARGET_SOURCE_FEATURES = [
    "current_diameter_mm",
    "interval_days"
]

# Same conceptual features as source:
# diameter at current time + prediction interval.
#
# The source scaler expects:
# diameter_start
# log_diameter_start
# interval_days
# log_interval_days

def build_source_style_target_features(df):

    x = pd.DataFrame()

    x["diameter_start_mm"] = (
        df["current_diameter_mm"]
    )

    x["log_diameter_start"] = np.log(
        df["current_diameter_mm"].clip(
            lower=0.001
        )
    )

    x["interval_days"] = (
        df["future_days"]
    )

    x["log_interval_days"] = np.log(
        df["future_days"].clip(
            lower=1.0
        )
    )

    return x[
        SOURCE_FEATURES
    ]


# ============================================================
# DIRECT TRANSFER
# ============================================================

for df_name, df in [
    ("train", train_cases),
    ("validation", validation_cases),
    ("test", test_cases)
]:

    if len(df) == 0:
        continue

    X_target_source = (
        build_source_style_target_features(
            df
        )
    )

    X_target_scaled = (
        source_scaler.transform(
            X_target_source.values
        )
    )

    df["transferred_rate"] = (
        source_model.predict(
            X_target_scaled
        )
    )

    df["pred_transferred"] = (
        df["current_diameter_mm"] *
        np.exp(
            df["transferred_rate"] *
            df["future_days"]
        )
    )

    if df_name == "train":
        train_cases = df

    elif df_name == "validation":
        validation_cases = df

    else:
        test_cases = df


# ============================================================
# TARGET DOMAIN ADAPTATION
# ============================================================

print("\nPerforming target-domain residual calibration...")

# IMPORTANT:
# Calibration uses historical intervals from TRAIN patients only.
#
# We estimate how source-transferred rates systematically differ
# from actual LUMIERE rates.

adaptation_rates = []

for patient_id, group in target[
    target["patient_id"].isin(
        train_patients
    )
].groupby("patient_id"):

    group = group.sort_values(
        time_col
    ).reset_index(drop=True)

    if len(group) < 2:
        continue

    for i in range(
        1,
        len(group)
    ):

        previous_d = float(
            group.iloc[i - 1][
                diameter_col
            ]
        )

        current_d = float(
            group.iloc[i][
                diameter_col
            ]
        )

        previous_t = float(
            group.iloc[i - 1][
                time_col
            ]
        )

        current_t = float(
            group.iloc[i][
                time_col
            ]
        )

        interval_days = (
            current_t -
            previous_t
        )

        if (
            interval_days <= 0
            or previous_d <= 0
            or current_d <= 0
        ):
            continue

        actual_rate = (
            np.log(
                current_d /
                previous_d
            ) /
            interval_days
        )

        source_features = pd.DataFrame({
            "diameter_start_mm": [
                previous_d
            ],
            "log_diameter_start": [
                np.log(
                    max(
                        previous_d,
                        0.001
                    )
                )
            ],
            "interval_days": [
                interval_days
            ],
            "log_interval_days": [
                np.log(
                    max(
                        interval_days,
                        1.0
                    )
                )
            ]
        })

        scaled = source_scaler.transform(
            source_features[
                SOURCE_FEATURES
            ].values
        )

        source_rate = float(
            source_model.predict(
                scaled
            )[0]
        )

        adaptation_rates.append(
            actual_rate -
            source_rate
        )


if len(adaptation_rates) > 0:

    residual_correction = float(
        np.median(
            adaptation_rates
        )
    )

else:

    residual_correction = 0.0


print(
    "Target adaptation intervals:",
    len(adaptation_rates)
)

print(
    "Median target residual correction:",
    "%.8f" %
    residual_correction
)


# ============================================================
# APPLY DOMAIN ADAPTATION
# ============================================================

for df_name, df in [
    ("train", train_cases),
    ("validation", validation_cases),
    ("test", test_cases)
]:

    if len(df) == 0:
        continue

    df["adapted_rate"] = (
        df["transferred_rate"] +
        residual_correction
    )

    df["pred_adapted"] = (
        df["current_diameter_mm"] *
        np.exp(
            df["adapted_rate"] *
            df["future_days"]
        )
    )

    if df_name == "train":
        train_cases = df

    elif df_name == "validation":
        validation_cases = df

    else:
        test_cases = df


# ============================================================
# GEOMETRY-ONLY TARGET MODEL
# ============================================================

print("\nTraining LUMIERE geometry-only baseline...")

GEOMETRY_FEATURES = [
    "previous_diameter_mm",
    "current_diameter_mm",
    "diameter_change_mm"
]

X_train_geometry = (
    train_cases[
        GEOMETRY_FEATURES
    ].fillna(0.0).values
)

y_train_geometry = (
    train_cases[
        "actual_future_diameter_mm"
    ].values
)

geometry_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    random_state=RANDOM_SEED
)

geometry_model.fit(
    X_train_geometry,
    y_train_geometry
)


def add_geometry_predictions(df):

    if len(df) == 0:
        return df

    X = df[
        GEOMETRY_FEATURES
    ].fillna(0.0).values

    df["pred_geometry"] = (
        geometry_model.predict(X)
    )

    return df


train_cases = add_geometry_predictions(
    train_cases
)

validation_cases = add_geometry_predictions(
    validation_cases
)

test_cases = add_geometry_predictions(
    test_cases
)


# ============================================================
# GEOMETRY + TRANSFERRED HBGD
# ============================================================

# A simple RF fusion model is trained ONLY on target TRAIN patients.
#
# Inputs:
#   geometry + transferred HBGD rate
#
# This asks whether the transferred growth representation contains
# information complementary to geometry.

FUSION_FEATURES = [
    "previous_diameter_mm",
    "current_diameter_mm",
    "diameter_change_mm",
    "transferred_rate"
]

fusion_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    random_state=RANDOM_SEED
)

fusion_model.fit(
    train_cases[
        FUSION_FEATURES
    ].fillna(0.0).values,
    train_cases[
        "actual_future_diameter_mm"
    ].values
)


def add_fusion_predictions(df):

    if len(df) == 0:
        return df

    df["pred_geometry_transfer"] = (
        fusion_model.predict(
            df[
                FUSION_FEATURES
            ].fillna(0.0).values
        )
    )

    return df


train_cases = add_fusion_predictions(
    train_cases
)

validation_cases = add_fusion_predictions(
    validation_cases
)

test_cases = add_fusion_predictions(
    test_cases
)


# ============================================================
# BASELINE PREDICTIONS
# ============================================================

for df_name, df in [
    ("train", train_cases),
    ("validation", validation_cases),
    ("test", test_cases)
]:

    if len(df) == 0:
        continue

    df["pred_no_change"] = (
        df["current_diameter_mm"]
    )

    if df_name == "train":
        train_cases = df

    elif df_name == "validation":
        validation_cases = df

    else:
        test_cases = df


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

MODEL_COLUMNS = {
    "No-change":
        "pred_no_change",

    "Geometry":
        "pred_geometry",

    "Transferred HBGD":
        "pred_transferred",

    "Domain-adapted HBGD":
        "pred_adapted",

    "Geometry + Transferred HBGD":
        "pred_geometry_transfer"
}


def evaluate(df, split_name):

    rows = []

    if len(df) == 0:
        return pd.DataFrame()

    y = df[
        "actual_future_diameter_mm"
    ].values

    for model_name, pred_col in MODEL_COLUMNS.items():

        pred = df[
            pred_col
        ].values

        rows.append({
            "split": split_name,
            "model": model_name,
            "n_cases": len(df),
            "n_patients":
                df["patient_id"].nunique(),
            "MAE_mm":
                mean_absolute_error(
                    y,
                    pred
                ),
            "RMSE_mm":
                rmse(
                    y,
                    pred
                )
        })

    return pd.DataFrame(rows)


performance = pd.concat(
    [
        evaluate(
            train_cases,
            "train"
        ),
        evaluate(
            validation_cases,
            "validation"
        ),
        evaluate(
            test_cases,
            "test"
        )
    ],
    ignore_index=True
)


# ============================================================
# TEST IMPROVEMENT
# ============================================================

test_perf = performance[
    performance["split"] == "test"
].copy()

baseline_row = test_perf[
    test_perf["model"] == "Geometry"
]

if len(baseline_row) > 0:

    geometry_mae = float(
        baseline_row[
            "MAE_mm"
        ].iloc[0]
    )

    test_perf["improvement_vs_geometry_pct"] = (
        (
            geometry_mae -
            test_perf["MAE_mm"]
        )
        /
        geometry_mae
        *
        100.0
    )

else:

    test_perf[
        "improvement_vs_geometry_pct"
    ] = np.nan


# ============================================================
# SAVE PREDICTIONS
# ============================================================

all_predictions = pd.concat(
    [
        train_cases,
        validation_cases,
        test_cases
    ],
    ignore_index=True
)

prediction_file = os.path.join(
    LUMIERE_DIR,
    "step19_hbgd_representation_transfer_predictions.csv"
)

all_predictions.to_csv(
    prediction_file,
    index=False
)


# ============================================================
# SAVE PERFORMANCE
# ============================================================

performance_file = os.path.join(
    LUMIERE_DIR,
    "step19_hbgd_representation_transfer_performance.csv"
)

performance.to_csv(
    performance_file,
    index=False
)


# ============================================================
# SAVE SOURCE PARAMETERS
# ============================================================

source_parameters = pd.DataFrame({
    "parameter": [
        "source_intervals",
        "source_patients",
        "source_training_MAE_mm",
        "source_training_RMSE_mm",
        "target_prediction_cases",
        "target_train_patients",
        "target_validation_patients",
        "target_test_patients",
        "target_train_cases",
        "target_validation_cases",
        "target_test_cases",
        "target_adaptation_intervals",
        "median_residual_correction_per_day"
    ],
    "value": [
        len(source),
        source["patient_id"].nunique(),
        source_train_mae,
        source_train_rmse,
        len(cases),
        len(train_patients),
        len(validation_patients),
        len(test_patients),
        len(train_cases),
        len(validation_cases),
        len(test_cases),
        len(adaptation_rates),
        residual_correction
    ]
})

parameters_file = os.path.join(
    LUMIERE_DIR,
    "step19_hbgd_representation_transfer_parameters.csv"
)

source_parameters.to_csv(
    parameters_file,
    index=False
)


# ============================================================
# REPORT
# ============================================================

report_file = os.path.join(
    LUMIERE_DIR,
    "STEP19_HBGD_REPRESENTATION_TRANSFER_REPORT.md"
)

with open(
    report_file,
    "w"
) as f:

    f.write(
        "# Step 19 — Cross-Cancer Transfer of HBGD-Derived Growth Representations\n\n"
    )

    f.write(
        "## Objective\n\n"
    )

    f.write(
        "This experiment evaluates whether a compact growth-dynamics "
        "representation learned from the validated source HBGD cohort "
        "can transfer to LUMIERE glioblastoma progression prediction.\n\n"
    )

    f.write(
        "The experiment does not assume that the numerical growth rate "
        "from the source cancer can simply be copied to GBM. Instead, "
        "a source HBGD growth-rate mapping is transferred and then "
        "optionally calibrated using historical intervals from target "
        "training patients only.\n\n"
    )

    f.write(
        "## Source cohort\n\n"
    )

    f.write(
        "- Source artifact: `17_hbgd_state_parameters.csv`\n"
    )

    f.write(
        "- Validated source intervals: %d\n"
        % len(source)
    )

    f.write(
        "- Source patients represented: %d\n"
        % source["patient_id"].nunique()
    )

    f.write(
        "- Source growth-rate model: Random Forest\n"
    )

    f.write(
        "- Source training MAE: %.8f mm-equivalent rate error\n"
        % source_train_mae
    )

    f.write(
        "- Source training RMSE: %.8f\n\n"
        % source_train_rmse
    )

    f.write(
        "## Target cohort\n\n"
    )

    f.write(
        "- LUMIERE longitudinal trajectory rows: %d\n"
        % len(target)
    )

    f.write(
        "- LUMIERE patients represented in prediction cases: %d\n"
        % cases["patient_id"].nunique()
    )

    f.write(
        "- Train patients: %d\n"
        % len(train_patients)
    )

    f.write(
        "- Validation patients: %d\n"
        % len(validation_patients)
    )

    f.write(
        "- Exact held-out Step-13 test patients: %d\n\n"
        % len(test_patients)
    )

    f.write(
        "## Leakage control\n\n"
    )

    f.write(
        "Target prediction inputs use only measurements available "
        "at or before the prediction time. The future diameter is "
        "used only as the evaluation target.\n\n"
    )

    f.write(
        "The target-domain residual calibration uses historical "
        "intervals from training patients only. The 13-patient "
        "test cohort is not used for fitting or calibration.\n\n"
    )

    f.write(
        "## Models\n\n"
    )

    for name in MODEL_COLUMNS.keys():
        f.write(
            "- %s\n" % name
        )

    f.write("\n")

    f.write(
        "## Performance\n\n"
    )

    f.write(
        "| Split | Model | Cases | Patients | MAE (mm) | RMSE (mm) |\n"
    )

    f.write(
        "|---|---|---:|---:|---:|---:|\n"
    )

    for _, row in performance.iterrows():

        f.write(
            "| %s | %s | %d | %d | %.4f | %.4f |\n"
            % (
                row["split"],
                row["model"],
                int(row["n_cases"]),
                int(row["n_patients"]),
                row["MAE_mm"],
                row["RMSE_mm"]
            )
        )

    f.write("\n")

    f.write(
        "## Held-out test interpretation\n\n"
    )

    geometry_test = test_perf[
        test_perf["model"] == "Geometry"
    ]

    transfer_test = test_perf[
        test_perf["model"] ==
        "Transferred HBGD"
    ]

    adapted_test = test_perf[
        test_perf["model"] ==
        "Domain-adapted HBGD"
    ]

    fusion_test = test_perf[
        test_perf["model"] ==
        "Geometry + Transferred HBGD"
    ]

    if len(geometry_test) > 0:

        gm = float(
            geometry_test[
                "MAE_mm"
            ].iloc[0]
        )

        f.write(
            "Geometry-only test MAE: %.4f mm\n\n"
            % gm
        )

        if len(transfer_test) > 0:

            tm = float(
                transfer_test[
                    "MAE_mm"
                ].iloc[0]
            )

            f.write(
                "Direct transferred HBGD test MAE: %.4f mm "
                "(%.2f%% vs geometry)\n\n"
                % (
                    tm,
                    (
                        (gm - tm) /
                        gm *
                        100.0
                    )
                )
            )

        if len(adapted_test) > 0:

            am = float(
                adapted_test[
                    "MAE_mm"
                ].iloc[0]
            )

            f.write(
                "Domain-adapted HBGD test MAE: %.4f mm "
                "(%.2f%% vs geometry)\n\n"
                % (
                    am,
                    (
                        (gm - am) /
                        gm *
                        100.0
                    )
                )
            )

        if len(fusion_test) > 0:

            fm = float(
                fusion_test[
                    "MAE_mm"
                ].iloc[0]
            )

            f.write(
                "Geometry + transferred HBGD test MAE: %.4f mm "
                "(%.2f%% vs geometry)\n\n"
                % (
                    fm,
                    (
                        (gm - fm) /
                        gm *
                        100.0
                    )
                )
            )

    f.write(
        "## Scientific interpretation rule\n\n"
    )

    f.write(
        "Cross-cancer HBGD transfer is considered preliminarily "
        "supported only if the transferred representation improves "
        "the untouched LUMIERE test cohort relative to the geometry-only "
        "baseline. A numerical improvement alone should not be described "
        "as statistically significant without patient-level statistical "
        "testing.\n\n"
    )

    f.write(
        "Because the source cohort contains only 12 patients and 49 "
        "validated intervals, results should be considered exploratory "
        "until independently replicated in larger longitudinal cohorts.\n"
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 72)
print("STEP 19 COMPLETE")
print("=" * 72)

print("\nSOURCE")
print(
    "Validated source intervals:",
    len(source)
)

print(
    "Source patients:",
    source["patient_id"].nunique()
)

print("\nTARGET")
print(
    "Prediction cases:",
    len(cases)
)

print(
    "Train patients:",
    len(train_patients)
)

print(
    "Validation patients:",
    len(validation_patients)
)

print(
    "Test patients:",
    len(test_patients)
)

print(
    "Test prediction cases:",
    len(test_cases)
)

print("\nTARGET ADAPTATION")
print(
    "Calibration intervals:",
    len(adaptation_rates)
)

print(
    "Residual correction:",
    "%.8f/day" %
    residual_correction
)

print("\nHELD-OUT TEST PERFORMANCE")

for _, row in test_perf.iterrows():

    print(
        "  %-32s MAE = %7.4f mm | RMSE = %7.4f mm"
        % (
            row["model"],
            row["MAE_mm"],
            row["RMSE_mm"]
        )
    )

print("\nOUTPUT FILES")
print(
    prediction_file
)

print(
    performance_file
)

print(
    parameters_file
)

print(
    report_file
)

print("\n" + "=" * 72)