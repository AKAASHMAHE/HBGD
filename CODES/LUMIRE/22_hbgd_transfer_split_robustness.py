# -*- coding: utf-8 -*-

"""
STEP 22
CROSS-CANCER HBGD TRANSFER SPLIT ROBUSTNESS

Purpose
-------
Evaluate whether the incremental contribution of the transferred
HBGD representation remains directionally stable across multiple
patient-level train/validation/test splits.

Comparison:
    Geometry
        vs
    Geometry + Transferred HBGD

IMPORTANT
---------
1. Source HBGD representation is learned once from the validated
   source artifact:
       17_hbgd_state_parameters.csv

2. Source representation is NOT relearned separately for every
   target split.

3. LUMIERE target models are trained independently inside each
   patient-level split.

4. No patient appears in more than one split.

5. Future target diameter is never used as an input feature.

6. Test patients are not used for model fitting.

7. This is a robustness experiment, NOT a statistical claim of
   clinical validation.

Python 3.6 compatible.
"""

import os
import math
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA"

LUMIERE_DIR = (
    r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"
)

SOURCE_STATE = os.path.join(
    BASE_DIR,
    "17_hbgd_state_parameters.csv"
)

TARGET_TRAJECTORIES = os.path.join(
    LUMIERE_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

OUTPUT_SPLITS = os.path.join(
    LUMIERE_DIR,
    "step22_hbgd_transfer_split_results.csv"
)

OUTPUT_PATIENT = os.path.join(
    LUMIERE_DIR,
    "step22_hbgd_transfer_patient_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    LUMIERE_DIR,
    "step22_hbgd_transfer_robustness_summary.csv"
)

OUTPUT_REPORT = os.path.join(
    LUMIERE_DIR,
    "STEP22_HBGD_TRANSFER_SPLIT_ROBUSTNESS_REPORT.md"
)


# ============================================================
# SETTINGS
# ============================================================

# Multiple independent patient-level splits.
SEEDS = [
    42,
    7,
    21,
    100,
    123,
    202,
    314,
    512,
    777,
    1001
]

TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.125

SOURCE_FEATURES = [
    "diameter_start_mm",
    "log_diameter_start",
    "interval_days",
    "log_interval_days"
]

GEOMETRY_FEATURES = [
    "previous_diameter_mm",
    "current_diameter_mm",
    "diameter_change_mm"
]

FUSION_FEATURES = [
    "previous_diameter_mm",
    "current_diameter_mm",
    "diameter_change_mm",
    "transferred_rate"
]


# ============================================================
# UTILITY
# ============================================================

def normalize_patient_id(value):

    s = str(value).strip()

    if s.lower().startswith(
        "patient-"
    ):
        return s

    try:
        return "Patient-%03d" % int(
            float(s)
        )
    except Exception:
        return s


def rmse(
    y_true,
    y_pred
):

    return math.sqrt(
        np.mean(
            (
                np.asarray(y_true)
                -
                np.asarray(y_pred)
            ) ** 2
        )
    )


# ============================================================
# LOAD SOURCE
# ============================================================

print("=" * 78)
print(
    "STEP 22 CROSS-CANCER HBGD TRANSFER SPLIT ROBUSTNESS"
)
print("=" * 78)

print(
    "\nLoading validated source HBGD intervals..."
)

if not os.path.exists(
    SOURCE_STATE
):

    raise IOError(
        "Missing source file:\n%s"
        % SOURCE_STATE
    )

source = pd.read_csv(
    SOURCE_STATE
)

source["patient_id"] = (
    source["patient_id"]
    .apply(
        normalize_patient_id
    )
)

required_source = [
    "patient_id",
    "time_start_days",
    "time_end_days",
    "diameter_start_mm",
    "diameter_end_mm",
    "growth_rate"
]

for c in required_source:

    if c not in source.columns:

        raise ValueError(
            "Missing source column: %s"
            % c
        )


for c in [
    "time_start_days",
    "time_end_days",
    "diameter_start_mm",
    "diameter_end_mm",
    "growth_rate"
]:

    source[c] = pd.to_numeric(
        source[c],
        errors="coerce"
    )


source["interval_days"] = (
    source["time_end_days"]
    -
    source["time_start_days"]
)

source = source[
    (source["interval_days"] > 0)
    &
    (source["diameter_start_mm"] > 0)
    &
    (source["diameter_end_mm"] > 0)
    &
    source["growth_rate"].notna()
].copy()

print(
    "Source intervals:",
    len(source)
)

print(
    "Source patients:",
    source["patient_id"].nunique()
)


# ============================================================
# SOURCE REPRESENTATION
# ============================================================

source["log_diameter_start"] = np.log(
    source[
        "diameter_start_mm"
    ]
)

source["log_interval_days"] = np.log(
    source[
        "interval_days"
    ].clip(
        lower=1.0
    )
)

X_source = source[
    SOURCE_FEATURES
].values

y_source = source[
    "growth_rate"
].values


# ============================================================
# SOURCE STANDARDIZATION
# ============================================================

source_scaler = StandardScaler()

X_source_scaled = (
    source_scaler.fit_transform(
        X_source
    )
)


# ============================================================
# SOURCE HBGD MODEL
# ============================================================

print(
    "\nTraining fixed source HBGD model..."
)

source_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)

source_model.fit(
    X_source_scaled,
    y_source
)

source_train_pred = (
    source_model.predict(
        X_source_scaled
    )
)

print(
    "Source training MAE:",
    "%.8f"
    % mean_absolute_error(
        y_source,
        source_train_pred
    )
)


# ============================================================
# LOAD LUMIERE
# ============================================================

print(
    "\nLoading LUMIERE trajectories..."
)

if not os.path.exists(
    TARGET_TRAJECTORIES
):

    raise IOError(
        "Missing target trajectory file:\n%s"
        % TARGET_TRAJECTORIES
    )

target = pd.read_csv(
    TARGET_TRAJECTORIES
)

target["patient_id"] = (
    target["patient_id"]
    .apply(
        normalize_patient_id
    )
)

required_target = [
    "patient_id",
    "week",
    "tumor_volume_cm3",
    "equivalent_diameter_mm"
]

for c in required_target:

    if c not in target.columns:

        raise ValueError(
            "Missing target column: %s"
            % c
        )


target["week"] = pd.to_numeric(
    target["week"],
    errors="coerce"
)

target[
    "equivalent_diameter_mm"
] = pd.to_numeric(
    target[
        "equivalent_diameter_mm"
    ],
    errors="coerce"
)

target[
    "tumor_volume_cm3"
] = pd.to_numeric(
    target[
        "tumor_volume_cm3"
    ],
    errors="coerce"
)

# LUMIERE week index → elapsed days.
target["time_days"] = (
    target["week"] * 7.0
)

target = target[
    target["time_days"].notna()
    &
    target[
        "equivalent_diameter_mm"
    ].notna()
    &
    (
        target[
            "equivalent_diameter_mm"
        ] > 0
    )
].copy()

target = target.sort_values(
    [
        "patient_id",
        "time_days"
    ]
).reset_index(
    drop=True
)

print(
    "Target trajectory rows:",
    len(target)
)

print(
    "Target patients:",
    target["patient_id"].nunique()
)


# ============================================================
# BUILD LEAKAGE-CONTROLLED CASES
# ============================================================

print(
    "\nConstructing target prediction cases..."
)

cases = []

for patient_id, group in target.groupby(
    "patient_id"
):

    group = group.sort_values(
        "time_days"
    ).reset_index(
        drop=True
    )

    if len(group) < 3:
        continue

    history_rates = []

    for i in range(
        1,
        len(group)
    ):

        previous = group.iloc[
            i - 1
        ]

        current = group.iloc[
            i
        ]

        previous_d = float(
            previous[
                "equivalent_diameter_mm"
            ]
        )

        current_d = float(
            current[
                "equivalent_diameter_mm"
            ]
        )

        previous_t = float(
            previous[
                "time_days"
            ]
        )

        current_t = float(
            current[
                "time_days"
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

        observed_rate = (
            np.log(
                current_d /
                previous_d
            )
            /
            interval_days
        )

        history_rates.append(
            observed_rate
        )

        # Future observation.
        if i + 1 >= len(group):
            continue

        future = group.iloc[
            i + 1
        ]

        future_t = float(
            future[
                "time_days"
            ]
        )

        future_d = float(
            future[
                "equivalent_diameter_mm"
            ]
        )

        future_days = (
            future_t -
            current_t
        )

        if (
            future_days <= 0
            or future_d <= 0
        ):
            continue

        cases.append({

            "patient_id":
                patient_id,

            "current_week":
                float(
                    current["week"]
                ),

            "future_week":
                float(
                    future["week"]
                ),

            "current_diameter_mm":
                current_d,

            "previous_diameter_mm":
                previous_d,

            "current_volume_cm3":
                float(
                    current[
                        "tumor_volume_cm3"
                    ]
                )
                if not pd.isna(
                    current[
                        "tumor_volume_cm3"
                    ]
                )
                else np.nan,

            "previous_volume_cm3":
                float(
                    previous[
                        "tumor_volume_cm3"
                    ]
                )
                if not pd.isna(
                    previous[
                        "tumor_volume_cm3"
                    ]
                )
                else np.nan,

            "diameter_change_mm":
                current_d -
                previous_d,

            "interval_days":
                interval_days,

            "future_days":
                future_days,

            "actual_future_diameter_mm":
                future_d,

            "history_mean_rate":
                np.mean(
                    history_rates
                ),

            "history_abs_mean_rate":
                np.mean(
                    np.abs(
                        history_rates
                    )
                ),

            "history_rate_std":
                np.std(
                    history_rates
                )
                if len(
                    history_rates
                ) >= 2
                else 0.0,

            "history_n_intervals":
                len(
                    history_rates
                )
        })


cases = pd.DataFrame(
    cases
)

print(
    "Prediction cases:",
    len(cases)
)

print(
    "Prediction patients:",
    cases[
        "patient_id"
    ].nunique()
)


# ============================================================
# REMOVE PATIENTS WITH NO CASES
# ============================================================

all_patients = sorted(
    cases[
        "patient_id"
    ].unique()
)

print(
    "Patients available for repeated splits:",
    len(all_patients)
)


# ============================================================
# FIXED SOURCE TRANSFER FUNCTION
# ============================================================

def predict_source_rate(
    df
):

    source_style = pd.DataFrame({

        "diameter_start_mm":
            df[
                "current_diameter_mm"
            ].values,

        "log_diameter_start":
            np.log(
                df[
                    "current_diameter_mm"
                ].values
                .clip(
                    min=0.001
                )
            ),

        "interval_days":
            df[
                "future_days"
            ].values,

        "log_interval_days":
            np.log(
                df[
                    "future_days"
                ].values
                .clip(
                    min=1.0
                )
            )
    })

    scaled = (
        source_scaler.transform(
            source_style[
                SOURCE_FEATURES
            ].values
        )
    )

    return source_model.predict(
        scaled
    )


# ============================================================
# REPEATED SPLITS
# ============================================================

split_rows = []
patient_rows = []

for seed in SEEDS:

    print(
        "\n"
        + "-" * 72
    )

    print(
        "SPLIT SEED:",
        seed
    )

    rng = np.random.RandomState(
        seed
    )

    shuffled = list(
        all_patients
    )

    rng.shuffle(
        shuffled
    )

    n_patients = len(
        shuffled
    )

    n_test = int(
        round(
            n_patients *
            TEST_FRACTION
        )
    )

    n_validation = int(
        round(
            n_patients *
            VALIDATION_FRACTION
        )
    )

    if n_test < 1:
        n_test = 1

    if n_validation < 1:
        n_validation = 1

    if (
        n_test +
        n_validation
        >=
        n_patients
    ):

        n_validation = 1

        n_test = max(
            1,
            n_patients - 2
        )

    test_patients = set(
        shuffled[
            :n_test
        ]
    )

    validation_patients = set(
        shuffled[
            n_test:
            n_test +
            n_validation
        ]
    )

    train_patients = set(
        shuffled[
            n_test +
            n_validation:
        ]
    )

    train = cases[
        cases[
            "patient_id"
        ].isin(
            train_patients
        )
    ].copy()

    validation = cases[
        cases[
            "patient_id"
        ].isin(
            validation_patients
        )
    ].copy()

    test = cases[
        cases[
            "patient_id"
        ].isin(
            test_patients
        )
    ].copy()

    print(
        "Patients:",
        "train=%d" % len(
            train_patients
        ),
        "validation=%d" % len(
            validation_patients
        ),
        "test=%d" % len(
            test_patients
        )
    )

    print(
        "Cases:",
        "train=%d" % len(train),
        "validation=%d"
        % len(validation),
        "test=%d"
        % len(test)
    )

    # --------------------------------------------------------
    # Skip unusable split
    # --------------------------------------------------------

    if (
        len(train) == 0
        or len(test) == 0
    ):

        print(
            "WARNING: unusable split"
        )

        continue

    # --------------------------------------------------------
    # TRANSFERRED HBGD RATE
    # --------------------------------------------------------

    train[
        "transferred_rate"
    ] = predict_source_rate(
        train
    )

    validation[
        "transferred_rate"
    ] = predict_source_rate(
        validation
    )

    test[
        "transferred_rate"
    ] = predict_source_rate(
        test
    )

    # --------------------------------------------------------
    # GEOMETRY MODEL
    # --------------------------------------------------------

    geometry_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        random_state=seed
    )

    geometry_model.fit(
        train[
            GEOMETRY_FEATURES
        ].fillna(
            0.0
        ).values,

        train[
            "actual_future_diameter_mm"
        ].values
    )

    geometry_pred = (
        geometry_model.predict(
            test[
                GEOMETRY_FEATURES
            ].fillna(
                0.0
            ).values
        )
    )

    # --------------------------------------------------------
    # GEOMETRY + TRANSFERRED HBGD
    # --------------------------------------------------------

    fusion_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=3,
        random_state=seed
    )

    fusion_model.fit(
        train[
            FUSION_FEATURES
        ].fillna(
            0.0
        ).values,

        train[
            "actual_future_diameter_mm"
        ].values
    )

    fusion_pred = (
        fusion_model.predict(
            test[
                FUSION_FEATURES
            ].fillna(
                0.0
            ).values
        )
    )

    # --------------------------------------------------------
    # NO CHANGE
    # --------------------------------------------------------

    no_change_pred = (
        test[
            "current_diameter_mm"
        ].values
    )

    # --------------------------------------------------------
    # DIRECT TRANSFER
    # --------------------------------------------------------

    direct_pred = (
        test[
            "current_diameter_mm"
        ].values
        *
        np.exp(
            test[
                "transferred_rate"
            ].values
            *
            test[
                "future_days"
            ].values
        )
    )

    y_test = test[
        "actual_future_diameter_mm"
    ].values

    # --------------------------------------------------------
    # POOLED METRICS
    # --------------------------------------------------------

    geometry_mae = (
        mean_absolute_error(
            y_test,
            geometry_pred
        )
    )

    fusion_mae = (
        mean_absolute_error(
            y_test,
            fusion_pred
        )
    )

    no_change_mae = (
        mean_absolute_error(
            y_test,
            no_change_pred
        )
    )

    direct_mae = (
        mean_absolute_error(
            y_test,
            direct_pred
        )
    )

    geometry_rmse = rmse(
        y_test,
        geometry_pred
    )

    fusion_rmse = rmse(
        y_test,
        fusion_pred
    )

    # --------------------------------------------------------
    # PATIENT-LEVEL RESULTS
    # --------------------------------------------------------

    test_eval = test.copy()

    test_eval[
        "geometry_pred"
    ] = geometry_pred

    test_eval[
        "fusion_pred"
    ] = fusion_pred

    test_eval[
        "no_change_pred"
    ] = no_change_pred

    test_eval[
        "direct_transfer_pred"
    ] = direct_pred

    test_eval[
        "geometry_error"
    ] = np.abs(
        test_eval[
            "actual_future_diameter_mm"
        ]
        -
        test_eval[
            "geometry_pred"
        ]
    )

    test_eval[
        "fusion_error"
    ] = np.abs(
        test_eval[
            "actual_future_diameter_mm"
        ]
        -
        test_eval[
            "fusion_pred"
        ]
    )

    test_eval[
        "patient_improvement_mm"
    ] = (
        test_eval[
            "geometry_error"
        ]
        -
        test_eval[
            "fusion_error"
        ]
    )

    for patient_id, group in test_eval.groupby(
        "patient_id"
    ):

        patient_geometry_mae = (
            group[
                "geometry_error"
            ].mean()
        )

        patient_fusion_mae = (
            group[
                "fusion_error"
            ].mean()
        )

        patient_improvement = (
            patient_geometry_mae
            -
            patient_fusion_mae
        )

        patient_rows.append({

            "seed":
                seed,

            "patient_id":
                patient_id,

            "n_cases":
                len(group),

            "geometry_MAE_mm":
                patient_geometry_mae,

            "geometry_hbgd_MAE_mm":
                patient_fusion_mae,

            "improvement_mm":
                patient_improvement,

            "improved":
                int(
                    patient_improvement > 0
                ),

            "worsened":
                int(
                    patient_improvement < 0
                )
        })

    # --------------------------------------------------------
    # SPLIT RESULT
    # --------------------------------------------------------

    fusion_improvement = (
        geometry_mae -
        fusion_mae
    )

    fusion_improvement_pct = (
        fusion_improvement /
        geometry_mae *
        100.0
        if geometry_mae != 0
        else np.nan
    )

    split_rows.append({

        "seed":
            seed,

        "train_patients":
            len(train_patients),

        "validation_patients":
            len(validation_patients),

        "test_patients":
            len(test_patients),

        "train_cases":
            len(train),

        "validation_cases":
            len(validation),

        "test_cases":
            len(test),

        "no_change_MAE_mm":
            no_change_mae,

        "direct_transfer_MAE_mm":
            direct_mae,

        "geometry_MAE_mm":
            geometry_mae,

        "geometry_RMSE_mm":
            geometry_rmse,

        "geometry_hbgd_MAE_mm":
            fusion_mae,

        "geometry_hbgd_RMSE_mm":
            fusion_rmse,

        "improvement_mm":
            fusion_improvement,

        "improvement_pct":
            fusion_improvement_pct,

        "hbgd_better_than_geometry":
            int(
                fusion_mae <
                geometry_mae
            )
    })

    print(
        "Geometry MAE:",
        "%.4f" % geometry_mae
    )

    print(
        "Geometry + HBGD MAE:",
        "%.4f" % fusion_mae
    )

    print(
        "Improvement:",
        "%.4f mm (%.2f%%)"
        % (
            fusion_improvement,
            fusion_improvement_pct
        )
    )


# ============================================================
# RESULTS DATAFRAMES
# ============================================================

split_results = pd.DataFrame(
    split_rows
)

patient_results = pd.DataFrame(
    patient_rows
)


if len(split_results) == 0:

    raise RuntimeError(
        "No valid repeated split results were produced."
    )


# ============================================================
# SPLIT-LEVEL SUMMARY
# ============================================================

improvements = (
    split_results[
        "improvement_mm"
    ].values
)

improvement_pcts = (
    split_results[
        "improvement_pct"
    ].values
)

n_splits = len(
    split_results
)

n_positive = int(
    np.sum(
        improvements > 0
    )
)

n_negative = int(
    np.sum(
        improvements < 0
    )
)

mean_improvement = np.mean(
    improvements
)

median_improvement = np.median(
    improvements
)

sd_improvement = (
    np.std(
        improvements,
        ddof=1
    )
    if n_splits > 1
    else np.nan
)

mean_pct = np.mean(
    improvement_pcts
)

median_pct = np.median(
    improvement_pcts
)

positive_fraction = (
    float(n_positive) /
    float(n_splits)
)


# ============================================================
# PATIENT-LEVEL SUMMARY
# ============================================================

patient_improvements = (
    patient_results[
        "improvement_mm"
    ].values
)

patient_n = len(
    patient_improvements
)

patient_positive = int(
    np.sum(
        patient_improvements > 0
    )
)

patient_negative = int(
    np.sum(
        patient_improvements < 0
    )
)

patient_mean = np.mean(
    patient_improvements
)

patient_median = np.median(
    patient_improvements
)


# ============================================================
# SAVE RESULTS
# ============================================================

split_results.to_csv(
    OUTPUT_SPLITS,
    index=False
)

patient_results.to_csv(
    OUTPUT_PATIENT,
    index=False
)


# ============================================================
# SUMMARY CSV
# ============================================================

summary_rows = [

    {
        "metric":
            "source_intervals",
        "value":
            len(source)
    },

    {
        "metric":
            "source_patients",
        "value":
            source[
                "patient_id"
            ].nunique()
    },

    {
        "metric":
            "target_patients",
        "value":
            cases[
                "patient_id"
            ].nunique()
    },

    {
        "metric":
            "target_prediction_cases",
        "value":
            len(cases)
    },

    {
        "metric":
            "number_of_valid_splits",
        "value":
            n_splits
    },

    {
        "metric":
            "splits_HBGD_improved",
        "value":
            n_positive
    },

    {
        "metric":
            "splits_HBGD_worsened",
        "value":
            n_negative
    },

    {
        "metric":
            "fraction_splits_improved",
        "value":
            positive_fraction
    },

    {
        "metric":
            "mean_split_improvement_mm",
        "value":
            mean_improvement
    },

    {
        "metric":
            "median_split_improvement_mm",
        "value":
            median_improvement
    },

    {
        "metric":
            "SD_split_improvement_mm",
        "value":
            sd_improvement
    },

    {
        "metric":
            "mean_split_improvement_pct",
        "value":
            mean_pct
    },

    {
        "metric":
            "median_split_improvement_pct",
        "value":
            median_pct
    },

    {
        "metric":
            "patient_evaluations",
        "value":
            patient_n
    },

    {
        "metric":
            "patient_evaluations_improved",
        "value":
            patient_positive
    },

    {
        "metric":
            "patient_evaluations_worsened",
        "value":
            patient_negative
    },

    {
        "metric":
            "patient_mean_improvement_mm",
        "value":
            patient_mean
    },

    {
        "metric":
            "patient_median_improvement_mm",
        "value":
            patient_median
    }
]

summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)


# ============================================================
# INTERPRETATION
# ============================================================

if positive_fraction >= 0.70:

    robustness_statement = (
        "The transferred HBGD representation showed a "
        "directionally favorable effect in at least 70% "
        "of valid patient-level splits."
    )

elif positive_fraction > 0.50:

    robustness_statement = (
        "The transferred HBGD representation showed a "
        "directionally favorable effect in a majority, "
        "but not a strong majority, of patient-level splits."
    )

else:

    robustness_statement = (
        "The transferred HBGD representation did not show "
        "a consistently favorable effect across patient-level "
        "splits."
    )


# ============================================================
# REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w"
) as f:

    f.write(
        "# Step 22 — Cross-Cancer HBGD Transfer Split Robustness\n\n"
    )

    f.write(
        "## Objective\n\n"
    )

    f.write(
        "This experiment evaluates whether the incremental "
        "predictive contribution of the transferred HBGD "
        "representation remains stable across multiple "
        "patient-level target-domain train/validation/test "
        "splits.\n\n"
    )

    f.write(
        "The source HBGD representation was learned once from "
        "49 validated source intervals representing 12 source "
        "patients. It was then transferred unchanged across "
        "all target-domain splits.\n\n"
    )

    f.write(
        "## Leakage control\n\n"
    )

    f.write(
        "- Splitting was performed at the patient level.\n"
    )

    f.write(
        "- Test patients were excluded from model fitting.\n"
    )

    f.write(
        "- Future target diameter was not used as an input.\n"
    )

    f.write(
        "- The source HBGD model was fixed across target splits.\n"
    )

    f.write(
        "- Geometry and fusion models were independently fitted "
        "within each target training split.\n\n"
    )

    f.write(
        "## Split-level results\n\n"
    )

    f.write(
        "| Seed | Train patients | Validation patients | "
        "Test patients | Geometry MAE | Geometry + HBGD MAE | "
        "Improvement | Improvement % |\n"
    )

    f.write(
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )

    for _, row in split_results.iterrows():

        f.write(
            "| %d | %d | %d | %d | %.4f | %.4f | %.4f | %.2f%% |\n"
            % (
                int(row["seed"]),
                int(row["train_patients"]),
                int(row["validation_patients"]),
                int(row["test_patients"]),
                row["geometry_MAE_mm"],
                row["geometry_hbgd_MAE_mm"],
                row["improvement_mm"],
                row["improvement_pct"]
            )
        )

    f.write("\n")

    f.write(
        "## Robustness summary\n\n"
    )

    f.write(
        "- Valid splits: %d\n"
        % n_splits
    )

    f.write(
        "- Splits improved by HBGD: %d/%d\n"
        % (
            n_positive,
            n_splits
        )
    )

    f.write(
        "- Splits worsened by HBGD: %d/%d\n"
        % (
            n_negative,
            n_splits
        )
    )

    f.write(
        "- Fraction of splits improved: %.2f%%\n"
        % (
            positive_fraction *
            100.0
        )
    )

    f.write(
        "- Mean split improvement: %.4f mm\n"
        % mean_improvement
    )

    f.write(
        "- Median split improvement: %.4f mm\n"
        % median_improvement
    )

    f.write(
        "- Mean percentage improvement: %.2f%%\n"
        % mean_pct
    )

    f.write(
        "- Median percentage improvement: %.2f%%\n\n"
        % median_pct
    )

    f.write(
        "## Patient-level robustness\n\n"
    )

    f.write(
        "- Patient evaluations across splits: %d\n"
        % patient_n
    )

    f.write(
        "- Improved evaluations: %d\n"
        % patient_positive
    )

    f.write(
        "- Worsened evaluations: %d\n"
        % patient_negative
    )

    f.write(
        "- Mean patient improvement: %.4f mm\n"
        % patient_mean
    )

    f.write(
        "- Median patient improvement: %.4f mm\n\n"
        % patient_median
    )

    f.write(
        "## Interpretation\n\n"
    )

    f.write(
        robustness_statement
        +
        "\n\n"
    )

    f.write(
        "This analysis evaluates stability of the observed "
        "directional effect across alternative patient cohorts. "
        "It does not constitute a formal hypothesis test and "
        "should not be interpreted as clinical validation.\n\n"
    )

    f.write(
        "## Scientific conclusion\n\n"
    )

    if (
        positive_fraction > 0.50
        and mean_improvement > 0
    ):

        f.write(
            "Across repeated patient-level splits, the "
            "Geometry + Transferred HBGD model showed a "
            "generally favorable directional pattern relative "
            "to Geometry alone. This strengthens the hypothesis "
            "that source-derived HBGD information may provide "
            "complementary cross-cancer predictive signal. "
            "However, the magnitude and consistency of the "
            "effect must still be evaluated with larger "
            "independent cohorts and formal statistical "
            "analysis.\n"
        )

    else:

        f.write(
            "The repeated split analysis did not demonstrate "
            "a sufficiently consistent directional advantage "
            "for the transferred HBGD representation. The "
            "cross-cancer transfer hypothesis therefore remains "
            "uncertain and requires further investigation.\n"
        )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 78)
print("STEP 22 COMPLETE")
print("=" * 78)

print(
    "\nVALID SPLITS:",
    n_splits
)

print(
    "HBGD improved:",
    n_positive,
    "/",
    n_splits
)

print(
    "HBGD worsened:",
    n_negative,
    "/",
    n_splits
)

print(
    "Fraction improved:",
    "%.2f%%"
    % (
        positive_fraction *
        100.0
    )
)

print(
    "\nMEAN SPLIT IMPROVEMENT:",
    "%.4f mm"
    % mean_improvement
)

print(
    "MEDIAN SPLIT IMPROVEMENT:",
    "%.4f mm"
    % median_improvement
)

print(
    "MEAN PERCENT IMPROVEMENT:",
    "%.2f%%"
    % mean_pct
)

print(
    "\nPATIENT-LEVEL EVALUATIONS:",
    patient_n
)

print(
    "Patient evaluations improved:",
    patient_positive
)

print(
    "Patient evaluations worsened:",
    patient_negative
)

print(
    "\nROBUSTNESS:"
)

print(
    robustness_statement
)

print(
    "\nOUTPUTS:"
)

print(
    OUTPUT_SPLITS
)

print(
    OUTPUT_PATIENT
)

print(
    OUTPUT_SUMMARY
)

print(
    OUTPUT_REPORT
)

print(
    "\n" + "=" * 78
)