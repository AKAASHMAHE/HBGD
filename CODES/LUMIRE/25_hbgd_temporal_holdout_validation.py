# -*- coding: utf-8 -*-

"""
STEP 25
STRICT TEMPORAL HOLDOUT VALIDATION

Goal
----
Evaluate whether transferred HBGD information remains useful when
prediction is performed strictly forward in time.

For each patient:

    Historical observations
            |
            v
    HBGD representation
            |
            v
    Predict later unseen observation

No future observation is used to construct historical features.

Models:
    M0 = No Change
    M1 = Geometry
    M2 = Geometry + Transferred HBGD
    M3 = Direct Transferred HBGD

Temporal forecasting protocol:
    For each patient with >= 4 timepoints:

        history = observations before prediction
        target   = next unseen observation

    Multiple origins are evaluated within each patient.

IMPORTANT
---------
The source HBGD model is fixed and learned only from:
    17_hbgd_state_parameters.csv

Target-domain models are trained using only earlier temporal
observations.

Patient-level separation is maintained for target model fitting:
the patient's own future observation is never used for training.

Python 3.6 compatible.
"""

import os
import math
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error


# ============================================================
# PATHS
# ============================================================

LUMIERE_DIR = (
    r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"
)

SOURCE_STATE = os.path.join(
    r"D:\VIT\PROJECT\NLP\PATENT\DATA",
    "17_hbgd_state_parameters.csv"
)

TRAJECTORIES = os.path.join(
    LUMIERE_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

OUTPUT_PREDICTIONS = os.path.join(
    LUMIERE_DIR,
    "step25_hbgd_temporal_predictions.csv"
)

OUTPUT_PERFORMANCE = os.path.join(
    LUMIERE_DIR,
    "step25_hbgd_temporal_performance.csv"
)

OUTPUT_PATIENT = os.path.join(
    LUMIERE_DIR,
    "step25_hbgd_temporal_patient_results.csv"
)

OUTPUT_PHENOTYPE = os.path.join(
    LUMIERE_DIR,
    "step25_hbgd_temporal_phenotype_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    LUMIERE_DIR,
    "step25_hbgd_temporal_summary.csv"
)

OUTPUT_REPORT = os.path.join(
    LUMIERE_DIR,
    "STEP25_HBGD_STRICT_TEMPORAL_HOLDOUT_REPORT.md"
)


# ============================================================
# SETTINGS
# ============================================================

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

RANDOM_SEED = 42


# ============================================================
# UTILITY
# ============================================================

def normalize_patient_id(value):

    s = str(value).strip()

    if s.lower().startswith("patient-"):
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
# HEADER
# ============================================================

print("=" * 78)
print(
    "STEP 25 STRICT TEMPORAL HOLDOUT VALIDATION"
)
print("=" * 78)


# ============================================================
# LOAD SOURCE HBGD
# ============================================================

print(
    "\nLoading source HBGD intervals..."
)

if not os.path.exists(
    SOURCE_STATE
):

    raise IOError(
        "Missing source HBGD file:\n%s"
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

for column in [
    "time_start_days",
    "time_end_days",
    "diameter_start_mm",
    "diameter_end_mm",
    "growth_rate"
]:

    source[column] = pd.to_numeric(
        source[column],
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
# FIXED SOURCE REPRESENTATION
# ============================================================

source_scaler = StandardScaler()

X_source_scaled = (
    source_scaler.fit_transform(
        X_source
    )
)

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

print(
    "Source intervals:",
    len(source)
)

print(
    "Source patients:",
    source[
        "patient_id"
    ].nunique()
)


# ============================================================
# SOURCE TRANSFER FUNCTION
# ============================================================

def transferred_rate(
    current_diameter,
    future_days
):

    future_days = max(
        float(future_days),
        1.0
    )

    current_diameter = max(
        float(current_diameter),
        0.001
    )

    X = np.array([[
        current_diameter,
        np.log(
            current_diameter
        ),
        future_days,
        np.log(
            future_days
        )
    ]])

    X_scaled = (
        source_scaler.transform(
            X
        )
    )

    return float(
        source_model.predict(
            X_scaled
        )[0]
    )


# ============================================================
# LOAD TARGET
# ============================================================

print(
    "\nLoading LUMIERE trajectories..."
)

if not os.path.exists(
    TRAJECTORIES
):

    raise IOError(
        "Missing LUMIERE trajectory file:\n%s"
        % TRAJECTORIES
    )

target = pd.read_csv(
    TRAJECTORIES
)

target["patient_id"] = (
    target["patient_id"]
    .apply(
        normalize_patient_id
    )
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

target = target[
    target["week"].notna()
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

target["time_days"] = (
    target["week"] * 7.0
)

target = target.sort_values(
    [
        "patient_id",
        "time_days"
    ]
).reset_index(
    drop=True
)

print(
    "Trajectory rows:",
    len(target)
)

print(
    "Patients:",
    target[
        "patient_id"
    ].nunique()
)


# ============================================================
# BUILD STRICT TEMPORAL CASES
# ============================================================

print(
    "\nConstructing strict forward-in-time cases..."
)

cases = []

eligible_patients = 0

for patient_id, group in target.groupby(
    "patient_id"
):

    group = group.sort_values(
        "time_days"
    ).reset_index(
        drop=True
    )

    if len(group) < 4:
        continue

    eligible_patients += 1

    # --------------------------------------------------------
    # Every prediction origin uses ONLY observations before
    # the target observation.
    # --------------------------------------------------------

    for target_index in range(
        2,
        len(group)
    ):

        current_index = (
            target_index - 1
        )

        current = group.iloc[
            current_index
        ]

        future = group.iloc[
            target_index
        ]

        # ----------------------------------------------------
        # Historical observations available before current
        # ----------------------------------------------------

        history = group.iloc[
            :current_index + 1
        ].copy()

        if len(history) < 3:
            continue

        historical_rates = []

        for j in range(
            1,
            len(history)
        ):

            p = history.iloc[
                j - 1
            ]

            c = history.iloc[
                j
            ]

            d1 = float(
                p[
                    "equivalent_diameter_mm"
                ]
            )

            d2 = float(
                c[
                    "equivalent_diameter_mm"
                ]
            )

            t1 = float(
                p[
                    "time_days"
                ]
            )

            t2 = float(
                c[
                    "time_days"
                ]
            )

            dt = (
                t2 - t1
            )

            if (
                d1 <= 0
                or
                d2 <= 0
                or
                dt <= 0
            ):
                continue

            rate = (
                np.log(
                    d2 / d1
                )
                /
                dt
            )

            historical_rates.append(
                rate
            )

        if len(
            historical_rates
        ) == 0:
            continue

        current_d = float(
            current[
                "equivalent_diameter_mm"
            ]
        )

        future_d = float(
            future[
                "equivalent_diameter_mm"
            ]
        )

        current_t = float(
            current[
                "time_days"
            ]
        )

        future_t = float(
            future[
                "time_days"
            ]
        )

        forecast_days = (
            future_t -
            current_t
        )

        if forecast_days <= 0:
            continue

        previous = history.iloc[
            -2
        ]

        previous_d = float(
            previous[
                "equivalent_diameter_mm"
            ]
        )

        current_volume = (
            current[
                "tumor_volume_cm3"
            ]
        )

        previous_volume = (
            previous[
                "tumor_volume_cm3"
            ]
        )

        diameter_change = (
            current_d -
            previous_d
        )

        mean_rate = np.mean(
            historical_rates
        )

        rate_std = (
            np.std(
                historical_rates,
                ddof=1
            )
            if len(
                historical_rates
            ) >= 2
            else 0.0
        )

        if mean_rate > 0.0005:

            phenotype = "Growth"

        elif mean_rate < -0.0010:

            phenotype = "Shrinkage"

        else:

            phenotype = "Stable"

        cases.append({

            "patient_id":
                patient_id,

            "current_week":
                float(
                    current[
                        "week"
                    ]
                ),

            "future_week":
                float(
                    future[
                        "week"
                    ]
                ),

            "current_time_days":
                current_t,

            "future_time_days":
                future_t,

            "future_days":
                forecast_days,

            "previous_diameter_mm":
                previous_d,

            "current_diameter_mm":
                current_d,

            "actual_future_diameter_mm":
                future_d,

            "current_volume_cm3":
                current_volume,

            "previous_volume_cm3":
                previous_volume,

            "diameter_change_mm":
                diameter_change,

            "history_n_timepoints":
                len(history),

            "history_n_intervals":
                len(
                    historical_rates
                ),

            "history_mean_rate":
                mean_rate,

            "history_rate_std":
                rate_std,

            "hbgd_phenotype":
                phenotype,

            "transferred_rate":
                transferred_rate(
                    current_d,
                    forecast_days
                )
        })


cases = pd.DataFrame(
    cases
)

print(
    "Eligible patients (>=4 timepoints):",
    eligible_patients
)

print(
    "Strict temporal prediction cases:",
    len(cases)
)

print(
    "Patients represented:",
    cases[
        "patient_id"
    ].nunique()
)


# ============================================================
# IMPORTANT TEMPORAL TRAINING DESIGN
# ============================================================

"""
For each prediction origin, the target-domain regression model
must not use the patient's future observation.

We therefore construct training examples from OTHER PATIENTS only.

For a given patient:
    - its historical data create its prediction features
    - its future target is test-only
    - that patient's data are never included in training

This gives a strict leave-one-patient-out temporal evaluation.

Because this is computationally heavier, the model is fitted
once per unique test patient.
"""


# ============================================================
# PATIENT-LEVEL TEMPORAL FORECASTING
# ============================================================

prediction_rows = []

patient_results = []

unique_patients = sorted(
    cases[
        "patient_id"
    ].unique()
)

print(
    "\nRunning leave-one-patient-out temporal validation..."
)

for patient_number, test_patient in enumerate(
    unique_patients
):

    print(
        "Patient %d/%d: %s"
        % (
            patient_number + 1,
            len(unique_patients),
            test_patient
        )
    )

    test_cases = cases[
        cases[
            "patient_id"
        ] == test_patient
    ].copy()

    train_cases = cases[
        cases[
            "patient_id"
        ] != test_patient
    ].copy()

    if len(
        train_cases
    ) < 10:

        continue

    # --------------------------------------------------------
    # Target geometry model
    # --------------------------------------------------------

    geometry_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        random_state=RANDOM_SEED
    )

    geometry_model.fit(
        train_cases[
            GEOMETRY_FEATURES
        ].fillna(
            0.0
        ).values,

        train_cases[
            "actual_future_diameter_mm"
        ].values
    )

    geometry_pred = (
        geometry_model.predict(
            test_cases[
                GEOMETRY_FEATURES
            ].fillna(
                0.0
            ).values
        )
    )

    # --------------------------------------------------------
    # Geometry + HBGD
    # --------------------------------------------------------

    fusion_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=3,
        random_state=RANDOM_SEED
    )

    fusion_model.fit(
        train_cases[
            FUSION_FEATURES
        ].fillna(
            0.0
        ).values,

        train_cases[
            "actual_future_diameter_mm"
        ].values
    )

    fusion_pred = (
        fusion_model.predict(
            test_cases[
                FUSION_FEATURES
            ].fillna(
                0.0
            ).values
        )
    )

    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    no_change_pred = (
        test_cases[
            "current_diameter_mm"
        ].values
    )

    direct_transfer_pred = (
        test_cases[
            "current_diameter_mm"
        ].values
        *
        np.exp(
            test_cases[
                "transferred_rate"
            ].values
            *
            test_cases[
                "future_days"
            ].values
        )
    )

    actual = test_cases[
        "actual_future_diameter_mm"
    ].values

    # --------------------------------------------------------
    # Store predictions
    # --------------------------------------------------------

    for i in range(
        len(test_cases)
    ):

        row = test_cases.iloc[
            i
        ]

        prediction_rows.append({

            "patient_id":
                test_patient,

            "current_week":
                row[
                    "current_week"
                ],

            "future_week":
                row[
                    "future_week"
                ],

            "future_days":
                row[
                    "future_days"
                ],

            "actual_diameter_mm":
                actual[i],

            "no_change_pred":
                no_change_pred[i],

            "direct_transfer_pred":
                direct_transfer_pred[i],

            "geometry_pred":
                geometry_pred[i],

            "geometry_hbgd_pred":
                fusion_pred[i],

            "geometry_error_mm":
                abs(
                    actual[i]
                    -
                    geometry_pred[i]
                ),

            "geometry_hbgd_error_mm":
                abs(
                    actual[i]
                    -
                    fusion_pred[i]
                ),

            "no_change_error_mm":
                abs(
                    actual[i]
                    -
                    no_change_pred[i]
                ),

            "direct_transfer_error_mm":
                abs(
                    actual[i]
                    -
                    direct_transfer_pred[i]
                ),

            "improvement_geometry_hbgd_mm":
                abs(
                    actual[i]
                    -
                    geometry_pred[i]
                )
                -
                abs(
                    actual[i]
                    -
                    fusion_pred[i]
                ),

            "hbgd_phenotype":
                row[
                    "hbgd_phenotype"
                ],

            "history_n_timepoints":
                row[
                    "history_n_timepoints"
                ],

            "history_n_intervals":
                row[
                    "history_n_intervals"
                ],

            "history_mean_rate":
                row[
                    "history_mean_rate"
                ],

            "history_rate_std":
                row[
                    "history_rate_std"
                ]
        })

    # --------------------------------------------------------
    # Patient-level performance
    # --------------------------------------------------------

    patient_geometry_mae = (
        mean_absolute_error(
            actual,
            geometry_pred
        )
    )

    patient_fusion_mae = (
        mean_absolute_error(
            actual,
            fusion_pred
        )
    )

    patient_no_change_mae = (
        mean_absolute_error(
            actual,
            no_change_pred
        )
    )

    patient_direct_mae = (
        mean_absolute_error(
            actual,
            direct_transfer_pred
        )
    )

    patient_results.append({

        "patient_id":
            test_patient,

        "n_forecasts":
            len(test_cases),

        "no_change_MAE_mm":
            patient_no_change_mae,

        "direct_transfer_MAE_mm":
            patient_direct_mae,

        "geometry_MAE_mm":
            patient_geometry_mae,

        "geometry_hbgd_MAE_mm":
            patient_fusion_mae,

        "improvement_mm":
            (
                patient_geometry_mae
                -
                patient_fusion_mae
            )
    })


# ============================================================
# DATAFRAMES
# ============================================================

predictions = pd.DataFrame(
    prediction_rows
)

patient_df = pd.DataFrame(
    patient_results
)

if len(
    predictions
) == 0:

    raise RuntimeError(
        "No temporal predictions generated."
    )


# ============================================================
# OVERALL PERFORMANCE
# ============================================================

actual = predictions[
    "actual_diameter_mm"
].values

no_change = predictions[
    "no_change_pred"
].values

direct = predictions[
    "direct_transfer_pred"
].values

geometry = predictions[
    "geometry_pred"
].values

fusion = predictions[
    "geometry_hbgd_pred"
].values


performance_rows = []

models = [

    (
        "M0_No_Change",
        no_change
    ),

    (
        "M1_Geometry",
        geometry
    ),

    (
        "M2_Geometry_Transferred_HBGD",
        fusion
    ),

    (
        "M3_Direct_Transferred_HBGD",
        direct
    )
]


geometry_mae = mean_absolute_error(
    actual,
    geometry
)

for model_name, pred in models:

    mae = mean_absolute_error(
        actual,
        pred
    )

    performance_rows.append({

        "model":
            model_name,

        "n_forecasts":
            len(actual),

        "n_patients":
            predictions[
                "patient_id"
            ].nunique(),

        "MAE_mm":
            mae,

        "RMSE_mm":
            rmse(
                actual,
                pred
            ),

        "improvement_vs_geometry_mm":
            (
                geometry_mae -
                mae
            ),

        "improvement_vs_geometry_pct":
            (
                (
                    geometry_mae -
                    mae
                )
                /
                geometry_mae
                *
                100.0
            )
            if geometry_mae != 0
            else np.nan
    })


performance = pd.DataFrame(
    performance_rows
)


# ============================================================
# PHENOTYPE PERFORMANCE
# ============================================================

phenotype_rows = []

for phenotype in [
    "Growth",
    "Stable",
    "Shrinkage"
]:

    subset = predictions[
        predictions[
            "hbgd_phenotype"
        ] == phenotype
    ]

    if len(subset) == 0:
        continue

    y = subset[
        "actual_diameter_mm"
    ].values

    geo = subset[
        "geometry_pred"
    ].values

    hbgd = subset[
        "geometry_hbgd_pred"
    ].values

    phenotype_rows.append({

        "phenotype":
            phenotype,

        "n_forecasts":
            len(subset),

        "n_patients":
            subset[
                "patient_id"
            ].nunique(),

        "geometry_MAE_mm":
            mean_absolute_error(
                y,
                geo
            ),

        "geometry_hbgd_MAE_mm":
            mean_absolute_error(
                y,
                hbgd
            ),

        "improvement_mm":
            (
                mean_absolute_error(
                    y,
                    geo
                )
                -
                mean_absolute_error(
                    y,
                    hbgd
                )
            )
    })


phenotype_df = pd.DataFrame(
    phenotype_rows
)


# ============================================================
# PATIENT-LEVEL SUMMARY
# ============================================================

patient_mean_improvement = (
    patient_df[
        "improvement_mm"
    ].mean()
)

patient_median_improvement = (
    patient_df[
        "improvement_mm"
    ].median()
)

patient_positive = int(
    np.sum(
        patient_df[
            "improvement_mm"
        ] > 0
    )
)

patient_negative = int(
    np.sum(
        patient_df[
            "improvement_mm"
        ] < 0
    )
)


# ============================================================
# SAVE
# ============================================================

predictions.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)

performance.to_csv(
    OUTPUT_PERFORMANCE,
    index=False
)

patient_df.to_csv(
    OUTPUT_PATIENT,
    index=False
)

phenotype_df.to_csv(
    OUTPUT_PHENOTYPE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

fusion_row = performance[
    performance[
        "model"
    ]
    ==
    "M2_Geometry_Transferred_HBGD"
].iloc[0]

geometry_row = performance[
    performance[
        "model"
    ]
    ==
    "M1_Geometry"
].iloc[0]

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
            target[
                "patient_id"
            ].nunique()
    },

    {
        "metric":
            "eligible_temporal_patients",
        "value":
            eligible_patients
    },

    {
        "metric":
            "temporal_forecasts",
        "value":
            len(predictions)
    },

    {
        "metric":
            "temporal_forecast_patients",
        "value":
            predictions[
                "patient_id"
            ].nunique()
    },

    {
        "metric":
            "geometry_MAE_mm",
        "value":
            geometry_row[
                "MAE_mm"
            ]
    },

    {
        "metric":
            "geometry_hbgd_MAE_mm",
        "value":
            fusion_row[
                "MAE_mm"
            ]
    },

    {
        "metric":
            "geometry_hbgd_improvement_mm",
        "value":
            fusion_row[
                "improvement_vs_geometry_mm"
            ]
    },

    {
        "metric":
            "geometry_hbgd_improvement_pct",
        "value":
            fusion_row[
                "improvement_vs_geometry_pct"
            ]
    },

    {
        "metric":
            "patient_mean_improvement_mm",
        "value":
            patient_mean_improvement
    },

    {
        "metric":
            "patient_median_improvement_mm",
        "value":
            patient_median_improvement
    },

    {
        "metric":
            "patients_improved",
        "value":
            patient_positive
    },

    {
        "metric":
            "patients_worsened",
        "value":
            patient_negative
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
# REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w"
) as f:

    f.write(
        "# Step 25 — Strict Temporal Holdout Validation\n\n"
    )

    f.write(
        "## Objective\n\n"
    )

    f.write(
        "This experiment evaluates whether transferred HBGD "
        "information remains useful when predicting a strictly "
        "later unseen tumor observation using only information "
        "available before the prediction time.\n\n"
    )

    f.write(
        "## Temporal protocol\n\n"
    )

    f.write(
        "For each eligible patient, prediction origins were "
        "constructed chronologically. Historical observations "
        "preceding the prediction origin were used to derive "
        "HBGD features, while the immediately subsequent "
        "observation served as the future target.\n\n"
    )

    f.write(
        "Target-domain regression models were trained using "
        "other patients only for each held-out patient. Thus, "
        "the held-out patient's future measurements were not "
        "used during model fitting.\n\n"
    )

    f.write(
        "## Dataset\n\n"
    )

    f.write(
        "- Source HBGD intervals: %d\n"
        % len(source)
    )

    f.write(
        "- Source patients: %d\n"
        % source[
            "patient_id"
        ].nunique()
    )

    f.write(
        "- LUMIERE patients: %d\n"
        % target[
            "patient_id"
        ].nunique()
    )

    f.write(
        "- Eligible temporal patients: %d\n"
        % eligible_patients
    )

    f.write(
        "- Temporal forecasts: %d\n"
        % len(predictions)
    )

    f.write(
        "- Forecast patients: %d\n\n"
        % predictions[
            "patient_id"
        ].nunique()
    )

    f.write(
        "## Overall performance\n\n"
    )

    f.write(
        "| Model | MAE (mm) | RMSE (mm) | "
        "Improvement vs Geometry (mm) | Improvement (%) |\n"
    )

    f.write(
        "|---|---:|---:|---:|---:|\n"
    )

    for _, row in performance.iterrows():

        f.write(
            "| %s | %.4f | %.4f | %.4f | %.2f%% |\n"
            % (
                row["model"],
                row["MAE_mm"],
                row["RMSE_mm"],
                row[
                    "improvement_vs_geometry_mm"
                ],
                row[
                    "improvement_vs_geometry_pct"
                ]
            )
        )

    f.write(
        "\n"
    )

    f.write(
        "## Phenotype-specific performance\n\n"
    )

    f.write(
        "| Phenotype | Forecasts | Patients | "
        "Geometry MAE | Geometry + HBGD MAE | Improvement |\n"
    )

    f.write(
        "|---|---:|---:|---:|---:|---:|\n"
    )

    for _, row in phenotype_df.iterrows():

        f.write(
            "| %s | %d | %d | %.4f | %.4f | %.4f |\n"
            % (
                row["phenotype"],
                row["n_forecasts"],
                row["n_patients"],
                row["geometry_MAE_mm"],
                row[
                    "geometry_hbgd_MAE_mm"
                ],
                row["improvement_mm"]
            )
        )

    f.write(
        "\n"
    )

    f.write(
        "## Patient-level performance\n\n"
    )

    f.write(
        "- Mean improvement: %.6f mm\n"
        % patient_mean_improvement
    )

    f.write(
        "- Median improvement: %.6f mm\n"
        % patient_median_improvement
    )

    f.write(
        "- Patients improved: %d/%d\n"
        % (
            patient_positive,
            len(patient_df)
        )
    )

    f.write(
        "- Patients worsened: %d/%d\n\n"
        % (
            patient_negative,
            len(patient_df)
        )
    )

    f.write(
        "## Interpretation\n\n"
    )

    if (
        fusion_row[
            "improvement_vs_geometry_mm"
        ] > 0
    ):

        f.write(
            "Under the strict temporal holdout protocol, "
            "Geometry + Transferred HBGD produced a lower pooled "
            "MAE than Geometry alone. This provides directional "
            "evidence that the transferred HBGD representation "
            "may retain complementary information when forecasting "
            "a later unseen observation.\n\n"
        )

    else:

        f.write(
            "Under the strict temporal holdout protocol, "
            "Geometry + Transferred HBGD did not outperform "
            "Geometry alone. This indicates that the previously "
            "observed transfer signal may not generalize to a "
            "strict chronological forecasting setting.\n\n"
        )

    f.write(
        "This analysis remains exploratory and does not establish "
        "clinical validity or universal cross-cancer transferability. "
        "The temporal interpretation also depends on the assumption "
        "that the LUMIERE week index accurately represents elapsed "
        "time.\n\n"
    )

    f.write(
        "## Scientific conclusion\n\n"
    )

    f.write(
        "Strict temporal validation is a stronger test of the "
        "disease-progression forecasting hypothesis than random "
        "case-level evaluation because future observations are "
        "explicitly withheld from feature construction and model "
        "fitting. The results should therefore be interpreted as "
        "evidence for or against temporal generalization of the "
        "transferred HBGD representation.\n"
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 78)
print("STEP 25 COMPLETE")
print("=" * 78)

print(
    "\nEligible temporal patients:",
    eligible_patients
)

print(
    "Temporal forecasts:",
    len(predictions)
)

print(
    "Forecast patients:",
    predictions[
        "patient_id"
    ].nunique()
)

print(
    "\nGEOMETRY MAE:",
    "%.4f mm"
    % geometry_row[
        "MAE_mm"
    ]
)

print(
    "GEOMETRY + HBGD MAE:",
    "%.4f mm"
    % fusion_row[
        "MAE_mm"
    ]
)

print(
    "HBGD IMPROVEMENT:",
    "%.4f mm (%.2f%%)"
    % (
        fusion_row[
            "improvement_vs_geometry_mm"
        ],
        fusion_row[
            "improvement_vs_geometry_pct"
        ]
    )
)

print(
    "\nPATIENT-LEVEL:"
)

print(
    "Mean improvement:",
    "%.4f mm"
    % patient_mean_improvement
)

print(
    "Median improvement:",
    "%.4f mm"
    % patient_median_improvement
)

print(
    "Patients improved:",
    patient_positive,
    "/",
    len(patient_df)
)

print(
    "Patients worsened:",
    patient_negative,
    "/",
    len(patient_df)
)

print(
    "\nOUTPUTS:"
)

print(
    OUTPUT_PREDICTIONS
)

print(
    OUTPUT_PERFORMANCE
)

print(
    OUTPUT_PATIENT
)

print(
    OUTPUT_PHENOTYPE
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