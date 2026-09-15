# -*- coding: utf-8 -*-

"""
STEP 24
HBGD TRANSFER RESPONDER / NON-RESPONDER ANALYSIS

Purpose
-------
Determine whether patients/cases that benefit from transferred HBGD
have distinguishable longitudinal characteristics.

Primary comparison:
    Geometry
        vs
    Geometry + Transferred HBGD

Responder:
    Geometry + HBGD has lower absolute prediction error.

Non-responder:
    Geometry + HBGD has equal or higher absolute prediction error.

Features are calculated only from historical observations available
BEFORE each prediction time.

No future tumor diameter is used to construct responder features.

Python 3.6 compatible.
"""

import os
import math
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

LUMIERE_DIR = (
    r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"
)

STEP22_PATIENT = os.path.join(
    LUMIERE_DIR,
    "step22_hbgd_transfer_patient_results.csv"
)

TRAJECTORIES = os.path.join(
    LUMIERE_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

OUTPUT_CASES = os.path.join(
    LUMIERE_DIR,
    "step24_hbgd_responder_case_features.csv"
)

OUTPUT_PATIENTS = os.path.join(
    LUMIERE_DIR,
    "step24_hbgd_responder_patient_profiles.csv"
)

OUTPUT_GROUPS = os.path.join(
    LUMIERE_DIR,
    "step24_hbgd_responder_group_comparison.csv"
)

OUTPUT_CORR = os.path.join(
    LUMIERE_DIR,
    "step24_hbgd_responder_correlations.csv"
)

OUTPUT_REPORT = os.path.join(
    LUMIERE_DIR,
    "STEP24_HBGD_TRANSFER_RESPONDER_ANALYSIS_REPORT.md"
)


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


def safe_mean(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def safe_median(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.median(values)
    )


def safe_std(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        return 0.0

    return float(
        np.std(
            values,
            ddof=1
        )
    )


def pearson_correlation(
    x,
    y
):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if (
        np.std(x) == 0
        or
        np.std(y) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print(
    "STEP 24 HBGD TRANSFER RESPONDER / NON-RESPONDER ANALYSIS"
)
print("=" * 78)


# ============================================================
# LOAD STEP 22 PATIENT RESULTS
# ============================================================

if not os.path.exists(
    STEP22_PATIENT
):

    raise IOError(
        "Missing Step 22 patient results:\n%s"
        % STEP22_PATIENT
    )

step22 = pd.read_csv(
    STEP22_PATIENT
)

required_step22 = [
    "seed",
    "patient_id",
    "n_cases",
    "geometry_MAE_mm",
    "geometry_hbgd_MAE_mm",
    "improvement_mm"
]

for column in required_step22:

    if column not in step22.columns:

        raise ValueError(
            "Missing Step 22 column: %s"
            % column
        )

step22["patient_id"] = (
    step22["patient_id"]
    .apply(
        normalize_patient_id
    )
)

print(
    "\nStep 22 patient evaluations:",
    len(step22)
)

print(
    "Unique patients:",
    step22["patient_id"].nunique()
)


# ============================================================
# LOAD LUMIERE TRAJECTORIES
# ============================================================

if not os.path.exists(
    TRAJECTORIES
):

    raise IOError(
        "Missing LUMIERE trajectories:\n%s"
        % TRAJECTORIES
    )

traj = pd.read_csv(
    TRAJECTORIES
)

traj["patient_id"] = (
    traj["patient_id"]
    .apply(
        normalize_patient_id
    )
)

required_traj = [
    "patient_id",
    "week",
    "tumor_volume_cm3",
    "equivalent_diameter_mm"
]

for column in required_traj:

    if column not in traj.columns:

        raise ValueError(
            "Missing trajectory column: %s"
            % column
        )

traj["week"] = pd.to_numeric(
    traj["week"],
    errors="coerce"
)

traj[
    "equivalent_diameter_mm"
] = pd.to_numeric(
    traj[
        "equivalent_diameter_mm"
    ],
    errors="coerce"
)

traj[
    "tumor_volume_cm3"
] = pd.to_numeric(
    traj[
        "tumor_volume_cm3"
    ],
    errors="coerce"
)

traj = traj[
    traj["week"].notna()
    &
    traj[
        "equivalent_diameter_mm"
    ].notna()
    &
    (
        traj[
            "equivalent_diameter_mm"
        ] > 0
    )
].copy()

traj["time_days"] = (
    traj["week"] * 7.0
)

traj = traj.sort_values(
    [
        "patient_id",
        "time_days"
    ]
).reset_index(
    drop=True
)


# ============================================================
# BUILD HISTORICAL CASE FEATURES
# ============================================================

print(
    "\nBuilding leakage-controlled historical features..."
)

case_features = []

for patient_id, group in traj.groupby(
    "patient_id"
):

    group = group.sort_values(
        "time_days"
    ).reset_index(
        drop=True
    )

    if len(group) < 3:
        continue

    historical_rates = []

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
            or
            previous_d <= 0
            or
            current_d <= 0
        ):
            continue

        rate = (
            np.log(
                current_d /
                previous_d
            )
            /
            interval_days
        )

        historical_rates.append(
            rate
        )

        # ----------------------------------------------------
        # Current observation becomes prediction origin.
        # Future observation is NOT used in features.
        # ----------------------------------------------------

        if i + 1 >= len(group):
            continue

        history = np.asarray(
            historical_rates,
            dtype=float
        )

        current_volume = (
            current[
                "tumor_volume_cm3"
            ]
        )

        if pd.isna(
            current_volume
        ):
            current_volume = np.nan

        # ----------------------------------------------------
        # HBGD phenotype based ONLY on historical rates
        # ----------------------------------------------------

        mean_rate = np.mean(
            history
        )

        abs_mean_rate = np.mean(
            np.abs(history)
        )

        rate_std = (
            np.std(
                history,
                ddof=1
            )
            if len(history) >= 2
            else 0.0
        )

        if mean_rate > 0.0005:

            phenotype = "Growth"

        elif mean_rate < -0.0010:

            phenotype = "Shrinkage"

        else:

            phenotype = "Stable"

        # ----------------------------------------------------
        # Direction consistency
        # ----------------------------------------------------

        positive_fraction = (
            float(
                np.sum(
                    history > 0
                )
            )
            /
            float(len(history))
        )

        negative_fraction = (
            float(
                np.sum(
                    history < 0
                )
            )
            /
            float(len(history))
        )

        # ----------------------------------------------------
        # Current prediction origin
        # ----------------------------------------------------

        case_features.append({

            "patient_id":
                patient_id,

            "current_week":
                float(
                    current[
                        "week"
                    ]
                ),

            "current_time_days":
                current_t,

            "current_diameter_mm":
                current_d,

            "current_volume_cm3":
                current_volume,

            "history_n_timepoints":
                i + 1,

            "history_n_intervals":
                len(history),

            "history_weeks":
                float(
                    current[
                        "week"
                    ]
                )
                -
                float(
                    group.iloc[0][
                        "week"
                    ]
                ),

            "mean_hbgd_rate_per_day":
                mean_rate,

            "absolute_mean_hbgd_rate_per_day":
                abs_mean_rate,

            "hbgd_rate_std_per_day":
                rate_std,

            "positive_rate_fraction":
                positive_fraction,

            "negative_rate_fraction":
                negative_fraction,

            "hbgd_phenotype":
                phenotype,

            "last_interval_rate_per_day":
                history[-1],

            "last_interval_abs_rate_per_day":
                abs(
                    history[-1]
                ),

            "rate_range_per_day":
                np.max(history)
                -
                np.min(history),

            "history_diameter_change_mm":
                current_d
                -
                float(
                    group.iloc[0][
                        "equivalent_diameter_mm"
                    ]
                ),

            "history_percent_change":
                (
                    (
                        current_d
                        /
                        float(
                            group.iloc[0][
                                "equivalent_diameter_mm"
                            ]
                        )
                    )
                    -
                    1.0
                )
                *
                100.0
        })


features = pd.DataFrame(
    case_features
)

print(
    "Historical prediction-origin cases:",
    len(features)
)


# ============================================================
# MATCH STEP 22 PATIENT-SEED EVALUATIONS
# ============================================================

print(
    "\nMatching historical features to Step 22 evaluations..."
)

# Step 22 patient results contain only aggregated patient/seed
# performance. We therefore aggregate the historical features
# for each patient. These features are descriptive of the
# patient's longitudinal history, not the future outcome.

patient_history = (
    features
    .groupby(
        "patient_id"
    )
    .agg({

        "current_diameter_mm":
            ["mean", "median"],

        "current_volume_cm3":
            ["mean", "median"],

        "history_n_timepoints":
            ["mean", "median", "max"],

        "history_n_intervals":
            ["mean", "median", "max"],

        "history_weeks":
            ["mean", "median", "max"],

        "mean_hbgd_rate_per_day":
            ["mean", "median"],

        "absolute_mean_hbgd_rate_per_day":
            ["mean", "median"],

        "hbgd_rate_std_per_day":
            ["mean", "median"],

        "positive_rate_fraction":
            ["mean"],

        "negative_rate_fraction":
            ["mean"],

        "last_interval_rate_per_day":
            ["mean"],

        "last_interval_abs_rate_per_day":
            ["mean"],

        "rate_range_per_day":
            ["mean"],

        "history_diameter_change_mm":
            ["mean"],

        "history_percent_change":
            ["mean"]
    })
)

patient_history.columns = [
    "_".join(
        [
            str(x)
            for x in column
            if str(x) != ""
        ]
    )
    for column in patient_history.columns
]

patient_history = (
    patient_history
    .reset_index()
)


# ============================================================
# PATIENT PHENOTYPE
# ============================================================

phenotype_table = []

for patient_id, group in features.groupby(
    "patient_id"
):

    phenotype_counts = (
        group[
            "hbgd_phenotype"
        ]
        .value_counts()
    )

    if len(
        phenotype_counts
    ) == 0:

        dominant = "Unknown"

    else:

        dominant = (
            phenotype_counts
            .idxmax()
        )

    phenotype_table.append({

        "patient_id":
            patient_id,

        "dominant_hbgd_phenotype":
            dominant,

        "growth_prediction_origins":
            int(
                phenotype_counts.get(
                    "Growth",
                    0
                )
            ),

        "stable_prediction_origins":
            int(
                phenotype_counts.get(
                    "Stable",
                    0
                )
            ),

        "shrinkage_prediction_origins":
            int(
                phenotype_counts.get(
                    "Shrinkage",
                    0
                )
            )
    })

phenotype_df = pd.DataFrame(
    phenotype_table
)


patient_profiles = (
    step22
    .merge(
        patient_history,
        on="patient_id",
        how="left"
    )
    .merge(
        phenotype_df,
        on="patient_id",
        how="left"
    )
)


# ============================================================
# RESPONDER LABEL
# ============================================================

patient_profiles[
    "responder"
] = (
    patient_profiles[
        "improvement_mm"
    ] > 0
).astype(int)

patient_profiles[
    "response_group"
] = np.where(
    patient_profiles[
        "improvement_mm"
    ] > 0,
    "Responder",
    "Non-responder"
)


# ============================================================
# SAVE PATIENT PROFILES
# ============================================================

patient_profiles.to_csv(
    OUTPUT_PATIENTS,
    index=False
)


# ============================================================
# GROUP COMPARISON
# ============================================================

feature_columns = [

    "current_diameter_mm_mean",
    "current_volume_cm3_mean",

    "history_n_timepoints_mean",
    "history_n_intervals_mean",
    "history_weeks_mean",

    "mean_hbgd_rate_per_day_mean",
    "absolute_mean_hbgd_rate_per_day_mean",

    "hbgd_rate_std_per_day_mean",

    "positive_rate_fraction_mean",
    "negative_rate_fraction_mean",

    "last_interval_rate_per_day_mean",
    "last_interval_abs_rate_per_day_mean",

    "rate_range_per_day_mean",

    "history_diameter_change_mm_mean",
    "history_percent_change_mean"
]


group_rows = []

for feature in feature_columns:

    if feature not in patient_profiles.columns:
        continue

    responder_values = (
        patient_profiles[
            patient_profiles[
                "responder"
            ] == 1
        ][feature]
        .dropna()
        .values
    )

    nonresponder_values = (
        patient_profiles[
            patient_profiles[
                "responder"
            ] == 0
        ][feature]
        .dropna()
        .values
    )

    if (
        len(responder_values) == 0
        or
        len(nonresponder_values) == 0
    ):
        continue

    responder_mean = np.mean(
        responder_values
    )

    nonresponder_mean = np.mean(
        nonresponder_values
    )

    difference = (
        responder_mean
        -
        nonresponder_mean
    )

    group_rows.append({

        "feature":
            feature,

        "responder_n":
            len(responder_values),

        "nonresponder_n":
            len(nonresponder_values),

        "responder_mean":
            responder_mean,

        "responder_median":
            np.median(
                responder_values
            ),

        "nonresponder_mean":
            nonresponder_mean,

        "nonresponder_median":
            np.median(
                nonresponder_values
            ),

        "mean_difference":
            difference
    })


group_comparison = pd.DataFrame(
    group_rows
)

group_comparison.to_csv(
    OUTPUT_GROUPS,
    index=False
)


# ============================================================
# CORRELATION WITH BENEFIT
# ============================================================

print(
    "\nCalculating descriptive correlations..."
)

corr_rows = []

for feature in feature_columns:

    if feature not in patient_profiles.columns:
        continue

    corr = pearson_correlation(
        patient_profiles[
            feature
        ].values,

        patient_profiles[
            "improvement_mm"
        ].values
    )

    corr_rows.append({

        "feature":
            feature,

        "n":
            int(
                patient_profiles[
                    [
                        feature,
                        "improvement_mm"
                    ]
                ]
                .dropna()
                .shape[0]
            ),

        "pearson_r":
            corr
    })


correlation_df = pd.DataFrame(
    corr_rows
)

correlation_df[
    "absolute_pearson_r"
] = np.abs(
    correlation_df[
        "pearson_r"
    ]
)

correlation_df = (
    correlation_df
    .sort_values(
        "absolute_pearson_r",
        ascending=False
    )
)


correlation_df.to_csv(
    OUTPUT_CORR,
    index=False
)


# ============================================================
# CASE-LEVEL OUTPUT
# ============================================================

# Add a simple patient-level response label to every historical
# prediction-origin feature for interpretive analysis.

case_output = features.merge(
    patient_profiles[
        [
            "seed",
            "patient_id",
            "improvement_mm",
            "responder",
            "response_group"
        ]
    ],
    on="patient_id",
    how="left"
)

case_output.to_csv(
    OUTPUT_CASES,
    index=False
)


# ============================================================
# RESPONDER COUNTS
# ============================================================

n_total = len(
    patient_profiles
)

n_responder = int(
    np.sum(
        patient_profiles[
            "responder"
        ] == 1
    )
)

n_nonresponder = int(
    np.sum(
        patient_profiles[
            "responder"
        ] == 0
    )
)


# ============================================================
# PHENOTYPE RESPONSE ANALYSIS
# ============================================================

phenotype_rows = []

for phenotype in [
    "Growth",
    "Stable",
    "Shrinkage"
]:

    subset = patient_profiles[
        patient_profiles[
            "dominant_hbgd_phenotype"
        ] == phenotype
    ]

    if len(subset) == 0:
        continue

    phenotype_rows.append({

        "phenotype":
            phenotype,

        "n_patient_evaluations":
            len(subset),

        "responders":
            int(
                np.sum(
                    subset[
                        "responder"
                    ] == 1
                )
            ),

        "nonresponders":
            int(
                np.sum(
                    subset[
                        "responder"
                    ] == 0
                )
            ),

        "responder_fraction":
            float(
                np.sum(
                    subset[
                        "responder"
                    ] == 1
                )
            )
            /
            float(len(subset)),

        "mean_improvement_mm":
            subset[
                "improvement_mm"
            ].mean(),

        "median_improvement_mm":
            subset[
                "improvement_mm"
            ].median()
    })

phenotype_response_df = pd.DataFrame(
    phenotype_rows
)


# ============================================================
# REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w"
) as f:

    f.write(
        "# Step 24 — HBGD Transfer Responder / Non-Responder Analysis\n\n"
    )

    f.write(
        "## Objective\n\n"
    )

    f.write(
        "Step 24 investigates whether the incremental benefit "
        "of transferred HBGD information is concentrated in "
        "specific longitudinal tumor-growth characteristics.\n\n"
    )

    f.write(
        "A patient-seed evaluation is considered a **Responder** "
        "when Geometry + Transferred HBGD has lower MAE than "
        "Geometry alone. Otherwise it is classified as a "
        "**Non-responder**.\n\n"
    )

    f.write(
        "## Leakage control\n\n"
    )

    f.write(
        "All longitudinal characteristics were calculated from "
        "observations available before the corresponding prediction "
        "origin. Future tumor diameter was not used to construct "
        "the historical HBGD features or phenotype labels.\n\n"
    )

    f.write(
        "## Response distribution\n\n"
    )

    f.write(
        "- Total patient-seed evaluations: %d\n"
        % n_total
    )

    f.write(
        "- Responders: %d\n"
        % n_responder
    )

    f.write(
        "- Non-responders: %d\n"
        % n_nonresponder
    )

    f.write(
        "- Responder fraction: %.2f%%\n\n"
        % (
            100.0 *
            float(n_responder) /
            float(n_total)
            if n_total > 0
            else np.nan
        )
    )

    f.write(
        "## Responder versus non-responder features\n\n"
    )

    f.write(
        "| Feature | Responder mean | Non-responder mean | Difference |\n"
    )

    f.write(
        "|---|---:|---:|---:|\n"
    )

    for _, row in group_comparison.iterrows():

        f.write(
            "| %s | %.6f | %.6f | %.6f |\n"
            % (
                row["feature"],
                row["responder_mean"],
                row["nonresponder_mean"],
                row["mean_difference"]
            )
        )

    f.write(
        "\n"
    )

    f.write(
        "## Phenotype response\n\n"
    )

    f.write(
        "| Phenotype | N | Responders | "
        "Non-responders | Responder fraction | "
        "Mean improvement |\n"
    )

    f.write(
        "|---|---:|---:|---:|---:|---:|\n"
    )

    for _, row in phenotype_response_df.iterrows():

        f.write(
            "| %s | %d | %d | %d | %.2f%% | %.6f mm |\n"
            % (
                row["phenotype"],
                row[
                    "n_patient_evaluations"
                ],
                row["responders"],
                row["nonresponders"],
                row["responder_fraction"] * 100.0,
                row["mean_improvement_mm"]
            )
        )

    f.write(
        "\n"
    )

    f.write(
        "## Descriptive benefit correlations\n\n"
    )

    f.write(
        "| Feature | N | Pearson r |\n"
    )

    f.write(
        "|---|---:|---:|\n"
    )

    for _, row in correlation_df.head(
        15
    ).iterrows():

        f.write(
            "| %s | %d | %.4f |\n"
            % (
                row["feature"],
                int(row["n"]),
                row["pearson_r"]
            )
        )

    f.write(
        "\n"
    )

    f.write(
        "## Interpretation\n\n"
    )

    f.write(
        "This responder analysis is exploratory and hypothesis-"
        "generating. Differences between responders and "
        "non-responders do not establish causality, statistical "
        "significance, or clinical utility.\n\n"
    )

    f.write(
        "The analysis is intended to identify longitudinal "
        "characteristics that may explain heterogeneity in the "
        "incremental contribution of HBGD information.\n\n"
    )

    f.write(
        "## Scientific conclusion\n\n"
    )

    f.write(
        "The cross-cancer HBGD transfer effect should not be "
        "interpreted as uniformly beneficial across all patients. "
        "Instead, Step 24 evaluates whether its usefulness may "
        "depend on the underlying longitudinal growth phenotype, "
        "growth-rate stability, temporal history, or tumor "
        "characteristics.\n\n"
    )

    f.write(
        "Any observed responder pattern should be considered "
        "hypothesis-generating until independently reproduced "
        "in a larger cohort.\n"
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

print("\n")
print("=" * 78)
print("STEP 24 COMPLETE")
print("=" * 78)

print(
    "\nTotal patient-seed evaluations:",
    n_total
)

print(
    "Responders:",
    n_responder
)

print(
    "Non-responders:",
    n_nonresponder
)

print(
    "Responder fraction:",
    "%.2f%%"
    % (
        100.0 *
        float(n_responder) /
        float(n_total)
    )
)

print(
    "\nTOP DESCRIPTIVE CORRELATIONS:"
)

for _, row in correlation_df.head(
    8
).iterrows():

    print(
        "  %-45s r = %+.4f"
        % (
            row["feature"],
            row["pearson_r"]
        )
    )

print(
    "\nPHENOTYPE RESPONSE:"
)

for _, row in phenotype_response_df.iterrows():

    print(
        "  %-10s N=%d  responders=%d  "
        "fraction=%.1f%%  mean improvement=%+.4f mm"
        % (
            row["phenotype"],
            row[
                "n_patient_evaluations"
            ],
            row["responders"],
            row[
                "responder_fraction"
            ] * 100.0,
            row[
                "mean_improvement_mm"
            ]
        )
    )

print(
    "\nOUTPUTS:"
)

print(
    OUTPUT_CASES
)

print(
    OUTPUT_PATIENTS
)

print(
    OUTPUT_GROUPS
)

print(
    OUTPUT_CORR
)

print(
    OUTPUT_REPORT
)

print(
    "\n" + "=" * 78
)