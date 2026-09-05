# 11_validate_growth_dynamics.py

import os
import sys
import warnings

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import ttest_1samp
from scipy.stats import wilcoxon


# ============================================================
# TERMINAL ENCODING
# ============================================================

try:
    sys.stdout.reconfigure(
        encoding="utf-8",
        errors="replace"
    )
except Exception:
    pass


warnings.filterwarnings(
    "ignore"
)


# ============================================================
# PATHS
# ============================================================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "DATA"
)

PAIRWISE_FILE = os.path.join(
    DATA_DIR,
    "8_recist_pairwise_changes.csv"
)

FEATURE_FILE = os.path.join(
    DATA_DIR,
    "9_hbgd_features.csv"
)

HOLDOUT_FILE = os.path.join(
    DATA_DIR,
    "10_growth_model_holdout.csv"
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
# LOAD FILE
# ============================================================

def load_csv(path, name):

    if not os.path.exists(path):

        print(
            "ERROR: Required file not found:"
        )

        print(path)

        sys.exit(1)

    df = pd.read_csv(path)

    print(
        name,
        "records:",
        len(df)
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

header(
    "STEP 11 - LOADING DATA"
)

pairwise = load_csv(
    PAIRWISE_FILE,
    "Pairwise RECIST"
)

features = load_csv(
    FEATURE_FILE,
    "HBGD features"
)

holdout = load_csv(
    HOLDOUT_FILE,
    "Growth-model holdout"
)


# ============================================================
# NORMALIZE PAIRWISE DATA
# ============================================================

header(
    "NORMALIZING PAIRWISE DATA"
)

numeric_columns = [

    "patient_id",

    "interval_days",

    "previous_diameter_mm",

    "current_diameter_mm",

    "diameter_change_mm",

    "percent_change"
]


for column in numeric_columns:

    if column in pairwise.columns:

        pairwise[column] = pd.to_numeric(
            pairwise[column],
            errors="coerce"
        )


pairwise = pairwise[
    pairwise[
        "interval_days"
    ] > 0
].copy()


pairwise = pairwise[
    pairwise[
        "previous_diameter_mm"
    ] > 0
].copy()


pairwise = pairwise[
    pairwise[
        "current_diameter_mm"
    ] > 0
].copy()


pairwise = pairwise.reset_index(
    drop=True
)


print(
    "Valid pairwise transitions:",
    len(pairwise)
)


# ============================================================
# 1. GROWTH RATE STABILITY
# ============================================================

header(
    "1 - GROWTH RATE STABILITY WITHIN LESIONS"
)


pairwise[
    "linear_rate_mm_per_day"
] = (
    pairwise[
        "diameter_change_mm"
    ]
    /
    pairwise[
        "interval_days"
    ]
)


pairwise[
    "log_growth_rate_per_day"
] = (
    np.log(
        pairwise[
            "current_diameter_mm"
        ]
        /
        pairwise[
            "previous_diameter_mm"
        ]
    )
    /
    pairwise[
        "interval_days"
    ]
)


rate_groups = []

for (
    patient_id,
    alias
), group in pairwise.groupby(
    [
        "patient_id",
        "lesion_alias"
    ]
):

    rates = group[
        "log_growth_rate_per_day"
    ].dropna()

    if len(rates) == 0:

        continue

    rate_groups.append({

        "patient_id":
            patient_id,

        "lesion_alias":
            alias,

        "transitions":
            len(rates),

        "mean_rate":
            rates.mean(),

        "median_rate":
            rates.median(),

        "std_rate":
            rates.std()
            if len(rates) > 1
            else 0.0,

        "min_rate":
            rates.min(),

        "max_rate":
            rates.max(),

        "rate_range":
            rates.max()
            -
            rates.min()
    })


rate_stability = pd.DataFrame(
    rate_groups
)


print(
    "Lesion trajectories analyzed:",
    len(rate_stability)
)


if len(rate_stability) > 0:

    multi_transition = rate_stability[
        rate_stability[
            "transitions"
        ] >= 2
    ]

    print(
        "Lesions with >=2 transitions:",
        len(multi_transition)
    )

    if len(multi_transition) > 0:

        print(
            "Median within-lesion rate range:",
            round(
                multi_transition[
                    "rate_range"
                ].median(),
                6
            )
        )

        print(
            "Mean within-lesion rate range:",
            round(
                multi_transition[
                    "rate_range"
                ].mean(),
                6
            )
        )


# ============================================================
# 2. PATIENT-TO-PATIENT VARIABILITY
# ============================================================

header(
    "2 - PATIENT-TO-PATIENT GROWTH VARIABILITY"
)


patient_rates = (
    pairwise
    .groupby(
        "patient_id"
    )[
        "log_growth_rate_per_day"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std"
        ]
    )
    .reset_index()
)


print(
    "Patients represented:",
    len(patient_rates)
)


if len(patient_rates) > 0:

    print(
        "Mean patient median growth rate:",
        round(
            patient_rates[
                "median"
            ].mean(),
            6
        )
    )

    print(
        "Median patient median growth rate:",
        round(
            patient_rates[
                "median"
            ].median(),
            6
        )
    )

    print(
        "Patient-level SD:",
        round(
            patient_rates[
                "median"
            ].std(),
            6
        )
    )


# ============================================================
# 3. TEST WHETHER GROWTH RATE IS DIFFERENT FROM ZERO
# ============================================================

header(
    "3 - TESTING WHETHER OBSERVED RATE DIFFERS FROM ZERO"
)


rates = pairwise[
    "log_growth_rate_per_day"
].dropna()


if len(rates) >= 2:

    t_stat, t_p = ttest_1samp(
        rates,
        0.0
    )

    print(
        "One-sample t-test:"
    )

    print(
        "  t-statistic:",
        round(
            t_stat,
            5
        )
    )

    print(
        "  p-value:",
        round(
            t_p,
            6
        )
    )


    try:

        w_stat, w_p = wilcoxon(
            rates
        )

        print()

        print(
            "Wilcoxon signed-rank test:"
        )

        print(
            "  statistic:",
            round(
                float(w_stat),
                5
            )
        )

        print(
            "  p-value:",
            round(
                float(w_p),
                6
            )
        )

    except Exception as exc:

        print(
            "Wilcoxon test unavailable:",
            exc
        )


# ============================================================
# 4. DIAMETER CHANGE VS TIME INTERVAL
# ============================================================

header(
    "4 - RELATIONSHIP BETWEEN CHANGE AND TIME"
)


valid = pairwise[
    pairwise[
        "interval_days"
    ].notna()
    &
    pairwise[
        "diameter_change_mm"
    ].notna()
].copy()


if len(valid) >= 3:

    pearson_r, pearson_p = pearsonr(
        valid[
            "interval_days"
        ],
        valid[
            "diameter_change_mm"
        ]
    )

    spearman_r, spearman_p = spearmanr(
        valid[
            "interval_days"
        ],
        valid[
            "diameter_change_mm"
        ]
    )


    print(
        "Pearson correlation:",
        round(
            pearson_r,
            5
        )
    )

    print(
        "Pearson p-value:",
        round(
            pearson_p,
            6
        )
    )

    print(
        "Spearman correlation:",
        round(
            spearman_r,
            5
        )
    )

    print(
        "Spearman p-value:",
        round(
            spearman_p,
            6
        )
    )


# ============================================================
# 5. BASELINE DIAMETER VS SUBSEQUENT CHANGE
# ============================================================

header(
    "5 - BASELINE DIAMETER VS SUBSEQUENT CHANGE"
)


if (
    "previous_diameter_mm"
    in pairwise.columns
    and
    "diameter_change_mm"
    in pairwise.columns
):

    x = pairwise[
        "previous_diameter_mm"
    ]

    y = pairwise[
        "diameter_change_mm"
    ]


    valid_xy = pd.DataFrame({

        "x": x,

        "y": y

    }).dropna()


    if len(valid_xy) >= 3:

        r, p = pearsonr(
            valid_xy["x"],
            valid_xy["y"]
        )

        sr, sp = spearmanr(
            valid_xy["x"],
            valid_xy["y"]
        )


        print(
            "Pearson correlation:",
            round(
                r,
                5
            )
        )

        print(
            "Pearson p-value:",
            round(
                p,
                6
            )
        )

        print(
            "Spearman correlation:",
            round(
                sr,
                5
            )
        )

        print(
            "Spearman p-value:",
            round(
                sp,
                6
            )
        )


# ============================================================
# 6. BASELINE DIAMETER VS GROWTH RATE
# ============================================================

header(
    "6 - BASELINE DIAMETER VS LOG GROWTH RATE"
)


valid = pairwise[
    pairwise[
        "previous_diameter_mm"
    ].notna()
    &
    pairwise[
        "log_growth_rate_per_day"
    ].notna()
].copy()


if len(valid) >= 3:

    r, p = pearsonr(
        valid[
            "previous_diameter_mm"
        ],
        valid[
            "log_growth_rate_per_day"
        ]
    )

    sr, sp = spearmanr(
        valid[
            "previous_diameter_mm"
        ],
        valid[
            "log_growth_rate_per_day"
        ]
    )


    print(
        "Pearson correlation:",
        round(
            r,
            5
        )
    )

    print(
        "Pearson p-value:",
        round(
            p,
            6
        )
    )

    print(
        "Spearman correlation:",
        round(
            sr,
            5
        )
    )

    print(
        "Spearman p-value:",
        round(
            sp,
            6
        )
    )


# ============================================================
# 7. VOLUME-DERIVED INFORMATION
# ============================================================

header(
    "7 - VOLUME VS RECIST DIAMETER"
)


required_volume_columns = [

    "volume_cm3",

    "diameter_mm",

    "equivalent_diameter_mm"
]


volume_missing = [

    c
    for c in required_volume_columns
    if c not in features.columns
]


if len(volume_missing) == 0:

    valid_volume = features[
        required_volume_columns
    ].dropna()


    print(
        "Valid volume observations:",
        len(valid_volume)
    )


    if len(valid_volume) >= 3:

        r, p = pearsonr(
            valid_volume[
                "volume_cm3"
            ],
            valid_volume[
                "diameter_mm"
            ]
        )

        sr, sp = spearmanr(
            valid_volume[
                "volume_cm3"
            ],
            valid_volume[
                "diameter_mm"
            ]
        )


        print(
            "Volume vs RECIST diameter Pearson r:",
            round(
                r,
                5
            )
        )

        print(
            "Pearson p-value:",
            round(
                p,
                6
            )
        )

        print(
            "Volume vs RECIST diameter Spearman rho:",
            round(
                sr,
                5
            )
        )

        print(
            "Spearman p-value:",
            round(
                sp,
                6
            )
        )


# ============================================================
# 8. VOLUME CHANGE VS DIAMETER CHANGE
# ============================================================

header(
    "8 - VOLUME CHANGE VS DIAMETER CHANGE"
)


# Build longitudinal volume change only when the
# same patient/alias has consecutive observations.

volume = features.copy()


volume["study_date"] = pd.to_datetime(
    volume["study_date"],
    errors="coerce"
)


volume = volume.sort_values(
    [
        "patient_id",
        "lesion_label_alias",
        "study_date"
    ]
)


volume[
    "previous_volume_cm3"
] = (
    volume
    .groupby(
        [
            "patient_id",
            "lesion_label_alias"
        ]
    )[
        "volume_cm3"
    ]
    .shift(1)
)


volume[
    "previous_diameter_mm"
] = (
    volume
    .groupby(
        [
            "patient_id",
            "lesion_label_alias"
        ]
    )[
        "diameter_mm"
    ]
    .shift(1)
)


volume[
    "volume_change_cm3"
] = (
    volume[
        "volume_cm3"
    ]
    -
    volume[
        "previous_volume_cm3"
    ]
)


volume[
    "diameter_change_mm"
] = (
    volume[
        "diameter_mm"
    ]
    -
    volume[
        "previous_diameter_mm"
    ]
)


volume_pairs = volume[
    volume[
        "previous_volume_cm3"
    ].notna()
    &
    volume[
        "previous_diameter_mm"
    ].notna()
].copy()


print(
    "Longitudinal volume pairs:",
    len(volume_pairs)
)


if len(volume_pairs) >= 3:

    r, p = pearsonr(
        volume_pairs[
            "volume_change_cm3"
        ],
        volume_pairs[
            "diameter_change_mm"
        ]
    )

    sr, sp = spearmanr(
        volume_pairs[
            "volume_change_cm3"
        ],
        volume_pairs[
            "diameter_change_mm"
        ]
    )


    print(
        "Pearson correlation:",
        round(
            r,
            5
        )
    )

    print(
        "Pearson p-value:",
        round(
            p,
            6
        )
    )

    print(
        "Spearman correlation:",
        round(
            sr,
            5
        )
    )

    print(
        "Spearman p-value:",
        round(
            sp,
            6
        )
    )


# ============================================================
# 9. BASELINE-ONLY PREDICTOR
# ============================================================

header(
    "9 - BASELINE-ONLY PREDICTOR"
)


# The simplest possible prediction is:
#
# future diameter = current diameter
#
# This means "no change".
#
# We compare this against the growth-rate prediction
# already calculated in Step 10.


if len(holdout) > 0:

    required_holdout = [

        "test_initial_diameter_mm",

        "observed_diameter_mm",

        "predicted_diameter_mm"

    ]


    missing_holdout = [

        c
        for c in required_holdout
        if c not in holdout.columns
    ]


    if len(missing_holdout) == 0:

        baseline_prediction = (
            holdout[
                "test_initial_diameter_mm"
            ]
        )

        observed = (
            holdout[
                "observed_diameter_mm"
            ]
        )

        growth_prediction = (
            holdout[
                "predicted_diameter_mm"
            ]
        )


        baseline_error = (
            baseline_prediction
            -
            observed
        ).abs()


        growth_error = (
            growth_prediction
            -
            observed
        ).abs()


        baseline_mae = (
            baseline_error.mean()
        )

        growth_mae = (
            growth_error.mean()
        )


        baseline_rmse = np.sqrt(
            np.mean(
                (
                    baseline_prediction
                    -
                    observed
                ) ** 2
            )
        )


        growth_rmse = np.sqrt(
            np.mean(
                (
                    growth_prediction
                    -
                    observed
                ) ** 2
            )
        )


        print(
            "Baseline/no-change MAE:",
            round(
                baseline_mae,
                3
            ),
            "mm"
        )

        print(
            "Growth-rate model MAE:",
            round(
                growth_mae,
                3
            ),
            "mm"
        )

        print()

        print(
            "Baseline/no-change RMSE:",
            round(
                baseline_rmse,
                3
            ),
            "mm"
        )

        print(
            "Growth-rate model RMSE:",
            round(
                growth_rmse,
                3
            ),
            "mm"
        )


        if baseline_mae > 0:

            improvement = (
                (
                    baseline_mae
                    -
                    growth_mae
                )
                /
                baseline_mae
            ) * 100.0

        else:

            improvement = np.nan


        print()

        print(
            "MAE improvement from growth model:",
            round(
                improvement,
                2
            ),
            "%"
        )


# ============================================================
# 10. MEASUREMENT-SCALE CONTEXT
# ============================================================

header(
    "10 - CHANGE MAGNITUDE CONTEXT"
)


changes = pairwise[
    "diameter_change_mm"
].dropna()


if len(changes) > 0:

    absolute_changes = (
        changes.abs()
    )


    print(
        "Median absolute diameter change:",
        round(
            absolute_changes.median(),
            3
        ),
        "mm"
    )

    print(
        "Mean absolute diameter change:",
        round(
            absolute_changes.mean(),
            3
        ),
        "mm"
    )

    print(
        "Minimum absolute change:",
        round(
            absolute_changes.min(),
            3
        ),
        "mm"
    )

    print(
        "Maximum absolute change:",
        round(
            absolute_changes.max(),
            3
        ),
        "mm"
    )


# ============================================================
# 11. OUTLIER CHECK
# ============================================================

header(
    "11 - EXTREME GROWTH-RATE CHECK"
)


if len(rates) > 0:

    q1 = rates.quantile(
        0.25
    )

    q3 = rates.quantile(
        0.75
    )

    iqr = (
        q3 - q1
    )


    lower = (
        q1
        -
        1.5 * iqr
    )

    upper = (
        q3
        +
        1.5 * iqr
    )


    outliers = rates[
        (
            rates < lower
        )
        |
        (
            rates > upper
        )
    ]


    print(
        "Q1:",
        round(
            q1,
            6
        )
    )

    print(
        "Q3:",
        round(
            q3,
            6
        )
    )

    print(
        "IQR:",
        round(
            iqr,
            6
        )
    )

    print(
        "Lower bound:",
        round(
            lower,
            6
        )
    )

    print(
        "Upper bound:",
        round(
            upper,
            6
        )
    )

    print(
        "Potential rate outliers:",
        len(outliers)
    )


# ============================================================
# 12. OVERALL INTERPRETATION
# ============================================================

header(
    "STEP 11 - OVERALL INTERPRETATION"
)


print("""
This step evaluates whether the observed longitudinal
trajectories contain a stable dynamical signal.

The key questions are:

1. Are growth rates reasonably consistent within lesions?

2. Is there meaningful variation between patients?

3. Is the observed change different from zero?

4. Does baseline size explain subsequent change?

5. Does segmented volume contain information beyond
   RECIST diameter?

6. Does a growth-rate model outperform a simple
   no-change baseline predictor?

These tests do not establish clinical validity.

They determine whether there is sufficient quantitative
evidence to justify developing the more complex HBGD model.
""")


# ============================================================
# FINAL DECISION
# ============================================================

header(
    "STEP 11 - FEASIBILITY DECISION"
)


has_longitudinal = (
    len(pairwise) > 0
)

has_multiple_patients = (
    len(patient_rates) >= 2
)

has_holdout = (
    len(holdout) > 0
)

has_volume = (
    len(volume_pairs) > 0
)


print(
    "["
    +
    (
        "YES"
        if has_longitudinal
        else "NO"
    )
    +
    "] Longitudinal growth observations"
)


print(
    "["
    +
    (
        "YES"
        if has_multiple_patients
        else "NO"
    )
    +
    "] Multiple-patient growth variability"
)


print(
    "["
    +
    (
        "YES"
        if has_holdout
        else "NO"
    )
    +
    "] Independent later-transition evaluation"
)


print(
    "["
    +
    (
        "YES"
        if has_volume
        else "NO"
    )
    +
    "] Longitudinal volume information"
)


# ============================================================
# SAVE RESULTS
# ============================================================

header(
    "SAVING STEP 11 RESULTS"
)


rate_file = os.path.join(
    DATA_DIR,
    "11_growth_rate_stability.csv"
)

patient_file = os.path.join(
    DATA_DIR,
    "11_patient_growth_statistics.csv"
)

volume_file = os.path.join(
    DATA_DIR,
    "11_volume_longitudinal_analysis.csv"
)


pairwise.to_csv(
    rate_file,
    index=False
)

patient_rates.to_csv(
    patient_file,
    index=False
)

volume_pairs.to_csv(
    volume_file,
    index=False
)


print(
    "Saved:",
    rate_file
)

print(
    "Saved:",
    patient_file
)

print(
    "Saved:",
    volume_file
)


# ============================================================
# COMPLETE
# ============================================================

header(
    "STEP 11 COMPLETE"
)

print("""
Run:

    python 11_validate_growth_dynamics.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A NEURAL NETWORK YET.
""")

print(
    "=" * 95
)

print(
    "END OF STEP 11"
)

print(
    "=" * 95
)