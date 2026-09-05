# 15_analyze_hbgd_state_transitions.py

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


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

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "DATA"
)

STATE_FILE = os.path.join(
    DATA_DIR,
    "14_hbgd_dynamic_states.csv"
)

TRANSITION_FILE = os.path.join(
    DATA_DIR,
    "14_hbgd_state_transitions.csv"
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
    "STEP 15 - LOADING HBGD STATE TRANSITION DATA"
)


if not os.path.exists(STATE_FILE):

    print("ERROR: Missing file:")
    print(STATE_FILE)
    sys.exit(1)


if not os.path.exists(TRANSITION_FILE):

    print("ERROR: Missing file:")
    print(TRANSITION_FILE)
    sys.exit(1)


states = pd.read_csv(
    STATE_FILE
)

transitions = pd.read_csv(
    TRANSITION_FILE
)


print(
    "Dynamic state records:",
    len(states)
)

print(
    "State transition records:",
    len(transitions)
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

header(
    "CHECKING REQUIRED VARIABLES"
)


required_state_columns = [
    "patient_id",
    "lesion_alias",
    "dynamic_state",
    "volume_state"
]


required_transition_columns = [
    "patient_id",
    "lesion_alias",
    "previous_state",
    "current_state"
]


missing_state = [
    c
    for c in required_state_columns
    if c not in states.columns
]


missing_transition = [
    c
    for c in required_transition_columns
    if c not in transitions.columns
]


if missing_state:

    print(
        "ERROR: Missing state columns:"
    )

    for c in missing_state:
        print("   ", c)

    sys.exit(1)


if missing_transition:

    print(
        "ERROR: Missing transition columns:"
    )

    for c in missing_transition:
        print("   ", c)

    sys.exit(1)


print(
    "All required variables are present."
)


# ============================================================
# STATE DISTRIBUTION
# ============================================================

header(
    "OBSERVED STATE DISTRIBUTION"
)


state_counts = (
    states["dynamic_state"]
    .value_counts()
)


print(
    state_counts.to_string()
)


# ============================================================
# STATE ORDER
# ============================================================

state_order = [
    "growth",
    "stable",
    "shrinkage"
]


# ============================================================
# TRANSITION MATRIX
# ============================================================

header(
    "OBSERVED STATE TRANSITION MATRIX"
)


transition_matrix = pd.crosstab(
    transitions["previous_state"],
    transitions["current_state"]
)


transition_matrix = transition_matrix.reindex(
    index=state_order,
    columns=state_order,
    fill_value=0
)


print(
    transition_matrix.to_string()
)


# ============================================================
# TRANSITION PROBABILITIES
# ============================================================

header(
    "STATE TRANSITION PROBABILITIES"
)


row_totals = transition_matrix.sum(
    axis=1
)


transition_probabilities = (
    transition_matrix
    .div(
        row_totals.replace(
            0,
            np.nan
        ),
        axis=0
    )
    .fillna(0)
)


print(
    transition_probabilities.round(
        4
    ).to_string()
)


print()
print("Rows = previous state")
print("Columns = next state")


# ============================================================
# STATE PERSISTENCE
# ============================================================

header(
    "STATE PERSISTENCE"
)


for state in state_order:

    total = transition_matrix.loc[
        state
    ].sum()

    same = transition_matrix.loc[
        state,
        state
    ]

    if total > 0:

        persistence = (
            same / total
        ) * 100.0

    else:

        persistence = np.nan

    print(
        state,
        ":",
        int(same),
        "/",
        int(total),
        "=",
        round(
            persistence,
            2
        ),
        "%"
    )


# ============================================================
# STATE SWITCHING
# ============================================================

header(
    "STATE SWITCHING"
)


same_state = (
    transitions["previous_state"]
    ==
    transitions["current_state"]
)


switching = ~same_state


print(
    "Total transitions:",
    len(transitions)
)

print(
    "Same-state transitions:",
    int(
        same_state.sum()
    )
)

print(
    "State switches:",
    int(
        switching.sum()
    )
)


if len(transitions) > 0:

    persistence_percent = (
        same_state.mean()
        *
        100.0
    )

    switching_percent = (
        switching.mean()
        *
        100.0
    )

else:

    persistence_percent = np.nan
    switching_percent = np.nan


print(
    "Overall persistence:",
    round(
        persistence_percent,
        2
    ),
    "%"
)

print(
    "Overall switching:",
    round(
        switching_percent,
        2
    ),
    "%"
)


# ============================================================
# TRANSITION TYPES
# ============================================================

header(
    "INDIVIDUAL TRANSITION TYPES"
)


transition_counts = (
    transitions
    .groupby(
        [
            "previous_state",
            "current_state"
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
)


transition_counts["percentage"] = (
    transition_counts["count"]
    /
    len(transitions)
    *
    100.0
)


print(
    transition_counts.to_string(
        index=False
    )
)


# ============================================================
# GROWTH STATE
# ============================================================

header(
    "GROWTH STATE ANALYSIS"
)


growth_transitions = transitions[
    transitions["previous_state"]
    ==
    "growth"
]


print(
    "Growth-state transitions:",
    len(growth_transitions)
)


if len(growth_transitions) > 0:

    print(
        "Next state distribution:"
    )

    print(
        growth_transitions[
            "current_state"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# SHRINKAGE STATE
# ============================================================

header(
    "SHRINKAGE STATE ANALYSIS"
)


shrinkage_transitions = transitions[
    transitions["previous_state"]
    ==
    "shrinkage"
]


print(
    "Shrinkage-state transitions:",
    len(shrinkage_transitions)
)


if len(shrinkage_transitions) > 0:

    print(
        "Next state distribution:"
    )

    print(
        shrinkage_transitions[
            "current_state"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# STABLE STATE
# ============================================================

header(
    "STABLE STATE ANALYSIS"
)


stable_transitions = transitions[
    transitions["previous_state"]
    ==
    "stable"
]


print(
    "Stable-state transitions:",
    len(stable_transitions)
)


if len(stable_transitions) > 0:

    print(
        "Next state distribution:"
    )

    print(
        stable_transitions[
            "current_state"
        ]
        .value_counts()
        .to_string()
    )


# ============================================================
# PATIENT LEVEL
# ============================================================

header(
    "PATIENT-LEVEL STATE TRANSITION STRUCTURE"
)


patient_transition_records = []


for patient_id, group in transitions.groupby(
    "patient_id"
):

    same = (
        group["previous_state"]
        ==
        group["current_state"]
    )

    patient_transition_records.append({

        "patient_id":
            patient_id,

        "transitions":
            len(group),

        "same_state":
            int(
                same.sum()
            ),

        "state_switches":
            int(
                (~same).sum()
            ),

        "persistence_percent":
            same.mean() * 100.0
    })


patient_transition_summary = pd.DataFrame(
    patient_transition_records
)


print(
    patient_transition_summary.to_string(
        index=False
    )
)


# ============================================================
# LESION LEVEL
# ============================================================

header(
    "LESION-LEVEL STATE TRANSITION STRUCTURE"
)


lesion_transition_records = []


for (
    patient_id,
    alias
), group in transitions.groupby(
    [
        "patient_id",
        "lesion_alias"
    ]
):

    same = (
        group["previous_state"]
        ==
        group["current_state"]
    )

    lesion_transition_records.append({

        "patient_id":
            patient_id,

        "lesion_alias":
            alias,

        "transitions":
            len(group),

        "same_state":
            int(
                same.sum()
            ),

        "switches":
            int(
                (~same).sum()
            ),

        "persistence_percent":
            same.mean() * 100.0
    })


lesion_transition_summary = pd.DataFrame(
    lesion_transition_records
)


print(
    lesion_transition_summary.to_string(
        index=False
    )
)


# ============================================================
# DIAMETER VS VOLUME STATE
# ============================================================

header(
    "DIAMETER STATE VS VOLUME STATE"
)


volume_crosstab = pd.crosstab(
    states["dynamic_state"],
    states["volume_state"]
)


print(
    volume_crosstab.to_string()
)


# ============================================================
# VOLUME STATE AGREEMENT
# ============================================================

header(
    "VOLUME STATE AGREEMENT RATE"
)


def expected_volume_state(
    diameter_state
):

    if diameter_state == "growth":
        return "volume_growth"

    if diameter_state == "shrinkage":
        return "volume_shrinkage"

    return "volume_stable"


states["expected_volume_state"] = (
    states["dynamic_state"]
    .apply(
        expected_volume_state
    )
)


states["volume_state_agreement"] = (
    states["expected_volume_state"]
    ==
    states["volume_state"]
)


agreement_rate = (
    states["volume_state_agreement"]
    .mean()
    *
    100.0
)


print(
    "Exact diameter-volume state agreement:",
    round(
        agreement_rate,
        2
    ),
    "%"
)


# ============================================================
# NULL MODEL
# ============================================================

header(
    "NULL TRANSITION COMPARISON"
)


print("""
The null model assumes that the next state follows the
overall observed state frequencies.

This is only a reference model.

It does NOT represent a biological model.
""")


overall_probabilities = (
    states["dynamic_state"]
    .value_counts(
        normalize=True
    )
)


expected_matrix = pd.DataFrame(
    0.0,
    index=state_order,
    columns=state_order
)


for previous_state in state_order:

    row_total = transition_matrix.loc[
        previous_state
    ].sum()

    if row_total > 0:

        for current_state in state_order:

            probability = (
                overall_probabilities.get(
                    current_state,
                    0.0
                )
            )

            expected_matrix.loc[
                previous_state,
                current_state
            ] = (
                row_total
                *
                probability
            )


print(
    "Observed:"
)

print(
    transition_matrix.to_string()
)

print()

print(
    "Expected under frequency-based null:"
)

print(
    expected_matrix.round(
        3
    ).to_string()
)


# ============================================================
# CHI-SQUARE TEST
# ============================================================

header(
    "TRANSITION INDEPENDENCE TEST"
)


chi_matrix = transition_matrix.copy()


active_rows = (
    chi_matrix.sum(
        axis=1
    )
    > 0
)


active_cols = (
    chi_matrix.sum(
        axis=0
    )
    > 0
)


chi_matrix = chi_matrix.loc[
    active_rows,
    active_cols
]


if (
    chi_matrix.shape[0] >= 2
    and
    chi_matrix.shape[1] >= 2
):

    chi2, p_value, dof, expected = (
        chi2_contingency(
            chi_matrix
        )
    )

    print(
        "Chi-square statistic:",
        round(
            chi2,
            5
        )
    )

    print(
        "Degrees of freedom:",
        dof
    )

    print(
        "p-value:",
        p_value
    )

    if p_value < 0.05:

        print(
            "Result: transition structure differs from"
            " independence at p < 0.05."
        )

    else:

        print(
            "Result: insufficient evidence that transition"
            " structure differs from independence."
        )

else:

    chi2 = np.nan
    p_value = np.nan

    print(
        "Chi-square test could not be performed."
    )


# ============================================================
# TRANSITION ENTROPY
# ============================================================

header(
    "STATE TRANSITION ENTROPY"
)


def entropy(probabilities):

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    if len(probabilities) == 0:
        return 0.0

    return -np.sum(
        probabilities
        *
        np.log2(
            probabilities
        )
    )


entropy_records = []


for state in state_order:

    row = transition_probabilities.loc[
        state
    ].values

    entropy_value = entropy(
        row
    )

    entropy_records.append({

        "previous_state":
            state,

        "transition_entropy_bits":
            entropy_value
    })


entropy_table = pd.DataFrame(
    entropy_records
)


print(
    entropy_table.to_string(
        index=False
    )
)


# ============================================================
# DOMINANT NEXT STATE
# ============================================================

header(
    "DOMINANT NEXT STATE"
)


dominant_records = []


for state in state_order:

    row = transition_probabilities.loc[
        state
    ]

    if row.sum() == 0:

        dominant = "insufficient_data"
        probability = np.nan

    else:

        dominant = row.idxmax()
        probability = row.max()

    dominant_records.append({

        "previous_state":
            state,

        "dominant_next_state":
            dominant,

        "probability":
            probability
    })


dominant_table = pd.DataFrame(
    dominant_records
)


print(
    dominant_table.to_string(
        index=False
    )
)


# ============================================================
# INTERPRETATION
# ============================================================

header(
    "STEP 15 - HBGD INTERPRETATION"
)


print("""
This step evaluates whether the observable dynamic states
identified in Step 14 have a structured temporal relationship.

The states are:

    growth
    stable
    shrinkage

The analysis examines:

    state persistence
    state switching
    patient-level behavior
    lesion-level behavior
    diameter-volume agreement
    transition probabilities
    transition entropy
    comparison with a frequency-based null model

A structured transition matrix would provide evidence that
the observable state representation can be used as the basis
for a dynamical-system formulation.

This does NOT establish biological mechanisms.

It does NOT establish clinical validity.
""")


# ============================================================
# FINAL FEASIBILITY
# ============================================================

header(
    "STEP 15 - FINAL FEASIBILITY"
)


total_transitions = len(
    transitions
)


if total_transitions >= 10:

    print(
        "[YES] Sufficient observed state transitions"
        " for exploratory transition analysis"
    )

else:

    print(
        "[NO] Too few state transitions"
        " for transition analysis"
    )


if persistence_percent > 50:

    print(
        "[YES] Observable states show >50% persistence"
    )

else:

    print(
        "[NO] Observable states show weak persistence"
    )


if (
    not np.isnan(p_value)
    and
    p_value < 0.05
):

    print(
        "[YES] Transition structure differs from"
        " independence"
    )

else:

    print(
        "[NO] Transition structure is not statistically"
        " distinguishable from independence"
    )


if agreement_rate >= 70:

    print(
        "[YES] Diameter and volume states show substantial"
        " agreement"
    )

else:

    print(
        "[NO] Diameter and volume state agreement is limited"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

header(
    "SAVING STEP 15 RESULTS"
)


transition_matrix_file = os.path.join(
    DATA_DIR,
    "15_hbgd_transition_matrix.csv"
)


transition_probability_file = os.path.join(
    DATA_DIR,
    "15_hbgd_transition_probabilities.csv"
)


patient_file = os.path.join(
    DATA_DIR,
    "15_hbgd_patient_transition_summary.csv"
)


lesion_file = os.path.join(
    DATA_DIR,
    "15_hbgd_lesion_transition_summary.csv"
)


entropy_file = os.path.join(
    DATA_DIR,
    "15_hbgd_transition_entropy.csv"
)


volume_file = os.path.join(
    DATA_DIR,
    "15_hbgd_volume_state_agreement.csv"
)


transition_matrix.to_csv(
    transition_matrix_file
)


transition_probabilities.to_csv(
    transition_probability_file
)


patient_transition_summary.to_csv(
    patient_file,
    index=False
)


lesion_transition_summary.to_csv(
    lesion_file,
    index=False
)


entropy_table.to_csv(
    entropy_file,
    index=False
)


volume_crosstab.to_csv(
    volume_file
)


print(
    "Saved:",
    transition_matrix_file
)

print(
    "Saved:",
    transition_probability_file
)

print(
    "Saved:",
    patient_file
)

print(
    "Saved:",
    lesion_file
)

print(
    "Saved:",
    entropy_file
)

print(
    "Saved:",
    volume_file
)


# ============================================================
# COMPLETE
# ============================================================

header(
    "STEP 15 COMPLETE"
)


print("""
Run:

    python 15_analyze_hbgd_state_transitions.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A NEURAL NETWORK YET.
DO NOT CLAIM BIOLOGICAL STATES.
""")


print(
    "=" * 95
)

print(
    "END OF STEP 15"
)

print(
    "=" * 95
)