# 16_test_state_transition_prediction.py

import os
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


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
    "STEP 16 - LOADING HBGD STATE DATA"
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
    "State records:",
    len(states)
)

print(
    "Transition records:",
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
    "dynamic_state"
]


required_transition_columns = [
    "patient_id",
    "lesion_alias",
    "previous_state",
    "current_state"
]


missing_states = [
    column
    for column in required_state_columns
    if column not in states.columns
]


missing_transitions = [
    column
    for column in required_transition_columns
    if column not in transitions.columns
]


if missing_states:

    print(
        "ERROR: Missing state columns:"
    )

    for column in missing_states:
        print(
            "   ",
            column
        )

    sys.exit(1)


if missing_transitions:

    print(
        "ERROR: Missing transition columns:"
    )

    for column in missing_transitions:
        print(
            "   ",
            column
        )

    sys.exit(1)


print(
    "All required variables are present."
)


# ============================================================
# OPTIONAL FEATURE COLUMNS
# ============================================================

header(
    "CHECKING AVAILABLE FEATURES"
)


possible_features = [
    "diameter_mm",
    "volume_cm3",
    "equivalent_diameter_mm",
    "days_from_baseline",
    "diameter_normalized",
    "volume_normalized",
    "diameter_percent_from_baseline",
    "volume_percent_from_baseline"
]


available_features = [
    column
    for column in possible_features
    if column in states.columns
]


print(
    "Available feature variables:"
)


for feature in available_features:

    print(
        "   ",
        feature
    )


if len(available_features) == 0:

    print(
        "WARNING: No optional numerical features found."
    )


# ============================================================
# NORMALIZE DATA
# ============================================================

header(
    "NORMALIZING DATA"
)


states["patient_id"] = pd.to_numeric(
    states["patient_id"],
    errors="coerce"
)


transitions["patient_id"] = pd.to_numeric(
    transitions["patient_id"],
    errors="coerce"
)


states = states.dropna(
    subset=[
        "patient_id",
        "lesion_alias",
        "dynamic_state"
    ]
).copy()


transitions = transitions.dropna(
    subset=[
        "patient_id",
        "lesion_alias",
        "previous_state",
        "current_state"
    ]
).copy()


for feature in available_features:

    states[feature] = pd.to_numeric(
        states[feature],
        errors="coerce"
    )


print(
    "Valid state records:",
    len(states)
)

print(
    "Valid transition records:",
    len(transitions)
)


# ============================================================
# CONSTRUCT TRANSITION FEATURES
# ============================================================

header(
    "BUILDING TRANSITION FEATURE TABLE"
)


feature_records = []


for _, row in transitions.iterrows():

    patient_id = row[
        "patient_id"
    ]

    alias = row[
        "lesion_alias"
    ]

    previous_state = row[
        "previous_state"
    ]

    current_state = row[
        "current_state"
    ]


    patient_features = states[
        states["patient_id"]
        ==
        patient_id
    ]


    patient_features = patient_features[
        patient_features["lesion_alias"]
        ==
        alias
    ]


    if len(patient_features) == 0:

        continue


    previous_rows = patient_features[
        patient_features["dynamic_state"]
        ==
        previous_state
    ]


    # --------------------------------------------------------
    # Prefer the first occurrence of the previous state.
    # This is deliberately simple and transparent.
    # --------------------------------------------------------

    if len(previous_rows) == 0:

        continue


    feature_row = previous_rows.iloc[0]


    record = {

        "patient_id":
            patient_id,

        "lesion_alias":
            alias,

        "previous_state":
            previous_state,

        "current_state":
            current_state
    }


    for feature in available_features:

        record[feature] = feature_row[
            feature
        ]


    feature_records.append(
        record
    )


feature_table = pd.DataFrame(
    feature_records
)


print(
    "Feature transition records:",
    len(feature_table)
)


if len(feature_table) == 0:

    print()
    print(
        "ERROR: No transition feature records were created."
    )
    print()
    print(
        "This indicates that the state representation"
        " cannot currently be linked to the transition table."
    )

    sys.exit(1)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

feature_table = (
    feature_table
    .drop_duplicates(
        subset=[
            "patient_id",
            "lesion_alias",
            "previous_state",
            "current_state"
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "Unique transition feature records:",
    len(feature_table)
)


# ============================================================
# STATE DISTRIBUTION
# ============================================================

header(
    "TARGET STATE DISTRIBUTION"
)


print(
    feature_table[
        "current_state"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# PATIENT HOLDOUT
# ============================================================

header(
    "CREATING PATIENT-LEVEL HOLDOUT"
)


patients = sorted(
    feature_table[
        "patient_id"
    ]
    .unique()
)


print(
    "Patients available:",
    len(patients)
)


if len(patients) < 4:

    print(
        "ERROR: Too few patients for patient-level holdout."
    )

    sys.exit(1)


# ------------------------------------------------------------
# Deterministic holdout
# Last 25% of sorted patients.
# ------------------------------------------------------------

test_count = max(
    1,
    int(
        np.ceil(
            len(patients) * 0.25
        )
    )
)


test_patients = patients[
    -test_count:
]


train_patients = patients[
    :-test_count
]


train = feature_table[
    feature_table["patient_id"].isin(
        train_patients
    )
].copy()


test = feature_table[
    feature_table["patient_id"].isin(
        test_patients
    )
].copy()


print(
    "Training patients:",
    len(train_patients)
)

print(
    "Testing patients:",
    len(test_patients)
)

print(
    "Training transitions:",
    len(train)
)

print(
    "Testing transitions:",
    len(test)
)

print(
    "Training patient IDs:",
    train_patients
)

print(
    "Testing patient IDs:",
    test_patients
)


if len(test) == 0:

    print(
        "ERROR: Test set is empty."
    )

    sys.exit(1)


# ============================================================
# LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()


all_states = [
    "growth",
    "stable",
    "shrinkage"
]


label_encoder.fit(
    all_states
)


y_train = label_encoder.transform(
    train["current_state"]
)


y_test = label_encoder.transform(
    test["current_state"]
)


# ============================================================
# MODEL A - PREVIOUS STATE PERSISTENCE
# ============================================================

header(
    "MODEL A - PREVIOUS STATE PERSISTENCE"
)


pred_previous = (
    test["previous_state"]
    .values
)


# ============================================================
# MODEL B - MOST COMMON STATE
# ============================================================

header(
    "MODEL B - MOST COMMON TRAINING STATE"
)


most_common_state = (
    train[
        "current_state"
    ]
    .value_counts()
    .idxmax()
)


print(
    "Most common training state:",
    most_common_state
)


pred_common = np.array(
    [
        most_common_state
    ]
    *
    len(test)
)


# ============================================================
# MODEL C - TRANSITION MATRIX
# ============================================================

header(
    "MODEL C - LEARNED TRANSITION MATRIX"
)


transition_counts = pd.crosstab(
    train["previous_state"],
    train["current_state"]
)


transition_counts = transition_counts.reindex(
    index=all_states,
    columns=all_states,
    fill_value=0
)


transition_probabilities = (
    transition_counts
    .div(
        transition_counts.sum(
            axis=1
        ).replace(
            0,
            np.nan
        ),
        axis=0
    )
    .fillna(0)
)


print(
    "Training transition matrix:"
)

print(
    transition_counts.to_string()
)

print()

print(
    "Training transition probabilities:"
)

print(
    transition_probabilities.round(
        4
    ).to_string()
)


pred_transition = []


for state in test[
    "previous_state"
]:

    probabilities = (
        transition_probabilities
        .loc[state]
    )


    if probabilities.sum() == 0:

        prediction = most_common_state

    else:

        prediction = probabilities.idxmax()


    pred_transition.append(
        prediction
    )


pred_transition = np.array(
    pred_transition
)


# ============================================================
# MODEL D/E - FEATURE MODELS
# ============================================================

feature_model_results = []


feature_sets = {}


if "diameter_mm" in available_features:

    feature_sets[
        "diameter"
    ] = [
        "diameter_mm"
    ]


if (
    "diameter_mm" in available_features
    and
    "days_from_baseline" in available_features
):

    feature_sets[
        "diameter_time"
    ] = [
        "diameter_mm",
        "days_from_baseline"
    ]


if (
    "diameter_mm" in available_features
    and
    "volume_cm3" in available_features
    and
    "days_from_baseline" in available_features
):

    feature_sets[
        "diameter_volume_time"
    ] = [
        "diameter_mm",
        "volume_cm3",
        "days_from_baseline"
    ]


if len(feature_sets) > 0:

    header(
        "FEATURE-BASED STATE MODELS"
    )


for model_name, feature_columns in feature_sets.items():

    print()
    print(
        "MODEL:",
        model_name
    )

    print(
        "Features:",
        feature_columns
    )


    model_train = train[
        feature_columns
        +
        ["current_state"]
    ].dropna()


    model_test = test[
        feature_columns
    ].copy()


    valid_test = (
        ~model_test.isna().any(
            axis=1
        )
    )


    if len(model_train) < 5:

        print(
            "Skipped: insufficient training records."
        )

        continue


    if (
        model_train["current_state"]
        .nunique()
        < 2
    ):

        print(
            "Skipped: training data contains"
            " fewer than two target classes."
        )

        continue


    model = LogisticRegression(
        max_iter=2000,
        random_state=42
    )


    X_train = model_train[
        feature_columns
    ]


    y_model_train = model_train[
        "current_state"
    ]


    model.fit(
        X_train,
        y_model_train
    )


    predictions = np.full(
        len(test),
        most_common_state,
        dtype=object
    )


    if valid_test.sum() > 0:

        predictions[
            valid_test.values
        ] = model.predict(
            model_test.loc[
                valid_test,
                feature_columns
            ]
        )


    feature_model_results.append({

        "model":
            model_name,

        "predictions":
            predictions
    })


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    model_name,
    y_true,
    predictions
):

    accuracy = accuracy_score(
        y_true,
        predictions
    )


    balanced = balanced_accuracy_score(
        y_true,
        predictions
    )


    f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )


    return {

        "model":
            model_name,

        "observations":
            len(y_true),

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced,

        "macro_f1":
            f1
    }


# ============================================================
# EVALUATE BASE MODELS
# ============================================================

header(
    "MODEL PERFORMANCE"
)


results = []


results.append(
    calculate_metrics(
        "previous_state",
        test["current_state"].values,
        pred_previous
    )
)


results.append(
    calculate_metrics(
        "most_common_state",
        test["current_state"].values,
        pred_common
    )
)


results.append(
    calculate_metrics(
        "transition_matrix",
        test["current_state"].values,
        pred_transition
    )
)


for item in feature_model_results:

    results.append(
        calculate_metrics(
            item["model"],
            test["current_state"].values,
            item["predictions"]
        )
    )


performance = pd.DataFrame(
    results
)


print(
    performance.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

header(
    "BEST MODEL"
)


best_index = performance[
    "macro_f1"
].idxmax()


best_model = performance.loc[
    best_index,
    "model"
]


best_f1 = performance.loc[
    best_index,
    "macro_f1"
]


print(
    "Best model by macro-F1:",
    best_model
)


print(
    "Best macro-F1:",
    round(
        best_f1,
        4
    )
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

header(
    "CONFUSION MATRICES"
)


prediction_dictionary = {

    "previous_state":
        pred_previous,

    "most_common_state":
        pred_common,

    "transition_matrix":
        pred_transition
}


for item in feature_model_results:

    prediction_dictionary[
        item["model"]
    ] = item["predictions"]


for model_name, predictions in (
    prediction_dictionary.items()
):

    print()
    print(
        "MODEL:",
        model_name
    )

    cm = confusion_matrix(
        test["current_state"],
        predictions,
        labels=all_states
    )


    cm_df = pd.DataFrame(
        cm,
        index=[
            "actual_" + s
            for s in all_states
        ],
        columns=[
            "pred_" + s
            for s in all_states
        ]
    )


    print(
        cm_df.to_string()
    )


# ============================================================
# PER-PATIENT PERFORMANCE
# ============================================================

header(
    "PATIENT-LEVEL PERFORMANCE"
)


patient_results = []


for patient_id, group in test.groupby(
    "patient_id"
):

    indices = group.index


    true_values = group[
        "current_state"
    ].values


    previous_values = group[
        "previous_state"
    ].values


    common_values = np.array(
        [
            most_common_state
        ]
        *
        len(group)
    )


    transition_values = []


    for state in previous_values:

        row = (
            transition_probabilities
            .loc[state]
        )


        if row.sum() == 0:

            prediction = most_common_state

        else:

            prediction = row.idxmax()


        transition_values.append(
            prediction
        )


    patient_results.append({

        "patient_id":
            patient_id,

        "observations":
            len(group),

        "previous_state_accuracy":
            accuracy_score(
                true_values,
                previous_values
            ),

        "most_common_accuracy":
            accuracy_score(
                true_values,
                common_values
            ),

        "transition_accuracy":
            accuracy_score(
                true_values,
                transition_values
            )
    })


patient_performance = pd.DataFrame(
    patient_results
)


print(
    patient_performance.round(
        4
    ).to_string(
        index=False
    )
)


# ============================================================
# STATE-SPECIFIC RECALL
# ============================================================

header(
    "STATE-SPECIFIC PERFORMANCE"
)


transition_predictions = pred_transition


state_cm = confusion_matrix(
    test["current_state"],
    transition_predictions,
    labels=all_states
)


for i, state in enumerate(
    all_states
):

    denominator = (
        state_cm[i].sum()
    )


    if denominator > 0:

        recall = (
            state_cm[i, i]
            /
            denominator
        )

    else:

        recall = np.nan


    print(
        state,
        "recall:",
        (
            round(
                recall * 100.0,
                2
            )
            if not np.isnan(recall)
            else "N/A"
        ),
        "%"
    )


# ============================================================
# INTERPRETATION
# ============================================================

header(
    "STEP 16 - INTERPRETATION"
)


print("""
This experiment tests whether the observable dynamic states
can predict the next observed state in completely unseen
patients.

The baselines are:

    A. Previous-state persistence
    B. Most-common-state prediction

The transition model learns:

    previous state
          ↓
    transition probabilities
          ↓
    predicted next state

Optional feature models test whether adding lesion geometry
improves state prediction.

The patient-level holdout is important because observations
from the same patient are not allowed to appear in both
training and testing data.

This remains an exploratory feasibility experiment.

It does NOT establish biological mechanisms.
It does NOT establish clinical validity.
""")


# ============================================================
# FINAL FEASIBILITY
# ============================================================

header(
    "STEP 16 - FINAL FEASIBILITY"
)


transition_row = performance[
    performance["model"]
    ==
    "transition_matrix"
]


previous_row = performance[
    performance["model"]
    ==
    "previous_state"
]


if len(transition_row) > 0:

    transition_f1 = transition_row[
        "macro_f1"
    ].iloc[0]

else:

    transition_f1 = np.nan


if len(previous_row) > 0:

    previous_f1 = previous_row[
        "macro_f1"
    ].iloc[0]

else:

    previous_f1 = np.nan


print(
    "[YES] Independent patient-level holdout available"
)


if (
    not np.isnan(transition_f1)
    and
    not np.isnan(previous_f1)
    and
    transition_f1 > previous_f1
):

    print(
        "[YES] Transition model beats previous-state"
        " persistence by macro-F1"
    )

else:

    print(
        "[NO] Transition model does not beat"
        " previous-state persistence"
    )


if best_model != "most_common_state":

    print(
        "[YES] At least one model provides information"
        " beyond the most-common-state baseline"
    )

else:

    print(
        "[NO] No tested model beats the"
        " most-common-state baseline"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

header(
    "SAVING STEP 16 RESULTS"
)


performance_file = os.path.join(
    DATA_DIR,
    "16_hbgd_state_prediction_performance.csv"
)


patient_file = os.path.join(
    DATA_DIR,
    "16_hbgd_state_patient_performance.csv"
)


prediction_file = os.path.join(
    DATA_DIR,
    "16_hbgd_state_predictions.csv"
)


transition_file = os.path.join(
    DATA_DIR,
    "16_hbgd_training_transition_probabilities.csv"
)


# ------------------------------------------------------------
# Add predictions to test table
# ------------------------------------------------------------

prediction_table = test.copy()


prediction_table[
    "pred_previous_state"
] = pred_previous


prediction_table[
    "pred_most_common_state"
] = pred_common


prediction_table[
    "pred_transition_matrix"
] = pred_transition


for item in feature_model_results:

    column_name = (
        "pred_"
        +
        item["model"]
    )

    prediction_table[
        column_name
    ] = item["predictions"]


performance.to_csv(
    performance_file,
    index=False
)


patient_performance.to_csv(
    patient_file,
    index=False
)


prediction_table.to_csv(
    prediction_file,
    index=False
)


transition_probabilities.to_csv(
    transition_file
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

print(
    "Saved:",
    transition_file
)


# ============================================================
# COMPLETE
# ============================================================

header(
    "STEP 16 COMPLETE"
)


print("""
Run:

    python 16_test_state_transition_prediction.py

Then send me the COMPLETE terminal output.

DO NOT TRAIN A NEURAL NETWORK YET.
DO NOT CLAIM BIOLOGICAL STATES.
""")


print(
    "=" * 95
)

print(
    "END OF STEP 16"
)

print(
    "=" * 95
)