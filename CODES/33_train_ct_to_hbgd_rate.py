import os
import random
import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass
# ============================================================
# CONFIG
# ============================================================

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

COVERAGE_FILE = os.path.join(
    ROOT,
    "32_ct_rate_repaired_coverage.csv"
)

OUTPUT_DIR = os.path.join(
    ROOT,
    "33_ct_rate_cnn_results"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

SEED = 42

TRAIN_PATIENTS = [
    3, 7, 8, 12, 13, 16, 22
]

TEST_PATIENTS = [
    23, 25, 30
]

# Because only patient 30 currently has a complete
# CT-to-rate trajectory, patient 30 is the actual
# held-out demonstration case.

BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 1e-4

TARGET_SIZE = (
    64,
    64,
    64
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# PRINT HELPERS
# ============================================================

def header(title):

    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ============================================================
# START
# ============================================================

header(
    "STEP 33 - CT TO HBGD RATE CNN PROTOTYPE"
)

print(
    "This experiment learns:"
)

print()
print(
    "CT crop"
)

print(
    "   to"
)

print(
    "CNN encoder"
)

print(
    "   to"
)

print(
    "individualized growth rate r"
)

print()
print(
    "The CNN does NOT predict a biological state."
)

print(
    "It predicts the mathematical HBGD parameter r."
)

print()
print(
    "This is a prototype experiment."
)

print(
    "It is NOT clinical validation."
)


# ============================================================
# PYTORCH
# ============================================================

header(
    "CHECKING PYTORCH"
)

print(
    "PyTorch:",
    torch.__version__
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# ============================================================
# LOAD COVERAGE
# ============================================================

header(
    "LOADING CORRECTED CT-TO-RATE COVERAGE"
)

coverage = pd.read_csv(
    COVERAGE_FILE
)

print(
    "Coverage records:",
    len(coverage)
)

required_columns = [
    "patient_id",
    "alias",
    "baseline_date",
    "second_date",
    "individualized_rate",
    "baseline_ct_file",
    "second_ct_file",
    "usable_ct_rate"
]

missing = [
    c
    for c in required_columns
    if c not in coverage.columns
]

if missing:

    raise SystemExit(
        "ERROR: Missing columns: "
        + str(missing)
    )


# ============================================================
# FILTER USABLE TRAJECTORIES
# ============================================================

header(
    "FILTERING USABLE CT-TO-RATE TRAJECTORIES"
)

coverage = coverage[
    coverage["usable_ct_rate"] == True
].copy()

coverage["patient_id"] = pd.to_numeric(
    coverage["patient_id"],
    errors="coerce"
)

coverage["individualized_rate"] = pd.to_numeric(
    coverage["individualized_rate"],
    errors="coerce"
)

coverage = coverage.dropna(
    subset=[
        "patient_id",
        "individualized_rate"
    ]
).copy()

coverage["patient_id"] = (
    coverage["patient_id"].astype(int)
)

print(
    "Usable trajectories:",
    len(coverage)
)

print(
    "Patients:",
    sorted(
        coverage["patient_id"].unique().tolist()
    )
)


# ============================================================
# ACTUAL SPLIT
# ============================================================

header(
    "PATIENT-INDEPENDENT SPLIT"
)

train_df = coverage[
    coverage["patient_id"].isin(
        TRAIN_PATIENTS
    )
].copy()

test_df = coverage[
    coverage["patient_id"].isin(
        TEST_PATIENTS
    )
].copy()

print(
    "Requested training patients:",
    TRAIN_PATIENTS
)

print(
    "Requested testing patients:",
    TEST_PATIENTS
)

print()

print(
    "Actual training trajectories:",
    len(train_df)
)

print(
    "Actual testing trajectories:",
    len(test_df)
)

print(
    "Training patients actually present:",
    sorted(
        train_df["patient_id"].unique().tolist()
    )
)

print(
    "Testing patients actually present:",
    sorted(
        test_df["patient_id"].unique().tolist()
    )
)

if len(train_df) < 3:

    raise SystemExit(
        "ERROR: Too few training trajectories."
    )

if len(test_df) < 1:

    raise SystemExit(
        "ERROR: No independent test trajectory."
    )


# ============================================================
# LOAD NIFTI
# ============================================================

def load_nifti(path):

    img = nib.load(
        str(path)
    )

    arr = img.get_fdata(
        dtype=np.float32
    )

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return arr


# ============================================================
# RESAMPLE / RESIZE
# ============================================================

def resize_volume(
    volume,
    target_size=TARGET_SIZE
):

    tensor = torch.from_numpy(
        volume
    ).float()

    # NIfTI generally gives:
    # X, Y, Z
    #
    # Convert to:
    # 1, 1, X, Y, Z

    tensor = tensor.unsqueeze(0).unsqueeze(0)

    tensor = torch.nn.functional.interpolate(
        tensor,
        size=target_size,
        mode="trilinear",
        align_corners=False
    )

    return tensor.squeeze(0)


# ============================================================
# NORMALIZE CT
# ============================================================

def normalize_ct(volume):

    volume = np.clip(
        volume,
        -1000,
        1000
    )

    volume = (
        volume + 1000.0
    ) / 2000.0

    volume = np.clip(
        volume,
        0.0,
        1.0
    )

    return volume.astype(
        np.float32
    )


# ============================================================
# DATASET
# ============================================================

class CTGrowthDataset(
    Dataset
):

    def __init__(
        self,
        dataframe
    ):

        self.df = (
            dataframe
            .reset_index(drop=True)
            .copy()
        )

    def __len__(self):

        return len(
            self.df
        )

    def __getitem__(
        self,
        index
    ):

        row = self.df.iloc[
            index
        ]

        path = row[
            "baseline_ct_file"
        ]

        volume = load_nifti(
            path
        )

        volume = normalize_ct(
            volume
        )

        tensor = resize_volume(
            volume
        )

        target = torch.tensor(
            [
                float(
                    row[
                        "individualized_rate"
                    ]
                )
            ],
            dtype=torch.float32
        )

        return (
            tensor,
            target
        )


# ============================================================
# DATASET CREATION
# ============================================================

header(
    "CREATING CT DATASETS"
)

train_dataset = CTGrowthDataset(
    train_df
)

test_dataset = CTGrowthDataset(
    test_df
)

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Testing samples:",
    len(test_dataset)
)


# ============================================================
# CNN MODEL
# ============================================================

header(
    "DEFINING 3D CNN"
)


class HBGD3DCNN(
    nn.Module
):

    def __init__(
        self
    ):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv3d(
                1,
                8,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(
                2
            ),

            nn.Conv3d(
                8,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool3d(
                2
            ),

            nn.Conv3d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool3d(
                (1, 1, 1)
            )
        )

        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                1
            )
        )

    def forward(
        self,
        x
    ):

        x = self.features(
            x
        )

        return self.regressor(
            x
        )


model = HBGD3DCNN().to(
    device
)

print(
    model
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

header(
    "TRAINING CNN"
)

history = []

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    epoch_losses = []

    for x, y in train_loader:

        x = x.to(
            device
        )

        y = y.to(
            device
        )

        optimizer.zero_grad()

        prediction = model(
            x
        )

        loss = criterion(
            prediction,
            y
        )

        loss.backward()

        optimizer.step()

        epoch_losses.append(
            loss.item()
        )

    mean_loss = float(
        np.mean(
            epoch_losses
        )
    )

    history.append({

        "epoch":
            epoch,

        "train_mse":
            mean_loss
    })

    if (
        epoch == 1
        or epoch % 5 == 0
        or epoch == EPOCHS
    ):

        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"| MSE = {mean_loss:.8f}"
        )


# ============================================================
# EVALUATION
# ============================================================

header(
    "HELD-OUT PATIENT EVALUATION"
)

model.eval()

predictions = []

with torch.no_grad():

    for x, y in test_loader:

        x = x.to(
            device
        )

        prediction = model(
            x
        )

        predictions.append(
            float(
                prediction
                .cpu()
                .numpy()
                .reshape(-1)[0]
            )
        )


actual = (
    test_df[
        "individualized_rate"
    ]
    .astype(float)
    .values
)

predicted = np.array(
    predictions
)


# ============================================================
# RATE METRICS
# ============================================================

rate_mae = float(
    np.mean(
        np.abs(
            predicted - actual
        )
    )
)

rate_rmse = float(
    np.sqrt(
        np.mean(
            (
                predicted - actual
            ) ** 2
        )
    )
)

print(
    "Testing trajectories:",
    len(actual)
)

print(
    "Rate MAE:",
    rate_mae
)

print(
    "Rate RMSE:",
    rate_rmse
)


# ============================================================
# RATE PREDICTIONS
# ============================================================

prediction_df = test_df[
    [
        "patient_id",
        "alias",
        "baseline_date",
        "second_date",
        "individualized_rate",
        "baseline_ct_file",
        "second_ct_file"
    ]
].copy()

prediction_df[
    "cnn_predicted_rate"
] = predicted

prediction_df[
    "absolute_rate_error"
] = np.abs(
    predicted - actual
)


# ============================================================
# HBGD FORWARD PREDICTION
# ============================================================

header(
    "CONVERTING CNN RATE TO HBGD FORWARD PREDICTION"
)

future_predictions = []

for i, row in test_df.reset_index(
    drop=True
).iterrows():

    d0 = None

    # We need baseline diameter from the
    # HBGD feature table.
    #
    # Recover it using patient + alias + date.

    patient_id = int(
        row["patient_id"]
    )

    alias = row["alias"]

    baseline_date = pd.to_datetime(
        row["baseline_date"]
    )

    feature_match = pd.read_csv(
        os.path.join(
            ROOT,
            "9_hbgd_features.csv"
        )
    )

    feature_match[
        "patient_id"
    ] = pd.to_numeric(
        feature_match[
            "patient_id"
        ],
        errors="coerce"
    )

    feature_match[
        "study_date"
    ] = pd.to_datetime(
        feature_match[
            "study_date"
        ],
        errors="coerce"
    )

    feature_match[
        "alias"
    ] = feature_match[
        "lesion_label_alias"
    ].astype(str).str.upper().str.strip()

    match = feature_match[
        (feature_match["patient_id"] == patient_id)
        &
        (feature_match["alias"] == alias)
        &
        (
            feature_match["study_date"]
            == baseline_date
        )
    ]

    if len(match) == 0:

        continue

    d0 = float(
        match.iloc[0][
            "diameter_mm"
        ]
    )

    second_date = pd.to_datetime(
        row["second_date"]
    )

    days = (
        second_date
        -
        baseline_date
    ).days

    r_pred = float(
        predicted[i]
    )

    r_true = float(
        row[
            "individualized_rate"
        ]
    )

    predicted_diameter = (
        d0
        *
        np.exp(
            r_pred * days
        )
    )

    actual_diameter = float(
        match.iloc[0][
            "diameter_mm"
        ]
    )

    # Find actual second observation
    second_match = feature_match[
        (feature_match["patient_id"] == patient_id)
        &
        (feature_match["alias"] == alias)
        &
        (
            feature_match["study_date"]
            == second_date
        )
    ]

    if len(second_match) > 0:

        actual_diameter = float(
            second_match.iloc[0][
                "diameter_mm"
            ]
        )

    future_predictions.append({

        "patient_id":
            patient_id,

        "alias":
            alias,

        "baseline_date":
            baseline_date,

        "second_date":
            second_date,

        "days_ahead":
            days,

        "baseline_diameter_mm":
            d0,

        "actual_diameter_mm":
            actual_diameter,

        "true_rate":
            r_true,

        "cnn_predicted_rate":
            r_pred,

        "predicted_diameter_mm":
            predicted_diameter,

        "diameter_absolute_error_mm":
            abs(
                predicted_diameter
                -
                actual_diameter
            )
    })


future_df = pd.DataFrame(
    future_predictions
)


# ============================================================
# FORWARD METRICS
# ============================================================

if len(future_df) > 0:

    cnn_diameter_mae = float(
        future_df[
            "diameter_absolute_error_mm"
        ].mean()
    )

    cnn_diameter_rmse = float(
        np.sqrt(
            np.mean(
                future_df[
                    "diameter_absolute_error_mm"
                ].values
                ** 2
            )
        )
    )

    baseline_error = np.abs(
        future_df[
            "baseline_diameter_mm"
        ]
        -
        future_df[
            "actual_diameter_mm"
        ]
    )

    baseline_mae = float(
        baseline_error.mean()
    )

    print(
        "Forward prediction observations:",
        len(future_df)
    )

    print(
        "CNN-HBGD diameter MAE:",
        round(
            cnn_diameter_mae,
            4
        ),
        "mm"
    )

    print(
        "CNN-HBGD diameter RMSE:",
        round(
            cnn_diameter_rmse,
            4
        ),
        "mm"
    )

    print(
        "No-change baseline MAE:",
        round(
            baseline_mae,
            4
        ),
        "mm"
    )

else:

    cnn_diameter_mae = np.nan
    cnn_diameter_rmse = np.nan
    baseline_mae = np.nan

    print(
        "WARNING: No forward prediction records."
    )


# ============================================================
# SAVE
# ============================================================

header(
    "SAVING STEP 33 RESULTS"
)

history_df = pd.DataFrame(
    history
)

history_file = os.path.join(
    OUTPUT_DIR,
    "33_training_history.csv"
)

rate_file = os.path.join(
    OUTPUT_DIR,
    "33_cnn_rate_predictions.csv"
)

forward_file = os.path.join(
    OUTPUT_DIR,
    "33_cnn_hbgd_forward_predictions.csv"
)

performance_file = os.path.join(
    OUTPUT_DIR,
    "33_cnn_hbgd_performance.csv"
)

history_df.to_csv(
    history_file,
    index=False
)

prediction_df.to_csv(
    rate_file,
    index=False
)

future_df.to_csv(
    forward_file,
    index=False
)

performance = pd.DataFrame([{

    "test_trajectories":
        len(test_df),

    "rate_mae":
        rate_mae,

    "rate_rmse":
        rate_rmse,

    "forward_observations":
        len(future_df),

    "cnn_hbgd_diameter_mae":
        cnn_diameter_mae,

    "cnn_hbgd_diameter_rmse":
        cnn_diameter_rmse,

    "no_change_mae":
        baseline_mae
}])

performance.to_csv(
    performance_file,
    index=False
)

print(
    "Saved:",
    history_file
)

print(
    "Saved:",
    rate_file
)

print(
    "Saved:",
    forward_file
)

print(
    "Saved:",
    performance_file
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

header(
    "STEP 33 - INTERPRETATION"
)

print(
    "The CNN attempts to learn:"
)

print()
print(
    "CT → individualized mathematical growth rate r"
)

print()
print(
    "The predicted r is then inserted into:"
)

print()
print(
    "dD/dt = rD"
)

print()
print(
    "and:"
)

print()
print(
    "D(t) = D0 exp(r t)"
)

print()
print(
    "The testing patient was kept completely separate "
    "from CNN training."
)

print()
print(
    "However, the independent test set contains only "
    f"{len(test_df)} complete trajectory."
)

print()
print(
    "Therefore this result MUST be treated as a prototype "
    "demonstration rather than model validation."
)


# ============================================================
# FINAL DECISION
# ============================================================

header(
    "STEP 33 - FINAL DECISION"
)

if len(train_df) >= 3:

    print(
        "[YES] CNN training data available."
    )

else:

    print(
        "[NO] Insufficient training data."
    )


if len(test_df) >= 3:

    print(
        "[YES] Sufficient independent test trajectories."
    )

else:

    print(
        "[NO] Independent test set remains extremely small."
    )


if not np.isnan(
    cnn_diameter_mae
):

    print(
        "[YES] End-to-end CT to r to HBGD prediction executed."
    )

else:

    print(
        "[NO] End-to-end HBGD prediction unavailable."
    )


print()
print(
    "NO CLINICAL VALIDATION IS CLAIMED."
)

print(
    "NO BIOLOGICAL STATE IS CLAIMED."
)

print(
    "This is an exploratory CNN prototype."
)

print()
print("=" * 100)
print(
    "STEP 33 COMPLETE"
)
print("=" * 100)