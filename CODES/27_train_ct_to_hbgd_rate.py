# 27_train_ct_to_hbgd_rate.py

import os
import sys
import math
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent / "DATA"

QUALITY_FILE = ROOT / "25_ct_crop_quality.csv"
FEATURE_FILE = ROOT / "9_hbgd_features.csv"
CROP_DIR = ROOT / "25_lesion_ct_crops"

PREDICTION_FILE = ROOT / "27_ct_hbgd_predictions.csv"
PERFORMANCE_FILE = ROOT / "27_ct_hbgd_performance.csv"
PATIENT_PERFORMANCE_FILE = ROOT / "27_ct_hbgd_patient_performance.csv"
MODEL_FILE = ROOT / "27_ct_to_hbgd_rate_model.pt"

SEED = 42

TARGET_SHAPE = (64, 64, 64)
EPOCHS = 25
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
RATE_SCALE = 1000.0

TRAIN_PATIENTS = [3, 7, 8, 12, 13, 16, 22]
TEST_PATIENTS = [23, 25, 30]

random.seed(SEED)
np.random.seed(SEED)


def header(title):
    print()
    print("=" * 95)
    print(title)
    print("=" * 95)


# ============================================================
# PYTORCH
# ============================================================

header("STEP 27 - CT TO HBGD RATE CNN")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception as e:
    print("ERROR: PyTorch import failed:")
    print(e)
    sys.exit(1)

print("PyTorch:", torch.__version__)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# LOAD
# ============================================================

header("LOADING DATA")

quality = pd.read_csv(QUALITY_FILE)
features = pd.read_csv(FEATURE_FILE)

print("Crop quality records:", len(quality))
print("Feature records:", len(features))

quality.columns = [str(c).strip() for c in quality.columns]
features.columns = [str(c).strip() for c in features.columns]


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_quality = [
    "patient_id",
    "study_date",
    "lesion_label_alias",
    "crop_file",
    "status",
]

required_features = [
    "patient_id",
    "study_date",
    "lesion_label_alias",
    "diameter_mm",
]

for c in required_quality:
    if c not in quality.columns:
        print("ERROR: Missing quality column:", c)
        print(list(quality.columns))
        sys.exit(1)

for c in required_features:
    if c not in features.columns:
        print("ERROR: Missing feature column:", c)
        print(list(features.columns))
        sys.exit(1)


# ============================================================
# NORMALIZATION
# ============================================================

header("NORMALIZING DATA")

for df in [quality, features]:

    df["patient_id"] = pd.to_numeric(
        df["patient_id"],
        errors="coerce"
    )

    df["study_date"] = pd.to_datetime(
        df["study_date"].astype(str),
        errors="coerce"
    )

    df["lesion_label_alias"] = (
        df["lesion_label_alias"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

quality = quality.dropna(
    subset=[
        "patient_id",
        "study_date",
        "lesion_label_alias",
    ]
).copy()

features = features.dropna(
    subset=[
        "patient_id",
        "study_date",
        "lesion_label_alias",
        "diameter_mm",
    ]
).copy()

quality["patient_id"] = quality["patient_id"].astype(int)
features["patient_id"] = features["patient_id"].astype(int)

features["diameter_mm"] = pd.to_numeric(
    features["diameter_mm"],
    errors="coerce"
)

features = features[
    features["diameter_mm"] > 0
].copy()


# ============================================================
# REMOVE AMBIGUOUS PATIENT 37
# ============================================================

header("REMOVING AMBIGUOUS RECIST RECORD")

ambiguous_mask = (
    (quality["patient_id"] == 37)
    & (quality["study_date"] == pd.Timestamp("2023-02-21"))
    & (quality["lesion_label_alias"] == "A")
)

print(
    "Ambiguous quality records:",
    int(ambiguous_mask.sum())
)

quality = quality[
    ~ambiguous_mask
].copy()

ambiguous_feature_mask = (
    (features["patient_id"] == 37)
    & (features["study_date"] == pd.Timestamp("2023-02-21"))
    & (features["lesion_label_alias"] == "A")
)

features = features[
    ~ambiguous_feature_mask
].copy()


# ============================================================
# VERIFIED CROPS
# ============================================================

header("FINDING VERIFIED CT CROPS")

status = (
    quality["status"]
    .astype(str)
    .str.strip()
    .str.lower()
)

quality_valid = quality[
    status.isin([
        "valid",
        "success",
        "successful",
        "true",
        "1",
        "yes",
    ])
].copy()

print("Verified crop records:", len(quality_valid))


def resolve_crop(value):

    if pd.isna(value):
        return None

    p = Path(str(value))

    candidates = []

    if p.is_absolute():
        candidates.append(p)

    candidates.extend([
        ROOT / p,
        CROP_DIR / p,
        CROP_DIR / p.name,
    ])

    for candidate in candidates:

        try:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        except Exception:
            pass

    return None


quality_valid["resolved_crop_path"] = (
    quality_valid["crop_file"]
    .apply(resolve_crop)
)

quality_valid = quality_valid[
    quality_valid["resolved_crop_path"].notna()
].copy()

print(
    "Physically available crop files:",
    len(quality_valid)
)


# ============================================================
# BUILD LONGITUDINAL TRAJECTORIES
# ============================================================

header("BUILDING LONGITUDINAL RATE TARGETS")

features = features.sort_values([
    "patient_id",
    "lesion_label_alias",
    "study_date",
])

targets = []

for (patient, alias), group in features.groupby(
    ["patient_id", "lesion_label_alias"]
):

    group = (
        group
        .sort_values("study_date")
        .drop_duplicates(
            subset=["study_date"],
            keep="first"
        )
    )

    if len(group) < 2:
        continue

    first = group.iloc[0]
    second = group.iloc[1]

    d0 = float(first["diameter_mm"])
    d1 = float(second["diameter_mm"])

    dt = (
        second["study_date"]
        - first["study_date"]
    ).days

    if dt <= 0:
        continue

    if d0 <= 0 or d1 <= 0:
        continue

    r = math.log(d1 / d0) / float(dt)

    targets.append({
        "patient_id": int(patient),
        "lesion_label_alias": alias,
        "baseline_date": first["study_date"],
        "second_date": second["study_date"],
        "baseline_diameter_mm": d0,
        "second_diameter_mm": d1,
        "delta_days": dt,
        "target_r": r,
    })

targets = pd.DataFrame(targets)

print("Eligible trajectories:", len(targets))

if len(targets) == 0:
    print("ERROR: No trajectories.")
    sys.exit(1)


# ============================================================
# IMPORTANT:
# FIND FIRST AVAILABLE CT CROP FOR EACH TARGET
# ============================================================

header("LINKING FIRST CT CROP TO INDIVIDUALIZED RATE")

quality_valid = quality_valid.sort_values(
    [
        "patient_id",
        "lesion_label_alias",
        "study_date",
    ]
)

samples = []

for _, target in targets.iterrows():

    patient = int(target["patient_id"])
    alias = target["lesion_label_alias"]
    date = target["baseline_date"]

    matches = quality_valid[
        (quality_valid["patient_id"] == patient)
        & (quality_valid["lesion_label_alias"] == alias)
        & (quality_valid["study_date"] == date)
    ]

    if len(matches) == 0:
        continue

    # If multiple exact crop records exist,
    # only accept an unambiguous crop.
    if len(matches) > 1:

        paths = (
            matches["resolved_crop_path"]
            .astype(str)
            .unique()
        )

        if len(paths) != 1:
            continue

    row = matches.iloc[0]

    sample = target.to_dict()

    sample["crop_path"] = str(
        row["resolved_crop_path"]
    )

    samples.append(sample)


dataset = pd.DataFrame(samples)

print(
    "CT-to-rate samples:",
    len(dataset)
)

if len(dataset) == 0:
    print("ERROR: No CT/rate samples.")
    sys.exit(1)


# ============================================================
# SPLIT
# ============================================================

header("PATIENT-INDEPENDENT SPLIT")

dataset = dataset[
    dataset["patient_id"].isin(
        TRAIN_PATIENTS + TEST_PATIENTS
    )
].copy()

train_df = dataset[
    dataset["patient_id"].isin(
        TRAIN_PATIENTS
    )
].copy()

test_df = dataset[
    dataset["patient_id"].isin(
        TEST_PATIENTS
    )
].copy()

print("Training patients:", TRAIN_PATIENTS)
print("Testing patients:", TEST_PATIENTS)

print("Training samples:", len(train_df))
print("Testing samples:", len(test_df))

print(
    "Training patient IDs actually present:",
    sorted(train_df["patient_id"].unique().tolist())
)

print(
    "Testing patient IDs actually present:",
    sorted(test_df["patient_id"].unique().tolist())
)

if len(train_df) < 6:
    print()
    print("ERROR: Too few training samples.")
    sys.exit(1)

if len(test_df) < 3:
    print()
    print(
        "ERROR: Fewer than 3 independent test samples."
    )
    print(
        "The CT crop coverage is insufficient for a meaningful"
    )
    print(
        "CNN patient-independent experiment."
    )
    sys.exit(1)


# ============================================================
# VOLUME PREPROCESSING
# ============================================================

header("PREPARING 3D CT INPUTS")


def normalize_volume(volume):

    volume = np.nan_to_num(
        volume,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    p1 = np.percentile(volume, 1)
    p99 = np.percentile(volume, 99)

    if p99 <= p1:
        p99 = p1 + 1.0

    volume = np.clip(
        volume,
        p1,
        p99
    )

    volume = (
        volume - p1
    ) / (
        p99 - p1
    )

    return (
        volume * 2.0 - 1.0
    ).astype(np.float32)


def resize_volume(volume):

    x = torch.from_numpy(
        volume
    ).float()

    x = x.unsqueeze(0).unsqueeze(0)

    x = F.interpolate(
        x,
        size=TARGET_SHAPE,
        mode="trilinear",
        align_corners=False,
    )

    return (
        x.squeeze(0)
        .squeeze(0)
        .numpy()
    )


class CTDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = Path(
            row["crop_path"]
        )

        img = nib.load(
            str(path)
        )

        volume = img.get_fdata(
            dtype=np.float32
        )

        volume = np.squeeze(
            volume
        )

        if volume.ndim != 3:
            raise ValueError(
                f"Expected 3D volume, got {volume.shape}"
            )

        volume = normalize_volume(
            volume
        )

        volume = resize_volume(
            volume
        )

        x = torch.from_numpy(
            volume
        ).float().unsqueeze(0)

        y = torch.tensor(
            float(row["target_r"]),
            dtype=torch.float32
        )

        return x, y


train_dataset = CTDataset(
    train_df
)

test_dataset = CTDataset(
    test_df
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print(
    "Input shape:",
    TARGET_SHAPE
)


# ============================================================
# CNN
# ============================================================

header("BUILDING 3D CNN")


class HBGDCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv3d(
                1, 8,
                3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(
                8, 16,
                3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(
                16, 32,
                3,
                padding=1
            ),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d(
                (1, 1, 1)
            ),
        )

        self.head = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                32,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                1
            ),
        )

    def forward(self, x):

        x = self.encoder(x)

        return self.head(x).squeeze(1)


model = HBGDCNN().to(
    DEVICE
)

print(model)


# ============================================================
# TRAIN
# ============================================================

header("TRAINING CNN")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)

criterion = nn.SmoothL1Loss()

best_loss = float("inf")
best_state = None

for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    total = 0.0
    count = 0

    for x, y in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y_scaled = (
            y * RATE_SCALE
        )

        optimizer.zero_grad()

        pred = model(x)

        loss = criterion(
            pred,
            y_scaled
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        optimizer.step()

        total += (
            float(loss.item())
            * len(x)
        )

        count += len(x)

    epoch_loss = (
        total / max(count, 1)
    )

    if epoch_loss < best_loss:

        best_loss = epoch_loss

        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    if (
        epoch == 1
        or epoch % 5 == 0
        or epoch == EPOCHS
    ):

        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
            f" | loss = {epoch_loss:.6f}"
        )


if best_state is not None:

    model.load_state_dict(
        best_state
    )


# ============================================================
# PREDICT RATE
# ============================================================

header("PREDICTING HELD-OUT GROWTH RATES")

model.eval()

predicted_rates = []

with torch.no_grad():

    for x, _ in test_loader:

        x = x.to(DEVICE)

        pred = model(x)

        pred = (
            pred.cpu().numpy()
            / RATE_SCALE
        )

        predicted_rates.extend(
            pred.tolist()
        )

test_df = test_df.reset_index(
    drop=True
)

test_df["cnn_r"] = (
    predicted_rates
)


# ============================================================
# GLOBAL RATE
# ============================================================

global_r = float(
    train_df["target_r"].mean()
)

test_df["global_r"] = global_r


# ============================================================
# FORWARD PREDICTION
# ============================================================

header("FORWARD HBGD PREDICTION")

test_df["actual_diameter_mm"] = (
    test_df["second_diameter_mm"]
)

test_df["no_change_mm"] = (
    test_df["baseline_diameter_mm"]
)

test_df["global_mm"] = (
    test_df["baseline_diameter_mm"]
    * np.exp(
        test_df["global_r"]
        * test_df["delta_days"]
    )
)

test_df["cnn_hbgd_mm"] = (
    test_df["baseline_diameter_mm"]
    * np.exp(
        test_df["cnn_r"]
        * test_df["delta_days"]
    )
)


# ============================================================
# METRICS
# ============================================================

def metrics(actual, pred):

    actual = np.asarray(
        actual,
        dtype=float
    )

    pred = np.asarray(
        pred,
        dtype=float
    )

    err = pred - actual

    mae = np.mean(
        np.abs(err)
    )

    rmse = np.sqrt(
        np.mean(err ** 2)
    )

    return mae, rmse


rows = []

for name, column in [
    ("no_change", "no_change_mm"),
    ("global_exponential", "global_mm"),
    ("cnn_hbgd", "cnn_hbgd_mm"),
]:

    mae, rmse = metrics(
        test_df["actual_diameter_mm"],
        test_df[column]
    )

    rows.append({
        "model": name,
        "observations": len(test_df),
        "MAE_mm": mae,
        "RMSE_mm": rmse,
    })

performance = pd.DataFrame(
    rows
)


# ============================================================
# PRINT
# ============================================================

header("MODEL PERFORMANCE")

print(
    performance.to_string(
        index=False
    )
)

baseline_mae = float(
    performance.loc[
        performance["model"] == "no_change",
        "MAE_mm"
    ].iloc[0]
)

cnn_mae = float(
    performance.loc[
        performance["model"] == "cnn_hbgd",
        "MAE_mm"
    ].iloc[0]
)

global_mae = float(
    performance.loc[
        performance["model"] == "global_exponential",
        "MAE_mm"
    ].iloc[0]
)

print()
print(
    "CNN-HBGD vs no-change:",
    round(
        100 * (
            baseline_mae - cnn_mae
        ) / baseline_mae,
        2
    ),
    "%"
)

print(
    "CNN-HBGD vs global exponential:",
    round(
        100 * (
            global_mae - cnn_mae
        ) / global_mae,
        2
    ),
    "%"
)


# ============================================================
# PATIENT PERFORMANCE
# ============================================================

header("PATIENT-LEVEL PERFORMANCE")

patient_rows = []

for patient, group in test_df.groupby(
    "patient_id"
):

    row = {
        "patient_id": int(patient),
        "observations": len(group),
    }

    for name, column in [
        ("no_change", "no_change_mm"),
        ("global_exponential", "global_mm"),
        ("cnn_hbgd", "cnn_hbgd_mm"),
    ]:

        mae, rmse = metrics(
            group["actual_diameter_mm"],
            group[column]
        )

        row[
            name + "_MAE_mm"
        ] = mae

        row[
            name + "_RMSE_mm"
        ] = rmse

    patient_rows.append(row)

patient_performance = pd.DataFrame(
    patient_rows
)

print(
    patient_performance.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

header("SAVING STEP 27 RESULTS")

test_df.to_csv(
    PREDICTION_FILE,
    index=False
)

performance.to_csv(
    PERFORMANCE_FILE,
    index=False
)

patient_performance.to_csv(
    PATIENT_PERFORMANCE_FILE,
    index=False
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),
        "target_shape":
            TARGET_SHAPE,
        "rate_scale":
            RATE_SCALE,
        "train_patients":
            TRAIN_PATIENTS,
        "test_patients":
            TEST_PATIENTS,
    },
    MODEL_FILE
)

print("Saved:", PREDICTION_FILE)
print("Saved:", PERFORMANCE_FILE)
print("Saved:", PATIENT_PERFORMANCE_FILE)
print("Saved:", MODEL_FILE)


# ============================================================
# FINAL
# ============================================================

header("STEP 27 - FINAL FEASIBILITY")

print(
    "[YES] CT-to-individualized-rate samples constructed"
)

print(
    "[YES] Patient-independent split maintained"
)

print(
    "[YES] 3D CNN trained"
)

print(
    "[YES] CNN output connected to HBGD equation"
)

if cnn_mae < baseline_mae:
    print(
        "[YES] CNN-HBGD beats no-change baseline"
    )
else:
    print(
        "[NO] CNN-HBGD does not beat no-change baseline"
    )

if cnn_mae < global_mae:
    print(
        "[YES] CNN-HBGD beats global exponential"
    )
else:
    print(
        "[NO] CNN-HBGD does not beat global exponential"
    )

print()
print(
    "This remains an exploratory prototype."
)

print(
    "It is NOT clinical validation."
)

print(
    "It does NOT establish biological causality."
)

print()
print(
    "STEP 27 COMPLETE"
)