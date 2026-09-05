import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ============================================================
# STEP 23
# CT IMAGE -> INDIVIDUALIZED HBGD GROWTH RATE
# ============================================================

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DATA")

FEATURE_FILE = os.path.join(
    DATA_DIR,
    "9_hbgd_features.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "23_ct_hbgd_rate_results.csv"
)


def header(text):
    print()
    print("=" * 95)
    print(text)
    print("=" * 95)


# ============================================================
# LOAD REQUIRED LIBRARIES
# ============================================================

header("STEP 23 - CHECKING CT PROCESSING LIBRARIES")

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is not installed.")
    print()
    print("Install with:")
    print("python -m pip install nibabel")
    sys.exit(1)

print("nibabel:", nib.__version__)


try:
    from scipy.ndimage import zoom
except ImportError:
    print("ERROR: scipy is not installed.")
    sys.exit(1)


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: PyTorch is not installed.")
    print()
    print("Install with:")
    print("python -m pip install torch")
    sys.exit(1)

print("PyTorch:", torch.__version__)


# ============================================================
# FIND DATASET ROOT
# ============================================================

header("FINDING CT DATASET")

possible_roots = [
    DATA_DIR,
    os.path.dirname(DATA_DIR),
    os.path.join(DATA_DIR, "images"),
]

dataset_root = None

for root in possible_roots:

    if not os.path.exists(root):
        continue

    for current, dirs, files in os.walk(root):

        image_count = sum(
            1
            for f in files
            if f.endswith(".nii.gz")
            and "mask" not in current.lower()
        )

        if image_count > 0:

            dataset_root = root
            break

    if dataset_root is not None:
        break


if dataset_root is None:

    print("ERROR: Could not locate CT dataset.")
    print()
    print("The script expected .nii.gz CT files somewhere")
    print("under:")
    print(DATA_DIR)
    sys.exit(1)


print("Dataset root:", dataset_root)


# ============================================================
# LOAD HBGD FEATURES
# ============================================================

header("LOADING HBGD FEATURES")

if not os.path.exists(FEATURE_FILE):

    print("ERROR: Missing:")
    print(FEATURE_FILE)
    sys.exit(1)


features = pd.read_csv(
    FEATURE_FILE
)

print(
    "Feature records:",
    len(features)
)


required = [
    "patient_id",
    "lesion_label_alias",
    "diameter_mm",
    "days_from_baseline"
]

missing = [
    c for c in required
    if c not in features.columns
]

if missing:

    print("ERROR: Missing columns:")

    for c in missing:
        print("   ", c)

    sys.exit(1)


# ============================================================
# NORMALIZE
# ============================================================

header("NORMALIZING FEATURE DATA")

features["patient_id"] = pd.to_numeric(
    features["patient_id"],
    errors="coerce"
)

features["diameter_mm"] = pd.to_numeric(
    features["diameter_mm"],
    errors="coerce"
)

features["days_from_baseline"] = pd.to_numeric(
    features["days_from_baseline"],
    errors="coerce"
)

features["lesion_label_alias"] = (
    features["lesion_label_alias"]
    .astype(str)
    .str.strip()
)


features = features.dropna(
    subset=[
        "patient_id",
        "lesion_label_alias",
        "diameter_mm",
        "days_from_baseline"
    ]
).copy()


features = features[
    features["diameter_mm"] > 0
].copy()


features = features.sort_values(
    [
        "patient_id",
        "lesion_label_alias",
        "days_from_baseline"
    ]
)


# ============================================================
# BUILD LONGITUDINAL TARGETS
# ============================================================

header("BUILDING LONGITUDINAL RATE TARGETS")

trajectory_records = []

for (
    patient_id,
    alias
), group in features.groupby(
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

    D0 = float(
        group.loc[0, "diameter_mm"]
    )

    D1 = float(
        group.loc[1, "diameter_mm"]
    )

    t0 = float(
        group.loc[0, "days_from_baseline"]
    )

    t1 = float(
        group.loc[1, "days_from_baseline"]
    )

    dt = t1 - t0

    if (
        D0 <= 0
        or D1 <= 0
        or dt <= 0
    ):
        continue

    r = np.log(D1 / D0) / dt

    trajectory_records.append(
        {
            "patient_id": patient_id,
            "lesion_alias": alias,
            "target_r": r,
            "baseline_diameter_mm": D0,
            "first_followup_diameter_mm": D1,
            "first_interval_days": dt
        }
    )


targets = pd.DataFrame(
    trajectory_records
)


print(
    "Eligible trajectories:",
    len(targets)
)

print(
    "Eligible patients:",
    targets["patient_id"].nunique()
)


if len(targets) < 6:

    print(
        "ERROR: Too few longitudinal trajectories."
    )

    sys.exit(1)


# ============================================================
# PATIENT SPLIT
# ============================================================

header("CREATING PATIENT-LEVEL SPLIT")

patients = sorted(
    targets["patient_id"]
    .unique()
)


test_patients = patients[-3:]

train_patients = [
    p
    for p in patients
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
# BUILD CT / MASK LOOKUP
# ============================================================

header("SEARCHING FOR CT AND MASK FILES")

ct_files = []
mask_files = []

for current, dirs, files in os.walk(
    dataset_root
):

    for filename in files:

        lower = filename.lower()

        if not lower.endswith(
            ".nii.gz"
        ):
            continue

        full_path = os.path.join(
            current,
            filename
        )

        if "mask" in current.lower():
            mask_files.append(
                full_path
            )

        else:
            ct_files.append(
                full_path
            )


print(
    "CT files found:",
    len(ct_files)
)

print(
    "Mask files found:",
    len(mask_files)
)


# ============================================================
# FILE INDEX
# ============================================================

header("CREATING FILE INDEX")

def clean_name(path):

    name = os.path.basename(path)

    if name.endswith(
        ".nii.gz"
    ):
        name = name[:-7]

    return name


ct_index = {
    clean_name(p): p
    for p in ct_files
}

mask_index = {
    clean_name(p): p
    for p in mask_files
}


print(
    "Indexed CT files:",
    len(ct_index)
)

print(
    "Indexed masks:",
    len(mask_index)
)


# ============================================================
# FIND SERIES USING PATIENT + DATE
# ============================================================

header("LINKING CT SERIES TO LONGITUDINAL LESIONS")

# The dataset filenames contain patient ID and DICOM UID.
# We therefore first identify files by patient ID.
#
# This script does NOT assume that a numerical mask label
# remains identical between scans.
#
# It only uses the mask corresponding to the CT series
# and selects the lesion that is linked to the RECIST alias
# through the existing feature table.


def patient_from_filename(filename):

    name = os.path.basename(filename)

    parts = name.split("_")

    if len(parts) == 0:
        return None

    try:
        return int(parts[0])
    except Exception:
        return None


patient_ct = {}

for name, path in ct_index.items():

    pid = patient_from_filename(
        name
    )

    if pid is None:
        continue

    patient_ct.setdefault(
        pid,
        []
    ).append(
        path
    )


print(
    "Patients with indexed CT files:",
    len(patient_ct)
)


# ============================================================
# FIND SERIES JSON
# ============================================================

series_json = None

for current, dirs, files in os.walk(
    dataset_root
):

    for filename in files:

        if filename.lower() == "series.json":

            series_json = os.path.join(
                current,
                filename
            )

            break

    if series_json:
        break


if series_json is None:

    print(
        "ERROR: series.json not found."
    )

    sys.exit(1)


print(
    "Series metadata:",
    series_json
)


series_records = pd.read_json(
    series_json
)


print(
    "Series records:",
    len(series_records)
)


# ============================================================
# BUILD SERIES LOOKUP
# ============================================================

header("BUILDING SERIES LOOKUP")

series_records["patient_id"] = pd.to_numeric(
    series_records["patient_id"],
    errors="coerce"
)

series_records["study_date"] = (
    series_records["study_date"]
    .astype(str)
    .str.replace(
        ".0",
        "",
        regex=False
    )
)


series_lookup = {}

for _, row in series_records.iterrows():

    pid = row["patient_id"]

    date = row["study_date"]

    uuid = str(
        row["uuid"]
    )

    key = (
        int(pid),
        date,
        uuid
    )

    series_lookup[key] = row.to_dict()


# ============================================================
# MATCH CT FILES TO SERIES
# ============================================================

ct_series_records = []

for name, path in ct_index.items():

    pid = patient_from_filename(
        name
    )

    if pid is None:
        continue

    # Search series UUID inside filename
    for _, row in series_records[
        series_records["patient_id"] == pid
    ].iterrows():

        uuid = str(
            row["uuid"]
        )

        if uuid in name:

            ct_series_records.append(
                {
                    "patient_id":
                        int(pid),

                    "study_date":
                        str(
                            row["study_date"]
                        ),

                    "uuid":
                        uuid,

                    "ct_path":
                        path,

                    "basename":
                        name
                }
            )

            break


ct_series = pd.DataFrame(
    ct_series_records
)


print(
    "CT series successfully linked:",
    len(ct_series)
)


# ============================================================
# MASK LINKING
# ============================================================

mask_lookup = {}

for name, path in mask_index.items():

    mask_lookup[name] = path


# ============================================================
# LESION CROP FUNCTION
# ============================================================

header("DEFINING 3D LESION CROP")

CROP_SIZE = (
    32,
    32,
    32
)


def load_lesion_crop(
    ct_path,
    mask_path,
    target_label,
    target_spacing=None
):

    try:

        ct_img = nib.load(
            ct_path
        )

        mask_img = nib.load(
            mask_path
        )

        ct = ct_img.get_fdata(
            dtype=np.float32
        )

        mask = mask_img.get_fdata()

    except Exception:

        return None


    if ct.shape != mask.shape:

        return None


    lesion_voxels = (
        mask == target_label
    )


    if not lesion_voxels.any():

        return None


    coords = np.argwhere(
        lesion_voxels
    )


    center = np.mean(
        coords,
        axis=0
    ).astype(int)


    z, y, x = center

    dz, dy, dx = (
        CROP_SIZE
    )


    def crop_axis(
        arr,
        center_value,
        size,
        axis
    ):

        half = size // 2

        start = (
            center_value
            -
            half
        )

        end = (
            start
            +
            size
        )


        pad_before = max(
            0,
            -start
        )

        pad_after = max(
            0,
            end - arr.shape[axis]
        )


        start = max(
            0,
            start
        )

        end = min(
            arr.shape[axis],
            end
        )


        slices = [
            slice(None),
            slice(None),
            slice(None)
        ]

        slices[axis] = slice(
            start,
            end
        )

        cropped = arr[
            tuple(slices)
        ]


        if (
            pad_before > 0
            or
            pad_after > 0
        ):

            pad_width = [
                (0, 0),
                (0, 0),
                (0, 0)
            ]

            pad_width[axis] = (
                pad_before,
                pad_after
            )

            cropped = np.pad(
                cropped,
                pad_width,
                mode="constant"
            )


        return cropped


    # Crop around lesion center
    crop = ct[
        max(0, z-dz//2):
        min(ct.shape[0], z+dz//2),

        max(0, y-dy//2):
        min(ct.shape[1], y+dy//2),

        max(0, x-dx//2):
        min(ct.shape[2], x+dx//2)
    ]


    # Pad to exact size
    pad_width = []

    for axis in range(3):

        missing = (
            CROP_SIZE[axis]
            -
            crop.shape[axis]
        )

        if missing > 0:

            before = missing // 2
            after = (
                missing
                -
                before
            )

            pad_width.append(
                (
                    before,
                    after
                )
            )

        else:

            pad_width.append(
                (0, 0)
            )


    crop = np.pad(
        crop,
        pad_width,
        mode="constant"
    )


    crop = crop[
        :CROP_SIZE[0],
        :CROP_SIZE[1],
        :CROP_SIZE[2]
    ]


    # CT intensity normalization
    crop = np.clip(
        crop,
        -1000,
        1000
    )

    crop = (
        crop + 1000
    ) / 2000.0


    crop = np.nan_to_num(
        crop,
        nan=0.5,
        posinf=1.0,
        neginf=0.0
    )


    return crop.astype(
        np.float32
    )


# ============================================================
# FIND AVAILABLE IMAGE SAMPLES
# ============================================================

header("BUILDING IMAGE SAMPLES")

# For the first CT experiment we use one baseline lesion
# crop per longitudinal trajectory.
#
# Because lesion alias -> numerical mask-label correspondence
# is dataset-specific, we search the mask labels and use
# the RECIST-associated label available in the feature data
# whenever that information exists.


samples = []

for _, target_row in targets.iterrows():

    pid = int(
        target_row["patient_id"]
    )

    alias = target_row[
        "lesion_alias"
    ]


    patient_series = ct_series[
        ct_series["patient_id"] == pid
    ].copy()


    if len(patient_series) == 0:
        continue


    # Use earliest study as baseline
    patient_series = (
        patient_series
        .sort_values(
            "study_date"
        )
    )


    baseline_series = (
        patient_series.iloc[0]
    )


    ct_path = baseline_series[
        "ct_path"
    ]


    basename = baseline_series[
        "basename"
    ]


    if basename not in mask_lookup:

        # Try basename matching without
        # possible directory differences.
        mask_path = None

        for mask_name, candidate in mask_lookup.items():

            if mask_name == basename:

                mask_path = candidate
                break

    else:

        mask_path = mask_lookup[
            basename
        ]


    if mask_path is None:
        continue


    try:

        mask_img = nib.load(
            mask_path
        )

        mask = mask_img.get_fdata()

    except Exception:

        continue


    labels = np.unique(
        mask
    )

    labels = labels[
        labels > 0
    ]


    if len(labels) == 0:
        continue


    # ========================================================
    # IMPORTANT
    # ========================================================
    #
    # We cannot safely infer that label 1=A, label 2=B, etc.
    # across the dataset.
    #
    # Therefore for this exploratory CT experiment we create
    # one sample for EACH lesion label and associate the target
    # r with the trajectory only if the number of lesions
    # permits an unambiguous one-to-one match.
    #
    # To avoid inventing lesion identity, only cases with
    # exactly one lesion in the baseline mask are used here.
    #

    if len(labels) != 1:
        continue


    label = labels[0]


    crop = load_lesion_crop(
        ct_path,
        mask_path,
        label
    )


    if crop is None:
        continue


    samples.append(
        {
            "patient_id":
                pid,

            "lesion_alias":
                alias,

            "target_r":
                float(
                    target_row[
                        "target_r"
                    ]
                ),

            "crop":
                crop
        }
    )


print(
    "Unambiguous baseline CT samples:",
    len(samples)
)


if len(samples) < 6:

    print()
    print(
        "WARNING: Fewer than 6 unambiguous CT samples "
        "were found."
    )

    print()
    print(
        "This means the current dataset linkage does not "
        "support a safe CT-to-lesion experiment yet."
    )

    print()
    print(
        "NO CNN WILL BE TRAINED."
    )

    print()
    print(
        "The correct next step is to explicitly map "
        "RECIST lesion aliases to segmentation labels."
    )

    sys.exit(0)


# ============================================================
# DATASET
# ============================================================

header("PREPARING 3D CT DATASET")


class CTRateDataset(
    Dataset
):

    def __init__(
        self,
        samples
    ):

        self.samples = samples


    def __len__(
        self
    ):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        sample = self.samples[
            index
        ]

        x = torch.tensor(
            sample["crop"],
            dtype=torch.float32
        ).unsqueeze(0)

        y = torch.tensor(
            sample["target_r"],
            dtype=torch.float32
        )

        return (
            x,
            y
        )


# ============================================================
# PATIENT SPLIT
# ============================================================

train_samples = [
    s
    for s in samples
    if s["patient_id"]
    in train_patients
]

test_samples = [
    s
    for s in samples
    if s["patient_id"]
    in test_patients
]


print(
    "Training CT samples:",
    len(train_samples)
)

print(
    "Testing CT samples:",
    len(test_samples)
)


if (
    len(train_samples) < 4
    or
    len(test_samples) < 2
):

    print(
        "ERROR: Too few CT samples for a meaningful "
        "patient-independent experiment."
    )

    sys.exit(0)


# ============================================================
# SMALL 3D CNN
# ============================================================

header("BUILDING SMALL 3D CNN")

class Small3DEncoder(
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
                1
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
        ).squeeze(1)


model = Small3DEncoder()

print(
    model
)


# ============================================================
# TRAIN
# ============================================================

header("TRAINING SMALL CT-TO-R MODEL")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


model = model.to(
    device
)


train_loader = DataLoader(
    CTRateDataset(
        train_samples
    ),
    batch_size=2,
    shuffle=True
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)


loss_function = nn.MSELoss()


EPOCHS = 50


for epoch in range(
    EPOCHS
):

    model.train()

    total_loss = 0.0


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


        loss = loss_function(
            prediction,
            y
        )


        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
        )


    if (
        epoch == 0
        or
        (epoch + 1) % 10 == 0
    ):

        average_loss = (
            total_loss
            /
            len(train_loader)
        )

        print(
            "Epoch",
            epoch + 1,
            "/",
            EPOCHS,
            "| loss =",
            round(
                average_loss,
                8
            )
        )


# ============================================================
# TEST
# ============================================================

header("PATIENT-INDEPENDENT CT-TO-R TEST")

model.eval()

test_loader = DataLoader(
    CTRateDataset(
        test_samples
    ),
    batch_size=1,
    shuffle=False
)


predicted_rates = []
true_rates = []


with torch.no_grad():

    for x, y in test_loader:

        x = x.to(
            device
        )

        prediction = model(
            x
        ).cpu().numpy()[0]


        predicted_rates.append(
            prediction
        )

        true_rates.append(
            y.numpy()[0]
        )


predicted_rates = np.array(
    predicted_rates
)

true_rates = np.array(
    true_rates
)


rate_mae = np.mean(
    np.abs(
        true_rates
        -
        predicted_rates
    )
)

rate_rmse = np.sqrt(
    np.mean(
        (
            true_rates
            -
            predicted_rates
        ) ** 2
    )
)


print(
    "Test observations:",
    len(true_rates)
)

print(
    "Rate MAE:",
    round(
        rate_mae,
        6
    )
)

print(
    "Rate RMSE:",
    round(
        rate_rmse,
        6
    )
)


# ============================================================
# FORWARD HBGD PREDICTION
# ============================================================

header("TESTING CT -> r -> HBGD FORWARD PREDICTION")

prediction_rows = []


for index, sample in enumerate(
    test_samples
):

    pid = sample[
        "patient_id"
    ]

    alias = sample[
        "lesion_alias"
    ]


    matching = [
        item
        for item in trajectories
        if (
            item["patient_id"] == pid
            and
            item["lesion_alias"] == alias
        )
    ]


    if len(matching) == 0:
        continue


    group = matching[0][
        "data"
    ]


    D0 = float(
        group.loc[
            0,
            "diameter_mm"
        ]
    )

    t0 = float(
        group.loc[
            0,
            "days_from_baseline"
        ]
    )


    predicted_r = predicted_rates[
        index
    ]


    for i in range(
        2,
        len(group)
    ):

        t = float(
            group.loc[
                i,
                "days_from_baseline"
            ]
        )

        observed = float(
            group.loc[
                i,
                "diameter_mm"
            ]
        )


        delta_t = (
            t - t0
        )


        predicted_diameter = (
            D0
            *
            np.exp(
                predicted_r
                *
                delta_t
            )
        )


        prediction_rows.append(
            {
                "patient_id":
                    pid,

                "lesion_alias":
                    alias,

                "days_ahead":
                    delta_t,

                "observed_diameter_mm":
                    observed,

                "predicted_r":
                    predicted_r,

                "predicted_diameter_mm":
                    predicted_diameter,

                "absolute_error_mm":
                    abs(
                        observed
                        -
                        predicted_diameter
                    )
            }
        )


predictions = pd.DataFrame(
    prediction_rows
)


if len(predictions) > 0:

    forward_mae = predictions[
        "absolute_error_mm"
    ].mean()

    forward_rmse = np.sqrt(
        np.mean(
            predictions[
                "absolute_error_mm"
            ] ** 2
        )
    )

else:

    forward_mae = np.nan
    forward_rmse = np.nan


print(
    "Forward observations:",
    len(predictions)
)

print(
    "CT -> r -> HBGD MAE:",
    round(
        forward_mae,
        4
    )
)

print(
    "CT -> r -> HBGD RMSE:",
    round(
        forward_rmse,
        4
    )
)


# ============================================================
# FINAL INTERPRETATION
# ============================================================

header("STEP 23 - INTERPRETATION")

print("""
This is the first experiment where the actual CT image
is used as an input to a dynamical model.

Architecture:

    CT lesion crop
          |
          v
      3D CNN
          |
          v
    predicted r
          |
          v
    dD/dt = rD
          |
          v
    future diameter

The CNN is deliberately very small because the dataset
contains very few independent patients.

This experiment is exploratory.

It does NOT establish clinical validity.

It does NOT establish biological causality.

A successful result would only indicate that the CT
lesion appearance contains information that may help
estimate an individualized dynamical parameter.

A failed result would NOT prove the HBGD architecture
is invalid because the present dataset is extremely small.
""")


# ============================================================
# FINAL
# ============================================================

header("STEP 23 - FINAL FEASIBILITY")

print(
    "[YES] Actual CT volumes were used"
)

print(
    "[YES] Patient-level separation maintained"
)

print(
    "[YES] Future observations kept as prediction targets"
)

print(
    "[YES] HBGD equation used after CT encoding"
)

print(
    "[YES] Small CNN used instead of a large model"
)


# ============================================================
# SAVE
# ============================================================

header("SAVING STEP 23 RESULTS")

if len(predictions) > 0:

    predictions.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "Saved:",
        OUTPUT_FILE
    )


summary_file = os.path.join(
    DATA_DIR,
    "23_ct_hbgd_summary.csv"
)


summary = pd.DataFrame(
    [
        {
            "test_samples":
                len(test_samples),

            "rate_MAE":
                rate_mae,

            "rate_RMSE":
                rate_rmse,

            "forward_observations":
                len(predictions),

            "forward_MAE_mm":
                forward_mae,

            "forward_RMSE_mm":
                forward_rmse
        }
    ]
)


summary.to_csv(
    summary_file,
    index=False
)


print(
    "Saved:",
    summary_file
)


header("STEP 23 COMPLETE")

print("""
IMPORTANT:

If the number of unambiguous baseline CT samples is very
small, DO NOT interpret the CNN result.

The next step should instead be explicit RECIST-label-to-mask
correspondence before attempting a larger imaging model.
""")

print("=" * 95)
print("END OF STEP 23")
print("=" * 95)