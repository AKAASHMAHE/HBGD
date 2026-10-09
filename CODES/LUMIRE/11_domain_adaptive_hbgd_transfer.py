# -*- coding: utf-8 -*-
"""
STEP 11 - LEAKAGE-SAFE DOMAIN-ADAPTIVE HBGD TRANSFER

Purpose
-------
Evaluate whether a source-cancer next-timepoint diameter model can be
transferred to the LUMIERE glioblastoma (GBM) cohort and whether a target-domain
residual correction improves prediction on held-out GBM patients.

Important methodological safeguards
-----------------------------------
1. Every feature for a prediction is computed from observations strictly before
   the target timepoint. The target diameter is used only as the training label
   or evaluation outcome, never as a feature.
2. Source training and target inference use exactly the same ordered source-model
   features and units (diameters in mm, elapsed time in days).
3. Target patients, rather than rows/timepoints, are split into train/validation/
   test partitions. All timepoints from one patient stay in one partition.
4. The final test patients are not used to fit the target adaptation or target
   comparison models.
5. The combined model uses out-of-fold (patient-grouped) adaptation corrections
   for its training rows, reducing in-sample stacking bias.
6. Source model generalization is reported on a patient-grouped source holdout;
   in-sample training metrics are not presented as validation evidence.

Inputs (edit paths if needed)
-----------------------------
Source trajectory CSV columns:
    patient_id, lesion_alias, time_days, observed_diameter_mm

Target trajectory CSV columns:
    patient_id, week, equivalent_diameter_mm, tumor_volume_cm3

Outputs
-------
step11_domain_adaptive_transfer_summary.csv
step11_domain_adaptive_transfer_predictions.csv
step11_source_model_performance.csv
step11_domain_adaptation_parameters.csv
STEP11_DOMAIN_ADAPTIVE_HBGD_TRANSFER_REPORT.md
step11_target_patient_performance.csv
step11_source_holdout_predictions.csv

Requires: numpy, scikit-learn
Compatible with Python 3.8+.
"""

from __future__ import print_function

import os
import csv
import math
import random
import warnings
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, GroupKFold


# ============================================================================
# PATHS - update these only if your project folder differs
# ============================================================================
SOURCE_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA"
TARGET_DIR = r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"

SOURCE_TRAJECTORY = os.path.join(SOURCE_DIR, "17_hbgd_diameter_trajectories.csv")
TARGET_TRAJECTORY = os.path.join(TARGET_DIR, "lumiere_longitudinal_trajectories.csv")

OUTPUT_SUMMARY = os.path.join(TARGET_DIR, "step11_domain_adaptive_transfer_summary.csv")
OUTPUT_PREDICTIONS = os.path.join(TARGET_DIR, "step11_domain_adaptive_transfer_predictions.csv")
OUTPUT_SOURCE_MODEL = os.path.join(TARGET_DIR, "step11_source_model_performance.csv")
OUTPUT_ADAPTATION = os.path.join(TARGET_DIR, "step11_domain_adaptation_parameters.csv")
OUTPUT_REPORT = os.path.join(TARGET_DIR, "STEP11_DOMAIN_ADAPTIVE_HBGD_TRANSFER_REPORT.md")
OUTPUT_PATIENT_PERFORMANCE = os.path.join(TARGET_DIR, "step11_target_patient_performance.csv")
OUTPUT_SOURCE_HOLDOUT = os.path.join(TARGET_DIR, "step11_source_holdout_predictions.csv")


# ============================================================================
# SETTINGS
# ============================================================================
RANDOM_SEED = 42
N_ESTIMATORS = 300
MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 2
DAYS_PER_WEEK = 7.0  # Verify this conversion against the dataset documentation.

# The source model feature order is identical for source training and target use.
SOURCE_FEATURE_NAMES = [
    "current_diameter_mm",
    "previous_diameter_mm",
    "recent_historical_rate_per_day",
    "mean_historical_rate_per_day",
    "std_historical_rate_per_day",
    "prediction_horizon_days",
    "history_observation_count",
    "elapsed_history_days",
    "historical_hbgd_prior_mm",
]

# Target-only comparator intentionally excludes explicit growth-rate features.
TARGET_BASELINE_FEATURE_NAMES = [
    "current_diameter_mm",
    "current_volume_cm3",
    "previous_diameter_mm",
    "diameter_change_before_target_mm",
    "prediction_horizon_days",
    "history_observation_count",
    "elapsed_history_days",
    "previous_volume_cm3",
]


# ============================================================================
# CSV / METRIC HELPERS
# ============================================================================
def read_csv(path):
    if not os.path.exists(path):
        raise IOError("Required input file not found:\n{}".format(path))
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(value, default=float("nan")):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mean(values):
    values = [float(value) for value in values if finite(value)]
    return float(np.mean(values)) if values else float("nan")


def rmse(actual, predicted):
    return float(math.sqrt(mean_squared_error(actual, predicted)))


def metrics(actual, predicted):
    if len(actual) == 0:
        return float("nan"), float("nan")
    return float(mean_absolute_error(actual, predicted)), rmse(actual, predicted)


def safe_hbgd_rate(d1, d2, elapsed_days):
    if not (finite(d1) and finite(d2) and finite(elapsed_days)):
        return float("nan")
    if d1 <= 0 or d2 <= 0 or elapsed_days <= 0:
        return float("nan")
    return math.log(d2 / d1) / float(elapsed_days)


def hbgd_predict(current_diameter, rate_per_day, future_days):
    """Numerically safe exponential prior; rate must be historical only."""
    if not (finite(current_diameter) and finite(rate_per_day) and finite(future_days)):
        return float(current_diameter) if finite(current_diameter) else float("nan")
    if current_diameter <= 0 or future_days <= 0:
        return float(current_diameter)
    exponent = max(-20.0, min(20.0, float(rate_per_day) * float(future_days)))
    return float(current_diameter * math.exp(exponent))


def make_regressor(seed=RANDOM_SEED):
    return RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=seed,
        n_jobs=-1,
    )


def ensure_required_columns(rows, required, path):
    if not rows:
        raise ValueError("Input CSV contains no rows: {}".format(path))
    missing = [column for column in required if column not in rows[0]]
    if missing:
        raise ValueError("Missing columns in {}: {}".format(path, ", ".join(missing)))


# ============================================================================
# LOAD AND NORMALIZE TRAJECTORIES
# ============================================================================
def load_source_trajectories():
    rows = read_csv(SOURCE_TRAJECTORY)
    ensure_required_columns(
        rows,
        ["patient_id", "lesion_alias", "time_days", "observed_diameter_mm"],
        SOURCE_TRAJECTORY,
    )
    trajectories = {}
    for row in rows:
        patient = str(row["patient_id"]).strip()
        lesion = str(row["lesion_alias"]).strip()
        key = (patient, lesion)
        time_days = safe_float(row["time_days"])
        diameter = safe_float(row["observed_diameter_mm"])
        if patient and lesion and finite(time_days) and finite(diameter) and diameter > 0:
            trajectories.setdefault(key, []).append({"time_days": time_days, "diameter": diameter})
    for key in trajectories:
        trajectories[key].sort(key=lambda item: item["time_days"])
        # Duplicate timestamps are ambiguous; keep the last row after stable sort.
        deduped = {}
        for item in trajectories[key]:
            deduped[item["time_days"]] = item["diameter"]
        trajectories[key] = [
            {"time_days": t, "diameter": deduped[t]} for t in sorted(deduped)
        ]
    return trajectories


def load_target_trajectories():
    rows = read_csv(TARGET_TRAJECTORY)
    ensure_required_columns(
        rows,
        ["patient_id", "week", "equivalent_diameter_mm", "tumor_volume_cm3"],
        TARGET_TRAJECTORY,
    )
    trajectories = {}
    for row in rows:
        patient = str(row["patient_id"]).strip()
        week = safe_float(row["week"])
        diameter = safe_float(row["equivalent_diameter_mm"])
        volume = safe_float(row.get("tumor_volume_cm3"))
        if patient and finite(week) and finite(diameter) and diameter > 0:
            trajectories.setdefault(patient, []).append(
                {"time_days": week * DAYS_PER_WEEK, "week": week, "diameter": diameter, "volume": volume}
            )
    for patient in trajectories:
        trajectories[patient].sort(key=lambda item: item["time_days"])
        deduped = {}
        for item in trajectories[patient]:
            deduped[item["time_days"]] = item
        trajectories[patient] = [deduped[t] for t in sorted(deduped)]
    return trajectories


# ============================================================================
# LEAKAGE-SAFE CASE CONSTRUCTION
# ============================================================================
def historical_summary(history):
    """Summarize history only. 'history' must not contain the target observation."""
    rates = []
    for index in range(1, len(history)):
        older = history[index - 1]
        newer = history[index]
        elapsed = newer["time_days"] - older["time_days"]
        rate = safe_hbgd_rate(older["diameter"], newer["diameter"], elapsed)
        if finite(rate):
            rates.append(rate)
    recent_rate = rates[-1] if rates else 0.0
    mean_rate = float(np.mean(rates)) if rates else 0.0
    std_rate = float(np.std(rates, ddof=0)) if rates else 0.0
    current = history[-1]
    previous = history[-2] if len(history) >= 2 else current
    elapsed_history = current["time_days"] - history[0]["time_days"] if len(history) >= 2 else 0.0
    return {
        "current_diameter_mm": float(current["diameter"]),
        "previous_diameter_mm": float(previous["diameter"]),
        "recent_historical_rate_per_day": float(recent_rate),
        "mean_historical_rate_per_day": float(mean_rate),
        "std_historical_rate_per_day": float(std_rate),
        "history_observation_count": int(len(history)),
        "elapsed_history_days": float(max(0.0, elapsed_history)),
        "previous_time_days": float(previous["time_days"]),
        "current_time_days": float(current["time_days"]),
        "current_volume_cm3": float(current.get("volume", 0.0)) if finite(current.get("volume", 0.0)) else 0.0,
        "previous_volume_cm3": float(previous.get("volume", 0.0)) if finite(previous.get("volume", 0.0)) else 0.0,
        "diameter_change_before_target_mm": float(current["diameter"] - previous["diameter"]),
    }


def build_cases(trajectories, target_domain=False):
    """Build next-timepoint cases. Each feature uses history[:target_index] only."""
    cases = []
    for group_key, observations in trajectories.items():
        if len(observations) < 2:
            continue
        patient_id = group_key[0] if isinstance(group_key, tuple) else group_key
        lesion_id = group_key[1] if isinstance(group_key, tuple) else ""
        for target_index in range(1, len(observations)):
            history = observations[:target_index]  # strictly before target
            target = observations[target_index]
            history_info = historical_summary(history)
            horizon_days = float(target["time_days"] - history[-1]["time_days"])
            if not finite(horizon_days) or horizon_days <= 0:
                continue
            current_d = history_info["current_diameter_mm"]
            recent_rate = history_info["recent_historical_rate_per_day"]
            prior_prediction = hbgd_predict(current_d, recent_rate, horizon_days) if len(history) >= 2 else current_d
            if not finite(prior_prediction):
                prior_prediction = current_d
            features = {
                "current_diameter_mm": current_d,
                "previous_diameter_mm": history_info["previous_diameter_mm"],
                "recent_historical_rate_per_day": recent_rate,
                "mean_historical_rate_per_day": history_info["mean_historical_rate_per_day"],
                "std_historical_rate_per_day": history_info["std_historical_rate_per_day"],
                "prediction_horizon_days": horizon_days,
                "history_observation_count": history_info["history_observation_count"],
                "elapsed_history_days": history_info["elapsed_history_days"],
                "historical_hbgd_prior_mm": prior_prediction,
            }
            baseline_features = {
                "current_diameter_mm": current_d,
                "current_volume_cm3": history_info["current_volume_cm3"] if target_domain else 0.0,
                "previous_diameter_mm": history_info["previous_diameter_mm"],
                "diameter_change_before_target_mm": history_info["diameter_change_before_target_mm"],
                "prediction_horizon_days": horizon_days,
                "history_observation_count": history_info["history_observation_count"],
                "elapsed_history_days": history_info["elapsed_history_days"],
                "previous_volume_cm3": history_info["previous_volume_cm3"] if target_domain else 0.0,
            }
            # Explicit invariant: the target observation is not in history/features.
            cases.append({
                "patient_id": str(patient_id),
                "lesion_id": str(lesion_id),
                "current_time_days": float(history[-1]["time_days"]),
                "target_time_days": float(target["time_days"]),
                "current_week": float(history[-1].get("week", history[-1]["time_days"] / DAYS_PER_WEEK)),
                "future_week": float(target.get("week", target["time_days"] / DAYS_PER_WEEK)),
                "future_delta_weeks": float(horizon_days / DAYS_PER_WEEK),
                "actual": float(target["diameter"]),
                "direct_hbgd_prediction": float(prior_prediction),
                "features": features,
                "baseline_features": baseline_features,
                "history_count": len(history),
            })
    return cases


def feature_matrix(cases, names):
    if not cases:
        return np.empty((0, len(names)), dtype=float)
    return np.asarray([[case["features"].get(name, np.nan) for name in names] for case in cases], dtype=float)


def baseline_matrix(cases):
    if not cases:
        return np.empty((0, len(TARGET_BASELINE_FEATURE_NAMES)), dtype=float)
    return np.asarray([[case["baseline_features"][name] for name in TARGET_BASELINE_FEATURE_NAMES] for case in cases], dtype=float)


def labels(cases):
    return np.asarray([case["actual"] for case in cases], dtype=float)


def assert_finite_matrix(matrix, label):
    if matrix.size and not np.isfinite(matrix).all():
        raise ValueError("Non-finite feature value detected in {} matrix.".format(label))


# ============================================================================
# SOURCE MODEL: GROUPED HOLDOUT EVALUATION + FINAL SOURCE FIT
# ============================================================================
def fit_source_models(source_cases):
    if len(source_cases) < 10:
        raise ValueError("Too few source prediction cases: {}".format(len(source_cases)))
    X = feature_matrix(source_cases, SOURCE_FEATURE_NAMES)
    y = labels(source_cases)
    groups = np.asarray([case["patient_id"] for case in source_cases])
    assert_finite_matrix(X, "source")
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Source cohort needs at least two unique patients for grouped holdout evaluation.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    train_idx, holdout_idx = next(splitter.split(X, y, groups))
    holdout_model = make_regressor(RANDOM_SEED)
    holdout_model.fit(X[train_idx], y[train_idx])
    holdout_pred = holdout_model.predict(X[holdout_idx])
    holdout_mae, holdout_rmse = metrics(y[holdout_idx], holdout_pred)
    holdout_rows = []
    for idx, pred in zip(holdout_idx, holdout_pred):
        case = source_cases[int(idx)]
        holdout_rows.append({
            "patient_id": case["patient_id"],
            "lesion_id": case["lesion_id"],
            "current_time_days": case["current_time_days"],
            "target_time_days": case["target_time_days"],
            "actual_diameter_mm": case["actual"],
            "predicted_diameter_mm": float(pred),
            "absolute_error_mm": abs(case["actual"] - float(pred)),
            "partition": "source_patient_holdout",
        })

    # Refit final transfer model on all source cases after obtaining holdout estimate.
    final_model = make_regressor(RANDOM_SEED)
    final_model.fit(X, y)
    source_metrics = {
        "source_cases": int(len(source_cases)),
        "source_patients": int(len(unique_groups)),
        "grouped_holdout_cases": int(len(holdout_idx)),
        "grouped_holdout_patients": int(len(np.unique(groups[holdout_idx]))),
        "grouped_holdout_mae_mm": holdout_mae,
        "grouped_holdout_rmse_mm": holdout_rmse,
        "final_model_training_mae_not_generalization_mm": float(mean_absolute_error(y, final_model.predict(X))),
        "final_model_training_rmse_not_generalization_mm": rmse(y, final_model.predict(X)),
    }
    return final_model, source_metrics, holdout_rows


# ============================================================================
# DOMAIN ADAPTATION
# ============================================================================
def source_predict(model, cases):
    X = feature_matrix(cases, SOURCE_FEATURE_NAMES)
    assert_finite_matrix(X, "source inference")
    return model.predict(X)


def adaptation_features(source_predictions, cases):
    rows = []
    for pred, case in zip(source_predictions, cases):
        f = case["features"]
        rows.append([
            float(pred),
            f["current_diameter_mm"],
            f["recent_historical_rate_per_day"],
            f["mean_historical_rate_per_day"],
            f["std_historical_rate_per_day"],
            f["prediction_horizon_days"],
            f["current_diameter_mm"] - float(pred),
        ])
    return np.asarray(rows, dtype=float)


def fit_adaptation_model(source_predictions, train_cases):
    X = adaptation_features(source_predictions, train_cases)
    residual = labels(train_cases) - np.asarray(source_predictions, dtype=float)
    assert_finite_matrix(X, "adaptation training")
    model = make_regressor(RANDOM_SEED + 1)
    model.fit(X, residual)
    return model


def predict_adapted(source_model, adaptation_model, cases):
    source_pred = source_predict(source_model, cases)
    X_adapt = adaptation_features(source_pred, cases)
    correction = adaptation_model.predict(X_adapt)
    adapted = source_pred + correction
    return source_pred, correction, adapted


def grouped_oof_adaptation_corrections(source_model, train_cases):
    """OOF residual corrections for training rows used by the stacked model."""
    groups = np.asarray([case["patient_id"] for case in train_cases])
    unique_groups = np.unique(groups)
    source_pred = source_predict(source_model, train_cases)
    if len(unique_groups) < 2:
        # Conservative fallback for tiny cohorts; caller is warned in report.
        warnings.warn("Fewer than two training patients; using zero OOF corrections.")
        return source_pred, np.zeros(len(train_cases), dtype=float), source_pred.copy()
    n_splits = min(5, len(unique_groups))
    splitter = GroupKFold(n_splits=n_splits)
    corrections = np.zeros(len(train_cases), dtype=float)
    for fit_idx, held_idx in splitter.split(np.zeros(len(train_cases)), labels(train_cases), groups):
        fold_cases = [train_cases[int(i)] for i in fit_idx]
        held_cases = [train_cases[int(i)] for i in held_idx]
        fold_source_pred = source_pred[fit_idx]
        fold_model = fit_adaptation_model(fold_source_pred, fold_cases)
        held_source_pred = source_pred[held_idx]
        held_X = adaptation_features(held_source_pred, held_cases)
        corrections[held_idx] = fold_model.predict(held_X)
    return source_pred, corrections, source_pred + corrections


# ============================================================================
# SPLIT TARGET PATIENTS BEFORE BUILDING PARTITIONS
# ============================================================================
def split_target_patients(patient_ids):
    ids = list(sorted(patient_ids))
    if len(ids) < 5:
        raise ValueError("Need at least 5 target patients for train/validation/test splitting; found {}.".format(len(ids)))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * 0.20)))
    remaining = ids[n_test:]
    n_val = max(1, int(round(len(remaining) * 0.125)))
    test_ids = set(ids[:n_test])
    val_ids = set(remaining[:n_val])
    train_ids = set(remaining[n_val:])
    if not train_ids or not val_ids or not test_ids:
        raise ValueError("Target patient split produced an empty partition.")
    assert train_ids.isdisjoint(val_ids) and train_ids.isdisjoint(test_ids) and val_ids.isdisjoint(test_ids)
    return train_ids, val_ids, test_ids


def subset_cases(cases, patient_ids):
    return [case for case in cases if case["patient_id"] in patient_ids]


def evaluate_direct(cases):
    actual = labels(cases)
    predicted = np.asarray([case["direct_hbgd_prediction"] for case in cases], dtype=float)
    return metrics(actual, predicted), predicted


def evaluate_no_change(cases):
    actual = labels(cases)
    predicted = np.asarray([case["features"]["current_diameter_mm"] for case in cases], dtype=float)
    return metrics(actual, predicted), predicted


def patient_metrics_rows(cases, prediction_dict):
    grouped = defaultdict(list)
    for idx, case in enumerate(cases):
        grouped[case["patient_id"]].append((case, idx))
    rows = []
    for patient, entries in sorted(grouped.items()):
        actual = [entry[0]["actual"] for entry in entries]
        row = {"patient_id": patient, "n_predictions": len(entries)}
        for model_name, preds in prediction_dict.items():
            pred = [float(preds[idx]) for _, idx in entries]
            mae, patient_rmse = metrics(actual, pred)
            row[model_name + "_mae_mm"] = mae
            row[model_name + "_rmse_mm"] = patient_rmse
        rows.append(row)
    return rows


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 78)
    print("STEP 11 - LEAKAGE-SAFE DOMAIN-ADAPTIVE HBGD TRANSFER")
    print("=" * 78)

    # Build source cases; every case feature uses only earlier observations.
    source_trajectories = load_source_trajectories()
    source_cases = build_cases(source_trajectories, target_domain=False)
    print("Source trajectories:", len(source_trajectories))
    print("Source next-timepoint cases:", len(source_cases))
    source_model, source_perf, source_holdout_rows = fit_source_models(source_cases)

    # Load target cohort and make patient-level split.
    target_trajectories = load_target_trajectories()
    print("LUMIERE patients:", len(target_trajectories))
    train_ids, val_ids, test_ids = split_target_patients(target_trajectories.keys())
    print("Patient split: train={}, validation={}, test={}".format(len(train_ids), len(val_ids), len(test_ids)))

    all_target_cases = build_cases(target_trajectories, target_domain=True)
    train_cases = subset_cases(all_target_cases, train_ids)
    val_cases = subset_cases(all_target_cases, val_ids)
    test_cases = subset_cases(all_target_cases, test_ids)
    if not train_cases or not val_cases or not test_cases:
        raise ValueError("At least one target partition has no eligible prediction cases. Check trajectory lengths.")
    print("Prediction cases: train={}, validation={}, test={}".format(len(train_cases), len(val_cases), len(test_cases)))

    # Target-only comparator: current/past morphology and observation timing; no rate features.
    X_train_base = baseline_matrix(train_cases)
    X_val_base = baseline_matrix(val_cases)
    X_test_base = baseline_matrix(test_cases)
    assert_finite_matrix(X_train_base, "target baseline train")
    assert_finite_matrix(X_val_base, "target baseline validation")
    assert_finite_matrix(X_test_base, "target baseline test")
    baseline_model = make_regressor(RANDOM_SEED + 2)
    baseline_model.fit(X_train_base, labels(train_cases))
    baseline_val_pred = baseline_model.predict(X_val_base)
    baseline_test_pred = baseline_model.predict(X_test_base)

    # Direct HBGD prior, no learned source model.
    (direct_val_mae, direct_val_rmse), direct_val_pred = evaluate_direct(val_cases)
    (direct_test_mae, direct_test_rmse), direct_test_pred = evaluate_direct(test_cases)

    # Source model transfer and target residual adaptation.
    source_val_pred = source_predict(source_model, val_cases)
    source_test_pred = source_predict(source_model, test_cases)
    source_val_mae, source_val_rmse = metrics(labels(val_cases), source_val_pred)
    source_test_mae, source_test_rmse = metrics(labels(test_cases), source_test_pred)

    train_source_pred = source_predict(source_model, train_cases)
    adaptation_model = fit_adaptation_model(train_source_pred, train_cases)
    adapted_val_source, correction_val, adapted_val_pred = predict_adapted(source_model, adaptation_model, val_cases)
    adapted_test_source, correction_test, adapted_test_pred = predict_adapted(source_model, adaptation_model, test_cases)
    adapted_val_mae, adapted_val_rmse = metrics(labels(val_cases), adapted_val_pred)
    adapted_test_mae, adapted_test_rmse = metrics(labels(test_cases), adapted_test_pred)

    # Stacked model: train with grouped OOF corrections, evaluate with final adaptation model.
    _, oof_train_correction, _ = grouped_oof_adaptation_corrections(source_model, train_cases)
    train_adapted_features = []
    for case, spred, corr in zip(train_cases, train_source_pred, oof_train_correction):
        train_adapted_features.append(
            [case["baseline_features"][name] for name in TARGET_BASELINE_FEATURE_NAMES]
            + [float(spred), float(corr), float(spred + corr)]
        )
    train_adapted_features = np.asarray(train_adapted_features, dtype=float)

    def combined_feature_matrix(cases):
        spred, correction, adapted = predict_adapted(source_model, adaptation_model, cases)
        rows = []
        for case, s, c, a in zip(cases, spred, correction, adapted):
            rows.append([case["baseline_features"][name] for name in TARGET_BASELINE_FEATURE_NAMES] + [float(s), float(c), float(a)])
        return np.asarray(rows, dtype=float)

    X_val_combined = combined_feature_matrix(val_cases)
    X_test_combined = combined_feature_matrix(test_cases)
    assert_finite_matrix(train_adapted_features, "combined train")
    assert_finite_matrix(X_val_combined, "combined validation")
    assert_finite_matrix(X_test_combined, "combined test")
    combined_model = make_regressor(RANDOM_SEED + 3)
    combined_model.fit(train_adapted_features, labels(train_cases))
    combined_val_pred = combined_model.predict(X_val_combined)
    combined_test_pred = combined_model.predict(X_test_combined)
    combined_val_mae, combined_val_rmse = metrics(labels(val_cases), combined_val_pred)
    combined_test_mae, combined_test_rmse = metrics(labels(test_cases), combined_test_pred)

    # No-change baseline for context.
    (nochange_val_mae, nochange_val_rmse), nochange_val_pred = evaluate_no_change(val_cases)
    (nochange_test_mae, nochange_test_rmse), nochange_test_pred = evaluate_no_change(test_cases)

    summary_rows = [
        {"model": "No_Change_Baseline", "validation_mae_mm": nochange_val_mae, "validation_rmse_mm": nochange_val_rmse, "test_mae_mm": nochange_test_mae, "test_rmse_mm": nochange_test_rmse, "test_predictions": len(test_cases)},
        {"model": "Direct_HBGD_Historical_Rate", "validation_mae_mm": direct_val_mae, "validation_rmse_mm": direct_val_rmse, "test_mae_mm": direct_test_mae, "test_rmse_mm": direct_test_rmse, "test_predictions": len(test_cases)},
        {"model": "Source_HBGD_Transfer", "validation_mae_mm": source_val_mae, "validation_rmse_mm": source_val_rmse, "test_mae_mm": source_test_mae, "test_rmse_mm": source_test_rmse, "test_predictions": len(test_cases)},
        {"model": "Target_Morphology_History_Baseline", "validation_mae_mm": metrics(labels(val_cases), baseline_val_pred)[0], "validation_rmse_mm": metrics(labels(val_cases), baseline_val_pred)[1], "test_mae_mm": metrics(labels(test_cases), baseline_test_pred)[0], "test_rmse_mm": metrics(labels(test_cases), baseline_test_pred)[1], "test_predictions": len(test_cases)},
        {"model": "Domain_Adapted_HBGD", "validation_mae_mm": adapted_val_mae, "validation_rmse_mm": adapted_val_rmse, "test_mae_mm": adapted_test_mae, "test_rmse_mm": adapted_test_rmse, "test_predictions": len(test_cases)},
        {"model": "Morphology_Plus_Adapted_HBGD", "validation_mae_mm": combined_val_mae, "validation_rmse_mm": combined_val_rmse, "test_mae_mm": combined_test_mae, "test_rmse_mm": combined_test_rmse, "test_predictions": len(test_cases)},
    ]
    write_csv(OUTPUT_SUMMARY, summary_rows, ["model", "validation_mae_mm", "validation_rmse_mm", "test_mae_mm", "test_rmse_mm", "test_predictions"])

    source_perf_rows = [{"metric": key, "value": value} for key, value in source_perf.items()]
    source_perf_rows.append({"metric": "source_feature_names", "value": ";".join(SOURCE_FEATURE_NAMES)})
    source_perf_rows.append({"metric": "source_features_use_target_diameter", "value": "NO - historical observations only"})
    write_csv(OUTPUT_SOURCE_MODEL, source_perf_rows, ["metric", "value"])

    # Prediction-level audit output.
    prediction_rows = []
    for idx, case in enumerate(test_cases):
        prediction_rows.append({
            "patient_id": case["patient_id"],
            "lesion_id": case["lesion_id"],
            "current_week": case["current_week"],
            "future_week": case["future_week"],
            "future_delta_weeks": case["future_delta_weeks"],
            "actual_diameter_mm": case["actual"],
            "direct_hbgd_prediction_mm": float(direct_test_pred[idx]),
            "source_model_prediction_mm": float(source_test_pred[idx]),
            "domain_correction_mm": float(correction_test[idx]),
            "domain_adapted_prediction_mm": float(adapted_test_pred[idx]),
            "target_morphology_history_prediction_mm": float(baseline_test_pred[idx]),
            "morphology_plus_adapted_prediction_mm": float(combined_test_pred[idx]),
            "no_change_prediction_mm": float(nochange_test_pred[idx]),
            "direct_hbgd_error_mm": abs(case["actual"] - float(direct_test_pred[idx])),
            "source_transfer_error_mm": abs(case["actual"] - float(source_test_pred[idx])),
            "adapted_hbgd_error_mm": abs(case["actual"] - float(adapted_test_pred[idx])),
            "target_baseline_error_mm": abs(case["actual"] - float(baseline_test_pred[idx])),
            "combined_error_mm": abs(case["actual"] - float(combined_test_pred[idx])),
            "no_change_error_mm": abs(case["actual"] - float(nochange_test_pred[idx])),
            "history_observation_count": case["history_count"],
        })
    write_csv(OUTPUT_PREDICTIONS, prediction_rows, list(prediction_rows[0].keys()))

    # Adaptation summary.
    adaptation_rows = [
        {"parameter": "mean_validation_correction_mm", "value": mean(correction_val)},
        {"parameter": "std_validation_correction_mm", "value": float(np.std(correction_val))},
        {"parameter": "mean_test_correction_mm", "value": mean(correction_test)},
        {"parameter": "std_test_correction_mm", "value": float(np.std(correction_test))},
        {"parameter": "source_feature_count", "value": len(SOURCE_FEATURE_NAMES)},
        {"parameter": "source_features_use_target_diameter", "value": "NO"},
        {"parameter": "target_test_patients_used_in_fit", "value": "NO"},
    ]
    write_csv(OUTPUT_ADAPTATION, adaptation_rows, ["parameter", "value"])

    # Per-patient metrics on the untouched target test cohort.
    patient_rows = patient_metrics_rows(test_cases, {
        "no_change": nochange_test_pred,
        "source_transfer": source_test_pred,
        "adapted_hbgd": adapted_test_pred,
        "target_baseline": baseline_test_pred,
        "combined": combined_test_pred,
    })
    write_csv(OUTPUT_PATIENT_PERFORMANCE, patient_rows, list(patient_rows[0].keys()) if patient_rows else ["patient_id", "n_predictions"])
    write_csv(OUTPUT_SOURCE_HOLDOUT, source_holdout_rows, [
        "patient_id", "lesion_id", "current_time_days", "target_time_days", "actual_diameter_mm", "predicted_diameter_mm", "absolute_error_mm", "partition"
    ])

    # Relative changes are positive when the second model has lower MAE.
    geometry_mae = metrics(labels(test_cases), baseline_test_pred)[0]
    combined_mae = combined_test_mae
    combined_improvement_pct = ((geometry_mae - combined_mae) / geometry_mae * 100.0) if geometry_mae else float("nan")
    source_to_adapt_improvement_pct = ((source_test_mae - adapted_test_mae) / source_test_mae * 100.0) if source_test_mae else float("nan")

    report_lines = [
        "# Step 11 — Leakage-Safe Domain-Adaptive HBGD Transfer",
        "",
        "## Objective",
        "",
        "Evaluate source-cancer next-timepoint diameter prediction in the LUMIERE GBM cohort, then learn a target-domain residual correction using target training patients only.",
        "",
        "## Leakage safeguards",
        "",
        "- Every prediction feature is constructed from observations strictly before the target timepoint.",
        "- The target diameter is used only as the supervised label or evaluation outcome.",
        "- Source training and target inference use the same ordered feature schema and day-based time units.",
        "- Target patient IDs are split before evaluation; no patient contributes timepoints to multiple partitions.",
        "- The combined model uses patient-grouped out-of-fold adaptation corrections for its training rows.",
        "- Source generalization is estimated using a patient-grouped source holdout. The separate source training metrics are explicitly labelled as in-sample only.",
        "",
        "## Patient split",
        "",
        "| Partition | Patients | Prediction cases |",
        "|---|---:|---:|",
        "| Train | {} | {} |".format(len(train_ids), len(train_cases)),
        "| Validation | {} | {} |".format(len(val_ids), len(val_cases)),
        "| Test | {} | {} |".format(len(test_ids), len(test_cases)),
        "",
        "## Results",
        "",
        "| Model | Validation MAE (mm) | Validation RMSE (mm) | Test MAE (mm) | Test RMSE (mm) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            row["model"], row["validation_mae_mm"], row["validation_rmse_mm"], row["test_mae_mm"], row["test_rmse_mm"]
        ))
    report_lines += [
        "",
        "## Key comparisons",
        "",
        "- Source-transfer test MAE: {:.4f} mm.".format(source_test_mae),
        "- Domain-adapted HBGD test MAE: {:.4f} mm.".format(adapted_test_mae),
        "- Relative MAE reduction from source transfer to domain-adapted HBGD: {:.2f}%.".format(source_to_adapt_improvement_pct),
        "- Target morphology/history baseline test MAE: {:.4f} mm.".format(geometry_mae),
        "- Morphology + adapted HBGD test MAE: {:.4f} mm.".format(combined_mae),
        "- Relative MAE reduction from target baseline to combined model: {:.2f}%. Positive means the combined model is better; negative means it is worse.".format(combined_improvement_pct),
        "",
        "## Source cohort holdout",
        "",
        "- Source cases: {} across {} source patients.".format(source_perf["source_cases"], source_perf["source_patients"]),
        "- Patient-grouped source holdout MAE: {:.4f} mm; RMSE: {:.4f} mm.".format(source_perf["grouped_holdout_mae_mm"], source_perf["grouped_holdout_rmse_mm"]),
        "- In-sample source training MAE/RMSE are provided for debugging only and must not be interpreted as generalization performance.",
        "",
        "## Interpretation",
        "",
        "This run evaluates transfer of a source-trained prediction model plus a target-domain residual correction. It is not, by itself, proof that a biological growth parameter transfers unchanged between cancer types. Any claimed benefit must be judged against the held-out target baselines and patient-level variability.",
        "",
        "## Limitations and checks",
        "",
        "- The target cohort has a limited number of held-out patients; confidence intervals or repeated patient-level splits are recommended.",
        "- Equivalent diameter and source observed diameter may be produced by different measurement pipelines.",
        "- The conversion of week to seven days must be confirmed from the target dataset documentation.",
        "- Results are next-observed-timepoint predictions; they do not establish long-horizon forecasting or clinical utility.",
        "- Review step11_target_patient_performance.csv; pooled metrics can conceal patient-level failures.",
        "",
        "## Final conclusion",
        "",
        "Interpret the regenerated metrics only after confirming input schemas, time units, and the patient split. Claim cross-cancer predictive benefit only if the corrected held-out target results support it. Distinguish predictive-model adaptation from direct biological parameter transfer.",
    ]
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines) + "\n")

    print("\nRESULTS")
    print("-" * 78)
    for row in summary_rows:
        print("{:<38} val MAE={:8.4f} mm | test MAE={:8.4f} mm | test RMSE={:8.4f} mm".format(
            row["model"], row["validation_mae_mm"], row["test_mae_mm"], row["test_rmse_mm"]
        ))
    print("-" * 78)
    print("Source patient-grouped holdout: MAE={:.4f} mm, RMSE={:.4f} mm".format(
        source_perf["grouped_holdout_mae_mm"], source_perf["grouped_holdout_rmse_mm"]
    ))
    print("Combined vs target baseline relative MAE reduction: {:.2f}%".format(combined_improvement_pct))
    print("\nOutputs written to:\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}\n  {}".format(
        OUTPUT_SUMMARY, OUTPUT_PREDICTIONS, OUTPUT_SOURCE_MODEL, OUTPUT_ADAPTATION,
        OUTPUT_REPORT, OUTPUT_PATIENT_PERFORMANCE, OUTPUT_SOURCE_HOLDOUT
    ))


if __name__ == "__main__":
    main()
