# -*- coding: utf-8 -*-
"""
STEP 27 - FAIR PATIENT-GROUPED CROSS-VALIDATION EXPERIMENT & BIOLOGICAL PARAMETER TRANSFER AUDIT

Purpose
-------
Run a leak-free, patient-grouped repeated cross-validation experiment using ONLY the
54 target development patients (train=47, validation=7). The final 13 test patients
are kept strictly untouched.

Models evaluated per fold:
1. No_Change_Baseline
2. Direct_HBGD_Historical_Rate
3. Source_HBGD_Transfer
4. Target_Morphology_History_Baseline
5. Domain_Adapted_HBGD
6. Morphology_Plus_Adapted_HBGD
7. Direct_Biological_Parameter_Transfer (Unlearned biological prior transfer)

Outputs saved to:
  DATA/LUMIRE/revised_outputs/
"""

from __future__ import print_function, division
import os
import csv
import math
import random
import json
import warnings
from collections import defaultdict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# ============================================================================
# PATHS & CONSTANTS
# ============================================================================
BASE_DIR = r"D:\VIT\PROJECT\NLP\PATENT"
SOURCE_TRAJECTORY = os.path.join(BASE_DIR, "DATA", "17_hbgd_diameter_trajectories.csv")
TARGET_TRAJECTORY = os.path.join(BASE_DIR, "DATA", "LUMIRE", "lumiere_longitudinal_trajectories.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "DATA", "LUMIRE", "revised_outputs")

RANDOM_SEED = 42
N_ESTIMATORS = 300
MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 2
DAYS_PER_WEEK = 7.0

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
# UTILITIES
# ============================================================================
def safe_float(v, default=float("nan")):
    try:
        val = float(v)
        return val if math.isfinite(val) else default
    except (TypeError, ValueError):
        return default

def finite(v):
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False

def rmse(y_true, y_pred):
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))

def metrics(y_true, y_pred):
    if len(y_true) == 0:
        return float("nan"), float("nan")
    return float(mean_absolute_error(y_true, y_pred)), rmse(y_true, y_pred)

def safe_hbgd_rate(d1, d2, elapsed_days):
    if not (finite(d1) and finite(d2) and finite(elapsed_days)):
        return float("nan")
    if d1 <= 0 or d2 <= 0 or elapsed_days <= 0:
        return float("nan")
    return math.log(d2 / d1) / float(elapsed_days)

def hbgd_predict(current_diameter, rate_per_day, future_days):
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

def read_csv(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ============================================================================
# DATA LOADING & CASE BUILDING
# ============================================================================
def load_source_trajectories():
    rows = read_csv(SOURCE_TRAJECTORY)
    trajectories = {}
    for r in rows:
        patient = str(r["patient_id"]).strip()
        lesion = str(r["lesion_alias"]).strip()
        t = safe_float(r["time_days"])
        d = safe_float(r["observed_diameter_mm"])
        if patient and lesion and finite(t) and finite(d) and d > 0:
            trajectories.setdefault((patient, lesion), []).append({"time_days": t, "diameter": d})
    for key in trajectories:
        trajectories[key].sort(key=lambda item: item["time_days"])
        deduped = {}
        for item in trajectories[key]:
            deduped[item["time_days"]] = item["diameter"]
        trajectories[key] = [{"time_days": t, "diameter": deduped[t]} for t in sorted(deduped)]
    return trajectories

def load_target_trajectories():
    rows = read_csv(TARGET_TRAJECTORY)
    trajectories = {}
    for r in rows:
        patient = str(r["patient_id"]).strip()
        w = safe_float(r["week"])
        d = safe_float(r["equivalent_diameter_mm"])
        v = safe_float(r.get("tumor_volume_cm3"))
        if patient and finite(w) and finite(d) and d > 0:
            trajectories.setdefault(patient, []).append({
                "time_days": w * DAYS_PER_WEEK,
                "week": w,
                "diameter": d,
                "volume": v
            })
    for p in trajectories:
        trajectories[p].sort(key=lambda item: item["time_days"])
        deduped = {}
        for item in trajectories[p]:
            deduped[item["time_days"]] = item
        trajectories[p] = [deduped[t] for t in sorted(deduped)]
    return trajectories

def historical_summary(history):
    rates = []
    for i in range(1, len(history)):
        older, newer = history[i - 1], history[i]
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
    cases = []
    for key, observations in trajectories.items():
        if len(observations) < 2:
            continue
        patient_id = key[0] if isinstance(key, tuple) else key
        lesion_id = key[1] if isinstance(key, tuple) else ""
        for target_index in range(1, len(observations)):
            history = observations[:target_index]
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
    return np.asarray([[c["features"].get(n, np.nan) for n in names] for c in cases], dtype=float)

def baseline_matrix(cases):
    if not cases:
        return np.empty((0, len(TARGET_BASELINE_FEATURE_NAMES)), dtype=float)
    return np.asarray([[c["baseline_features"][n] for n in TARGET_BASELINE_FEATURE_NAMES] for c in cases], dtype=float)

def labels(cases):
    return np.asarray([c["actual"] for c in cases], dtype=float)


# ============================================================================
# SOURCE MODEL FITTING
# ============================================================================
def fit_source_model(source_cases):
    X = feature_matrix(source_cases, SOURCE_FEATURE_NAMES)
    y = labels(source_cases)
    model = make_regressor(RANDOM_SEED)
    model.fit(X, y)
    return model


# ============================================================================
# BIOLOGICAL PARAMETER ESTIMATION & UNLEARNED PARAMETER TRANSFER
# ============================================================================
def estimate_source_biological_parameter(source_trajectories):
    """
    Estimate population biological growth parameter (mean logarithmic growth rate r)
    from source trajectories without using target labels.
    """
    rates = []
    for key, obs in source_trajectories.items():
        for i in range(1, len(obs)):
            dt = obs[i]["time_days"] - obs[i-1]["time_days"]
            if dt > 0 and obs[i-1]["diameter"] > 0 and obs[i]["diameter"] > 0:
                r = math.log(obs[i]["diameter"] / obs[i-1]["diameter"]) / dt
                if math.isfinite(r):
                    rates.append(r)
    mean_r = float(np.mean(rates)) if rates else 0.0
    median_r = float(np.median(rates)) if rates else 0.0
    std_r = float(np.std(rates)) if rates else 0.0
    return {"mean_growth_rate_per_day": mean_r, "median_growth_rate_per_day": median_r, "std_growth_rate_per_day": std_r}


def predict_biological_parameter_transfer(cases, mean_source_rate):
    """
    Apply pure source-estimated biological parameter r to predict target trajectories:
    D_predicted = D_current * exp(mean_source_rate * horizon_days)
    """
    preds = []
    for c in cases:
        d_curr = c["features"]["current_diameter_mm"]
        horizon = c["features"]["prediction_horizon_days"]
        p = hbgd_predict(d_curr, mean_source_rate, horizon)
        preds.append(p)
    return np.asarray(preds, dtype=float)


# ============================================================================
# PATIENT SPLITTING & REPEATED GROUPED CROSS-VALIDATION
# ============================================================================
def get_development_and_test_patients(target_trajectories):
    ids = sorted(list(target_trajectories.keys()))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(ids)
    n = len(ids)
    n_test = max(1, int(round(n * 0.20))) # 13 test patients
    test_ids = set(ids[:n_test])
    dev_ids = set(ids[n_test:]) # 54 development patients
    assert dev_ids.isdisjoint(test_ids)
    return sorted(list(dev_ids)), sorted(list(test_ids))


def run_patient_grouped_cv(target_trajectories, source_cases, source_trajectories):
    dev_ids, test_ids = get_development_and_test_patients(target_trajectories)
    print("Development patients count:", len(dev_ids))
    print("Final held-out test patients count (UNTOUCHED):", len(test_ids))

    source_model = fit_source_model(source_cases)
    bio_params = estimate_source_biological_parameter(source_trajectories)
    mean_source_r = bio_params["mean_growth_rate_per_day"]

    all_cases = build_cases(target_trajectories, target_domain=True)
    dev_cases = [c for c in all_cases if c["patient_id"] in dev_ids]

    print("Total development prediction cases:", len(dev_cases))

    # Setup 5-fold GroupKFold repeated over 5 seeds = 25 total folds
    n_splits = 5
    n_repeats = 5
    
    fold_assignment_rows = []
    cv_prediction_rows = []
    fold_metric_rows = []
    
    models_to_evaluate = [
        "No_Change_Baseline",
        "Direct_HBGD_Historical_Rate",
        "Source_HBGD_Transfer",
        "Target_Morphology_History_Baseline",
        "Domain_Adapted_HBGD",
        "Morphology_Plus_Adapted_HBGD",
        "Direct_Biological_Parameter_Transfer"
    ]

    fold_metrics_by_model = defaultdict(list)
    patient_errors_by_model = defaultdict(lambda: defaultdict(list))
    
    for r in range(n_repeats):
        seed = RANDOM_SEED + r * 10
        # Shuffle dev_ids reproducibly for repeated split
        shuffled_dev_ids = list(dev_ids)
        rng = random.Random(seed)
        rng.shuffle(shuffled_dev_ids)
        
        # Map patient_id to group index
        patient_to_group = {p: i % n_splits for i, p in enumerate(shuffled_dev_ids)}
        
        for fold in range(n_splits):
            repeat_fold_id = f"repeat_{r+1}_fold_{fold+1}"
            
            fold_val_patients = set([p for p in shuffled_dev_ids if patient_to_group[p] == fold])
            fold_train_patients = set([p for p in shuffled_dev_ids if patient_to_group[p] != fold])
            
            assert fold_train_patients.isdisjoint(fold_val_patients)
            
            # Save fold assignments
            for p in sorted(fold_val_patients):
                fold_assignment_rows.append({
                    "repeat": r + 1,
                    "fold": fold + 1,
                    "repeat_fold_id": repeat_fold_id,
                    "patient_id": p,
                    "partition": "validation"
                })
            for p in sorted(fold_train_patients):
                fold_assignment_rows.append({
                    "repeat": r + 1,
                    "fold": fold + 1,
                    "repeat_fold_id": repeat_fold_id,
                    "patient_id": p,
                    "partition": "train"
                })

            train_fold_cases = [c for c in dev_cases if c["patient_id"] in fold_train_patients]
            val_fold_cases = [c for c in dev_cases if c["patient_id"] in fold_val_patients]

            if not train_fold_cases or not val_fold_cases:
                continue

            # 1. No change baseline
            no_change_preds = np.asarray([c["features"]["current_diameter_mm"] for c in val_fold_cases], dtype=float)
            
            # 2. Direct HBGD historical rate baseline
            direct_hbgd_preds = np.asarray([c["direct_hbgd_prediction"] for c in val_fold_cases], dtype=float)
            
            # 3. Source HBGD Transfer
            X_val_src = feature_matrix(val_fold_cases, SOURCE_FEATURE_NAMES)
            source_transfer_preds = source_model.predict(X_val_src)
            
            # 4. Target Morphology History Baseline (trained on fold_train_patients ONLY)
            X_train_base = baseline_matrix(train_fold_cases)
            y_train_base = labels(train_fold_cases)
            X_val_base = baseline_matrix(val_fold_cases)
            target_base_model = make_regressor(seed + fold)
            target_base_model.fit(X_train_base, y_train_base)
            target_base_preds = target_base_model.predict(X_val_base)
            
            # 5. Domain Adapted HBGD (adaptation model trained on fold_train_patients ONLY)
            X_train_src = feature_matrix(train_fold_cases, SOURCE_FEATURE_NAMES)
            train_src_preds = source_model.predict(X_train_src)
            train_residuals = labels(train_fold_cases) - train_src_preds
            
            def make_adapt_features(s_preds, cases_list):
                rows = []
                for pred, c in zip(s_preds, cases_list):
                    f = c["features"]
                    rows.append([
                        float(pred),
                        f["current_diameter_mm"],
                        f["recent_historical_rate_per_day"],
                        f["mean_historical_rate_per_day"],
                        f["std_historical_rate_per_day"],
                        f["prediction_horizon_days"],
                        f["current_diameter_mm"] - float(pred)
                    ])
                return np.asarray(rows, dtype=float)

            X_train_adapt = make_adapt_features(train_src_preds, train_fold_cases)
            adapt_model = make_regressor(seed + fold + 100)
            adapt_model.fit(X_train_adapt, train_residuals)
            
            X_val_adapt = make_adapt_features(source_transfer_preds, val_fold_cases)
            val_corrections = adapt_model.predict(X_val_adapt)
            adapted_hbgd_preds = source_transfer_preds + val_corrections

            # 6. Combined Morphology + Adapted HBGD Model (trained using nested OOF on fold_train_patients)
            # Generate inner OOF corrections for train_fold_cases
            fold_train_groups = np.asarray([c["patient_id"] for c in train_fold_cases])
            unique_ftg = np.unique(fold_train_groups)
            oof_corrections = np.zeros(len(train_fold_cases), dtype=float)
            
            if len(unique_ftg) >= 2:
                inner_splits = min(5, len(unique_ftg))
                inner_gkf = GroupKFold(n_splits=inner_splits)
                for in_tr_idx, in_val_idx in inner_gkf.split(np.zeros(len(train_fold_cases)), labels(train_fold_cases), fold_train_groups):
                    in_tr_cases = [train_fold_cases[i] for i in in_tr_idx]
                    in_val_cases = [train_fold_cases[i] for i in in_val_idx]
                    
                    in_tr_spred = train_src_preds[in_tr_idx]
                    in_val_spred = train_src_preds[in_val_idx]
                    
                    in_tr_X_adapt = make_adapt_features(in_tr_spred, in_tr_cases)
                    in_tr_res = labels(in_tr_cases) - in_tr_spred
                    
                    in_adapt_model = make_regressor(seed + fold + 500)
                    in_adapt_model.fit(in_tr_X_adapt, in_tr_res)
                    
                    in_val_X_adapt = make_adapt_features(in_val_spred, in_val_cases)
                    oof_corrections[in_val_idx] = in_adapt_model.predict(in_val_X_adapt)

            train_comb_features = []
            for c, sp, corr in zip(train_fold_cases, train_src_preds, oof_corrections):
                train_comb_features.append(
                    [c["baseline_features"][n] for n in TARGET_BASELINE_FEATURE_NAMES] + [float(sp), float(corr), float(sp + corr)]
                )
            train_comb_features = np.asarray(train_comb_features, dtype=float)
            
            val_comb_features = []
            for c, sp, corr in zip(val_fold_cases, source_transfer_preds, val_corrections):
                val_comb_features.append(
                    [c["baseline_features"][n] for n in TARGET_BASELINE_FEATURE_NAMES] + [float(sp), float(corr), float(sp + corr)]
                )
            val_comb_features = np.asarray(val_comb_features, dtype=float)
            
            comb_model = make_regressor(seed + fold + 1000)
            comb_model.fit(train_comb_features, labels(train_fold_cases))
            combined_preds = comb_model.predict(val_comb_features)

            # 7. Direct Biological Parameter Transfer Baseline
            bio_param_preds = predict_biological_parameter_transfer(val_fold_cases, mean_source_r)

            # Store predictions dictionary
            preds_map = {
                "No_Change_Baseline": no_change_preds,
                "Direct_HBGD_Historical_Rate": direct_hbgd_preds,
                "Source_HBGD_Transfer": source_transfer_preds,
                "Target_Morphology_History_Baseline": target_base_preds,
                "Domain_Adapted_HBGD": adapted_hbgd_preds,
                "Morphology_Plus_Adapted_HBGD": combined_preds,
                "Direct_Biological_Parameter_Transfer": bio_param_preds,
            }

            y_val = labels(val_fold_cases)

            for model_name, p_arr in preds_map.items():
                mae, r_val = metrics(y_val, p_arr)
                fold_metrics_by_model[model_name].append((mae, r_val))
                fold_metric_rows.append({
                    "repeat": r + 1,
                    "fold": fold + 1,
                    "repeat_fold_id": repeat_fold_id,
                    "model": model_name,
                    "n_val_patients": len(fold_val_patients),
                    "n_val_cases": len(val_fold_cases),
                    "validation_mae_mm": mae,
                    "validation_rmse_mm": r_val,
                })

            for idx, c in enumerate(val_fold_cases):
                p_id = c["patient_id"]
                actual_val = c["actual"]
                row_dict = {
                    "repeat": r + 1,
                    "fold": fold + 1,
                    "repeat_fold_id": repeat_fold_id,
                    "patient_id": p_id,
                    "current_week": c["current_week"],
                    "future_week": c["future_week"],
                    "future_delta_weeks": c["future_delta_weeks"],
                    "actual_diameter_mm": actual_val,
                }
                for model_name, p_arr in preds_map.items():
                    p_val = float(p_arr[idx])
                    err = abs(actual_val - p_val)
                    row_dict[f"{model_name}_pred_mm"] = p_val
                    row_dict[f"{model_name}_err_mm"] = err
                    patient_errors_by_model[model_name][p_id].append(err)
                cv_prediction_rows.append(row_dict)

    # Save fold assignments, predictions, fold metrics
    write_csv(
        os.path.join(OUTPUT_DIR, "step27_fold_patient_assignments.csv"),
        fold_assignment_rows,
        ["repeat", "fold", "repeat_fold_id", "patient_id", "partition"]
    )
    
    write_csv(
        os.path.join(OUTPUT_DIR, "step27_cross_validation_predictions.csv"),
        cv_prediction_rows,
        list(cv_prediction_rows[0].keys())
    )
    
    write_csv(
        os.path.join(OUTPUT_DIR, "step27_fold_level_metrics.csv"),
        fold_metric_rows,
        ["repeat", "fold", "repeat_fold_id", "model", "n_val_patients", "n_val_cases", "validation_mae_mm", "validation_rmse_mm"]
    )

    # Aggregated model comparison across folds
    agg_rows = []
    for model_name in models_to_evaluate:
        maes = [m[0] for m in fold_metrics_by_model[model_name]]
        rmses = [m[1] for m in fold_metrics_by_model[model_name]]
        mean_mae = float(np.mean(maes))
        std_mae = float(np.std(maes, ddof=1))
        median_mae = float(np.median(maes))
        mean_rmse = float(np.mean(rmses))
        ci_lower = float(np.percentile(maes, 2.5))
        ci_upper = float(np.percentile(maes, 97.5))
        agg_rows.append({
            "model": model_name,
            "mean_cv_mae_mm": mean_mae,
            "std_cv_mae_mm": std_mae,
            "median_cv_mae_mm": median_mae,
            "mean_cv_rmse_mm": mean_rmse,
            "ci_95_lower_mae_mm": ci_lower,
            "ci_95_upper_mae_mm": ci_upper,
            "n_folds_evaluated": len(maes),
        })
    write_csv(
        os.path.join(OUTPUT_DIR, "step27_aggregated_model_comparison.csv"),
        agg_rows,
        ["model", "mean_cv_mae_mm", "std_cv_mae_mm", "median_cv_mae_mm", "mean_cv_rmse_mm", "ci_95_lower_mae_mm", "ci_95_upper_mae_mm", "n_folds_evaluated"]
    )

    # Patient-level performance & paired differences
    patient_rows = []
    for p_id in dev_ids:
        p_row = {"patient_id": p_id}
        for model_name in models_to_evaluate:
            errs = patient_errors_by_model[model_name].get(p_id, [])
            p_row[f"{model_name}_mean_mae_mm"] = float(np.mean(errs)) if errs else float("nan")
            p_row[f"{model_name}_n_evaluations"] = len(errs)
        patient_rows.append(p_row)
        
    write_csv(
        os.path.join(OUTPUT_DIR, "step27_patient_level_performance.csv"),
        patient_rows,
        list(patient_rows[0].keys())
    )

    # Paired differences between Domain_Adapted_HBGD / Morphology_Plus_Adapted vs Target_Morphology_History_Baseline
    paired_rows = []
    base_errs = patient_errors_by_model["Target_Morphology_History_Baseline"]
    adapt_errs = patient_errors_by_model["Domain_Adapted_HBGD"]
    comb_errs = patient_errors_by_model["Morphology_Plus_Adapted_HBGD"]
    src_errs = patient_errors_by_model["Source_HBGD_Transfer"]
    
    for p_id in dev_ids:
        b_mean = np.mean(base_errs.get(p_id, [np.nan]))
        a_mean = np.mean(adapt_errs.get(p_id, [np.nan]))
        c_mean = np.mean(comb_errs.get(p_id, [np.nan]))
        s_mean = np.mean(src_errs.get(p_id, [np.nan]))
        
        diff_adapt_vs_base = b_mean - a_mean # positive means adaptation improved
        diff_comb_vs_base = b_mean - c_mean
        diff_adapt_vs_src = s_mean - a_mean
        
        paired_rows.append({
            "patient_id": p_id,
            "target_baseline_mae_mm": float(b_mean),
            "source_transfer_mae_mm": float(s_mean),
            "domain_adapted_mae_mm": float(a_mean),
            "combined_model_mae_mm": float(c_mean),
            "improvement_adapted_vs_baseline_mm": float(diff_adapt_vs_base),
            "improvement_combined_vs_baseline_mm": float(diff_comb_vs_base),
            "improvement_adapted_vs_source_mm": float(diff_adapt_vs_src),
        })
    write_csv(
        os.path.join(OUTPUT_DIR, "step27_paired_differences.csv"),
        paired_rows,
        ["patient_id", "target_baseline_mae_mm", "source_transfer_mae_mm", "domain_adapted_mae_mm", "combined_model_mae_mm", "improvement_adapted_vs_baseline_mm", "improvement_combined_vs_baseline_mm", "improvement_adapted_vs_source_mm"]
    )

    # Machine-readable experiment config
    config_dict = {
        "experiment_name": "Step 27 Fair Patient-Grouped Cross-Validation & Biological Transfer Audit",
        "random_seed": RANDOM_SEED,
        "n_repeats": n_repeats,
        "n_splits": n_splits,
        "n_total_folds": len(fold_metric_rows) // len(models_to_evaluate),
        "development_patients_count": len(dev_ids),
        "test_patients_count_untouched": len(test_ids),
        "development_cases_count": len(dev_cases),
        "estimated_source_biological_parameter_r_day": bio_params,
        "days_per_week": DAYS_PER_WEEK,
    }
    with open(os.path.join(OUTPUT_DIR, "step27_experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    print("\n" + "=" * 78)
    print("STEP 27 PATIENT-GROUPED CROSS-VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 78)
    for row in agg_rows:
        print("{:<38} Mean CV MAE={:7.4f} +/- {:6.4f} mm | Median={:7.4f} mm | 95% CI=[{:6.4f}, {:6.4f}]".format(
            row["model"], row["mean_cv_mae_mm"], row["std_cv_mae_mm"], row["median_cv_mae_mm"], row["ci_95_lower_mae_mm"], row["ci_95_upper_mae_mm"]
        ))
    print("=" * 78)

if __name__ == "__main__":
    src_traj = load_source_trajectories()
    tgt_traj = load_target_trajectories()
    src_cases = build_cases(src_traj, target_domain=False)
    run_patient_grouped_cv(tgt_traj, src_cases, src_traj)
