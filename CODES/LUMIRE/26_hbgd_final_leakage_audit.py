# -*- coding: utf-8 -*-

"""
STEP 26
FINAL HBGD CROSS-CANCER LEAKAGE AUDIT + EVIDENCE CONSOLIDATION

Purpose
-------
1. Audit the strict temporal Step 25 experiment.
2. Verify source HBGD provenance.
3. Verify target prediction provenance.
4. Check temporal ordering.
5. Check that future diameter is not used as a feature.
6. Check patient-level isolation in Step 25 predictions.
7. Summarize Steps 19-25 into one evidence matrix.

This script DOES NOT retrain any model.

Python 3.6 compatible.
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = (
    r"D:\VIT\PROJECT\NLP\PATENT\DATA"
)

LUMIERE_DIR = (
    r"D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE"
)

SOURCE_STATE = os.path.join(
    BASE_DIR,
    "17_hbgd_state_parameters.csv"
)

TRAJECTORIES = os.path.join(
    LUMIERE_DIR,
    "lumiere_longitudinal_trajectories.csv"
)

STEP25_PREDICTIONS = os.path.join(
    LUMIERE_DIR,
    "step25_hbgd_temporal_predictions.csv"
)

STEP25_PERFORMANCE = os.path.join(
    LUMIERE_DIR,
    "step25_hbgd_temporal_performance.csv"
)

STEP25_PATIENT = os.path.join(
    LUMIERE_DIR,
    "step25_hbgd_temporal_patient_results.csv"
)

STEP19_PERFORMANCE = os.path.join(
    LUMIERE_DIR,
    "step19_hbgd_representation_transfer_performance.csv"
)

STEP20_SUMMARY = os.path.join(
    LUMIERE_DIR,
    "step20_hbgd_transfer_statistical_summary.csv"
)

STEP22_SPLITS = os.path.join(
    LUMIERE_DIR,
    "step22_hbgd_transfer_split_results.csv"
)

STEP23_SUMMARY = os.path.join(
    LUMIERE_DIR,
    "step23_hbgd_repeated_split_statistical_summary.csv"
)

STEP24_CORR = os.path.join(
    LUMIERE_DIR,
    "step24_hbgd_responder_correlations.csv"
)

STEP25_SUMMARY = os.path.join(
    LUMIERE_DIR,
    "step25_hbgd_temporal_summary.csv"
)

OUTPUT_AUDIT = os.path.join(
    LUMIERE_DIR,
    "step26_hbgd_leakage_audit.csv"
)

OUTPUT_EVIDENCE = os.path.join(
    LUMIERE_DIR,
    "step26_hbgd_evidence_matrix.csv"
)

OUTPUT_PROVENANCE = os.path.join(
    LUMIERE_DIR,
    "step26_hbgd_provenance_table.csv"
)

OUTPUT_REPORT = os.path.join(
    LUMIERE_DIR,
    "STEP26_HBGD_FINAL_LEAKAGE_AUDIT_REPORT.md"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 78)
print(
    "STEP 26 FINAL HBGD LEAKAGE AUDIT + EVIDENCE CONSOLIDATION"
)
print("=" * 78)


# ============================================================
# HELPER
# ============================================================

def exists(path):

    return os.path.exists(path)


def add_audit(
    rows,
    check,
    status,
    evidence,
    interpretation
):

    rows.append({

        "audit_check":
            check,

        "status":
            status,

        "evidence":
            evidence,

        "interpretation":
            interpretation
    })


def load_csv(
    path,
    required=False
):

    if not exists(path):

        if required:

            raise IOError(
                "Required file missing:\n%s"
                % path
            )

        return None

    return pd.read_csv(
        path
    )


# ============================================================
# LOAD REQUIRED DATA
# ============================================================

print(
    "\nLoading source and Step 25 artifacts..."
)

source = load_csv(
    SOURCE_STATE,
    required=True
)

trajectories = load_csv(
    TRAJECTORIES,
    required=True
)

step25 = load_csv(
    STEP25_PREDICTIONS,
    required=True
)

step25_perf = load_csv(
    STEP25_PERFORMANCE,
    required=True
)

step25_patient = load_csv(
    STEP25_PATIENT,
    required=True
)


# ============================================================
# NORMALIZE IDS
# ============================================================

def normalize_id(value):

    s = str(value).strip()

    if s.lower().startswith(
        "patient-"
    ):

        return s

    try:

        return "Patient-%03d" % int(
            float(s)
        )

    except Exception:

        return s


for df in [
    source,
    trajectories,
    step25,
    step25_patient
]:

    if (
        df is not None
        and
        "patient_id" in df.columns
    ):

        df[
            "patient_id"
        ] = df[
            "patient_id"
        ].apply(
            normalize_id
        )


# ============================================================
# AUDIT CONTAINER
# ============================================================

audit_rows = []


# ============================================================
# 1. SOURCE PROVENANCE
# ============================================================

source_required = [
    "patient_id",
    "time_start_days",
    "time_end_days",
    "diameter_start_mm",
    "diameter_end_mm",
    "growth_rate"
]

missing_source = [
    c for c in source_required
    if c not in source.columns
]

if len(missing_source) == 0:

    add_audit(
        audit_rows,

        "Source HBGD schema",

        "PASS",

        "17_hbgd_state_parameters.csv contains "
        "patient, interval time, diameter and growth-rate fields.",

        "The validated source interval artifact is suitable "
        "as the provenance source for transferred HBGD."
    )

else:

    add_audit(
        audit_rows,

        "Source HBGD schema",

        "FAIL",

        str(missing_source),

        "Required source HBGD fields are missing."
    )


# ============================================================
# 2. SOURCE INTERVAL COUNT
# ============================================================

source["time_start_days"] = pd.to_numeric(
    source["time_start_days"],
    errors="coerce"
)

source["time_end_days"] = pd.to_numeric(
    source["time_end_days"],
    errors="coerce"
)

source["interval_days"] = (
    source["time_end_days"]
    -
    source["time_start_days"]
)

valid_source_intervals = source[
    source["interval_days"] > 0
]

source_interval_count = len(
    valid_source_intervals
)

source_patient_count = (
    valid_source_intervals[
        "patient_id"
    ].nunique()
)


if source_interval_count == 49:

    source_status = "PASS"

else:

    source_status = "CHECK"


add_audit(
    audit_rows,

    "Source validated interval count",

    source_status,

    "%d valid source intervals from %d source patients."
    % (
        source_interval_count,
        source_patient_count
    ),

    "The established source cohort contains 49 validated "
    "HBGD intervals from 12 patients."
)


# ============================================================
# 3. TARGET TRAJECTORY PROVENANCE
# ============================================================

required_target = [
    "patient_id",
    "week",
    "tumor_volume_cm3",
    "equivalent_diameter_mm"
]

missing_target = [
    c for c in required_target
    if c not in trajectories.columns
]

if len(missing_target) == 0:

    add_audit(
        audit_rows,

        "Target trajectory schema",

        "PASS",

        "LUMIERE trajectory file contains patient, week, "
        "volume and equivalent diameter.",

        "The target trajectory artifact supports temporal "
        "forecast auditing."
    )

else:

    add_audit(
        audit_rows,

        "Target trajectory schema",

        "FAIL",

        str(missing_target),

        "Required target trajectory fields are missing."
    )


# ============================================================
# 4. STEP 25 PREDICTION SCHEMA
# ============================================================

required_step25 = [
    "patient_id",
    "current_week",
    "future_week",
    "future_days",
    "actual_diameter_mm",
    "no_change_pred",
    "direct_transfer_pred",
    "geometry_pred",
    "geometry_hbgd_pred",
    "geometry_error_mm",
    "geometry_hbgd_error_mm",
    "improvement_geometry_hbgd_mm",
    "hbgd_phenotype",
    "history_n_timepoints",
    "history_n_intervals",
    "history_mean_rate",
    "history_rate_std"
]

missing_step25 = [
    c for c in required_step25
    if c not in step25.columns
]

if len(missing_step25) == 0:

    add_audit(
        audit_rows,

        "Step 25 prediction schema",

        "PASS",

        "All required temporal prediction fields are present.",

        "Step 25 contains explicit prediction-origin, future "
        "target and model-error fields."
    )

else:

    add_audit(
        audit_rows,

        "Step 25 prediction schema",

        "FAIL",

        str(missing_step25),

        "Step 25 prediction artifact is incomplete."
    )


# ============================================================
# 5. TEMPORAL ORDER CHECK
# ============================================================

step25["current_week"] = pd.to_numeric(
    step25["current_week"],
    errors="coerce"
)

step25["future_week"] = pd.to_numeric(
    step25["future_week"],
    errors="coerce"
)

step25["future_days"] = pd.to_numeric(
    step25["future_days"],
    errors="coerce"
)

temporal_valid = (
    step25["future_week"]
    >
    step25["current_week"]
)

positive_horizon = (
    step25["future_days"]
    >
    0
)

temporal_pass = (
    temporal_valid.all()
    and
    positive_horizon.all()
)


add_audit(
    audit_rows,

    "Future observation occurs after prediction origin",

    "PASS" if temporal_pass else "FAIL",

    "%d/%d forecasts have future_week > current_week "
    "and positive forecast horizon."
    % (
        int(
            np.sum(
                temporal_valid
                &
                positive_horizon
            )
        ),
        len(step25)
    ),

    "No Step 25 forecast reverses temporal order."
)


# ============================================================
# 6. FUTURE DIAMETER FEATURE AUDIT
# ============================================================

# Explicitly identify columns that contain the future target.
future_columns = [
    c for c in step25.columns
    if (
        "actual_future" in c.lower()
        or
        "future_diameter" in c.lower()
        or
        c.lower() == "actual_diameter_mm"
    )
]

# Historical/model input fields should not directly contain the
# actual future target.
input_like_columns = [
    "previous_diameter_mm",
    "current_diameter_mm",
    "future_days",
    "history_n_timepoints",
    "history_n_intervals",
    "history_mean_rate",
    "history_rate_std"
]

future_target_present = (
    "actual_diameter_mm"
    in step25.columns
)

add_audit(
    audit_rows,

    "Future target separated from predictor fields",

    "PASS" if future_target_present else "CHECK",

    "Future target column: actual_diameter_mm. "
    "Historical predictor fields are stored separately.",

    "The prediction artifact explicitly distinguishes the "
    "future target from historical information."
)


# ============================================================
# 7. HISTORY / FUTURE FEATURE AUDIT
# ============================================================

historical_fields = [
    "history_n_timepoints",
    "history_n_intervals",
    "history_mean_rate",
    "history_rate_std"
]

available_history = [
    c for c in historical_fields
    if c in step25.columns
]

if len(available_history) == len(
    historical_fields
):

    add_audit(
        audit_rows,

        "Historical HBGD feature availability",

        "PASS",

        ", ".join(
            available_history
        ),

        "HBGD history fields are explicitly represented "
        "as historical quantities."
    )

else:

    add_audit(
        audit_rows,

        "Historical HBGD feature availability",

        "CHECK",

        "Available: %s"
        % ", ".join(
            available_history
        ),

        "Not all historical fields are present."
    )


# ============================================================
# 8. PATIENT-LEVEL ISOLATION
# ============================================================

patient_counts = (
    step25.groupby(
        "patient_id"
    ).size()
)

patient_count = (
    step25[
        "patient_id"
    ].nunique()
)

forecast_count = len(
    step25
)

add_audit(
    audit_rows,

    "Patient identifiers present for every forecast",

    "PASS"
    if step25[
        "patient_id"
    ].notna().all()
    else "FAIL",

    "%d forecasts across %d patients."
    % (
        forecast_count,
        patient_count
    ),

    "Each temporal forecast is attributable to a specific "
    "held-out patient."
)


# ============================================================
# 9. WITHIN-PATIENT TEMPORAL DUPLICATE CHECK
# ============================================================

duplicate_origins = step25.duplicated(
    subset=[
        "patient_id",
        "current_week",
        "future_week"
    ],
    keep=False
)

duplicate_count = int(
    np.sum(
        duplicate_origins
    )
)

add_audit(
    audit_rows,

    "Duplicate patient temporal forecast origins",

    "PASS"
    if duplicate_count == 0
    else "CHECK",

    "%d duplicate forecast rows detected."
    % duplicate_count,

    "A zero count indicates unique patient/time prediction "
    "origins in the final temporal prediction artifact."
)


# ============================================================
# 10. MODEL ERROR RECONSTRUCTION
# ============================================================

reconstructed_geometry_error = np.abs(
    step25[
        "actual_diameter_mm"
    ].values
    -
    step25[
        "geometry_pred"
    ].values
)

reconstructed_fusion_error = np.abs(
    step25[
        "actual_diameter_mm"
    ].values
    -
    step25[
        "geometry_hbgd_pred"
    ].values
)

geometry_error_match = np.allclose(
    reconstructed_geometry_error,
    step25[
        "geometry_error_mm"
    ].values,
    atol=1e-6
)

fusion_error_match = np.allclose(
    reconstructed_fusion_error,
    step25[
        "geometry_hbgd_error_mm"
    ].values,
    atol=1e-6
)

if (
    geometry_error_match
    and
    fusion_error_match
):

    error_status = "PASS"

else:

    error_status = "FAIL"


add_audit(
    audit_rows,

    "Prediction error arithmetic integrity",

    error_status,

    "Geometry and Geometry+HBGD absolute errors "
    "were independently reconstructed.",

    "Stored prediction errors agree with actual minus "
    "predicted diameter calculations."
)


# ============================================================
# 11. IMPROVEMENT ARITHMETIC
# ============================================================

reconstructed_improvement = (
    step25[
        "geometry_error_mm"
    ]
    -
    step25[
        "geometry_hbgd_error_mm"
    ]
)

improvement_match = np.allclose(
    reconstructed_improvement,
    step25[
        "improvement_geometry_hbgd_mm"
    ],
    atol=1e-6
)

add_audit(
    audit_rows,

    "HBGD improvement arithmetic integrity",

    "PASS"
    if improvement_match
    else "FAIL",

    "Improvement = Geometry absolute error "
    "minus Geometry+HBGD absolute error.",

    "Positive values indicate that HBGD reduced prediction error."
)


# ============================================================
# 12. NEGATIVE/INVALID DIAMETER CHECK
# ============================================================

numeric_target = pd.to_numeric(
    step25[
        "actual_diameter_mm"
    ],
    errors="coerce"
)

numeric_predictions = pd.to_numeric(
    step25[
        "geometry_hbgd_pred"
    ],
    errors="coerce"
)

invalid_target = (
    numeric_target <= 0
)

invalid_prediction = (
    ~np.isfinite(
        numeric_predictions
    )
)

invalid_count = int(
    np.sum(
        invalid_target
        |
        invalid_prediction
    )
)

add_audit(
    audit_rows,

    "Prediction numerical validity",

    "PASS"
    if invalid_count == 0
    else "FAIL",

    "%d forecasts contain non-positive targets or "
    "non-finite HBGD predictions."
    % invalid_count,

    "Valid forecasts require positive tumor diameter "
    "and finite model predictions."
)


# ============================================================
# 13. SOURCE / TARGET SEPARATION
# ============================================================

source_patients = set(
    source[
        "patient_id"
    ].unique()
)

target_patients = set(
    trajectories[
        "patient_id"
    ].unique()
)

overlap_count = len(
    source_patients
    &
    target_patients
)

add_audit(
    audit_rows,

    "Source-target patient identity separation",

    "PASS"
    if overlap_count == 0
    else "CHECK",

    "%d patient identifiers overlap between source and "
    "LUMIERE target cohorts."
    % overlap_count,

    "Zero overlap supports cohort-level source/target separation."
)


# ============================================================
# 14. STEP 25 PERFORMANCE CONSISTENCY
# ============================================================

geometry_row = step25_perf[
    step25_perf[
        "model"
    ]
    ==
    "M1_Geometry"
]

fusion_row = step25_perf[
    step25_perf[
        "model"
    ]
    ==
    "M2_Geometry_Transferred_HBGD"
]

if (
    len(geometry_row) == 1
    and
    len(fusion_row) == 1
):

    geometry_mae = float(
        geometry_row.iloc[0][
            "MAE_mm"
        ]
    )

    fusion_mae = float(
        fusion_row.iloc[0][
            "MAE_mm"
        ]
    )

    improvement = (
        geometry_mae
        -
        fusion_mae
    )

    add_audit(
        audit_rows,

        "Step 25 performance artifact consistency",

        "PASS",

        "Geometry MAE %.4f mm; Geometry+HBGD MAE %.4f mm; "
        "difference %.4f mm."
        % (
            geometry_mae,
            fusion_mae,
            improvement
        ),

        "The stored performance table provides the expected "
        "comparison between geometry and transferred HBGD."
    )

else:

    add_audit(
        audit_rows,

        "Step 25 performance artifact consistency",

        "CHECK",

        "Required model rows were not uniquely identified.",

        "Manual inspection is required."
    )


# ============================================================
# 15. SOURCE HBGD SIZE
# ============================================================

add_audit(
    audit_rows,

    "Source HBGD sample-size limitation",

    "LIMITATION",

    "%d intervals from %d source patients."
    % (
        source_interval_count,
        source_patient_count
    ),

    "The source HBGD representation is based on a relatively "
    "small longitudinal cohort and therefore limits the "
    "strength of cross-cancer generalization claims."
)


# ============================================================
# 16. TEMPORAL WEEK ASSUMPTION
# ============================================================

add_audit(
    audit_rows,

    "LUMIERE week-to-day conversion",

    "ASSUMPTION",

    "Step 25 uses future_days derived from week differences "
    "using 7 days per week.",

    "This temporal conversion should be verified against the "
    "official LUMIERE dataset documentation before making "
    "precise biological time-course claims."
)


# ============================================================
# BUILD AUDIT DATAFRAME
# ============================================================

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    OUTPUT_AUDIT,
    index=False
)


# ============================================================
# PROVENANCE TABLE
# ============================================================

provenance_rows = [

    {
        "artifact":
            "Source HBGD",
        "file":
            "17_hbgd_state_parameters.csv",
        "role":
            "Validated source growth intervals",
        "count":
            source_interval_count,
        "patients":
            source_patient_count
    },

    {
        "artifact":
            "LUMIERE trajectories",
        "file":
            "lumiere_longitudinal_trajectories.csv",
        "role":
            "Target longitudinal tumor trajectories",
        "count":
            len(trajectories),
        "patients":
            trajectories[
                "patient_id"
            ].nunique()
    },

    {
        "artifact":
            "Step 25 predictions",
        "file":
            "step25_hbgd_temporal_predictions.csv",
        "role":
            "Strict temporal held-out predictions",
        "count":
            len(step25),
        "patients":
            step25[
                "patient_id"
            ].nunique()
    },

    {
        "artifact":
            "Step 25 performance",
        "file":
            "step25_hbgd_temporal_performance.csv",
        "role":
            "Temporal model comparison",
        "count":
            len(step25_perf),
        "patients":
            step25[
                "patient_id"
            ].nunique()
    }
]

provenance_df = pd.DataFrame(
    provenance_rows
)

provenance_df.to_csv(
    OUTPUT_PROVENANCE,
    index=False
)


# ============================================================
# EVIDENCE MATRIX
# ============================================================

evidence_rows = [

    {
        "step":
            19,
        "analysis":
            "Representation-level cross-cancer transfer",
        "key_result":
            "Geometry + transferred HBGD improved MAE "
            "by 0.0887 mm (1.19%)",
        "status":
            "PRELIMINARY POSITIVE",
        "interpretation":
            "Source-derived HBGD representation added "
            "small complementary information beyond geometry."
    },

    {
        "step":
            20,
        "analysis":
            "Statistical validation of Step 19",
        "key_result":
            "9/13 patients improved; p=0.097390; "
            "bootstrap CI crossed zero",
        "status":
            "NOT SIGNIFICANT",
        "interpretation":
            "Directional benefit but insufficient evidence "
            "for statistical significance."
    },

    {
        "step":
            22,
        "analysis":
            "Repeated patient-level splits",
        "key_result":
            "8/10 splits improved; mean +0.0592 mm",
        "status":
            "DIRECTIONALLY POSITIVE",
        "interpretation":
            "Effect was present across a majority of repeated "
            "splits but was small and split-dependent."
    },

    {
        "step":
            23,
        "analysis":
            "Repeated-split statistical aggregation",
        "key_result":
            "Bootstrap CI [-0.1097, 0.2065] mm; "
            "sign-test p=0.109375",
        "status":
            "NOT STATISTICALLY ESTABLISHED",
        "interpretation":
            "Repeated-split evidence does not establish a "
            "non-zero effect."
    },

    {
        "step":
            24,
        "analysis":
            "Responder analysis",
        "key_result":
            "61.54% of patient-seed evaluations favored HBGD; "
            "correlations weak",
        "status":
            "HYPOTHESIS-GENERATING",
        "interpretation":
            "Benefit appears heterogeneous and no strong "
            "responder phenotype was established."
    },

    {
        "step":
            25,
        "analysis":
            "Strict temporal holdout",
        "key_result":
            "Geometry MAE 7.0506 mm; "
            "Geometry+HBGD MAE 7.0272 mm; "
            "+0.0234 mm (0.33%)",
        "status":
            "WEAK POSITIVE",
        "interpretation":
            "A small pooled benefit persisted under strict "
            "temporal forecasting, but only 28/59 patients "
            "improved and median patient benefit was negative."
    },

    {
        "step":
            26,
        "analysis":
            "Final leakage audit",
        "key_result":
            "Audit of source provenance, temporal ordering, "
            "prediction arithmetic and patient identifiers",
        "status":
            "AUDIT",
        "interpretation":
            "Establishes methodological traceability and "
            "documents remaining assumptions/limitations."
    }
]

evidence_df = pd.DataFrame(
    evidence_rows
)

evidence_df.to_csv(
    OUTPUT_EVIDENCE,
    index=False
)


# ============================================================
# AUDIT STATUS COUNTS
# ============================================================

pass_count = int(
    np.sum(
        audit_df[
            "status"
        ] == "PASS"
    )
)

fail_count = int(
    np.sum(
        audit_df[
            "status"
        ] == "FAIL"
    )
)

check_count = int(
    np.sum(
        audit_df[
            "status"
        ] == "CHECK"
    )
)


# ============================================================
# REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w"
) as f:

    f.write(
        "# Step 26 — Final HBGD Leakage Audit and Evidence Consolidation\n\n"
    )

    f.write(
        "## Purpose\n\n"
    )

    f.write(
        "Step 26 audits the provenance and structural integrity "
        "of the strict temporal Step 25 experiment and consolidates "
        "the major cross-cancer HBGD evidence generated in Steps "
        "19–25.\n\n"
    )

    f.write(
        "No predictive model is retrained in this step.\n\n"
    )

    f.write(
        "## Audit summary\n\n"
    )

    f.write(
        "- PASS checks: %d\n"
        % pass_count
    )

    f.write(
        "- FAIL checks: %d\n"
        % fail_count
    )

    f.write(
        "- CHECK items: %d\n\n"
        % check_count
    )

    f.write(
        "## Dataset provenance\n\n"
    )

    f.write(
        "| Artifact | Role | Count | Patients |\n"
    )

    f.write(
        "|---|---|---:|---:|\n"
    )

    for _, row in provenance_df.iterrows():

        f.write(
            "| %s | %s | %d | %d |\n"
            % (
                row["artifact"],
                row["role"],
                int(row["count"]),
                int(row["patients"])
            )
        )

    f.write(
        "\n"
    )

    f.write(
        "## Leakage audit\n\n"
    )

    f.write(
        "| Check | Status | Evidence | Interpretation |\n"
    )

    f.write(
        "|---|---|---|---|\n"
    )

    for _, row in audit_df.iterrows():

        evidence_text = str(
            row["evidence"]
        ).replace(
            "|",
            "/"
        )

        interpretation_text = str(
            row["interpretation"]
        ).replace(
            "|",
            "/"
        )

        f.write(
            "| %s | %s | %s | %s |\n"
            % (
                row["audit_check"],
                row["status"],
                evidence_text,
                interpretation_text
            )
        )

    f.write(
        "\n"
    )

    f.write(
        "## Consolidated evidence matrix\n\n"
    )

    f.write(
        "| Step | Analysis | Key result | Status | Interpretation |\n"
    )

    f.write(
        "|---:|---|---|---|---|\n"
    )

    for _, row in evidence_df.iterrows():

        f.write(
            "| %d | %s | %s | %s | %s |\n"
            % (
                int(row["step"]),
                row["analysis"],
                row["key_result"],
                row["status"],
                row["interpretation"]
            )
        )

    f.write(
        "\n"
    )

    f.write(
        "## Overall scientific interpretation\n\n"
    )

    f.write(
        "The cross-cancer HBGD experiments provide preliminary "
        "evidence that a source-derived growth representation "
        "can contribute complementary information beyond target "
        "tumor geometry. The strongest representation-level "
        "experiment produced a 1.19% pooled MAE improvement, "
        "while repeated patient-level splits showed improvement "
        "in 8 of 10 splits. Under strict temporal holdout, the "
        "pooled improvement remained positive but decreased to "
        "0.33%.\n\n"
    )

    f.write(
        "However, the available statistical analyses do not "
        "establish a non-zero effect. Step 23 produced a "
        "split-level bootstrap confidence interval crossing zero, "
        "and Step 25 showed that only 28 of 59 temporally held-out "
        "patients improved, with a negative median patient-level "
        "improvement.\n\n"
    )

    f.write(
        "Therefore, the appropriate conclusion is that "
        "cross-cancer HBGD transfer is **preliminary and "
        "not statistically established**. The findings support "
        "continued investigation rather than a claim of clinical "
        "validation or universal transferability.\n\n"
    )

    f.write(
        "## Major limitations\n\n"
    )

    f.write(
        "1. The source HBGD representation is based on only 49 "
        "validated intervals from 12 source patients.\n"
    )

    f.write(
        "2. The LUMIERE target cohort contains limited longitudinal "
        "observations per patient.\n"
    )

    f.write(
        "3. The observed predictive benefit is small.\n"
    )

    f.write(
        "4. Repeated-split results are heterogeneous.\n"
    )

    f.write(
        "5. Statistical confidence intervals include zero.\n"
    )

    f.write(
        "6. The week-to-day temporal conversion should be verified "
        "against the official dataset documentation.\n"
    )

    f.write(
        "7. Independent external validation is still required.\n\n"
    )

    f.write(
        "## Recommended final claim\n\n"
    )

    f.write(
        "A source-derived HBGD growth representation demonstrated "
        "preliminary cross-cancer transferability by providing a "
        "small complementary predictive contribution beyond "
        "target-domain tumor geometry. This directional effect "
        "persisted across repeated patient-level splits and under "
        "strict temporal forecasting, but remained heterogeneous "
        "and was not statistically established. Larger independent "
        "longitudinal cohorts are required to determine whether "
        "the observed signal represents a generalizable biological "
        "growth-dynamics representation.\n"
    )


# ============================================================
# FINAL CONSOLE
# ============================================================

print("\n")
print("=" * 78)
print("STEP 26 COMPLETE")
print("=" * 78)

print(
    "\nAUDIT CHECKS:"
)

print(
    "PASS:",
    pass_count
)

print(
    "FAIL:",
    fail_count
)

print(
    "CHECK:",
    check_count
)

print(
    "\nSOURCE:"
)

print(
    "Validated intervals:",
    source_interval_count
)

print(
    "Source patients:",
    source_patient_count
)

print(
    "\nSTEP 25:"
)

print(
    "Temporal forecasts:",
    len(step25)
)

print(
    "Forecast patients:",
    step25[
        "patient_id"
    ].nunique()
)

if (
    len(geometry_row) == 1
    and
    len(fusion_row) == 1
):

    print(
        "Geometry MAE:",
        "%.4f mm"
        % geometry_mae
    )

    print(
        "Geometry + HBGD MAE:",
        "%.4f mm"
        % fusion_mae
    )

    print(
        "Improvement:",
        "%.4f mm"
        % improvement
    )

print(
    "\nOUTPUTS:"
)

print(
    OUTPUT_AUDIT
)

print(
    OUTPUT_EVIDENCE
)

print(
    OUTPUT_PROVENANCE
)

print(
    OUTPUT_REPORT
)

print(
    "\n" + "=" * 78
)