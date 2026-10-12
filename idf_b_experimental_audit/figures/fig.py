"""
fig4_only.py - standalone script for Figure 4 (MAE improvement across protocols).
Reads values from the verified reconciliation CSV; no other script needed.
"""
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = r"C:\HBGD\idf_b_experimental_audit\figures"
RECON_CSV = r"C:\HBGD\idf_b_experimental_audit\HBGD_METRIC_RECONCILIATION.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- load VERIFIED rows only ----
verified_rows = []
with open(RECON_CSV, mode="r", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("Verification_Status", "").strip() == "VERIFIED":
            verified_rows.append(row)
print(f"Loaded {len(verified_rows)} verified rows.")


def get_metric(exp_id, model_name, metric_name, agg_level=None):
    for r in verified_rows:
        if (r["Experiment_ID"] == exp_id and r["Model_Name"] == model_name
                and r["Metric_Name"] == metric_name):
            if agg_level is None or r["Aggregation_Level"] == agg_level:
                return float(r["Metric_Value"])
    return None


def improvement(exp_id, base_model, hbgd_model, agg, expected):
    base = get_metric(exp_id, base_model, "MAE", agg)
    hbgd = get_metric(exp_id, hbgd_model, "MAE", agg)
    assert base is not None and hbgd is not None, f"Missing values for {exp_id}/{hbgd_model}"
    imp = base - hbgd
    assert abs(imp - expected) < 1e-4, f"{exp_id}/{hbgd_model}: expected {expected:+.4f}, got {imp:+.4f}"
    print(f"OK {exp_id} {hbgd_model}: {imp:+.4f} mm")
    return imp


imp_cv = improvement("EXP-LUM-09", "Target_Morphology_History_Baseline", "Domain_Adapted_HBGD", "Fold-Level Mean", 0.1680)
imp_rep = improvement("EXP-LUM-05", "Geometry", "Geometry_Plus_Transferred_HBGD", "Pooled Case-Level", 0.0887)
imp_temp = improvement("EXP-LUM-08", "M1_Geometry", "M2_Geometry_Transferred_HBGD", "Pooled Case-Level", 0.0234)
imp_res = improvement("EXP-LUM-03", "Target_Morphology_History_Baseline", "Domain_Adapted_HBGD", "Pooled Case-Level", -0.0362)
imp_comb = improvement("EXP-LUM-09", "Target_Morphology_History_Baseline", "Morphology_Plus_Adapted_HBGD", "Fold-Level Mean", -0.1159)

fig4_data = [
    {"label": "25-Fold Patient-Grouped CV (Step 27)\n[Baseline: Target Morphology, n = 54 pts, 1,605 rows]", "val": imp_cv},
    {"label": "Representation Transfer (Step 19)\n[Baseline: Geometry, n = 13 pts, 75 cases]", "val": imp_rep},
    {"label": "Strict Temporal Holdout (Step 25)\n[Baseline: Geometry, n = 59 pts, 276 cases]", "val": imp_temp},
    {"label": "Held-Out Residual Correction (Step 11)\n[Baseline: Target Morphology, n = 13 pts, 88 cases]", "val": imp_res},
    {"label": "Morphology + Adapted HBGD (Step 27)\n[Baseline: Target Morphology, n = 54 pts, 1,605 rows]", "val": imp_comb},
]

# ---- plot ----
GREEN, RED = "#2E7D32", "#C62828"

fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
y_pos = np.arange(len(fig4_data))
y_vals = [d["val"] for d in fig4_data]
y_labels = [d["label"] for d in fig4_data]

bars = ax.barh(
    y_pos, y_vals, height=0.55,
    color=[GREEN if v >= 0 else RED for v in y_vals],
    edgecolor="black", linewidth=1.2,
)  # solid bars, no hatch

ax.axvline(0, color="black", linewidth=1.2)

for bar, v in zip(bars, y_vals):
    yc = bar.get_y() + bar.get_height() / 2
    if v >= 0:
        ax.annotate(f"+{v:.4f} mm", xy=(v, yc), xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, fontweight="bold")
    else:
        ax.annotate(f"-{abs(v):.4f} mm", xy=(v, yc), xytext=(-6, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=10, fontweight="bold")

ax.set_xlim(-0.22, 0.26)
ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=9.5)
ax.invert_yaxis()
ax.tick_params(axis="y", pad=8)
ax.set_xlabel("Improvement in MAE (mm)  [MAE_baseline - MAE_HBGD]", fontsize=11, fontweight="bold")
ax.set_title(
    "Forecasting Improvement Across Independent Protocols\n"
    "Each Experiment Evaluated Against Its Own Defined Baseline",
    fontsize=12, fontweight="bold", pad=12,
)
ax.grid(axis="x", linestyle=":", alpha=0.6)
ax.set_axisbelow(True)

fig.text(
    0.5, -0.01,
    "Values > 0 indicate error reduction. "
    "Observed improvements were not statistically significant at alpha = 0.05.",
    ha="center", fontsize=8.5, style="italic",
)

plt.tight_layout()
fig4_path = os.path.join(OUTPUT_DIR, "fig4_improvement.png")
fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Created {fig4_path}")