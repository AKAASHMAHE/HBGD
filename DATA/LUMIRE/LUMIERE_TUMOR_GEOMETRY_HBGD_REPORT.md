# LUMIERE TUMOR GEOMETRY + HBGD FEATURE EXTRACTION

This stage converts validated LUMIERE tumor segmentation masks into tumor geometry and longitudinal HBGD growth features.

## 1. Input

| Metric | Value |
|---|---:|
| Step 2 VALID timepoints | 476 |
| Geometry extraction successes | 476 |
| Geometry extraction failures | 0 |

## 2. Tumor Geometry

| Metric | Value |
|---|---:|
| Valid tumor volumes | 476 |
| Minimum volume | 0.026 cm3 |
| Maximum volume | 293.871 cm3 |
| Mean volume | 44.704 cm3 |
| Minimum equivalent diameter | 3.653 mm |
| Maximum equivalent diameter | 82.487 mm |
| Mean equivalent diameter | 39.656 mm |

## 3. Longitudinal HBGD Features

| Metric | Value |
|---|---:|
| Patients with >=2 successful timepoints | 67 |
| Longitudinal intervals | 409 |

## 4. HBGD Growth-State Distribution

| State | Intervals | Percentage |
|---|---:|---:|
| GROWTH | 189 | 46.21% |
| STABLE | 103 | 25.18% |
| SHRINKAGE | 117 | 28.61% |
| UNKNOWN | 0 | 0.00% |

## 5. HBGD Mathematical Formulation

For equivalent tumor diameter D:

```text
r_D = ln(D2 / D1) / Δt
```

where Δt is represented in days using the LUMIERE week identifier multiplied by 7.

Tumor volume is calculated directly from segmentation voxel count:

```text
V = N_voxels × voxel_volume
```

Equivalent spherical diameter is calculated using:

```text
D = (6V / π)^(1/3)
```

Because V is proportional to D³ under the equivalent-sphere representation, the theoretical relationship is approximately r_V = 3r_D.

## 6. Important Temporal Assumption

This analysis currently uses the difference between LUMIERE week identifiers as the longitudinal time interval and converts weeks to days using 7 days per week.

This should be confirmed against the official LUMIERE dataset documentation before the resulting growth rates are used as final biological estimates.

## 7. Next Step

The resulting longitudinal HBGD features can now be used to evaluate within-LUMIERE forward prediction before performing the cross-cancer transfer experiment.

The next stage should perform patient-level train/validation/test splitting and predict future tumor diameter or volume from previous longitudinal observations.