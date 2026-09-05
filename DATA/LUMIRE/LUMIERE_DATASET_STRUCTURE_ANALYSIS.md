# LUMIERE DATASET STRUCTURE ANALYSIS

Metadata-only structural analysis of the LUMIERE ZIP archive.

## 1. Dataset Information

| Property | Value |
|---|---:|
| ZIP path | `D:\VIT\PROJECT\NLP\PATENT\DATA\LUMIRE.zip` |
| ZIP size | 30.33 GB |
| Total ZIP entries | 31,370 |
| Total files | 24,051 |
| Total directories | 7,319 |

## 2. Patient Summary

| Metric | Value |
|---|---:|
| Total patients | 83 |
| First patient ID | Patient-001 |
| Last patient ID | Patient-091 |
| Minimum timepoints/patient | 1 |
| Maximum timepoints/patient | 18 |
| Mean timepoints/patient | 6.52 |
| Median timepoints/patient | 5.0 |

## 3. Timepoint Distribution

| Category | Patients | Percentage |
|---|---:|---:|
| Exactly 1 | 7 | 8.43% |
| Exactly 2 | 7 | 8.43% |
| Exactly 3 | 9 | 10.84% |
| Exactly 4 | 5 | 6.02% |
| 5 or more | 55 | 66.27% |
| At least 2 | 76 | 91.57% |
| At least 3 | 69 | 83.13% |
| At least 4 | 60 | 72.29% |
| At least 5 | 55 | 66.27% |

## 4. Imaging Modality Availability

| Modality | Patient-timepoints | % of timepoints | Patients | Patients with modality at every week |
|---|---:|---:|---:|---:|
| FLAIR.nii.gz | 524 | 96.86% | 83 | 69 |
| CT1.nii.gz | 535 | 98.89% | 83 | 77 |
| T1.nii.gz | 527 | 97.41% | 83 | 70 |
| T2.nii.gz | 534 | 98.71% | 83 | 76 |

**Interpretation:** modality availability is based only on file existence in the ZIP.

## 5. Segmentation Availability

| Segmentation category | Patient-timepoints | Percentage |
|---|---:|---:|
| HD-GLIO-AUTO | 512 | 94.64% |
| DeepBraTumIA | 512 | 94.64% |
| Any segmentation | 512 | 94.64% |

### Individual HD-GLIO-AUTO files

- `segmentation_T1_origspace.nii.gz`: 512 patient-timepoints
- `segmentation_CT1_origspace.nii.gz`: 512 patient-timepoints
- `segmentation_FLAIR_origspace.nii.gz`: 512 patient-timepoints
- `segmentation_T2_origspace.nii.gz`: 512 patient-timepoints
- `segmentation.nii.gz`: 512 patient-timepoints

### Individual DeepBraTumIA files

- `t2_seg_mask.nii.gz`: 512 patient-timepoints
- `ct1_seg_mask.nii.gz`: 512 patient-timepoints
- `flair_seg_mask.nii.gz`: 512 patient-timepoints
- `t1_seg_mask.nii.gz`: 512 patient-timepoints

## 6. Structural Timepoint Usability

| Category | Count | Percentage |
|---|---:|---:|
| Total patient-timepoints | 541 | 100% |
| Structurally usable timepoints | 512 | 94.64% |
| Images but no segmentation | 29 | 5.36% |
| Segmentation but no images | 0 | 0.00% |

**Important:** A structurally usable timepoint means that at least one image file and at least one segmentation file are present. NIfTI voxel content, tumor volume, mask quality, shape, affine, spacing, and biological validity have NOT been checked.

## 7. Multimodal Completeness

| Category | Patient-timepoints | Percentage |
|---|---:|---:|
| At least 1 modality | 541 | 100.00% |
| At least 2 modalities | 538 | 99.45% |
| At least 3 modalities | 529 | 97.78% |
| All 4 modalities | 512 | 94.64% |

## 8. Candidate Longitudinal Trajectories

| Criterion | Patients | Percentage |
|---|---:|---:|
| At least 2 structurally usable timepoints | 74 | 89.16% |
| At least 3 structurally usable timepoints | 67 | 80.72% |
| At least 4 structurally usable timepoints | 60 | 72.29% |
| At least 5 structurally usable timepoints | 50 | 60.24% |

## 9. Longitudinal Interval Analysis

| Metric | Value |
|---|---:|
| Total candidate longitudinal intervals | 429 |
| Minimum interval | 1 weeks |
| Maximum interval | 155 weeks |
| Mean interval | 12.15 weeks |
| Median interval | 12.0 weeks |

## 10. Metadata File Availability

| File type | Patient-timepoints containing file |
|---|---:|
| `.mat` | 512 |
| `.tfm` | 512 |
| `.json` | 512 |

## 11. File Extension Summary

| Extension | Count |
|---|---:|
| `.nii.gz` | 18,660 |
| `.mat` | 2,396 |
| `.tfm` | 2,396 |
| `.json` | 599 |

## 12. Directory Structure

### Top-level directories

- `Imaging`: 24,051 file entries

Expected structure:
```text
Imaging/
    Patient-XXX/
        week-XXX/
            imaging / segmentation / metadata
```

## 13. Missing Data Analysis

### Patient-timepoints with images but no segmentation

- Patient-009 / week-026
- Patient-019 / week-000
- Patient-023 / week-070
- Patient-023 / week-116
- Patient-024 / week-000
- Patient-026 / week-000
- Patient-028 / week-024
- Patient-028 / week-046
- Patient-028 / week-048
- Patient-029 / week-240
- Patient-033 / week-011
- Patient-033 / week-088
- Patient-037 / week-049
- Patient-041 / week-051
- Patient-049 / week-003
- Patient-054 / week-000
- Patient-058 / week-000
- Patient-059 / week-148
- Patient-063 / week-000
- Patient-063 / week-178
- Patient-066 / week-128
- Patient-070 / week-000
- Patient-071 / week-037
- Patient-076 / week-000
- Patient-079 / week-000
- Patient-083 / week-022
- Patient-089 / week-013
- Patient-089 / week-026
- Patient-089 / week-039

### Patient-timepoints with segmentation but no image

None detected.

## 14. Scientific Interpretation

The ZIP inspection confirms the presence of longitudinal patient/week organization and multiple imaging and segmentation resources.

However, this analysis is metadata-only. It does not validate image contents, segmentation quality, tumor voxel presence, image orientation, voxel spacing, affine consistency, registration accuracy, or biological longitudinal consistency.

Therefore, the patient counts reported as structurally usable or candidate longitudinal trajectories must be treated as preliminary until the corresponding NIfTI files are selectively extracted and validated.

For future HBGD modelling, week numbers should not automatically be converted to biological elapsed time unless the LUMIERE dataset documentation confirms that the week identifier represents a reliable temporal scale.

## 15. Recommended Next Step

The next analysis should selectively extract only the structurally promising longitudinal patient-timepoints and validate the NIfTI images and tumor masks by checking voxel dimensions, affine/orientation, voxel spacing, non-zero mask voxels, tumor volume, and longitudinal consistency.

After this validation, the dataset can be used to calculate tumor geometry and HBGD growth parameters for the cross-cancer transfer experiment.