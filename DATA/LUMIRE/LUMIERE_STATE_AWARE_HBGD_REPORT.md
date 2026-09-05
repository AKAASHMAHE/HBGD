# LUMIERE STATE-AWARE HBGD ANALYSIS

## 1. Objective

This experiment evaluates whether the current HBGD dynamic state contains useful information about subsequent tumor behavior.

The analysis follows the same patient-level test split used in Steps 4 and 5.

## 2. Dataset

| Metric | Value |
|---|---:|
| Total patients | 67 |
| Training patients | 54 |
| Test patients | 13 |
| Test prediction cases | 75 |

## 3. State-Specific Training Rates

| State | Training intervals | Estimated r/day |
|---|---:|---:|
| GROWTH | 151 | 0.00323376 |
| STABLE | 77 | 0.00001868 |
| SHRINKAGE | 93 | -0.00589190 |

## 4. Overall Prediction

| Model | MAE (mm) | RMSE (mm) |
|---|---:|---:|
| No-change | 8.9621 | 13.3902 |
| State-aware HBGD | 14.8608 | 21.9946 |

State-aware HBGD improvement versus no-change: -65.82%.

## 5. State-Specific Future Behavior

| Current state | N | State-aware MAE | No-change MAE | Mean future diameter change | State persistence |
|---|---:|---:|---:|---:|---:|
| GROWTH | 33 | 15.4195 | 7.2517 | 0.2952 mm | 39.39% |
| STABLE | 21 | 5.4265 | 5.4272 | 3.2971 mm | 47.62% |
| SHRINKAGE | 21 | 23.4170 | 15.1848 | 3.0372 mm | 28.57% |

## 6. HBGD State Transition Matrix

| Current → Future | Growth | Stable | Shrinkage |
|---|---:|---:|---:|
| GROWTH | 39.39% | 33.33% | 27.27% |
| STABLE | 47.62% | 47.62% | 4.76% |
| SHRINKAGE | 52.38% | 19.05% | 28.57% |

### Transition counts

| Current → Future | Growth | Stable | Shrinkage |
|---|---:|---:|---:|
| GROWTH | 13 | 11 | 9 |
| STABLE | 10 | 10 | 1 |
| SHRINKAGE | 11 | 4 | 6 |

## 7. Interpretation

**The state-aware model did not outperform the no-change baseline overall.** Therefore, HBGD state alone does not provide sufficient predictive accuracy for this test set.

For **GROWTH**, state-aware prediction did not outperform no-change.
For **STABLE**, state-aware prediction outperformed no-change.
For **SHRINKAGE**, state-aware prediction did not outperform no-change.

## 8. Scientific Interpretation

The purpose of this analysis is diagnostic as well as predictive. If HBGD states show meaningful persistence or distinct future behavior even when the direct exponential forecast is inaccurate, the states may remain useful as features in a larger predictive-intelligence model.

Conversely, if states show weak persistence and poor state-aware prediction, HBGD state alone should not be presented as a reliable GBM forecasting mechanism.

## 9. Temporal Limitation

Time intervals are currently derived from LUMIERE week identifiers and converted using 7 days per week. This temporal interpretation must be confirmed against official dataset documentation.

## 10. Next Decision

The next stage should combine HBGD-derived features with predictive machine-learning models rather than relying on exponential HBGD alone. Candidate features include current diameter, volume, recent growth rate, growth state, rate variability, interval duration, and longitudinal trend.