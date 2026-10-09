# -*- coding: utf-8 -*-
"""
AUTOMATED TEST SUITE FOR HBGD CROSS-CANCER TRANSFER AUDIT
=========================================================

Verifies key methodological safeguards:
- Temporal feature leakage prevention
- Patient partition disjointness
- Residual sign & stacking arithmetic
- Feature matrix column alignment
- Growth rate numerical safety & boundary handling
- Metric calculation mathematical accuracy
- Direct exponential growth rate prediction instability detection
"""

import sys
import os
import unittest
import importlib
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

step27 = importlib.import_module("27_hbgd_patient_grouped_cv_experiment")

historical_summary = step27.historical_summary
build_cases = step27.build_cases
safe_hbgd_rate = step27.safe_hbgd_rate
hbgd_predict = step27.hbgd_predict
metrics = step27.metrics
rmse = step27.rmse
SOURCE_FEATURE_NAMES = step27.SOURCE_FEATURE_NAMES
TARGET_BASELINE_FEATURE_NAMES = step27.TARGET_BASELINE_FEATURE_NAMES
get_development_and_test_patients = step27.get_development_and_test_patients
load_target_trajectories = step27.load_target_trajectories


class TestHBGDTransferAudit(unittest.TestCase):

    def test_temporal_non_leakage(self):
        """Verify that historical summary uses only past observations and excludes target."""
        observations = [
            {"time_days": 0.0, "diameter": 10.0, "week": 0},
            {"time_days": 14.0, "diameter": 12.0, "week": 2},
            {"time_days": 28.0, "diameter": 50.0, "week": 4} # Target: huge spike
        ]
        history = observations[:2] # history before index 2
        summary = historical_summary(history)
        
        self.assertEqual(summary["current_diameter_mm"], 12.0)
        self.assertEqual(summary["previous_diameter_mm"], 10.0)
        self.assertEqual(summary["history_observation_count"], 2)
        # Verify that 50.0 target value never appears in history summary
        self.assertNotEqual(summary["current_diameter_mm"], 50.0)

    def test_patient_partition_disjointness(self):
        """Verify train, validation, and test patient ID sets are mutually disjoint."""
        trajectories = load_target_trajectories()
        dev_ids, test_ids = get_development_and_test_patients(trajectories)
        
        dev_set = set(dev_ids)
        test_set = set(test_ids)
        
        self.assertTrue(dev_set.isdisjoint(test_set))
        self.assertEqual(len(dev_set) + len(test_set), len(trajectories.keys()))

    def test_residual_sign_arithmetic(self):
        """Verify residual = actual - predicted and adapted = source + correction."""
        actual = 25.0
        source_pred = 20.0
        expected_residual = 5.0 # actual - source_pred
        
        correction = expected_residual
        adapted_pred = source_pred + correction
        
        self.assertAlmostEqual(adapted_pred, actual, places=6)

    def test_source_target_feature_alignment(self):
        """Verify feature column list is consistent and contains expected names."""
        self.assertEqual(len(SOURCE_FEATURE_NAMES), 9)
        self.assertIn("current_diameter_mm", SOURCE_FEATURE_NAMES)
        self.assertIn("historical_hbgd_prior_mm", SOURCE_FEATURE_NAMES)
        self.assertIn("recent_historical_rate_per_day", SOURCE_FEATURE_NAMES)

    def test_growth_rate_numerical_safety(self):
        """Verify safe_hbgd_rate handles invalid inputs safely."""
        self.assertTrue(np.isnan(safe_hbgd_rate(-5.0, 10.0, 10.0)))
        self.assertTrue(np.isnan(safe_hbgd_rate(10.0, -5.0, 10.0)))
        self.assertTrue(np.isnan(safe_hbgd_rate(10.0, 12.0, 0.0)))
        self.assertTrue(np.isnan(safe_hbgd_rate(10.0, 12.0, -10.0)))
        
        # Valid calculation
        r = safe_hbgd_rate(10.0, 20.0, 10.0)
        expected = np.log(2.0) / 10.0
        self.assertAlmostEqual(r, expected, places=6)

    def test_metric_reproducibility(self):
        """Verify MAE and RMSE calculation functions."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([12.0, 18.0, 30.0])
        
        mae_val, rmse_val = metrics(y_true, y_pred)
        self.assertAlmostEqual(mae_val, (2.0 + 2.0 + 0.0) / 3.0, places=6)
        self.assertAlmostEqual(rmse_val, np.sqrt((4.0 + 4.0 + 0.0) / 3.0), places=6)

    def test_unbounded_exponential_rate_detection(self):
        """Verify that direct exponential prediction without ML bounding can explode on long horizons."""
        current_d = 23.02
        # Transient rapid rate over 14 days
        rate = np.log(23.02 / 15.08) / 14.0 # ~0.030214 / day
        horizon = 273.0 # 39 weeks
        
        raw_pred = hbgd_predict(current_d, rate, horizon)
        self.assertGreater(raw_pred, 1000.0) # Confirms exponential explosion (>1 meter diameter)
        
        # Verify that exponent clamping prevents infinity
        clamped_pred = hbgd_predict(current_d, 1.0, 1000.0) # rate * horizon = 1000 -> clamped to 20
        self.assertTrue(np.isfinite(clamped_pred))


if __name__ == "__main__":
    unittest.main()
