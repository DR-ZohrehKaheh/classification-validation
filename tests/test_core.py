# Project supervisor: Dr. Zohreh Kaheh.
# Undergraduate project collaborator: Ruihan Sun.
# Initial student coding was developed with supervisory teaching and improvement guidance.
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis_pipeline import (
    PlattCalibrator,
    expected_calibration_error,
    metrics,
    optimal_f1_threshold,
)


class CoreAnalysisTests(unittest.TestCase):
    def test_positive_class_metrics(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.9, 0.8, 0.2])
        result = metrics(y, p, 0.5)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tn"], 1)
        self.assertAlmostEqual(result["precision_wildfire"], 0.5)
        self.assertAlmostEqual(result["recall_wildfire"], 0.5)

    def test_ece_is_zero_for_perfect_group_calibration(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(expected_calibration_error(y, p), 0.0)

    def test_optimal_threshold_is_in_probability_range(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.3, 0.7, 0.9])
        threshold, score = optimal_f1_threshold(y, p)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertAlmostEqual(score, 1.0)

    def test_platt_logit_transform_is_finite_at_probability_limits(self):
        probability = np.array([0.0, 0.25, 0.75, 1.0])
        transformed = PlattCalibrator._logit(probability)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertEqual(transformed.shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
