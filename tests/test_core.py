import unittest

import numpy as np
import pandas as pd

from data.load_dataset import _synth_segment, _uci_recording_groups
from src.models.calibrated import CalibratedProbModel
from src.models.severity import compute_severity
from src.preprocessing.feature_extraction import (
    DWT_FEATURES,
    FEATURE_NAMES,
    extract_features,
)
from src.preprocessing.features_cache import _segments_fingerprint


class _Member:
    def predict_proba(self, X):
        p1 = np.asarray(X, dtype=float).reshape(-1)
        return np.column_stack([1.0 - p1, p1])


class CoreTests(unittest.TestCase):
    def test_severity_probability_boundaries(self):
        probabilities = [0, .2, .2001, .4, .4001, .6, .6001, .8, .8001, 1]
        levels = [compute_severity(value, {})["level"] for value in probabilities]
        self.assertEqual(levels, [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_severity_preserves_precision_used_for_tier(self):
        for probability, level in ((0.20001, 2), (0.80001, 5)):
            with self.subTest(probability=probability):
                result = compute_severity(probability, {})
                self.assertEqual(result["level"], level)
                self.assertEqual(result["score"], probability)
                self.assertEqual(result["seizure_probability"], probability)

    def test_short_window_dwt_does_not_mislabel_a4_as_a5(self):
        signal = np.sin(np.linspace(0, 20, 178))
        vector = extract_features(signal)
        dwt_start = len(FEATURE_NAMES) - len(DWT_FEATURES)
        self.assertEqual(vector.shape, (len(FEATURE_NAMES),))
        self.assertEqual(vector[dwt_start], 0.0)      # a5 unavailable
        self.assertEqual(vector[dwt_start + 1], 0.0)  # d5 unavailable
        self.assertTrue(np.isfinite(vector).all())

    def test_invalid_signal_is_rejected(self):
        for signal in ([], [1.0], [1.0, np.nan], [[1, 2], [3, 4]]):
            with self.subTest(signal=signal):
                with self.assertRaises(ValueError):
                    extract_features(signal)

    def test_cache_fingerprint_changes_with_content(self):
        first = np.arange(12.0).reshape(3, 4)
        second = first.copy()
        second[0, 0] = 99.0
        self.assertNotEqual(
            _segments_fingerprint(first), _segments_fingerprint(second))

    def test_degenerate_uci_groups_are_rejected(self):
        frame = pd.DataFrame({"id": ["unique-a", "unique-b"], "y": [1, 2]})
        with self.assertRaisesRegex(ValueError, "recording groups"):
            _uci_recording_groups(frame)

    def test_synthetic_fallback_is_reproducible_in_process(self):
        np.testing.assert_array_equal(
            _synth_segment("Z", 7), _synth_segment("Z", 7))

    def test_persisted_model_predict_uses_deployed_threshold(self):
        model = CalibratedProbModel([_Member()], calibrator=None)
        model.decision_threshold = 0.3
        np.testing.assert_array_equal(model.predict([[.2], [.3], [.8]]), [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
