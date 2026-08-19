"""
Deployable probability model: an optional soft-vote ensemble wrapped in a
leakage-free probability calibrator.

One small class, ``CalibratedProbModel``, is what actually gets persisted and
loaded by the web app / inference layer. It:

  * averages the positive-class probability of one or more fitted base models
    (a single model is just a list of length 1), and
  * maps that raw probability through a calibrator (isotonic or sigmoid/Platt)
    fitted on a held-out CALIBRATION set, so the seizure probability shown to a
    clinician is trustworthy.

Keeping calibration here (rather than sklearn's CalibratedClassifierCV) lets the
exact same code path calibrate a single classical pipeline, a deep network, or an
ensemble — and it is trivially picklable (it only holds fitted estimators + a
fitted 1-D calibrator).
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _mean_pos_proba(members, X) -> np.ndarray:
    """Mean positive-class probability across base models."""
    return np.mean([m.predict_proba(X)[:, 1] for m in members], axis=0)


class _SigmoidCalibrator:
    """Platt scaling: 1-D logistic map from raw probability to calibrated one."""

    def __init__(self):
        self._lr = LogisticRegression()

    def fit(self, raw, y):
        self._lr.fit(np.asarray(raw).reshape(-1, 1), np.asarray(y))
        return self

    def transform(self, raw):
        return self._lr.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]


class _IsotonicCalibrator:
    """Monotonic, non-parametric calibration (flexible; needs enough data)."""

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, raw, y):
        self._iso.fit(np.asarray(raw), np.asarray(y))
        return self

    def transform(self, raw):
        return np.clip(self._iso.predict(np.asarray(raw)), 0.0, 1.0)


class CalibratedProbModel:
    """Persisted model: (soft-vote) base members + a fitted probability calibrator."""

    def __init__(self, members, calibrator, input_kind: str = "features"):
        self.members = list(members)
        self.calibrator = calibrator
        self.input_kind = input_kind          # "features" or "window"
        self.classes_ = np.array([0, 1])

    def raw_proba(self, X) -> np.ndarray:
        return _mean_pos_proba(self.members, X)

    def predict_proba(self, X) -> np.ndarray:
        p1 = self.calibrator.transform(self.raw_proba(X)) \
            if self.calibrator is not None else self.raw_proba(X)
        p1 = np.clip(np.asarray(p1, dtype=float), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X) -> np.ndarray:
        threshold = float(getattr(self, "decision_threshold", 0.5))
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def fit_calibrated(members, X_cal, y_cal, method: str = "isotonic",
                   input_kind: str = "features") -> CalibratedProbModel:
    """Fit a calibrator for ``members`` on the held-out calibration set."""
    raw = _mean_pos_proba(members, X_cal)
    calibrator = (_SigmoidCalibrator() if method == "sigmoid"
                  else _IsotonicCalibrator())
    calibrator.fit(raw, y_cal)
    return CalibratedProbModel(members, calibrator, input_kind)
