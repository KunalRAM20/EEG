"""
Prediction helper shared by the web app and the demo pipeline.

Loads the persisted best model (+ metadata) once and turns a raw EEG segment
into a full result: seizure probability, binary call, the named feature vector,
and the derived 5-level EEG evidence tier.
"""
from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version

import joblib
import numpy as np

import config
from src.models.calibrated import CalibratedProbModel  # noqa: F401 (unpickling)
from src.models.severity import compute_severity
from src.preprocessing.feature_extraction import (FEATURE_NAMES,
                                                   FEATURE_VERSION,
                                                   extract_features,
                                                   features_to_dict)
from src.preprocessing.signal_processing import to_dl_window


class Predictor:
    """Lazy-loaded singleton-ish wrapper around the trained artefacts."""

    def __init__(self):
        self.best = None
        self.metadata = None
        self.feature_names = FEATURE_NAMES
        self._loaded = False

    def is_trained(self) -> bool:
        return os.path.exists(config.BEST_MODEL_PATH) and \
            os.path.exists(config.METRICS_PATH)

    def load(self):
        if self._loaded:
            return
        if not self.is_trained():
            raise FileNotFoundError(
                "No trained model found. Run:  python -m src.models.train")
        with open(config.METRICS_PATH) as fh:
            self.metadata = json.load(fh)
        if os.path.exists(config.FEATURE_NAMES_PATH):
            with open(config.FEATURE_NAMES_PATH) as fh:
                self.feature_names = json.load(fh)
        schema_hash = hashlib.sha256(
            "\0".join(self.feature_names).encode("utf-8")).hexdigest()
        trained_schema_hash = self.metadata.get("feature_schema_sha256")
        if trained_schema_hash and schema_hash != trained_schema_hash:
            raise RuntimeError(
                "The saved feature order does not match model metadata. "
                "Retrain with: python setup.py")
        expected = self.metadata.get("n_features")
        if expected is not None and int(expected) != len(self.feature_names):
            raise RuntimeError(
                "The saved model metadata and feature-name artifact disagree. "
                "Retrain with: python setup.py")
        trained_feature_version = self.metadata.get("feature_version")
        if (trained_feature_version is not None and
                int(trained_feature_version) != FEATURE_VERSION):
            raise RuntimeError(
                f"Model uses feature version {trained_feature_version}, but "
                f"runtime uses version {FEATURE_VERSION}. Retrain with: "
                "python setup.py")
        trained_sklearn = self.metadata.get("runtime", {}).get("scikit-learn")
        try:
            runtime_sklearn = version("scikit-learn")
        except PackageNotFoundError:
            runtime_sklearn = None
        if trained_sklearn and runtime_sklearn != trained_sklearn:
            raise RuntimeError(
                f"Model was trained with scikit-learn {trained_sklearn}, but "
                f"runtime has {runtime_sklearn}. Retrain with: python setup.py")
        self.best = joblib.load(config.BEST_MODEL_PATH)
        expected_run = self.metadata.get("artifact_run_id")
        if (expected_run and
                getattr(self.best, "artifact_run_id", None) != expected_run):
            raise RuntimeError(
                "Model artifacts come from different training runs. Retrain "
                "with: python setup.py")
        if self.metadata.get("best_input") == "features":
            for member in getattr(self.best, "members", []):
                model_width = getattr(member, "n_features_in_", None)
                if model_width is not None and int(model_width) != len(self.feature_names):
                    raise RuntimeError(
                        "The saved estimator and feature schema disagree. "
                        "Retrain with: python setup.py")
        self._loaded = True

    def expected_segment_length(self) -> int:
        """Return the exact raw EEG length used to train the persisted model."""
        self.load()
        stored = self.metadata.get("segment_length")
        if stored is not None:
            return int(stored)
        return 178 if self.metadata.get("data_source") == "real-uci" \
            else config.SEGMENT_LENGTH

    def predict_segment(self, segment: np.ndarray) -> dict:
        """Full prediction for one raw EEG segment."""
        self.load()
        segment = np.asarray(segment, dtype=np.float64).squeeze()
        expected_length = self.expected_segment_length()
        if segment.ndim != 1 or len(segment) != expected_length:
            raise ValueError(
                f"This model requires exactly {expected_length} EEG values; "
                f"received shape {segment.shape}.")
        if not np.all(np.isfinite(segment)):
            raise ValueError("EEG signal contains non-finite values.")
        runtime_features = extract_features(segment)
        feat_dict = features_to_dict(runtime_features)

        # Consume the training-time named schema in its persisted order. This
        # safely supports older 32-feature artifacts while the runtime also
        # computes the newer DWT features.
        missing = [name for name in self.feature_names if name not in feat_dict]
        if missing:
            raise RuntimeError(
                "The trained model requires unavailable EEG features: "
                + ", ".join(missing) + ". Retrain with: python setup.py")
        features = np.asarray([feat_dict[name] for name in self.feature_names],
                              dtype=np.float64)

        if self.metadata.get("best_input") == "window":
            X = to_dl_window(segment).reshape(1, -1)
        else:
            X = features.reshape(1, -1)

        prob = float(self.best.predict_proba(X)[0, 1])
        # Clinically chosen operating point (persisted at training time), not 0.5.
        threshold = float(self.metadata.get("decision_threshold",
                                            config.DEFAULT_THRESHOLD))
        label = int(prob >= threshold)
        severity = compute_severity(prob, feat_dict)

        return {
            # Keep the exact calibrated value in storage; templates/reporting
            # are responsible for presentation rounding.
            "seizure_probability": prob,
            "seizure_prediction": label,
            "prediction_label": "Seizure" if label else "No seizure",
            "decision_threshold": threshold,
            "operating_point": self.metadata.get("operating_point", "balanced"),
            "severity": severity,
            "features": feat_dict,
            "feature_vector": features,
            "model_name": self.metadata.get("best_model"),
            "data_source": self.metadata.get("data_source"),
        }


# Module-level shared instance
PREDICTOR = Predictor()


def predict_segment(segment: np.ndarray) -> dict:
    return PREDICTOR.predict_segment(segment)
