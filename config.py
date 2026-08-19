"""
Central configuration for the Epilepsy Clinical Decision Support System.

All paths are resolved relative to this file so the project runs from any
working directory. Hyperparameters, signal constants, frequency bands and
severity thresholds live here so they can be tuned in one place.
"""
from __future__ import annotations

import os

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
BONN_DIR = os.path.join(DATA_DIR, "bonn")
MODELS_DIR = os.path.join(ROOT_DIR, "models_store")
WEBAPP_DIR = os.path.join(ROOT_DIR, "webapp")
DB_PATH = os.path.join(WEBAPP_DIR, "cdss.db")
UPLOADS_DIR = os.path.join(WEBAPP_DIR, "uploads")
MAX_UPLOAD_BYTES = int(os.environ.get("CDSS_MAX_UPLOAD_BYTES", 100 * 1024 * 1024))
ENVIRONMENT = os.environ.get("CDSS_ENV", "development").lower()
RESEARCH_SANDBOX_ENABLED = (
    os.environ.get("CDSS_RESEARCH_SANDBOX", "0") == "1"
    and ENVIRONMENT != "production"
)

for _d in (DATA_DIR, BONN_DIR, MODELS_DIR, UPLOADS_DIR):
    os.makedirs(_d, exist_ok=True)

# Persisted training artefacts
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
# The best model is persisted already CALIBRATED (see src/models/train.py) so the
# seizure probability shown to clinicians is meaningful.
# A feature-based classical model kept as a surrogate so XAI always works, even
# when the best overall model is a deep network trained on raw windows.
EXPLAINER_MODEL_PATH = os.path.join(MODELS_DIR, "explainer_model.joblib")
BACKGROUND_PATH = os.path.join(MODELS_DIR, "background.npy")  # XAI background sample
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")       # full run metadata

# --------------------------------------------------------------------------- #
# EEG signal constants (Bonn University dataset)
# --------------------------------------------------------------------------- #
# The Bonn dataset was sampled at 173.61 Hz; each segment is 4097 samples
# (~23.6 s) and each of the 5 sets contains 100 single-channel segments.
SAMPLING_RATE = 173.61          # Hz
SEGMENT_LENGTH = 4097           # samples per raw Bonn segment
N_SEGMENTS_PER_SET = 100

# The UCI "Epileptic Seizure Recognition" CSV reshapes each 4097-sample recording
# into 23 consecutive 178-sample chunks (rows). Chunks of one recording share an
# original-recording id, so they must never be split across train/test — see the
# recording-wise grouping in data/load_dataset.py and src/models/train.py.
N_CHUNKS_PER_RECORDING = 23

# Preprocessing
BANDPASS_LOW = 0.5              # Hz
BANDPASS_HIGH = 40.0            # Hz
FILTER_ORDER = 4

# For the deep-learning models we resample each 4097-sample segment down to a
# fixed window to keep the networks small and CPU friendly.
DL_WINDOW = 512

# Frequency bands (Hz) used for band-power features
FREQ_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
}

# --------------------------------------------------------------------------- #
# Bonn dataset set -> clinical meaning
# --------------------------------------------------------------------------- #
# Z (Set A) & O (Set B): healthy volunteers, eyes open / eyes closed
# N (Set C) & F (Set D): interictal (seizure-free interval), epileptogenic zone
# S (Set E):             ictal (active seizure)
# Binary label: seizure (S) = 1, everything else = 0
BONN_SETS = {
    "Z": {"label": 0, "group": "healthy"},
    "O": {"label": 0, "group": "healthy"},
    "N": {"label": 0, "group": "interictal"},
    "F": {"label": 0, "group": "interictal"},
    "S": {"label": 1, "group": "ictal"},
}

# --------------------------------------------------------------------------- #
# EEG evidence thresholds (Level 1 Very low -> Level 5 Very high)
# --------------------------------------------------------------------------- #
# Severity is a derived probability tier (see src/models/severity.py), NOT a
# clinical-severity label present in a public dataset. These cut-points map the
# calibrated seizure probability directly into five visible levels. EEG
# intensity is reported separately and does not change the tier.
SEVERITY_LEVELS = {
    1: "Very low",
    2: "Low",
    3: "Intermediate",
    4: "High",
    5: "Very high",
}
# Upper bounds (inclusive) on seizure probability for levels 1..4;
# anything above the last bound is Level 5.
SEVERITY_THRESHOLDS = [0.20, 0.40, 0.60, 0.80]

# --------------------------------------------------------------------------- #
# Composite (multi-modal) severity — synthesized cases only
# --------------------------------------------------------------------------- #
# The EEG evidence tier above is a single-modality, calibrated-probability tier
# and is never modified. The composite grade below is a SEPARATE, transparent,
# additive rule layer over corroborating modalities, used only for generated
# composite cases where every contributing measurement is available and traced.
# It is a decision-support summary, not a validated clinical severity scale.
#
# Every term is reported with its own weight and rationale so the whole
# derivation is auditable; there is no hidden model here.
COMPOSITE_SEVERITY_WEIGHTS = {
    # EEG evidence tier, contributing (level - 1) * 0.5 so a maximal EEG tier
    # alone reaches 2.0 and cannot by itself produce a maximal composite grade.
    "eeg_evidence": 2.0,
    # Ictal heart-rate rise measured from the participant's own ECG.
    "ictal_heart_rate": 1.0,
    # Recorded seizure frequency during monitoring.
    "seizure_burden": 1.0,
    # Awareness impairment encoded in the annotated event type.
    "impaired_awareness": 0.5,
    # Motor / hyperkinetic semiology encoded in the annotated event type.
    "motor_semiology": 0.5,
    # A structural lesion reported for the matched MRI donor.
    "lesional_mri": 0.5,
}
# Maximum attainable total is the sum of the weights above (5.5).
# Upper bounds (inclusive) for composite levels 1..4; above the last is Level 5.
COMPOSITE_SEVERITY_THRESHOLDS = [0.75, 1.75, 2.75, 3.75]

# Generated composite cases analyse at most this many seconds of EEG. A long
# annotated seizure can exceed 350 analysis windows, and each window costs a
# full feature-extraction plus ensemble inference pass. Truncation is always
# reported in the case provenance — it is never applied silently.
COMPOSITE_ANALYSIS_SECONDS = 30.0

# --------------------------------------------------------------------------- #
# Model hyperparameters
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --------------------------------------------------------------------------- #
# Real-world evaluation & clinical operating point
# --------------------------------------------------------------------------- #
# Seizure detection is evaluated RECORDING-WISE: every chunk of a patient
# recording stays wholly in train or wholly in test (StratifiedGroupKFold /
# GroupShuffleSplit), so reported metrics estimate performance on unseen
# recordings rather than memorised chunks. UCI does not expose a reliable
# patient identifier, so patient-disjoint performance requires another dataset.
#
# Missing a seizure is far worse than a false alarm, so the decision threshold is
# chosen to reach a minimum sensitivity (recall on the seizure class) while
# maximising specificity — NOT the naive 0.5 cut-point. The chosen threshold is
# persisted in metrics.json and honoured at inference time.
SENSITIVITY_TARGET = 0.95        # minimum acceptable seizure recall
DEFAULT_THRESHOLD = 0.5          # fallback if no threshold was persisted

# How the deployed decision threshold is chosen (both are computed on the
# held-out CALIBRATION set, never on test):
#   "balanced"    -> maximise balanced accuracy (best overall accuracy; default)
#   "sensitivity" -> highest threshold still reaching SENSITIVITY_TARGET recall
#                    (clinically safest: minimises missed seizures)
# The non-chosen point is always reported too, so the trade-off stays visible.
OPERATING_POINT = "balanced"

# Probability calibration so the displayed seizure probability is trustworthy.
# "isotonic" (flexible, needs enough data) or "sigmoid" (Platt scaling).
CALIBRATION_METHOD = "sigmoid"

# Class-imbalance handling for models that support it (2,300 seizure vs 9,200
# non-seizure in the UCI binary task; real-world ictal periods are rarer still).
CLASS_WEIGHT = "balanced"

CLASSICAL_PARAMS = {
    "RandomForest": {"n_estimators": 400, "max_depth": None, "min_samples_leaf": 1,
                     "max_features": "sqrt"},
    "SVM": {"C": 4.0, "kernel": "rbf", "gamma": "scale", "probability": True},
    "KNN": {"n_neighbors": 7},
    "GradientBoosting": {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 3},
    # XGBoost (optional; strong CPU learner). tree_method="hist" is the fast path.
    "XGBoost": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 2,
                "reg_lambda": 1.0, "tree_method": "hist", "eval_metric": "logloss"},
}

# Accuracy search space used by the CPU pipeline.  These are intentionally a
# small set of high-value candidates rather than a huge brute-force grid: each
# candidate is evaluated with recording-disjoint cross-validation and the
# extracted feature matrix is reused from disk.
XGBOOST_SEARCH = [
    {"max_depth": 4, "min_child_weight": 1, "subsample": 0.9,
     "colsample_bytree": 0.9, "reg_lambda": 2.0},
    {"max_depth": 6, "min_child_weight": 2, "subsample": 0.9,
     "colsample_bytree": 0.8, "reg_lambda": 1.0},
    {"max_depth": 8, "min_child_weight": 3, "subsample": 0.85,
     "colsample_bytree": 0.85, "reg_lambda": 3.0},
]

# Number of top base models combined into the soft-voting ensemble candidate.
ENSEMBLE_TOP_K = 3

DL_PARAMS = {
    "epochs": 15,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "hidden_size": 64,
    "patience": 4,          # early-stopping patience (on validation loss)
    "val_fraction": 0.2,    # recording-wise validation split held out for stopping
}

# Number of top features surfaced by the XAI module in reports
XAI_TOP_K = 6
