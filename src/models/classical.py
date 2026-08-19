"""
Classical machine-learning models operating on the extracted feature vectors.

Each builder returns an unfitted scikit-learn ``Pipeline`` that standardizes the
features and then applies the estimator, matching the PDR's list of Random
Forest, SVM, KNN and Gradient Boosting.

Random Forest and SVM use ``class_weight`` (config.CLASS_WEIGHT) to counter the
seizure/non-seizure imbalance. KNN and Gradient Boosting have no such parameter;
they lean on the calibrated decision threshold chosen in ``src/models/train.py``.
Probability calibration is applied to the *selected* model in the training script
(leakage-free, grouped CV), not baked in here.
"""
from __future__ import annotations

from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:  # pragma: no cover - XGBoost is optional
    XGB_AVAILABLE = False

P = config.CLASSICAL_PARAMS


def _pipe(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def build_random_forest() -> Pipeline:
    # class_weight balances the rarer seizure class (see config.CLASS_WEIGHT).
    return _pipe(RandomForestClassifier(
        random_state=config.RANDOM_STATE, n_jobs=1,
        class_weight=config.CLASS_WEIGHT, **P["RandomForest"]))


def build_svm() -> Pipeline:
    return _pipe(SVC(random_state=config.RANDOM_STATE,
                     class_weight=config.CLASS_WEIGHT, **P["SVM"]))


def build_knn() -> Pipeline:
    return _pipe(KNeighborsClassifier(**P["KNN"]))


def build_gradient_boosting() -> Pipeline:
    return _pipe(GradientBoostingClassifier(
        random_state=config.RANDOM_STATE, **P["GradientBoosting"]))


def build_extra_trees() -> Pipeline:
    """Highly randomized trees: a strong, low-cost complement to boosting."""
    return _pipe(ExtraTreesClassifier(
        n_estimators=700, max_features="sqrt", min_samples_leaf=1,
        class_weight=config.CLASS_WEIGHT, random_state=config.RANDOM_STATE,
        n_jobs=1))


def build_xgboost(scale_pos_weight: float = 1.0, **overrides) -> Pipeline:
    """Gradient-boosted trees (XGBoost) — strong on CPU; imbalance via
    ``scale_pos_weight`` (= n_negative / n_positive on the training split)."""
    if not XGB_AVAILABLE:
        raise RuntimeError("XGBoost is not installed.")
    params = dict(P["XGBoost"])
    params.update(overrides)
    return _pipe(XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=config.RANDOM_STATE, n_jobs=1, **params))


def build_all_classical() -> dict:
    """Return {name: unfitted pipeline} for every classical model."""
    models = {
        "RandomForest": build_random_forest(),
        "SVM": build_svm(),
        "KNN": build_knn(),
        "GradientBoosting": build_gradient_boosting(),
        "ExtraTrees": build_extra_trees(),
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = build_xgboost()
    return models
