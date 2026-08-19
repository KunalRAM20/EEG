"""
Explainable-AI layer: why did the model flag (or clear) this segment?

Explanations are always computed on the feature-based surrogate classical model
persisted during training, so they remain available and interpretable even when
the best overall model is a deep network. Three strategies are attempted in
order of fidelity, degrading gracefully so a report can always be produced:

    1. SHAP  (TreeExplainer for tree models; general Explainer otherwise)
    2. LIME  (local linear approximation)
    3. Permutation / built-in feature importance  (last-resort fallback)

Each returns a ranked list of contributing features with a signed value that is
positive when the feature pushes the prediction toward "seizure".
"""
from __future__ import annotations

import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version

import joblib
import numpy as np

import config
from src.preprocessing.feature_extraction import FEATURE_NAMES, FEATURE_VERSION


# --------------------------------------------------------------------------- #
# Artefact loading
# --------------------------------------------------------------------------- #
def _load_surrogate():
    if not os.path.exists(config.EXPLAINER_MODEL_PATH):
        raise FileNotFoundError("No surrogate model — run training first.")
    model = joblib.load(config.EXPLAINER_MODEL_PATH)
    background = (np.load(config.BACKGROUND_PATH)
                  if os.path.exists(config.BACKGROUND_PATH) else None)
    names = FEATURE_NAMES
    if os.path.exists(config.FEATURE_NAMES_PATH):
        with open(config.FEATURE_NAMES_PATH) as fh:
            names = json.load(fh)
    if os.path.exists(config.METRICS_PATH):
        with open(config.METRICS_PATH) as fh:
            metadata = json.load(fh)
        if (metadata.get("feature_version") is not None and
                int(metadata["feature_version"]) != FEATURE_VERSION):
            raise RuntimeError("XAI feature version does not match the model.")
        schema_hash = hashlib.sha256(
            "\0".join(names).encode("utf-8")).hexdigest()
        if (metadata.get("feature_schema_sha256") and
                schema_hash != metadata["feature_schema_sha256"]):
            raise RuntimeError("XAI feature order does not match the model.")
        if (metadata.get("artifact_run_id") and
                getattr(model, "artifact_run_id", None) !=
                metadata["artifact_run_id"]):
            raise RuntimeError(
                "XAI artifacts come from different training runs.")
        trained_sklearn = metadata.get("runtime", {}).get("scikit-learn")
        try:
            runtime_sklearn = version("scikit-learn")
        except PackageNotFoundError:
            runtime_sklearn = None
        if trained_sklearn and runtime_sklearn != trained_sklearn:
            raise RuntimeError(
                "XAI scikit-learn runtime does not match the trained model.")
    if background is not None and (
            background.ndim != 2 or background.shape[1] != len(names) or
            not np.all(np.isfinite(background))):
        raise RuntimeError("XAI background does not match the feature schema.")
    return model, background, names


def _split_pipeline(model):
    """Return (scaler, clf) if a sklearn Pipeline, else (None, model)."""
    if hasattr(model, "named_steps"):
        return model.named_steps.get("scaler"), model.named_steps.get("clf")
    return None, model


# --------------------------------------------------------------------------- #
# Strategy 1: SHAP
# --------------------------------------------------------------------------- #
def _shap_contributions(model, background, feature_vector):
    import shap

    scaler, clf = _split_pipeline(model)
    x = feature_vector.reshape(1, -1)
    x_scaled = scaler.transform(x) if scaler is not None else x

    if hasattr(clf, "feature_importances_"):
        explainer = shap.TreeExplainer(clf)
        vals = explainer.shap_values(x_scaled)
    else:
        bg = background[:50] if background is not None else x_scaled
        bg_scaled = scaler.transform(bg) if scaler is not None else bg
        explainer = shap.KernelExplainer(clf.predict_proba, bg_scaled)
        vals = explainer.shap_values(x_scaled, nsamples=100)

    # Normalize the many possible SHAP return shapes to a (n_features,) vector
    arr = np.asarray(vals, dtype=object) if isinstance(vals, list) else np.asarray(vals)
    if isinstance(vals, list):                       # [class0, class1]
        contrib = np.asarray(vals[-1])[0]
    elif arr.ndim == 3:                              # (n, features, classes)
        contrib = arr[0, :, -1]
    elif arr.ndim == 2:                              # (n, features)
        contrib = arr[0]
    else:
        contrib = np.ravel(arr)
    return np.asarray(contrib, dtype=np.float64), "SHAP"


# --------------------------------------------------------------------------- #
# Strategy 2: LIME
# --------------------------------------------------------------------------- #
def _lime_contributions(model, background, names, feature_vector, top_k):
    from lime.lime_tabular import LimeTabularExplainer

    if background is None:
        raise RuntimeError("LIME needs a background sample.")
    explainer = LimeTabularExplainer(
        background, feature_names=names,
        class_names=["no_seizure", "seizure"], mode="classification",
        discretize_continuous=False)
    exp = explainer.explain_instance(
        feature_vector, model.predict_proba,
        num_features=len(names), labels=(1,))
    weights = dict(exp.as_map()[1])                  # {feature_index: weight}
    contrib = np.array([weights.get(i, 0.0) for i in range(len(names))])
    return contrib, "LIME"


# --------------------------------------------------------------------------- #
# Strategy 3: built-in importance with approximate background-relative sign
# --------------------------------------------------------------------------- #
def _importance_contributions(model, background, feature_vector):
    _, clf = _split_pipeline(model)
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=np.float64)
    elif hasattr(clf, "coef_"):
        imp = np.abs(np.ravel(clf.coef_)).astype(np.float64)
    else:
        imp = np.ones(len(feature_vector))
    if len(imp) != len(feature_vector):
        raise ValueError("Estimator importance width does not match feature schema.")
    center = (np.mean(background, axis=0) if background is not None
              else np.zeros(len(feature_vector), dtype=np.float64))
    if len(center) != len(feature_vector):
        raise ValueError("XAI background width does not match feature schema.")
    # Built-in tree importances are unsigned. Use the feature's position relative
    # to the training background only as an explicitly approximate direction.
    signed = imp * np.sign(np.asarray(feature_vector) - center)
    return signed, "SignedFeatureImportance"


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def explain(feature_vector: np.ndarray, top_k: int = config.XAI_TOP_K) -> dict:
    """Return the top-k contributing features for one prediction."""
    feature_vector = np.asarray(feature_vector, dtype=np.float64).reshape(-1)
    model, background, names = _load_surrogate()
    if len(feature_vector) != len(names):
        raise ValueError(
            f"Expected {len(names)} explanation features, got "
            f"{len(feature_vector)}."
        )
    if not np.all(np.isfinite(feature_vector)):
        raise ValueError("Explanation features must all be finite.")
    top_k = max(0, min(int(top_k), len(names)))

    contrib, method = None, None
    for strategy in (
        lambda: _shap_contributions(model, background, feature_vector),
        lambda: _lime_contributions(model, background, names, feature_vector, top_k),
        lambda: _importance_contributions(model, background, feature_vector),
    ):
        try:
            contrib, method = strategy()
            if contrib is not None and len(contrib) == len(names):
                break
        except Exception:
            continue

    if contrib is None:
        contrib, method = _importance_contributions(
            model, background, feature_vector)

    order = np.argsort(np.abs(contrib))[::-1][:top_k]
    top = [
        {
            "feature": names[i],
            "value": round(float(feature_vector[i]), 4),
            "contribution": round(float(contrib[i]), 6),
            "direction": ("increases" if contrib[i] > 0 else
                          "decreases" if contrib[i] < 0 else "neutral"),
        }
        for i in order
    ]
    return {"method": method, "top_features": top}


if __name__ == "__main__":
    from data.load_dataset import load_dataset
    from src.preprocessing.feature_extraction import extract_features

    seg, lab, recs, groups, src = load_dataset(verbose=False)
    fv = extract_features(seg[-1])          # a seizure segment
    print(json.dumps(explain(fv), indent=2))
