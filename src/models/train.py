"""
Train leakage-controlled classical deployment candidates, optionally run deep
and ensemble diagnostics, then calibrate + threshold the selected classical
model and persist it.

Leakage-controlled internal evaluation protocol:

  * THREE recording-disjoint sets by original-recording id:
    TRAIN, CALIBRATION, TEST. No recording's chunks ever appear in more than one
    set, so metrics estimate performance on unseen recordings rather than
    memorised chunks. UCI has no reliable patient ID, so this is not
    patient-disjoint external validation. (A plain random split leaks — it shuffles 23
    chunks per recording — which is why the old pipeline scored a fake 1.0.)
  * Classical models and hyperparameters are ranked by recording-stratified
    cross-validation inside TRAIN using PR-AUC (imbalance-aware).
  * A soft-vote ensemble is reported as a diagnostic candidate; deployment is
    selected only from candidates with leakage-free TRAIN cross-validation.
  * The winner is probability-CALIBRATED on CALIBRATION (leakage-free) so the
    seizure probability shown to clinicians is trustworthy.
  * Sensitivity-first and maximum-balanced-accuracy thresholds are chosen on
    CALIBRATION; the configured policy is deployed instead of a naive 0.5.
  * FINAL headline metrics (sensitivity, specificity, PPV, NPV, balanced accuracy,
    ROC-AUC, PR-AUC, Brier) are reported on the untouched TEST set.

Run with:  python -m src.models.train        (classical + XGBoost + ensemble)
           python -m src.models.train --deep  (also train the CPU-heavy DL models)
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

import joblib
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, roc_auc_score)
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import config
from data.load_dataset import load_dataset
from src.models.calibrated import fit_calibrated
from src.models.classical import (XGB_AVAILABLE, build_gradient_boosting,
                                  build_extra_trees, build_knn,
                                  build_random_forest, build_svm, build_xgboost)
from src.models.deep_learning import TORCH_AVAILABLE, TorchClassifier, build_all_deep
from src.preprocessing.feature_extraction import FEATURE_NAMES, FEATURE_VERSION
from src.preprocessing.features_cache import get_feature_matrix, get_window_matrix


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _temporary_path(path: str, suffix: str) -> str:
    fd, temp_path = tempfile.mkstemp(
        prefix="artifact_", suffix=suffix,
        dir=os.path.dirname(path))
    os.close(fd)
    return temp_path


def _atomic_joblib_dump(value, path: str):
    temp_path = _temporary_path(path, ".joblib")
    try:
        joblib.dump(value, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _atomic_numpy_save(value, path: str):
    temp_path = _temporary_path(path, ".npy")
    try:
        np.save(temp_path, value)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _atomic_json_dump(value, path: str):
    temp_path = _temporary_path(path, ".json")
    try:
        with open(temp_path, "w", encoding="utf-8") as fh:
            json.dump(value, fh, indent=2, default=str)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _rank_metrics(y_true, prob) -> dict:
    """Threshold-free metrics (need only the probability ranking)."""
    out = {"pr_auc": round(float(average_precision_score(y_true, prob)), 4)}
    try:
        out["roc_auc"] = round(float(roc_auc_score(y_true, prob)), 4)
    except ValueError:
        out["roc_auc"] = None
    return out


def _threshold_metrics(y_true, prob, thr: float) -> dict:
    """Clinical metrics at a fixed decision threshold."""
    y_pred = (np.asarray(prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0          # recall on seizures
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0           # precision
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "threshold": round(float(thr), 4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
        "ppv": round(float(ppv), 4),
        "npv": round(float(npv), 4),
        "balanced_accuracy": round(float((sens + spec) / 2), 4),
        "accuracy": round(float((tp + tn) / max(1, tp + tn + fp + fn)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _max_balanced_threshold(y_true, prob) -> float:
    """Threshold maximising balanced accuracy (peak-accuracy operating point)."""
    cands = np.unique(np.concatenate([[0.0], np.asarray(prob), [1.0]]))
    best_thr, best_bal = config.DEFAULT_THRESHOLD, -1.0
    for t in cands:
        m = _threshold_metrics(y_true, prob, t)
        if (m["balanced_accuracy"] > best_bal or
                (m["balanced_accuracy"] == best_bal and t > best_thr)):
            best_bal, best_thr = m["balanced_accuracy"], t
    return float(best_thr)


def _choose_threshold(y_true, prob, target_sens: float) -> float:
    """
    Highest threshold that still reaches ``target_sens`` sensitivity, tie-broken
    by specificity. Falls back to the threshold maximising Youden's J if the
    target is unreachable.
    """
    cands = np.unique(np.concatenate([[0.0], np.asarray(prob), [1.0]]))
    best_thr, best_spec, best_j, best_j_thr = None, -1.0, -1.0, config.DEFAULT_THRESHOLD
    for t in cands:
        m = _threshold_metrics(y_true, prob, t)
        j = m["sensitivity"] + m["specificity"] - 1
        if j > best_j:
            best_j, best_j_thr = j, t
        if (m["sensitivity"] >= target_sens and
                (m["specificity"] > best_spec or
                 (m["specificity"] == best_spec and
                  (best_thr is None or t > best_thr)))):
            best_spec, best_thr = m["specificity"], t
    return float(best_thr if best_thr is not None else best_j_thr)


# --------------------------------------------------------------------------- #
# Recording-disjoint three-way split
# --------------------------------------------------------------------------- #
def _grouped_three_way(n, y, groups):
    """Stratified 60/20/20 split with each recording in exactly one set."""
    idx = np.arange(n)
    unique_groups, first = np.unique(groups, return_index=True)
    group_y = y[first]
    # This dataset assigns one class to every chunk from a recording. Fail
    # loudly if a future dataset violates that assumption.
    for group, label in zip(unique_groups, group_y):
        if np.any(y[groups == group] != label):
            raise ValueError(f"recording {group!r} contains mixed labels")

    outer = StratifiedShuffleSplit(n_splits=1, test_size=config.TEST_SIZE,
                                    random_state=config.RANDOM_STATE)
    dev_g_rel, test_g_rel = next(outer.split(unique_groups, group_y))
    dev_groups, test_groups = unique_groups[dev_g_rel], unique_groups[test_g_rel]

    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.25,
                                    random_state=config.RANDOM_STATE + 1)
    tr_g_rel, cal_g_rel = next(inner.split(dev_groups, group_y[dev_g_rel]))
    train_groups, calib_groups = dev_groups[tr_g_rel], dev_groups[cal_g_rel]
    train = idx[np.isin(groups, train_groups)]
    calib = idx[np.isin(groups, calib_groups)]
    test = idx[np.isin(groups, test_groups)]

    g = {k: set(groups[v].tolist()) for k, v in
         (("train", train), ("calib", calib), ("test", test))}
    assert not (g["train"] & g["calib"]), "leakage: train/calib share a recording"
    assert not (g["train"] & g["test"]), "leakage: train/test share a recording"
    assert not (g["calib"] & g["test"]), "leakage: calib/test share a recording"
    return train, calib, test


# --------------------------------------------------------------------------- #
# Base-model factory
# --------------------------------------------------------------------------- #
def _classical_models(y_tr):
    """Fresh unfitted classical pipelines (XGBoost gets an imbalance weight)."""
    models = {
        "RandomForest": build_random_forest(),
        "SVM": build_svm(),
        "KNN": build_knn(),
        "GradientBoosting": build_gradient_boosting(),
        "ExtraTrees": build_extra_trees(),
    }
    if XGB_AVAILABLE:
        spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
        for i, params in enumerate(config.XGBOOST_SEARCH, 1):
            models[f"XGBoost-{i}"] = build_xgboost(spw, **params)
    return models


def _group_cv_pr_auc(model, X, y, groups) -> tuple[float, float]:
    """Mean/std PR-AUC on recording-disjoint folds, used only for selection."""
    unique_groups, first = np.unique(groups, return_index=True)
    group_y = y[first]
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE + 2)
    scores = []
    for fit_g, val_g in cv.split(unique_groups, group_y):
        fit_idx = np.flatnonzero(np.isin(groups, unique_groups[fit_g]))
        val_idx = np.flatnonzero(np.isin(groups, unique_groups[val_g]))
        candidate = clone(model)
        candidate.fit(X[fit_idx], y[fit_idx])
        prob = candidate.predict_proba(X[val_idx])[:, 1]
        scores.append(average_precision_score(y[val_idx], prob))
    return float(np.mean(scores)), float(np.std(scores))


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_and_evaluate(verbose: bool = True, include_deep: bool = False,
                       chunk_size: int = 512, n_jobs: int = -1):
    # ------------------------------------------------------------------ load
    segments, labels, records, groups, source = load_dataset(verbose=verbose)
    if source == "synthetic":
        raise RuntimeError(
            "Training on synthetic fallback data is disabled. Add the real "
            "Bonn/UCI dataset under data/bonn/ and rerun."
        )
    if include_deep and not TORCH_AVAILABLE:
        raise RuntimeError(
            "--deep was requested, but PyTorch cannot be imported in this "
            "Python environment."
        )
    labels = np.asarray(labels)
    groups = np.asarray(groups, dtype=object)
    n_recordings = len(set(groups.tolist()))

    X_feat = get_feature_matrix(segments, source, chunk_size=chunk_size,
                                n_jobs=n_jobs, progress=verbose)
    X_win = None
    if include_deep and TORCH_AVAILABLE:
        X_win = get_window_matrix(segments, source, chunk_size=chunk_size,
                                  n_jobs=n_jobs, progress=verbose)

    train_idx, calib_idx, test_idx = _grouped_three_way(len(labels), labels, groups)
    if verbose:
        print(f"[train] Recording-wise split: {len(train_idx)} train / "
              f"{len(calib_idx)} calib / {len(test_idx)} test rows "
              f"({n_recordings} recordings, none shared).")

    y_tr, y_cal, y_te = labels[train_idx], labels[calib_idx], labels[test_idx]
    results, fitted, inputs = {}, {}, {}

    def _slice(X, part):
        return X[{"train": train_idx, "calib": calib_idx, "test": test_idx}[part]]

    # -------------------------------------------------------- classical models
    for name, model in _classical_models(y_tr).items():
        if verbose:
            print(f"[train] {name}: {config.CV_FOLDS}-fold grouped CV + fit ...")
        cv_mean, cv_std = _group_cv_pr_auc(
            model, _slice(X_feat, "train"), y_tr, groups[train_idx])
        model.fit(_slice(X_feat, "train"), y_tr)
        p_cal = model.predict_proba(_slice(X_feat, "calib"))[:, 1]
        m = _rank_metrics(y_cal, p_cal)
        m.update(_threshold_metrics(y_cal, p_cal, config.DEFAULT_THRESHOLD))
        m["type"], m["input"] = "classical", "features"
        m["selection_pr_auc_mean"] = round(cv_mean, 4)
        m["selection_pr_auc_std"] = round(cv_std, 4)
        results[name], fitted[name], inputs[name] = m, model, "features"

    # ------------------------------------------------------------- deep models
    if include_deep and TORCH_AVAILABLE:
        for name in build_all_deep():
            if verbose:
                print(f"[train] {name} (deep) ...")
            model = TorchClassifier(name)
            model.fit(_slice(X_win, "train"), y_tr, groups=groups[train_idx],
                      verbose=verbose)
            p_cal = model.predict_proba(_slice(X_win, "calib"))[:, 1]
            m = _rank_metrics(y_cal, p_cal)
            m.update(_threshold_metrics(y_cal, p_cal, config.DEFAULT_THRESHOLD))
            m["type"], m["input"] = "deep", "window"
            m["selection_eligible"] = False
            results[name], fitted[name], inputs[name] = m, model, "window"

    # ------------------------------------------------------ soft-vote ensemble
    feature_ranked = sorted(
        (n for n in results if inputs[n] == "features"),
        key=lambda n: -results[n]["pr_auc"])
    ensemble_members = feature_ranked[:config.ENSEMBLE_TOP_K]
    if len(ensemble_members) >= 2:
        members = [fitted[n] for n in ensemble_members]
        p_cal = np.mean([m.predict_proba(_slice(X_feat, "calib"))[:, 1]
                         for m in members], axis=0)
        m = _rank_metrics(y_cal, p_cal)
        m.update(_threshold_metrics(y_cal, p_cal, config.DEFAULT_THRESHOLD))
        m["type"], m["input"] = "ensemble", "features"
        m["selection_eligible"] = False
        m["members"] = ensemble_members
        results["Ensemble"] = m
        inputs["Ensemble"] = "features"

    # ------------------------------------------------------------ select best
    # Classical candidates are selected exclusively by grouped CV inside TRAIN.
    # CALIBRATION remains untouched until probability calibration/thresholding.
    selectable = [n for n in results if "selection_pr_auc_mean" in results[n]]
    best_name = max(selectable, key=lambda n: (
        results[n]["selection_pr_auc_mean"], -results[n]["selection_pr_auc_std"]))
    best_input = inputs[best_name]

    deploy_members = [fitted[best_name]]

    # ------------------------------------------- calibrate + threshold + test
    Xin = X_feat if best_input == "features" else X_win
    deployed = fit_calibrated(deploy_members, _slice(Xin, "calib"), y_cal,
                              method=config.CALIBRATION_METHOD, input_kind=best_input)

    p_cal_final = deployed.predict_proba(_slice(Xin, "calib"))[:, 1]
    # Both operating points are derived on CALIBRATION; the config policy selects
    # which one is DEPLOYED, and the other is reported as the alternative.
    sens_thr = _choose_threshold(y_cal, p_cal_final, config.SENSITIVITY_TARGET)
    bal_thr = _max_balanced_threshold(y_cal, p_cal_final)
    if config.OPERATING_POINT == "sensitivity":
        threshold, alt_thr = sens_thr, bal_thr
    else:
        threshold, alt_thr = bal_thr, sens_thr
    deployed.decision_threshold = float(threshold)

    p_te = deployed.predict_proba(_slice(Xin, "test"))[:, 1]
    test_metrics = _rank_metrics(y_te, p_te)
    test_metrics.update(_threshold_metrics(y_te, p_te, threshold))
    test_metrics["brier"] = round(float(brier_score_loss(y_te, p_te)), 4)
    results[best_name]["test"] = test_metrics

    # The alternative operating point, reported on TEST for full transparency.
    peak_metrics = _threshold_metrics(y_te, p_te, alt_thr)

    # ----------------------------------------------------- XAI surrogate model
    dev_idx = np.concatenate([train_idx, calib_idx])
    classical_names = [n for n in results if results[n].get("type") == "classical"]
    surrogate_name = max(classical_names, key=lambda n: results[n]["pr_auc"])
    surrogate = _classical_models(labels[dev_idx])[surrogate_name]
    surrogate.fit(X_feat[dev_idx], labels[dev_idx])

    deployed_label = best_name
    artifact_run_id = uuid.uuid4().hex
    deployed.artifact_run_id = artifact_run_id
    surrogate.artifact_run_id = artifact_run_id
    metadata = {
        "artifact_run_id": artifact_run_id,
        "best_model": deployed_label,
        "best_type": results[best_name]["type"],
        "best_input": best_input,
        "surrogate_model": surrogate_name,
        "data_source": source,
        "evaluation": "stratified recording-wise train/calibration/test; recording-level CV selection",
        "calibrated": True,
        "calibration_method": config.CALIBRATION_METHOD,
        "operating_point": config.OPERATING_POINT,
        "decision_threshold": round(float(threshold), 4),
        "alt_threshold": round(float(alt_thr), 4),
        "sensitivity_target": config.SENSITIVITY_TARGET,
        "n_samples": int(len(labels)),
        "segment_length": int(np.asarray(segments).shape[1]),
        "n_recordings": int(n_recordings),
        "n_train": int(len(train_idx)),
        "n_calibration": int(len(calib_idx)),
        "n_test": int(len(test_idx)),
        "n_seizure": int(labels.sum()),
        "n_features": len(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "feature_schema_sha256": hashlib.sha256(
            "\0".join(FEATURE_NAMES).encode("utf-8")).hexdigest(),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "scikit-learn": _package_version("scikit-learn"),
            "PyWavelets": _package_version("PyWavelets"),
            "xgboost": _package_version("xgboost"),
        },
        "torch_available": bool(TORCH_AVAILABLE),
        "xgboost_available": bool(XGB_AVAILABLE),
        "deep_trained": bool(include_deep and TORCH_AVAILABLE),
        "test_metrics": test_metrics,
        "alternative_operating_point": peak_metrics,
        "metrics": results,
    }

    # Replace each complete file atomically and publish metadata last. A process
    # interrupted midway is rejected at load time by artifact_run_id rather than
    # silently combining model files from different training runs.
    _atomic_joblib_dump(deployed, config.BEST_MODEL_PATH)
    _atomic_joblib_dump(surrogate, config.EXPLAINER_MODEL_PATH)
    _atomic_numpy_save(X_feat[train_idx][:100], config.BACKGROUND_PATH)
    _atomic_json_dump(FEATURE_NAMES, config.FEATURE_NAMES_PATH)
    _atomic_json_dump(metadata, config.METRICS_PATH)

    if verbose:
        _print_table(results, best_name, deployed_label, surrogate_name, source,
                     test_metrics, threshold)
        alt_name = ("sensitivity-first" if config.OPERATING_POINT != "sensitivity"
                    else "peak-accuracy")
        print(f"  alternative {alt_name} point (thr={alt_thr:.3f}): "
              f"accuracy={peak_metrics['accuracy']:.4f} "
              f"balanced={peak_metrics['balanced_accuracy']:.4f} "
              f"sens={peak_metrics['sensitivity']:.4f} "
              f"spec={peak_metrics['specificity']:.4f}")
        print("=" * 84)
    return metadata


def _print_table(results, best_name, deployed_label, surrogate_name, source,
                 test_metrics, thr):
    print("\n" + "=" * 84)
    print(f"MODEL COMPARISON  (ranked on held-out CALIBRATION recordings; "
          f"data: {source})")
    print("=" * 84)
    print(f"{'Model':<18}{'PR-AUC':>8}{'ROC-AUC':>9}{'Sens@.5':>9}{'Spec@.5':>9}"
          f"{'BalAcc':>8}")
    print("-" * 84)
    for name, m in sorted(results.items(), key=lambda kv: -kv[1]["pr_auc"]):
        roc = f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "  -  "
        star = " *" if name == best_name else ""
        print(f"{name:<18}{m['pr_auc']:>8.3f}{roc:>9}{m['sensitivity']:>9.3f}"
              f"{m['specificity']:>9.3f}{m['balanced_accuracy']:>8.3f}{star}")
    print("-" * 84)
    print(f"Deployed model  : {deployed_label}  (calibrated -> best_model.joblib)")
    print(f"XAI surrogate   : {surrogate_name}  (feature-based, for explanations)")
    policy = ("max balanced accuracy" if config.OPERATING_POINT != "sensitivity"
              else f"sensitivity >= {config.SENSITIVITY_TARGET}")
    print(f"Decision thresh : {thr:.3f}  (operating point: {policy})")
    print("-" * 84)
    print("HONEST TEST-SET PERFORMANCE (untouched recordings, at the threshold above):")
    tm = test_metrics
    print(f"  accuracy    = {tm['accuracy']:.4f}      balanced_accuracy = "
          f"{tm['balanced_accuracy']:.4f}")
    print(f"  sensitivity = {tm['sensitivity']:.4f}      specificity       = "
          f"{tm['specificity']:.4f}")
    print(f"  PPV         = {tm['ppv']:.4f}      NPV               = {tm['npv']:.4f}")
    print(f"  ROC-AUC     = {tm.get('roc_auc')}      PR-AUC            = "
          f"{tm['pr_auc']:.4f}      Brier = {tm.get('brier')}")
    print(f"  confusion   = {tm['confusion']}")
    print(f"Artefacts saved : {config.MODELS_DIR}")
    print("=" * 84)


if __name__ == "__main__":
    train_and_evaluate(include_deep="--deep" in sys.argv)
