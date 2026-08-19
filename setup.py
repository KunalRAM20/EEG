"""
One-command setup + training orchestrator for the Epilepsy CDSS.

    python setup.py

Runs the whole accuracy pipeline end-to-end on a GPU-less laptop:

  1. checks (and optionally installs) Python dependencies,
  2. ensures the real EEG dataset is present (downloads if missing),
  3. extracts features from the ~11,500 EEG segments IN CHUNKS across CPU cores,
     caching them to disk so reruns are instant,
  4. trains + compares the models recording-wise, calibrates the best, and
  5. prints the honest, real-world (patient-disjoint) accuracy.

This is intentionally an orchestration script, not a packaging file — it exists
so a single command reproduces the trained model.

Flags:
    --deep            also train the CPU-heavy deep models (CNN/LSTM/CNN-LSTM)
    --chunk-size N    feature-extraction batch size (default 512; lower = lighter)
    --jobs N          parallel worker processes (default: all cores)
    --install         pip-install any missing dependencies first
    --force-extract   ignore the feature cache and re-extract
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Import name -> pip package name (differs for a few).
REQUIRED = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
    "sklearn": "scikit-learn", "joblib": "joblib",
}
OPTIONAL = {
    "xgboost": "xgboost",        # strong CPU booster (recommended)
    "pywt": "PyWavelets",        # DWT sub-band features (recommended)
    "flask": "Flask",            # web app only
    "torch": "torch",            # deep models only (--deep)
    "shap": "shap", "lime": "lime",   # explainability (optional)
}


def _hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def _check_deps(install: bool):
    _hr("STEP 1/4  Dependencies")
    missing_req, missing_opt = [], []
    for mod, pkg in REQUIRED.items():
        try:
            __import__(mod)
            print(f"  [ ok ] {pkg}")
        except Exception:
            missing_req.append(pkg)
            print(f"  [MISS] {pkg}  (required)")
    for mod, pkg in OPTIONAL.items():
        try:
            __import__(mod)
            print(f"  [ ok ] {pkg}")
        except Exception:
            missing_opt.append(pkg)
            print(f"  [ -- ] {pkg}  (optional)")

    to_install = missing_req + (missing_opt if install else [])
    if to_install and install:
        print(f"\n  Installing: {', '.join(to_install)}")
        subprocess.run([sys.executable, "-m", "pip", "install", *to_install],
                       check=False)
    elif missing_req:
        print("\n  Missing REQUIRED packages. Re-run with --install, or:")
        print(f"    pip install {' '.join(missing_req)}")
        sys.exit(1)
    if missing_opt and not install:
        print("\n  (Optional packages missing — pipeline still runs. "
              "Use --install for max accuracy: xgboost + PyWavelets.)")


def _ensure_data():
    _hr("STEP 2/4  Dataset")
    import config
    from data.load_dataset import load_eeg_segments
    have_csv = any(f.lower().endswith(".csv") for f in os.listdir(config.BONN_DIR)) \
        if os.path.isdir(config.BONN_DIR) else False
    have_bonn = os.path.isdir(os.path.join(config.BONN_DIR, "Z"))
    if not (have_csv or have_bonn):
        print("  No dataset found — attempting download ...")
        try:
            from data.download_data import download_all
            download_all()
        except Exception as exc:
            print(f"  Download failed ({exc}); will use synthetic fallback.")
    # Report what will actually be used.
    _, labels, _, groups, source = load_eeg_segments(verbose=False)
    print(f"  Source     : {source}")
    print(f"  Segments   : {len(labels)}  "
          f"(seizure={int(sum(labels))}, non-seizure={len(labels) - int(sum(labels))})")
    print(f"  Recordings : {len(set(groups.tolist()))}")
    if source == "synthetic":
        print("  NOTE: using SYNTHETIC data — drop the UCI CSV into data/bonn/ "
              "for real-world accuracy (see README).")


def _train(args):
    _hr("STEP 3/4  Feature extraction (chunked) + training")
    from src.models.train import train_and_evaluate
    if args.force_extract:
        _clear_feature_cache()
    meta = train_and_evaluate(verbose=True, include_deep=args.deep,
                              chunk_size=args.chunk_size, n_jobs=args.jobs)
    return meta


def _clear_feature_cache():
    import config
    cache = os.path.join(config.MODELS_DIR, "cache")
    if os.path.isdir(cache):
        for f in os.listdir(cache):
            os.remove(os.path.join(cache, f))
        print("  [cache] cleared feature cache (forced re-extraction).")


def _summary(meta):
    _hr("STEP 4/4  Result  —  honest real-world performance")
    tm = meta["test_metrics"]
    print(f"  Deployed model : {meta['best_model']}")
    print(f"  Evaluation     : {meta['evaluation']}")
    print(f"  Data source    : {meta['data_source']}  "
          f"({meta['n_recordings']} recordings)")
    print(f"  Threshold      : {meta['decision_threshold']} "
          f"(sensitivity target {meta['sensitivity_target']})")
    print("  ---- held-out TEST recordings (never seen in training) ----")
    print(f"    accuracy          : {tm['accuracy']:.4f}")
    print(f"    balanced accuracy : {tm['balanced_accuracy']:.4f}")
    print(f"    sensitivity       : {tm['sensitivity']:.4f}")
    print(f"    specificity       : {tm['specificity']:.4f}")
    print(f"    ROC-AUC / PR-AUC  : {tm.get('roc_auc')} / {tm['pr_auc']}")
    print(f"    Brier (calib.)    : {tm.get('brier')}")
    print("\n  Next:")
    print("    python run_pipeline.py      # end-to-end demo reports")
    print("    python webapp/app.py        # clinical web app -> http://127.0.0.1:5000")


def main():
    ap = argparse.ArgumentParser(description="Epilepsy CDSS setup + training")
    ap.add_argument("--deep", action="store_true", help="also train DL models")
    ap.add_argument("--chunk-size", type=int, default=256,
                    help="signals processed per chunk (lower uses less RAM)")
    ap.add_argument("--jobs", type=int, default=-1,
                    help="feature workers; use 1-2 if the laptop becomes busy")
    ap.add_argument("--install", action="store_true", help="pip install missing deps")
    ap.add_argument("--force-extract", action="store_true", help="ignore feature cache")
    args = ap.parse_args()

    if args.chunk_size < 1:
        ap.error("--chunk-size must be at least 1")
    if args.jobs == 0 or args.jobs < -1:
        ap.error("--jobs must be -1 or a positive integer")

    _check_deps(args.install)
    _ensure_data()
    meta = _train(args)
    _summary(meta)


if __name__ == "__main__":
    main()
