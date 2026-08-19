"""
One-command setup + training orchestrator for the Epilepsy CDSS.

    python scripts/train.py

Runs the whole accuracy pipeline end-to-end on a GPU-less laptop:

  1. checks (and optionally installs) Python dependencies,
  2. ensures the real EEG dataset is present (downloads if missing),
  3. extracts features from the ~11,500 EEG segments IN CHUNKS across CPU cores,
     caching them to disk so reruns are instant,
  4. trains + compares the models recording-wise, calibrates the best, and
  5. prints the leakage-controlled, recording-held-out internal accuracy.

This is intentionally an orchestration script, not a packaging file — it exists
so a single command reproduces the trained model.

Flags:
    --deep            also run CPU-heavy deep diagnostics (CNN/LSTM/CNN-LSTM)
    --chunk-size N    feature-extraction batch size (default 256; lower = lighter)
    --jobs N          parallel worker processes (default: all cores)
    --install         pip-install any missing dependencies first
    --force-extract   ignore the feature cache and re-extract
"""
from __future__ import annotations

import os
import sys

# Temporarily drop the entry Python injected for this script (the scripts/
# directory) so standard-library imports below are not resolved against the
# OneDrive/network filesystem.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
if sys.path and sys.path[0] in (SCRIPT_DIR, ROOT, "", "."):
    sys.path.pop(0)

# Import standard libraries securely from the system path
import argparse
import importlib.util
import subprocess

# Put the project root on sys.path so `config`, `src`, `data` and `webapp`
# import the same way they do when running from the root directory.
sys.path.insert(0, ROOT)

# Import name -> pip package name (differs for a few).
REQUIRED = {
    "numpy": "numpy", "scipy": "scipy", "pandas": "pandas",
    "sklearn": "scikit-learn", "joblib": "joblib",
    "xgboost": "xgboost",       # evaluated as a CPU candidate
    "pywt": "PyWavelets",       # fixed DWT feature schema
    "flask": "Flask",           # connected dataset/case UI
    "requests": "requests",     # dataset downloader
}
OPTIONAL = {
    "torch": "torch",            # deep models only (--deep)
    "shap": "shap", "lime": "lime",   # explainability (optional)
}


def _hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)


def _check_deps(install: bool, include_deep: bool = False):
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
        # Optional packages include heavyweight modules such as torch and shap.
        # Importing all of them here can make setup appear to hang even though
        # this step only needs to determine whether they are installed.
        try:
            available = importlib.util.find_spec(mod) is not None
        except (ImportError, AttributeError, ValueError):
            available = False
        if available:
            print(f"  [ ok ] {pkg}")
        else:
            missing_opt.append(pkg)
            print(f"  [ -- ] {pkg}  (optional)")

    if include_deep and "torch" in missing_opt:
        missing_opt.remove("torch")
        missing_req.append("torch")
        print("  [MISS] torch  (required by --deep)")

    optional_to_install = [
        pkg for mod, pkg in OPTIONAL.items()
        if pkg in missing_opt and (mod != "torch" or include_deep)
    ]
    to_install = missing_req + (optional_to_install if install else [])
    if to_install and install:
        print(f"\n  Installing: {', '.join(to_install)}")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", *to_install],
                check=True)
        except subprocess.CalledProcessError as exc:
            print(f"  Dependency installation failed (exit {exc.returncode}).")
            sys.exit(1)
        # A successful pip exit does not guarantee that wheels can be imported
        # in this interpreter (ABI/DLL errors are possible).
        importlib.invalidate_caches()
        failed_imports = []
        required_modules = list(REQUIRED)
        if include_deep:
            required_modules.append("torch")
        for mod in required_modules:
            try:
                __import__(mod)
            except Exception as exc:
                failed_imports.append(f"{mod}: {exc}")
        if failed_imports:
            print("  Installed dependencies still fail to import:")
            for failure in failed_imports:
                print(f"    {failure}")
            sys.exit(1)
    elif missing_req:
        print("\n  Missing REQUIRED packages. Re-run with --install, or:")
        print(f"    pip install {' '.join(missing_req)}")
        sys.exit(1)
    if missing_opt and not install:
        print("\n  Optional explainability/deep-learning packages are missing. "
              "Use --install if you need those optional paths.")


def _ensure_data():
    _hr("STEP 2/4  Dataset")
    import config
    from data.load_dataset import load_eeg_segments
    have_csv = any(f.lower().endswith(".csv") for f in os.listdir(config.BONN_DIR)) \
        if os.path.isdir(config.BONN_DIR) else False
    from data.download_data import is_set_present
    have_bonn = all(is_set_present(name) for name in config.BONN_SETS)
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
        print("  ERROR: only synthetic fallback data is available. Training a "
              "real-world model on it is disabled. Add the Bonn/UCI CSV to "
              "data/bonn/ and rerun scripts/train.py (see README).")
        sys.exit(1)

    # The web database and registry are lightweight and are initialized here so
    # the same one-command setup prepares both training and dataset review.
    from data.dataset_registry import dataset_catalog
    from webapp import db
    db.init_db()
    print("  Connected datasets:")
    for item in dataset_catalog():
        print(f"    {item['name']}: {item['status']}")


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
            path = os.path.join(cache, f)
            if os.path.isfile(path):
                os.remove(path)
        print("  [cache] cleared feature cache (forced re-extraction).")


def _summary(meta):
    _hr("STEP 4/4  Result  —  recording-held-out internal performance")
    tm = meta["test_metrics"]
    print(f"  Deployed model : {meta['best_model']}")
    print(f"  Evaluation     : {meta['evaluation']}")
    print(f"  Data source    : {meta['data_source']}  "
          f"({meta['n_recordings']} recordings)")
    print(f"  Threshold      : {meta['decision_threshold']} "
          f"({meta['operating_point']} policy; sensitivity target "
          f"{meta['sensitivity_target']})")
    print("  ---- held-out TEST recordings (never seen in training) ----")
    print(f"    accuracy          : {tm['accuracy']:.4f}")
    print(f"    balanced accuracy : {tm['balanced_accuracy']:.4f}")
    print(f"    sensitivity       : {tm['sensitivity']:.4f}")
    print(f"    specificity       : {tm['specificity']:.4f}")
    print(f"    ROC-AUC / PR-AUC  : {tm.get('roc_auc')} / {tm['pr_auc']}")
    print(f"    Brier (calib.)    : {tm.get('brier')}")
    print("\n  Next:")
    print("    python scripts/demo.py      # end-to-end demo reports")
    print("    python webapp/app.py        # clinical web app -> http://127.0.0.1:5000")


def main():
    ap = argparse.ArgumentParser(description="Epilepsy CDSS setup + training")
    ap.add_argument("--deep", action="store_true",
                    help="also run diagnostic DL model comparisons")
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

    _check_deps(args.install, include_deep=args.deep)
    _ensure_data()
    meta = _train(args)
    _summary(meta)


if __name__ == "__main__":
    main()
