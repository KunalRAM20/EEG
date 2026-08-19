"""
End-to-end demo pipeline.

Downloads the dataset, trains the models if no
trained artefacts exist yet, then runs prediction -> XAI -> report generation on
a few deidentified dataset samples and prints the resulting reports.

    python scripts/demo.py
"""
from __future__ import annotations

import os
import sys

# Allow "python scripts/demo.py" from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from data.load_dataset import load_dataset  # noqa: E402
from src.models.inference import PREDICTOR  # noqa: E402
from src.reporting.report_generator import build_full_result  # noqa: E402


def ensure_trained():
    if PREDICTOR.is_trained():
        try:
            PREDICTOR.load()
            print("[pipeline] Compatible trained model found — skipping training.")
            return
        except RuntimeError as exc:
            print(f"[pipeline] Existing model is incompatible ({exc}); retraining.")
    print("[pipeline] No trained model — training now ...")
    from src.models.train import train_and_evaluate
    train_and_evaluate(verbose=True)


def ensure_data():
    """Download Bonn data only when no supported real dataset is present."""
    have_csv = os.path.isdir(config.BONN_DIR) and any(
        name.lower().endswith(".csv") for name in os.listdir(config.BONN_DIR)
    )
    have_txt = all(
        os.path.isdir(os.path.join(config.BONN_DIR, set_name))
        and len([
            name for name in os.listdir(os.path.join(config.BONN_DIR, set_name))
            if name.lower().endswith(".txt")
        ]) >= config.N_SEGMENTS_PER_SET
        for set_name in config.BONN_SETS
    )
    if have_csv or have_txt:
        print("[pipeline] Existing real Bonn/UCI dataset found - skipping download.")
        return

    try:
        from data.download_data import download_all
        download_all()
    except Exception as exc:  # pragma: no cover
        print(f"[pipeline] download step skipped: {exc}")


def main():
    # 1. Data (download only if no supported real data is already available)
    ensure_data()

    # 2. Refuse to train or report accuracy on the synthetic software fallback.
    segments, labels, records, groups, source = load_dataset(verbose=False)
    if source == "synthetic":
        raise RuntimeError(
            "No real Bonn/UCI dataset is available. Run `python "
            "scripts/train.py --install` after adding/downloading the real "
            "CSV; synthetic data will not be reported as real-world "
            "performance.")

    # 3. Model
    ensure_trained()

    # 4. Pick one sample from each clinical group and produce an EEG-only report
    print(f"\n[pipeline] Data source: {source}\n")

    seen = set()
    for i, rec in enumerate(records):
        grp = rec["eeg_group"]
        if grp in seen:
            continue
        seen.add(grp)
        patient = dict(rec)
        patient["name"] = "Not provided (deidentified dataset sample)"
        patient["patient_id"] = f"Bonn-row-{i}"
        result = build_full_result(patient, segments[i])
        print(result["report"])
        print()
        if len(seen) >= 3:
            break

    print("[pipeline] Done. Launch the web app with:  python webapp/app.py")



if __name__ == "__main__":
    main()
