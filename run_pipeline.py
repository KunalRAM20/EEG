"""
End-to-end demo pipeline.

Downloads the dataset (or uses the synthetic fallback), trains the models if no
trained artefacts exist yet, then runs prediction -> XAI -> report generation on
a few sample patients and prints the resulting clinical reports.

    python run_pipeline.py
"""
from __future__ import annotations

import os

import config
from data.load_dataset import load_dataset
from src.models.inference import PREDICTOR
from src.reporting.report_generator import build_full_result


def ensure_trained():
    if PREDICTOR.is_trained():
        print("[pipeline] Trained model found — skipping training.")
        return
    print("[pipeline] No trained model — training now ...")
    from src.models.train import train_and_evaluate
    train_and_evaluate(verbose=True)


def main():
    # 1. Data (attempt real download; falls back to synthetic automatically)
    try:
        from data.download_data import download_all
        download_all()
    except Exception as exc:  # pragma: no cover
        print(f"[pipeline] download step skipped: {exc}")

    # 2. Model
    ensure_trained()

    # 3. Pick one sample from each clinical group and produce a report
    segments, labels, records, groups, source = load_dataset(verbose=False)
    print(f"\n[pipeline] Data source: {source}\n")

    seen = set()
    for i, rec in enumerate(records):
        grp = rec["eeg_group"]
        if grp in seen:
            continue
        seen.add(grp)
        patient = dict(rec)
        patient["name"] = f"Demo-{grp}"
        patient["patient_id"] = f"DEMO-{i}"
        result = build_full_result(patient, segments[i])
        print(result["report"])
        print()
        if len(seen) >= 3:
            break

    print("[pipeline] Done. Launch the web app with:  python webapp/app.py")


if __name__ == "__main__":
    main()
