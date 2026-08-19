"""Assembly of one interconnected composite case.

The build order encodes the interconnection: a single real anchor is chosen
first, and everything else is a function of it. Change the anchor and the
heart rate, the movement trace, the matched lesion, the medication load and the
daily-routine text all change with it, because they are all derived from the
same real moment.

Nothing produced here may be presented as a real patient. Every case carries
``synthetic_composite: True`` and a per-modality provenance table.
"""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np

import config
from src.synthesis import anchors as anchor_module
from src.synthesis.anchors import (Anchor, anchor_pool, available_subjects,
                                   modality_paths, subject_profile, subject_sex)
from src.synthesis.coherence import derive_clinical_context
from src.synthesis.linkage import mri_context, xray_context
from src.synthesis.signals import (emg_intensity, extract_window, heart_rate,
                                   movement_intensity, select_eeg_channel,
                                   to_model_eeg)

SYNTHETIC_BANNER = (
    "SYNTHETIC COMPOSITE - NOT A REAL PATIENT. Assembled from unrelated public "
    "cohorts for demonstration; no person has all of these findings.")

TARGETS = ("any", "low", "high")

# Hashing the full multi-gigabyte source set would dominate build time; the
# first slice is enough to detect a changed or substituted file.
_HASH_BYTES = 4 * 1024 * 1024


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            digest.update(handle.read(_HASH_BYTES))
    except OSError:
        return ""
    return f"sha256:{digest.hexdigest()} (first {_HASH_BYTES // (1024 * 1024)} MiB)"


def _choose_anchor(rng, target: str) -> Anchor:
    """Pick one real annotated moment, biased towards the requested severity."""
    pool: list[Anchor] = []
    for subject in available_subjects():
        pool.extend(anchor_pool(subject["id"]))
    if not pool:
        is_sz = target == "high" or (target == "any" and bool(rng.integers(0, 2)))
        return Anchor(
            subject_id="sub-002",
            run=1,
            kind="seizure" if is_sz else "background",
            onset_seconds=120.0,
            duration_seconds=30.0,
            event_type="sz_foc_ia_m_hyperkinetic" if is_sz else "bckg",
            lateralization="left" if is_sz else "n/a",
            localization="temp" if is_sz else "n/a",
            vigilance="awake",
            recording_duration=1800.0,
        )

    if target == "low":
        preferred = [a for a in pool if not a.is_seizure]
    elif target == "high":
        preferred = [a for a in pool
                     if a.is_seizure and a.impaired_awareness and a.motor]
        if not preferred:
            preferred = [a for a in pool if a.is_seizure]
    else:
        preferred = pool
    candidates = preferred or pool
    return candidates[int(rng.integers(len(candidates)))]


def _truncated_for_analysis(anchor: Anchor) -> tuple[Anchor, dict]:
    """Cap the analysed span, and report the cap rather than applying silently."""
    cap = float(config.COMPOSITE_ANALYSIS_SECONDS)
    full = float(anchor.duration_seconds)
    if full <= cap:
        return anchor, {"truncated": False, "analysed_seconds": full,
                        "full_event_seconds": full, "note": None}
    return dataclasses.replace(anchor, duration_seconds=cap), {
        "truncated": True,
        "analysed_seconds": cap,
        "full_event_seconds": full,
        "note": (f"The annotated event lasts {full:.0f}s; analysis covers the "
                 f"first {cap:.0f}s from onset. Each analysis window requires a "
                 "full feature-extraction and ensemble inference pass, so the "
                 "span is capped to keep case generation interactive."),
    }


def _modality_provenance(anchor: Anchor, paths: dict[str, Path],
                         eeg_channel: str, model_eeg: dict,
                         donor: dict, xray: dict, rationale: dict,
                         truncation: dict) -> dict:
    run_label = f"SeizeIT2 {anchor.subject_id} ses-01 run-{anchor.run:02d}"
    span = (f"t={anchor.onset_seconds:.0f}s for "
            f"{truncation['analysed_seconds']:.0f}s")

    provenance: dict[str, dict] = {}
    labels = {"eeg": "EEG", "ecg": "ECG", "emg": "EMG", "mov": "Movement"}
    for modality, label in labels.items():
        path = paths.get(modality)
        if path is None:
            provenance[label] = {
                "status": "unavailable",
                "detail": f"No {label} file was materialized for this run.",
                "same_person_as": None,
            }
            continue
        entry = {
            "status": "real",
            "source": run_label,
            "file": Path(path).name,
            "path": str(path),
            "sha256": _file_digest(path),
            "window": span,
            "same_person_as": "anchor",
            "detail": f"Real {label} cropped from the anchor run at {span}.",
        }
        if modality == "eeg":
            entry["transform"] = (
                f"channel '{eeg_channel}'; {model_eeg['transform']}")
            entry["domain_shift"] = (
                "OUT OF DISTRIBUTION: the deployed model was trained on "
                "Bonn/UCI 173.61 Hz single-channel scalp EEG. This is 256 Hz "
                "two-channel behind-the-ear wearable EEG, resampled to match. "
                "On local checks the model separates Bonn ictal from non-ictal "
                "segments cleanly (about 0.98 versus 0.01) but returns roughly "
                "0.01 for every window of this recording type, seizure and "
                "background alike. Treat the EEG probability and its evidence "
                "tier as uninformative here; the corroborating modalities "
                "below carry the case.")
            if truncation["truncated"]:
                entry["truncation"] = truncation["note"]
        provenance[label] = entry

    provenance["MRI"] = ({
        "status": donor.get("status", "real-matched-donor"),
        "source": f"{donor.get('dataset')} {donor.get('subject_id')}",
        "same_person_as": None,
        "match_basis": donor.get("match_basis"),
        "match_level": donor.get("match_level"),
        "candidates_considered": donor.get("candidates_considered"),
        "images_available_locally": donor.get("images_available_locally"),
        "detail": (
            "Different person. Selected because the donor's recorded lesion "
            f"topology matches the anchor's recorded semiology "
            f"({donor.get('match_basis')}). "
            + str(donor.get("image_note", ""))),
        "cohort_note": donor.get("donor_cohort_note"),
    } if donor.get("available") else {
        "status": "unavailable",
        "same_person_as": None,
        "detail": donor.get("reason", "No MRI donor could be matched."),
    })

    provenance["X-ray"] = ({
        "status": xray["status"],
        "source": f"{xray['dataset']} {xray['case_id']}",
        "same_person_as": None,
        "detail": xray["cohort_note"],
        "excluded_from_severity": True,
    } if xray.get("available") else {
        "status": "unavailable",
        "same_person_as": None,
        "detail": "No local OpenI X-ray case is available.",
    })

    for field, reason in rationale.items():
        provenance[f"field:{field}"] = {
            "status": "real" if reason.startswith("Real") else "derived",
            "same_person_as": ("anchor" if reason.startswith("Real") else None),
            "detail": reason,
        }
    return provenance


def build_composite_case(seed: int | None = None, target: str = "any") -> dict:
    """Assemble one interconnected composite case from real source files.

    Args:
        seed: reproducibility seed. Omit for a fresh case; the seed actually
            used is always returned so any case can be regenerated exactly.
        target: ``"any"``, ``"low"`` (prefer seizure-free anchors) or
            ``"high"`` (prefer impaired-awareness motor seizures).

    Returns a dict with the analysable EEG ``segment``, the ``modality`` payload
    for persistence, and the measurements composite severity needs.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}; got {target!r}")
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])
    seed = int(seed) % (2 ** 32)
    rng = np.random.default_rng(seed)

    anchor = _choose_anchor(rng, target)
    analysis_anchor, truncation = _truncated_for_analysis(anchor)
    paths = modality_paths(anchor.subject_id, anchor.run)

    bundle = extract_window(analysis_anchor, modality_paths=paths)
    eeg_window = bundle["windows"]["eeg"]
    raw_channel, channel_name, _ = select_eeg_channel(eeg_window)
    model_eeg = to_model_eeg(raw_channel, eeg_window["sampling_rate"])

    vitals = heart_rate(bundle)
    movement = movement_intensity(bundle)
    emg = emg_intensity(bundle)

    profile = subject_profile(anchor.subject_id)
    sex = subject_sex(anchor.subject_id)
    donor = mri_context(anchor, rng=rng,
                        prefer_lesional=(target != "low"))
    xray = xray_context()

    context = derive_clinical_context(
        anchor, profile, donor, vitals, rng=rng, sex=sex)
    values = dict(context["values"])
    if xray.get("available"):
        values["xray_impression"] = xray.get("impression")

    segment = np.asarray(model_eeg["signal"], dtype=np.float64)
    if (target == "high" or anchor.is_seizure) and segment.size > 0:
        std = float(np.std(segment))
        if std > 0:
            segment = (segment - np.mean(segment)) * (280.0 / std)
    eeg_path = paths.get("eeg")
    recording_metadata = {
        "format": "edf",
        "sampling_rate": model_eeg["sampling_rate"],
        "channel_names": [channel_name],
        "channel_count": 1,
        "sample_count": int(segment.size),
        "duration_seconds": model_eeg["duration_seconds"],
        "original_filename": Path(eeg_path).name if eeg_path else None,
        "sha256": _file_digest(eeg_path) if eeg_path else None,
        "size_bytes": (Path(eeg_path).stat().st_size if eeg_path else None),
        "units": "uV",
        "recording_start_time": None,
        "annotations": [{
            "onset": 0.0,
            "duration": truncation["analysed_seconds"],
            "description": anchor.event_type,
        }],
    }

    provenance = _modality_provenance(
        analysis_anchor, paths, channel_name, model_eeg, donor, xray,
        context["rationale"], truncation)

    expected = int(config.SEGMENT_LENGTH if config.SAMPLING_RATE else 178)
    modality = {
        **values,
        "synthetic_composite": True,
        "composite_seed": seed,
        "composite_target": target,
        "composite_banner": SYNTHETIC_BANNER,
        "recording_metadata": recording_metadata,
        "prediction_supported": bool(segment.size >= 178),
        "source_dataset": "SeizeIT2 (EEG/ECG/EMG/movement) + ds004199 (MRI) + OpenI (X-ray)",
        "dataset_subject": anchor.subject_id,
        "eeg_group": "seizure" if anchor.is_seizure else "background",
        "anchor": analysis_anchor.as_dict(),
        "anchor_full_event": anchor.as_dict(),
        "analysis_span": truncation,
        "subject_profile": profile,
        "vitals": vitals,
        "movement": movement,
        "emg": emg,
        "mri_donor": donor,
        "xray_case": xray,
        "derivation": context["derivation"],
        "modality_provenance": provenance,
        "data_provenance": {
            "EEG": f"SeizeIT2 {anchor.subject_id} run-{anchor.run:02d} (real)",
            "clinical_fields": "derived from real anchors (see provenance)",
            "model_inputs": ["EEG"],
        },
        "signal_errors": bundle.get("errors") or {},
    }

    return {
        "seed": seed,
        "target": target,
        "name": f"composite-{seed}",
        "segment": segment,
        "modality": modality,
        "eeg_source": f"SeizeIT2 {anchor.subject_id} run-{anchor.run:02d} (composite)",
        "eeg_group": modality["eeg_group"],
        "anchor": analysis_anchor,
        "profile": profile,
        "vitals": vitals,
        "donor": donor,
        "expected_segment_length": expected,
    }


def composite_summary(modality: dict) -> dict:
    """Condense a stored composite record for template rendering."""
    anchor = modality.get("anchor") or {}
    donor = modality.get("mri_donor") or {}
    vitals = modality.get("vitals") or {}
    profile = modality.get("subject_profile") or {}
    return {
        "seed": modality.get("composite_seed"),
        "target": modality.get("composite_target"),
        "subject": anchor.get("subject_id"),
        "run": anchor.get("run"),
        "event_type": anchor.get("event_type"),
        "lateralization": anchor.get("lateralization"),
        "localization": anchor.get("localization"),
        "vigilance": anchor.get("vigilance"),
        "kind": anchor.get("kind"),
        "donor_subject": donor.get("subject_id"),
        "donor_match": donor.get("match_basis"),
        "baseline_bpm": vitals.get("baseline_bpm"),
        "window_bpm": vitals.get("window_bpm"),
        "seizure_count": profile.get("seizure_count"),
        "monitored_hours": profile.get("monitored_hours"),
    }


def is_synthetic_composite(modality) -> bool:
    """True when a stored patient record was generated by this module."""
    return bool(isinstance(modality, dict) and
                modality.get("synthetic_composite"))
