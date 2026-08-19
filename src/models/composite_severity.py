"""Transparent multi-modal composite severity grading (Level 1 -> Level 5).

This module is deliberately NOT a model. It is an additive rule layer whose
every term is reported with its weight, its measured input and the reason it
fired, so a reviewing clinician can check the arithmetic by hand.

It sits beside — never replaces — the calibrated single-modality EEG evidence
tier in ``src/models/severity.py``, which is left untouched. The two are shown
together precisely because they answer different questions:

    EEG evidence tier   "how strongly does this EEG window resemble the
                         model's seizure training class?"
    Composite severity  "how concerning is this case once the corroborating
                         modalities are taken into account?"

The composite grade is decision support for a synthesized demonstration case.
It is not a validated clinical severity scale and carries no urgency or
treatment meaning on its own.
"""
from __future__ import annotations

import config

# Heart-rate ratio (ictal / baseline) bands. Ictal tachycardia is a well
# described autonomic accompaniment of focal seizures; these bands are graded
# rather than binary so a modest rise is not treated like a marked one.
HEART_RATE_BANDS = (
    (1.30, 1.00, "marked ictal tachycardia"),
    (1.20, 0.70, "moderate ictal tachycardia"),
    (1.10, 0.40, "mild ictal heart-rate rise"),
)

# Recorded seizures per 24 h of monitoring.
BURDEN_BANDS = (
    (2.00, 1.00, "high recorded seizure frequency"),
    (0.50, 0.60, "moderate recorded seizure frequency"),
    (0.01, 0.30, "low but non-zero recorded seizure frequency"),
)


def _term(name: str, label: str, points: float, weight: float,
          detail: str, *, available: bool = True,
          measurement=None) -> dict:
    return {
        "name": name,
        "label": label,
        "points": round(float(points), 3),
        "max_points": round(float(weight), 3),
        "detail": detail,
        "available": bool(available),
        "measurement": measurement,
    }


def _eeg_term(eeg_level) -> dict:
    weight = config.COMPOSITE_SEVERITY_WEIGHTS["eeg_evidence"]
    if eeg_level is None:
        return _term("eeg_evidence", "EEG evidence tier", 0.0, weight,
                     "No EEG evidence tier was produced for this case.",
                     available=False)
    level = int(eeg_level)
    # (level - 1) * 0.5 -> L1 = 0.0 ... L5 = 2.0. A maximal EEG tier alone
    # cannot reach the top composite band; corroboration is required.
    points = (level - 1) * (weight / 4.0)
    return _term(
        "eeg_evidence", "EEG evidence tier", points, weight,
        f"EEG evidence Level {level} of 5 contributes "
        f"(level - 1) x {weight / 4.0:.2f}.",
        measurement=f"Level {level}")


def _heart_rate_term(vitals: dict, anchor) -> dict:
    weight = config.COMPOSITE_SEVERITY_WEIGHTS["ictal_heart_rate"]
    if not vitals or not vitals.get("available") or vitals.get("ratio") is None:
        return _term(
            "ictal_heart_rate", "Ictal heart-rate rise", 0.0, weight,
            vitals.get("reason") if vitals else
            "No ECG was available for this case.",
            available=False)
    ratio = float(vitals["ratio"])
    baseline = vitals.get("baseline_bpm")
    window = vitals.get("window_bpm")
    measurement = (f"{baseline:.0f} -> {window:.0f} bpm (x{ratio:.2f})"
                   if baseline and window else f"x{ratio:.2f}")

    # A heart-rate change across a seizure-free window is ordinary variability,
    # not ictal tachycardia. Report the real measurement, but score it zero so
    # normal autonomic drift cannot inflate a background case.
    if anchor is not None and not anchor.is_seizure:
        return _term(
            "ictal_heart_rate", "Heart-rate change (non-ictal window)",
            0.0, weight,
            f"Measured {measurement} across a seizure-free window; scored zero "
            "because this is ordinary variability, not an ictal rise.",
            measurement=measurement)

    for threshold, fraction, description in HEART_RATE_BANDS:
        if ratio >= threshold:
            return _term("ictal_heart_rate", "Ictal heart-rate rise",
                         weight * fraction, weight,
                         f"{description}: measured {measurement} from the "
                         "participant's own ECG.",
                         measurement=measurement)
    return _term("ictal_heart_rate", "Ictal heart-rate rise", 0.0, weight,
                 f"No meaningful heart-rate rise: measured {measurement}.",
                 measurement=measurement)


def _burden_term(profile: dict) -> dict:
    weight = config.COMPOSITE_SEVERITY_WEIGHTS["seizure_burden"]
    if not profile:
        return _term("seizure_burden", "Recorded seizure burden", 0.0, weight,
                     "No monitoring statistics were available.",
                     available=False)
    per_day = float(profile.get("seizures_per_24h", 0.0))
    count = int(profile.get("seizure_count", 0))
    hours = float(profile.get("monitored_hours", 0.0))
    measurement = f"{count} seizure(s) in {hours:.0f} h ({per_day:.2f}/24 h)"
    for threshold, fraction, description in BURDEN_BANDS:
        if per_day >= threshold:
            return _term("seizure_burden", "Recorded seizure burden",
                         weight * fraction, weight,
                         f"{description}: {measurement}.",
                         measurement=measurement)
    return _term("seizure_burden", "Recorded seizure burden", 0.0, weight,
                 f"No seizures were recorded during monitoring ({measurement}).",
                 measurement=measurement)


def _awareness_term(anchor) -> dict:
    weight = config.COMPOSITE_SEVERITY_WEIGHTS["impaired_awareness"]
    if anchor is None or not anchor.is_seizure:
        return _term("impaired_awareness", "Impaired awareness", 0.0, weight,
                     "The anchor window is seizure-free background.",
                     measurement="not applicable")
    if anchor.impaired_awareness:
        return _term("impaired_awareness", "Impaired awareness", weight, weight,
                     "The annotated event type records impaired awareness "
                     f"({anchor.event_type}).",
                     measurement="impaired")
    return _term("impaired_awareness", "Impaired awareness", 0.0, weight,
                 f"Awareness was retained ({anchor.event_type}).",
                 measurement="retained")


def _motor_term(anchor) -> dict:
    weight = config.COMPOSITE_SEVERITY_WEIGHTS["motor_semiology"]
    if anchor is None or not anchor.is_seizure:
        return _term("motor_semiology", "Motor semiology", 0.0, weight,
                     "The anchor window is seizure-free background.",
                     measurement="not applicable")
    if anchor.motor:
        kind = "hyperkinetic" if anchor.hyperkinetic else "motor"
        return _term("motor_semiology", "Motor semiology", weight, weight,
                     f"The annotated event type records {kind} semiology "
                     f"({anchor.event_type}).",
                     measurement=kind)
    return _term("motor_semiology", "Motor semiology", 0.0, weight,
                 f"Non-motor semiology ({anchor.event_type}).",
                 measurement="non-motor")


def _mri_term(donor: dict) -> dict:
    weight = config.COMPOSITE_SEVERITY_WEIGHTS["lesional_mri"]
    if not donor or not donor.get("available"):
        return _term("lesional_mri", "Structural MRI finding", 0.0, weight,
                     "No MRI donor was matched for this case.",
                     available=False)
    diagnosis = (donor.get("mri_diagnosis") or "").lower()
    histology = donor.get("histopathology")
    lesional = diagnosis in ("suspicion", "other") or (
        histology and histology != "n/a")
    where = " ".join(filter(None, [donor.get("hemisphere"), donor.get("lobe")]))
    if lesional:
        descriptor = (f"FCD type {histology}" if histology and histology != "n/a"
                      else "suspected focal cortical dysplasia")
        return _term("lesional_mri", "Structural MRI finding", weight, weight,
                     f"Matched donor {donor.get('subject_id')} reports "
                     f"{descriptor} ({where}).",
                     measurement=f"{descriptor} ({where})")
    return _term("lesional_mri", "Structural MRI finding", 0.0, weight,
                 f"Matched donor {donor.get('subject_id')} is MRI-negative.",
                 measurement="MRI-negative")


def compute_composite_severity(*, eeg_level, anchor, profile: dict,
                               vitals: dict, donor: dict) -> dict:
    """Grade a composite case 1-5 from its corroborating modalities.

    Returns the level, the total, and the full itemised breakdown. Terms whose
    input is missing contribute zero and are flagged ``available: False`` so a
    low grade caused by absent data is never mistaken for a reassuring one.
    """
    terms = [
        _eeg_term(eeg_level),
        _heart_rate_term(vitals, anchor),
        _burden_term(profile),
        _awareness_term(anchor),
        _motor_term(anchor),
        _mri_term(donor),
    ]

    total = sum(term["points"] for term in terms)
    attainable = sum(term["max_points"] for term in terms)
    available_max = sum(term["max_points"] for term in terms if term["available"])
    missing = [term["label"] for term in terms if not term["available"]]

    level = 5
    for index, bound in enumerate(config.COMPOSITE_SEVERITY_THRESHOLDS, start=1):
        if total <= bound:
            level = index
            break

    return {
        "level": int(level),
        "label": config.SEVERITY_LEVELS[level],
        "score": round(float(total), 3),
        "max_score": round(float(attainable), 3),
        "available_max_score": round(float(available_max), 3),
        "terms": terms,
        "missing_inputs": missing,
        "thresholds": list(config.COMPOSITE_SEVERITY_THRESHOLDS),
        "grading_basis": "additive multi-modal rule layer (itemised)",
        "caveat": (
            "Composite severity is a transparent rule-based summary over "
            "corroborating modalities, not a validated clinical severity scale "
            "and not a model output. It conveys no urgency or treatment "
            "recommendation on its own."
            + (f" Reduced confidence: no input for {', '.join(missing)}."
               if missing else "")),
    }
