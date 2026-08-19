"""
Five-level EEG evidence grading (Level 1 Very low -> Level 5 Very high).

IMPORTANT — this is a probability tier, not a clinical-severity label present
in the dataset. The calibrated seizure probability maps directly to five
non-overlapping bands, while electrographic intensity is reported separately as
context. Keeping intensity out of the tier prevents a 100% seizure probability
from being incorrectly reduced to Level 4.
"""
from __future__ import annotations

import numpy as np

import config


def _clip01(x: float) -> float:
    value = float(x)
    if not np.isfinite(value):
        raise ValueError("Severity inputs must be finite numbers.")
    return float(np.clip(value, 0.0, 1.0))


def intensity_score(feature_dict: dict) -> float:
    """
    Scale-invariant 0-1 intensity from high-frequency content, signal complexity
    and spectral spread. Features are extracted from filtered (non-z-scored)
    signals now, so intensity relies only on ratios / bounded spectral measures —
    never on absolute amplitude — to stay comparable across recordings.
    """
    beta = feature_dict.get("beta_rel_power", 0.0)
    gamma = feature_dict.get("gamma_rel_power", 0.0)
    mobility = feature_dict.get("hjorth_mobility", 0.0)
    complexity = feature_dict.get("hjorth_complexity", 1.0)
    edge90 = feature_dict.get("spectral_edge_90", 0.0)

    hf = _clip01(beta + gamma)                       # high-freq relative power
    mob = _clip01(mobility / 0.5)                    # ~dominant-frequency proxy
    comp = _clip01((complexity - 1.0) / 2.0)         # waveform irregularity
    edge = _clip01(edge90 / config.BANDPASS_HIGH)    # spectral spread (0-40 Hz)

    return _clip01(0.35 * hf + 0.25 * mob + 0.20 * comp + 0.20 * edge)


def compute_severity(seizure_prob: float, feature_dict: dict) -> dict:
    """
    Map calibrated seizure probability to a transparent five-level grade.

    Bands are 0-20% (L1), >20-40% (L2), >40-60% (L3), >60-80% (L4),
    and >80-100% (L5). Intensity is returned for context but cannot lower or
    raise the probability tier.
    """
    intensity = intensity_score(feature_dict)
    risk_score = _clip01(float(seizure_prob))

    level = 5
    for i, bound in enumerate(config.SEVERITY_THRESHOLDS, start=1):
        if risk_score <= bound:
            level = i
            break

    return {
        "level": int(level),
        "label": config.SEVERITY_LEVELS[level],
        # Persist the exact values used for grading. Presentation layers may
        # round them, but stored values must not cross a tier boundary merely
        # because (for example) 0.80001 is displayed as 0.8000.
        "score": risk_score,
        "seizure_probability": risk_score,
        "intensity": intensity,
        "grading_basis": "calibrated seizure-probability band",
    }
