"""Interconnected composite-case synthesis.

The project's three cohorts (Bonn/UCI, SeizeIT2, OpenNeuro ds004199, OpenI) share
no patients, so no real record spans EEG, ECG, MRI, X-ray and clinical history.
This package assembles an *internally coherent composite* from real files only:

  * EEG / ECG / EMG / movement come from ONE real SeizeIT2 participant at ONE
    real annotated moment, so those four are genuinely same-person, same-second.
  * The MRI donor is drawn from ds004199 by matching the anchor's real seizure
    lateralization and localization, and is labelled a matched donor, never the
    same person.
  * The chest X-ray is a single unrelated-cohort OpenI case, labelled as such.
  * Demographics, history, medication and daily routine are DERIVED from those
    real anchors by documented rules, never sampled freely.

Every composite is stamped ``synthetic_composite`` and carries a per-modality
provenance table. It is a demonstration and development artefact, and must never
be presented, exported or interpreted as a real patient.
"""

from .anchors import (Anchor, anchor_pool, available_subjects, subject_events,
                      subject_profile)
from .case_builder import build_composite_case
from .coherence import derive_clinical_context
from .linkage import match_mri_donor, xray_context
from .signals import (extract_window, heart_rate, movement_intensity,
                      to_model_eeg)

__all__ = [
    "Anchor",
    "anchor_pool",
    "available_subjects",
    "subject_events",
    "subject_profile",
    "build_composite_case",
    "derive_clinical_context",
    "match_mri_donor",
    "xray_context",
    "extract_window",
    "heart_rate",
    "movement_intensity",
    "to_model_eeg",
]
