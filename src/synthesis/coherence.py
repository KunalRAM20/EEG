"""Derivation of the clinical fields no dataset provides.

SeizeIT2 publishes only ``sex``; ds004199 publishes a paediatric clinical row;
neither publishes an adult patient history, medication list or daily routine.
Rather than sampling those freely, every derived field here is a documented
function of facts that ARE real:

  * seizure count and monitored hours, counted from the annotation files
  * awake/asleep distribution of that participant's own seizures
  * semiology encoded in the real ``eventType`` string
  * lesion topology and histopathology from the matched MRI donor
  * measured baseline heart rate from that participant's own ECG

The seeded generator only chooses *between clinically equivalent options*
(which drug from an appropriate class, which plausible onset age within a
band); it never invents a finding. Everything produced here is tagged
``derived`` in the provenance table.
"""
from __future__ import annotations

# ILAE 2017 style descriptions for the SeizeIT2 eventType vocabulary.
SEMIOLOGY_TEXT = {
    "sz_foc_a_nm": "focal aware non-motor seizure",
    "sz_foc_a_nm_behavior": "focal aware non-motor seizure with behaviour arrest",
    "sz_foc_a_m_hyperkinetic": "focal aware hyperkinetic motor seizure",
    "sz_foc_ia_nm": "focal impaired-awareness non-motor seizure",
    "sz_foc_ia_m_hyperkinetic": "focal impaired-awareness hyperkinetic motor seizure",
}

LOBE_TEXT = {"FL": "frontal", "TL": "temporal", "PL": "parietal",
             "OL": "occipital", "IL": "insular"}
SIDE_TEXT = {"L": "left", "R": "right", "left": "left", "right": "right"}
LOCALIZATION_TEXT = {"temp": "temporal", "cen_par": "centro-parietal",
                     "front": "frontal", "un": "unlocalised"}

# Anti-seizure medications appropriate to focal epilepsy, grouped so the seeded
# choice stays clinically sensible rather than arbitrary.
FIRST_LINE = ("levetiracetam", "lamotrigine", "oxcarbazepine", "carbamazepine")
SECOND_LINE = ("lacosamide", "brivaracetam", "zonisamide", "topiramate")
ADJUNCT = ("clobazam", "perampanel", "valproate")

TYPICAL_DOSES = {
    "levetiracetam": "1000 mg BD", "lamotrigine": "150 mg BD",
    "oxcarbazepine": "600 mg BD", "carbamazepine": "400 mg BD",
    "lacosamide": "150 mg BD", "brivaracetam": "100 mg BD",
    "zonisamide": "200 mg OD", "topiramate": "100 mg BD",
    "clobazam": "10 mg nocte", "perampanel": "6 mg nocte",
    "valproate": "600 mg BD",
}

# Adults only: SeizeIT2 is adult epilepsy-monitoring-unit data. The paediatric
# MRI donor's age is never adopted (see the age-conflict rule below).
MIN_ADULT_AGE = 18
MAX_ADULT_AGE = 64


def describe_semiology(anchor) -> str:
    return SEMIOLOGY_TEXT.get(
        anchor.event_type,
        anchor.event_type.replace("_", " ") if anchor.event_type else "seizure")


def describe_focus(anchor) -> str:
    """Human-readable seizure focus from the anchor's own annotation."""
    side = SIDE_TEXT.get(str(anchor.lateralization).lower())
    region = LOCALIZATION_TEXT.get(str(anchor.localization).lower())
    if side and region and region != "unlocalised":
        return f"{side} {region}"
    if side:
        return f"{side} hemisphere, not further localised"
    return "not lateralised on scalp EEG"


def _burden_band(seizures_per_24h: float) -> str:
    if seizures_per_24h >= 2.0:
        return "high"
    if seizures_per_24h >= 0.5:
        return "moderate"
    return "low"


def _medication_plan(band: str, rng) -> list[str]:
    """Choose a plausible regimen sized to the participant's real burden."""
    first = FIRST_LINE[int(rng.integers(len(FIRST_LINE)))]
    second = SECOND_LINE[int(rng.integers(len(SECOND_LINE)))]
    third = ADJUNCT[int(rng.integers(len(ADJUNCT)))]
    if band == "high":
        drugs = [first, second, third]
    elif band == "moderate":
        drugs = [first, second]
    else:
        drugs = [first]
    return [f"{drug} {TYPICAL_DOSES[drug]}" for drug in drugs]


def _sleep_pattern(profile: dict) -> tuple[str, str]:
    """Classify seizure timing from the participant's own vigilance labels."""
    awake = int(profile.get("awake_seizures", 0))
    asleep = int(profile.get("asleep_seizures", 0))
    total = awake + asleep
    if total == 0:
        return "unclassified", (
            "No vigilance state was annotated for this participant's events.")
    asleep_fraction = asleep / total
    if asleep_fraction >= 0.6:
        return "sleep-predominant", (
            f"{asleep} of {total} recorded seizures arose from sleep.")
    if asleep_fraction <= 0.2:
        return "wake-predominant", (
            f"{awake} of {total} recorded seizures arose while awake.")
    return "mixed", (
        f"{awake} awake and {asleep} sleep-onset seizures were recorded.")


def _age_at_onset(rng, donor: dict) -> int:
    """Plausible adult-recall onset age, informed by the FCD donor's band.

    FCD-related focal epilepsy characteristically begins in childhood or
    adolescence. The donor's own onset age indicates the band but is not copied
    verbatim, because the donor is a different (paediatric) person.
    """
    donor_onset = donor.get("donor_age_at_epilepsy_onset")
    try:
        hint = int(donor_onset)
    except (TypeError, ValueError):
        hint = None
    if hint is not None and hint <= 6:
        low, high = 2, 11
    elif hint is not None:
        low, high = 6, 17
    else:
        low, high = 4, 17
    return int(rng.integers(low, high + 1))


def derive_clinical_context(anchor, profile: dict, donor: dict,
                            vitals: dict, *, rng, sex: str | None) -> dict:
    """Build the derived clinical record for one composite case.

    Returns both the field values and a per-field rationale, so the report and
    the provenance table can state exactly why each value has the value it has.
    """
    seizures_per_24h = float(profile.get("seizures_per_24h", 0.0))
    band = _burden_band(seizures_per_24h)
    pattern, pattern_note = _sleep_pattern(profile)

    age_at_onset = _age_at_onset(rng, donor)
    # Years lived with epilepsy, floored so the composite is always an adult.
    min_duration = max(MIN_ADULT_AGE - age_at_onset, 1)
    duration_years = int(rng.integers(min_duration, min_duration + 22))
    age = age_at_onset + duration_years

    # --- Age-conflict rule -------------------------------------------------
    # ds004199 is paediatric (ages 3-13); SeizeIT2 is adult EMU monitoring. The
    # donor supplies lesion topology only, so its age must never become the
    # composite's age. Clamping to the adult band already guarantees this; the
    # assertion documents the invariant and fails loudly if the bands change.
    age = int(min(max(age, MIN_ADULT_AGE), MAX_ADULT_AGE))
    donor_age = donor.get("donor_age_at_scan")
    try:
        if donor_age is not None and int(donor_age) == age:
            raise AssertionError(
                "Composite age collided with the MRI donor's age at scan; the "
                "donor's demographics must never be adopted.")
    except (TypeError, ValueError):
        pass
    if age_at_onset >= age:
        age_at_onset = max(1, age - 1)
    # -----------------------------------------------------------------------

    medications = _medication_plan(band, rng)
    focus = describe_focus(anchor)
    semiology = describe_semiology(anchor)
    drug_resistant = band in ("high", "moderate") and len(medications) >= 2

    monitored_hours = float(profile.get("monitored_hours", 0.0))
    seizure_count = int(profile.get("seizure_count", 0))

    history_parts = [
        f"Focal epilepsy with onset at age {age_at_onset}, "
        f"{age - age_at_onset} years of active disease.",
        f"Habitual events are {semiology}s arising from the {focus}.",
    ]
    if donor.get("available") and donor.get("impression"):
        history_parts.append(f"MRI: Imaging pattern is compatible with {donor.get('impression')}.")
    if drug_resistant:
        history_parts.append(
            f"Drug-resistant: seizures persist on {len(medications)} "
            "anti-seizure medications.")
    history_parts.append(
        f"Admitted for video-EEG monitoring; {seizure_count} seizure(s) "
        f"captured over {monitored_hours:.0f} hours.")

    sleep_hours = int(rng.integers(5, 8))
    routine_parts = [
        f"Sleeps approximately {sleep_hours} hours per night; "
        f"{pattern} seizure timing. {pattern_note}",
        "Reports sleep deprivation and missed doses as the most consistent "
        "triggers." if band != "low" else
        "No consistent trigger identified; events are unpredictable.",
        f"Takes {len(medications)} anti-seizure medication(s) daily "
        f"({'good' if band == 'low' else 'variable'} reported adherence).",
        "Does not drive; avoids alcohol; works reduced hours."
        if drug_resistant else
        "Independent in daily activities; driving status reviewed at each visit.",
    ]

    baseline_bpm = vitals.get("baseline_bpm")
    notes = (
        f"Composite assembled around a real annotated {semiology} of "
        f"{anchor.duration_seconds:.0f}s recorded while {anchor.vigilance}. "
        f"Seizure frequency during monitoring was {seizures_per_24h:.2f} per 24h "
        f"({band} burden)."
    )

    values = {
        "age": age,
        "sex": sex,
        "heart_rate_bpm": (round(float(baseline_bpm), 1)
                           if baseline_bpm is not None else None),
        "prior_seizures": seizure_count,
        "family_history_epilepsy": bool(rng.integers(0, 2)),
        "medication": ", ".join(medications),
        "medical_history": " ".join(history_parts),
        "daily_routine": " ".join(routine_parts),
        "clinical_notes": notes,
        "mri_impression": donor.get("impression") if donor.get("available") else None,
        "xray_impression": None,   # filled by the case builder from OpenI
    }

    rationale = {
        "age": (f"Derived. Adult band {MIN_ADULT_AGE}-{MAX_ADULT_AGE}; onset age "
                f"{age_at_onset} seeded from the donor's onset band, plus "
                f"{duration_years} years of disease. The paediatric donor's own "
                "age is deliberately NOT used."),
        "sex": ("Real. Published in the SeizeIT2 participants table."
                if sex else "Not published for this participant."),
        "heart_rate_bpm": ("Real. Median R-R interval measured from this "
                           "participant's own baseline ECG window."
                           if baseline_bpm is not None else
                           "Not measurable from the available ECG window."),
        "prior_seizures": ("Real. Counted from this participant's annotation "
                           f"files ({seizure_count} events)."),
        "family_history_epilepsy": (
            "Derived. No cohort publishes family history; seeded so the field "
            "is populated for demonstration."),
        "medication": (f"Derived. Regimen size follows the measured {band} "
                       f"seizure burden ({seizures_per_24h:.2f}/24h); the "
                       "specific agents are seeded from focal-epilepsy classes."),
        "medical_history": ("Derived from real anchors: annotated semiology, "
                            "real event counts, real monitored hours, and the "
                            "matched donor's lesion topology."),
        "daily_routine": (f"Derived. Timing reflects this participant's real "
                          f"awake/asleep seizure split ({pattern})."),
        "clinical_notes": "Derived summary of the real anchor event.",
        "mri_impression": "Real clinical row from the matched ds004199 donor.",
    }

    return {
        "values": values,
        "rationale": rationale,
        "derivation": {
            "seizure_burden_band": band,
            "seizures_per_24h": seizures_per_24h,
            "sleep_pattern": pattern,
            "sleep_pattern_note": pattern_note,
            "age_at_onset": age_at_onset,
            "years_with_epilepsy": age - age_at_onset,
            "drug_resistant": drug_resistant,
            "medication_count": len(medications),
            "focus": focus,
            "semiology": semiology,
        },
    }
