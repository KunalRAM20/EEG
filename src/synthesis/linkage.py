"""Cross-cohort linkage rules.

SeizeIT2, OpenNeuro ds004199 and OpenI share no patients. Rather than pretend
otherwise, this module performs an explicit, auditable *modality match*: it
selects an MRI donor whose recorded lesion topology is consistent with the
anchor participant's recorded seizure lateralization and localization, and it
emits the exact criteria used so a reviewer can check the join.

A matched donor is never claimed to be the same person. Every record produced
here carries ``same_person_as: None`` and a human-readable ``match_basis``.
"""
from __future__ import annotations

import csv
from pathlib import Path

from data.dataset_registry import (MRI_ROOT, OPENI_ROOT, REMOTE_DATASETS,
                                   _load_json, _report_section)

# SeizeIT2 lateralization -> ds004199 hemisphere column.
HEMISPHERE_MAP = {"left": "L", "right": "R"}

# SeizeIT2 localization -> acceptable ds004199 lobe codes.
# ds004199 lobes: FL frontal, TL temporal, PL parietal, OL occipital,
# IL insular; commas denote multi-lobar involvement.
LOCALIZATION_MAP = {
    "temp": ("TL", "TL,OL", "TL,PL"),
    "cen_par": ("PL", "FL,PL", "TL,PL"),
    "front": ("FL", "FL,PL"),
    "occ": ("OL", "TL,OL"),
    "ins": ("IL",),
}

# The only ds004199 subject whose imaging volumes are materialized locally.
LOCAL_IMAGE_SUBJECT = "sub-00043"

# Engel classification, used for the derived surgical-history narrative.
ENGEL_LABELS = {
    "IA": "seizure-free since surgery",
    "IB": "non-disabling auras only",
    "IC": "some disabling seizures after surgery, free for >=2 years",
    "ID": "generalised convulsions with drug withdrawal only",
    "IIA": "initially seizure-free, now rare disabling seizures",
    "IIB": "rare disabling seizures",
    "IIC": "more than rare disabling seizures, rare for >=2 years",
    "IIIA": "worthwhile seizure reduction",
    "IVB": "no worthwhile seizure reduction",
    "IVC": "seizures worse than before surgery",
}


def _participants() -> list[dict]:
    """Read the real ds004199 clinical table (BOM-prefixed, tab separated)."""
    path = MRI_ROOT / "participants.tsv"
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]
    except OSError:
        return []


def _clean(value) -> str:
    return str(value or "").strip()


def _has_local_images(subject_id: str) -> bool:
    anat = MRI_ROOT / subject_id / "anat"
    if not anat.is_dir():
        return False
    volumes = list(anat.glob("*.nii.gz"))
    # git-annex pointer stubs are ~107 bytes; real volumes are megabytes.
    return bool(volumes) and all(p.stat().st_size > 1000 for p in volumes)


def _local_image_files(subject_id: str) -> list[dict]:
    anat = MRI_ROOT / subject_id / "anat"
    if not anat.is_dir():
        return []
    files = []
    for path in sorted(anat.glob("*.nii.gz")):
        files.append({"name": path.name, "path": str(path),
                      "kind": "NIfTI", "size_bytes": path.stat().st_size})
    for path in sorted(anat.glob("*.json")):
        files.append({"name": path.name, "path": str(path),
                      "kind": "JSON sidecar", "size_bytes": path.stat().st_size})
    return files


def _candidate_lobes(localization: str) -> tuple[str, ...] | None:
    """Acceptable lobe codes, or None when the anchor does not localise."""
    key = _clean(localization).lower()
    if key in ("", "n/a", "un", "unknown"):
        return None
    return LOCALIZATION_MAP.get(key)


def match_mri_donor(anchor, *, rng, prefer_lesional: bool = True) -> dict:
    """Select an MRI donor consistent with the anchor's recorded semiology.

    Matching is progressively relaxed and the level actually used is reported:

      1. hemisphere AND lobe both match
      2. hemisphere matches (anchor did not localise, or no lobe match exists)
      3. any lesional subject (anchor gave neither side nor lobe)

    ``rng`` is the case's seeded generator, so the same seed always yields the
    same donor.
    """
    rows = [row for row in _participants() if _clean(row.get("group")) == "fcd"]
    if not rows:
        return {
            "available": False,
            "reason": "The ds004199 participants table could not be read.",
            "same_person_as": None,
        }

    hemisphere = HEMISPHERE_MAP.get(_clean(anchor.lateralization).lower())
    lobes = _candidate_lobes(anchor.localization)

    strict = [
        row for row in rows
        if (hemisphere is None or _clean(row.get("hemisphere")) == hemisphere)
        and (lobes is None or _clean(row.get("lobe")) in lobes)
    ]
    if strict and (hemisphere is not None or lobes is not None):
        pool = strict
        criteria = []
        if hemisphere is not None:
            criteria.append(f"hemisphere={hemisphere}")
        if lobes is not None:
            criteria.append(f"lobe in {{{', '.join(lobes)}}}")
        match_level = "hemisphere+lobe" if (hemisphere and lobes) else "partial"
    else:
        side_only = [
            row for row in rows
            if hemisphere is None or _clean(row.get("hemisphere")) == hemisphere
        ]
        pool = side_only or rows
        criteria = ([f"hemisphere={hemisphere}"] if hemisphere else
                    ["lesional cohort only"])
        match_level = "hemisphere-only" if side_only and hemisphere else "cohort-only"

    if not prefer_lesional:
        pool = [row for row in pool
                if _clean(row.get("mri_diagnosis")).lower() == "none"] or pool

    # Prefer the one subject whose actual volumes are on disk, so the case can
    # show a real image rather than only a clinical row.
    with_images = [row for row in pool
                   if _clean(row.get("participant_id")) == LOCAL_IMAGE_SUBJECT
                   and _has_local_images(LOCAL_IMAGE_SUBJECT)]
    chosen = with_images[0] if with_images else pool[int(rng.integers(len(pool)))]

    subject_id = _clean(chosen.get("participant_id"))
    images_local = _has_local_images(subject_id)
    outcome = _clean(chosen.get("latest_outcome"))
    operated = _clean(chosen.get("op")) == "1"

    return {
        "available": True,
        "same_person_as": None,
        "status": "real-matched-donor",
        "dataset": "OpenNeuro ds004199 (FCD-II epilepsy MRI)",
        "subject_id": subject_id,
        "source_url": REMOTE_DATASETS["epilepsy_mri"]["url"],
        "license": "CC0",
        "match_level": match_level,
        "match_basis": ", ".join(criteria) if criteria else "unconstrained",
        "matched_against": {
            "anchor_lateralization": anchor.lateralization,
            "anchor_localization": anchor.localization,
        },
        "candidates_considered": len(pool),
        "hemisphere": _clean(chosen.get("hemisphere")) or None,
        "lobe": _clean(chosen.get("lobe")) or None,
        "mri_diagnosis": _clean(chosen.get("mri_diagnosis")) or None,
        "histopathology": _clean(chosen.get("histopathology")) or None,
        "operated": operated,
        "engel_outcome": outcome if outcome not in ("", "n/a") else None,
        "engel_meaning": ENGEL_LABELS.get(outcome),
        # Donor demographics are reported as the DONOR's, never adopted as the
        # composite patient's — ds004199 is a paediatric cohort.
        "donor_sex": _clean(chosen.get("sex")) or None,
        "donor_age_at_scan": _clean(chosen.get("age_scan")) or None,
        "donor_age_at_epilepsy_onset": _clean(chosen.get("age_epilepsyonset")) or None,
        "donor_cohort_note": (
            "ds004199 is a paediatric FCD cohort (ages 3-13). This donor "
            "supplies lesion topology and histopathology only; its age and sex "
            "are NOT adopted by the composite patient."),
        "images_available_locally": images_local,
        "files": _local_image_files(subject_id) if images_local else [],
        "image_note": (
            "Real T1/FLAIR volumes and lesion ROI are available locally."
            if images_local else
            "Only the clinical row is local; this donor's imaging volumes are "
            "git-annex pointers that were never downloaded."),
    }


def _impression_text(anchor, donor: dict) -> str:
    """A radiology-style impression line assembled from the donor's real row."""
    if not donor.get("available"):
        return "No MRI is available for this case."
    side = {"L": "left", "R": "right"}.get(donor.get("hemisphere") or "", "")
    lobe_names = {"FL": "frontal", "TL": "temporal", "PL": "parietal",
                  "OL": "occipital", "IL": "insular"}
    lobes = ", ".join(
        lobe_names.get(part, part)
        for part in (donor.get("lobe") or "").split(",") if part)
    diagnosis = (donor.get("mri_diagnosis") or "").lower()
    if diagnosis == "none":
        finding = "No definite structural lesion identified on this study"
    elif diagnosis == "suspicion":
        finding = (f"Findings suspicious for focal cortical dysplasia in the "
                   f"{side} {lobes} region".strip())
    else:
        finding = f"Structural abnormality reported in the {side} {lobes} region".strip()
    histology = donor.get("histopathology")
    if histology and histology != "n/a":
        finding += f"; resection histopathology FCD type {histology}"
    return finding + "."


def mri_context(anchor, *, rng, prefer_lesional: bool = True) -> dict:
    """MRI donor plus a report-ready impression line."""
    donor = match_mri_donor(anchor, rng=rng, prefer_lesional=prefer_lesional)
    donor["impression"] = _impression_text(anchor, donor)
    return donor


def xray_context() -> dict:
    """The single local OpenI chest X-ray, always flagged unrelated-cohort.

    There is exactly one X-ray case on disk and it belongs to a general
    radiology collection, not an epilepsy cohort. It is included as realistic
    ancillary imaging and is explicitly excluded from severity scoring.
    """
    case_id = "CXR1151"
    case_root = OPENI_ROOT / case_id
    images = sorted(case_root.glob("*.png")) if case_root.is_dir() else []
    report = _load_json(case_root / "report.json", {})
    entries = report.get("list") or [{}]
    first = entries[0] if entries else {}
    abstract = first.get("abstract", "")

    available = len(images) >= 1 and all(p.stat().st_size > 1000 for p in images)
    return {
        "available": available,
        "same_person_as": None,
        "status": "real-unrelated-cohort",
        "dataset": "Indiana University / OpenI chest X-ray",
        "case_id": case_id,
        "source_url": REMOTE_DATASETS["openi_xray"]["url"],
        "license": "CC BY-NC-ND 4.0",
        "indication": _report_section(abstract, "Indication"),
        "findings": _report_section(abstract, "Findings"),
        "impression": first.get("impression"),
        "problem": first.get("Problems"),
        "files": [{"name": p.name, "path": str(p), "kind": "PNG X-ray",
                   "size_bytes": p.stat().st_size} for p in images],
        "cohort_note": (
            "This chest X-ray comes from a general radiology collection with no "
            "relationship to either epilepsy cohort. It is ancillary context "
            "only and contributes nothing to the severity grade."),
    }
