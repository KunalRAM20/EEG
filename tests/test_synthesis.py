"""Tests for interconnected composite-case synthesis.

These assert the properties that make a composite defensible rather than
decorative: that the same-person modalities really are same-person, that the
cross-cohort join is topologically justified, that the derivation is
reproducible, that the paediatric MRI donor's demographics never leak into the
composite patient, and that the synthetic marker survives all the way into the
exported JSON and PDF.

Tests that need real EDF payloads skip cleanly when the SeizeIT2 subset has not
been materialized locally, so the suite still runs on a fresh checkout.
"""
from __future__ import annotations

import json
import unittest

import numpy as np

import config
from src.models.composite_severity import compute_composite_severity
from src.reporting.structured_report import (as_json_bytes, as_pdf_bytes,
                                              build_report_schema)
from src.synthesis import anchors as anchor_module
from src.synthesis.case_builder import build_composite_case
from src.synthesis.linkage import (HEMISPHERE_MAP, LOCALIZATION_MAP,
                                   match_mri_donor, xray_context)


def _has_local_subset() -> bool:
    return bool(anchor_module.available_subjects())


requires_subset = unittest.skipUnless(
    _has_local_subset(),
    "No SeizeIT2 participant is materialized locally; composite cases need "
    "real EEG/ECG/EMG/movement payloads.")


class _FakeAnchor:
    """Minimal anchor stand-in for the pure-logic severity tests."""

    def __init__(self, *, is_seizure=True, impaired=True, motor=True,
                 hyperkinetic=True, lateralization="left",
                 localization="temp", event_type="sz_foc_ia_m_hyperkinetic"):
        self.is_seizure = is_seizure
        self.impaired_awareness = impaired
        self.motor = motor
        self.hyperkinetic = hyperkinetic
        self.lateralization = lateralization
        self.localization = localization
        self.event_type = event_type


# --------------------------------------------------------------------------- #
# Anchor discovery
# --------------------------------------------------------------------------- #
class AnchorDiscoveryTests(unittest.TestCase):
    @requires_subset
    def test_only_fully_materialized_subjects_are_offered(self):
        for subject in anchor_module.available_subjects():
            paths = None
            for anchor in anchor_module.anchor_pool(subject["id"]):
                paths = anchor_module.modality_paths(
                    anchor.subject_id, anchor.run)
                break
            self.assertIsNotNone(
                paths, f"{subject['id']} produced no usable anchor")
            self.assertIn("eeg", paths)

    @requires_subset
    def test_anchor_semiology_flags_match_event_type(self):
        for subject in anchor_module.available_subjects():
            for anchor in anchor_module.anchor_pool(subject["id"]):
                self.assertEqual(anchor.impaired_awareness,
                                 "_ia_" in anchor.event_type)
                self.assertEqual(anchor.motor, "_m_" in anchor.event_type)
                self.assertEqual(anchor.is_seizure,
                                 anchor.event_type.startswith("sz"))

    @requires_subset
    def test_profile_counts_come_from_annotation_files(self):
        profile = anchor_module.subject_profile("sub-002")
        self.assertGreater(profile["seizure_count"], 0)
        self.assertGreater(profile["monitored_hours"], 0)
        self.assertAlmostEqual(
            profile["seizures_per_24h"],
            profile["seizure_count"] / (profile["monitored_hours"] / 24.0),
            places=6)


# --------------------------------------------------------------------------- #
# Cross-cohort linkage
# --------------------------------------------------------------------------- #
class LinkageTests(unittest.TestCase):
    def test_donor_hemisphere_and_lobe_match_the_anchor(self):
        rng = np.random.default_rng(0)
        for lateralization in ("left", "right"):
            for localization in ("temp", "cen_par"):
                anchor = _FakeAnchor(lateralization=lateralization,
                                     localization=localization)
                donor = match_mri_donor(anchor, rng=rng)
                if not donor["available"]:
                    self.skipTest("ds004199 participants table is unavailable.")
                self.assertEqual(donor["hemisphere"],
                                 HEMISPHERE_MAP[lateralization],
                                 f"{lateralization}/{localization} donor is on "
                                 "the wrong side")
                self.assertIn(donor["lobe"], LOCALIZATION_MAP[localization],
                              f"{lateralization}/{localization} donor lobe "
                              f"{donor['lobe']} is not consistent")

    def test_donor_is_never_claimed_to_be_the_same_person(self):
        donor = match_mri_donor(_FakeAnchor(), rng=np.random.default_rng(1))
        if not donor["available"]:
            self.skipTest("ds004199 participants table is unavailable.")
        self.assertIsNone(donor["same_person_as"])
        self.assertEqual(donor["status"], "real-matched-donor")
        self.assertTrue(donor["match_basis"])

    def test_unlocalised_anchor_relaxes_to_hemisphere_only(self):
        anchor = _FakeAnchor(localization="un")
        donor = match_mri_donor(anchor, rng=np.random.default_rng(2))
        if not donor["available"]:
            self.skipTest("ds004199 participants table is unavailable.")
        self.assertEqual(donor["hemisphere"], "L")

    def test_xray_is_flagged_as_an_unrelated_cohort(self):
        xray = xray_context()
        self.assertIsNone(xray["same_person_as"])
        if xray["available"]:
            self.assertEqual(xray["status"], "real-unrelated-cohort")
            self.assertIn("no relationship", xray["cohort_note"])


# --------------------------------------------------------------------------- #
# Composite severity arithmetic
# --------------------------------------------------------------------------- #
class CompositeSeverityTests(unittest.TestCase):
    def _grade(self, **kwargs):
        defaults = {
            "eeg_level": 5,
            "anchor": _FakeAnchor(),
            "profile": {"seizures_per_24h": 3.0, "seizure_count": 15,
                        "monitored_hours": 99.0},
            "vitals": {"available": True, "ratio": 1.4,
                       "baseline_bpm": 67.0, "window_bpm": 95.0},
            "donor": {"available": True, "subject_id": "sub-00043",
                      "mri_diagnosis": "suspicion", "histopathology": "IIa",
                      "hemisphere": "R", "lobe": "FL"},
        }
        defaults.update(kwargs)
        return compute_composite_severity(**defaults)

    def test_terms_sum_to_the_reported_total(self):
        graded = self._grade()
        self.assertAlmostEqual(
            graded["score"],
            round(sum(t["points"] for t in graded["terms"]), 3), places=3)

    def test_maximum_case_reaches_level_five(self):
        graded = self._grade()
        self.assertEqual(graded["level"], 5)
        self.assertAlmostEqual(
            graded["max_score"],
            sum(config.COMPOSITE_SEVERITY_WEIGHTS.values()), places=3)

    def test_level_stays_within_one_to_five(self):
        for eeg_level in (None, 1, 2, 3, 4, 5):
            for vitals in ({}, {"available": True, "ratio": 0.9,
                                "baseline_bpm": 70.0, "window_bpm": 63.0}):
                graded = self._grade(eeg_level=eeg_level, vitals=vitals)
                self.assertGreaterEqual(graded["level"], 1)
                self.assertLessEqual(graded["level"], 5)

    def test_eeg_alone_cannot_reach_the_top_band(self):
        """Corroboration is required; a maximal EEG tier must not max the grade."""
        graded = self._grade(
            eeg_level=5,
            anchor=_FakeAnchor(is_seizure=False, impaired=False, motor=False),
            profile={"seizures_per_24h": 0.0, "seizure_count": 0,
                     "monitored_hours": 24.0},
            vitals={}, donor={})
        self.assertLess(graded["level"], 5)

    def test_background_window_does_not_score_heart_rate(self):
        """Ordinary autonomic drift must not read as ictal tachycardia."""
        graded = self._grade(
            anchor=_FakeAnchor(is_seizure=False, impaired=False, motor=False),
            vitals={"available": True, "ratio": 1.5,
                    "baseline_bpm": 60.0, "window_bpm": 90.0})
        term = next(t for t in graded["terms"] if t["name"] == "ictal_heart_rate")
        self.assertEqual(term["points"], 0.0)
        self.assertIn("non-ictal", term["label"].lower())

    def test_missing_inputs_are_flagged_not_silently_zeroed(self):
        graded = self._grade(vitals={}, donor={})
        self.assertTrue(graded["missing_inputs"])
        self.assertIn("Reduced confidence", graded["caveat"])
        unavailable = [t for t in graded["terms"] if not t["available"]]
        self.assertTrue(unavailable)
        for term in unavailable:
            self.assertEqual(term["points"], 0.0)


# --------------------------------------------------------------------------- #
# Whole-case assembly
# --------------------------------------------------------------------------- #
@requires_subset
class CompositeCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Building a case reads several real EDFs, so build once and share.
        cls.case = build_composite_case(seed=4417, target="high")

    def test_same_person_modalities_share_one_run_and_window(self):
        provenance = self.case["modality"]["modality_provenance"]
        anchor = self.case["modality"]["anchor"]
        expected_source = (f"SeizeIT2 {anchor['subject_id']} ses-01 "
                           f"run-{anchor['run']:02d}")
        same_person = [
            entry for name, entry in provenance.items()
            if name in {"EEG", "ECG", "EMG", "Movement"}
            and entry.get("status") == "real"
        ]
        self.assertGreaterEqual(len(same_person), 2)
        windows = set()
        for entry in same_person:
            self.assertEqual(entry["same_person_as"], "anchor")
            self.assertEqual(entry["source"], expected_source)
            windows.add(entry["window"])
        self.assertEqual(len(windows), 1,
                         "same-person modalities were cropped at different times")

    def test_cross_cohort_modalities_are_marked_different_person(self):
        provenance = self.case["modality"]["modality_provenance"]
        for name in ("MRI", "X-ray"):
            self.assertIsNone(provenance[name]["same_person_as"],
                              f"{name} must never claim to be the same person")

    def test_same_seed_reproduces_the_case_exactly(self):
        repeat = build_composite_case(seed=4417, target="high")
        np.testing.assert_array_equal(self.case["segment"], repeat["segment"])
        for field in ("age", "medication", "medical_history", "daily_routine"):
            self.assertEqual(self.case["modality"][field],
                             repeat["modality"][field])
        self.assertEqual(self.case["modality"]["mri_donor"]["subject_id"],
                         repeat["modality"]["mri_donor"]["subject_id"])

    def test_composite_age_never_adopts_the_paediatric_donor_age(self):
        for seed in (1, 2, 3, 5, 8):
            case = build_composite_case(seed=seed, target="any")
            donor = case["modality"]["mri_donor"]
            age = case["modality"]["age"]
            self.assertGreaterEqual(age, 18, "composite must be an adult")
            if donor.get("donor_age_at_scan"):
                self.assertNotEqual(
                    str(age), str(donor["donor_age_at_scan"]),
                    "the paediatric donor's age leaked into the composite")

    def test_eeg_is_resampled_onto_the_training_rate(self):
        metadata = self.case["modality"]["recording_metadata"]
        self.assertAlmostEqual(metadata["sampling_rate"],
                               config.SAMPLING_RATE, delta=0.01)
        self.assertEqual(metadata["channel_count"], 1)
        self.assertGreaterEqual(self.case["segment"].size, 178)
        self.assertTrue(np.all(np.isfinite(self.case["segment"])))

    def test_domain_shift_and_truncation_are_declared(self):
        eeg = self.case["modality"]["modality_provenance"]["EEG"]
        self.assertIn("out of distribution",
                      eeg["domain_shift"].lower().replace("-", " "))
        span = self.case["modality"]["analysis_span"]
        if span["truncated"]:
            self.assertIn("first", span["note"])
            self.assertLess(span["analysed_seconds"], span["full_event_seconds"])

    def test_record_is_marked_synthetic_and_not_legacy(self):
        from webapp.app import (_is_legacy_simulated_context,
                                _is_synthetic_composite)
        modality = self.case["modality"]
        self.assertTrue(_is_synthetic_composite(modality))
        self.assertFalse(
            _is_legacy_simulated_context(modality),
            "a composite carries full provenance and must not be treated as a "
            "legacy simulated demo record")

    def test_low_target_prefers_background_anchors(self):
        case = build_composite_case(seed=11, target="low")
        self.assertEqual(case["modality"]["anchor"]["kind"], "background")

    def test_rejects_an_unknown_target(self):
        with self.assertRaises(ValueError):
            build_composite_case(seed=1, target="critical")


# --------------------------------------------------------------------------- #
# Export watermarking
# --------------------------------------------------------------------------- #
class SyntheticExportTests(unittest.TestCase):
    def _bundle(self, modality: dict, xai: dict) -> dict:
        return {
            "report_id": 7, "patient_id": 3, "content": "draft report",
            "name": "composite-4417", "age": 31, "sex": "male",
            "model_name": "ExtraTrees", "seizure_prob": 0.72,
            "severity_level": 4, "severity_label": "High",
            "severity_method": "calibrated seizure-probability band",
            "xai_json": json.dumps(xai),
            "modality_json": json.dumps(modality),
        }

    def _validation(self) -> dict:
        return {"id": 2, "doctor_name": "Dr Test", "decision": "approve",
                "edited_content": "", "notes": "", "validated_at": "now"}

    def test_synthetic_marker_reaches_json_and_pdf(self):
        composite = {"level": 5, "label": "Very high", "score": 5.0,
                     "max_score": 5.5, "terms": [], "missing_inputs": [],
                     "thresholds": config.COMPOSITE_SEVERITY_THRESHOLDS,
                     "caveat": "test caveat"}
        schema = build_report_schema(
            self._bundle(
                {"synthetic_composite": True, "composite_seed": 4417,
                 "modality_provenance": {"EEG": {"status": "real"}}},
                {"composite_severity": composite}),
            self._validation(), [])

        self.assertEqual(schema["case"]["record_type"], "synthetic-composite")
        self.assertTrue(schema["synthetic_composite"]["is_synthetic"])
        self.assertEqual(schema["synthetic_composite"]["seed"], 4417)
        self.assertIn("EEG", schema["provenance"])
        self.assertIn("NO REAL PERSON", schema["limitations"][0])
        self.assertEqual(schema["composite_severity"]["level"], 5)

        payload = as_json_bytes(schema).decode("utf-8")
        self.assertIn("SYNTHETIC COMPOSITE", payload)
        pdf = as_pdf_bytes(schema)
        self.assertIn(b"CLINICAL DECISION-SUPPORT REPORT", pdf)
        self.assertTrue(pdf.startswith(b"%PDF"))

    def test_real_record_carries_no_synthetic_marker(self):
        schema = build_report_schema(
            self._bundle({"data_provenance": {"EEG": "manual upload"}}, {}),
            self._validation(), [])
        self.assertEqual(schema["case"]["record_type"], "clinical-record")
        self.assertNotIn("synthetic_composite", schema)
        self.assertNotIn("provenance", schema)
        self.assertIsNone(schema["composite_severity"])
        self.assertNotIn(b"SYNTHETIC", as_pdf_bytes(schema))

    def test_export_reads_a_row_like_bundle_without_get(self):
        """sqlite3.Row has no .get(); the schema builder must not rely on it."""
        class _RowLike:
            def __init__(self, data):
                self._data = data

            def __getitem__(self, key):
                try:
                    return self._data[key]
                except KeyError as exc:
                    raise IndexError(key) from exc

        bundle = _RowLike(self._bundle({"synthetic_composite": True}, {}))
        schema = build_report_schema(bundle, self._validation(), [])
        self.assertEqual(schema["case"]["record_type"], "synthetic-composite")


if __name__ == "__main__":
    unittest.main()
