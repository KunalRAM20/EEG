import io
import json
import os
import shutil
import unittest
from unittest.mock import patch

import numpy as np

import config
from webapp import db
from webapp.app import app


def _prediction():
    return {
        "seizure_probability": 0.91,
        "seizure_prediction": 1,
        "prediction_label": "Seizure",
        "decision_threshold": 0.4,
        "operating_point": "balanced",
        "severity": {
            "level": 5,
            "label": "Very high",
            "score": 0.91,
            "seizure_probability": 0.91,
            "intensity": 0.2,
            "grading_basis": "calibrated seizure-probability band",
        },
        "features": {"mean": 0.0},
        "feature_vector": np.zeros(44),
        "model_name": "TestModel",
        "data_source": "real-uci",
    }


class WebWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.old_db = config.DB_PATH
        self.old_uploads = config.UPLOADS_DIR
        fixture_root = os.path.join(config.ROOT_DIR, "tests")
        config.DB_PATH = os.path.join(fixture_root, ".web-workflow-test.db")
        config.UPLOADS_DIR = os.path.join(
            fixture_root, ".web-workflow-uploads")
        if os.path.exists(config.DB_PATH):
            os.remove(config.DB_PATH)
        shutil.rmtree(config.UPLOADS_DIR, ignore_errors=True)
        os.makedirs(config.UPLOADS_DIR, exist_ok=True)
        db.init_db()
        app.config.update(TESTING=True, SECRET_KEY="test-secret")
        self.client = app.test_client()
        self.client.get("/upload")
        with self.client.session_transaction() as session:
            self.csrf = session["_csrf_token"]

    def tearDown(self):
        if os.path.exists(config.DB_PATH):
            os.remove(config.DB_PATH)
        shutil.rmtree(config.UPLOADS_DIR, ignore_errors=True)
        config.DB_PATH = self.old_db
        config.UPLOADS_DIR = self.old_uploads

    def _patient(self, name="José Smith", *, legacy=False, segment=None):
        if segment is None:
            segment = np.zeros(178)
        modality = ({"age": 55, "sex": "Female"} if legacy else {
            "age": None,
            "sex": None,
            "prediction_supported": True,
            "data_provenance": {
                "EEG": "manual upload",
                "clinical_fields": "not provided",
                "model_inputs": ["EEG"],
            },
        })
        return db.create_patient(
            name, modality.get("age"), modality.get("sex"), "manual upload",
            "uploaded recording", segment, modality)

    def _model_patches(self):
        return (
            patch("webapp.app._model_meta", return_value={
                "data_source": "real-uci", "segment_length": 178}),
            patch("webapp.app.PREDICTOR.predict_segment",
                  return_value=_prediction()),
            patch("webapp.app.explain", return_value={
                "method": "test", "top_features": []}),
            patch("webapp.app.generate_report", return_value="test report"),
        )

    def test_unknown_name_creates_nothing(self):
        response = self.client.post("/upload", data={
            "_csrf_token": self.csrf, "patient_name": "Missing Person"})
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No data is provided with this name.", response.data)
        self.assertEqual(db.dashboard_stats()["predictions"], 0)

    def test_exact_casefolded_name_creates_report_without_new_patient(self):
        self._patient()
        patches = self._model_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            response = self.client.post("/upload", data={
                "_csrf_token": self.csrf, "patient_name": "JOSÉ SMITH"},
                follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Clinical Assessment Report", response.data)
        stats = db.dashboard_stats()
        self.assertEqual(stats["patients"], 1)
        self.assertEqual(stats["predictions"], 1)
        self.assertEqual(stats["reports"], 1)

    def test_duplicate_name_requires_patient_id(self):
        self._patient("Same Name")
        self._patient("same name")
        response = self.client.post("/upload", data={
            "_csrf_token": self.csrf, "patient_name": "SAME NAME"})
        self.assertIn(b"More than one registered patient", response.data)
        self.assertEqual(db.dashboard_stats()["predictions"], 0)

    def test_legacy_patient_is_not_reanalyzed(self):
        self._patient("Old Demo", legacy=True)
        response = self.client.post("/upload", data={
            "_csrf_token": self.csrf, "patient_name": "old demo"})
        self.assertIn(b"legacy demo record", response.data)
        self.assertEqual(db.dashboard_stats()["predictions"], 0)

    def test_manual_registration_does_not_invent_age_or_sex(self):
        eeg = "\n".join("0" for _ in range(178)).encode("ascii")
        with patch("webapp.app._model_meta", return_value={
                "data_source": "real-uci", "segment_length": 178}):
            response = self.client.post(
                "/patients/register",
                data={
                    "_csrf_token": self.csrf,
                    "case_source": "manual",
                    "name": "Registered Person",
                    "eeg_sampling_rate": "173.61",
                    "eeg_file": (io.BytesIO(eeg), "eeg.txt"),
                },
                content_type="multipart/form-data")
        self.assertEqual(response.status_code, 302)
        patient = db.find_patients_by_exact_name("registered person")[0]
        modality = json.loads(patient["modality_json"])
        self.assertIsNone(patient["age"])
        self.assertIsNone(patient["sex"])
        self.assertIsNone(modality["age"])
        self.assertIsNone(modality["sex"])
        self.assertEqual(len(json.loads(patient["segment_json"])), 178)

    def test_upload_route_accepts_eeg_file_without_name_lookup(self):
        eeg = "\n".join("0" for _ in range(178)).encode("ascii")
        with patch("webapp.app._model_meta", return_value={
                "data_source": "real-uci", "segment_length": 178}):
            response = self.client.post(
                "/upload",
                data={
                    "_csrf_token": self.csrf,
                    "eeg_file": (io.BytesIO(eeg), "sample.txt"),
                    "eeg_sampling_rate": "173.61",
                },
                content_type="multipart/form-data",
                follow_redirects=False)
        self.assertEqual(response.status_code, 302)
        self.assertGreaterEqual(db.dashboard_stats()["patients"], 1)

    def test_long_recording_creates_window_analysis_payload(self):
        long_segment = np.zeros(178 * 3)
        patient_id = self._patient("Windowed Case", segment=long_segment)
        patches = self._model_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            response = self.client.post(
                f"/predict/{patient_id}",
                data={"_csrf_token": self.csrf},
                follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Window-Level EEG Analysis", response.data)

        report = db.list_reports()[0]
        bundle = db.get_report_bundle(report["report_id"])
        xai = json.loads(bundle["xai_json"])
        self.assertTrue(xai.get("window_analysis"))
        self.assertIn("candidate_events", xai)

    def test_upload_page_offers_the_composite_generator(self):
        response = self.client.get("/patients/register")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Target Clinical Profile", response.data)
        self.assertIn(b"generate-composite", response.data)

    def test_generate_composite_rejects_an_unknown_target(self):
        response = self.client.post(
            "/patients/generate-composite",
            data={"_csrf_token": self.csrf, "composite_target": "critical"},
            follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"valid target profile", response.data)
        self.assertEqual(db.dashboard_stats()["patients"], 0)

    def test_generate_composite_requires_csrf(self):
        response = self.client.post(
            "/patients/generate-composite", data={"composite_target": "any"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(db.dashboard_stats()["patients"], 0)

    def test_generate_composite_registers_a_watermarked_case(self):
        segment = np.zeros(178 * 2)
        fake_case = {
            "seed": 4417, "target": "high", "name": "composite-4417",
            "segment": segment,
            "eeg_source": "SeizeIT2 sub-002 run-09 (composite)",
            "eeg_group": "seizure",
            "modality": {
                "age": 31, "sex": "male", "synthetic_composite": True,
                "composite_seed": 4417, "prediction_supported": True,
                "anchor": {"subject_id": "sub-002", "run": 9,
                           "kind": "seizure", "onset_seconds": 22216.0,
                           "event_type": "sz_foc_ia_m_hyperkinetic",
                           "lateralization": "left", "localization": "cen_par",
                           "vigilance": "awake", "impaired_awareness": True,
                           "motor": True, "hyperkinetic": True},
                "subject_profile": {"seizure_count": 15,
                                    "monitored_hours": 99.0,
                                    "seizures_per_24h": 3.63},
                "vitals": {"available": True, "ratio": 1.41,
                           "baseline_bpm": 67.0, "window_bpm": 95.0},
                "mri_donor": {"available": True, "subject_id": "sub-00043",
                              "mri_diagnosis": "suspicion",
                              "histopathology": "IIa", "hemisphere": "R",
                              "lobe": "FL", "match_basis": "hemisphere=R",
                              "images_available_locally": True},
                "modality_provenance": {
                    "EEG": {"status": "real", "same_person_as": "anchor",
                            "source": "SeizeIT2 sub-002 ses-01 run-09",
                            "detail": "Real EEG."},
                },
                "recording_metadata": {"format": "edf",
                                       "sampling_rate": 173.6092,
                                       "sample_count": int(segment.size),
                                       "channel_count": 1,
                                       "channel_names": ["BTEleft SD"]},
                "data_provenance": {"EEG": "SeizeIT2 (real)",
                                    "model_inputs": ["EEG"]},
            },
        }
        with patch("src.synthesis.case_builder.build_composite_case",
                   return_value=fake_case):
            response = self.client.post(
                "/patients/generate-composite",
                data={"_csrf_token": self.csrf, "composite_target": "high",
                      "composite_seed": "4417"},
                follow_redirects=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Registered patient", response.data)

        patient = db.find_patients_by_exact_name("composite-4417")[0]
        modality = json.loads(patient["modality_json"])
        self.assertTrue(modality["synthetic_composite"])
        self.assertEqual(modality["composite_seed"], 4417)

    def test_composite_report_shows_itemised_composite_severity(self):
        modality = {
            "synthetic_composite": True, "composite_seed": 99,
            "prediction_supported": True,
            "anchor": {"subject_id": "sub-002", "run": 9, "kind": "seizure",
                       "event_type": "sz_foc_ia_m_hyperkinetic",
                       "impaired_awareness": True, "motor": True,
                       "hyperkinetic": True, "onset_seconds": 1.0,
                       "lateralization": "left", "localization": "cen_par",
                       "vigilance": "awake"},
            "subject_profile": {"seizure_count": 15, "monitored_hours": 99.0,
                                "seizures_per_24h": 3.63},
            "vitals": {"available": True, "ratio": 1.41,
                       "baseline_bpm": 67.0, "window_bpm": 95.0},
            "mri_donor": {"available": True, "subject_id": "sub-00043",
                          "mri_diagnosis": "suspicion", "histopathology": "IIa",
                          "hemisphere": "R", "lobe": "FL"},
            "modality_provenance": {"EEG": {"status": "real",
                                            "same_person_as": "anchor",
                                            "detail": "Real EEG."}},
            "data_provenance": {"EEG": "SeizeIT2 (real)",
                                "model_inputs": ["EEG"]},
        }
        patient_id = db.create_patient(
            "composite-99", 31, "male", "SeizeIT2 (composite)", "seizure",
            np.zeros(178), modality)

        patches = self._model_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            response = self.client.post(
                f"/predict/{patient_id}",
                data={"_csrf_token": self.csrf}, follow_redirects=True)

        self.assertEqual(response.status_code, 200)

        bundle = db.get_report_bundle(db.list_reports()[0]["report_id"])
        composite = json.loads(bundle["xai_json"])["composite_severity"]
        self.assertEqual(composite["level"], 5)
        self.assertTrue(composite["terms"])
        self.assertAlmostEqual(
            composite["score"],
            round(sum(t["points"] for t in composite["terms"]), 3), places=3)

    def test_uploaded_record_is_not_watermarked(self):
        patient_id = self._patient("Genuine Upload")
        patches = self._model_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            response = self.client.post(
                f"/predict/{patient_id}",
                data={"_csrf_token": self.csrf}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"SYNTHETIC COMPOSITE", response.data)
        self.assertNotIn(b"Composite clinical severity", response.data)

    def test_nonfinite_clinical_number_is_rejected(self):
        response = self.client.post(
            "/patients/register",
            data={
                "_csrf_token": self.csrf,
                "case_source": "manual",
                "name": "Invalid Numbers",
                "heart_rate_bpm": "NaN",
            },
            follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"must be a finite number", response.data)
        self.assertEqual(db.dashboard_stats()["patients"], 0)

    def test_synthetic_bonn_fallback_cannot_be_registered(self):
        fake_dataset = (
            np.zeros((1, 178)), np.array([0]), np.array(["Z"]),
            np.array(["Z-0"]), "synthetic")
        with patch("webapp.app._model_meta", return_value={
                "data_source": "real-uci", "segment_length": 178}), \
                patch("webapp.app.config.RESEARCH_SANDBOX_ENABLED", True), \
                patch("webapp.app.load_eeg_segments",
                      return_value=fake_dataset), \
                patch.dict("webapp.app._CACHE", {}, clear=True):
            response = self.client.post(
                "/patients/register",
                data={
                    "_csrf_token": self.csrf,
                    "case_source": "bonn_sample",
                    "group": "random",
                },
                follow_redirects=True)
        self.assertIn(b"synthetic fallback data cannot be registered",
                      response.data)
        self.assertEqual(db.dashboard_stats()["patients"], 0)

    def test_delete_confirmation_and_cascade(self):
        patient_id = self._patient("Delete Me")
        prediction = _prediction()
        prediction.pop("feature_vector")
        prediction["xai"] = {"method": "test", "top_features": []}
        _, report_id = db.create_prediction_and_report(
            patient_id, prediction, "report")
        db.create_validation(report_id, "Dr Test", "approve", "", "")

        wrong = self.client.post(
            f"/patients/{patient_id}/delete",
            data={"_csrf_token": self.csrf, "confirmation": "999"})
        self.assertEqual(wrong.status_code, 302)
        self.assertIsNotNone(db.get_patient(patient_id))

        deleted = self.client.post(
            f"/patients/{patient_id}/delete",
            data={"_csrf_token": self.csrf,
                  "confirmation": str(patient_id)})
        self.assertEqual(deleted.status_code, 302)
        self.assertIsNone(db.get_patient(patient_id))
        self.assertEqual(db.dashboard_stats()["patients"], 0)

    def test_failed_upload_removal_stays_in_retry_queue(self):
        patient_id = self._patient("Locked Upload")
        patient_dir = os.path.join(config.UPLOADS_DIR, str(patient_id))
        os.makedirs(patient_dir, exist_ok=True)
        stored_path = os.path.join(patient_dir, "locked.bin")
        with open(stored_path, "wb") as handle:
            handle.write(b"test")
        db.create_case_file(
            patient_id, "EEG", "locked.bin", stored_path,
            "application/octet-stream", 4, "user upload")

        with patch("webapp.app.os.remove",
                   side_effect=PermissionError("file is locked")):
            response = self.client.post(
                f"/patients/{patient_id}/delete",
                data={"_csrf_token": self.csrf,
                      "confirmation": str(patient_id)})

        self.assertEqual(response.status_code, 302)
        self.assertIsNone(db.get_patient(patient_id))
        queued = db.list_pending_file_deletions()
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0]["stored_path"], stored_path)
        self.assertIn("locked", queued[0]["last_error"])

    def test_legacy_report_is_corrected_for_display_and_read_only(self):
        patient_id = self._patient("Old Report", legacy=True)
        prediction = _prediction()
        prediction.pop("feature_vector")
        prediction["severity"] = {
            "level": 4, "label": "Critical", "score": 1.0}
        prediction["xai"] = {"method": "legacy", "top_features": []}
        _, report_id = db.create_prediction_and_report(
            patient_id, prediction, "UNVERIFIED BODY")

        response = self.client.get(f"/report/{report_id}")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Level 4", response.data)

        validation = self.client.get(
            f"/validate/{report_id}", follow_redirects=True)
        self.assertEqual(validation.status_code, 200)

    def test_missing_csrf_is_rejected(self):
        response = self.client.post(
            "/upload", data={"patient_name": "Anyone"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
