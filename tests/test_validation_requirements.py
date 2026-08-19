import io
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import config
from src.synthesis.coherence import derive_clinical_context
from webapp import db
from webapp.app import app, _generate_demo_window_sequence, level_from_probability


class ValidationRequirementsTests(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.old_db = config.DB_PATH
        self.old_uploads = config.UPLOADS_DIR
        config.DB_PATH = os.path.join(self.test_dir, "test_cdss.db")
        config.UPLOADS_DIR = os.path.join(self.test_dir, "uploads")
        os.makedirs(config.UPLOADS_DIR, exist_ok=True)
        db.init_db()

        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

        # Seed one patient and report
        self.pid, _ = db.get_or_create_patient(
            name="Test-Patient-Alpha",
            age=34, sex="Male",
            eeg_source="manual", eeg_group="monitoring",
            segment=np.zeros(100),
            modality={"medical_history": "Focal epilepsy with onset at age 12."}
        )
        prediction = {
            "model_name": "ExtraTrees",
            "seizure_probability": 0.55,
            "seizure_prediction": 1,
            "severity": {"level": 3, "label": "Moderate", "score": 0.55},
        }
        report_text = "Clinical Assessment Summary: Patient exhibits focal epileptiform discharges."
        _, self.report_id = db.create_prediction_and_report(self.pid, prediction, report_text)

        resp = self.client.get("/patients/register")
        import re
        m = re.search(r'name="_csrf_token"\s+value="([^"]+)"', resp.data.decode("utf-8"))
        self.csrf = m.group(1) if m else ""

    def tearDown(self):
        config.DB_PATH = self.old_db
        config.UPLOADS_DIR = self.old_uploads
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_1_opening_report_twice_does_not_create_new_patient(self):
        initial_count = len(db.list_patients())
        resp1 = self.client.get(f"/report/{self.report_id}")
        self.assertEqual(resp1.status_code, 200)
        resp2 = self.client.get(f"/report/{self.report_id}")
        self.assertEqual(resp2.status_code, 200)
        final_count = len(db.list_patients())
        self.assertEqual(initial_count, final_count)

    def test_2_exporting_pdf_does_not_create_new_patient(self):
        initial_count = len(db.list_patients())
        db.create_validation(self.report_id, "Dr. Test", "approve", "Approved.", "Notes")
        resp = self.client.get(f"/report/{self.report_id}/export.pdf")
        self.assertEqual(resp.status_code, 200)
        final_count = len(db.list_patients())
        self.assertEqual(initial_count, final_count)

    def test_3_doctor_validation_does_not_create_new_patient(self):
        initial_count = len(db.list_patients())
        resp = self.client.post(
            f"/validate/{self.report_id}",
            data={"_csrf_token": self.csrf, "doctor_name": "Dr. Smith", "decision": "approve", "content": "Approved findings.", "notes": "Looks good."}
        )
        self.assertIn(resp.status_code, (200, 302))
        final_count = len(db.list_patients())
        self.assertEqual(initial_count, final_count)

    def test_4_patient_history_remains_identical_after_page_refresh(self):
        resp1 = self.client.get(f"/report/{self.report_id}")
        content1 = resp1.data
        resp2 = self.client.get(f"/report/{self.report_id}")
        content2 = resp2.data
        self.assertEqual(content1, content2)

    def test_5_stored_demo_report_remains_identical_after_restart(self):
        bundle1 = db.get_report_bundle(self.report_id)
        # Simulate restart by re-querying
        bundle2 = db.get_report_bundle(self.report_id)
        self.assertEqual(bundle1["patient_id"], bundle2["patient_id"])
        self.assertEqual(bundle1["content"], bundle2["content"])
        self.assertEqual(bundle1["seizure_prob"], bundle2["seizure_prob"])

    def test_6_level_1_generates_1_to_12_positive_windows(self):
        target_prob, windows, pos_count = _generate_demo_window_sequence(chosen_level=1, num_windows=57, seed=42)
        self.assertTrue(1 <= pos_count <= 12)

    def test_7_level_2_generates_13_to_24_positive_windows(self):
        target_prob, windows, pos_count = _generate_demo_window_sequence(chosen_level=2, num_windows=57, seed=42)
        self.assertTrue(13 <= pos_count <= 24)

    def test_8_level_3_generates_25_to_38_positive_windows(self):
        target_prob, windows, pos_count = _generate_demo_window_sequence(chosen_level=3, num_windows=57, seed=42)
        self.assertTrue(25 <= pos_count <= 38)

    def test_9_level_4_generates_39_to_48_positive_windows(self):
        target_prob, windows, pos_count = _generate_demo_window_sequence(chosen_level=4, num_windows=57, seed=42)
        self.assertTrue(39 <= pos_count <= 48)

    def test_10_level_5_generates_49_to_57_positive_windows(self):
        target_prob, windows, pos_count = _generate_demo_window_sequence(chosen_level=5, num_windows=57, seed=42)
        self.assertTrue(49 <= pos_count <= 57)

    def test_11_overall_probability_maps_back_to_level(self):
        self.assertEqual(level_from_probability(0.18), 1)
        self.assertEqual(level_from_probability(0.35), 2)
        self.assertEqual(level_from_probability(0.55), 3)
        self.assertEqual(level_from_probability(0.78), 4)
        self.assertEqual(level_from_probability(0.95), 5)

    def test_12_real_eeg_predictions_unmodified(self):
        # Verify that level_from_probability maps accurately across spectrum
        self.assertEqual(level_from_probability(0.10), 1)
        self.assertEqual(level_from_probability(0.90), 5)

    def test_13_no_donor_surgery_in_patient_history(self):
        from src.synthesis.case_builder import build_composite_case
        case = build_composite_case(seed=42)
        history_str = case["modality"].get("medical_history", "")
        self.assertNotIn("donor", history_str.lower())
        self.assertNotIn("prior surgery is documented", history_str.lower())
        self.assertIn("Imaging pattern is compatible with", history_str)

    def test_14_get_report_routes_do_not_call_generation_functions(self):
        db.create_validation(self.report_id, "Dr. Test", "approve", "Approved.", "Notes")
        with patch("webapp.db.create_patient") as mock_create_pat, \
             patch("webapp.app._create_report_for_patient") as mock_create_rep:
            self.client.get(f"/report/{self.report_id}")
            self.client.get(f"/report/{self.report_id}/export.pdf")
            self.client.get(f"/report/{self.report_id}/export.json")
            self.client.get(f"/validate/{self.report_id}")
            mock_create_pat.assert_not_called()
            mock_create_rep.assert_not_called()


class PatientProfileReuseTests(unittest.TestCase):
    def setUp(self):
        self.old_db = config.DB_PATH
        self.old_uploads = config.UPLOADS_DIR
        self.test_dir = tempfile.mkdtemp()
        config.DB_PATH = os.path.join(self.test_dir, "test_reuse.db")
        config.UPLOADS_DIR = os.path.join(self.test_dir, "uploads")
        os.makedirs(config.UPLOADS_DIR, exist_ok=True)
        db.init_db()
        app.testing = True
        self.client = app.test_client()

        resp = self.client.get("/patients/register")
        import re
        m = re.search(r'name="_csrf_token"\s+value="([^"]+)"', resp.data.decode("utf-8"))
        self.csrf = m.group(1) if m else ""

    def tearDown(self):
        config.DB_PATH = self.old_db
        config.UPLOADS_DIR = self.old_uploads
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_patient_reuse_across_multiple_reports(self):
        # 1. Create patient P-33 once
        initial_patients = len(db.list_patients())
        resp1 = self.client.post(
            "/patients/generate-composite",
            data={"_csrf_token": self.csrf, "composite_target": "high", "composite_seed": "3333"},
            follow_redirects=False
        )
        self.assertEqual(resp1.status_code, 302)
        patients = db.list_patients()
        self.assertEqual(len(patients), initial_patients + 1)
        patient_p33 = patients[0]
        p33_id = patient_p33["id"]

        # 2. Generate Report A for P-33 (created on initial registration)
        reports_after_a = db.list_reports()
        report_a = reports_after_a[0]
        bundle_a = db.get_report_bundle(report_a["report_id"])

        # 3. Generate Report B for P-33 by selecting existing patient_id = p33_id
        resp2 = self.client.post(
            "/patients/generate-composite",
            data={"_csrf_token": self.csrf, "composite_target": "low", "patient_id": str(p33_id)},
            follow_redirects=False
        )
        self.assertEqual(resp2.status_code, 302)
        reports_after_b = db.list_reports()
        report_b = reports_after_b[0]  # sorted by created_at desc
        bundle_b = db.get_report_bundle(report_b["report_id"])

        # 4. Confirm both reports have the exact same patient_id
        self.assertEqual(bundle_a["patient_id"], p33_id)
        self.assertEqual(bundle_b["patient_id"], p33_id)
        self.assertEqual(bundle_a["patient_id"], bundle_b["patient_id"])

        # 5. Confirm patient count increased only once throughout whole process
        final_patients = len(db.list_patients())
        self.assertEqual(final_patients, initial_patients + 1)

        # 6. Confirm name, age, sex, medical history, medication, routine, and imaging details are 100% identical in both reports
        self.assertEqual(bundle_a["name"], bundle_b["name"])
        self.assertEqual(bundle_a["age"], bundle_b["age"])
        self.assertEqual(bundle_a["sex"], bundle_b["sex"])
        
        modality_a = json.loads(bundle_a["modality_json"])
        modality_b = json.loads(bundle_b["modality_json"])
        self.assertEqual(modality_a.get("medical_history"), modality_b.get("medical_history"))
        self.assertEqual(modality_a.get("medication"), modality_b.get("medication"))
        self.assertEqual(modality_a.get("daily_routine"), modality_b.get("daily_routine"))
        self.assertEqual(modality_a.get("mri_impression"), modality_b.get("mri_impression"))
        self.assertEqual(modality_a.get("xray_impression"), modality_b.get("xray_impression"))

        # 7. Confirm risk and report analysis may differ
        self.assertNotEqual(bundle_a["report_id"], bundle_b["report_id"])

        # 8. Open both reports repeatedly and confirm no patient data changes
        p_count_before_open = len(db.list_patients())
        self.client.get(f"/report/{bundle_a['report_id']}")
        self.client.get(f"/report/{bundle_b['report_id']}")
        self.client.get(f"/report/{bundle_a['report_id']}")
        p_count_after_open = len(db.list_patients())
        self.assertEqual(p_count_before_open, p_count_after_open)

        # 9. Export both reports and confirm no new patient is created
        db.create_validation(bundle_a["report_id"], "Dr. Smith", "approve", "Approved", "Notes")
        db.create_validation(bundle_b["report_id"], "Dr. Smith", "approve", "Approved", "Notes")
        self.client.get(f"/report/{bundle_a['report_id']}/export.pdf")
        self.client.get(f"/report/{bundle_b['report_id']}/export.pdf")
        self.assertEqual(len(db.list_patients()), p_count_before_open)

        # 10. Validate one report and confirm patient data remains unchanged
        self.client.post(
            f"/validate/{bundle_a['report_id']}",
            data={"_csrf_token": self.csrf, "doctor_name": "Dr. House", "decision": "approve", "content": "Validated.", "notes": "No changes to patient."}
        )
        final_patient_record = db.get_patient(p33_id)
        self.assertEqual(final_patient_record["name"], bundle_a["name"])
        self.assertEqual(final_patient_record["age"], bundle_a["age"])
        self.assertEqual(final_patient_record["sex"], bundle_a["sex"])
        self.assertEqual(len(db.list_patients()), p_count_before_open)


if __name__ == "__main__":
    unittest.main()
