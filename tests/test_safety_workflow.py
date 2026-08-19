import io
import json
import unittest

from src.processing import processing_manifest
from src.reporting.structured_report import as_pdf_bytes, build_report_schema
from webapp.file_security import validate_upload
from webapp.app import app
from werkzeug.datastructures import FileStorage


class SafetyWorkflowTests(unittest.TestCase):
    def test_health_and_security_headers(self):
        response = app.test_client().get("/health/live")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["status"], "ok")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertIn("frame-ancestors", response.headers["Content-Security-Policy"])

    def test_unsupported_modality_never_gets_findings(self):
        result = processing_manifest([{
            "id": 7, "modality": "MRI", "sha256": "a" * 64,
            "size_bytes": 8, "mime_type": "application/dicom",
        }])
        self.assertEqual(result[0]["status"], "contextual-only")
        self.assertEqual(result[0]["extracted_findings"], [])
        self.assertIsNone(result[0]["model_name"])

    def test_magic_byte_mismatch_is_rejected(self):
        upload = FileStorage(
            stream=io.BytesIO(b"not a pdf"), filename="report.pdf",
            content_type="application/pdf")
        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_upload(upload, "medical_report", 1000)

    def test_final_export_requires_doctor_approval(self):
        with self.assertRaisesRegex(ValueError, "doctor-approved"):
            build_report_schema({}, None, [])

    def test_pdf_and_json_share_structured_facts(self):
        bundle = {
            "report_id": 4, "patient_id": 9, "name": "Test", "age": 20,
            "sex": "Female", "content": "ORIGINAL AI DRAFT", "model_name": "M",
            "seizure_prob": .75, "severity_level": 4,
            "severity_label": "High",
            "severity_method": "calibrated seizure-probability band",
            "xai_json": json.dumps({
                "window_analysis": [{
                    "window_index": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 1.02,
                    "quality_status": "suitable",
                    "seizure_probability": 0.75,
                    "predicted_class": "Seizure",
                    "abstention_reason": None,
                }],
                "candidate_events": [{
                    "start_seconds": 0.0,
                    "end_seconds": 1.02,
                    "duration_seconds": 1.02,
                    "max_probability": 0.75,
                    "mean_probability": 0.75,
                    "supporting_windows": 1,
                }],
            }),
        }
        validation = {
            "id": 2, "doctor_name": "Dr Reviewer", "decision": "modify",
            "edited_content": "DOCTOR EDIT", "notes": "", "validated_at": "now",
        }
        report = build_report_schema(bundle, validation, [])
        self.assertTrue(report["report"]["ai_draft_preserved"])
        self.assertEqual(report["doctor_review"]["reviewed_content"], "DOCTOR EDIT")
        self.assertTrue(report["analysis"]["window_level_analysis"])
        self.assertTrue(report["analysis"]["candidate_model_evidence_events"])
        encoded = json.dumps(report)
        pdf = as_pdf_bytes(report)
        self.assertIn("report-4", encoded)
        self.assertTrue(pdf.startswith(b"%PDF-1.4"))
        self.assertIn(b"report-4", pdf)


if __name__ == "__main__":
    unittest.main()
