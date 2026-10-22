import io
import os
import unittest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from app.main import app, EVIDENCE_DIR


class DetectionPipelineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_analyze_invalid_content_type(self):
        """Upload with invalid content type returns HTTP 400."""
        fake_file = io.BytesIO(b"not an image file")
        response = self.client.post(
            "/api/detections/analyze",
            files={"file": ("report.pdf", fake_file, "application/pdf")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("valid image", response.json().get("detail", "").lower())

    def test_02_analyze_corrupted_image_bytes(self):
        """Upload with corrupted/invalid image bytes returns HTTP 400."""
        corrupted_file = io.BytesIO(b"corrupted binary noise data")
        response = self.client.post(
            "/api/detections/analyze",
            files={"file": ("bad_image.jpg", corrupted_file, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("unable to decode", response.json().get("detail", "").lower())

    def test_03_analyze_mock_inference_mode(self):
        """Upload with use_mock=True executes mock inference without claiming genuine AI.

        Verifies upload validation, response schema, evidence creation, and database persistence.
        """
        # Create a synthetic image in memory
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(test_img, (320, 240), 50, (200, 200, 200), -1)
        _, encoded_img = cv2.imencode(".jpg", test_img)
        img_bytes = io.BytesIO(encoded_img.tobytes())

        response = self.client.post(
            "/api/detections/analyze",
            files={"file": ("test_frame.jpg", img_bytes, "image/jpeg")},
            data={
                "detection_source": "panel_review_test",
                "conf_threshold": "0.25",
                "use_mock": "true",
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Check explicit mock transparency
        self.assertTrue(data.get("is_mock_inference"))
        self.assertEqual(data.get("model_status"), "mock_inference_mode")

        # Verify required response structure
        self.assertIn("violation", data)
        self.assertIn("helmet_classification", data)
        self.assertIn("number_plate_detection", data)
        self.assertIn("ocr_output", data)
        self.assertIn("evidence_path", data)
        self.assertIn("summary", data)

        # Verify mock mode returns empty helmet detection list (no arbitrary fabricated bounding boxes)
        self.assertEqual(len(data["helmet_classification"]), 0)
        self.assertIn("[Simulated Test]", data["summary"])

        # Verify database record persistence
        violation = data["violation"]
        self.assertIn("id", violation)
        self.assertIn("created_at", violation)
        self.assertIn("panel_review_test", violation["detection_source"])

        # Verify evidence image was written to disk and is statically retrievable
        evidence_url = data["evidence_path"]
        self.assertTrue(evidence_url.startswith("/evidence/"))
        filename = os.path.basename(evidence_url)
        disk_path = os.path.join(EVIDENCE_DIR, filename)
        self.assertTrue(os.path.exists(disk_path))

        static_resp = self.client.get(evidence_url)
        self.assertEqual(static_resp.status_code, 200)
        self.assertEqual(static_resp.headers.get("content-type"), "image/jpeg")

    def test_04_analyze_live_pipeline_execution(self):
        """Test model pipeline execution with sample image."""
        sample_path = os.path.abspath(
            os.path.join(EVIDENCE_DIR, "..", "..", "output_annotated.jpg")
        )
        if os.path.exists(sample_path):
            with open(sample_path, "rb") as f:
                content = f.read()

            response = self.client.post(
                "/api/detections/analyze",
                files={"file": ("sample.jpg", io.BytesIO(content), "image/jpeg")},
                data={
                    "detection_source": "camera_cctv_01",
                    "conf_threshold": "0.2",
                    "use_mock": "false",
                },
            )
            self.assertEqual(response.status_code, 200)
            data = response.json()

            # Ensure response structure is consistent
            self.assertIn("violation", data)
            self.assertIn("helmet_classification", data)
            self.assertIn("number_plate_detection", data)
            self.assertIn("ocr_output", data)
            self.assertFalse(data.get("is_mock_inference"))
            self.assertIn("violation", data)
            self.assertIsNotNone(data["violation"]["id"])


if __name__ == "__main__":
    unittest.main()
