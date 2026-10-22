import os
import unittest
from fastapi.testclient import TestClient

from app.main import app, EVIDENCE_DIR


class HelmetDetectionAPITestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_check(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("service"), "helmet-detection-api")

    def test_02_database_health(self):
        response = self.client.get("/api/health/database")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertEqual(data.get("database"), "connected")

    def test_03_create_detection(self):
        payload = {
            "number_plate": "GJ-01-XX-9999",
            "helmet_detected": False,
            "detection_source": "test_suite",
            "evidence_path": "evidence/sample_crop.jpg",
        }
        response = self.client.post("/api/detections", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("id", data)
        self.assertEqual(data["number_plate"], "GJ-01-XX-9999")
        self.assertFalse(data["helmet_detected"])
        self.assertEqual(data["review_status"], "pending")
        self.__class__.created_id = data["id"]

    def test_04_list_violations(self):
        response = self.client.get("/api/violations?page=1&page_size=10")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total", data)
        self.assertIn("items", data)
        self.assertGreaterEqual(data["total"], 1)
        self.assertIsInstance(data["items"], list)

    def test_05_filter_and_search_violations(self):
        # Search by number plate substring
        response = self.client.get("/api/violations?search=XX-9999")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["total"], 1)
        self.assertTrue(
            any(item["number_plate"] == "GJ-01-XX-9999" for item in data["items"])
        )

        # Filter by status
        response = self.client.get("/api/violations?review_status=pending")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for item in data["items"]:
            self.assertEqual(item["review_status"], "pending")

    def test_06_get_violation_by_id(self):
        target_id = getattr(self.__class__, "created_id", 1)
        response = self.client.get(f"/api/violations/{target_id}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], target_id)
        self.assertIn("number_plate", data)
        self.assertIn("created_at", data)

        # Nonexistent ID
        missing_resp = self.client.get("/api/violations/9999999")
        self.assertEqual(missing_resp.status_code, 404)

    def test_07_update_violation_status(self):
        target_id = getattr(self.__class__, "created_id", 1)

        # Update to approved
        response = self.client.patch(
            f"/api/violations/{target_id}/status",
            json={"review_status": "approved"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["review_status"], "approved")

        # Update to rejected
        response = self.client.patch(
            f"/api/violations/{target_id}/status",
            json={"review_status": "rejected"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["review_status"], "rejected")

        # Invalid status should return 422
        bad_response = self.client.patch(
            f"/api/violations/{target_id}/status",
            json={"review_status": "invalid_status_value"},
        )
        self.assertEqual(bad_response.status_code, 422)

        # Nonexistent ID should return 404
        missing_resp = self.client.patch(
            "/api/violations/9999999/status",
            json={"review_status": "approved"},
        )
        self.assertEqual(missing_resp.status_code, 404)

    def test_08_get_stats(self):
        response = self.client.get("/api/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_detections", data)
        self.assertIn("helmet_compliant", data)
        self.assertIn("violations", data)
        self.assertIn("pending_review", data)
        self.assertIn("approved_violations", data)
        self.assertIn("rejected_violations", data)
        self.assertGreaterEqual(data["total_detections"], 1)

    def test_09_evidence_static_serving(self):
        test_file_path = os.path.join(EVIDENCE_DIR, "test_evidence.txt")
        with open(test_file_path, "w") as f:
            f.write("sample evidence content")

        try:
            response = self.client.get("/evidence/test_evidence.txt")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.text, "sample evidence content")
        finally:
            if os.path.exists(test_file_path):
                os.remove(test_file_path)


if __name__ == "__main__":
    unittest.main()
