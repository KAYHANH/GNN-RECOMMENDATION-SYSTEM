from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from api.main import app


class FashionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_root_returns_api_metadata(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertEqual(payload["message"], "Fashion Recommender API is running.")
        self.assertIn("api", payload)
        self.assertIn("version", payload["api"])
        self.assertIn("environment", payload["api"])

    def test_healthcheck_returns_liveness(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("api", payload)

    def test_readiness_endpoint_reports_snapshot(self) -> None:
        response = self.client.get("/health/ready")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertIn(payload["status"], {"ready", "degraded"})
        self.assertIn("snapshot", payload)
        self.assertIn("engines", payload["snapshot"])
        self.assertIn("catalog", payload["snapshot"])

    def test_recommend_endpoint_returns_items_and_meta(self) -> None:
        response = self.client.get("/recommend/demo-customer?k=3&mode=hybrid")
        self.assertEqual(response.status_code, 200)
        self.assertIn("X-Request-ID", response.headers)
        self.assertIn("X-Process-Time-Ms", response.headers)

        payload = response.json()
        self.assertEqual(payload["customer_id"], "demo-customer")
        self.assertEqual(payload["mode"], "hybrid")
        self.assertLessEqual(len(payload["recommendations"]), 3)
        self.assertIn("meta", payload)
        self.assertIn("snapshot", payload["meta"])

    def test_search_endpoint_returns_matches_and_meta(self) -> None:
        response = self.client.get("/search?q=black%20dress&k=2")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertEqual(payload["query"], "black dress")
        self.assertLessEqual(len(payload["results"]), 2)
        self.assertIn("meta", payload)

    def test_explain_endpoint_returns_reasons_and_meta(self) -> None:
        response = self.client.get("/explain/demo-customer/0926246001")
        self.assertEqual(response.status_code, 200)

        payload = response.json()
        self.assertEqual(payload["customer_id"], "demo-customer")
        self.assertEqual(payload["article_id"], "0926246001")
        self.assertGreaterEqual(len(payload["reasons"]), 1)
        self.assertIn("meta", payload)


if __name__ == "__main__":
    unittest.main()
