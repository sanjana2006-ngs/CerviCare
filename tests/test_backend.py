import json
import joblib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from backend import app as backend
from backend.preprocessing import EXCLUDED_COLUMNS, FEATURE_NAMES, FEATURE_SCHEMA, TARGET_COLUMN


def valid_responses():
    values = {}
    for feature in FEATURE_SCHEMA:
        if feature["type"] == "boolean":
            values[feature["name"]] = 0
        else:
            values[feature["name"]] = feature["min"]
    return values


class BackendApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backend.app.config.update(TESTING=True)
        cls.client = backend.app.test_client()

    def test_health_page_and_static_routes(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.get_json(), {"status": "ok"})
        for route in ("/", "/questionnaire", "/result", "/analysis", "/dashboard", "/about"):
            with self.subTest(route=route):
                with self.client.get(route) as response:
                    self.assertEqual(response.status_code, 200)
        self.assertEqual(self.client.get("/api/unknown").get_json()["error"], "API endpoint not found.")

    def test_features_return_model_rankings_and_questionnaire(self):
        response = self.client.get("/api/features")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["features"]), min(10, len(FEATURE_SCHEMA)))
        self.assertEqual(len(payload["questionnaire"]), len(FEATURE_SCHEMA))
        self.assertEqual(
            [item["rank"] for item in payload["features"]],
            list(range(1, len(payload["features"]) + 1)),
        )
        self.assertGreaterEqual(
            payload["features"][0]["importance"],
            payload["features"][-1]["importance"],
        )

    def test_selected_features_exclude_targets_and_leakage_columns(self):
        self.assertNotIn(TARGET_COLUMN, FEATURE_NAMES)
        self.assertTrue(set(FEATURE_NAMES).isdisjoint(EXCLUDED_COLUMNS))
        self.assertEqual(
            FEATURE_NAMES,
            ["Age", "Smokes", "STDs"],
        )
        self.assertTrue(
            {
                "Number of sexual partners",
                "First sexual intercourse",
                "Num of pregnancies",
                "Hormonal Contraceptives",
                "Hormonal Contraceptives (years)",
                "IUD",
                "IUD (years)",
                "STDs (number)",
                "STDs: Number of diagnosis",
                "STDs:HIV",
                "STDs:HPV",
                "Smokes (years)",
                "Smokes (packs/year)",
            }.issubset(EXCLUDED_COLUMNS)
        )

    def test_questionnaire_has_only_dataset_supported_questions(self):
        self.assertEqual(
            [feature["name"] for feature in FEATURE_SCHEMA],
            ["Age", "Smokes", "STDs"],
        )
        self.assertEqual(
            [feature["question"] for feature in FEATURE_SCHEMA],
            [
                "What is your age?",
                "Do you currently smoke?",
                "Have you ever been diagnosed with a sexually transmitted infection (STI)?",
            ],
        )
        self.assertTrue(FEATURE_SCHEMA[1]["allow_unknown"])
        self.assertTrue(FEATURE_SCHEMA[-1]["allow_unknown"])

    def test_saved_model_and_preprocessor_use_exact_questionnaire_features(self):
        model = joblib.load(backend.MODEL_PATH)
        preprocessor = joblib.load(
            backend.ROOT / "model" / "preprocessing_pipeline.pkl"
        )
        self.assertEqual(list(model.feature_names_in_), FEATURE_NAMES)
        self.assertEqual(list(preprocessor.feature_names_in_), FEATURE_NAMES)
        self.assertEqual(
            set(preprocessor.get_feature_names_out()),
            set(FEATURE_NAMES),
        )

    def test_metrics_are_actual_saved_report(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        for metric in ("accuracy", "precision", "recall", "f1_score"):
            self.assertGreaterEqual(payload[metric], 0)
            self.assertLessEqual(payload[metric], 1)
        self.assertEqual(payload["metadata"]["target"], "Biopsy")
        self.assertEqual(payload["metadata"]["records"], 858)
        self.assertEqual(payload["metadata"]["original_features"], 36)
        self.assertEqual(
            payload["metadata"]["class_distribution"],
            {"negative": 803, "positive": 55},
        )
        self.assertGreaterEqual(payload["roc_auc"], 0)
        self.assertLessEqual(payload["roc_auc"], 1)
        self.assertGreaterEqual(payload["average_precision"], 0)
        self.assertLessEqual(payload["average_precision"], 1)
        self.assertEqual(len(payload["confusion_matrix"]), 2)
        self.assertEqual(len(payload["model_comparison"]), 3)

    def test_predict_returns_a_screening_classification(self):
        response = self.client.post(
            "/api/predict",
            json={"responses": valid_responses()},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertIn(payload["prediction"], (0, 1))
        self.assertEqual(
            payload["classification"],
            "Model predicts positive biopsy class."
            if payload["prediction"] == 1
            else "Model predicts negative biopsy class.",
        )

    def test_predict_class_matches_saved_model_for_multiple_answers(self):
        model = joblib.load(backend.MODEL_PATH)
        combinations = [
            {"Age": 13, "Smokes": 0, "STDs": 0},
            {"Age": 25, "Smokes": 0, "STDs": 0},
            {"Age": 35, "Smokes": 0, "STDs": 0},
            {"Age": 25, "Smokes": 1, "STDs": 0},
            {"Age": 25, "Smokes": 0, "STDs": 1},
        ]
        for responses in combinations:
            with self.subTest(responses=responses):
                sample = pd.DataFrame([responses], columns=FEATURE_NAMES)
                predicted_class = int(model.predict(sample)[0])
                response = self.client.post(
                    "/api/predict",
                    json={"responses": responses},
                )
                payload = response.get_json()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(payload["prediction"], predicted_class)

    def test_predict_uses_model_class_without_exposing_probability(self):
        class ReversedClassOrderModel:
            classes_ = [1, 0]

            def predict(self, sample):
                return [0]

        with patch.object(backend, "load_model", return_value=ReversedClassOrderModel()):
            response = self.client.post(
                "/api/predict",
                json={"responses": valid_responses()},
            )
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["prediction"], 0)
        self.assertEqual(payload["classification"], "Model predicts negative biopsy class.")
        self.assertNotIn("probability", payload)
        self.assertNotIn("positive_class_probability", payload)

    def test_predict_accepts_unknown_answers(self):
        responses = valid_responses()
        responses["Smokes"] = None
        responses["STDs"] = None
        response = self.client.post("/api/predict", json={"responses": responses})
        self.assertEqual(response.status_code, 200)

        responses["Age"] = None
        response = self.client.post("/api/predict", json={"responses": responses})
        self.assertEqual(response.status_code, 400)
        self.assertIn("is required", response.get_json()["error"])

    def test_predict_rejects_missing_invalid_and_unknown_fields(self):
        responses = valid_responses()
        missing = dict(responses)
        missing.pop("Age")
        self.assertEqual(
            self.client.post("/api/predict", json={"responses": missing}).status_code,
            400,
        )

        invalid = dict(responses)
        invalid["Age"] = 100
        self.assertEqual(
            self.client.post("/api/predict", json={"responses": invalid}).status_code,
            400,
        )

        wrong_type = dict(responses)
        wrong_type["Smokes"] = "yes"
        self.assertEqual(
            self.client.post("/api/predict", json={"responses": wrong_type}).status_code,
            400,
        )

        unknown = dict(responses, Biopsy=1)
        unknown["Number of sexual partners"] = 2
        self.assertEqual(
            self.client.post("/api/predict", json={"responses": unknown}).status_code,
            400,
        )

    def test_predict_rejects_malformed_json(self):
        response = self.client.post(
            "/api/predict",
            data="{",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.get_json()["success"])

    def test_model_missing_returns_clear_service_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.pkl"
            backend.load_model.cache_clear()
            with patch.object(backend, "MODEL_PATH", missing_path):
                response = self.client.post(
                    "/api/predict",
                    json={"responses": valid_responses()},
                )
            backend.load_model.cache_clear()
        self.assertEqual(response.status_code, 503)
        self.assertIn("Run `python train_model.py`", response.get_json()["error"])

    def test_missing_metrics_return_json_service_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "missing.json"
            with patch.object(backend, "METRICS_PATH", missing_path):
                response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["success"])

    def test_chart_files_are_served(self):
        for chart in (
            "confusion_matrix.png",
            "class_distribution.png",
            "model_comparison.png",
            "feature_importance.png",
            "roc_curve.png",
            "precision_recall_curve.png",
        ):
            with self.subTest(chart=chart):
                with self.client.get(f"/charts/{chart}") as response:
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.mimetype, "image/png")


if __name__ == "__main__":
    unittest.main()
