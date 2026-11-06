"""Flask API and static-page server for the CerviCare prototype."""

import json
import math
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory
from werkzeug.exceptions import BadRequest, UnsupportedMediaType

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.preprocessing import FEATURE_NAMES, FEATURE_SCHEMA

FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = ROOT / "model" / "random_forest_model.pkl"
METRICS_PATH = ROOT / "reports" / "model_metrics.json"
FEATURES_PATH = ROOT / "model" / "feature_importance.json"
PAGE_FILES = {
    "questionnaire": "questionnaire.html",
    "result": "result.html",
    "analysis": "analysis.html",
    "dashboard": "dashboard.html",
    "about": "about.html",
}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="/static")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Trained model not found. Run `python train_model.py` before making predictions."
        )
    return joblib.load(MODEL_PATH)


def read_json_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required project artifact is missing: {path.name}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def error_response(message, status_code, details=None):
    payload = {"success": False, "error": message}
    if details:
        payload["details"] = details
    return jsonify(payload), status_code


def validate_responses(responses):
    if not isinstance(responses, dict):
        return None, "The 'responses' value must be an object."

    expected = set(FEATURE_NAMES)
    provided = set(responses)
    missing = sorted(expected - provided)
    unknown = sorted(provided - expected)
    if missing:
        return None, f"Please answer or mark as unsure for every question. Missing: {', '.join(missing)}."
    if unknown:
        return None, f"Unrecognized questionnaire fields: {', '.join(unknown)}."

    cleaned = {}
    for feature in FEATURE_SCHEMA:
        name = feature["name"]
        value = responses[name]
        if value is None:
            if not feature.get("allow_unknown", False):
                return None, f"'{feature['label']}' is required."
            cleaned[name] = np.nan
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, f"'{feature['label']}' must be a number or null for unsure."
        if not math.isfinite(value):
            return None, f"'{feature['label']}' must be a finite number."

        if feature["type"] == "boolean":
            if value not in (0, 1):
                return None, f"'{feature['label']}' must be Yes, No, or unsure."
        else:
            if value < feature["min"] or value > feature["max"]:
                return None, (
                    f"'{feature['label']}' must be between {feature['min']} and "
                    f"{feature['max']}."
                )
            if feature["step"] == 1 and not float(value).is_integer():
                return None, f"'{feature['label']}' must be a whole number."
        cleaned[name] = float(value)

    return cleaned, None


@app.get("/")
def home_page():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/<page>")
def static_page(page):
    file_name = PAGE_FILES.get(page)
    if file_name is None:
        abort(404)
    return send_from_directory(FRONTEND_DIR, file_name)


@app.get("/charts/<path:filename>")
def chart_file(filename):
    return send_from_directory(ROOT / "charts", filename)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json()
    except (BadRequest, UnsupportedMediaType):
        return error_response("Request body must contain valid JSON.", 400)

    if not isinstance(payload, dict):
        return error_response("Request body must be a JSON object.", 400)
    responses, validation_error = validate_responses(payload.get("responses"))
    if validation_error:
        return error_response(validation_error, 400)

    try:
        model = load_model()
    except FileNotFoundError as error:
        return error_response(str(error), 503)
    except (OSError, EOFError, ValueError, ImportError, AttributeError, pickle.UnpicklingError) as error:
        app.logger.error("Unable to load the saved model: %s", error)
        return error_response("The saved model could not be loaded.", 503)

    try:
        sample = pd.DataFrame([responses], columns=FEATURE_NAMES)
        predicted_class = int(model.predict(sample)[0])
    except (ValueError, TypeError, AttributeError) as error:
        app.logger.error("Prediction failed: %s", error)
        return error_response("The assessment could not be processed. Please try again.", 500)

    classification = "positive biopsy class" if predicted_class == 1 else "negative biopsy class"
    return jsonify(
        {
            "success": True,
            "prediction": predicted_class,
            "classification": f"Model predicts {classification}.",
            "model": "Random Forest Classifier",
        }
    )


@app.get("/api/metrics")
def metrics():
    try:
        return jsonify(read_json_file(METRICS_PATH))
    except FileNotFoundError as error:
        return error_response(str(error), 503)
    except (OSError, json.JSONDecodeError) as error:
        app.logger.error("Unable to read model metrics: %s", error)
        return error_response("Model metrics are currently unavailable.", 500)


@app.get("/api/features")
def features():
    try:
        report = read_json_file(FEATURES_PATH)
    except FileNotFoundError as error:
        return error_response(str(error), 503)
    except (OSError, json.JSONDecodeError) as error:
        app.logger.error("Unable to read feature rankings: %s", error)
        return error_response("Feature rankings are currently unavailable.", 500)

    return jsonify(
        {
            "features": report["features"][:10],
            "questionnaire": FEATURE_SCHEMA,
            "feature_count": len(FEATURE_SCHEMA),
        }
    )


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return error_response("API endpoint not found.", 404)
    return send_from_directory(FRONTEND_DIR, "index.html"), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
