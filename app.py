"""
Heart Disease Prediction — Flask API Server
=============================================
Loads the trained best_model.pkl (sklearn Pipeline with preprocessor)
and serves a prediction endpoint for the frontend.
"""

import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

# ============================================================
# App setup
# ============================================================
app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")

# Load model once at startup (Pipeline includes preprocessor)
print(f"Loading model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

# ============================================================
# Feature definitions for validation
# ============================================================
FEATURE_SCHEMA = {
    "age":        {"type": "int",   "min": 1,   "max": 120},
    "sex":        {"type": "int",   "min": 0,   "max": 1},
    "chest_pain": {"type": "int",   "min": 0,   "max": 3},
    "restbps":    {"type": "int",   "min": 50,  "max": 300},
    "chol":       {"type": "int",   "min": 50,  "max": 800},
    "fbs":        {"type": "int",   "min": 0,   "max": 1},
    "restecg":    {"type": "int",   "min": 0,   "max": 2},
    "thalach":    {"type": "int",   "min": 40,  "max": 250},
    "exang":      {"type": "int",   "min": 0,   "max": 1},
    "oldpeak":    {"type": "float", "min": 0.0, "max": 10.0},
    "slope":      {"type": "int",   "min": 0,   "max": 2},
    "ca":         {"type": "int",   "min": 0,   "max": 4},
    "thal":       {"type": "int",   "min": 0,   "max": 3},
}

FEATURE_ORDER = list(FEATURE_SCHEMA.keys())


def validate_input(data: dict) -> tuple[dict | None, str | None]:
    """Validate and cast input data. Returns (clean_data, error_message)."""
    clean = {}
    for feature, schema in FEATURE_SCHEMA.items():
        if feature not in data:
            return None, f"Thiếu trường: {feature}"

        raw = data[feature]
        try:
            if schema["type"] == "int":
                val = int(float(raw))
            else:
                val = float(raw)
        except (ValueError, TypeError):
            return None, f"Giá trị không hợp lệ cho '{feature}': {raw}"

        if val < schema["min"] or val > schema["max"]:
            return None, (
                f"'{feature}' phải nằm trong khoảng "
                f"[{schema['min']}, {schema['max']}], nhận được: {val}"
            )
        clean[feature] = val

    return clean, None


# ============================================================
# Routes
# ============================================================
@app.route("/")
def index():
    """Serve the frontend page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive JSON with 13 features, validate, predict, and return result.

    Request body:
        { "age": 52, "sex": 1, "chest_pain": 0, ... }

    Response:
        {
            "prediction": 0 or 1,
            "probability": 0.85,
            "message": "Có nguy cơ mắc bệnh tim" or "Không có nguy cơ..."
        }
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Request body phải là JSON hợp lệ"}), 400

    # Validate
    clean_data, error = validate_input(data)
    if error:
        return jsonify({"error": error}), 400

    # Build DataFrame with correct column order
    df = pd.DataFrame([clean_data], columns=FEATURE_ORDER)

    # Predict
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])  # probability of class 1 (disease)

    message = (
        "Có nguy cơ mắc bệnh tim"
        if prediction == 1
        else "Không có nguy cơ mắc bệnh tim"
    )

    return jsonify({
        "prediction": prediction,
        "probability": round(probability, 4),
        "message": message,
    })


# ============================================================
# Run
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
