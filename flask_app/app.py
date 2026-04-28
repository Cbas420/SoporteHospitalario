"""
Flask Application for Hospital AI System.
Professional interface for hospital workers to use AI predictions.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = Flask(__name__)

# Configuration
app.config["SECRET_KEY"] = "hospital-ai-secret-key-2026"

# API Endpoints (use environment variables for Docker compatibility)
RADIOLOGY_API = os.getenv("RADIOLOGY_API_URL", "http://localhost:8000")
CLINICAL_API = os.getenv("CLINICAL_API_URL", "http://localhost:8001")


# ============================================================================
# Health Check
# ============================================================================

@app.route("/health")
def health_check():
    """Check health of all services."""
    radio_health = {"status": "unavailable"}
    clinical_health = {"status": "unavailable"}

    try:
        resp = requests.get(f"{RADIOLOGY_API}/health", timeout=2)
        if resp.status_code == 200:
            radio_health = resp.json()
    except Exception:
        pass

    try:
        resp = requests.get(f"{CLINICAL_API}/clinical/health", timeout=2)
        if resp.status_code == 200:
            clinical_health = resp.json()
    except Exception:
        pass

    return jsonify({
        "radiology_api": radio_health,
        "clinical_api": clinical_health,
        "flask_app": {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    })


# ============================================================================
# Main Pages
# ============================================================================

@app.route("/")
def index():
    """Home page - Dashboard overview."""
    stats = {}
    recent_predictions = []
    alerts = []

    # Get stats from radiology API
    try:
        resp = requests.get(f"{RADIOLOGY_API}/stats", timeout=3)
        if resp.status_code == 200:
            stats = resp.json()
    except Exception:
        stats = {"error": "Radiology API not available"}

    # Get recent predictions
    try:
        resp = requests.get(f"{RADIOLOGY_API}/predictions/recent?limit=10", timeout=3)
        if resp.status_code == 200:
            recent_predictions = resp.json()
    except Exception:
        pass

    # Get alerts
    try:
        resp = requests.get(f"{RADIOLOGY_API}/alerts?limit=5", timeout=3)
        if resp.status_code == 200:
            alerts = resp.json()
    except Exception:
        pass

    return render_template(
        "index.html",
        stats=stats,
        recent_predictions=recent_predictions,
        alerts=alerts,
        active_page="home"
    )


@app.route("/radiology")
def radiology():
    """Radiology page - Upload X-rays for classification."""
    return render_template("radiology.html", active_page="radiology")


@app.route("/radiology/predict", methods=["POST"])
def radiology_predict():
    """Proxy to radiology API for prediction."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]
        patient_id = request.form.get("patient_id", "")

        files = {"file": (file.filename, file.stream, file.content_type)}
        data = {"patient_id": patient_id} if patient_id else {}

        resp = requests.post(
            f"{RADIOLOGY_API}/predict",
            files=files,
            data=data,
            timeout=30
        )

        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/clinical")
def clinical():
    """Clinical prediction page - Predict diseases from symptoms."""
    return render_template("clinical.html", active_page="clinical")


@app.route("/clinical/predict", methods=["POST"])
def clinical_predict():
    """Proxy to clinical API for prediction."""
    try:
        data = request.get_json()

        resp = requests.post(
            f"{CLINICAL_API}/clinical/predict",
            json=data,
            timeout=10
        )

        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/patients")
def patients():
    """Patient management page."""
    patients_list = []
    try:
        resp = requests.get(f"{RADIOLOGY_API}/patients?limit=100", timeout=3)
        if resp.status_code == 200:
            patients_list = resp.json()
    except Exception:
        pass

    return render_template("patients.html", patients=patients_list, active_page="patients")


@app.route("/api/patients")
def api_patients():
    """Get patients from radiology API."""
    try:
        limit = request.args.get("limit", 100)
        resp = requests.get(f"{RADIOLOGY_API}/patients?limit={limit}", timeout=3)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/alerts")
def api_alerts():
    """Get alerts from radiology API."""
    try:
        limit = request.args.get("limit", 50)
        resp = requests.get(f"{RADIOLOGY_API}/alerts?limit={limit}", timeout=3)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/resolve-alert", methods=["PATCH"])
def api_resolve_alert():
    """Resolve alert via radiology API."""
    try:
        data = request.get_json()
        alert_id = data.get("alert_id")
        if not alert_id:
            return jsonify({"error": "alert_id required"}), 400

        resp = requests.patch(
            f"{RADIOLOGY_API}/alerts/{alert_id}",
            timeout=5
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/alerts")
def alerts():
    """Alerts management page."""
    alerts_list = []
    try:
        resp = requests.get(f"{RADIOLOGY_API}/alerts?limit=50", timeout=3)
        if resp.status_code == 200:
            alerts_list = resp.json()
    except Exception:
        pass

    return render_template("alerts.html", alerts=alerts_list, active_page="alerts")


@app.route("/api-info")
def api_info():
    """API info page."""
    radio_info = {}
    clinical_info = {}

    try:
        resp = requests.get(f"{RADIOLOGY_API}/model/info", timeout=3)
        if resp.status_code == 200:
            radio_info = resp.json()
    except Exception:
        pass

    try:
        resp = requests.get(f"{CLINICAL_API}/clinical/info", timeout=3)
        if resp.status_code == 200:
            clinical_info = resp.json()
    except Exception:
        pass

    return render_template(
        "api_info.html",
        radio_info=radio_info,
        clinical_info=clinical_info,
        active_page="api_info"
    )


# ============================================================================
# API Proxy Endpoints (for dashboard AJAX calls)
# ============================================================================

@app.route("/api/stats")
def api_stats():
    """Get stats from radiology API."""
    try:
        resp = requests.get(f"{RADIOLOGY_API}/stats", timeout=3)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/recent-predictions")
def api_recent():
    """Get recent predictions."""
    try:
        limit = request.args.get("limit", 10)
        resp = requests.get(f"{RADIOLOGY_API}/predictions/recent?limit={limit}", timeout=3)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 503


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
