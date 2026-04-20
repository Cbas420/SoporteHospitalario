"""
Tests basicos de la API FastAPI.
"""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

import src.api.app as app_module

from src.data.repository import HospitalRepository


def test_api_exposes_patients_stats_and_predict(monkeypatch):
    repository = HospitalRepository("sqlite:///:memory:")
    repository.initialize()
    repository.upsert_patients(
        [
            {
                "patient_id": "patient_001",
                "image_name": "patient_001.png",
                "image_path": "patient_001.png",
                "diagnosis": "Normal",
                "age": 34,
                "sex": "F",
                "admission_date": "2026-01-01",
                "source": "test",
            }
        ]
    )

    monkeypatch.setattr(app_module, "repository", repository)
    monkeypatch.setattr(app_module, "prepare_data_pipeline", lambda: {"status": "ready"})
    monkeypatch.setattr(app_module, "seed_repository_from_pipeline", lambda repo: 0)
    monkeypatch.setattr(app_module, "load_model", lambda: object())
    monkeypatch.setattr(
        app_module,
        "predict_single",
        lambda image_bytes: {
            "prediction": "COVID19",
            "confidence": 0.92,
            "probabilities": {"COVID19": 0.92, "Normal": 0.05, "Pneumonia": 0.03},
            "requires_review": True,
            "inference_time_ms": 42.0,
        },
    )
    monkeypatch.setattr(app_module.inference_service, "_model_cache", object())

    with TestClient(app_module.app) as client:
        health = client.get("/health")
        patients = client.get("/patients")
        stats_before = client.get("/stats")
        prediction = client.post(
            "/predict",
            files={"file": ("patient_001.png", b"fake-image", "image/png")},
            data={"patient_id": "patient_001"},
        )
        stats_after = client.get("/stats")
        alerts = client.get("/alerts")

    assert health.status_code == 200
    assert patients.status_code == 200
    assert len(patients.json()) == 1
    assert stats_before.json()["patients_total"] == 1
    assert prediction.status_code == 200
    assert prediction.json()["prediction"] == "COVID19"
    assert stats_after.json()["predictions_total"] == 1
    assert len(alerts.json()) >= 1
