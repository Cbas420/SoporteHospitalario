"""
Tests del flujo de automatizacion.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.automation import scheduler as scheduler_module
from src.data.repository import HospitalRepository


def test_run_automation_cycle_processes_incoming_files(monkeypatch, tmp_path):
    input_dir = tmp_path / "incoming"
    processed_dir = tmp_path / "processed"
    reports_dir = tmp_path / "reports"
    predictions_audit = tmp_path / "predictions.jsonl"
    alerts_audit = tmp_path / "alerts.jsonl"
    input_dir.mkdir()
    (input_dir / "case_001.png").write_bytes(b"fake-image")

    repository = HospitalRepository("sqlite:///:memory:")
    repository.initialize()

    monkeypatch.setattr(scheduler_module, "AUTOMATION_PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(scheduler_module, "AUTOMATION_REPORTS_DIR", reports_dir)
    monkeypatch.setattr(scheduler_module, "PREDICTIONS_AUDIT_PATH", predictions_audit)
    monkeypatch.setattr(scheduler_module, "ALERTS_AUDIT_PATH", alerts_audit)
    monkeypatch.setattr(scheduler_module, "prepare_data_pipeline", lambda: {"status": "ready"})
    monkeypatch.setattr(scheduler_module, "seed_repository_from_pipeline", lambda repo: 0)
    monkeypatch.setattr(scheduler_module, "load_model", lambda: object())
    monkeypatch.setattr(
        scheduler_module,
        "predict_single",
        lambda image_path, model=None: {
            "prediction": "COVID19",
            "confidence": 0.88,
            "probabilities": {"COVID19": 0.88, "Normal": 0.07, "Pneumonia": 0.05},
            "requires_review": False,
            "inference_time_ms": 55.0,
        },
    )

    report = scheduler_module.run_automation_cycle(
        input_dir=input_dir,
        repository=repository,
    )

    stats = repository.get_stats()
    assert report["status"] == "completed"
    assert report["processed_files"] == 1
    assert stats["predictions_total"] == 1
    assert processed_dir.joinpath("case_001.png").exists()
    assert predictions_audit.exists()
