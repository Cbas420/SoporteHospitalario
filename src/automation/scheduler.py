"""
Automatizacion de alertas, informes y procesamiento de imagenes entrantes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from apscheduler.schedulers.blocking import BlockingScheduler

from config.settings import (
    ALERTS_AUDIT_PATH,
    AUTOMATION_INPUT_DIR,
    AUTOMATION_INTERVAL_SECONDS,
    AUTOMATION_PROCESSED_DIR,
    AUTOMATION_REPORTS_DIR,
    COVID_CRITICAL_THRESHOLD,
    COVID_WARNING_THRESHOLD,
    PREDICTIONS_AUDIT_PATH,
)
from src.api.inference import load_model, predict_single
from src.data.pipeline import prepare_data_pipeline, seed_repository_from_pipeline
from src.data.repository import HospitalRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_handle:
        file_handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def build_alerts_from_prediction(
    prediction: dict[str, Any],
    patient_id: Optional[str] = None,
    image_path: Optional[str] = None,
    source: str = "automation",
) -> list[dict[str, Any]]:
    """
    Genera alertas clinicas segun SDD §04:
    - COVID19 con confianza > COVID_CRITICAL_THRESHOLD (85%) → severity='critical'
    - COVID19 con confianza entre COVID_WARNING_THRESHOLD (60%) y 85% → severity='warning'
    - Baja confianza general → severity='medium' (requiere revision manual)
    """
    alerts = []
    predicted_class = prediction.get("prediction", "")
    confidence = float(prediction.get("confidence", 0.0))
    probabilities = prediction.get("probabilities", {})
    covid_prob = float(probabilities.get("COVID19", confidence if predicted_class == "COVID19" else 0.0))

    if predicted_class == "COVID19" and covid_prob >= COVID_CRITICAL_THRESHOLD:
        alerts.append(
            {
                "patient_id": patient_id,
                "image_path": image_path,
                "alert_type": "covid19_critical",
                "severity": "critical",
                "message": (
                    f"CRITICO: COVID-19 detectado con confianza {covid_prob:.1%}. "
                    "Activar protocolo de aislamiento inmediato y confirmacion PCR."
                ),
                "source": source,
            }
        )
    elif predicted_class == "COVID19" and covid_prob >= COVID_WARNING_THRESHOLD:
        alerts.append(
            {
                "patient_id": patient_id,
                "image_path": image_path,
                "alert_type": "covid19_warning",
                "severity": "warning",
                "message": (
                    f"ADVERTENCIA: Posible COVID-19 con confianza {covid_prob:.1%}. "
                    "Revisar aislamiento preventivo y solicitar confirmacion."
                ),
                "source": source,
            }
        )
    elif covid_prob >= COVID_WARNING_THRESHOLD:
        # Prediccion no-COVID pero con probabilidad COVID relevante
        alerts.append(
            {
                "patient_id": patient_id,
                "image_path": image_path,
                "alert_type": "covid19_suspicion",
                "severity": "warning",
                "message": (
                    f"Probabilidad COVID-19 elevada ({covid_prob:.1%}) aunque clasificado "
                    f"como {predicted_class}. Revisar manualmente."
                ),
                "source": source,
            }
        )

    if prediction.get("requires_review") and predicted_class != "COVID19":
        alerts.append(
            {
                "patient_id": patient_id,
                "image_path": image_path,
                "alert_type": "manual_review",
                "severity": "medium",
                "message": (
                    f"Prediccion con baja confianza ({confidence:.1%}). "
                    "Requiere revision manual por radiologo."
                ),
                "source": source,
            }
        )

    return alerts


def persist_prediction_outcome(
    repository: HospitalRepository,
    prediction: dict[str, Any],
    patient_id: Optional[str] = None,
    image_path: Optional[str] = None,
    source: str = "api",
) -> dict[str, Any]:
    prediction_record = {
        "patient_id": patient_id,
        "image_path": image_path,
        "prediction": prediction["prediction"],
        "confidence": prediction["confidence"],
        "requires_review": prediction["requires_review"],
        "inference_time_ms": prediction["inference_time_ms"],
        "probabilities": prediction["probabilities"],
        "source": source,
        "created_at": datetime.now(timezone.utc),
    }
    repository.save_prediction(prediction_record)
    append_jsonl(PREDICTIONS_AUDIT_PATH, prediction_record)

    alerts = build_alerts_from_prediction(
        prediction=prediction,
        patient_id=patient_id,
        image_path=image_path,
        source=source,
    )
    for alert in alerts:
        alert["created_at"] = datetime.now(timezone.utc)
        repository.save_alert(alert)
        append_jsonl(ALERTS_AUDIT_PATH, alert)

    return {
        "prediction": prediction_record,
        "alerts": alerts,
    }


def collect_incoming_images(input_dir: Path = AUTOMATION_INPUT_DIR) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def run_automation_cycle(
    input_dir: Path = AUTOMATION_INPUT_DIR,
    repository: Optional[HospitalRepository] = None,
) -> dict[str, Any]:
    repository = repository or HospitalRepository()
    repository.initialize()

    prepare_data_pipeline()
    seed_repository_from_pipeline(repository)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "processed_files": 0,
        "generated_alerts": 0,
        "results": [],
    }

    incoming_files = collect_incoming_images(input_dir)
    if not incoming_files:
        report["status"] = "idle"
        return report

    try:
        model = load_model()
    except FileNotFoundError as exc:
        report["status"] = "error"
        report["error"] = str(exc)
        return report

    AUTOMATION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in incoming_files:
        patient_id = image_path.stem
        prediction = predict_single(image_path, model=model)
        persisted = persist_prediction_outcome(
            repository=repository,
            prediction=prediction,
            patient_id=patient_id,
            image_path=str(image_path),
            source="automation",
        )
        processed_destination = AUTOMATION_PROCESSED_DIR / image_path.name
        shutil.move(str(image_path), str(processed_destination))

        report["processed_files"] += 1
        report["generated_alerts"] += len(persisted["alerts"])
        report["results"].append(
            {
                "patient_id": patient_id,
                "image_path": str(processed_destination),
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
                "requires_review": prediction["requires_review"],
                "alerts": persisted["alerts"],
            }
        )

    report["status"] = "completed"
    report_path = (
        AUTOMATION_REPORTS_DIR
        / f"automation_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    report["report_path"] = str(report_path)
    logger.info("Ciclo de automatizacion completado: %s", report_path)
    return report


def build_scheduler(
    interval_seconds: int = AUTOMATION_INTERVAL_SECONDS,
) -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        run_automation_cycle,
        "interval",
        seconds=interval_seconds,
        max_instances=1,
        coalesce=True,
    )
    return scheduler


def main() -> None:
    parser = argparse.ArgumentParser(description="Scheduler de automatizacion hospitalaria")
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Ejecuta un ciclo y termina",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=AUTOMATION_INTERVAL_SECONDS,
        help="Segundos entre ejecuciones del scheduler",
    )
    args = parser.parse_args()

    repository = HospitalRepository()
    repository.initialize()

    if args.run_once:
        run_automation_cycle(repository=repository)
        return

    scheduler = build_scheduler(interval_seconds=args.interval)
    logger.info("Scheduler iniciado cada %s segundos", args.interval)
    scheduler.start()


if __name__ == "__main__":
    main()
