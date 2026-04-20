"""
API principal del Sistema Inteligente de Soporte Hospitalario.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import src.api.inference as inference_service

from config.settings import API_HOST, API_PORT, CLASS_NAMES, MODELS_DIR, PIPELINE_REPORT_PATH
from src.api.inference import load_model, predict_single
from src.automation.scheduler import persist_prediction_outcome
from src.data.pipeline import prepare_data_pipeline, seed_repository_from_pipeline
from src.data.repository import HospitalRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)
repository = HospitalRepository()
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/bmp"}


class APIModel(BaseModel):
    model_config = {"protected_namespaces": ()}


class PredictionResponse(APIModel):
    prediction: str = Field(..., description="Clase predicha")
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict = Field(..., description="Probabilidades por clase")
    requires_review: bool
    inference_time_ms: float
    timestamp: str
    patient_id: Optional[str] = None


class HealthResponse(APIModel):
    status: str
    model_loaded: bool
    database_ready: bool
    timestamp: str


class ModelInfoResponse(APIModel):
    model_name: str
    version: str
    classes: list[str]
    model_path: str
    status: str


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_upload_file(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Tipo de archivo no soportado: {file.content_type}. "
                f"Usar: {', '.join(sorted(ALLOWED_TYPES))}"
            ),
        )


async def _predict_from_upload(
    file: UploadFile,
    patient_id: Optional[str] = None,
    source: str = "api",
) -> dict:
    _validate_upload_file(file)
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Archivo vacio")

    result = predict_single(image_bytes)
    result["timestamp"] = _utcnow_iso()
    result["patient_id"] = patient_id

    persist_prediction_outcome(
        repository=repository,
        prediction=result,
        patient_id=patient_id,
        image_path=file.filename,
        source=source,
    )
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Inicializando API hospitalaria...")
    repository.initialize()

    try:
        prepare_data_pipeline()
        seed_repository_from_pipeline(repository)
    except Exception as exc:
        logger.warning("No se pudo preparar el pipeline de datos: %s", exc)

    try:
        load_model()
        logger.info("Modelo cargado correctamente.")
    except Exception as exc:
        logger.warning("No se pudo cargar el modelo al arranque: %s", exc)

    yield

    logger.info("API hospitalaria detenida.")


app = FastAPI(
    title="Sistema Inteligente de Soporte Hospitalario",
    description=(
        "API FastAPI para clasificacion de radiografias, gestion basica de pacientes, "
        "estadisticas operativas y soporte a dashboard/automatizacion."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=inference_service._model_cache is not None,
        database_ready=True,
        timestamp=_utcnow_iso(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
):
    try:
        return PredictionResponse(
            **await _predict_from_upload(file=file, patient_id=patient_id, source="api")
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error en prediccion: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch")
async def predict_batch_endpoint(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        try:
            result = await _predict_from_upload(file=file, patient_id=None, source="batch")
            results.append(result)
        except HTTPException as exc:
            results.append({"file_name": file.filename, "error": exc.detail})
        except Exception as exc:
            results.append({"file_name": file.filename, "error": str(exc)})
    return {"count": len(results), "results": results}


@app.get("/patients")
async def get_patients(
    limit: int = Query(50, ge=1, le=500),
    diagnosis: Optional[str] = Query(None),
):
    return repository.list_patients(limit=limit, diagnosis=diagnosis)


@app.get("/stats")
async def get_stats():
    stats = repository.get_stats()
    stats["timestamp"] = _utcnow_iso()
    return stats


@app.get("/predictions/recent")
async def get_recent_predictions(limit: int = Query(20, ge=1, le=200)):
    return repository.recent_predictions(limit=limit)


@app.get("/patients/{patient_id}")
async def get_patient(patient_id: str):
    patient = repository.get_patient_by_id(patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail=f"Paciente '{patient_id}' no encontrado.")
    return patient


@app.get("/alerts")
async def get_alerts(limit: int = Query(20, ge=1, le=200)):
    return repository.recent_alerts(limit=limit)


@app.patch("/alerts/{alert_id}")
async def resolve_alert(alert_id: int):
    resolved = repository.resolve_alert(alert_id)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Alerta '{alert_id}' no encontrada.")
    return {"alert_id": alert_id, "resolved": True, "timestamp": _utcnow_iso()}


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    return ModelInfoResponse(
        model_name="chest_xray_classifier",
        version="2.0.0",
        classes=CLASS_NAMES,
        model_path=str(MODELS_DIR / "chest_xray_classifier.keras"),
        status="loaded" if inference_service._model_cache is not None else "not_loaded",
    )


@app.get("/evaluation")
async def get_evaluation():
    report_path = MODELS_DIR / "evaluation_report.json"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Informe de evaluacion no disponible. Ejecutar entrenamiento/evaluacion.",
        )

    with report_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


@app.get("/pipeline/report")
async def get_pipeline_report():
    if not PIPELINE_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Pipeline report no disponible.")
    with PIPELINE_REPORT_PATH.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
