"""
API REST para el modulo de prediccion clinica.
Endpoints para predecir riesgos de enfermedades basado en sintomas.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
from model.clinical_classifier import build_clinical_model, preprocess_clinical_input
from config.clinical_settings import (
    CLINICAL_API_HOST,
    CLINICAL_API_PORT,
    CLINICAL_MODELS_DIR,
    DISEASE_COLUMNS,
    MODEL_PATH,
)

model_cache: Optional[tf.keras.Model] = None


def load_clinical_model() -> tf.keras.Model:
    """Carga el modelo clinico."""
    global model_cache
    if model_cache is not None:
        return model_cache

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Modelo clinico no encontrado. Ejecutar train_clinical_model.py primero.",
        )

    model_cache = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    return model_cache


class ClinicalInput(BaseModel):
    age: int
    sex: str
    height_cm: float
    weight_kg: float
    has_cough: bool = False
    has_chest_pain: bool = False
    has_fatigue: bool = False
    has_fever: bool = False
    has_dizziness: bool = False
    has_breathing_difficulty: bool = False
    has_headache: bool = False
    has_nausea: bool = False
    has_loss_of_appetite: bool = False
    has_night_sweats: bool = False


class ClinicalPrediction(BaseModel):
    diabetes_risk: bool
    hypertension_risk: bool
    heart_disease_risk: bool
    probabilities: dict[str, float]
    confidence: float
    risk_level: str
    inference_time_ms: float
    timestamp: str


def predict_risk(
    clinical_data: ClinicalInput,
) -> ClinicalPrediction:
    """Realiza la prediccion de riesgos clinicos."""
    start = datetime.now(timezone.utc)

    model = load_clinical_model()

    symptoms = {
        "has_cough": clinical_data.has_cough,
        "has_chest_pain": clinical_data.has_chest_pain,
        "has_fatigue": clinical_data.has_fatigue,
        "has_fever": clinical_data.has_fever,
        "has_dizziness": clinical_data.has_dizziness,
        "has_breathing_difficulty": clinical_data.has_breathing_difficulty,
        "has_headache": clinical_data.has_headache,
        "has_nausea": clinical_data.has_nausea,
        "has_loss_of_appetite": clinical_data.has_loss_of_appetite,
        "has_night_sweats": clinical_data.has_night_sweats,
    }

    features = preprocess_clinical_input(
        age=clinical_data.age,
        sex=clinical_data.sex,
        height_cm=clinical_data.height_cm,
        weight_kg=clinical_data.weight_kg,
        symptoms=symptoms,
    )

    probabilities = model.predict(features, verbose=0)[0]

    prob_dict = {
        disease: float(prob)
        for disease, prob in zip(DISEASE_COLUMNS, probabilities)
    }

    diabetes_risk = bool(probabilities[0] >= 0.5)
    hypertension_risk = bool(probabilities[1] >= 0.5)
    heart_disease_risk = bool(probabilities[2] >= 0.5)

    confidence = float(max(probabilities))
    avg_prob = float(probabilities.mean())

    if avg_prob >= 0.6:
        risk_level = "alto"
    elif avg_prob >= 0.4:
        risk_level = "moderado"
    else:
        risk_level = "bajo"

    inference_time = (datetime.now(timezone.utc) - start).total_seconds() * 1000

    return ClinicalPrediction(
        diabetes_risk=diabetes_risk,
        hypertension_risk=hypertension_risk,
        heart_disease_risk=heart_disease_risk,
        probabilities=prob_dict,
        confidence=confidence,
        risk_level=risk_level,
        inference_time_ms=round(inference_time, 2),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


app = FastAPI(
    title="Clinical AI API",
    description="API para prediccion de riesgos clinicos",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/clinical/health")
async def health_check():
    """Health check del modulo clinico."""
    model_loaded = model_cache is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/clinical/info")
async def model_info():
    """Informacion del modelo clinico."""
    return {
        "model_name": "clinical_risk_classifier",
        "version": "1.0.0",
        "diseases": DISEASE_COLUMNS,
        "features": [
            "age",
            "sex",
            "height_cm",
            "weight_kg",
            "bmi (calculado)",
            "10 sintomas binarios",
        ],
        "model_path": str(MODEL_PATH),
        "status": "loaded" if model_cache is not None else "not_loaded",
    }


@app.post("/clinical/predict", response_model=ClinicalPrediction)
async def predict(
    data: ClinicalInput,
):
    """Predice riesgos clinicos en base a sintomas."""
    try:
        return predict_risk(data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    try:
        load_clinical_model()
        print(f"Modelo clinico cargado desde: {MODEL_PATH}")
    except Exception as e:
        print(f"ADVERTENCIA: No se pudo cargar el modelo: {e}")

    uvicorn.run(
        app,
        host=CLINICAL_API_HOST,
        port=CLINICAL_API_PORT,
        reload=False,
    )