"""
Configuracion centralizada del modulo de IA clinica.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CLINICAL_DATA_DIR = Path(os.getenv("CLINICAL_DATA_DIR", str(PROJECT_ROOT / "clinical_ai" / "data")))
CLINICAL_MODELS_DIR = Path(os.getenv("CLINICAL_MODELS_DIR", str(PROJECT_ROOT / "clinical_ai" / "models")))

CLINICAL_DATABASE_URL = os.getenv(
    "CLINICAL_DATABASE_URL",
    "postgresql://clinical:clinical@localhost:5432/clinical_db"
)

CLINICAL_DB_HOST = os.getenv("CLINICAL_DB_HOST", "localhost")
CLINICAL_DB_PORT = int(os.getenv("CLINICAL_DB_PORT", "5432"))
CLINICAL_DB_NAME = os.getenv("CLINICAL_DB_NAME", "clinical_db")
CLINICAL_DB_USER = os.getenv("CLINICAL_DB_USER", "clinical")
CLINICAL_DB_PASSWORD = os.getenv("CLINICAL_DB_PASSWORD", "clinical")

SYMPTOM_COLUMNS = [
    "has_cough",
    "has_chest_pain",
    "has_fatigue",
    "has_fever",
    "has_dizziness",
    "has_breathing_difficulty",
    "has_headache",
    "has_nausea",
    "has_loss_of_appetite",
    "has_night_sweats",
]

DISEASE_COLUMNS = [
    "diabetes_risk",
    "hypertension_risk",
    "heart_disease_risk",
]

ALL_FEATURES = [
    "age",
    "sex",
    "height_cm",
    "weight_kg",
    "bmi",
] + SYMPTOM_COLUMNS

CLINICAL_API_PORT = int(os.getenv("CLINICAL_API_PORT", "8001"))
CLINICAL_API_HOST = os.getenv("CLINICAL_API_HOST", "0.0.0.0")

MODEL_PATH = CLINICAL_MODELS_DIR / "clinical_risk_classifier.keras"
MODEL_THRESHOLD = 0.5

for runtime_dir in (CLINICAL_DATA_DIR, CLINICAL_MODELS_DIR):
    runtime_dir.mkdir(parents=True, exist_ok=True)