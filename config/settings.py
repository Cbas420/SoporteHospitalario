"""
Configuracion centralizada del Sistema Inteligente de Soporte Hospitalario.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
AUTOMATION_INPUT_DIR = DATA_DIR / "incoming"
AUTOMATION_PROCESSED_DIR = AUTOMATION_INPUT_DIR / "processed"
AUTOMATION_REPORTS_DIR = DATA_DIR / "automation_reports"
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(PROJECT_ROOT / "models")))

PATIENTS_CSV_PATH = Path(
    os.getenv("PATIENTS_CSV_PATH", str(RAW_DATA_DIR / "patients.csv"))
)
PATIENTS_CLEAN_PATH = PROCESSED_DATA_DIR / "patients_clean.csv"
PIPELINE_REPORT_PATH = PROCESSED_DATA_DIR / "pipeline_report.json"
DATASET_MANIFEST_PATH = PROCESSED_DATA_DIR / "dataset_manifest.json"
PREDICTIONS_AUDIT_PATH = DATA_DIR / "predictions.jsonl"
ALERTS_AUDIT_PATH = DATA_DIR / "alerts.jsonl"

SQLITE_DB_PATH = Path(os.getenv("SQLITE_DB_PATH", str(DATA_DIR / "hospital.db")))
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Buckets MinIO segun SDD §01 y §06
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "medical-images")          # retrocompatibilidad
MINIO_BUCKET_RAW = os.getenv("MINIO_BUCKET_RAW", "raw-xrays")
MINIO_BUCKET_MASKS = os.getenv("MINIO_BUCKET_MASKS", "lung-masks")
MINIO_BUCKET_PROCESSED = os.getenv("MINIO_BUCKET_PROCESSED", "processed-xrays")
MINIO_BUCKET_QUARANTINE = os.getenv("MINIO_BUCKET_QUARANTINE", "quarantine")
MINIO_BUCKET_REPORTS = os.getenv("MINIO_BUCKET_REPORTS", "reports")

MINIO_ALL_BUCKETS = [
    MINIO_BUCKET_RAW,
    MINIO_BUCKET_MASKS,
    MINIO_BUCKET_PROCESSED,
    MINIO_BUCKET_QUARANTINE,
    MINIO_BUCKET_REPORTS,
]

CLASS_NAMES = ["COVID19", "Normal", "Pneumonia"]
NUM_CLASSES = len(CLASS_NAMES)

PATIENT_ID_FIELD = "patient_id"
PATIENT_CANONICAL_FIELDS = [
    "patient_id",
    "image_name",
    "image_path",
    "diagnosis",
    "age",
    "sex",
    "admission_date",
    "source",
]
PATIENTS_CANONICAL_FIELDS = PATIENT_CANONICAL_FIELDS

IMG_SIZE = (224, 224)
IMG_CHANNELS = 3

AUGMENTATION_CONFIG = {
    "rotation_range": 15,
    "horizontal_flip": True,
    "vertical_flip": False,
    "zoom_range": 0.1,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "brightness_range": (0.9, 1.1),
    "fill_mode": "constant",
    "cval": 0,
}

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_MIN = 1e-7

MODEL_BACKBONE = "ResNet50"
FREEZE_BASE_LAYERS = True
UNFREEZE_FROM_LAYER = 140
DENSE_UNITS = 256
DROPOUT_RATE = 0.5

MODEL_FILENAME = "chest_xray_classifier.keras"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
MODEL_WEIGHTS_FILENAME = "chest_xray_classifier.weights.h5"
MODEL_WEIGHTS_PATH = MODELS_DIR / MODEL_WEIGHTS_FILENAME
TRAINING_HISTORY_PATH = MODELS_DIR / "training_history.json"

MAX_INFERENCE_TIME_MS = 1000
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{API_PORT}")
CONFIDENCE_THRESHOLD_LOW = 0.6
DASHBOARD_REFRESH_SECONDS = int(os.getenv("DASHBOARD_REFRESH_SECONDS", "10"))

# Umbrales clinicos para alertas COVID (SDD §04)
# COVID >85% → severidad "critical", COVID 60-85% → severidad "warning"
COVID_CRITICAL_THRESHOLD = float(os.getenv("COVID_CRITICAL_THRESHOLD", "0.85"))
COVID_WARNING_THRESHOLD = float(os.getenv("COVID_WARNING_THRESHOLD", "0.60"))
AUTOMATION_INTERVAL_SECONDS = int(
    os.getenv("AUTOMATION_INTERVAL_SECONDS", "300")
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_database_url() -> str:
    """Devuelve una URL de BD compatible con SQLAlchemy."""
    if not DATABASE_URL:
        return f"sqlite:///{SQLITE_DB_PATH.as_posix()}"
    if DATABASE_URL.startswith("postgresql://"):
        return DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
    return DATABASE_URL


for runtime_dir in (
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    AUTOMATION_INPUT_DIR,
    AUTOMATION_PROCESSED_DIR,
    AUTOMATION_REPORTS_DIR,
    MODELS_DIR,
    SQLITE_DB_PATH.parent,
):
    runtime_dir.mkdir(parents=True, exist_ok=True)
