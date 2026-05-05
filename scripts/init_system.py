"""
Inicializacion automatica del sistema hospitalario.

Ejecutado por el servicio hospital-init al hacer docker-compose up.
Garantiza que antes de arrancar la API existan:
  1. Buckets de MinIO inicializados
  2. Dataset de imagenes (sintetico si no hay dataset real)
  3. Modelo de radiografias entrenado

El modelo clinico (clinical_risk_classifier.keras) viene pre-entrenado
en el repositorio y no requiere inicializacion adicional.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "chest_xray_classifier.keras"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def log(msg: str) -> None:
    print(f"[hospital-init] {msg}", flush=True)


# ---------------------------------------------------------------------------
# 1. MinIO
# ---------------------------------------------------------------------------

def init_minio_buckets() -> bool:
    log("Inicializando buckets de MinIO...")
    result = subprocess.run(
        [sys.executable, "scripts/init_minio_buckets.py"],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        log("Buckets de MinIO listos.")
    else:
        log("ERROR: No se pudieron inicializar los buckets de MinIO.")
    return result.returncode == 0


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------

def has_dataset() -> bool:
    """Comprueba si existe algun dataset de radiografias (real o sintetico)."""
    # Estructura Kaggle original: xrays/<Clase>/images/
    for class_dir in ["COVID", "Normal", "Viral Pneumonia", "Lung_Opacity"]:
        images_dir = DATA_RAW_DIR / "xrays" / class_dir / "images"
        if images_dir.exists() and any(images_dir.iterdir()):
            return True
    # Estructura plana generada por generate_sample_data.py: <Clase>/
    for class_dir in ["COVID", "COVID19", "Normal", "Pneumonia"]:
        class_path = DATA_RAW_DIR / class_dir
        if class_path.exists() and any(class_path.iterdir()):
            return True
    return False


def generate_sample_data() -> bool:
    log("No se encontro dataset de radiografias.")
    log("Generando datos sinteticos (50 imagenes por clase)...")
    result = subprocess.run(
        [sys.executable, "generate_sample_data.py", "--num-per-class", "50"],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        log("Datos sinteticos generados.")
    else:
        log("ERROR: Fallo la generacion de datos sinteticos.")
    return result.returncode == 0


# ---------------------------------------------------------------------------
# 3. Modelo de radiografias
# ---------------------------------------------------------------------------

def train_radiology_model() -> bool:
    log("Modelo de radiografias no encontrado. Iniciando entrenamiento...")
    log("  Parametros: 5 epochs, sin fine-tuning (inicio rapido).")
    log("  Para un modelo de mayor calidad ejecuta despues:")
    log("    docker-compose run ai-train")
    result = subprocess.run(
        [
            sys.executable, "train_pipeline.py",
            "--epochs", "5",
            "--no-finetune",
        ],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode == 0:
        log("Modelo de radiografias entrenado y guardado.")
    else:
        log("ERROR: Fallo el entrenamiento del modelo de radiografias.")
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    log("=" * 60)
    log("INICIALIZACION DEL SISTEMA HOSPITALARIO")
    log("=" * 60)

    # 1. MinIO
    if not init_minio_buckets():
        return 1

    # 2. Dataset
    if has_dataset():
        log("Dataset de radiografias encontrado. Omitiendo generacion.")
    else:
        if not generate_sample_data():
            return 1

    # 3. Modelo de radiografias
    if MODEL_PATH.exists():
        log(f"Modelo encontrado en: {MODEL_PATH.name}")
        log("Omitiendo entrenamiento.")
    else:
        if not train_radiology_model():
            return 1

    log("=" * 60)
    log("Sistema listo. Arrancando servicios...")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
