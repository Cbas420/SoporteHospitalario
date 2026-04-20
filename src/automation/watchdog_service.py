"""
Servicio Watchdog para monitorizacion en tiempo real del directorio de entrada.

Implementa SDD §04: detecta nuevos ficheros (CSV/imagenes) en /data/incoming/
en menos de 10 segundos y lanza el pipeline automaticamente.
"""

from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from watchdog.events import FileCreatedEvent, FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from config.settings import (
    AUTOMATION_INPUT_DIR,
    AUTOMATION_PROCESSED_DIR,
    PATIENTS_CSV_PATH,
    RAW_DATA_DIR,
)
from src.api.inference import load_model, predict_single
from src.automation.scheduler import persist_prediction_outcome
from src.data.pipeline import prepare_data_pipeline, seed_repository_from_pipeline
from src.data.repository import HospitalRepository
from src.utils.logger import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
CSV_EXTENSIONS = {".csv"}

FAILED_DIR = AUTOMATION_INPUT_DIR / "failed"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class IncomingFileHandler(FileSystemEventHandler):
    """Manejador de eventos de ficheros nuevos en el directorio de entrada."""

    def __init__(self, repository: HospitalRepository) -> None:
        super().__init__()
        self.repository = repository
        self._model_cache = None

    def _get_model(self):
        if self._model_cache is None:
            try:
                self._model_cache = load_model()
                logger.info("Modelo cargado para watchdog.")
            except FileNotFoundError as exc:
                logger.error("No se puede cargar el modelo: %s", exc)
        return self._model_cache

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        path = Path(str(event.src_path))
        suffix = path.suffix.lower()

        if suffix in IMAGE_EXTENSIONS:
            logger.info("[Watchdog] Nueva imagen detectada: %s", path.name)
            self._handle_image(path)
        elif suffix in CSV_EXTENSIONS:
            logger.info("[Watchdog] Nuevo CSV detectado: %s", path.name)
            self._handle_csv(path)
        else:
            logger.debug("[Watchdog] Fichero ignorado (extension no soportada): %s", path.name)

    def _wait_for_file_ready(self, path: Path, timeout: float = 5.0) -> bool:
        """Espera hasta que el fichero se haya escrito completamente."""
        deadline = time.monotonic() + timeout
        last_size = -1
        while time.monotonic() < deadline:
            try:
                current_size = path.stat().st_size
                if current_size == last_size and current_size > 0:
                    return True
                last_size = current_size
            except OSError:
                pass
            time.sleep(0.3)
        return last_size > 0

    def _move_to_processed(self, path: Path) -> Optional[Path]:
        AUTOMATION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        dest = AUTOMATION_PROCESSED_DIR / path.name
        try:
            shutil.move(str(path), str(dest))
            return dest
        except Exception as exc:
            logger.error("[Watchdog] No se pudo mover %s a processed: %s", path.name, exc)
            return None

    def _move_to_failed(self, path: Path) -> None:
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        dest = FAILED_DIR / path.name
        try:
            shutil.move(str(path), str(dest))
            logger.warning("[Watchdog] Fichero movido a failed: %s", path.name)
        except Exception as exc:
            logger.error("[Watchdog] No se pudo mover %s a failed: %s", path.name, exc)

    def _handle_image(self, path: Path) -> dict[str, Any]:
        if not self._wait_for_file_ready(path):
            logger.warning("[Watchdog] Fichero no disponible aun: %s", path.name)
            return {"status": "skipped", "reason": "file_not_ready"}

        model = self._get_model()
        if model is None:
            self._move_to_failed(path)
            return {"status": "error", "reason": "model_not_available"}

        try:
            image_bytes = path.read_bytes()
            prediction = predict_single(image_bytes, model=model)
            prediction["timestamp"] = _utcnow_iso()

            patient_id = path.stem
            persist_prediction_outcome(
                repository=self.repository,
                prediction=prediction,
                patient_id=patient_id,
                image_path=str(path),
                source="watchdog",
            )

            dest = self._move_to_processed(path)
            logger.info(
                "[Watchdog] Imagen procesada: %s → %s (confianza=%.2f)",
                path.name,
                prediction["prediction"],
                prediction["confidence"],
            )
            return {
                "status": "ok",
                "patient_id": patient_id,
                "prediction": prediction["prediction"],
                "confidence": prediction["confidence"],
                "destination": str(dest) if dest else None,
            }
        except Exception as exc:
            logger.error("[Watchdog] Error procesando imagen %s: %s", path.name, exc)
            self._move_to_failed(path)
            return {"status": "error", "reason": str(exc)}

    def _handle_csv(self, path: Path) -> dict[str, Any]:
        if not self._wait_for_file_ready(path):
            logger.warning("[Watchdog] CSV no disponible aun: %s", path.name)
            return {"status": "skipped", "reason": "file_not_ready"}

        try:
            dest = AUTOMATION_PROCESSED_DIR / path.name
            AUTOMATION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(dest))

            prepare_data_pipeline(csv_path=path, raw_data_dir=RAW_DATA_DIR)
            seed_repository_from_pipeline(self.repository, csv_path=path)

            path.unlink(missing_ok=True)
            logger.info("[Watchdog] CSV procesado y pacientes sincronizados: %s", path.name)
            return {"status": "ok", "csv": path.name}
        except Exception as exc:
            logger.error("[Watchdog] Error procesando CSV %s: %s", path.name, exc)
            self._move_to_failed(path)
            return {"status": "error", "reason": str(exc)}


def run_watchdog(
    input_dir: Path = AUTOMATION_INPUT_DIR,
    repository: Optional[HospitalRepository] = None,
) -> None:
    """
    Inicia el watchdog bloqueante que monitoriza input_dir en tiempo real.
    Cumple SDD §04: deteccion en <10 segundos.
    """
    repository = repository or HospitalRepository()
    repository.initialize()

    input_dir.mkdir(parents=True, exist_ok=True)
    AUTOMATION_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    handler = IncomingFileHandler(repository=repository)
    observer = Observer()
    observer.schedule(handler, str(input_dir), recursive=False)
    observer.start()

    logger.info("[Watchdog] Monitorizando directorio: %s", input_dir)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("[Watchdog] Detenido por el usuario.")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    run_watchdog()
