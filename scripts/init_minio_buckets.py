"""
Inicializacion de buckets MinIO al arrancar el sistema.

Crea los 5 buckets requeridos por SDD §01/§06 si no existen:
  - raw-xrays       : imagenes originales recibidas
  - lung-masks      : mascaras de segmentacion pulmonar
  - processed-xrays : imagenes preprocesadas (224x224, normalizadas)
  - quarantine      : imagenes/registros rechazados por errores de calidad
  - reports         : informes diarios generados por el scheduler
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.settings import (
    MINIO_ACCESS_KEY,
    MINIO_ALL_BUCKETS,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def init_buckets(retries: int = 10, delay: float = 3.0) -> bool:
    """
    Inicializa todos los buckets MinIO necesarios.
    Reintenta hasta `retries` veces esperando `delay` segundos entre intentos.
    """
    try:
        from minio import Minio
        from minio.error import S3Error
    except ImportError:
        logger.error("La libreria minio no esta instalada.")
        return False

    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )

    for attempt in range(1, retries + 1):
        try:
            for bucket in MINIO_ALL_BUCKETS:
                if not client.bucket_exists(bucket):
                    client.make_bucket(bucket)
                    logger.info("Bucket creado: %s", bucket)
                else:
                    logger.info("Bucket ya existe: %s", bucket)
            logger.info("Inicializacion de buckets completada.")
            return True
        except Exception as exc:
            logger.warning(
                "Intento %d/%d fallido al inicializar buckets: %s",
                attempt,
                retries,
                exc,
            )
            if attempt < retries:
                time.sleep(delay)

    logger.error("No se pudieron inicializar los buckets de MinIO tras %d intentos.", retries)
    return False


if __name__ == "__main__":
    success = init_buckets()
    sys.exit(0 if success else 1)
