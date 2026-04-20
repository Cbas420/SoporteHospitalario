"""
Logging estructurado con formato JSON y rotacion de ficheros.

Implementa SDD §07: logs JSON a /logs/<modulo>.log con rotacion (10 MB, 5 copias).
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
from datetime import datetime, timezone
from pathlib import Path

from config.settings import LOG_LEVEL

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"


class JsonFormatter(logging.Formatter):
    """Formateador que serializa cada registro como una linea JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger configurado para el modulo dado.

    - Consola: formato texto legible.
    - Fichero: formato JSON con RotatingFileHandler (10 MB, 5 copias).
      Ruta: logs/<nombre_raiz>.log  (ej. logs/src.api.log)

    Args:
        name: Nombre del modulo (usar __name__ del caller).

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    # --- Handler de consola (texto) ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(console_handler)

    # --- Handler de fichero (JSON con rotacion) ---
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        # Nombre de fichero: primer segmento del nombre del modulo
        # src.api.app → api.log, src.data.pipeline → data.log, etc.
        parts = name.split(".")
        file_stem = parts[1] if len(parts) > 1 else parts[0]
        log_file = LOGS_DIR / f"{file_stem}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(JsonFormatter())
        logger.addHandler(file_handler)
    except Exception as exc:
        # No impedir que la app arranque si no se puede escribir logs
        logger.warning("No se pudo crear el handler de fichero: %s", exc)

    logger.propagate = False
    return logger
