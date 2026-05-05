"""
Procesamiento distribuido de datos de pacientes usando Dask.

Justificacion tecnica (vs PySpark):
  - Dask no requiere JVM, reduciendo la imagen Docker ~400MB.
  - API compatible con pandas: menor curva de aprendizaje y menos codigo.
  - Escalable horizontalmente añadiendo workers si el volumen crece.
  - Adecuado para el volumen hospitalario actual (~5k pacientes, ~21k imagenes).
  - PySpark justificaria su overhead solo con clusters multi-nodo reales (>1TB).

Flujo:
  CSV raw  →  dd.read_csv (particionado)
           →  normalize_columns
           →  clean_diagnoses (por particion)
           →  deduplicate
           →  validate_images (dask.delayed en paralelo)
           →  pandas DataFrame limpio
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

import dask
import dask.dataframe as dd
import pandas as pd
from dask import delayed

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Mapeo de columnas (mismo que pipeline.py para consistencia)
# ---------------------------------------------------------------------------
_COLUMN_ALIASES: dict[str, str] = {
    "patient_id": "patient_id",
    "id": "patient_id",
    "patientid": "patient_id",
    "image_name": "image_name",
    "image": "image_name",
    "filename": "image_name",
    "file_name": "image_name",
    "image_path": "image_path",
    "path": "image_path",
    "filepath": "image_path",
    "diagnosis": "diagnosis",
    "label": "diagnosis",
    "class": "diagnosis",
    "age": "age",
    "sex": "sex",
    "gender": "sex",
    "admission_date": "admission_date",
    "date": "admission_date",
    "source": "source",
}

_DIAGNOSIS_MAP: dict[str, str] = {
    "covid": "COVID19",
    "covid19": "COVID19",
    "normal": "Normal",
    "sana": "Normal",
    "healthy": "Normal",
    "pneumonia": "Pneumonia",
    "neumonia": "Pneumonia",
    "lungopacity": "Pneumonia",
    "viralpneumonia": "Pneumonia",
}

_VALID_DIAGNOSES = {"COVID19", "Normal", "Pneumonia"}

_CANONICAL_FIELDS = [
    "patient_id",
    "image_name",
    "image_path",
    "diagnosis",
    "age",
    "sex",
    "admission_date",
    "source",
]


# ---------------------------------------------------------------------------
# Funciones de transformacion por particion (ejecutadas en paralelo por Dask)
# ---------------------------------------------------------------------------

def _normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _normalize_columns_partition(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas segun alias canonicos. Se aplica por particion."""
    rename_map = {
        col: _COLUMN_ALIASES.get(_normalize_col(col), _normalize_col(col))
        for col in df.columns
    }
    df = df.rename(columns=rename_map)
    for field in _CANONICAL_FIELDS:
        if field not in df.columns:
            df[field] = None
    return df[_CANONICAL_FIELDS]


def _clean_diagnosis_value(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = (
        str(value).strip().lower()
        .replace("-", "").replace("_", "").replace(" ", "")
    )
    return _DIAGNOSIS_MAP.get(text)


def _clean_diagnoses_partition(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza el campo diagnosis. Se aplica por particion."""
    df = df.copy()
    df["diagnosis"] = df["diagnosis"].apply(_clean_diagnosis_value)
    return df


def _fill_defaults_partition(df: pd.DataFrame) -> pd.DataFrame:
    """Rellena valores nulos con defaults seguros. Se aplica por particion."""
    df = df.copy()
    df["image_name"] = df["image_name"].fillna("").astype(str)
    df["image_path"] = df["image_path"].fillna("").astype(str)
    df["source"] = df["source"].fillna("patients_csv")
    df["patient_id"] = df["patient_id"].fillna("").astype(str)
    return df


# ---------------------------------------------------------------------------
# Validacion de imagenes en paralelo con dask.delayed
# ---------------------------------------------------------------------------

@delayed
def _check_image_exists(image_path: str) -> bool:
    """Comprueba si una imagen existe en disco. Se ejecuta en paralelo."""
    if not image_path:
        return False
    return Path(image_path).exists()


def validate_images_parallel(image_paths: list[str]) -> dict[str, Any]:
    """
    Valida la existencia de imagenes en paralelo usando dask.delayed.

    Returns:
        dict con 'existing', 'missing' y 'total'.
    """
    if not image_paths:
        return {"existing": 0, "missing": 0, "total": 0, "missing_paths": []}

    tasks = [_check_image_exists(p) for p in image_paths]
    results = dask.compute(*tasks)

    existing = sum(1 for r in results if r)
    missing_paths = [p for p, r in zip(image_paths, results) if not r]

    return {
        "existing": existing,
        "missing": len(missing_paths),
        "total": len(image_paths),
        "missing_paths": missing_paths[:20],  # limitar para el informe
    }


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_dask_csv_processing(
    csv_path: Path,
    npartitions: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Procesa el CSV de pacientes con Dask.

    Fases:
      1. Carga con dd.read_csv (particionado para procesamiento paralelo)
      2. Normalizacion de columnas por particion
      3. Limpieza de diagnosticos por particion
      4. Deduplicacion
      5. Validacion de imagenes en paralelo con dask.delayed
      6. .compute() para materializar el resultado

    Args:
        csv_path: Ruta al CSV de pacientes.
        npartitions: Numero de particiones Dask (workers logicos).

    Returns:
        Tupla (DataFrame limpio, informe de procesamiento).
    """
    t_start = time.monotonic()

    # --- 1. Carga lazy con Dask ---
    if not csv_path.exists():
        logger.warning("[Dask] CSV no encontrado: %s — devolviendo DataFrame vacio", csv_path)
        empty_df = pd.DataFrame(columns=_CANONICAL_FIELDS)
        return empty_df, _empty_report()

    logger.info("[Dask] Cargando CSV: %s", csv_path)
    ddf = dd.read_csv(
        str(csv_path),
        dtype=str,           # todo como string para evitar inferencia erronea
        assume_missing=True,
    ).repartition(npartitions=npartitions)

    input_rows = len(ddf)
    logger.info("[Dask] Filas leidas: %s en %s particiones", input_rows, ddf.npartitions)

    # --- 2. Normalizacion de columnas (por particion, en paralelo) ---
    ddf = ddf.map_partitions(_normalize_columns_partition)

    # --- 3. Limpieza de diagnosticos (por particion, en paralelo) ---
    ddf = ddf.map_partitions(_clean_diagnoses_partition)

    # --- 4. Relleno de defaults (por particion, en paralelo) ---
    ddf = ddf.map_partitions(_fill_defaults_partition)

    # --- 5. Materializacion ---
    df: pd.DataFrame = ddf.compute()

    # --- 6. Deduplicacion (requiere vista global, post-compute) ---
    df["record_uid"] = (
        df["patient_id"].astype(str) + ":" + df["image_name"].astype(str)
    )
    duplicates_removed = int(df.duplicated(subset=["record_uid"]).sum())
    df = df.drop_duplicates(subset=["record_uid"], keep="first").copy()

    # --- 7. Filtrado por diagnostico valido ---
    invalid_mask = ~df["diagnosis"].isin(_VALID_DIAGNOSES)
    invalid_count = int(invalid_mask.sum())
    df = df[~invalid_mask].copy()

    # --- 8. Validacion de imagenes en paralelo (dask.delayed) ---
    image_paths = df["image_path"].tolist()
    img_report = validate_images_parallel(image_paths)

    elapsed_ms = (time.monotonic() - t_start) * 1000

    report: dict[str, Any] = {
        "engine": "dask",
        "npartitions": npartitions,
        "input_rows": input_rows,
        "after_dedup": len(df) + invalid_count,
        "duplicates_removed": duplicates_removed,
        "invalid_diagnoses_removed": invalid_count,
        "records_output": len(df),
        "image_validation": img_report,
        "processing_time_ms": round(elapsed_ms, 1),
    }

    logger.info(
        "[Dask] Pipeline completado: %s registros validos en %.1fms",
        len(df),
        elapsed_ms,
    )
    return df[_CANONICAL_FIELDS], report


def _empty_report() -> dict[str, Any]:
    return {
        "engine": "dask",
        "npartitions": 0,
        "input_rows": 0,
        "after_dedup": 0,
        "duplicates_removed": 0,
        "invalid_diagnoses_removed": 0,
        "records_output": 0,
        "image_validation": {"existing": 0, "missing": 0, "total": 0},
        "processing_time_ms": 0.0,
    }
