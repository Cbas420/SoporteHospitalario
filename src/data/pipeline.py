"""
Pipeline de datos para pacientes e imagenes medicas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config.settings import (
    CLASS_NAMES,
    DATASET_MANIFEST_PATH,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECURE,
    MINIO_SECRET_KEY,
    PATIENTS_CANONICAL_FIELDS,
    PATIENTS_CLEAN_PATH,
    PATIENTS_CSV_PATH,
    PIPELINE_REPORT_PATH,
    RAW_DATA_DIR,
)
from src.data.loader import generate_dataset_report, scan_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)


COLUMN_ALIASES = {
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


def _normalize_column_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def normalize_diagnosis(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = (
        str(value)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
    )
    mapping = {
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
    return mapping.get(text)


def build_image_manifest(raw_data_dir: Path = RAW_DATA_DIR) -> list[dict[str, Any]]:
    dataset = scan_dataset(raw_data_dir)
    manifest = []
    for diagnosis, file_paths in dataset.items():
        for item in file_paths:
            if isinstance(item, dict):
                path = Path(item["image_path"])
                patient_id = str(item.get("patient_id") or path.stem)
                image_name = str(item.get("image_name") or path.name)
                source = str(item.get("source_class") or "image_dataset")
            else:
                path = Path(item)
                patient_id = path.stem
                image_name = path.name
                source = "image_dataset"

            manifest.append(
                {
                    "patient_id": patient_id,
                    "image_name": image_name,
                    "image_path": str(path),
                    "diagnosis": diagnosis,
                    "source": source,
                }
            )
    return manifest


def get_object_storage_status() -> dict[str, Any]:
    status = {
        "backend": "MinIO",
        "endpoint": MINIO_ENDPOINT,
        "bucket": MINIO_BUCKET,
        "secure": MINIO_SECURE,
        "reachable": False,
    }
    try:
        from minio import Minio

        client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        status["reachable"] = True
        status["bucket_exists"] = client.bucket_exists(MINIO_BUCKET)
    except Exception as exc:  # pragma: no cover - depende de infraestructura
        status["error"] = str(exc)
    return status


def _standardize_patient_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    renamed_columns = {
        column: COLUMN_ALIASES.get(_normalize_column_name(column), _normalize_column_name(column))
        for column in df.columns
    }
    df = df.rename(columns=renamed_columns).copy()

    for field in PATIENTS_CANONICAL_FIELDS:
        if field not in df.columns:
            df[field] = None

    return df[PATIENTS_CANONICAL_FIELDS].copy()


def _resolve_image_path(primary_path: Any, manifest_path: Any) -> str:
    primary = str(primary_path).strip() if primary_path is not None else ""
    manifest = str(manifest_path).strip() if manifest_path is not None else ""

    if primary and Path(primary).exists():
        return primary
    if manifest:
        return manifest
    return primary


def prepare_patient_dataset(
    csv_path: Path = PATIENTS_CSV_PATH,
    raw_data_dir: Path = RAW_DATA_DIR,
    output_path: Path = PATIENTS_CLEAN_PATH,
    manifest_path: Path = DATASET_MANIFEST_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_records = build_image_manifest(raw_data_dir)
    manifest_df = pd.DataFrame(manifest_records)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if csv_path.exists():
        raw_df = pd.read_csv(csv_path)
    else:
        raw_df = pd.DataFrame(columns=PATIENTS_CANONICAL_FIELDS)

    standardized = _standardize_patient_dataframe(raw_df)
    input_rows = len(standardized)

    standardized["diagnosis"] = standardized["diagnosis"].apply(normalize_diagnosis)
    standardized["image_name"] = standardized["image_name"].fillna("").astype(str)
    standardized["image_path"] = standardized["image_path"].fillna("").astype(str)
    standardized["source"] = standardized["source"].fillna("patients_csv")
    standardized["in_manifest"] = False

    if not manifest_df.empty:
        standardized = standardized.merge(
            manifest_df[["image_name", "image_path", "diagnosis"]].rename(
                columns={
                    "image_path": "manifest_image_path",
                    "diagnosis": "manifest_diagnosis",
                }
            ),
            on="image_name",
            how="outer",
        )
        standardized["in_manifest"] = standardized["manifest_image_path"].notna()
        standardized["image_path"] = standardized.apply(
            lambda row: _resolve_image_path(
                row.get("image_path"),
                row.get("manifest_image_path"),
            ),
            axis=1,
        )
        standardized["diagnosis"] = standardized["diagnosis"].fillna(
            standardized["manifest_diagnosis"]
        )
        standardized["patient_id"] = standardized["patient_id"].where(
            standardized["patient_id"].notna()
            & (standardized["patient_id"].astype(str).str.len() > 0),
            standardized["image_name"].str.replace(r"\.[^.]+$", "", regex=True),
        )
        standardized["source"] = standardized["source"].fillna("image_dataset")
        standardized = standardized.drop(
            columns=["manifest_image_path", "manifest_diagnosis"]
        )

    standardized["patient_id"] = standardized["patient_id"].fillna("").astype(str)
    standardized["patient_id"] = standardized["patient_id"].where(
        standardized["patient_id"].str.len() > 0,
        "patient_" + standardized.index.astype(str),
    )
    standardized["image_name"] = standardized["image_name"].where(
        standardized["image_name"].str.len() > 0,
        standardized["image_path"].apply(lambda path: Path(path).name if path else ""),
    )
    standardized["diagnosis"] = standardized["diagnosis"].apply(normalize_diagnosis)
    standardized["record_uid"] = (
        standardized["patient_id"].astype(str)
        + ":"
        + standardized["image_name"].fillna("").astype(str)
    )

    duplicates_removed = int(standardized.duplicated(subset=["record_uid"]).sum())
    standardized = standardized.drop_duplicates(subset=["record_uid"], keep="first")

    standardized["image_exists"] = standardized["image_path"].apply(
        lambda value: Path(value).exists() if value else False
    )

    invalid_diagnoses_removed = int(
        (~standardized["diagnosis"].isin(CLASS_NAMES)).sum()
    )
    missing_images_removed = int((~standardized["image_exists"]).sum())

    cleaned = standardized[
        standardized["diagnosis"].isin(CLASS_NAMES)
        & standardized["image_exists"]
        & standardized["in_manifest"]
    ].copy()
    cleaned = cleaned[PATIENTS_CANONICAL_FIELDS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    report = {
        "input_rows": int(input_rows),
        "records_after_cleaning": int(len(cleaned)),
        "duplicates_removed": duplicates_removed,
        "invalid_diagnoses_removed": invalid_diagnoses_removed,
        "missing_images_removed": missing_images_removed,
        "manifest_records": int(len(manifest_records)),
    }

    return cleaned, report


def prepare_data_pipeline(
    csv_path: Path = PATIENTS_CSV_PATH,
    raw_data_dir: Path = RAW_DATA_DIR,
    output_path: Path = PATIENTS_CLEAN_PATH,
    report_path: Path = PIPELINE_REPORT_PATH,
) -> dict[str, Any]:
    dataset = scan_dataset(raw_data_dir)
    dataset_report = generate_dataset_report(dataset)
    patients_df, cleaning_report = prepare_patient_dataset(
        csv_path=csv_path,
        raw_data_dir=raw_data_dir,
        output_path=output_path,
    )

    report = {
        "status": "ready" if len(patients_df) > 0 else "empty",
        "patients": cleaning_report,
        "dataset": dataset_report,
        "storage": {
            "database": "PostgreSQL/SQLite via SQLAlchemy",
            "object_storage": get_object_storage_status(),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Pipeline de datos preparado en %s", report_path)
    return report


def seed_repository_from_pipeline(
    repository,
    csv_path: Path = PATIENTS_CSV_PATH,
    raw_data_dir: Path = RAW_DATA_DIR,
) -> int:
    patients_df, _ = prepare_patient_dataset(csv_path=csv_path, raw_data_dir=raw_data_dir)
    if patients_df.empty:
        return 0
    inserted = repository.upsert_patients(patients_df.to_dict(orient="records"))
    logger.info("Pacientes sincronizados en repositorio: %s", inserted)
    return inserted
