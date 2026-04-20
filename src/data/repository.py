"""
Capa de persistencia para pacientes, predicciones y alertas.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.pool import StaticPool

from config.settings import get_database_url
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HospitalRepository:
    """Repositorio con soporte SQLite/PostgreSQL via SQLAlchemy."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or get_database_url()
        engine_kwargs = {"future": True}
        if self.database_url.startswith("sqlite:///:memory:"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
            engine_kwargs["poolclass"] = StaticPool
        elif self.database_url.startswith("sqlite:///"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        self.engine = create_engine(self.database_url, **engine_kwargs)
        self.metadata = MetaData()

        self.patients = Table(
            "patients",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("record_uid", String(255), nullable=False, unique=True),
            Column("patient_id", String(128), nullable=False),
            Column("image_name", String(255)),
            Column("image_path", Text),
            Column("diagnosis", String(64)),
            Column("age", Integer),
            Column("sex", String(32)),
            Column("admission_date", String(64)),
            Column("source", String(64)),
            Column("last_updated", DateTime(timezone=True), nullable=False),
        )

        self.predictions = Table(
            "predictions",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("patient_id", String(128)),
            Column("image_path", Text),
            Column("prediction", String(64), nullable=False),
            Column("confidence", Float, nullable=False),
            Column("requires_review", Boolean, nullable=False, default=False),
            Column("inference_time_ms", Float),
            Column("probabilities", Text, nullable=False),
            Column("source", String(64)),
            Column("created_at", DateTime(timezone=True), nullable=False),
        )

        self.alerts = Table(
            "alerts",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("patient_id", String(128)),
            Column("image_path", Text),
            Column("alert_type", String(64), nullable=False),
            Column("severity", String(32), nullable=False),
            Column("message", Text, nullable=False),
            Column("source", String(64)),
            Column("resolved", Boolean, nullable=False, default=False),
            Column("created_at", DateTime(timezone=True), nullable=False),
        )

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _normalize_record_uid(record: dict[str, Any]) -> str:
        patient_id = str(record.get("patient_id") or "unknown").strip()
        image_name = str(
            record.get("image_name")
            or Path(str(record.get("image_path") or "record")).name
        ).strip()
        return f"{patient_id}:{image_name}"

    def initialize(self) -> None:
        self.metadata.create_all(self.engine)
        logger.info("Repositorio inicializado en %s", self.database_url)

    def upsert_patients(self, records: Iterable[dict[str, Any]]) -> int:
        prepared_records = []
        now = self._utcnow()

        for record in records:
            patient_id = str(record.get("patient_id") or "").strip()
            if not patient_id:
                continue

            raw_age = record.get("age")
            if raw_age in (None, "") or str(raw_age).lower() == "nan":
                age = None
            else:
                age = int(float(raw_age))

            payload = {
                "record_uid": self._normalize_record_uid(record),
                "patient_id": patient_id,
                "image_name": record.get("image_name"),
                "image_path": record.get("image_path"),
                "diagnosis": record.get("diagnosis"),
                "age": age,
                "sex": record.get("sex"),
                "admission_date": record.get("admission_date"),
                "source": record.get("source") or "pipeline",
                "last_updated": now,
            }
            prepared_records.append(payload)

        if not prepared_records:
            return 0

        with self.engine.begin() as connection:
            for payload in prepared_records:
                existing = connection.execute(
                    select(self.patients.c.id).where(
                        self.patients.c.record_uid == payload["record_uid"]
                    )
                ).first()
                if existing:
                    connection.execute(
                        self.patients.update()
                        .where(self.patients.c.id == existing.id)
                        .values(**payload)
                    )
                else:
                    connection.execute(self.patients.insert().values(**payload))

        return len(prepared_records)

    def list_patients(
        self,
        limit: int = 100,
        diagnosis: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        statement = (
            select(
                self.patients.c.patient_id,
                self.patients.c.image_name,
                self.patients.c.image_path,
                self.patients.c.diagnosis,
                self.patients.c.age,
                self.patients.c.sex,
                self.patients.c.admission_date,
                self.patients.c.source,
                self.patients.c.last_updated,
            )
            .order_by(self.patients.c.last_updated.desc())
            .limit(limit)
        )

        if diagnosis:
            statement = statement.where(self.patients.c.diagnosis == diagnosis)

        with self.engine.begin() as connection:
            rows = connection.execute(statement).mappings().all()

        return [dict(row) for row in rows]

    def save_prediction(self, record: dict[str, Any]) -> int:
        created_at = record.get("created_at") or self._utcnow()
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        payload = {
            "patient_id": record.get("patient_id"),
            "image_path": record.get("image_path"),
            "prediction": record["prediction"],
            "confidence": float(record["confidence"]),
            "requires_review": bool(record.get("requires_review", False)),
            "inference_time_ms": float(record.get("inference_time_ms", 0.0)),
            "probabilities": json.dumps(record.get("probabilities", {})),
            "source": record.get("source") or "api",
            "created_at": created_at,
        }

        with self.engine.begin() as connection:
            result = connection.execute(self.predictions.insert().values(**payload))

        inserted_id = result.inserted_primary_key[0]
        return int(inserted_id) if inserted_id is not None else 0

    def save_alert(self, record: dict[str, Any]) -> int:
        created_at = record.get("created_at") or self._utcnow()
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        payload = {
            "patient_id": record.get("patient_id"),
            "image_path": record.get("image_path"),
            "alert_type": record["alert_type"],
            "severity": record["severity"],
            "message": record["message"],
            "source": record.get("source") or "automation",
            "resolved": bool(record.get("resolved", False)),
            "created_at": created_at,
        }

        with self.engine.begin() as connection:
            result = connection.execute(self.alerts.insert().values(**payload))

        inserted_id = result.inserted_primary_key[0]
        return int(inserted_id) if inserted_id is not None else 0

    def recent_predictions(self, limit: int = 20) -> list[dict[str, Any]]:
        statement = (
            select(
                self.predictions.c.patient_id,
                self.predictions.c.image_path,
                self.predictions.c.prediction,
                self.predictions.c.confidence,
                self.predictions.c.requires_review,
                self.predictions.c.inference_time_ms,
                self.predictions.c.probabilities,
                self.predictions.c.source,
                self.predictions.c.created_at,
            )
            .order_by(self.predictions.c.created_at.desc())
            .limit(limit)
        )

        with self.engine.begin() as connection:
            rows = connection.execute(statement).mappings().all()

        results = []
        for row in rows:
            payload = dict(row)
            payload["probabilities"] = json.loads(payload["probabilities"])
            results.append(payload)
        return results

    def get_patient_by_id(self, patient_id: str) -> Optional[dict[str, Any]]:
        with self.engine.begin() as connection:
            row = connection.execute(
                select(self.patients).where(self.patients.c.patient_id == patient_id).limit(1)
            ).mappings().first()

        if row is None:
            return None

        patient = dict(row)
        # Historial de predicciones del paciente
        preds = connection if False else None  # evitar uso de conexion cerrada
        with self.engine.begin() as connection:
            pred_rows = connection.execute(
                select(
                    self.predictions.c.prediction,
                    self.predictions.c.confidence,
                    self.predictions.c.requires_review,
                    self.predictions.c.inference_time_ms,
                    self.predictions.c.probabilities,
                    self.predictions.c.source,
                    self.predictions.c.created_at,
                )
                .where(self.predictions.c.patient_id == patient_id)
                .order_by(self.predictions.c.created_at.desc())
                .limit(20)
            ).mappings().all()

        prediction_history = []
        for pr in pred_rows:
            record = dict(pr)
            record["probabilities"] = json.loads(record["probabilities"])
            prediction_history.append(record)

        patient["prediction_history"] = prediction_history
        return patient

    def resolve_alert(self, alert_id: int) -> bool:
        with self.engine.begin() as connection:
            result = connection.execute(
                self.alerts.update()
                .where(self.alerts.c.id == alert_id)
                .values(resolved=True)
            )
        return result.rowcount > 0

    def recent_alerts(self, limit: int = 20) -> list[dict[str, Any]]:
        statement = (
            select(
                self.alerts.c.patient_id,
                self.alerts.c.image_path,
                self.alerts.c.alert_type,
                self.alerts.c.severity,
                self.alerts.c.message,
                self.alerts.c.source,
                self.alerts.c.resolved,
                self.alerts.c.created_at,
            )
            .order_by(self.alerts.c.created_at.desc())
            .limit(limit)
        )

        with self.engine.begin() as connection:
            rows = connection.execute(statement).mappings().all()

        return [dict(row) for row in rows]

    def get_stats(self) -> dict[str, Any]:
        with self.engine.begin() as connection:
            patients_total = connection.execute(
                select(func.count()).select_from(self.patients)
            ).scalar_one()
            predictions_total = connection.execute(
                select(func.count()).select_from(self.predictions)
            ).scalar_one()
            alerts_total = connection.execute(
                select(func.count()).select_from(self.alerts)
            ).scalar_one()
            low_confidence_total = connection.execute(
                select(func.count()).select_from(self.predictions).where(
                    self.predictions.c.requires_review.is_(True)
                )
            ).scalar_one()

            diagnosis_rows = connection.execute(
                select(
                    self.patients.c.diagnosis,
                    func.count().label("count"),
                )
                .group_by(self.patients.c.diagnosis)
                .order_by(self.patients.c.diagnosis)
            ).all()

            prediction_rows = connection.execute(
                select(
                    self.predictions.c.prediction,
                    func.count().label("count"),
                )
                .group_by(self.predictions.c.prediction)
                .order_by(self.predictions.c.prediction)
            ).all()

        return {
            "patients_total": int(patients_total or 0),
            "predictions_total": int(predictions_total or 0),
            "alerts_total": int(alerts_total or 0),
            "low_confidence_total": int(low_confidence_total or 0),
            "patients_by_diagnosis": {
                str(diagnosis or "Unknown"): int(count) for diagnosis, count in diagnosis_rows
            },
            "predictions_by_class": {
                str(prediction or "Unknown"): int(count)
                for prediction, count in prediction_rows
            },
            "database_backend": self.database_url.split(":", 1)[0],
        }
