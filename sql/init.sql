-- =============================================================================
-- Esquema de base de datos del Sistema Inteligente de Soporte Hospitalario
-- SDD §06: Inicialización automática de PostgreSQL en docker-compose up
-- =============================================================================

-- Tabla de pacientes
CREATE TABLE IF NOT EXISTS patients (
    id              SERIAL PRIMARY KEY,
    record_uid      VARCHAR(255) NOT NULL UNIQUE,
    patient_id      VARCHAR(128) NOT NULL,
    image_name      VARCHAR(255),
    image_path      TEXT,
    diagnosis       VARCHAR(64),
    age             INTEGER CHECK (age IS NULL OR (age >= 0 AND age <= 120)),
    sex             VARCHAR(32),
    admission_date  VARCHAR(64),
    source          VARCHAR(64),
    last_updated    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_patients_patient_id  ON patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_patients_diagnosis   ON patients(diagnosis);
CREATE INDEX IF NOT EXISTS idx_patients_admission   ON patients(admission_date);

-- Tabla de predicciones del modelo
CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL PRIMARY KEY,
    patient_id          VARCHAR(128),
    image_path          TEXT,
    prediction          VARCHAR(64)  NOT NULL,
    confidence          FLOAT        NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    requires_review     BOOLEAN      NOT NULL DEFAULT FALSE,
    inference_time_ms   FLOAT,
    probabilities       TEXT         NOT NULL,   -- JSON serializado
    source              VARCHAR(64),
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_patient_id ON predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_prediction ON predictions(prediction);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at DESC);

-- Tabla de alertas clinicas
CREATE TABLE IF NOT EXISTS alerts (
    id          SERIAL PRIMARY KEY,
    patient_id  VARCHAR(128),
    image_path  TEXT,
    alert_type  VARCHAR(64)  NOT NULL,
    severity    VARCHAR(32)  NOT NULL CHECK (severity IN ('critical', 'high', 'warning', 'medium', 'low', 'info')),
    message     TEXT         NOT NULL,
    source      VARCHAR(64),
    resolved    BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alerts_patient_id ON alerts(patient_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity   ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved   ON alerts(resolved);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at DESC);

-- Tabla de registro de calidad de datos (SDD §07)
-- Registra incidencias detectadas durante el pipeline de datos
CREATE TABLE IF NOT EXISTS data_quality_log (
    id              SERIAL PRIMARY KEY,
    source_file     VARCHAR(512),                -- Fichero origen del problema
    issue_type      VARCHAR(128) NOT NULL,       -- duplicate, null_field, corrupted_image, invalid_range, etc.
    severity        VARCHAR(32)  NOT NULL CHECK (severity IN ('error', 'warning', 'info')),
    description     TEXT         NOT NULL,       -- Descripcion detallada del problema
    field_name      VARCHAR(128),                -- Campo afectado (si aplica)
    record_uid      VARCHAR(255),                -- UID del registro afectado (si aplica)
    resolved        BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quality_log_severity    ON data_quality_log(severity);
CREATE INDEX IF NOT EXISTS idx_quality_log_issue_type  ON data_quality_log(issue_type);
CREATE INDEX IF NOT EXISTS idx_quality_log_created_at  ON data_quality_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_quality_log_resolved    ON data_quality_log(resolved);
