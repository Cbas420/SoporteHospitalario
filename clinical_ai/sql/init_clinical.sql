-- Inicializacion de la base de datos clinica
-- Ejecutar: psql -U postgres -d clinical_db -f init_clinical.sql

-- Crear tabla de predicciones clinicas
CREATE TABLE IF NOT EXISTS clinical_predictions (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(128),
    age INTEGER,
    sex VARCHAR(1),
    height_cm FLOAT,
    weight_kg FLOAT,

    -- Sintomas
    has_cough BOOLEAN DEFAULT FALSE,
    has_chest_pain BOOLEAN DEFAULT FALSE,
    has_fatigue BOOLEAN DEFAULT FALSE,
    has_fever BOOLEAN DEFAULT FALSE,
    has_dizziness BOOLEAN DEFAULT FALSE,
    has_breathing_difficulty BOOLEAN DEFAULT FALSE,
    has_headache BOOLEAN DEFAULT FALSE,
    has_nausea BOOLEAN DEFAULT FALSE,
    has_loss_of_appetite BOOLEAN DEFAULT FALSE,
    has_night_sweats BOOLEAN DEFAULT FALSE,

    -- Resultados
    diabetes_risk BOOLEAN,
    hypertension_risk BOOLEAN,
    heart_disease_risk BOOLEAN,
    diabetes_probability FLOAT,
    hypertension_probability FLOAT,
    heart_disease_probability FLOAT,
    risk_level VARCHAR(20),

    -- Metadatos
    confidence FLOAT,
    inference_time_ms FLOAT,
    source VARCHAR(64) DEFAULT 'api',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices para busquedas rapidas
CREATE INDEX IF NOT EXISTS idx_predictions_patient ON clinical_predictions(patient_id);
CREATE INDEX IF NOT EXISTS idx_predictions_diabetes ON clinical_predictions(diabetes_risk);
CREATE INDEX IF NOT EXISTS idx_predictions_hypertension ON clinical_predictions(hypertension_risk);
CREATE INDEX IF NOT EXISTS idx_predictions_heart ON clinical_predictions(heart_disease_risk);
CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON clinical_predictions(risk_level);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON clinical_predictions(created_at);

-- Tabla de pacientes clinicos (datos basicos)
CREATE TABLE IF NOT EXISTS clinical_patients (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(128) UNIQUE NOT NULL,
    age INTEGER,
    sex VARCHAR(1),
    height_cm FLOAT,
    weight_kg FLOAT,
    bmi FLOAT,
    source VARCHAR(64) DEFAULT 'clinical_api',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON clinical_patients(patient_id);

-- Tabla de alertas clinicas
CREATE TABLE IF NOT EXISTS clinical_alerts (
    id SERIAL PRIMARY KEY,
    patient_id VARCHAR(128),
    alert_type VARCHAR(64) NOT NULL,
    severity VARCHAR(32) NOT NULL,
    message TEXT,
    disease VARCHAR(32),
    probability FLOAT,
    resolved BOOLEAN DEFAULT FALSE,
    source VARCHAR(64) DEFAULT 'clinical_api',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_patient ON clinical_alerts(patient_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON clinical_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON clinical_alerts(resolved);

COMMENT ON TABLE clinical_predictions IS 'Almacena las predicciones de riesgos clinicos';
COMMENT ON TABLE clinical_patients IS 'Almacena datos basicos de pacientes clinicos';
COMMENT ON TABLE clinical_alerts IS 'Almacena alertas generadas por el modulo clinico';