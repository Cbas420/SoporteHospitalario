# Resumen de Implementacion - Modulo Clinical AI

## Descripcion general

Se ha implementado un nuevo modulo de IA clinica para prediccion de riesgos basado en sintomas. Este modulo es **completamente independiente** del sistema de clasificacion de radiografias (COVID/Normal/Neumonia).

## Archivos implementados

### Estructura de carpetas

```
clinical_ai/
├── __init__.py                      # Package init
├── config/
│   ├── __init__.py                 # Package init
│   └── clinical_settings.py          # Configuracion centralizada
├── model/
│   ├── __init__.py                 # Package init
│   └── clinical_classifier.py         # Arquitectura del modelo MLP
├── api/
│   ├── __init__.py                 # Package init
│   └── clinical_api.py             # API REST (FastAPI)
├── data/                          # Datos del modulo
├── models/                        # Modelos entrenados
├── tests/
│   ├── __init__.py                 # Package init
│   └── test_clinical.py          # Tests unitarios
├── sql/
│   └── init_clinical.sql         # SQL de inicializacion BD
├── generate_clinical_data.py      # Generador de datos sintéticos
├── train_clinical_model.py        # Pipeline de entrenamiento
└── README.md                    # Documentacion
```

### Archivos nuevos (no en clinical_ai/)

```
SDD_Hospital/03_clinical_ai.md     # SDD del modulo clinico
```

## Archivos detallados

### 1. Configuracion

| Archivo | Descripcion |
|---------|-----------|
| `config/clinical_settings.py` | Configuracion: paths, features, diseases, API settings |

**Configuracion clave:**
- `SYMPTOM_COLUMNS`: 10 sintomas binarios
- `DISEASE_COLUMNS`: 3 enfermedades (diabetes, hypertension, heart_disease)
- `ALL_FEATURES`: 15 features (5 demograficos + 10 sintomas)
- `CLINICAL_API_PORT`: 8001

### 2. Modelo

| Archivo | Descripcion |
|---------|-----------|
| `model/clinical_classifier.py` | Arquitectura MLP, preprocesamiento |

**Arquitectura:**
- Input: 15 features
- Dense(64, ReLU) + BatchNorm + Dropout(0.3)
- Dense(32, ReLU) + BatchNorm + Dropout(0.3)
- Output: Dense(3, Sigmoid) - 3 enfermedades

### 3. Generador de datos

| Archivo | Descripcion |
|---------|-----------|
| `generate_clinical_data.py` | Genera pacientes sintéticos con sintomas y diagnosticos |

**Funciones:**
- `generate_patient()`: Genera un paciente con datos demograficos y sintomas
- `determine_risks()`: Calcula riesgos basados en correlaciones medicas reales
- `generate_clinical_dataset()`: Genera el dataset completo

**Correlaciones implementadas:**
- Diabetes: +riesgo con IMC alto, edad, fatiga, nausea
- Hipertension: +riesgo con IMC alto, edad, dolor cabeza, mareos
- Cardiaca: +riesgo con edad, dolor pecho, dificultad respiratoria

### 4. Entrenamiento

| Archivo | Descripcion |
|---------|-----------|
| `train_clinical_model.py` | Pipeline de entrenamiento con metricas |

**Pasos:**
1. Carga dataset CSV
2. Preprocesa features (normalizacion)
3. Divide 70/15/15 (train/val/test)
4. Entrena con Callbacks (EarlyStopping, ReduceLROnPlateau)
5. Evalua en test set
6. Genera metricas por enfermedad (precision, recall, F1)

### 5. API REST

| Archivo | Descripcion |
|---------|-----------|
| `api/clinical_api.py` | Endpoints FastAPI |

**Endpoints:**
- `POST /clinical/predict`: Predice riesgos clinicos
- `GET /clinical/info`: Informacion del modelo
- `GET /clinical/health`: Health check

**Request ejemplo:**
```json
{
  "age": 55,
  "sex": "M",
  "height_cm": 175,
  "weight_kg": 85,
  "has_cough": true,
  "has_chest_pain": true,
  "has_fatigue": true
}
```

**Response ejemplo:**
```json
{
  "diabetes_risk": true,
  "hypertension_risk": true,
  "heart_disease_risk": false,
  "probabilities": {
    "diabetes_risk": 0.72,
    "hypertension_risk": 0.65,
    "heart_disease_risk": 0.35
  },
  "confidence": 0.72,
  "risk_level": "moderado",
  "inference_time_ms": 12.5,
  "timestamp": "2026-04-22T19:00:00Z"
}
```

### 6. Tests

| Archivo | Descripcion |
|---------|-----------|
| `tests/test_clinical.py` | Tests unitarios |

**Test suites:**
- `TestClinicalPreprocessing`: Tests de preprocesamiento
- `TestClinicalModel`: Tests del modelo (build, prediction, training)

### 7. Base de datos

| Archivo | Descripcion |
|---------|-----------|
| `sql/init_clinical.sql` | SQL de inicializacion |

**Tablas:**
- `clinical_predictions`: Almacena predicciones
- `clinical_patients`: Datos basicos de pacientes
- `clinical_alerts`: Alertas clinicas

## Dependencias externas

### Requeridas (ya en requirements.txt)

- tensorflow==2.15.0
- fastapi==0.109.0
- uvicorn==0.27.0
- pandas==2.1.4
- numpy==1.26.3
- pytest==8.3.2

## Diferencias con el modulo de radiografias

| Aspecto | Radiografias | Clinico |
|---------|-----------|---------|
| Input | Imagen 224x224 | Features tabulares (15) |
| Modelo | ResNet50 (CNN) | MLP (red densa) |
| Clases | 3 (COVID/Normal/Neumonia) | 3 enfermedades multi-label |
| Puerto API | 8000 | 8001 |
| BD | PostgreSQL (hospital) | PostgreSQL (clinical) |
| Generador | generate_sample_data.py | generate_clinical_data.py |
| Entrenamiento | train_pipeline.py | train_clinical_model.py |

## Uso

### 1. Generar datos

```bash
python clinical_ai/generate_clinical_data.py --num-patients 1000
```

### 2. Entrenar modelo

```bash
python clinical_ai/train_clinical_model.py --epochs 50
```

### 3. Levantar API

```bash
python -m uvicorn clinical_ai.api.clinical_api:app --host 0.0.0.0 --port 8001
```

### 4. Hacer prediccion

```bash
curl -X POST "http://localhost:8001/clinical/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 55, "sex": "M", "height_cm": 175, "weight_kg": 85, "has_cough": true}'
```

## Pendiente

- [ ] Integracion con Docker Compose
- [ ] Vista en Dashboard Streamlit
- [ ] Inicializacion de BD PostgreSQL
- [ ] Tests de integracion

## Nota

El modulo clinico es **completamente independiente** del sistema de radiografias. No comparte:
- Modelo ni entrenamiento
- API ni endpoints
- Base de datos
- Dataset ni generador de datos