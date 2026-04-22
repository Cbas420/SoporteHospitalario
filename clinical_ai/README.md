# Clinical AI Module - Module de Prediccion Clinica

## Descripcion

Modulo de Inteligencia Artificial para prediccion de riesgos clinicos basado en sintomas y datos demograficos de pacientes. Este modulo opera de forma **independiente** del sistema de clasificacion de radiografias.

## Funcionalidades

- **Clasificacion multi-label**: Predice simultaneamente el riesgo de:
  - Diabetes
  - Hipertension
  - Enfermedades cardiacas

- **Entrada de datos estructurados**:
  - Datos demograficos: edad, sexo, altura, peso
  - Sintomas binarios: tos, dolor de pecho, fatiga, fiebre, mareos, etc.

## Estructura del Proyecto

```
clinical_ai/
├── config/
│   └── clinical_settings.py      # Configuracion centralizada
├── model/
│   └── clinical_classifier.py # Arquitectura del modelo
├── api/
│   └── clinical_api.py       # API REST (puerto 8001)
├── data/
│   └── clinical_dataset.csv   # Dataset generado
├── models/
│   └── clinical_risk_classifier.keras # Modelo entrenado
├── tests/
│   └── test_clinical.py     # Tests unitarios
├── generate_clinical_data.py  # Generador de datos
├── train_clinical_model.py   # Pipeline de entrenamiento
└── README.md                # Este archivo
```

## Uso

### 1. Generar datos de prueba

```bash
python clinical_ai/generate_clinical_data.py --num-patients 1000
```

### 2. Entrenar el modelo

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
  -d '{
    "age": 55,
    "sex": "M",
    "height_cm": 175,
    "weight_kg": 85,
    "has_cough": true,
    "has_chest_pain": true,
    "has_fatigue": true
  }'
```

### Respuesta de ejemplo

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

## Features de Entrada

| Feature | Tipo | Descripcion |
|---------|------|-------------|
| age | Numero | Edad del paciente (18-85) |
| sex | String | 'M' o 'F' |
| height_cm | Numero | Altura en centimetros |
| weight_kg | Numero | Peso en kilogramos |
| has_cough | Booleano | ¿Tiene tos? |
| has_chest_pain | Booleano | ¿Tiene dolor de pecho? |
| has_fatigue | Booleano | ¿Tiene fatiga? |
| has_fever | Booleano | ¿Tiene fiebre? |
| has_dizziness | Booleano | ¿Tiene mareos? |
| has_breathing_difficulty | Booleano | ¿Tiene dificultad respiratoria? |
| has_headache | Booleano | ¿Tiene dolor de cabeza? |
| has_nausea | Booleano | ¿Tiene nausea? |
| has_loss_of_appetite | Booleano | ¿Tiene perdida de apetito? |
| has_night_sweats | Booleano | ¿Tiene sudores nocturnos? |

## Modelo

- **Arquitectura**: MLP (Multi-Layer Perceptron)
- **Capas ocultas**: Dense(64, ReLU) -> Dense(32, ReLU)
- **Regularizacion**: L2 + Dropout(0.3) + BatchNormalization
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: Binary Crossentropy
- **Output**: Sigmoid (3 enfermedades, multi-label)

## Endpoints API

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/clinical/predict` | POST | Predice riesgos clinicos |
| `/clinical/info` | GET | Informacion del modelo |
| `/clinical/health` | GET | Health check |

## Base de Datos

El modulo usa su **propia base de datos PostgreSQL** (separada del sistema principal):

- **Host**: Configurado via variables de entorno
- **Tablas**: `clinical_predictions` (futuro)

## Integracion

- **Puerto API**: 8001 (separado del 8000 de radiografias)
- **Docker**: Servicio opcional en docker-compose (futuro)
- **Dashboard**: Vista separada en Streamlit (futuro)

## Tests

```bash
pytest clinical_ai/tests/ -v
```

## Requisitos

- Python 3.11+
- TensorFlow 2.15.0
- FastAPI
- Uvicorn
- Pandas
- NumPy
- Scikit-learn (para metricas)

## Notas

- Los datos son **sinteticos** y se generan para desarrollo/pruebas
- El modelo usa **correlaciones medicas simplificadas** para generar etiquetas
- Para produccion, se recomienda usar datos reales anonimizados