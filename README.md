# 🏥 Módulo de IA — Clasificación de Radiografías de Tórax

## Sistema Inteligente de Soporte Hospitalario

Módulo de Deep Learning para clasificación automática de radiografías de tórax en tres categorías clínicas: **Sana (Normal)**, **Neumonía** y **COVID-19**.

---

## 📋 Tabla de Contenidos

- [Descripción](#descripción)
- [Arquitectura del Modelo](#arquitectura-del-modelo)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación y Uso](#instalación-y-uso)
- [API REST](#api-rest)
- [Integración con el Sistema](#integración-con-el-sistema)
- [Evaluación y Métricas](#evaluación-y-métricas)
- [Análisis Clínico](#análisis-clínico)
- [Decisiones Técnicas](#decisiones-técnicas)
- [Limitaciones](#limitaciones)
- [Mejoras Futuras](#mejoras-futuras)

---

## Descripción

Este módulo implementa un sistema de clasificación automática de radiografías de tórax utilizando Transfer Learning con ResNet50. Está diseñado como componente autónomo del Sistema Inteligente de Soporte Hospitalario, con interfaces claras para integración con el resto de módulos (pipeline de datos, API, dashboard, automatización).

### Capacidades

- Clasificación en 3 categorías: COVID-19, Normal, Neumonía
- API REST para predicciones en tiempo real (< 1s por imagen)
- Evaluación completa con métricas clínicas
- Análisis de impacto de errores en contexto hospitalario
- Containerización completa con Docker

---

## Arquitectura del Modelo

```
Imagen (224x224x3)
       │
       ▼
┌──────────────────┐
│  ResNet50 Base   │  ← Pesos preentrenados de ImageNet
│  (175 capas)     │    Capas 1-140: congeladas
│                  │    Capas 140-175: fine-tuning
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  Global Avg Pool │  ← Reduce dimensionalidad (2048 features)
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  Dense(256)      │
│  BatchNorm       │  ← Capas personalizadas
│  ReLU            │
│  Dropout(0.5)    │
└──────────────────┘
       │
       ▼
┌──────────────────┐
│  Dense(3)        │
│  Softmax         │  ← Salida: [COVID19, Normal, Pneumonia]
└──────────────────┘
```

### Entrenamiento en 2 fases

1. **Transfer Learning** (backbone congelado): se entrenan solo las capas densas personalizadas con LR=1e-4.
2. **Fine-Tuning** (capas profundas descongeladas): se descongelan los últimos bloques conv5 de ResNet50 con LR=1e-5 para adaptar features de alto nivel al dominio médico.

---

## Estructura del Proyecto

```
ai_model/
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuración centralizada
├── data/
│   ├── raw/                     # Dataset original (metadata/ + xrays/COVID|Normal|Lung_Opacity|Viral Pneumonia)
│   ├── processed/               # Dataset dividido (train/, val/, test/)
│   └── sample/                  # Datos de muestra para testing
├── models/                      # Modelo entrenado y artefactos de evaluación
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py            # Carga, split, augmentation
│   ├── model/
│   │   ├── __init__.py
│   │   ├── classifier.py        # Arquitectura ResNet50 + capas custom
│   │   └── trainer.py           # Orquestador de entrenamiento
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py         # Métricas, confusion matrix, análisis clínico
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py               # FastAPI endpoints
│   │   └── inference.py         # Servicio de predicción
│   └── utils/
│       ├── __init__.py
│       └── logger.py            # Logging centralizado
├── tests/
│   ├── conftest.py
│   ├── test_data_loader.py
│   ├── test_evaluator.py
│   └── test_inference.py
├── notebooks/                   # Jupyter notebooks de exploración
├── docs/                        # Documentación adicional
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── train_pipeline.py            # Script principal de entrenamiento
├── generate_sample_data.py      # Generador de datos de prueba
├── .env.example
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Requisitos

- Python 3.11+
- Docker y Docker Compose (para despliegue containerizado)
- ~4GB RAM mínimo (6GB recomendado para entrenamiento)
- Dataset de radiografías de tórax (o usar datos de muestra)

---

## Instalación y Uso

### Opción 1: Docker (recomendado)

```bash
# 1. Generar datos de muestra (si no tienes dataset real)
docker-compose --profile setup run ai-sample-data

# 2. Entrenar el modelo
docker-compose --profile training run ai-train

# 3. Levantar la API
docker-compose up ai-model
```

### Opción 2: Local

```bash
# 1. Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Generar datos de muestra (opcional)
python generate_sample_data.py --num-per-class 50

# 4. Entrenar
python train_pipeline.py --epochs 20

# 5. Levantar API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Dataset real

Colocar el dataset original en `data/raw/` con la siguiente estructura:

```
data/raw/
├── patients.csv
├── README.md.txt
├── metadata/
│   ├── COVID.metadata.xlsx
│   ├── Normal.metadata.xlsx
│   ├── Lung_Opacity.metadata.xlsx
│   └── Viral Pneumonia.metadata.xlsx
└── xrays/
    ├── COVID/
    │   ├── images/
    │   └── masks/
    ├── Normal/
    │   ├── images/
    │   └── masks/
    ├── Lung_Opacity/
    │   ├── images/
    │   └── masks/
    └── Viral Pneumonia/
        ├── images/
        └── masks/
```

La carpeta física se llama `COVID`, pero el pipeline la mapea a la clase clínica `COVID19` para entrenamiento e inferencia.

Fuentes recomendadas:
- [COVID-19 Radiography Database (Kaggle)](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## API REST

Una vez levantada, la API está disponible en `http://localhost:8000`.

### Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/predict` | Clasificar una radiografía |
| `GET` | `/health` | Health check |
| `GET` | `/model/info` | Información del modelo |
| `GET` | `/evaluation` | Último informe de evaluación |

### Ejemplo de uso

```bash
# Clasificar una imagen
curl -X POST "http://localhost:8000/predict" \
  -F "file=@radiografia.jpg"

# Respuesta:
{
  "prediction": "COVID19",
  "confidence": 0.923,
  "probabilities": {
    "COVID19": 0.923,
    "Normal": 0.042,
    "Pneumonia": 0.035
  },
  "requires_review": false,
  "inference_time_ms": 45.2,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

### Documentación interactiva

Swagger UI disponible en: `http://localhost:8000/docs`

---

## Integración con el Sistema

Este módulo se conecta con el resto del sistema hospitalario a través de:

### Con el Pipeline de Datos
- **Entrada**: Las imágenes llegan a `data/raw/` desde el pipeline de ingestión (MinIO/S3 → volumen Docker compartido).
- **Salida**: Los resultados de clasificación se pueden escribir en PostgreSQL a través de la API.

### Con la API General (FastAPI)
- Este módulo expone sus propios endpoints que el API gateway del sistema puede proxificar.
- El endpoint `/predict` es consumido directamente por otros módulos.

### Con el Dashboard
- El dashboard consume `/evaluation` para mostrar métricas y matrices de confusión.
- Los resultados de `/predict` se visualizan en tiempo real.

### Con la Automatización
- El pipeline de automatización puede invocar `/predict` para procesar lotes de radiografías nuevas.
- Las alertas se generan cuando `requires_review: true`.

### Diagrama de integración

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   MinIO /   │────▶│  Pipeline de │────▶│  data/raw/  │
│   Ingestión │     │    Datos     │     │  (volumen)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                                                ▼
                                    ┌───────────────────┐
                                    │  MÓDULO DE IA     │
                                    │  ┌─────────────┐  │
                                    │  │ /predict    │  │◀── Dashboard
                                    │  │ /evaluation │  │◀── Automatización
                                    │  │ /health     │  │◀── Monitorización
                                    │  └─────────────┘  │
                                    └────────┬──────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │   PostgreSQL    │
                                    │  (resultados)   │
                                    └─────────────────┘
```

---

## Evaluación y Métricas

El pipeline genera automáticamente:

- **Accuracy, Precision, Recall, F1-Score** (global y por clase)
- **Especificidad** por clase
- **Matriz de confusión** (PNG)
- **Curvas de entrenamiento** (PNG)
- **Informe completo** (JSON)

Todos los artefactos se guardan en `models/`.

---

## Análisis Clínico

El módulo incluye un análisis obligatorio del impacto de errores:

- **Falso Negativo COVID-19**: Riesgo ALTO — paciente contagioso no aislado.
- **Falso Negativo Neumonía**: Riesgo MEDIO-ALTO — retraso en tratamiento.
- **Confusión COVID ↔ Neumonía**: Especialmente relevante por patrones radiológicos similares.
- **Falsos Positivos**: Sobrecarga del sistema sanitario.

Predicciones con confianza < 60% se marcan automáticamente como "requiere revisión manual".

---

## Decisiones Técnicas

| Decisión | Justificación |
|----------|---------------|
| ResNet50 | Balance rendimiento/coste. Validado en imágenes médicas. Conexiones residuales evitan vanishing gradient. |
| Transfer Learning 2 fases | Evita corromper pesos preentrenados. Fase 1 rápida, fase 2 adapta al dominio. |
| Dropout 0.5 | Regularización agresiva necesaria con datasets médicos pequeños. |
| Adam optimizer | Convergencia rápida, adaptativo, estándar para transfer learning. |
| Data Augmentation | Sin vertical flip (las radiografías tienen orientación). Rotaciones y zoom moderados. |
| FastAPI | Validación automática, docs OpenAPI, async, estándar del SDD. |

---

## Limitaciones

1. **Dataset limitado**: Los datos de muestra son sintéticos. El rendimiento real depende de la calidad y cantidad del dataset clínico.
2. **Sesgo potencial**: Si el dataset no representa equitativamente poblaciones diversas, el modelo puede tener sesgo.
3. **No sustituye diagnóstico médico**: Es una herramienta de apoyo, nunca de decisión final.
4. **Sin explicabilidad**: El modelo actual no proporciona mapas de activación (Grad-CAM). Se deja como mejora futura.
5. **Clase COVID-19**: Los patrones radiológicos de COVID-19 evolucionan con nuevas variantes.

---

## Mejoras Futuras

- **EfficientNet-B4**: Mayor rendimiento potencial con menos parámetros.
- **Grad-CAM**: Mapas de activación para explicar qué regiones de la imagen influyen en la decisión.
- **Class Weights / Oversampling**: Manejo de desbalanceo de clases.
- **Ensemble**: Combinar múltiples modelos para mayor robustez.
- **MLflow**: Tracking de experimentos para comparar configuraciones.
- **Integración real con MinIO**: Lectura directa desde object storage sin volumen intermedio.

---

## Tests

```bash
# Ejecutar todos los tests
pytest tests/ -v

# Solo tests de datos
pytest tests/test_data_loader.py -v

# Solo tests de evaluación
pytest tests/test_evaluator.py -v
```

---

## Licencia

Proyecto académico — Sistema Inteligente de Soporte Hospitalario.
