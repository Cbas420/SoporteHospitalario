# AGENTS.md - Sistema Hospitalario AI

## Comandos principales

```bash
# Generar datos de muestra
python generate_sample_data.py --num-per-class 50

# Entrenar modelo (2 fases: transfer learning + fine-tuning)
python train_pipeline.py --epochs 20 --finetune-epochs 15

# Omitir fine-tuning
python train_pipeline.py --no-finetune

# Solo evaluar modelo existente
python train_pipeline.py --skip-training

# Levantar API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Levantar dashboard
streamlit run streamlit_app.py --server.port 8501

# Tests
pytest tests/ -v
pytest tests/test_inference.py -v
```

## Docker

```bash
# Setup + entrenamiento
docker-compose --profile setup run ai-sample-data
docker-compose --profile training run ai-train

# Servicios
docker-compose up db minio api dashboard automation
```

## Arquitectura

- **API**: FastAPI en `src/api/app.py` (puerto 8000)
- **Dashboard**: Streamlit en `streamlit_app.py` (puerto 8501)
- **Automatización**: `src/automation/scheduler.py` (cada 5 min)
- **Watchdog**: `src/automation/watchdog_service.py`

## Configuración

`config/settings.py` es la fuente principal de configuración.

Directorios clave:
- `data/raw/` - Imágenes originales
- `data/processed/` - Dataset dividido
- `data/sample/` - Datos de prueba
- `models/` - Modelo entrenado y métricas

## Datos

Clases: `COVID19`, `Normal`, `Pneumonia`

Carpeta física `COVID/` se mapea a clase `COVID19`.

Augmentación: sin vertical flip (orientación fija en radiografías).

## API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/predict` | POST | Clasificar radiografía |
| `/health` | GET | Health check |
| `/model/info` | GET | Info del modelo |
| `/evaluation` | GET | Métricas |
| `/patients` | GET | Listar pacientes |
| `/stats` | GET | Estadísticas |

## Dependencias

- Python 3.11
- TensorFlow 2.15.0 (Windows: tensorflow-intel)
- PostgreSQL 16, MinIO
- FastAPI, Streamlit, SQLAlchemy, APScheduler

## Notas importantes

- El modelo usa formato `.keras` (Keras 3), no `.h5`
- Tests de inferencia se saltan si no hay modelo entrenado
- Predicciones con confianza < 60% se marcan como `requires_review`
- Umbral COVID crítico: 85%, warning: 60%
