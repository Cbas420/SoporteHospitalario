# Las 3 Áreas del Proyecto - Sistema Hospitalario AI

## Resumen Visual

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROYECTO HOSPITALARIO                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  1. IA - CLASIFICACIÓN DE RADIOGRAFÍAS                  │   │
│  │  ─────────────────────────────────────────────            │   │
│  │  train_pipeline.py                                        │   │
│  │  src/model/ (classifier.py, trainer.py)                  │   │
│  │  src/evaluation/evaluator.py                              │   │
│  │  src/api/inference.py                                     │   │
│  │  models/*.keras                                           │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  2. INFRAESTRUCTURA - BIG DATA                          │   │
│  │  ─────────────────────────────────────────────            │   │
│  │  docker-compose.yml                                       │   │
│  │  Dockerfile                                               │   │
│  │  mnt/postgres/                                            │   │
│  │  mnt/minio/                                               │   │
│  │  sql/init.sql                                             │   │
│  │  config/settings.py                                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  3. DATOS + AUTOMATIZACIÓN + DASHBOARD                  │   │
│  │  ─────────────────────────────────────────────            │   │
│  │  src/data/repository.py  →  CRUD pacientes/predicciones  │   │
│  │  src/data/pipeline.py   →  Limpieza de datos            │   │
│  │  src/automation/scheduler.py →  Scheduler (5 min)       │   │
│  │  src/automation/watchdog_service.py →  Detector archivos │   │
│  │  streamlit_app.py     →  Dashboard visual               │   │
│  │  src/api/app.py       →  API REST (expone todo)         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Áreas Detalladas

### 1. IA - Clasificación de Radiografías (Deep Learning)

| Archivo/Carpeta | Función |
|-----------------|---------|
| `src/model/classifier.py` | Define la arquitectura ResNet50 |
| `src/model/trainer.py` | Lógica de entrenamiento |
| `src/evaluation/evaluator.py` | Métricas (accuracy, recall, F1, matriz de confusión) |
| `src/api/inference.py` | Predicción en tiempo real (carga modelo, procesa imagen) |
| `train_pipeline.py` | Script principal de entrenamiento |
| `models/chest_xray_classifier.keras` | Modelo ya entrenado |

**Flujo**: `train_pipeline.py` → entrena → `models/*.keras` → usado por `inference.py` para predecir

---

### 2. Infraestructura - Big Data

| Archivo/Carpeta | Función |
|-----------------|---------|
| `docker-compose.yml` | Define servicios: PostgreSQL, MinIO, API, Dashboard |
| `Dockerfile` | Imagen del contenedor Python |
| `mnt/postgres/` | Archivos de la base de datos PostgreSQL |
| `mnt/minio/` | Archivos del object storage MinIO |
| `sql/init.sql` | Esquema de base de datos (tablas patients, predictions, alerts) |
| `config/settings.py` | Configuración centralizada (conexiones BD, buckets MinIO) |

**Servicios Docker**:
- **PostgreSQL** (puerto 5432): Base de datos relacional
- **MinIO** (puertos 9000/9001): Almacenamiento de imágenes (compatible S3)

---

### 3. Gestión de Datos + Automatización + Dashboard

#### 3a. Gestión de Pacientes y Datos

| Archivo/Carpeta | Función |
|-----------------|---------|
| `src/data/repository.py` | **CRUD de pacientes, predicciones, alertas** (tablas SQL) |
| `src/data/pipeline.py` | Limpia y transforma datos de pacientes |
| `src/data/loader.py` | Carga y split de imágenes para entrenamiento |
| `generate_sample_data.py` | Genera datos sintéticos de prueba |

#### 3b. Automatización

| Archivo/Carpeta | Función |
|-----------------|---------|
| `src/automation/scheduler.py` | Scheduler (cada 5 min): procesa imágenes, genera informes, alertas |
| `src/automation/watchdog_service.py` | Watchdog: detecta archivos nuevos en carpetas |
| `src/api/app.py` | API REST con endpoints para todo |

#### 3c. Dashboard (Visualización)

| Archivo/Carpeta | Función |
|-----------------|---------|
| `streamlit_app.py` | Dashboard web con 5 vistas: Inicio, Diagnósticos, Modelo IA, Alertas, Calidad de Datos |

---

## API REST: Endpoints

| Endpoint | Método | Pertenece a | Función |
|----------|--------|-------------|---------|
| `/predict` | POST | IA + Datos | Clasifica una radiografía |
| `/predict/batch` | POST | IA | Clasifica múltiples imágenes |
| `/patients` | GET | Datos | Lista pacientes |
| `/patients/{id}` | GET | Datos | Obtiene un paciente |
| `/stats` | GET | Datos | Estadísticas (KPIs) |
| `/predictions/recent` | GET | Datos | Predicciones recientes |
| `/alerts` | GET | Automatización | Lista alertas |
| `/alerts/{id}` | PATCH | Automatización | Resuelve una alerta |
| `/model/info` | GET | IA | Info del modelo |
| `/evaluation` | GET | IA | Métricas del modelo |
| `/health` | GET | Sistema | Health check |

---

## Dependencias entre Áreas

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DEPENDENCIAS DETECTADAS                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                   │
│   src/model/          →  config/, src/utils                           │
│   src/evaluation/    →  config/, src/utils                          │
│   src/data/           →  config/, src/utils                          │
│                                                                   │
│   ═══════════════════════════════════════════════════════════════════   │
│                                                                   │
│   src/api/            →  config/, src/utils, src/api/inference       │
│                       →  src/automation/scheduler                    │
│                       →  src/data/pipeline, src/data/repository       │
│                                                                   │
│   ═══════════════════════════════════════════════════════════════════   │
│                                                                   │
│   src/automation/     →  config/, src/utils                          │
│   (watchdog)         →  src/api/inference  ←─┐                      │
│                       →  src/data/pipeline    →──┐                   │
│                       →  src/data/repository    →──┘                 │
│                                                                   │
│   src/automation/     →  config/, src/utils                          │
│   (scheduler)        →  src/api/inference      ←─┐                  │
│                       →  src/data/pipeline    →──┤                   │
│                       →  src/data/repository    →──┘                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Notas Importantes

1. **Clasificación de pacientes**: No existe como tal en el código actual. El sistema:
   - Almacena pacientes en base de datos
   - Clasifica radiografías (COVID/Normal/Neumonía)
   - NO predice riesgo de pacientes

2. **Modelo actual**: Solo acepta imágenes de radiografías. Para clasificar pacientes por riesgo, se necesitaría:
   - Datos tabulares del paciente (edad, síntomas, historial)
   - Un modelo adicional o ampliar el existente

3. **Orden de ejecución recomendado**:
   - `docker-compose up db minio` (primero servicios de datos)
   - `python generate_sample_data.py --num-per-class 50` (generar datos)
   - `python train_pipeline.py --epochs 20 --finetune-epochs 15` (entrenar)
   - `docker-compose up api dashboard` (levantar servicios)
