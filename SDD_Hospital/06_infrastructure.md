# SDD – Infraestructura y Despliegue

### 1. Descripción funcional

Infraestructura completamente containerizada con Docker y orquestada con Docker Compose. El objetivo es que cualquier persona pueda levantar el sistema completo con un solo comando (`docker-compose up`). Cada componente del sistema corre en su propio contenedor con responsabilidades claramente separadas, comunicándose a través de una red interna Docker.

### 2. Servicios (contenedores)

| Servicio         | Imagen base              | Puerto expuesto | Responsabilidad                           |
|------------------|--------------------------|-----------------|-------------------------------------------|
| `api`            | `python:3.11-slim`       | `8000`          | API REST FastAPI + modelo IA en memoria   |
| `dashboard`      | `python:3.11-slim`       | `8501`          | Dashboard Streamlit                       |
| `postgres`       | `postgres:16-alpine`     | `5432`          | Base de datos relacional                  |
| `minio`          | `minio/minio:latest`     | `9000` / `9001` | Object storage (imágenes, informes)       |
| `pipeline`       | `python:3.11-slim` + PySpark | —           | Procesamiento batch de datos              |
| `watchdog`       | `python:3.11-slim`       | —               | Monitorización de nuevos ficheros + alertas + scheduler |

**Total: 6 contenedores** con responsabilidades claramente separadas.

### 3. Arquitectura de red

```
                    ┌─────────────────────────────────────┐
                    │        Docker Network: hospital_net  │
                    │                                      │
  Host :8000 ◄─────┤  api ◄──────► postgres               │
  Host :8501 ◄─────┤  dashboard ◄─► postgres               │
  Host :9000 ◄─────┤  minio                               │
  Host :9001 ◄─────┤  minio (console)                     │
                    │  pipeline ──► postgres + minio        │
                    │  watchdog ──► minio + api + postgres  │
                    └─────────────────────────────────────┘
```

Todos los servicios se comunican a través de la red interna `hospital_net`. Solo los puertos de la API, dashboard y MinIO se exponen al host.

### 4. Docker Compose: estructura

```yaml
version: "3.9"

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      retries: 5

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_ROOT_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_ROOT_PASSWORD}
    volumes:
      - minio_data:/data
    ports:
      - "9000:9000"
      - "9001:9001"

  api:
    build: ./services/api
    environment:
      - DATABASE_URL=postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=${MINIO_ROOT_USER}
      - MINIO_SECRET_KEY=${MINIO_ROOT_PASSWORD}
      - MODEL_PATH=/models/resnet50_v1.0.pth
    volumes:
      - ./models:/models:ro
      - shared_logs:/logs
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_started

  dashboard:
    build: ./services/dashboard
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - API_URL=http://api:8000
      - MINIO_ENDPOINT=minio:9000
    ports:
      - "8501:8501"
    depends_on:
      - api

  pipeline:
    build: ./services/pipeline
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=${MINIO_ROOT_USER}
      - MINIO_SECRET_KEY=${MINIO_ROOT_PASSWORD}
    volumes:
      - ./data:/data
      - shared_logs:/logs
    depends_on:
      postgres:
        condition: service_healthy
      minio:
        condition: service_started

  watchdog:
    build: ./services/watchdog
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
      - MINIO_ENDPOINT=minio:9000
      - API_URL=http://api:8000
      - REPORT_SCHEDULE=${REPORT_SCHEDULE:-0 6 * * *}
    volumes:
      - ./data/incoming:/data/incoming
      - ./data/processed:/data/processed
      - ./data/failed:/data/failed
      - shared_logs:/logs
    depends_on:
      - api

volumes:
  postgres_data:
  minio_data:
  shared_logs:

networks:
  default:
    name: hospital_net
```

### 5. Variables de entorno (`.env`)

```env
# PostgreSQL
POSTGRES_DB=hospital_db
POSTGRES_USER=hospital_admin
POSTGRES_PASSWORD=secure_password_here

# MinIO
MINIO_ROOT_USER=minio_admin
MINIO_ROOT_PASSWORD=minio_secure_password

# Scheduler
REPORT_SCHEDULE=0 6 * * *
```

Todas las credenciales se externalizan en un fichero `.env` (incluido en `.gitignore`, nunca en el repositorio). Se proporciona un `.env.example` con valores de ejemplo.

### 6. Volúmenes

| Volumen          | Propósito                                         | Tipo        |
|------------------|----------------------------------------------------|-------------|
| `postgres_data`  | Persistencia de la base de datos                   | Named volume|
| `minio_data`     | Persistencia del object storage                    | Named volume|
| `shared_logs`    | Logs centralizados accesibles por todos los servicios | Named volume|
| `./data`         | Directorio de datos (incoming, processed, failed)  | Bind mount  |
| `./models`       | Modelo IA entrenado (`.pth`)                       | Bind mount (ro) |
| `./sql`          | Scripts de inicialización de base de datos         | Bind mount  |

### 7. Estructura del repositorio

```
hospital-ai-system/
├── docker-compose.yml
├── .env.example
├── .gitignore
├── README.md
├── docs/
│   ├── SDD/
│   │   ├── 00_system_overview.md
│   │   ├── 01_data_pipeline.md
│   │   ├── 02_ai_model.md
│   │   ├── 03_api.md
│   │   ├── 04_automation.md
│   │   ├── 05_dashboard.md
│   │   └── 06_infrastructure.md
│   └── memoria_tecnica.md
├── sql/
│   └── init.sql
├── models/
│   └── resnet50_v1.0.pth
├── data/
│   ├── incoming/
│   ├── processed/
│   ├── failed/
│   └── raw/
│       ├── patients.csv
│       └── generate_patients.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
└── services/
    ├── api/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── app/
    │       ├── main.py
    │       ├── model.py
    │       ├── schemas.py
    │       └── database.py
    ├── dashboard/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── app/
    │       └── dashboard.py
    ├── pipeline/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── app/
    │       ├── ingestion.py
    │       ├── cleaning.py
    │       ├── preprocessing.py
    │       └── quality.py
    └── watchdog/
        ├── Dockerfile
        ├── requirements.txt
        └── app/
            ├── watcher.py
            ├── alerts.py
            └── reports.py
```

### 8. Healthchecks y dependencias

- PostgreSQL incluye healthcheck (`pg_isready`). Los servicios que dependen de la BD esperan a que esté healthy.
- MinIO se valida con `condition: service_started` (no expone healthcheck nativo fácilmente).
- La API depende de PostgreSQL y MinIO; el dashboard depende de la API; el watchdog depende de la API.

### 9. Instrucciones de despliegue (README)

```bash
# 1. Clonar el repositorio
git clone https://github.com/equipo/hospital-ai-system.git
cd hospital-ai-system

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env con las credenciales deseadas

# 3. Levantar el sistema completo
docker-compose up --build

# 4. Acceder a los servicios
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
# MinIO:     http://localhost:9001

# 5. Ejecutar pipeline inicial (carga de datos)
docker-compose exec pipeline python app/ingestion.py

# 6. Parar el sistema
docker-compose down
```

### 10. Criterios de aceptación

- [ ] `docker-compose up --build` levanta los 6 servicios sin errores.
- [ ] PostgreSQL está accesible y contiene las tablas definidas en `init.sql`.
- [ ] MinIO está accesible y los buckets se crean automáticamente al iniciar.
- [ ] La API responde en `http://localhost:8000/docs`.
- [ ] El dashboard es accesible en `http://localhost:8501`.
- [ ] Los logs de todos los servicios se escriben en el volumen compartido `shared_logs`.
- [ ] Las variables de entorno están externalizadas en `.env` (no hardcodeadas).
- [ ] El sistema puede pararse y reiniciarse sin pérdida de datos (persistencia en volúmenes).
