# SDD – API REST

### 1. Descripción funcional

API REST que expone los servicios del sistema hospitalario: predicción de radiografías, consulta de pacientes y diagnósticos, estadísticas operativas y gestión de alertas. Actúa como punto de entrada principal para el dashboard y para integraciones externas. Construida con FastAPI por su rendimiento asíncrono, validación automática con Pydantic y documentación OpenAPI generada.

### 2. Inputs

| Input                       | Formato                  | Origen                  |
|-----------------------------|--------------------------|-------------------------|
| Imagen radiografía          | `multipart/form-data`    | Upload directo (cliente) |
| Parámetros de consulta      | Query params / JSON body | Dashboard / cliente     |
| Datos del modelo IA         | Fichero `.pth` en memoria| Volumen Docker          |

### 3. Outputs

| Output                      | Formato        | Destino                |
|-----------------------------|----------------|------------------------|
| Predicción + probabilidades | JSON           | Cliente / Dashboard    |
| Listado de pacientes        | JSON (paginado)| Dashboard              |
| Estadísticas del sistema    | JSON           | Dashboard              |
| Alertas clínicas            | JSON           | Dashboard              |

### 4. Endpoints

#### 4.1 Predicción

```
POST /predict
Content-Type: multipart/form-data
```

**Input**: campo `file` con imagen PNG/JPEG de la radiografía, campo opcional `patient_id` (UUID).

**Flujo interno**:
1. Recibe la imagen vía `UploadFile`.
2. Sube la imagen original al bucket `raw-xrays` de MinIO.
3. Ejecuta preprocesamiento: resize 224×224, aplicación de máscara pulmonar, normalización ImageNet.
4. Ejecuta inferencia con el modelo CNN cargado en memoria.
5. Guarda la predicción en la tabla `diagnoses` de PostgreSQL.
6. Si predicción = COVID-19 y probabilidad > 0.85, genera alerta en tabla `alerts`.
7. Retorna respuesta.

**Output** (200 OK):
```json
{
  "prediction": "COVID-19",
  "probabilities": {
    "Sana": 0.05,
    "Neumonía": 0.10,
    "COVID-19": 0.85
  },
  "confidence": 0.85,
  "patient_id": "a1b2c3d4-...",
  "xray_id": "xray_20260415_001.png",
  "model_version": "resnet50_v1.0",
  "timestamp": "2026-04-15T10:30:00Z",
  "alert_generated": true
}
```

**Errores**:
- `400`: imagen no válida (formato incorrecto, corrupta).
- `422`: patient_id no encontrado en PostgreSQL (si se proporcionó).
- `500`: error interno en inferencia.

#### 4.2 Pacientes

```
GET /patients
```

**Query params**: `department` (filtro), `status` (filtro), `diagnosis` (filtro), `page` (default 1), `per_page` (default 20).

**Output** (200 OK):
```json
{
  "patients": [...],
  "total": 5000,
  "page": 1,
  "per_page": 20
}
```

```
GET /patients/{patient_id}
```

**Output**: datos completos del paciente + historial de diagnósticos asociados.

#### 4.3 Estadísticas

```
GET /stats
```

**Output** (200 OK):
```json
{
  "total_patients": 5000,
  "total_xrays_processed": 21165,
  "xrays_today": 45,
  "diagnosis_distribution": {
    "Sana": 10192,
    "Neumonía": 7357,
    "COVID-19": 3616
  },
  "active_alerts": 12,
  "departments": {
    "UCI": {"patients": 120, "covid_cases": 35},
    "Urgencias": {"patients": 450, "covid_cases": 89}
  },
  "model_version": "resnet50_v1.0",
  "system_uptime": "3d 14h 22m"
}
```

#### 4.4 Alertas

```
GET /alerts
```

**Query params**: `status` (pending/resolved), `severity`, `page`, `per_page`.

```
PATCH /alerts/{alert_id}
```

**Body**: `{"status": "resolved"}` — permite marcar alertas como revisadas desde el dashboard.

### 5. Tecnología

| Componente            | Tecnología         | Justificación                                              |
|-----------------------|--------------------|-------------------------------------------------------------|
| Framework             | FastAPI            | Async nativo, validación Pydantic, docs OpenAPI automáticas |
| Validación            | Pydantic v2        | Esquemas tipados para request/response                      |
| Servidor              | Uvicorn            | Servidor ASGI de alto rendimiento                           |
| Conexión DB           | SQLAlchemy + asyncpg| ORM async para PostgreSQL                                  |
| Conexión MinIO        | boto3 / minio-py   | SDK S3-compatible                                           |

### 6. Restricciones técnicas

- La API carga el modelo CNN una sola vez al iniciar (evita recarga por request).
- Timeout máximo de 30 segundos por request de predicción.
- Tamaño máximo de upload: 10 MB por imagen.
- La API no implementa autenticación completa (fuera de alcance), pero incluye un header `X-API-Key` básico como demostración.

### 7. Criterios de aceptación

- [ ] `POST /predict` recibe una imagen, ejecuta inferencia y devuelve clase + probabilidades en < 500ms.
- [ ] `GET /patients` devuelve resultados paginados con filtros funcionales.
- [ ] `GET /stats` devuelve estadísticas agregadas correctas.
- [ ] `GET /alerts` lista alertas filtradas por estado.
- [ ] `PATCH /alerts/{id}` actualiza el estado de una alerta.
- [ ] La documentación OpenAPI es accesible en `/docs` (Swagger UI).
- [ ] Los errores devuelven códigos HTTP apropiados con mensajes descriptivos.
