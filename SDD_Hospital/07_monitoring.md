# SDD – Monitorización y Calidad de Datos

### 1. Descripción funcional

Módulo transversal que garantiza la observabilidad del sistema y la integridad de los datos procesados. Incluye logging centralizado de todos los servicios, validación automática de calidad de datos en cada ejecución del pipeline, y mecanismos de alerta ante fallos de procesamiento. Este módulo responde al requisito del enunciado de monitorización y calidad de datos.

### 2. Logging centralizado

#### Arquitectura
Todos los servicios (API, pipeline, watchdog, scheduler) escriben logs en formato JSON estructurado a un volumen Docker compartido (`/logs/`). Cada servicio tiene su propio fichero de log con prefijo identificativo.

#### Formato de log

```json
{
  "timestamp": "2026-04-15T10:30:00.123Z",
  "service": "pipeline",
  "level": "INFO",
  "module": "cleaning",
  "message": "Limpieza completada: 4.987 registros procesados, 13 duplicados eliminados",
  "details": {
    "records_in": 5000,
    "records_out": 4987,
    "duplicates_removed": 13,
    "duration_seconds": 12.4
  }
}
```

#### Ficheros de log

| Fichero                  | Servicio   | Contenido                                        |
|--------------------------|------------|--------------------------------------------------|
| `/logs/api.log`          | API        | Requests, respuestas, errores, tiempos           |
| `/logs/pipeline.log`     | Pipeline   | Ingesta, limpieza, transformación, duraciones    |
| `/logs/watchdog.log`     | Watchdog   | Ficheros detectados, procesados, fallidos         |
| `/logs/scheduler.log`    | Scheduler  | Ejecuciones de informes, éxitos/fallos           |
| `/logs/inference.log`    | API/IA     | Predicciones realizadas, tiempos de inferencia   |

#### Implementación
- Librería estándar `logging` de Python con `JSONFormatter` personalizado.
- Configuración compartida en un módulo `common/logger.py` importado por todos los servicios.
- Nivel de log configurable vía variable de entorno `LOG_LEVEL` (default: `INFO`).
- Rotación de logs: `RotatingFileHandler` con max 10 MB por fichero, 5 backups.

### 3. Validación de calidad de datos

#### Momento de ejecución
La validación se ejecuta automáticamente como paso del pipeline, antes de cualquier transformación. También se puede disparar manualmente vía la API.

#### Checks para datos tabulares (CSV de pacientes)

| Check                     | Criterio                                             | Severidad |
|---------------------------|------------------------------------------------------|-----------|
| Duplicados                | `patient_id` repetido                                | `warning` |
| Campos nulos obligatorios | `patient_id`, `age`, `sex`, `admission_date` vacíos  | `error`   |
| Rango de edad             | `age` < 0 o > 120                                    | `error`   |
| Fecha futura              | `admission_date` posterior a hoy                     | `warning` |
| Departamento válido       | `department` fuera de lista permitida                | `warning` |
| Status válido             | `status` fuera de `[Ingresado, Alta, En observación]`| `warning` |

#### Checks para imágenes (radiografías)

| Check                     | Criterio                                            | Severidad |
|---------------------------|-----------------------------------------------------|-----------|
| Imagen corrupta           | No se puede abrir con PIL                           | `error`   |
| Dimensión inesperada      | No es 1024×1024 (original) ni 224×224 (procesada)   | `warning` |
| Formato incorrecto        | No es PNG                                           | `warning` |
| Duplicado por hash        | Hash MD5 ya existe en el sistema                    | `info`    |
| Máscara ausente           | Imagen sin máscara pulmonar correspondiente         | `warning` |

#### Acciones según severidad

| Severidad | Acción                                                         |
|-----------|----------------------------------------------------------------|
| `error`   | Registro se rechaza. Fichero movido a `/data/quarantine/`. Se registra en `data_quality_log`. |
| `warning` | Registro se procesa pero se registra la incidencia. Visible en dashboard. |
| `info`    | Solo registro informativo. No bloquea procesamiento.           |

### 4. Tabla `data_quality_log`

```sql
CREATE TABLE data_quality_log (
    log_id        SERIAL PRIMARY KEY,
    timestamp     TIMESTAMP DEFAULT NOW(),
    source_file   VARCHAR(255),
    record_id     VARCHAR(100),
    issue_type    VARCHAR(50),    -- 'duplicate', 'null_field', 'corrupt_image', 'invalid_range', etc.
    severity      VARCHAR(20),    -- 'error', 'warning', 'info'
    description   TEXT,
    resolved      BOOLEAN DEFAULT FALSE,
    resolved_at   TIMESTAMP,
    resolved_by   VARCHAR(100)
);
```

### 5. Alertas ante fallos de procesamiento

Cuando el pipeline falla (excepción no controlada, timeout, servicio no disponible), se genera automáticamente:

1. **Log de error** con stack trace completo en el fichero correspondiente.
2. **Entrada en `alerts`** con `alert_type = 'pipeline_error'` y `severity = 'error'`.
3. **Entrada en `data_quality_log`** con los detalles del fallo.
4. **Visibilidad en dashboard**: la pestaña de alertas muestra los errores de pipeline junto con las alertas clínicas.

### 6. Métricas del pipeline (visibles en dashboard)

| Métrica                             | Fuente                    |
|-------------------------------------|---------------------------|
| Registros procesados (total/hoy)    | `data_quality_log` + PostgreSQL |
| Tasa de rechazo                     | `data_quality_log` (errores / total) |
| Duplicados detectados               | `data_quality_log` (type='duplicate') |
| Imágenes en cuarentena              | Conteo bucket `quarantine` de MinIO |
| Tiempo medio de procesamiento       | Logs del pipeline (campo `duration_seconds`) |
| Último pipeline ejecutado           | Último timestamp en logs del pipeline |

### 7. Tecnología

| Componente        | Tecnología               | Justificación                                                |
|-------------------|--------------------------|--------------------------------------------------------------|
| Logging           | Python `logging` + JSON  | Estándar, sin dependencias externas, fácil de parsear.       |
| Validación        | Python (custom module)   | Checks específicos del dominio, no necesita framework externo.|
| Almacenamiento    | PostgreSQL               | Los logs de calidad son datos estructurados consultables.    |
| Visualización     | Streamlit (pestaña)      | Integrado en el dashboard existente.                         |

**Justificación de no usar ELK/Prometheus**: para el alcance de este proyecto, un stack de observabilidad dedicado añadiría 3-4 contenedores adicionales y complejidad de configuración desproporcionada. El enfoque elegido (logs JSON + tabla en PostgreSQL + pestaña en dashboard) cumple los requisitos del enunciado con una solución pragmática y mantenible.

### 8. Criterios de aceptación

- [ ] Todos los servicios escriben logs en formato JSON en el volumen compartido.
- [ ] La validación de calidad detecta y registra: duplicados, campos nulos, imágenes corruptas.
- [ ] Los registros con severidad `error` se rechazan y se mueven a cuarentena.
- [ ] Los fallos de pipeline generan alertas automáticas visibles en el dashboard.
- [ ] La pestaña de calidad de datos del dashboard muestra las incidencias registradas.
- [ ] Los logs rotan automáticamente al superar 10 MB.
