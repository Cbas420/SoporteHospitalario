# SDD – Automatización de Procesos

### 1. Descripción funcional

Módulo de automatización que mejora la eficiencia operativa del hospital mediante tres mecanismos: vigilancia automática de nuevos ficheros (watchdog), generación de alertas clínicas ante predicciones críticas, e informes periódicos programados. Todos los procesos corren dentro de contenedores Docker y se orquestan con APScheduler.

### 2. Automatización 1: Watchdog de ingesta

#### Descripción
Proceso que monitoriza en tiempo real un directorio de entrada (`/data/incoming/`) para detectar nuevos ficheros (CSV de pacientes o imágenes PNG de radiografías). Al detectar un fichero nuevo, dispara automáticamente el pipeline correspondiente.

#### Inputs
| Input            | Formato    | Origen                          |
|------------------|------------|---------------------------------|
| Nuevo fichero    | CSV o PNG  | Directorio `/data/incoming/`    |

#### Outputs
| Output                        | Destino         |
|-------------------------------|-----------------|
| Datos procesados              | PostgreSQL      |
| Imagen almacenada             | MinIO           |
| Predicción (si es imagen)     | PostgreSQL + API|
| Log de procesamiento          | `/logs/` + PostgreSQL |

#### Flujo
1. Watchdog (librería `watchdog` de Python) detecta evento `FileCreated` en el directorio vigilado.
2. Identifica el tipo de fichero por extensión.
3. **Si CSV**: ejecuta pipeline de limpieza PySpark → inserta en PostgreSQL.
4. **Si PNG/JPEG**: sube a MinIO (`raw-xrays`), ejecuta preprocesamiento, lanza inferencia IA, guarda resultado en `diagnoses`.
5. Mueve el fichero original a `/data/processed/` tras éxito, o a `/data/failed/` si hay error.
6. Registra el evento en el log centralizado.

#### Tecnología
- `watchdog` (Python): monitorización de filesystem, ligero y sin dependencias pesadas.
- Corre como proceso persistente dentro de su propio contenedor Docker.

### 3. Automatización 2: Alertas clínicas

#### Descripción
Sistema de alertas automáticas que se dispara cuando el modelo de IA clasifica una radiografía como COVID-19 con probabilidad superior al umbral configurado. Las alertas se persisten en PostgreSQL y son visibles en el dashboard.

#### Inputs
| Input                    | Formato              | Origen                |
|--------------------------|----------------------|-----------------------|
| Resultado de predicción  | Dict (clase + probs) | Módulo de inferencia  |

#### Outputs
| Output              | Destino                      |
|---------------------|------------------------------|
| Registro de alerta  | Tabla `alerts` (PostgreSQL)  |
| Log de alerta       | `/logs/alerts.log`           |
| Notificación visual | Dashboard (pestaña alertas)  |

#### Reglas de generación de alertas

| Condición                                | Tipo de alerta    | Severidad |
|------------------------------------------|-------------------|-----------|
| Predicción COVID-19 con prob > 0.85      | `covid_high_conf` | `critical`|
| Predicción COVID-19 con prob 0.60-0.85   | `covid_review`    | `warning` |
| Error en pipeline de procesamiento       | `pipeline_error`  | `error`   |
| Imagen no supera validación de calidad   | `quality_issue`   | `info`    |

#### Esquema de alerta (tabla `alerts`)
```json
{
  "alert_id": 42,
  "diagnosis_id": 1205,
  "patient_id": "a1b2c3d4-...",
  "alert_type": "covid_high_conf",
  "severity": "critical",
  "message": "Paciente A. García (UCI) clasificado como COVID-19 con 92% de confianza. Requiere revisión inmediata.",
  "status": "pending",
  "created_at": "2026-04-15T10:30:00Z",
  "resolved_at": null
}
```

### 4. Automatización 3: Informes periódicos

#### Descripción
Generación automática de informes resumen del estado operativo del hospital. Se ejecuta cada 24 horas (configurable) mediante APScheduler y produce un informe en formato JSON almacenado en MinIO.

#### Inputs
| Input                    | Formato          | Origen      |
|--------------------------|------------------|-------------|
| Datos de pacientes       | Consulta SQL     | PostgreSQL  |
| Datos de diagnósticos    | Consulta SQL     | PostgreSQL  |
| Datos de alertas         | Consulta SQL     | PostgreSQL  |

#### Outputs
| Output               | Formato  | Destino                         |
|----------------------|----------|---------------------------------|
| Informe diario       | JSON     | MinIO (bucket `reports`)        |
| Log de generación    | Texto    | `/logs/reports.log`             |

#### Contenido del informe
- Fecha y rango temporal cubierto.
- Número total de radiografías procesadas en el período.
- Distribución de diagnósticos (Sana / Neumonía / COVID-19).
- Número de alertas generadas y su desglose por severidad.
- Pacientes nuevos ingresados por departamento.
- Incidencias de calidad de datos detectadas.
- Estado general del pipeline (OK / errores).

#### Scheduler
- **Tecnología**: APScheduler (Advanced Python Scheduler).
- **Justificación**: ligero, se integra nativamente en Python, no requiere infraestructura adicional (a diferencia de Airflow o Celery, que serían overengineering para este alcance).
- **Configuración**: cron trigger, ejecuta a las 06:00 cada día (configurable vía variable de entorno `REPORT_SCHEDULE`).

### 5. Logging centralizado

Todas las automatizaciones escriben logs estructurados (formato JSON) en un volumen Docker compartido (`/logs/`):

```json
{
  "timestamp": "2026-04-15T10:30:00Z",
  "service": "watchdog",
  "level": "INFO",
  "message": "Nuevo fichero detectado: xray_0042.png",
  "details": {"action": "pipeline_triggered", "file_type": "image"}
}
```

Los logs se escriben con la librería estándar `logging` de Python, configurada con `JSONFormatter`. Cada servicio (watchdog, alertas, scheduler) tiene su propio logger con nombre identificativo.

### 6. Criterios de aceptación

- [ ] El watchdog detecta un nuevo fichero en `/data/incoming/` en menos de 10 segundos.
- [ ] Un PNG nuevo se procesa end-to-end (ingesta → inferencia → resultado en BD) sin intervención manual.
- [ ] Un CSV nuevo se limpia y carga en PostgreSQL automáticamente.
- [ ] Se genera una alerta `critical` cuando una predicción COVID-19 supera el 85% de confianza.
- [ ] Se genera una alerta `warning` para predicciones COVID-19 entre 60-85%.
- [ ] El informe diario se genera automáticamente y se almacena en MinIO.
- [ ] Todos los eventos de automatización quedan registrados en los logs centralizados.
- [ ] Los ficheros procesados se mueven a `/data/processed/`; los fallidos a `/data/failed/`.
