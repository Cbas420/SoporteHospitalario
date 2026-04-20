# SDD – System Overview
## Sistema Inteligente de Soporte Hospitalario – laSalle Health Center

### 1. Descripción general

Sistema integral de soporte hospitalario que combina análisis de datos clínicos, clasificación automática de radiografías de tórax mediante Deep Learning y automatización de procesos operativos. El sistema ingesta, procesa y sirve datos tanto estructurados (registros de pacientes en PostgreSQL) como no estructurados (imágenes médicas en MinIO), exponiendo los resultados a través de una API REST y un dashboard interactivo.

### 2. Problema que resuelve

El hospital laSalle Health Center genera diariamente grandes volúmenes de datos clínicos y operativos sin herramientas que permitan extraer conocimiento, detectar patrones ni automatizar tareas. Este sistema aporta:

- Clasificación automática de radiografías de tórax en tres categorías (Sana, Neumonía, COVID-19) para apoyar al equipo médico.
- Pipeline de datos escalable para ingesta, limpieza y transformación de datos clínicos.
- Automatización de alertas, informes y procesamiento de nuevas imágenes.
- Dashboard centralizado para la toma de decisiones médicas y operativas.

### 3. Arquitectura de alto nivel

```
[Ingesta]                     [Almacenamiento]              [Servicio]
  │                               │                             │
  ├─ CSV pacientes ──────► PostgreSQL (estructurados)    ► FastAPI REST
  ├─ Radiografías ───────► MinIO (imágenes + máscaras)   ► Dashboard Streamlit
  └─ Watchdog auto ──────► Pipeline PySpark              ► Alertas automáticas
                               │
                        [Procesamiento]
                               │
                    ├─ Preprocesamiento imágenes
                    ├─ Inferencia CNN (ResNet50)
                    ├─ Validación de calidad de datos
                    └─ Logging centralizado
```

### 4. Componentes del sistema

| Componente           | Responsabilidad                                      | Tecnología principal   |
|----------------------|------------------------------------------------------|------------------------|
| Data Pipeline        | Ingesta, limpieza, transformación                    | Python, PySpark        |
| Almacenamiento SQL   | Pacientes, diagnósticos, alertas, logs de calidad    | PostgreSQL             |
| Almacenamiento NoSQL | Imágenes médicas, máscaras, radiografías procesadas  | MinIO (S3-compatible)  |
| Modelo IA            | Clasificación de radiografías (3 clases)             | PyTorch, ResNet50      |
| API REST             | Exposición de servicios: predicción, datos, stats    | FastAPI                |
| Dashboard            | Visualización de KPIs, diagnósticos, alertas, modelo | Streamlit              |
| Automatización       | Watchdog, alertas clínicas, informes periódicos      | APScheduler, Watchdog  |
| Infraestructura      | Containerización y orquestación                      | Docker, Docker Compose |
| Monitorización       | Logging centralizado, validación de calidad de datos | Python logging, PostgreSQL |

### 5. Flujo principal de datos

1. **Ingesta**: los datos entran por dos vías — carga manual vía API (`POST /predict`) o detección automática de nuevos ficheros en MinIO (watchdog).
2. **Almacenamiento**: los datos estructurados (pacientes, diagnósticos) se persisten en PostgreSQL; las imágenes y máscaras en MinIO.
3. **Procesamiento**: PySpark ejecuta la limpieza y transformación de datos tabulares. El módulo de preprocesamiento aplica resize (224×224), segmentación con máscara pulmonar y normalización a las imágenes.
4. **Inferencia**: el modelo CNN clasifica la radiografía y devuelve clase + probabilidades.
5. **Persistencia de resultados**: la predicción se guarda en PostgreSQL vinculada al paciente.
6. **Automatización**: si la predicción es COVID-19 con probabilidad > 0.85, se genera una alerta. Un scheduler genera informes periódicos.
7. **Servicio**: la API expone endpoints para consulta de datos y predicción; el dashboard visualiza métricas, diagnósticos y alertas en tiempo real.

### 6. Datasets

- **Radiografías**: COVID-19 Radiography Database (Kaggle, Tawsifur Rahman). 21.165 imágenes PNG (1024×1024) en 4 clases originales, reducidas a 3 (fusionando Lung Opacity + Viral Pneumonia → Neumonía). Incluye máscaras de segmentación pulmonar.
- **Pacientes**: dataset sintético generado con Faker (~5.000 registros) con campos: `patient_id`, `name`, `age`, `sex`, `admission_date`, `department`, `diagnosis`, `xray_filename`, `risk_score`, `status`, `attending_physician`.

### 7. Criterios de aceptación globales

- [ ] El sistema completo se levanta con `docker-compose up` en un solo comando.
- [ ] El pipeline procesa datos de ambas fuentes (CSV + imágenes) sin intervención manual.
- [ ] El modelo clasifica radiografías en tres clases con métricas documentadas.
- [ ] La API responde en < 500ms para predicciones individuales.
- [ ] El dashboard muestra datos actualizados tras cada ciclo de procesamiento.
- [ ] Las alertas se generan automáticamente ante predicciones COVID-19 de alta confianza.
- [ ] Los logs centralizados registran cada etapa del pipeline con timestamps.
- [ ] La validación de calidad de datos detecta y reporta registros duplicados, nulos o corruptos.

### 8. Restricciones técnicas

- Toda la infraestructura debe correr en contenedores Docker sin dependencias locales.
- Las imágenes médicas no se almacenan en PostgreSQL; se usa MinIO como object storage.
- El modelo debe poder ejecutar inferencia en CPU (no se asume disponibilidad de GPU en despliegue).
- Los datos de pacientes son sintéticos para evitar problemas de privacidad (GDPR/LOPDGDD).

### 9. Restricciones de negocio

- El sistema es un soporte a la decisión médica, nunca un sustituto del diagnóstico profesional.
- Los falsos negativos en COVID-19 tienen mayor coste clínico que los falsos positivos (priorizar recall en la clase COVID).
- Toda predicción debe almacenarse con trazabilidad completa (timestamp, versión del modelo, probabilidades por clase).
