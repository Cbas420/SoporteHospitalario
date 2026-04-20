# SDD – Dashboard

### 1. Descripción funcional

Dashboard interactivo para la visualización de datos clínicos, resultados del modelo de IA, alertas y métricas operativas del hospital. Dirigido a dos perfiles de usuario: **personal médico** (diagnósticos, alertas, pacientes) y **administración** (KPIs operativos, calidad de datos, estado del sistema). Construido con Streamlit por su rapidez de desarrollo y capacidad de generar interfaces reactivas sin frontend dedicado.

### 2. Inputs

| Input                       | Formato       | Origen          |
|-----------------------------|---------------|-----------------|
| Datos de pacientes          | SQL query     | PostgreSQL      |
| Diagnósticos y predicciones | SQL query     | PostgreSQL      |
| Alertas clínicas            | SQL query     | PostgreSQL      |
| Logs de calidad de datos    | SQL query     | PostgreSQL      |
| Imágenes de radiografías    | PNG           | MinIO           |
| Estadísticas agregadas      | JSON          | API `/stats`    |

### 3. Outputs

| Output                  | Formato     | Destino           |
|-------------------------|-------------|-------------------|
| Visualizaciones         | Gráficos    | Navegador (UI)    |
| Acciones sobre alertas  | PATCH request| API `/alerts`    |

### 4. Vistas del dashboard

#### 4.1 Vista: Resumen operativo (Home)

**Público**: administración y dirección médica.

**Componentes**:
- **KPI cards** (fila superior): pacientes totales, radiografías procesadas hoy, alertas activas, tasa de ocupación por departamento.
- **Gráfico de barras**: distribución de pacientes por departamento (Urgencias, UCI, Planta, Consultas Externas).
- **Línea temporal**: evolución de ingresos en los últimos 30 días.
- **Filtros**: rango de fechas, departamento.

#### 4.2 Vista: Diagnósticos

**Público**: personal médico.

**Componentes**:
- **Gráfico de tarta/donut**: distribución Sana / Neumonía / COVID-19 sobre el total de radiografías clasificadas.
- **Gráfico de línea**: evolución temporal de diagnósticos por categoría (tendencias de COVID, neumonía).
- **Tabla detallada**: últimos diagnósticos con columnas: paciente, fecha, predicción, confianza, enlace a radiografía.
- **Filtros**: rango de fechas, departamento, tipo de diagnóstico, umbral de confianza.

#### 4.3 Vista: Modelo IA

**Público**: equipo técnico y dirección médica.

**Componentes**:
- **Matriz de confusión interactiva**: heatmap con valores absolutos y porcentajes. Click en celda muestra ejemplos de ese tipo de error.
- **Tabla de métricas por clase**: precision, recall, F1-score, support para cada categoría.
- **Histograma de probabilidades**: distribución de la confianza del modelo para cada clase predicha (permite ver si el modelo es "decidido" o "dudoso").
- **AUC-ROC curves**: una curva por clase (one-vs-rest).
- **Comparativa con/sin máscara**: tabla o gráfico mostrando el impacto de la segmentación pulmonar en las métricas.

#### 4.4 Vista: Alertas

**Público**: personal médico y supervisores.

**Componentes**:
- **Contador por severidad**: badges con el número de alertas `critical`, `warning`, `info` pendientes.
- **Tabla de alertas**: columnas: fecha, paciente, tipo, severidad, mensaje, estado. Ordenada por severidad descendente.
- **Acción**: botón para marcar alerta como "revisada" (envía `PATCH /alerts/{id}`).
- **Filtros**: estado (pendiente/resuelta), severidad, rango de fechas.

#### 4.5 Vista: Calidad de datos

**Público**: equipo técnico.

**Componentes**:
- **KPI cards**: registros duplicados detectados, campos nulos encontrados, imágenes en cuarentena.
- **Tabla de incidencias**: datos de `data_quality_log` con columnas: timestamp, fichero, tipo de incidencia, severidad, descripción, estado.
- **Gráfico de barras**: incidencias por tipo en el último mes.

### 5. Tecnología

| Componente         | Tecnología       | Justificación                                        |
|--------------------|------------------|------------------------------------------------------|
| Framework UI       | Streamlit        | Desarrollo rápido, reactivo, sin necesidad de JS/React separado. Ideal para prototipos de dashboards de datos. |
| Gráficos           | Plotly           | Gráficos interactivos (hover, zoom, click) integrados nativamente en Streamlit. |
| Conexión DB        | psycopg2 / SQLAlchemy | Queries directas a PostgreSQL.                  |
| Conexión MinIO     | boto3 / minio-py | Recuperación de imágenes para visualización.         |
| Conexión API       | requests / httpx | Llamadas a endpoints de la API FastAPI.              |

### 6. Restricciones técnicas

- El dashboard consulta datos directamente de PostgreSQL para vistas tabulares y de la API para estadísticas agregadas.
- Las imágenes de radiografías se cargan desde MinIO bajo demanda (no se precargan todas).
- Refresco automático configurable (por defecto cada 60 segundos vía `st.auto_refresh` o botón manual).
- El dashboard corre en su propio contenedor Docker, expuesto en el puerto `8501`.

### 7. Criterios de aceptación

- [ ] La vista Home muestra KPIs actualizados con datos reales de PostgreSQL.
- [ ] La vista Diagnósticos muestra la distribución de clases con filtros funcionales.
- [ ] La vista Modelo IA muestra la matriz de confusión y métricas por clase.
- [ ] La vista Alertas lista las alertas pendientes y permite marcarlas como revisadas.
- [ ] La vista Calidad de datos muestra las incidencias registradas.
- [ ] Todos los gráficos son interactivos (hover con detalle, zoom).
- [ ] El dashboard es accesible desde el navegador en `http://localhost:8501`.
- [ ] Los filtros de fecha y departamento funcionan en todas las vistas relevantes.
