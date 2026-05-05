# SDD – Data Pipeline

### 1. Descripción funcional

Pipeline de datos completo que cubre las fases de ingesta, almacenamiento, limpieza, transformación y servicio. Gestiona dos flujos paralelos: datos estructurados de pacientes (CSV → PostgreSQL) y datos no estructurados de imágenes médicas (radiografías → MinIO). Diseñado para ser escalable, automatizado y con validación de calidad integrada.

### 2. Inputs

| Input                    | Formato       | Origen                          | Volumen estimado        |
|--------------------------|---------------|---------------------------------|-------------------------|
| Registros de pacientes   | CSV           | Script generador (Faker)        | ~5.000 registros        |
| Radiografías de tórax    | PNG 1024×1024 | COVID-19 Radiography Database   | ~21.165 imágenes        |
| Máscaras pulmonares      | PNG 1024×1024 | COVID-19 Radiography Database   | ~21.165 máscaras        |
| Nuevos ficheros (runtime)| CSV / PNG     | Directorio vigilado (watchdog)  | Variable                |

### 3. Outputs

| Output                         | Destino       | Formato                          |
|--------------------------------|---------------|-----------------------------------|
| Tabla `patients`               | PostgreSQL    | Registros limpios y validados     |
| Tabla `diagnoses`              | PostgreSQL    | Resultados de predicción IA       |
| Tabla `data_quality_log`       | PostgreSQL    | Incidencias detectadas            |
| Imágenes preprocesadas         | MinIO         | PNG 224×224, segmentadas          |
| Imágenes originales            | MinIO         | PNG 1024×1024 (backup)            |

### 4. Fases del pipeline

#### 4.1 Ingesta
- **Datos tabulares**: lectura de CSVs con **Dask** (`dd.read_csv`), particionando el fichero para procesamiento paralelo. Detección automática de esquema con validación posterior por partición.
- **Imágenes**: carga batch al bucket `raw-xrays` de MinIO. Las máscaras van al bucket `lung-masks`.
- **Ingesta continua**: proceso watchdog monitoriza el directorio `/data/incoming/` y dispara el pipeline para nuevos ficheros.

#### 4.2 Validación de calidad de datos
Antes de cualquier transformación, se ejecutan comprobaciones:
- **Duplicados**: detección por `record_uid` (`patient_id:image_name`) con Dask.
- **Campos nulos**: verificación de campos obligatorios (`patient_id`, `age`, `sex`, `admission_date`).
- **Integridad referencial**: todo `xray_filename` debe existir en MinIO.
- **Imágenes corruptas**: intento de apertura con PIL; las que fallan se mueven a `/data/quarantine/` y se registra la incidencia.
- **Rangos válidos**: `age` entre 0-120, `admission_date` no futura, `department` dentro de valores permitidos.
- **Validación paralela de imágenes**: usando `dask.delayed`, se comprueban en paralelo la existencia de todas las imágenes referenciadas en el CSV.

Todas las incidencias se registran en la tabla `data_quality_log` con: `timestamp`, `source_file`, `issue_type`, `severity`, `description`, `resolved`.

#### 4.3 Limpieza y transformación (datos tabulares)
- Normalización de columnas por partición (`map_partitions`): nombres de columnas, capitalización consistente.
- Limpieza de diagnósticos por partición: mapeo a clases canónicas (`COVID19`, `Normal`, `Pneumonia`).
- Relleno de valores nulos por partición con defaults seguros.
- Eliminación de registros duplicados (keep first) tras materialización.
- Motor: **Dask** para procesamiento distribuido/paralelo.

#### 4.4 Preprocesamiento de imágenes
- Resize de 1024×1024 a 224×224 píxeles (dimensión de entrada de ResNet50).
- Aplicación de máscara pulmonar: multiplicación element-wise `imagen × máscara` para eliminar fondo (bordes torácicos, artefactos, texto superpuesto).
- Normalización de píxeles al rango [0, 1] y estandarización con media/std de ImageNet (para transfer learning).
- Almacenamiento de imágenes preprocesadas en bucket `processed-xrays` de MinIO.

#### 4.5 Servicio
- Los datos limpios quedan disponibles en PostgreSQL para consumo vía API REST y dashboard.
- Las imágenes procesadas en MinIO están accesibles para el módulo de inferencia IA.

### 5. Tecnologías

| Componente         | Tecnología                | Justificación                                                |
|--------------------|---------------------------|--------------------------------------------------------------|
| Procesamiento      | **Dask**                  | Framework distribuido/escalable. No requiere JVM (vs PySpark, −400MB imagen Docker). API compatible con pandas. Escalable añadiendo workers. Adecuado para el volumen hospitalario (~5k pacientes, ~21k imágenes). |
| Almacenamiento SQL | PostgreSQL                | Base de datos relacional robusta para datos estructurados.   |
| Object Storage     | MinIO                     | Compatible con API S3, ideal para imágenes. Ligero y dockerizable. |
| Preprocesamiento   | Pillow, NumPy             | Estándar para manipulación de imágenes en Python.            |
| Watchdog           | watchdog (Python library) | Monitorización de filesystem en tiempo real, ligero.         |
| Generación datos   | Faker                     | Generación de datos sintéticos realistas y reproducibles.    |

**Justificación de Dask sobre PySpark**: El enunciado acepta Spark/PySpark, Dask o Apache Beam. Se eligió Dask porque: (1) no requiere instalación de JVM ni imagen Docker dedicada de Spark, reduciendo la complejidad operativa; (2) su API es un superset de pandas, minimizando el código y facilitando el mantenimiento; (3) permite escalar horizontalmente añadiendo workers Dask si el volumen del hospital crece, sin cambiar el código; (4) PySpark estaría justificado para clústeres multi-nodo con >1TB de datos, escenario que excede el contexto del proyecto actual.

### 6. Esquema de base de datos (PostgreSQL)

```sql
-- Tabla principal de pacientes
CREATE TABLE patients (
    patient_id    UUID PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    age           INTEGER CHECK (age BETWEEN 0 AND 120),
    sex           VARCHAR(1) CHECK (sex IN ('M', 'F')),
    admission_date DATE NOT NULL,
    department    VARCHAR(50) NOT NULL,
    status        VARCHAR(20) DEFAULT 'Ingresado',
    attending_physician VARCHAR(100),
    risk_score    FLOAT,
    created_at    TIMESTAMP DEFAULT NOW()
);

-- Tabla de diagnósticos (vinculada a predicciones IA)
CREATE TABLE diagnoses (
    diagnosis_id  SERIAL PRIMARY KEY,
    patient_id    UUID REFERENCES patients(patient_id),
    xray_filename VARCHAR(255) NOT NULL,
    prediction    VARCHAR(20) NOT NULL,
    prob_normal   FLOAT NOT NULL,
    prob_pneumonia FLOAT NOT NULL,
    prob_covid    FLOAT NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    predicted_at  TIMESTAMP DEFAULT NOW()
);

-- Tabla de alertas clínicas
CREATE TABLE alerts (
    alert_id      SERIAL PRIMARY KEY,
    diagnosis_id  INTEGER REFERENCES diagnoses(diagnosis_id),
    patient_id    UUID REFERENCES patients(patient_id),
    alert_type    VARCHAR(50) NOT NULL,
    severity      VARCHAR(20) NOT NULL,
    message       TEXT,
    status        VARCHAR(20) DEFAULT 'pending',
    created_at    TIMESTAMP DEFAULT NOW(),
    resolved_at   TIMESTAMP
);

-- Tabla de calidad de datos
CREATE TABLE data_quality_log (
    log_id        SERIAL PRIMARY KEY,
    timestamp     TIMESTAMP DEFAULT NOW(),
    source_file   VARCHAR(255),
    issue_type    VARCHAR(50),
    severity      VARCHAR(20),
    description   TEXT,
    resolved      BOOLEAN DEFAULT FALSE
);
```

### 7. Estructura de MinIO (buckets)

```
raw-xrays/           → Imágenes originales (1024×1024)
lung-masks/          → Máscaras de segmentación pulmonar
processed-xrays/     → Imágenes preprocesadas (224×224, segmentadas)
quarantine/          → Imágenes que no pasaron validación
reports/             → Informes automáticos generados
```

### 8. Criterios de aceptación

- [x] **Dask** procesa el CSV completo de pacientes en particiones paralelas sin errores y los datos se cargan en PostgreSQL.
- [x] La validación paralela de imágenes usa `dask.delayed` para comprobar existencia en paralelo.
- [ ] Las imágenes se almacenan en MinIO organizadas por bucket según su estado.
- [x] La validación de calidad detecta al menos: duplicados, campos nulos e imágenes corruptas.
- [x] Toda incidencia de calidad queda registrada en `data_quality_log`.
- [ ] El preprocesamiento genera imágenes 224×224 segmentadas y normalizadas.
- [x] El watchdog detecta nuevos ficheros en < 10 segundos y dispara el pipeline.
- [ ] El pipeline completo (ingesta → limpieza → transformación) ejecuta en < 5 minutos para el dataset completo.
