# SDD - Modulo de Prediccion Clinica

## 1. Descripcion funcional

Modulo de Inteligencia Artificial para prediccion de riesgos clinicos basado en sintomas y datos demograficos de pacientes. Opera de forma **independiente** del sistema de clasificacion de radiografias (COVID/Normal/Neumonia).

## 2. Problema que resuelve

El sistema de radiografias solo clasifica imagenes medicas. Este modulo permite:

- **Clasificacion de pacientes** por nivel de riesgo
- **Prediccion de enfermedades** basada en sintomas
- **Segmentacion de perfiles clinicos** (bajo/medio/alto riesgo)

## 3. Alcance tecnologico

### Modelo de IA (Clasificacion de sintomas)

- [x] Clasificacion de pacientes
- [x] Prediccion de enfermedades
- [x] Segmentacion de perfiles clinicos

### Procesamiento de datos / Big Data

- [x] Volumen medio (dataset sintético)
- [x] Datos estructurados (CSV, PostgreSQL)
- [x] Pipeline de datos (ingesta, entrenamiento)

### Automatizacion de procesos

- [x] Generacion automatica de informes
- [x] Prediccion automatica de riesgos

## 4. Inputs

### Features de entrada

| Feature | Tipo | Rango | Fuente |
|---------|------|-------|--------|
| age | Entero | 18-85 | Formulario |
| sex | String | M/F | Formulario |
| height_cm | Float | 150-210 | Formulario |
| weight_kg | Float | 45-150 | Formulario |
| has_cough | Booleano | 0/1 | Formulario |
| has_chest_pain | Booleano | 0/1 | Formulario |
| has_fatigue | Booleano | 0/1 | Formulario |
| has_fever | Booleano | 0/1 | Formulario |
| has_dizziness | Booleano | 0/1 | Formulario |
| has_breathing_difficulty | Booleano | 0/1 | Formulario |
| has_headache | Booleano | 0/1 | Formulario |
| has_nausea | Booleano | 0/1 | Formulario |
| has_loss_of_appetite | Booleano | 0/1 | Formulario |
| has_night_sweats | Booleano | 0/1 | Formulario |

## 5. Outputs

### Enfermedades predichas

| Enfermedad | Tipo | Descripcion |
|-------------|------|-------------|
| diabetes_risk | Booleano | Riesgo de diabetes |
| hypertension_risk | Booleano | Riesgo de hipertension |
| heart_disease_risk | Booleano | Riesgo de enfermedad cardiaca |

### Metadatos

| Campo | Tipo | Descripcion |
|-------|------|-------------|
| probabilities | Dict | Probabilidades por enfermedad |
| confidence | Float | Confianza maxima |
| risk_level | String | Nivel de riesgo: bajo/moderado/alto |
| inference_time_ms | Float | Tiempo de inference |

## 6. Arquitectura del modelo

### Eleccion: MLP (Multi-Layer Perceptron)

| Aspecto | Valor |
|--------|-------|
| Tipo | Red neuronal densa |
| Capas ocultas | Dense(64, ReLU) -> Dense(32, ReLU) |
| Regularizacion | L2 + Dropout(0.3) + BatchNorm |
| Output | Sigmoid (multi-label) |
| Optimizer | Adam (lr=1e-4) |
| Loss | Binary Crossentropy |

### Justificacion

- Datos estructurados (no imagenes)
- Features tabulares
- Clasificacion multi-label
- Comparable con el modelo de radiografias (misma arquitectura base)

## 7. Dataset

### Generacion sintetica

- **Numero de pacientes**: 1000 (configurable)
- **Features**: 15 (5 demograficos + 10 sintomas)
- **Labels**: 3 enfermedades binarias

### Correlaciones medicas

| Factor | Diabetes | Hipertension | Cardiaca |
|---------|----------|--------------|----------|
| Edad > 40 | +0.003/yr | +0.005/yr | +0.004/yr |
| IMC > 25 | +0.008/yr | +0.01/yr | - |
| Obesidad (IMC > 30) | +0.015 | +0.02 | +0.01 |
| Tos | +0.05 | - | - |
| Dolor de pecho | - | - | +0.08 |
| Fatiga | +0.05 | +0.03 | +0.03 |
| Mareos | - | +0.04 | - |

## 8. Integracion

### Con el sistema principal

| Aspecto | Integracion |
|--------|------------|
| Puerto API | 8001 (separado de 8000) |
| BD | PostgreSQL separado (clinical_db) |
| Dashboard | Vista separada en Streamlit |
| Docker | Servicio opcional |

### Endpoints

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/clinical/predict` | POST | Predice riesgos |
| `/clinical/info` | GET | Info del modelo |
| `/clinical/health` | GET | Health check |

## 9. Criterios de aceptacion

- [ ] El modelo predice 3 enfermedades simultaneamente
- [ ] Devuelve probabilidades por enfermedad
- [ ] Clasifica nivel de riesgo (bajo/moderado/alto)
- [ ] La API responde en < 500ms
- [ ] Tests unitarios pasan
- [ ] Dataset generable sinteticamente
- [ ] Modelo entrenable con 1000 samples

## 10. Limitaciones

1. **Datos sinteticos**: Las correlaciones son simplificadas
2. **Sin validacion clinica**: No es un diagnostico medico real
3. **Threshold fijo**: 0.5 para todas las enfermedades
4. **Sin consideracion de historial**: Solo datos puntuales

## 11. Mejoras futuras

- [ ] Validacion con datos clinicos reales
- [ ] Ajuste de thresholds por enfermedad
- [ ] Integracion con historial medico
- [ ] Modelo de evolucion temporal
- [ ] Explicabilidad (SHAP values)