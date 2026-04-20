# SDD – Modelo de Inteligencia Artificial

### 1. Descripción funcional

Módulo de Deep Learning para clasificación automática de radiografías de tórax en tres categorías clínicas: **Sana**, **Neumonía** y **COVID-19**. Utiliza transfer learning sobre ResNet50 pre-entrenada en ImageNet, con fine-tuning de las capas superiores. El modelo recibe imágenes preprocesadas (224×224, segmentadas con máscara pulmonar) y devuelve la clase predicha junto con las probabilidades por categoría.

### 2. Dataset

**Fuente**: COVID-19 Radiography Database (Kaggle, Tawsifur Rahman et al.)

| Clase original    | Imágenes | Mapeo a 3 clases | Imágenes finales |
|-------------------|----------|-------------------|------------------|
| Normal            | 10.192   | Sana              | 10.192           |
| Lung Opacity      | 6.012    | Neumonía          | 7.357            |
| Viral Pneumonia   | 1.345    | Neumonía          |                  |
| COVID-19          | 3.616    | COVID-19          | 3.616            |

**Justificación de la fusión Lung Opacity + Viral Pneumonia → Neumonía**: La opacidad pulmonar es un hallazgo radiológico que se manifiesta en diversas neumopatías, incluyendo la neumonía viral. Ambas categorías representan patología pulmonar no-COVID y clínicamente se abordan de forma similar a nivel de triaje inicial. La fusión produce tres clases coherentes con el reto planteado por el enunciado.

### 3. Inputs

| Input                      | Formato              | Origen                     |
|----------------------------|----------------------|----------------------------|
| Radiografía preprocesada   | Tensor 224×224×3     | MinIO (bucket `processed-xrays`) |
| Máscara pulmonar           | PNG 1024×1024        | MinIO (bucket `lung-masks`) — aplicada en preprocesamiento |

### 4. Outputs

| Output            | Tipo              | Ejemplo                                              |
|-------------------|-------------------|-------------------------------------------------------|
| Clase predicha    | String            | `"COVID-19"`                                          |
| Probabilidades    | Dict[str, float]  | `{"Sana": 0.05, "Neumonía": 0.10, "COVID-19": 0.85}` |
| Confianza         | Float             | `0.85` (max de probabilidades)                        |

### 5. Arquitectura del modelo

#### 5.1 Elección: ResNet50 con transfer learning

**Alternativas consideradas**:

| Arquitectura    | Parámetros | Pros                                    | Contras                                | Decisión    |
|-----------------|------------|-----------------------------------------|----------------------------------------|-------------|
| VGG16/19        | ~138M      | Simple, bien documentada                | Muy pesada, lenta en inferencia        | Descartada  |
| ResNet50        | ~25.6M     | Skip connections, buen balance peso/rendimiento, ampliamente validada en imagen médica | Más pesada que EfficientNet | **Elegida** |
| EfficientNet-B0 | ~5.3M      | Muy ligera, eficiente                   | Menos validada en radiografías         | Alternativa |
| DenseNet121     | ~8M        | Feature reuse, buena en datasets pequeños | Uso alto de memoria por concatenaciones | Alternativa |

**Justificación**: ResNet50 ofrece el mejor equilibrio entre rendimiento demostrado en clasificación de radiografías (ampliamente citada en literatura), complejidad manejable y compatibilidad con inferencia en CPU. Las skip connections mitigan el vanishing gradient y permiten fine-tuning estable de capas profundas.

#### 5.2 Estrategia de transfer learning

- **Base**: ResNet50 pre-entrenada en ImageNet (pesos congelados inicialmente).
- **Fine-tuning progresivo**:
  - Fase 1: Entrenar solo la cabeza de clasificación (capas fully connected nuevas) durante 5-10 epochs.
  - Fase 2: Descongelar las últimas 15-20 capas de ResNet50 y entrenar end-to-end con learning rate reducido (1e-5 a 1e-4).
- **Cabeza de clasificación**: Global Average Pooling → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.3) → Dense(3, Softmax).

### 6. Preprocesamiento de imágenes

1. **Segmentación**: aplicación de máscara pulmonar (imagen × máscara) para eliminar fondo.
2. **Resize**: 1024×1024 → 224×224 píxeles (bilinear interpolation).
3. **Normalización**: píxeles al rango [0, 1], luego estandarización con media y desviación estándar de ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### 7. Estrategia para desbalanceo de clases

El dataset presenta desbalanceo significativo tras la fusión: Sana (10.192) vs COVID-19 (3.616).

| Técnica                | Aplicación                                                      |
|------------------------|-----------------------------------------------------------------|
| Undersampling          | Reducir Normal a ~5.500 muestras (random sampling estratificado) |
| Data augmentation      | Rotaciones ±15°, flip horizontal, ajustes de brillo/contraste ±10%. Sin transformaciones agresivas que distorsionen anatomía. |
| Class weights          | Pesos inversamente proporcionales a frecuencia en la loss function. Mayor peso a COVID-19 para priorizar recall en la clase más crítica clínicamente. |

**Justificación clínica del weighting**: En un entorno hospitalario, un falso negativo de COVID-19 (paciente infectado clasificado como sano) tiene consecuencias graves — riesgo de contagio a otros pacientes y personal, retraso en tratamiento. Un falso positivo (paciente sano clasificado como COVID) genera coste adicional (aislamiento, pruebas confirmatorias) pero no pone en riesgo vidas directamente. Por tanto, se prioriza recall sobre precision en la clase COVID-19.

### 8. Entrenamiento

| Hiperparámetro        | Valor                                |
|------------------------|--------------------------------------|
| Optimizer              | Adam                                 |
| Learning rate (fase 1) | 1e-3                                 |
| Learning rate (fase 2) | 1e-5                                 |
| Batch size             | 32                                   |
| Epochs (fase 1)        | 10                                   |
| Epochs (fase 2)        | 20                                   |
| Loss function          | CrossEntropyLoss con class weights   |
| Early stopping         | Patience 5 epochs sobre val_loss     |
| Split                  | 70% train / 15% val / 15% test       |

### 9. Métricas de evaluación

| Métrica                    | Propósito                                                |
|----------------------------|----------------------------------------------------------|
| Accuracy global            | Rendimiento general del modelo.                          |
| Precision por clase        | ¿Cuántos de los predichos como X son realmente X?        |
| Recall por clase           | ¿Cuántos de los reales X fueron detectados?              |
| F1-score por clase         | Balance precision-recall por categoría.                  |
| Matriz de confusión        | Identificar errores específicos entre clases.            |
| AUC-ROC (one-vs-rest)     | Capacidad discriminativa del modelo por clase.           |

**Análisis obligatorio de la matriz de confusión**: se documentará específicamente qué tipos de error comete el modelo (COVID↔Neumonía, COVID↔Sana, Neumonía↔Sana) y se evaluará el impacto clínico de cada tipo de error.

### 10. Integración con el sistema

- El modelo entrenado se serializa como fichero `.pth` (PyTorch) y se monta como volumen en el contenedor de la API.
- La API carga el modelo al iniciar y mantiene una instancia en memoria para inferencia.
- Cada predicción se registra en la tabla `diagnoses` de PostgreSQL con trazabilidad completa: `patient_id`, `xray_filename`, `prediction`, `probabilidades`, `model_version`, `timestamp`.

### 11. Experimento comparativo (bonus)

Para la memoria técnica se documentará una comparación:
- Rendimiento **con** vs **sin** segmentación por máscara pulmonar.
- Impacto del data augmentation vs sin augmentation.
- Esto justifica técnicamente las decisiones de preprocesamiento.

### 12. Criterios de aceptación

- [ ] El modelo clasifica radiografías en 3 clases y devuelve probabilidades.
- [ ] Se documenta la matriz de confusión con análisis de errores clínicos.
- [ ] Se reportan precision, recall y F1-score por clase.
- [ ] El recall de COVID-19 es superior al de las otras clases (priorizado por class weights).
- [ ] La inferencia de una imagen individual ejecuta en < 2 segundos en CPU.
- [ ] El modelo está versionado y la versión se registra con cada predicción.
- [ ] Se documenta el experimento comparativo con/sin máscara pulmonar.
