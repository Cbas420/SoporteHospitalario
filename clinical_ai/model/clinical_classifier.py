"""
Arquitectura del modelo de clasificacion de riesgo clinico.
"""

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2


def build_clinical_model(
    input_dim: int,
    num_diseases: int = 3,
    hidden_units: int = 64,
    dropout_rate: float = 0.3,
) -> Model:
    """
    Construye el modelo de clasificacion multi-label para riesgos clinicos.

    Args:
        input_dim: Dimension del vector de features de entrada
        num_diseases: Numero de enfermedades a predecir (default 3)
        hidden_units: Unidades en capas ocultas (default 64)
        dropout_rate: Tasa de dropout para regularizacion (default 0.3)

    Returns:
        Modelo Keras compilado
    """
    inputs = layers.Input(shape=(input_dim,), name="clinical_input")

    x = layers.Dense(hidden_units, activation="relu", kernel_regularizer=l2(0.001))(
        inputs
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(hidden_units // 2, activation="relu", kernel_regularizer=l2(0.001))(
        x
    )
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_diseases, activation="sigmoid", name="risk_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="clinical_risk_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def preprocess_clinical_input(
    age: int,
    sex: str,
    height_cm: float,
    weight_kg: float,
    symptoms: dict[str, bool],
) -> np.ndarray:
    """
    Preprocesa los datos clinicos de entrada para el modelo.

    Args:
        age: Edad del paciente
        sex: Sexo ('M' o 'F')
        height_cm: Altura en cm
        weight_kg: Peso en kg
        symptoms: Diccionario de sintomas {nombre: valor_boolean}

    Returns:
        Vector numpy listo para inference
    """
    bmi = weight_kg / ((height_cm / 100) ** 2)

    sex_encoded = 1.0 if sex.upper() == "M" else 0.0

    features = [
        float(age) / 100.0,
        sex_encoded,
        height_cm / 250.0,
        weight_kg / 200.0,
        bmi / 50.0,
    ]

    symptom_map = {
        "has_cough": 0,
        "has_chest_pain": 1,
        "has_fatigue": 2,
        "has_fever": 3,
        "has_dizziness": 4,
        "has_breathing_difficulty": 5,
        "has_headache": 6,
        "has_nausea": 7,
        "has_loss_of_appetite": 8,
        "has_night_sweats": 9,
    }

    for i in range(10):
        features.append(0.0)

    for symptom, value in symptoms.items():
        if symptom in symptom_map:
            features[5 + symptom_map[symptom]] = 1.0 if value else 0.0

    return np.array([features], dtype=np.float32)


def get_feature_names() -> list[str]:
    """Devuelve los nombres de features en orden."""
    base = [
        "age",
        "sex",
        "height_cm",
        "weight_kg",
        "bmi",
    ]
    return base + [
        "has_cough",
        "has_chest_pain",
        "has_fatigue",
        "has_fever",
        "has_dizziness",
        "has_breathing_difficulty",
        "has_headache",
        "has_nausea",
        "has_loss_of_appetite",
        "has_night_sweats",
    ]