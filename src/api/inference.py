"""
Servicio de inferencia para predicciones individuales.

Este módulo proporciona la interfaz de predicción que será consumida
por la API REST (FastAPI) del sistema.

Responsabilidades:
- Cargar y cachear el modelo en memoria.
- Preprocesar una imagen individual.
- Ejecutar inferencia y devolver resultado estructurado.
- Validar tiempo de inferencia (< 1s según SDD).

Decisiones técnicas:

1. Singleton del modelo:
   - Se carga una sola vez al iniciar el servicio.
   - Se mantiene en memoria para evitar overhead de carga en cada request.

2. Resultado estructurado:
   - Incluye clase predicha, confianza, todas las probabilidades.
   - Flag de "baja confianza" para derivar a revisión manual.
   - Compatible con el formato esperado por el dashboard.

Interfaz de integración:
- La API REST invoca predict_single() para cada request a /predict.
- El pipeline de automatización puede usarlo para procesamiento batch.
- Los resultados se estructuran para registro en base de datos (auditoría).
"""

import io
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from config.settings import (
    CLASS_NAMES, IMG_SIZE, MODEL_PATH,
    MAX_INFERENCE_TIME_MS, CONFIDENCE_THRESHOLD_LOW, MODEL_WEIGHTS_PATH,
)
from src.model.classifier import build_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Cache global del modelo (singleton)
_model_cache = None


def load_model(
    model_path: Path = MODEL_PATH,
    weights_path: Path = MODEL_WEIGHTS_PATH,
):
    """
    Carga el modelo en el cache global (singleton).
    
    Se invoca una vez al iniciar la API. Las llamadas
    posteriores devuelven el modelo cacheado.
    
    Args:
        model_path: Ruta al modelo .keras.
    
    Returns:
        Modelo Keras cargado.
    """
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    import tensorflow as tf
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado: {model_path}. "
            "Ejecuta primero el entrenamiento con train_pipeline.py"
        )
    
    try:
        _model_cache = tf.keras.models.load_model(str(model_path), compile=False)
        logger.info(f"Modelo cargado en cache desde {model_path}")
    except Exception as exc:
        if not weights_path.exists():
            raise

        logger.warning(
            "Fallo al cargar el modelo serializado (%s). "
            "Se reconstruira la arquitectura y se cargaran pesos desde %s",
            exc,
            weights_path,
        )
        _model_cache = build_model(freeze_base=False, weights=None)
        _model_cache.load_weights(str(weights_path))
        logger.info("Modelo reconstruido desde arquitectura + pesos")
    
    # Warm-up: primera inferencia es más lenta por compilación de grafos
    dummy_input = np.zeros((1, *IMG_SIZE, 3))
    _model_cache.predict(dummy_input, verbose=0)
    logger.info("Warm-up de inferencia completado")
    
    return _model_cache


def preprocess_image(
    image_data: Union[bytes, str, Path, np.ndarray],
) -> np.ndarray:
    """
    Preprocesa una imagen individual para inferencia.
    
    Acepta múltiples formatos de entrada para flexibilidad
    en la integración (upload HTTP, ruta de fichero, array numpy).
    
    Args:
        image_data: Imagen como bytes, ruta, o numpy array.
    
    Returns:
        Numpy array preprocesado (1, 224, 224, 3).
    """
    import tensorflow as tf
    
    if isinstance(image_data, (str, Path)):
        # Desde archivo en disco
        img = tf.keras.utils.load_img(
            str(image_data),
            target_size=IMG_SIZE,
        )
        img_array = tf.keras.utils.img_to_array(img)
    
    elif isinstance(image_data, bytes):
        # Desde bytes (upload HTTP)
        img = tf.keras.utils.load_img(
            io.BytesIO(image_data),
            target_size=IMG_SIZE,
        )
        img_array = tf.keras.utils.img_to_array(img)
    
    elif isinstance(image_data, np.ndarray):
        # Ya es un array numpy
        if image_data.shape[:2] != IMG_SIZE:
            img = tf.image.resize(image_data, IMG_SIZE)
            img_array = img.numpy()
        else:
            img_array = image_data.copy()
    
    else:
        raise ValueError(
            f"Formato de imagen no soportado: {type(image_data)}. "
            "Usar bytes, str, Path, o numpy array."
        )
    
    # Expandir dimensión de batch y aplicar normalización ResNet50
    img_batch = np.expand_dims(img_array.astype(np.float32), axis=0)
    img_batch = img_batch / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    img_batch = (img_batch - mean) / std

    return img_batch


def predict_single(
    image_data: Union[bytes, str, Path, np.ndarray],
    model=None,
) -> Dict:
    """
    Ejecuta predicción sobre una imagen individual.
    
    Esta es la función principal que la API REST invocará.
    
    Args:
        image_data: Imagen en cualquier formato soportado.
        model: Modelo Keras (usa cache si no se proporciona).
    
    Returns:
        Dict con resultado estructurado:
        {
            "prediction": "COVID19",
            "confidence": 0.92,
            "probabilities": {
                "COVID19": 0.92,
                "Normal": 0.05,
                "Pneumonia": 0.03
            },
            "requires_review": False,
            "inference_time_ms": 45.2
        }
    """
    if model is None:
        model = load_model()
    
    # Preprocesar
    img_batch = preprocess_image(image_data)
    
    # Inferencia con medición de tiempo
    start = time.time()
    predictions = model.predict(img_batch, verbose=0)
    inference_time_ms = (time.time() - start) * 1000
    
    # Interpretar resultado
    proba = predictions[0]
    predicted_idx = int(np.argmax(proba))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(proba[predicted_idx])
    
    # Flag de revisión manual si confianza baja
    requires_review = confidence < CONFIDENCE_THRESHOLD_LOW
    
    # Alerta si excede tiempo máximo
    if inference_time_ms > MAX_INFERENCE_TIME_MS:
        logger.warning(
            f"Inferencia lenta: {inference_time_ms:.1f}ms "
            f"(máximo: {MAX_INFERENCE_TIME_MS}ms)"
        )
    
    result = {
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {
            name: round(float(proba[i]), 4)
            for i, name in enumerate(CLASS_NAMES)
        },
        "requires_review": requires_review,
        "inference_time_ms": round(inference_time_ms, 2),
    }
    
    logger.info(
        f"Predicción: {predicted_class} "
        f"(confianza={confidence:.2%}, "
        f"tiempo={inference_time_ms:.1f}ms, "
        f"revisión={'SÍ' if requires_review else 'NO'})"
    )
    
    return result


def predict_batch(
    image_paths: list,
    model=None,
) -> list:
    """
    Predicción en lote para procesamiento automatizado.
    
    Interfaz de integración:
    - El pipeline de automatización puede procesar múltiples
      radiografías de una vez (ej: lote nocturno).
    
    Args:
        image_paths: Lista de rutas a imágenes.
        model: Modelo Keras (usa cache si no se proporciona).
    
    Returns:
        Lista de resultados (misma estructura que predict_single).
    """
    results = []
    for path in image_paths:
        try:
            result = predict_single(path, model=model)
            result["file_path"] = str(path)
            results.append(result)
        except Exception as e:
            logger.error(f"Error procesando {path}: {e}")
            results.append({
                "file_path": str(path),
                "error": str(e),
            })
    
    return results
