"""
Orquestador de entrenamiento del modelo.

Responsabilidades:
- Entrenar el modelo en dos fases (transfer learning + fine-tuning)
- Gestionar callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Guardar el modelo y el historial de entrenamiento
- Exportar métricas para monitorización

Decisiones técnicas:

1. Entrenamiento en dos fases:
   - Fase 1: Solo capas densas → convergencia rápida (~5 epochs).
   - Fase 2: Fine-tuning de capas profundas → adaptación al dominio (~15 epochs).
   Esto evita corromper los pesos preentrenados al inicio.

2. EarlyStopping monitorizando val_loss:
   - No val_accuracy, porque loss es más sensible a overfitting temprano.
   - Patience=5 para dar margen a fluctuaciones normales.

3. ModelCheckpoint guardando solo el mejor modelo:
   - Ahorra espacio y evita guardar modelos overfitteados.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict

import tensorflow as tf

from config.settings import (
    EPOCHS, BATCH_SIZE,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    REDUCE_LR_FACTOR, REDUCE_LR_PATIENCE, REDUCE_LR_MIN,
    MODEL_PATH, MODEL_WEIGHTS_PATH, TRAINING_HISTORY_PATH, MODELS_DIR,
)
from src.model.classifier import (
    build_model, compile_model, unfreeze_for_finetuning,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_callbacks(phase: str = "transfer_learning") -> list:
    """
    Configura callbacks para el entrenamiento.
    
    Args:
        phase: Fase de entrenamiento ('transfer_learning' o 'fine_tuning').
    
    Returns:
        Lista de callbacks de Keras.
    """
    callbacks = [
        # Detener si no mejora para evitar overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reducir LR si el progreso se estanca
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,
            verbose=1,
        ),
        # Guardar solo el mejor modelo
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
    
    # TensorBoard para visualización (integrable con dashboard)
    log_dir = MODELS_DIR / "logs" / phase
    log_dir.mkdir(parents=True, exist_ok=True)
    callbacks.append(
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir),
            histogram_freq=1,
        )
    )
    
    return callbacks


def train_model(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int = EPOCHS,
    fine_tune: bool = True,
    fine_tune_epochs: int = 15,
    class_weights: Optional[Dict[int, float]] = None,
) -> tuple:
    """
    Ejecuta el entrenamiento completo del modelo en dos fases.
    
    Fase 1 - Transfer Learning:
        - Backbone congelado
        - Solo se entrenan las capas densas personalizadas
        - LR estándar (1e-4)
    
    Fase 2 - Fine-Tuning (opcional):
        - Se descongelan los últimos bloques del backbone
        - LR reducido (1e-5)
        - Permite adaptar features de alto nivel al dominio médico
    
    Args:
        train_ds: Dataset de entrenamiento.
        val_ds: Dataset de validación.
        epochs: Epochs para fase 1.
        fine_tune: Si True, ejecuta también fase 2.
        fine_tune_epochs: Epochs adicionales para fase 2.
    
    Returns:
        Tupla (modelo entrenado, historial completo).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # =================================================================
    # FASE 1: Transfer Learning (backbone congelado)
    # =================================================================
    logger.info("=" * 60)
    logger.info("FASE 1: Transfer Learning (capas base congeladas)")
    logger.info("=" * 60)
    
    model = build_model(freeze_base=True)
    model = compile_model(model)
    
    start_time = time.time()
    
    history_phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=get_callbacks("transfer_learning"),
        class_weight=class_weights,
        verbose=1,
    )
    
    phase1_time = time.time() - start_time
    logger.info(f"Fase 1 completada en {phase1_time:.1f}s")
    
    # Historial combinado
    full_history = {k: list(v) for k, v in history_phase1.history.items()}
    
    # =================================================================
    # FASE 2: Fine-Tuning (descongelar capas profundas)
    # =================================================================
    if fine_tune:
        logger.info("=" * 60)
        logger.info("FASE 2: Fine-Tuning (capas profundas descongeladas)")
        logger.info("=" * 60)
        
        model = unfreeze_for_finetuning(model)
        
        start_time = time.time()
        
        history_phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=fine_tune_epochs,
            callbacks=get_callbacks("fine_tuning"),
            class_weight=class_weights,
            verbose=1,
        )
        
        phase2_time = time.time() - start_time
        logger.info(f"Fase 2 completada en {phase2_time:.1f}s")
        
        # Acumular historial
        for key, values in history_phase2.history.items():
            if key in full_history:
                full_history[key].extend(list(values))
            else:
                full_history[key] = list(values)
    
    # =================================================================
    # Guardar historial de entrenamiento
    # =================================================================
    # Convertir a tipos serializables
    serializable_history = {}
    for key, values in full_history.items():
        serializable_history[key] = [float(v) for v in values]
    
    with open(TRAINING_HISTORY_PATH, "w") as f:
        json.dump(serializable_history, f, indent=2)

    model.save_weights(str(MODEL_WEIGHTS_PATH))

    logger.info(f"Historial guardado en {TRAINING_HISTORY_PATH}")
    logger.info(f"Mejor modelo guardado en {MODEL_PATH}")
    logger.info(f"Pesos del modelo guardados en {MODEL_WEIGHTS_PATH}")
    
    return model, full_history


def load_trained_model(model_path: Path = MODEL_PATH) -> tf.keras.Model:
    """
    Carga un modelo previamente entrenado desde disco.
    
    Interfaz de integración:
    - La API REST usará esta función para cargar el modelo al iniciar.
    - El pipeline de automatización puede invocarla para predicciones batch.
    
    Args:
        model_path: Ruta al archivo .keras del modelo.
    
    Returns:
        Modelo Keras cargado y listo para inferencia.
    
    Raises:
        FileNotFoundError: Si el modelo no existe.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {model_path}. "
            f"Ejecuta el entrenamiento primero."
        )
    
    model = tf.keras.models.load_model(str(model_path))
    logger.info(f"Modelo cargado desde {model_path}")
    
    return model
