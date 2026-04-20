"""
Constructor del modelo de clasificación de radiografías.

Arquitectura:
- Base: ResNet50 preentrenado en ImageNet (sin top)
- Global Average Pooling
- Dense 256 + BatchNorm + ReLU + Dropout(0.5)
- Dense 3 + Softmax (salida: COVID19, Normal, Pneumonia)

Decisiones técnicas:

1. ResNet50 sobre alternativas:
   - VGG16: Más parámetros, peor ratio rendimiento/coste.
   - InceptionV3: Bueno, pero ResNet50 tiene mejor rendimiento
     documentado en clasificación de radiografías de tórax.
   - EfficientNet: Mejor rendimiento potencial, pero más complejo
     de ajustar. Se deja como mejora futura.

2. Congelación parcial:
   - Se congelan todas las capas inicialmente para entrenamiento rápido.
   - En fase de fine-tuning, se descongelan las últimas ~35 capas
     (bloques conv5 de ResNet50) para adaptar features de alto nivel
     al dominio médico.

3. Dropout 0.5:
   - Agresivo pero necesario con datasets médicos pequeños.
   - Combinado con data augmentation para máxima regularización.

4. BatchNormalization después de Dense:
   - Estabiliza el entrenamiento durante fine-tuning.
   - Permite learning rates más altos sin divergencia.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_KERAS_HOME = _PROJECT_ROOT / ".keras"
_KERAS_TMP = _KERAS_HOME / "tmp"
_KERAS_HOME.mkdir(parents=True, exist_ok=True)
_KERAS_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("KERAS_HOME", str(_KERAS_HOME))
os.environ.setdefault("TMP", str(_KERAS_TMP))
os.environ.setdefault("TEMP", str(_KERAS_TMP))
os.environ.setdefault("TMPDIR", str(_KERAS_TMP))
tempfile.tempdir = str(_KERAS_TMP)

_RESNET50_WEIGHTS = (
    _KERAS_HOME / "models" / "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
)

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

from config.settings import (
    IMG_SIZE, IMG_CHANNELS, NUM_CLASSES,
    DENSE_UNITS, DROPOUT_RATE,
    FREEZE_BASE_LAYERS, UNFREEZE_FROM_LAYER,
    LEARNING_RATE,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_model(
    num_classes: int = NUM_CLASSES,
    img_size: tuple = IMG_SIZE,
    freeze_base: bool = FREEZE_BASE_LAYERS,
    weights: Optional[str] = "imagenet",
) -> Model:
    """
    Construye el modelo de clasificación con transfer learning.
    
    Args:
        num_classes: Número de clases de salida.
        img_size: Tupla (height, width) de la imagen de entrada.
        freeze_base: Si True, congela todas las capas del backbone.
    
    Returns:
        Modelo Keras compilado.
    """
    input_shape = (*img_size, IMG_CHANNELS)
    keras_data_home = Path(os.path.expanduser("~")) / ".keras" / "models"
    try:
        keras_data_home.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.debug("No se pudo preparar cache local de Keras en %s", keras_data_home)
    
    resolved_weights = weights
    if weights == "imagenet" and _RESNET50_WEIGHTS.exists():
        resolved_weights = str(_RESNET50_WEIGHTS)
        logger.info("Usando pesos locales de ResNet50 desde %s", _RESNET50_WEIGHTS)

    # --- Backbone: ResNet50 preentrenado ---
    base_model = ResNet50(
        weights=resolved_weights,
        include_top=False,          # Excluir capas de clasificación de ImageNet
        input_shape=input_shape,
    )
    
    # Congelar capas base para transfer learning
    base_model.trainable = not freeze_base
    
    logger.info(
        f"ResNet50 cargado: {len(base_model.layers)} capas, "
        f"trainable={base_model.trainable}"
    )
    
    # --- Capas de clasificación personalizadas ---
    inputs = tf.keras.Input(shape=input_shape, name="xray_input")
    
    # Forward pass por el backbone
    x = base_model(inputs, training=False)
    
    # Global Average Pooling: reduce dimensionalidad espacial
    # Decisión: GAP sobre Flatten para reducir parámetros y overfitting
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    
    # Capa densa con regularización
    x = layers.Dense(DENSE_UNITS, name="dense_features")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Activation("relu", name="relu_activation")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_regularization")(x)
    
    # Capa de salida: softmax para clasificación multiclase
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classification_output",
    )(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="chest_xray_classifier")
    
    logger.info(f"Modelo construido: {model.count_params():,} parámetros totales")
    
    return model


def compile_model(
    model: Model,
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """
    Compila el modelo con optimizador, loss y métricas.
    
    Decisiones:
    - Adam: convergencia rápida, adaptativo, estándar para transfer learning.
    - Categorical Crossentropy: apropiado para clasificación multiclase
      con labels one-hot.
    - Métricas: accuracy + AUC para evaluación durante entrenamiento.
    
    Args:
        model: Modelo Keras sin compilar.
        learning_rate: Tasa de aprendizaje inicial.
    
    Returns:
        Modelo compilado.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )
    
    logger.info(f"Modelo compilado (lr={learning_rate})")
    return model


def unfreeze_for_finetuning(
    model: Model,
    unfreeze_from: int = UNFREEZE_FROM_LAYER,
    learning_rate: float = LEARNING_RATE / 10,
) -> Model:
    """
    Descongela las últimas capas del backbone para fine-tuning.
    
    Estrategia:
    - Se descongelan las últimas capas del ResNet50 (bloques conv5)
      que capturan features de alto nivel.
    - Se reduce el learning rate 10x para evitar destruir los pesos
      preentrenados con gradientes grandes.
    - Las capas BatchNormalization se mantienen en modo inferencia
      para estabilidad.
    
    Args:
        model: Modelo previamente entrenado con capas congeladas.
        unfreeze_from: Índice de capa desde la cual descongelar.
        learning_rate: LR reducido para fine-tuning.
    
    Returns:
        Modelo recompilado para fine-tuning.
    """
    base_model = model.layers[1]  # ResNet50 es la segunda capa
    base_model.trainable = True
    
    # Congelar capas iniciales, descongelar las finales
    frozen_count = 0
    unfrozen_count = 0
    
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
        frozen_count += 1
    
    for layer in base_model.layers[unfreeze_from:]:
        # Mantener BatchNorm en inferencia para estabilidad
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            frozen_count += 1
        else:
            unfrozen_count += 1
    
    logger.info(
        f"Fine-tuning: {frozen_count} capas congeladas, "
        f"{unfrozen_count} capas descongeladas"
    )
    
    # Recompilar con LR reducido
    model = compile_model(model, learning_rate=learning_rate)
    
    return model


def get_model_summary(model: Model) -> str:
    """Captura el summary del modelo como string para logging/documentación."""
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    return "\n".join(string_list)
