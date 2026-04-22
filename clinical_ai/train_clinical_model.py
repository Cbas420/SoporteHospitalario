"""
Pipeline de entrenamiento para el modelo clínico de riesgos.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.clinical_settings import (
    ALL_FEATURES,
    CLINICAL_DATA_DIR,
    CLINICAL_MODELS_DIR,
    DISEASE_COLUMNS,
    MODEL_PATH,
)

from model.clinical_classifier import build_clinical_model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga y preprocesa el dataset.

    Returns:
        X: Features (n_samples, n_features)
        y: Labels (n_samples, n_diseases)
    """
    df = pd.read_csv(csv_path)

    X = df[ALL_FEATURES].values.astype(np.float32)

    X[:, 0] = X[:, 0] / 100.0
    X[:, 2] = X[:, 2] / 250.0
    X[:, 3] = X[:, 3] / 200.0
    X[:, 4] = X[:, 4] / 50.0

    y = df[DISEASE_COLUMNS].values.astype(np.float32)

    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple:
    """Divide datos en train/val/test."""
    n = len(X)
    indices = np.random.permutation(n)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        X[train_idx],
        X[val_idx],
        X[test_idx],
        y[train_idx],
        y[val_idx],
        y[test_idx],
    )


def train_clinical_model(
    data_path: Path,
    epochs: int = 50,
    batch_size: int = 32,
    output_path: Path | None = None,
) -> dict:
    """
    Entrena el modelo clínico.

    Args:
        data_path: Path al CSV del dataset
        epochs: Numero de epochs
        batch_size: Tamano de batch
        output_path: Path de salida del modelo

    Returns:
        Diccionario con historial
    """
    output_path = output_path or MODEL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ENTRENAMIENTO MODELO CLINICO")
    print("=" * 60)

    print(f"\n1. Cargando dataset...")
    X, y = load_dataset(data_path)
    print(f"   Samples: {X.shape[0]}, Features: {X.shape[1]}, Enfermedades: {y.shape[1]}")

    print(f"\n2. Dividiendo datos (70/15/15)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print(f"\n3. Construyendo modelo...")
    model = build_clinical_model(input_dim=X.shape[1], num_diseases=y.shape[1])
    model.summary()

    print(f"\n4. Entrenando...")
    start_time = time.time()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
            ModelCheckpoint(
                str(output_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
        ],
        verbose=1,
    )

    train_time = time.time() - start_time
    print(f"\n   Tiempo de entrenamiento: {train_time:.1f}s")

    print(f"\n5. Evaluando en test set...")
    results = model.evaluate(X_test, y_test, verbose=1)
    print(f"   Loss: {results[0]:.4f}")
    print(f"   Accuracy: {results[1]:.4f}")
    print(f"   AUC: {results[2]:.4f}")

    print(f"\n6. Evaluando por enfermedad...")
    y_pred_proba = model.predict(X_test, verbose=0)
    threshold = 0.5

    for i, disease in enumerate(DISEASE_COLUMNS):
        y_pred = (y_pred_proba[:, i] >= threshold).astype(int)
        y_true = y_test[:, i]

        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n   {disease}:")
        print(f"     Precision: {precision:.4f}")
        print(f"     Recall: {recall:.4f}")
        print(f"     F1-Score: {f1:.4f}")
        print(f"     TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

    history_dict = {
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "training_time_seconds": float(train_time),
        "test_metrics": {
            "loss": float(results[0]),
            "accuracy": float(results[1]),
            "auc": float(results[2]),
        },
    }

    history_path = CLINICAL_MODELS_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=2, default=str)
    print(f"\n   Historial guardado en: {history_path}")

    model.save(str(output_path))
    print(f"   Modelo guardado en: {output_path}")

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)

    return history_dict


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo clinico"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(CLINICAL_DATA_DIR / "clinical_dataset.csv"),
        help="Path al dataset CSV",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Numero de epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamano de batch",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(MODEL_PATH),
        help="Path de salida del modelo",
    )
    args = parser.parse_args()

    train_clinical_model(
        data_path=Path(args.data),
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()