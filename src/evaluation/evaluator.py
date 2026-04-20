"""
Evaluacion del modelo y analisis clinico.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from config.settings import CLASS_NAMES, CONFIDENCE_THRESHOLD_LOW, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)
_MPL_CONFIG_DIR = Path(__file__).resolve().parents[2] / ".matplotlib"


def _prepare_matplotlib_runtime() -> None:
    _MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CONFIG_DIR))


def predict_dataset(model, test_ds) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import tensorflow as tf

    start_time = time.time()
    y_true_list = []
    y_proba_list = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_proba_list.append(predictions)
        y_true_list.append(labels.numpy())

    y_proba = np.concatenate(y_proba_list, axis=0)
    y_true_onehot = np.concatenate(y_true_list, axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_proba, axis=1)

    elapsed = time.time() - start_time
    avg_time = (elapsed / len(y_true)) * 1000 if len(y_true) else 0.0
    logger.info(
        "Predicciones completadas: %s imagenes, tiempo medio=%.1fms/imagen",
        len(y_true),
        avg_time,
    )
    return y_true, y_pred, y_proba


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list = CLASS_NAMES,
) -> Dict:
    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, predicted_label in zip(y_true, y_pred):
        cm[int(true_label), int(predicted_label)] += 1

    total_samples = int(len(y_true))
    accuracy = float(np.trace(cm) / total_samples) if total_samples else 0.0

    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    specificity_per_class = []

    for index in range(num_classes):
        tp = cm[index, index]
        fp = cm[:, index].sum() - tp
        fn = cm[index, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        precision_per_class.append(float(precision))
        recall_per_class.append(float(recall))
        f1_per_class.append(float(f1_score))
        specificity_per_class.append(float(specificity))

    metrics = {
        "global": {
            "accuracy": accuracy,
            "precision_macro": float(np.mean(precision_per_class)) if precision_per_class else 0.0,
            "recall_macro": float(np.mean(recall_per_class)) if recall_per_class else 0.0,
            "f1_macro": float(np.mean(f1_per_class)) if f1_per_class else 0.0,
            "total_samples": total_samples,
        },
        "per_class": {},
        "confusion_matrix": cm.tolist(),
    }

    for index, class_name in enumerate(class_names):
        metrics["per_class"][class_name] = {
            "precision": precision_per_class[index],
            "recall": recall_per_class[index],
            "f1_score": f1_per_class[index],
            "specificity": specificity_per_class[index],
            "support": int(np.sum(y_true == index)),
        }

    return metrics


def clinical_error_analysis(
    metrics: Dict,
    class_names: list = CLASS_NAMES,
) -> Dict:
    cm = np.array(metrics["confusion_matrix"])
    analysis = {"risk_assessment": [], "recommendations": []}

    covid_idx = class_names.index("COVID19")
    covid_fn = cm[covid_idx, :].sum() - cm[covid_idx, covid_idx]
    covid_recall = metrics["per_class"]["COVID19"]["recall"]

    analysis["covid19_false_negatives"] = {
        "count": int(covid_fn),
        "recall": float(covid_recall),
        "clinical_impact": "ALTO",
        "description": (
            "Un falso negativo en COVID-19 significa que un paciente contagioso "
            "no es identificado, lo que puede generar brotes hospitalarios."
        ),
        "mitigation": (
            "Predicciones con confianza < 60% deben derivarse a revision manual."
        ),
    }

    pneumonia_idx = class_names.index("Pneumonia")
    pneumonia_fn = cm[pneumonia_idx, :].sum() - cm[pneumonia_idx, pneumonia_idx]
    pneumonia_recall = metrics["per_class"]["Pneumonia"]["recall"]

    analysis["pneumonia_false_negatives"] = {
        "count": int(pneumonia_fn),
        "recall": float(pneumonia_recall),
        "clinical_impact": "MEDIO-ALTO",
        "description": (
            "Un falso negativo en neumonia retrasa el tratamiento y puede agravar al paciente."
        ),
    }

    for class_name in class_names:
        index = class_names.index(class_name)
        false_positives = cm[:, index].sum() - cm[index, index]
        precision = metrics["per_class"][class_name]["precision"]
        if false_positives > 0:
            analysis[f"{class_name.lower()}_false_positives"] = {
                "count": int(false_positives),
                "precision": float(precision),
                "clinical_impact": "MEDIO" if class_name != "Normal" else "BAJO",
                "description": (
                    f"Falsos positivos de {class_name}: {int(false_positives)} casos."
                ),
            }

    covid_as_pneumonia = cm[covid_idx, pneumonia_idx]
    pneumonia_as_covid = cm[pneumonia_idx, covid_idx]

    analysis["covid_pneumonia_confusion"] = {
        "covid_classified_as_pneumonia": int(covid_as_pneumonia),
        "pneumonia_classified_as_covid": int(pneumonia_as_covid),
        "clinical_impact": "ALTO",
        "description": (
            "La confusion COVID-19 <-> Neumonia es clinicamente relevante por similitud radiologica."
        ),
    }

    if covid_recall < 0.85:
        analysis["risk_assessment"].append(
            "ALERTA: Recall de COVID-19 por debajo del 85%."
        )

    if pneumonia_recall < 0.80:
        analysis["risk_assessment"].append(
            "ALERTA: Recall de Neumonia por debajo del 80%."
        )

    analysis["recommendations"].extend(
        [
            "El modelo debe usarse como herramienta de apoyo.",
            "Implementar revision manual obligatoria para confianza < 60%.",
            "Monitorizar drift del modelo.",
            "Actualizar periodicamente con datos nuevos del hospital.",
        ]
    )

    return analysis


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = CLASS_NAMES,
    output_path: Optional[Path] = None,
) -> Path:
    _prepare_matplotlib_runtime()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if output_path is None:
        output_path = MODELS_DIR / "confusion_matrix.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={"label": "Numero de predicciones"},
    )
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusion")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Matriz de confusion guardada en %s", output_path)
    return output_path


def plot_training_history(
    history: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    _prepare_matplotlib_runtime()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = MODELS_DIR / "training_history.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.get("loss", []), label="Train Loss", linewidth=2)
    axes[0].plot(history.get("val_loss", []), label="Val Loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.get("accuracy", []), label="Train Accuracy", linewidth=2)
    axes[1].plot(history.get("val_accuracy", []), label="Val Accuracy", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Curvas de entrenamiento guardadas en %s", output_path)
    return output_path


def generate_evaluation_report(
    metrics: Dict,
    clinical_analysis: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    if output_path is None:
        output_path = MODELS_DIR / "evaluation_report.json"

    report = {
        "model_name": "chest_xray_classifier",
        "version": "2.0.0",
        "evaluation_metrics": metrics,
        "clinical_analysis": clinical_analysis,
        "metadata": {
            "classes": CLASS_NAMES,
            "confidence_threshold_low": CONFIDENCE_THRESHOLD_LOW,
            "disclaimer": (
                "Este modelo es una herramienta de apoyo diagnostico. "
                "No sustituye el criterio medico profesional."
            ),
        },
    }

    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, indent=2, ensure_ascii=False)

    logger.info("Informe de evaluacion guardado en %s", output_path)
    return output_path
