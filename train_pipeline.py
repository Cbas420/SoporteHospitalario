"""
Pipeline principal de entrenamiento y evaluacion del sistema hospitalario.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import BATCH_SIZE, CLASS_NAMES, EPOCHS, MODELS_DIR, PATIENTS_CSV_PATH, RAW_DATA_DIR
from src.data.loader import (
    compute_class_weights,
    create_data_generators,
    generate_dataset_report,
    organize_split_directories,
    scan_dataset,
    split_dataset,
)
from src.data.pipeline import prepare_data_pipeline
from src.evaluation.evaluator import (
    clinical_error_analysis,
    compute_metrics,
    generate_evaluation_report,
    plot_confusion_matrix,
    plot_training_history,
    predict_dataset,
)
from src.model.trainer import train_model
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de entrenamiento del clasificador de radiografias"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Directorio con imagenes organizadas por clase",
    )
    parser.add_argument(
        "--patients-csv",
        type=str,
        default=str(PATIENTS_CSV_PATH),
        help="CSV opcional de pacientes para el pipeline de datos",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Numero de epochs para transfer learning",
    )
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=15,
        help="Numero de epochs para fine-tuning",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Tamano de batch",
    )
    parser.add_argument(
        "--no-finetune",
        action="store_true",
        help="Omitir la fase de fine-tuning",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Omitir entrenamiento y solo evaluar un modelo existente",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_total = time.time()
    data_dir = Path(args.data_dir)

    logger.info("=" * 70)
    logger.info("PIPELINE DE ENTRENAMIENTO - SISTEMA HOSPITALARIO")
    logger.info("=" * 70)

    logger.info("PASO 0: Preparando pipeline de datos clinicos...")
    pipeline_report = prepare_data_pipeline(
        csv_path=Path(args.patients_csv),
        raw_data_dir=data_dir,
    )
    logger.info(
        "Pacientes validos: %s | Imagenes totales: %s",
        pipeline_report["patients"]["records_after_cleaning"],
        pipeline_report["dataset"]["total_images"],
    )

    logger.info("PASO 1: Escaneando dataset...")
    dataset = scan_dataset(data_dir)
    report = generate_dataset_report(dataset)

    total_images = report["total_images"]
    if total_images == 0:
        logger.error(
            "No se encontraron imagenes en %s. Se espera un dataset plano por clase "
            "o la estructura original con images/ y masks/",
            data_dir,
        )
        sys.exit(1)

    logger.info("Total de imagenes: %s", total_images)
    for cls, count in report["classes"].items():
        logger.info("%s: %s (%.1f%%)", cls, count, report["balance_ratio"][cls] * 100)

    if report.get("imbalance_detected"):
        logger.warning("Desbalanceo significativo detectado en el dataset.")

    logger.info("PASO 2: Dividiendo dataset (70/15/15)...")
    train_data, val_data, test_data = split_dataset(dataset)
    organize_split_directories(train_data, val_data, test_data)
    class_weights = compute_class_weights(train_data)

    logger.info("PASO 3: Creando generadores de datos...")
    train_ds, val_ds, test_ds, _ = create_data_generators(batch_size=args.batch_size)

    if not args.skip_training:
        logger.info("PASO 4: Entrenando modelo...")
        model, history = train_model(
            train_ds=train_ds,
            val_ds=val_ds,
            epochs=args.epochs,
            fine_tune=not args.no_finetune,
            fine_tune_epochs=args.finetune_epochs,
            class_weights=class_weights,
        )
        plot_training_history(history)
    else:
        logger.info("PASO 4: Cargando modelo existente...")
        from src.model.trainer import load_trained_model

        model = load_trained_model()
        history_path = MODELS_DIR / "training_history.json"
        if history_path.exists():
            with history_path.open("r", encoding="utf-8") as file_handle:
                history = json.load(file_handle)
        else:
            history = {}

    logger.info("PASO 5: Evaluando modelo...")
    y_true, y_pred, _ = predict_dataset(model, test_ds)
    metrics = compute_metrics(y_true, y_pred)

    logger.info("Accuracy global: %.4f", metrics["global"]["accuracy"])
    logger.info("Precision macro: %.4f", metrics["global"]["precision_macro"])
    logger.info("Recall macro: %.4f", metrics["global"]["recall_macro"])
    logger.info("F1 macro: %.4f", metrics["global"]["f1_macro"])

    for cls_name in CLASS_NAMES:
        cls_metrics = metrics["per_class"][cls_name]
        logger.info(
            "%s -> P=%.3f R=%.3f F1=%.3f Spec=%.3f (n=%s)",
            cls_name,
            cls_metrics["precision"],
            cls_metrics["recall"],
            cls_metrics["f1_score"],
            cls_metrics["specificity"],
            cls_metrics["support"],
        )

    import numpy as np

    plot_confusion_matrix(np.array(metrics["confusion_matrix"]))

    logger.info("PASO 6: Analisis clinico de errores...")
    clinical = clinical_error_analysis(metrics)
    for alert in clinical.get("risk_assessment", []):
        logger.warning(alert)

    generate_evaluation_report(metrics, clinical)

    total_time = time.time() - start_total
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETADO EN %.1fs", total_time)
    logger.info("Modelo guardado en: %s", MODELS_DIR / "chest_xray_classifier.keras")
    logger.info("Informe en: %s", MODELS_DIR / "evaluation_report.json")
    logger.info("Matriz de confusion en: %s", MODELS_DIR / "confusion_matrix.png")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
