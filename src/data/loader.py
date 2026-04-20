"""
Carga y preprocesamiento de datos de imagen.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from config.settings import (
    AUGMENTATION_CONFIG,
    BATCH_SIZE,
    CLASS_NAMES,
    IMG_SIZE,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
RAW_TO_TARGET_CLASS = {
    "COVID": "COVID19",
    "Normal": "Normal",
    "Lung_Opacity": "Pneumonia",
    "Viral Pneumonia": "Pneumonia",
}


def _resolve_xrays_root(data_dir: Path) -> Path:
    xrays_root = data_dir / "xrays"
    return xrays_root if xrays_root.exists() else data_dir


def _dataset_uses_images_and_masks(data_dir: Path) -> bool:
    xrays_root = _resolve_xrays_root(data_dir)
    return any((xrays_root / raw_class / "images").exists() for raw_class in RAW_TO_TARGET_CLASS)


def _build_dataset_entry(
    image_path: Path,
    target_class: str,
    source_class: str,
    mask_path: Path | None = None,
) -> dict[str, str]:
    return {
        "image_path": str(image_path),
        "image_name": image_path.name,
        "patient_id": image_path.stem,
        "source_class": source_class,
        "target_class": target_class,
        "mask_path": str(mask_path) if mask_path and mask_path.exists() else "",
    }


def _scan_dataset_with_masks(data_dir: Path) -> Dict[str, list]:
    dataset = {class_name: [] for class_name in CLASS_NAMES}
    xrays_root = _resolve_xrays_root(data_dir)

    for raw_class, target_class in RAW_TO_TARGET_CLASS.items():
        images_dir = xrays_root / raw_class / "images"
        masks_dir = xrays_root / raw_class / "masks"

        if not images_dir.exists():
            logger.warning("Directorio de imagenes no encontrado: %s", images_dir)
            continue

        image_paths = sorted(
            file_path
            for file_path in images_dir.iterdir()
            if file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )

        entries = [
            _build_dataset_entry(
                image_path=image_path,
                target_class=target_class,
                source_class=raw_class,
                mask_path=masks_dir / image_path.name if masks_dir.exists() else None,
            )
            for image_path in image_paths
        ]
        dataset[target_class].extend(entries)

        logger.info(
            "Clase origen %s -> %s: %s imagenes%s",
            raw_class,
            target_class,
            len(entries),
            " con mascaras" if masks_dir.exists() else " sin mascaras",
        )

    for target_class, entries in dataset.items():
        logger.info("Clase final %s: %s imagenes", target_class, len(entries))

    return dataset


def _scan_flat_dataset(data_dir: Path) -> Dict[str, list]:
    dataset = {}
    xrays_root = _resolve_xrays_root(data_dir)

    for class_name in CLASS_NAMES:
        class_dir = xrays_root / class_name
        if not class_dir.exists():
            logger.warning("Directorio de clase no encontrado: %s", class_dir)
            dataset[class_name] = []
            continue

        images = [
            str(file_path)
            for file_path in class_dir.iterdir()
            if file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        dataset[class_name] = sorted(images)
        logger.info("Clase %s: %s imagenes encontradas", class_name, len(images))

    return dataset


def scan_dataset(data_dir: Path) -> Dict[str, list]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Directorio de datos no encontrado: {data_dir}")

    if _dataset_uses_images_and_masks(data_dir):
        return _scan_dataset_with_masks(data_dir)

    return _scan_flat_dataset(data_dir)


def split_dataset(
    dataset: Dict[str, list],
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
) -> Tuple[Dict[str, list], Dict[str, list], Dict[str, list]]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    train_data: Dict[str, list] = {}
    val_data: Dict[str, list] = {}
    test_data: Dict[str, list] = {}

    for class_name, items in dataset.items():
        if len(items) < 3:
            logger.warning(
                "Clase %s tiene menos de 3 imagenes. No se puede dividir correctamente.",
                class_name,
            )
            train_data[class_name] = list(items)
            val_data[class_name] = []
            test_data[class_name] = []
            continue

        shuffled_items = list(items)
        random.Random(seed).shuffle(shuffled_items)

        total_items = len(shuffled_items)
        train_count = max(1, int(round(total_items * train_ratio)))
        val_count = max(1, int(round(total_items * val_ratio)))
        test_count = total_items - train_count - val_count

        if test_count <= 0:
            test_count = 1
            if train_count >= val_count and train_count > 1:
                train_count -= 1
            elif val_count > 1:
                val_count -= 1

        train_data[class_name] = shuffled_items[:train_count]
        val_data[class_name] = shuffled_items[train_count:train_count + val_count]
        test_data[class_name] = shuffled_items[train_count + val_count:]

        logger.info(
            "Clase %s: train=%s, val=%s, test=%s",
            class_name,
            len(train_data[class_name]),
            len(val_data[class_name]),
            len(test_data[class_name]),
        )

    return train_data, val_data, test_data


def _entry_image_path(item: Any) -> Path:
    if isinstance(item, dict):
        return Path(item["image_path"])
    return Path(str(item))


def _entry_mask_path(item: Any) -> Path | None:
    if isinstance(item, dict):
        mask_path = str(item.get("mask_path", "")).strip()
        if mask_path:
            return Path(mask_path)
    return None


def _entry_output_name(item: Any) -> str:
    if not isinstance(item, dict):
        return _entry_image_path(item).name

    source_class = str(item.get("source_class", "")).strip().replace(" ", "_")
    image_name = str(item.get("image_name") or _entry_image_path(item).name)
    return f"{source_class}__{image_name}" if source_class else image_name


def _save_preprocessed_image(item: Any, destination: Path) -> None:
    image_path = _entry_image_path(item)
    mask_path = _entry_mask_path(item)

    if not isinstance(item, dict):
        shutil.copy2(image_path, destination)
        return

    image = Image.open(image_path).convert("RGB")
    mask = None
    if mask_path and mask_path.exists():
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            resampling = getattr(Image, "Resampling", Image)
            mask = mask.resize(image.size, resampling.BILINEAR)

    image_array = np.asarray(image, dtype=np.float32)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=np.float32) / 255.0
        mask_array = (mask_array > 0.5).astype(np.float32)
        image_array = image_array * mask_array[..., None]

    image = Image.fromarray(image_array.clip(0, 255).astype("uint8"))
    resampling = getattr(Image, "Resampling", Image)
    image = image.resize(IMG_SIZE, resampling.BILINEAR)
    image.save(destination)


def organize_split_directories(
    train_data: Dict[str, list],
    val_data: Dict[str, list],
    test_data: Dict[str, list],
    output_dir: Path = PROCESSED_DATA_DIR,
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    splits = {"train": train_data, "val": val_data, "test": test_data}

    for split_name, split_data in splits.items():
        for class_name, items in split_data.items():
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for item in items:
                destination = dest_dir / _entry_output_name(item)
                _save_preprocessed_image(item, destination)

            logger.info(
                "[%s/%s] %s imagenes preparadas",
                split_name,
                class_name,
                len(items),
            )

    return output_dir


def _collect_split_samples(split_dir: Path) -> Tuple[list[str], list[int]]:
    file_paths: list[str] = []
    labels: list[int] = []

    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            logger.warning("Directorio de split no encontrado: %s", class_dir)
            continue

        class_files = sorted(
            str(file_path)
            for file_path in class_dir.iterdir()
            if file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )
        file_paths.extend(class_files)
        labels.extend([label_index] * len(class_files))

    return file_paths, labels


def _build_dataset_from_directory(
    split_dir: Path,
    batch_size: int,
    img_size: Tuple[int, int],
    shuffle: bool,
):
    import tensorflow as tf

    file_paths, labels = _collect_split_samples(split_dir)
    if not file_paths:
        raise ValueError(f"No se encontraron imagenes en {split_dir}")

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(file_paths),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    def load_example(file_path, label):
        image_bytes = tf.io.read_file(file_path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32)
        label = tf.one_hot(label, depth=len(CLASS_NAMES))
        return image, label

    autotune = tf.data.AUTOTUNE
    return dataset.map(load_example, num_parallel_calls=autotune).batch(batch_size)


def _imagenet_standardize(images):
    import tensorflow as tf

    mean = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
    std = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
    images = tf.cast(images, tf.float32) / 255.0
    return (images - mean) / std


def create_data_generators(
    processed_dir: Path = PROCESSED_DATA_DIR,
    batch_size: int = BATCH_SIZE,
    img_size: Tuple[int, int] = IMG_SIZE,
) -> Tuple:
    import tensorflow as tf

    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"

    train_ds = _build_dataset_from_directory(
        train_dir,
        batch_size=batch_size,
        img_size=img_size,
        shuffle=True,
    )

    val_ds = _build_dataset_from_directory(
        val_dir,
        batch_size=batch_size,
        img_size=img_size,
        shuffle=False,
    )

    test_ds = _build_dataset_from_directory(
        test_dir,
        batch_size=batch_size,
        img_size=img_size,
        shuffle=False,
    )

    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(AUGMENTATION_CONFIG["rotation_range"] / 360.0),
            tf.keras.layers.RandomZoom(AUGMENTATION_CONFIG["zoom_range"]),
            tf.keras.layers.RandomTranslation(
                height_factor=AUGMENTATION_CONFIG["height_shift_range"],
                width_factor=AUGMENTATION_CONFIG["width_shift_range"],
            ),
        ],
        name="data_augmentation",
    )

    def preprocess_train(images, labels):
        images = data_augmentation(images, training=True)
        images = _imagenet_standardize(images)
        return images, labels

    def preprocess_eval(images, labels):
        images = _imagenet_standardize(images)
        return images, labels

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.map(preprocess_train, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_ds.map(preprocess_eval, num_parallel_calls=autotune).prefetch(autotune)
    test_ds = test_ds.map(preprocess_eval, num_parallel_calls=autotune).prefetch(autotune)

    logger.info("Generadores de datos creados exitosamente")
    return train_ds, val_ds, test_ds, CLASS_NAMES


def compute_class_weights(train_data: Dict[str, list]) -> Dict[int, float]:
    counts = {class_name: len(train_data.get(class_name, [])) for class_name in CLASS_NAMES}
    total = sum(counts.values())

    if total == 0:
        return {index: 1.0 for index, _ in enumerate(CLASS_NAMES)}

    class_weights: Dict[int, float] = {}
    for index, class_name in enumerate(CLASS_NAMES):
        count = counts[class_name]
        class_weights[index] = total / (len(CLASS_NAMES) * max(count, 1))

    logger.info("Class weights calculados: %s", class_weights)
    return class_weights


def generate_dataset_report(dataset: Dict[str, list]) -> Dict:
    total = sum(len(values) for values in dataset.values())
    report = {
        "total_images": total,
        "classes": {},
        "balance_ratio": {},
        "imbalance_detected": False,
    }

    for class_name, items in dataset.items():
        count = len(items)
        report["classes"][class_name] = count
        report["balance_ratio"][class_name] = round(count / total, 3) if total > 0 else 0

    ratios = list(report["balance_ratio"].values())
    if ratios:
        max_ratio = max(ratios)
        min_ratio = min(ratios)
        report["imbalance_detected"] = (max_ratio / min_ratio) > 2.0 if min_ratio > 0 else True

    logger.info("Dataset report: %s", json.dumps(report, indent=2))
    return report
