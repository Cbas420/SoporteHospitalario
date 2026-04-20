"""
Tests del modulo de carga y preprocesamiento de datos.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import CLASS_NAMES
from src.data.loader import (
    compute_class_weights,
    generate_dataset_report,
    organize_split_directories,
    scan_dataset,
    split_dataset,
)


@pytest.fixture
def temp_dataset(tmp_path):
    counts = {"COVID19": 20, "Normal": 25, "Pneumonia": 15}

    for class_name, count in counts.items():
        class_dir = tmp_path / class_name
        class_dir.mkdir()
        for i in range(count):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(class_dir / f"img_{i:04d}.png")

    return tmp_path, counts


def test_scan_dataset(temp_dataset):
    data_dir, expected_counts = temp_dataset
    dataset = scan_dataset(data_dir)

    for class_name in CLASS_NAMES:
        assert class_name in dataset
        assert len(dataset[class_name]) == expected_counts[class_name]


def test_scan_dataset_nonexistent():
    with pytest.raises(FileNotFoundError):
        scan_dataset(Path("/nonexistent/path"))


def test_split_dataset(temp_dataset):
    data_dir, _ = temp_dataset
    dataset = scan_dataset(data_dir)

    train, val, test = split_dataset(dataset)

    for class_name in CLASS_NAMES:
        total = len(dataset[class_name])
        train_count = len(train[class_name])
        val_count = len(val[class_name])
        test_count = len(test[class_name])

        assert train_count + val_count + test_count == total
        assert abs(train_count / total - 0.70) < 0.10
        assert abs(val_count / total - 0.15) < 0.10
        assert abs(test_count / total - 0.15) < 0.10


def test_split_no_overlap(temp_dataset):
    data_dir, _ = temp_dataset
    dataset = scan_dataset(data_dir)
    train, val, test = split_dataset(dataset)

    for class_name in CLASS_NAMES:
        train_set = set(train[class_name])
        val_set = set(val[class_name])
        test_set = set(test[class_name])

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0


def test_dataset_report(temp_dataset):
    data_dir, _ = temp_dataset
    dataset = scan_dataset(data_dir)
    report = generate_dataset_report(dataset)

    assert report["total_images"] == 60
    assert "COVID19" in report["classes"]
    assert "balance_ratio" in report
    assert isinstance(report["imbalance_detected"], bool)


def test_dataset_report_empty():
    dataset = {cls: [] for cls in CLASS_NAMES}
    report = generate_dataset_report(dataset)
    assert report["total_images"] == 0


def test_scan_dataset_merges_original_classes_with_masks(tmp_path):
    raw_dir = tmp_path / "raw"
    class_specs = {
        "COVID": ("COVID19", 2),
        "Normal": ("Normal", 3),
        "Lung_Opacity": ("Pneumonia", 4),
        "Viral Pneumonia": ("Pneumonia", 1),
    }

    for source_class, (_, count) in class_specs.items():
        image_dir = raw_dir / source_class / "images"
        mask_dir = raw_dir / source_class / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        for index in range(count):
            image = Image.fromarray(
                np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            )
            mask = Image.fromarray(
                np.random.randint(0, 255, (32, 32), dtype=np.uint8)
            )
            image.save(image_dir / f"{source_class}_{index:02d}.png")
            mask.save(mask_dir / f"{source_class}_{index:02d}.png")

    dataset = scan_dataset(raw_dir)

    assert len(dataset["COVID19"]) == 2
    assert len(dataset["Normal"]) == 3
    assert len(dataset["Pneumonia"]) == 5
    assert dataset["Pneumonia"][0]["target_class"] == "Pneumonia"
    assert dataset["Pneumonia"][0]["mask_path"]


def test_organize_split_directories_applies_resize_with_masks(tmp_path):
    raw_dir = tmp_path / "raw"
    image_dir = raw_dir / "COVID" / "images"
    mask_dir = raw_dir / "COVID" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image_path = image_dir / "covid_01.png"
    mask_path = mask_dir / "covid_01.png"
    Image.new("RGB", (128, 128), color=(255, 0, 0)).save(image_path)
    Image.new("L", (128, 128), color=255).save(mask_path)

    dataset = scan_dataset(raw_dir)
    output_dir = organize_split_directories(
        train_data={"COVID19": dataset["COVID19"], "Normal": [], "Pneumonia": []},
        val_data={class_name: [] for class_name in CLASS_NAMES},
        test_data={class_name: [] for class_name in CLASS_NAMES},
        output_dir=tmp_path / "processed",
    )

    generated_files = list((output_dir / "train" / "COVID19").glob("*.png"))
    assert len(generated_files) == 1
    generated_image = Image.open(generated_files[0])
    assert generated_image.size == (224, 224)


def test_compute_class_weights_prioritizes_minority_class():
    train_data = {
        "COVID19": ["a"] * 10,
        "Normal": ["b"] * 100,
        "Pneumonia": ["c"] * 50,
    }

    class_weights = compute_class_weights(train_data)

    assert class_weights[0] > class_weights[2] > class_weights[1]
