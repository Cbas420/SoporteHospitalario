"""
Tests del pipeline clinico de pacientes e imagenes.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.pipeline import prepare_patient_dataset
from src.data.repository import HospitalRepository


@pytest.fixture
def sample_pipeline_assets(tmp_path):
    raw_dir = tmp_path / "raw"
    csv_path = tmp_path / "patients.csv"

    records = []
    for diagnosis in ["COVID19", "Normal", "Pneumonia"]:
        diagnosis_dir = raw_dir / diagnosis
        diagnosis_dir.mkdir(parents=True)
        for index in range(2):
            image_name = f"{diagnosis.lower()}_{index:02d}.png"
            image_path = diagnosis_dir / image_name
            Image.new("RGB", (16, 16), color=(index * 30, 10, 10)).save(image_path)
            records.append(
                {
                    "patient_id": f"{diagnosis.lower()}_{index:02d}",
                    "image_name": image_name,
                    "image_path": str(image_path),
                    "diagnosis": diagnosis,
                    "age": 25 + index,
                    "sex": "F" if index % 2 == 0 else "M",
                    "admission_date": "2026-01-01",
                    "source": "test",
                }
            )

    duplicated_record = records[0].copy()
    invalid_record = {
        "patient_id": "invalid_01",
        "image_name": "missing.png",
        "image_path": str(tmp_path / "missing.png"),
        "diagnosis": "Unknown",
        "age": 50,
        "sex": "F",
        "admission_date": "2026-01-02",
        "source": "test",
    }
    pd.DataFrame(records + [duplicated_record, invalid_record]).to_csv(csv_path, index=False)

    return {
        "raw_dir": raw_dir,
        "csv_path": csv_path,
        "output_path": tmp_path / "patients_clean.csv",
        "manifest_path": tmp_path / "dataset_manifest.json",
    }


def test_prepare_patient_dataset_removes_invalid_rows(sample_pipeline_assets):
    cleaned_df, report = prepare_patient_dataset(
        csv_path=sample_pipeline_assets["csv_path"],
        raw_data_dir=sample_pipeline_assets["raw_dir"],
        output_path=sample_pipeline_assets["output_path"],
        manifest_path=sample_pipeline_assets["manifest_path"],
    )

    assert report["duplicates_removed"] == 1
    assert report["invalid_diagnoses_removed"] >= 1
    assert all(cleaned_df["diagnosis"].isin(["COVID19", "Normal", "Pneumonia"]))
    assert sample_pipeline_assets["output_path"].exists()
    assert sample_pipeline_assets["manifest_path"].exists()


def test_repository_upserts_cleaned_patients(sample_pipeline_assets):
    cleaned_df, _ = prepare_patient_dataset(
        csv_path=sample_pipeline_assets["csv_path"],
        raw_data_dir=sample_pipeline_assets["raw_dir"],
        output_path=sample_pipeline_assets["output_path"],
        manifest_path=sample_pipeline_assets["manifest_path"],
    )

    repository = HospitalRepository("sqlite:///:memory:")
    repository.initialize()
    inserted = repository.upsert_patients(cleaned_df.to_dict(orient="records"))

    stats = repository.get_stats()
    assert inserted == len(cleaned_df)
    assert stats["patients_total"] == len(cleaned_df)
