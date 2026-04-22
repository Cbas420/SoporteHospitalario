"""
Generador de datos sintéticos para el modelo clínico.
Genera pacientes con síntomas y diagnósticos para entrenamiento.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.clinical_settings import (
    ALL_FEATURES,
    CLINICAL_DATA_DIR,
    DISEASE_COLUMNS,
    SYMPTOM_COLUMNS,
)

np.random.seed(42)


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Calcula el IMC."""
    return weight_kg / ((height_cm / 100) ** 2)


def generate_patient(
    patient_id: int,
    age_range: tuple[int, int] = (18, 85),
    sex_probs: tuple[float, float] = (0.48, 0.52),
) -> dict:
    """Genera un paciente sintético."""
    age = np.random.randint(*age_range)
    sex = np.random.choice(["M", "F"], p=sex_probs)

    if sex == "M":
        height = np.random.normal(175, 7)
        weight = np.random.normal(78, 15)
    else:
        height = np.random.normal(162, 6)
        weight = np.random.normal(66, 13)

    height = np.clip(height, 150, 210)
    weight = np.clip(weight, 45, 150)
    bmi = calculate_bmi(height, weight)

    symptoms = {
        "has_cough": bool(np.random.choice([0, 1], p=[0.7, 0.3])),
        "has_chest_pain": bool(np.random.choice([0, 1], p=[0.8, 0.2])),
        "has_fatigue": bool(np.random.choice([0, 1], p=[0.6, 0.4])),
        "has_fever": bool(np.random.choice([0, 1], p=[0.85, 0.15])),
        "has_dizziness": bool(np.random.choice([0, 1], p=[0.75, 0.25])),
        "has_breathing_difficulty": bool(np.random.choice([0, 1], p=[0.8, 0.2])),
        "has_headache": bool(np.random.choice([0, 1], p=[0.7, 0.3])),
        "has_nausea": bool(np.random.choice([0, 1], p=[0.8, 0.2])),
        "has_loss_of_appetite": bool(np.random.choice([0, 1], p=[0.75, 0.25])),
        "has_night_sweats": bool(np.random.choice([0, 1], p=[0.85, 0.15])),
    }

    return {
        "patient_id": f"CLIN_{patient_id:06d}",
        "age": age,
        "sex": sex,
        "height_cm": round(height, 1),
        "weight_kg": round(weight, 1),
        "bmi": round(bmi, 2),
        **symptoms,
    }


def determine_risks(patient: dict, patient_id: int) -> dict:
    """
    Determina los riesgos de enfermedad basados en correlaciones médicas reales.
    """
    age = patient["age"]
    bmi = patient["bmi"]
    has_fatigue = int(patient["has_fatigue"])
    has_chest_pain = int(patient["has_chest_pain"])
    has_breathing = int(patient["has_breathing_difficulty"])
    has_night_sweats = int(patient["has_night_sweats"])
    has_dizziness = int(patient["has_dizziness"])
    has_nausea = int(patient["has_nausea"])

    base_diabetes = (
        0.02
        + 0.008 * (bmi - 25)
        + 0.003 * max(0, age - 40)
        + 0.05 * has_fatigue
        + 0.03 * has_nausea
    )
    diabetes_risk = min(1.0, max(0.0, base_diabetes))

    base_hypertension = (
        0.03
        + 0.01 * (bmi - 25)
        + 0.005 * max(0, age - 35)
        + 0.06 * has_headache
        + 0.04 * has_dizziness
        + 0.03 * has_fatigue
    )
    hypertension_risk = min(1.0, max(0.0, base_hypertension))

    base_heart = (
        0.01
        + 0.004 * max(0, age - 45)
        + 0.02 * has_chest_pain
        + 0.08 * has_breathing
        + 0.03 * has_fatigue
        + 0.02 * has_night_sweats
    )
    heart_risk = min(1.0, max(0.0, base_heart))

    rng = np.random.RandomState(patient_id)
    return {
        "diabetes_risk": int(rng.random() < diabetes_risk),
        "hypertension_risk": int(rng.random() < hypertension_risk),
        "heart_disease_risk": int(rng.random() < heart_risk),
    }


def generate_clinical_dataset(
    num_patients: int = 1000,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Genera el dataset completo de pacientes clínicos.

    Args:
        num_patients: Numero de pacientes a generar
        output_path: Path de salida (opcional)

    Returns:
        DataFrame con todos los pacientes
    """
    print(f"Generando {num_patients} pacientes clinicos...")

    patients = []
    for i in range(num_patients):
        patient = generate_patient(i)
        risks = determine_risks(patient, i)
        patient.update(risks)
        patients.append(patient)

    df = pd.DataFrame(patients)
    print(f"  - Pacientes生成ados: {len(df)}")
    print(f"  - Diabetes: {df['diabetes_risk'].sum()} casos ({df['diabetes_risk'].mean()*100:.1f}%)")
    print(f"  - Hipertension: {df['hypertension_risk'].sum()} casos ({df['hypertension_risk'].mean()*100:.1f}%)")
    print(f"  - Cardiaca: {df['heart_disease_risk'].sum()} casos ({df['heart_disease_risk'].mean()*100:.1f}%)")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  - Guardado en: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generador de datos clinicos sintéticos"
    )
    parser.add_argument(
        "--num-patients",
        type=int,
        default=1000,
        help="Numero de pacientes a generar",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(CLINICAL_DATA_DIR / "clinical_dataset.csv"),
        help="Path de salida",
    )
    args = parser.parse_args()

    generate_clinical_dataset(
        num_patients=args.num_patients,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()