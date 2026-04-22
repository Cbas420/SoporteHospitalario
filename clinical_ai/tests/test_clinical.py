"""
Tests para el modulo clinico.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.clinical_classifier import build_clinical_model, preprocess_clinical_input


class TestClinicalPreprocessing:
    """Tests de preprocesamiento."""

    def test_preprocess_basic(self):
        """Test con datos basicos."""
        features = preprocess_clinical_input(
            age=45,
            sex="M",
            height_cm=175,
            weight_kg=80,
            symptoms={"has_cough": True, "has_fever": False},
        )

        assert features.shape == (1, 15)
        assert features.dtype == np.float32

    def test_preprocess_female(self):
        """Test con sexo femenino."""
        features = preprocess_clinical_input(
            age=30,
            sex="F",
            height_cm=165,
            weight_kg=60,
            symptoms={},
        )

        assert features[0, 1] == 0.0

    def test_preprocess_symptoms(self):
        """Test que los sintomas se codifican correctamente."""
        features = preprocess_clinical_input(
            age=50,
            sex="M",
            height_cm=180,
            weight_kg=90,
            symptoms={
                "has_cough": True,
                "has_chest_pain": True,
                "has_fatigue": False,
            },
        )

        assert features[0, 5] == 1.0
        assert features[0, 6] == 1.0
        assert features[0, 7] == 0.0


class TestClinicalModel:
    """Tests del modelo."""

    def test_model_build(self):
        """Test construccion del modelo."""
        model = build_clinical_model(input_dim=15, num_diseases=3)

        assert model.input_shape == (None, 15)
        assert model.output_shape == (None, 3)

    def test_model_prediction_shape(self):
        """Test que la prediccion tiene el shape correcto."""
        model = build_clinical_model(input_dim=15, num_diseases=3)
        model.compile(optimizer="adam", loss="binary_crossentropy")

        dummy_input = np.random.rand(5, 15).astype(np.float32)
        prediction = model.predict(dummy_input, verbose=0)

        assert prediction.shape == (5, 3)
        assert np.all((prediction >= 0) & (prediction <= 1))

    def test_model_training(self):
        """Test que el modelo puede entrenar."""
        model = build_clinical_model(input_dim=15, num_diseases=3)

        X_train = np.random.rand(100, 15).astype(np.float32)
        y_train = np.random.randint(0, 2, size=(100, 3)).astype(np.float32)

        history = model.fit(
            X_train,
            y_train,
            epochs=3,
            batch_size=16,
            verbose=0,
        )

        assert "loss" in history.history
        assert len(history.history["loss"]) == 3


def test_bmi_calculation():
    """Test que el BMI se calcula correctamente."""
    features = preprocess_clinical_input(
        age=30,
        sex="M",
        height_cm=170,
        weight_kg=70,
        symptoms={},
    )

    expected_bmi = 70 / (1.7**2)
    expected_bmi_normalized = expected_bmi / 50.0

    assert abs(features[0, 4] - expected_bmi_normalized) < 0.01