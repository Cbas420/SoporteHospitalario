"""
Tests del servicio de inferencia.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import CLASS_NAMES, IMG_SIZE, MODEL_PATH


def _runtime_model_is_available() -> bool:
    if not MODEL_PATH.exists():
        return False

    try:
        import tensorflow as tf

        tf.keras.models.load_model(str(MODEL_PATH), compile=False)
        return True
    except Exception:
        return False


class TestPreprocessing:
    def test_preprocess_from_numpy(self):
        from src.api.inference import preprocess_image

        img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        result = preprocess_image(img)

        assert result.shape == (1, *IMG_SIZE, 3)
        assert result.dtype == np.float32 or result.dtype == np.float64

    def test_preprocess_from_file(self, tmp_path):
        from src.api.inference import preprocess_image

        img = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        )
        filepath = tmp_path / "test.png"
        img.save(filepath)

        result = preprocess_image(filepath)
        assert result.shape == (1, *IMG_SIZE, 3)

    def test_preprocess_from_bytes(self):
        from src.api.inference import preprocess_image
        import io

        img = Image.fromarray(
            np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        )
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        result = preprocess_image(image_bytes)
        assert result.shape == (1, *IMG_SIZE, 3)

    def test_preprocess_invalid_type(self):
        from src.api.inference import preprocess_image

        with pytest.raises(ValueError):
            preprocess_image(12345)


@pytest.mark.skipif(
    not _runtime_model_is_available(),
    reason="Modelo entrenado compatible con este runtime no disponible",
)
class TestPrediction:
    def test_predict_single_returns_valid_format(self):
        from src.api.inference import predict_single

        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = predict_single(img)

        assert "prediction" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "requires_review" in result
        assert "inference_time_ms" in result

        assert result["prediction"] in CLASS_NAMES
        assert 0 <= result["confidence"] <= 1
        assert len(result["probabilities"]) == len(CLASS_NAMES)

    def test_predict_probabilities_sum_to_one(self):
        from src.api.inference import predict_single

        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = predict_single(img)

        prob_sum = sum(result["probabilities"].values())
        assert abs(prob_sum - 1.0) < 0.01
