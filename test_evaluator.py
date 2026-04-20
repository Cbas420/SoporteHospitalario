"""
Tests del modulo de evaluacion.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import CLASS_NAMES
from src.evaluation.evaluator import clinical_error_analysis, compute_metrics


@pytest.fixture
def perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return y_true, y_pred


@pytest.fixture
def imperfect_predictions():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    y_pred = np.array([0, 0, 2, 1, 1, 1, 1, 0, 2, 2, 0, 2])
    return y_true, y_pred


def test_perfect_accuracy(perfect_predictions):
    y_true, y_pred = perfect_predictions
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["global"]["accuracy"] == 1.0


def test_perfect_f1(perfect_predictions):
    y_true, y_pred = perfect_predictions
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["global"]["f1_macro"] == 1.0


def test_imperfect_accuracy(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    metrics = compute_metrics(y_true, y_pred)
    assert 0 < metrics["global"]["accuracy"] < 1.0


def test_confusion_matrix_shape(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    metrics = compute_metrics(y_true, y_pred)
    cm = np.array(metrics["confusion_matrix"])
    assert cm.shape == (3, 3)


def test_confusion_matrix_sum(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    metrics = compute_metrics(y_true, y_pred)
    cm = np.array(metrics["confusion_matrix"])
    assert cm.sum() == len(y_true)


def test_per_class_metrics_present(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    metrics = compute_metrics(y_true, y_pred)
    for cls in CLASS_NAMES:
        assert cls in metrics["per_class"]
        cls_m = metrics["per_class"][cls]
        assert "precision" in cls_m
        assert "recall" in cls_m
        assert "f1_score" in cls_m
        assert "specificity" in cls_m


def test_clinical_analysis_structure(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    metrics = compute_metrics(y_true, y_pred)
    analysis = clinical_error_analysis(metrics)

    assert "covid19_false_negatives" in analysis
    assert "pneumonia_false_negatives" in analysis
    assert "recommendations" in analysis


def test_covid_fn_detection(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    metrics = compute_metrics(y_true, y_pred)
    analysis = clinical_error_analysis(metrics)
    assert analysis["covid19_false_negatives"]["count"] > 0


def test_recommendations_not_empty(imperfect_predictions):
    y_true, y_pred = imperfect_predictions
    metrics = compute_metrics(y_true, y_pred)
    analysis = clinical_error_analysis(metrics)
    assert len(analysis["recommendations"]) > 0
