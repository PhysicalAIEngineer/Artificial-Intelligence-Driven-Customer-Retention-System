import numpy as np

from src.evaluation import (
    classification_metrics,
    optimize_threshold,
)


def test_metrics():
    y_true = np.array([0, 1, 1, 0])
    probabilities = np.array([0.1, 0.9, 0.8, 0.2])

    metrics = classification_metrics(
        y_true,
        probabilities,
        0.5,
    )

    assert metrics["recall"] == 1.0


def test_threshold():
    y_true = np.array([0, 0, 1, 1, 1])
    probabilities = np.array([0.1, 0.2, 0.4, 0.8, 0.9])

    result = optimize_threshold(
        y_true,
        probabilities,
        0.8,
    )

    assert result["metrics"]["recall"] >= 0.8
