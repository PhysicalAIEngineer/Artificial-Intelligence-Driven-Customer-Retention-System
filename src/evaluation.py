from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y_true,
    probabilities,
    threshold: float,
):
    predictions = (
        np.asarray(probabilities) >= threshold
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
    }


def optimize_threshold(
    y_true,
    probabilities,
    min_recall: float = 0.80,
):
    candidates = np.linspace(
        0.05,
        0.95,
        181,
    )
    best = None

    for threshold in candidates:
        metrics = classification_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        if metrics["recall"] >= min_recall:
            key = (
                metrics["f1"],
                metrics["precision"],
                float(threshold),
            )
            candidate = {
                "threshold": float(threshold),
                "metrics": metrics,
                "key": key,
            }

            if (
                best is None
                or candidate["key"] > best["key"]
            ):
                best = candidate

    if best is not None:
        best.pop("key")
        return best

    fallback = []
    for threshold in candidates:
        fallback.append(
            {
                "threshold": float(threshold),
                "metrics": classification_metrics(
                    y_true,
                    probabilities,
                    float(threshold),
                ),
            }
        )

    return max(
        fallback,
        key=lambda item: (
            item["metrics"]["recall"],
            item["metrics"]["precision"],
        ),
    )
