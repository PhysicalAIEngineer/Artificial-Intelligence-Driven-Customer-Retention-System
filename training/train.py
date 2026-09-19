from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from src.data import load_dataframe, prepare_xy
from src.evaluation import (
    classification_metrics,
    optimize_threshold,
)
from src.model import build_model


def main():
    parser = argparse.ArgumentParser(
        description="Train the production churn model"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", default=None)
    parser.add_argument("--out", default="artifacts")
    parser.add_argument(
        "--min-recall",
        type=float,
        default=float(
            os.getenv(
                "MIN_RECALL_TARGET",
                "0.80",
            )
        ),
    )
    args = parser.parse_args()

    dataframe = load_dataframe(args.data)
    features, target, target_column = prepare_xy(
        dataframe,
        args.target,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=42,
    )

    model = build_model(X_train)

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000",
        )
    )
    mlflow.set_experiment(
        os.getenv(
            "MLFLOW_EXPERIMENT",
            "customer-retention",
        )
    )

    with mlflow.start_run() as run:
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        default_metrics = classification_metrics(
            y_test,
            probabilities,
            0.50,
        )
        optimized = optimize_threshold(
            y_test,
            probabilities,
            args.min_recall,
        )

        mlflow.log_params(
            {
                "model": "logistic_regression",
                "class_weight": "balanced",
                "random_state": 42,
                "target_column": target_column,
                "min_recall_target": args.min_recall,
                "train_rows": len(X_train),
                "test_rows": len(X_test),
            }
        )

        mlflow.log_metrics(
            {
                f"default_{key}": value
                for key, value
                in default_metrics.items()
            }
        )
        mlflow.log_metrics(
            {
                f"optimized_{key}": value
                for key, value
                in optimized["metrics"].items()
            }
        )
        mlflow.log_param(
            "optimized_threshold",
            optimized["threshold"],
        )

        register_name = os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME"
        )
        if register_name:
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=register_name,
            )
        else:
            mlflow.sklearn.log_model(
                model,
                "model",
            )

        output = Path(args.out)
        output.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            output / "churn_model.joblib",
        )

        (output / "threshold.json").write_text(
            json.dumps(
                {
                    "threshold": optimized["threshold"],
                    "min_recall_target": args.min_recall,
                },
                indent=2,
            )
        )

        (output / "metadata.json").write_text(
            json.dumps(
                {
                    "run_id": run.info.run_id,
                    "target_column": target_column,
                    "feature_columns": list(
                        features.columns
                    ),
                    "metrics": optimized["metrics"],
                    "default_metrics": default_metrics,
                },
                indent=2,
            )
        )

        print(json.dumps(optimized, indent=2))


if __name__ == "__main__":
    main()
