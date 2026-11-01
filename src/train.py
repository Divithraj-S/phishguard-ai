"""Model training and evaluation module for phishing email detection.

Trains baseline models:
1. Logistic Regression with TF-IDF features.
2. Random Forest with TF-IDF features.

Strict Data Leakage Controls:
- Stratified 80/20 train/test split (random_state=42).
- Vectorization and model fitting encapsulated in scikit-learn Pipelines.
- Held-out test set is evaluated strictly once with default decision threshold (0.50).
- Zero threshold tuning or feature selection on test data.
- Outputs machine-readable evaluation results and serialized pipeline artifacts.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    build_classification_pipeline,
    calculate_classification_metrics,
    create_reproducible_split,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = os.path.join("data", "processed", "cleaned_emails.csv")
DEFAULT_MODELS_DIR = "models"
DEFAULT_RESULTS_FILE = "evaluation_results.json"

# Documented Hyperparameters suitable for beginner CPU workstations
DEFAULT_TFIDF_PARAMS = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
}

LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": DEFAULT_RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": None,
    "random_state": DEFAULT_RANDOM_STATE,
    "n_jobs": -1,  # Utilize multi-core CPU parallelism
}


def train_and_evaluate_baseline_models(
    data_path: str = DEFAULT_DATA_PATH,
    models_dir: str = DEFAULT_MODELS_DIR,
    results_filename: str = DEFAULT_RESULTS_FILE,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
) -> Dict[str, Any]:
    """Train and evaluate baseline Logistic Regression and Random Forest models.

    Args:
        data_path: Path to cleaned dataset CSV.
        models_dir: Directory where serialized model artifacts will be saved.
        results_filename: File name for saving evaluation results JSON.
        random_state: Random seed for reproducibility.
        test_size: Proportion for test partition.

    Returns:
        Structured dictionary containing metrics, timings, hyperparameters, and artifact paths.
    """
    os.makedirs(models_dir, exist_ok=True)

    # 1. Reproducible Split (Zero exact text overlap, strictly stratified)
    X_train, X_test, y_train, y_test, split_meta = create_reproducible_split(
        data_input=data_path,
        test_size=test_size,
        random_state=random_state,
    )

    models_config = {
        "Logistic_Regression": {
            "classifier": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
            "hyperparameters": LOGISTIC_REGRESSION_PARAMS,
            "filename": "logistic_regression_pipeline.joblib",
        },
        "Random_Forest": {
            "classifier": RandomForestClassifier(**RANDOM_FOREST_PARAMS),
            "hyperparameters": RANDOM_FOREST_PARAMS,
            "filename": "random_forest_pipeline.joblib",
        },
    }

    results: Dict[str, Any] = {
        "split_metadata": split_meta,
        "tfidf_hyperparameters": {
            "max_features": DEFAULT_TFIDF_PARAMS["max_features"],
            "ngram_range": list(DEFAULT_TFIDF_PARAMS["ngram_range"]),
            "min_df": DEFAULT_TFIDF_PARAMS["min_df"],
            "max_df": DEFAULT_TFIDF_PARAMS["max_df"],
            "sublinear_tf": True,
        },
        "models": {},
    }

    for model_name, cfg in models_config.items():
        logger.info("------------------------------------------------------------")
        logger.info("Training pipeline for: %s ...", model_name)

        # Build pipeline (TF-IDF + Classifier)
        pipeline = build_classification_pipeline(
            classifier=cfg["classifier"],
            max_features=DEFAULT_TFIDF_PARAMS["max_features"],
            ngram_range=DEFAULT_TFIDF_PARAMS["ngram_range"],
            min_df=DEFAULT_TFIDF_PARAMS["min_df"],
            max_df=DEFAULT_TFIDF_PARAMS["max_df"],
        )

        # Fit strictly on training set (measuring training time)
        train_start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        train_duration = time.perf_counter() - train_start

        logger.info("Evaluating on held-out test set (%d samples)...", len(X_test))
        # Evaluate strictly on test set (measuring inference time)
        eval_start = time.perf_counter()
        y_pred = pipeline.predict(X_test)
        eval_duration = time.perf_counter() - eval_start

        # Compute all evaluation metrics
        metrics = calculate_classification_metrics(y_true=y_test, y_pred=y_pred)

        # Save pipeline artifact
        artifact_path = os.path.join(models_dir, cfg["filename"])
        joblib.dump(pipeline, artifact_path)
        file_size_mb = os.path.getsize(artifact_path) / (1024 * 1024)

        results["models"][model_name] = {
            "metrics": metrics,
            "training_time_seconds": round(train_duration, 3),
            "evaluation_time_seconds": round(eval_duration, 3),
            "hyperparameters": cfg["hyperparameters"],
            "artifact_path": artifact_path,
            "artifact_size_mb": round(file_size_mb, 2),
        }

        logger.info(
            "%s Results: Acc=%.4f, Prec=%.4f, Rec=%.4f, F1=%.4f, FPR=%.4f (Train: %.2fs, Eval: %.3fs)",
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1_score"],
            metrics["false_positive_rate"],
            train_duration,
            eval_duration,
        )

    # 2. Save Machine-Readable Results
    results_path = os.path.join(models_dir, results_filename)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved machine-readable results to: '%s'", results_path)

    # 3. Print Comparison Report
    print_comparison_report(results)

    return results


def print_comparison_report(results: Dict[str, Any]) -> None:
    """Print a clean, formatted comparison report of model metrics."""
    split_meta = results.get("split_metadata", {})
    models = results.get("models", {})

    print("\n" + "=" * 75)
    print("           BASELINE MODELS EVALUATION REPORT (HELD-OUT TEST SET)")
    print("=" * 75)
    print(
        f"Test Set Size:    {split_meta.get('test_records', 0):,} emails "
        f"(Safe: {split_meta.get('test_safe_count', 0):,}, "
        f"Phishing: {split_meta.get('test_phish_count', 0):,})"
    )
    print(f"Random Seed:      {split_meta.get('random_state')}")
    print(f"Decision Rule:    Standard Default Threshold (p >= 0.50)")
    print("-" * 75)
    print(
        f"{'Metric':<25} | "
        + " | ".join(f"{name.replace('_', ' '):<20}" for name in models.keys())
    )
    print("-" * 75)

    metric_keys = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1-Score", "f1_score"),
        ("False Positive Rate", "false_positive_rate"),
        ("Training Time (s)", "training_time_seconds"),
        ("Evaluation Time (s)", "evaluation_time_seconds"),
    ]

    for label, key in metric_keys:
        row_vals = []
        for model_name, data in models.items():
            if key in data["metrics"]:
                val = data["metrics"][key]
                row_vals.append(f"{val:.4f}")
            elif key in data:
                val = data[key]
                row_vals.append(f"{val:.3f} s")
            else:
                row_vals.append("N/A")
        print(f"{label:<25} | " + " | ".join(f"{v:<20}" for v in row_vals))

    print("-" * 75)
    print("CONFUSION MATRICES: [[TN, FP], [FN, TP]]")
    for name, data in models.items():
        cm = data["metrics"]["confusion_matrix"]
        tn = data["metrics"]["true_negatives"]
        fp = data["metrics"]["false_positives"]
        fn = data["metrics"]["false_negatives"]
        tp = data["metrics"]["true_positives"]
        fpr = data["metrics"]["false_positive_rate"]
        print(f"\n  * {name}:")
        print(f"    [[TN={tn:<5}, FP={fp:<5}],")
        print(f"     [FN={fn:<5}, TP={tp:<5}]]")
        print(
            f"    -> False Positive Rate: {fpr:.4f} "
            f"({fp} legitimate emails misclassified out of {fp + tn} safe emails)"
        )

    print("=" * 75 + "\n")


if __name__ == "__main__":
    import sys

    data_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    res = train_and_evaluate_baseline_models(data_path=data_file)
