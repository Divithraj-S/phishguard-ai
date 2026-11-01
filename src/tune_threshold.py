"""Decision threshold tuning and evaluation module for phishing email detection.

This module addresses the core project goal: Reducing False Positives in Phishing Email Detection.

By default, classification models use a decision threshold of p >= 0.50.
In real-world email filtering:
- False Positives (Safe email flagged as Phishing) block vital business communications,
  inconvenience users, and erode trust in security tools.
- False Negatives (Phishing email flagged as Safe) allow potential threats into inboxes.

By tuning the probability threshold on a strictly held-out VALIDATION partition
(carved out from the training data only), we can systematically evaluate the trade-offs
between False Positive Rate (FPR) and Recall (threat detection coverage).

Crucial Data Leakage Safeguards:
- Threshold search is performed EXCLUSIVELY on a validation set (X_val, y_val)
  split from the training partition.
- The held-out test set (3,504 emails) is kept completely untouched during the search.
- The selected threshold is evaluated strictly ONCE on the held-out test set at the end.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluation import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    build_classification_pipeline,
    calculate_classification_metrics,
    create_reproducible_split,
)
from src.train import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODELS_DIR,
    DEFAULT_TFIDF_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: List[float] = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
DEFAULT_TUNING_RESULTS_FILE = "threshold_tuning_results.json"
DEFAULT_VAL_SIZE = 0.20
DEFAULT_MIN_RECALL = 0.90


def create_validation_split(
    X_train: pd.Series,
    y_train: pd.Series,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    """Carve a stratified validation set out of the training set.

    Ensures zero contamination of the final held-out test set.

    Args:
        X_train: Full training set features (email texts).
        y_train: Full training set labels (0 or 1).
        val_size: Proportion of training data to reserve for validation (default 0.20).
        random_state: Fixed random seed for complete reproducibility.

    Returns:
        Tuple of (X_sub_train, X_val, y_sub_train, y_val, split_metadata).
    """
    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Cannot split empty training data.")

    if len(X_train) != len(y_train):
        raise ValueError(
            f"Length mismatch: X_train has {len(X_train)} rows, y_train has {len(y_train)} rows."
        )

    X_sub_train, X_val, y_sub_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train,
    )

    # Verify zero exact text overlap between sub-train and validation sets
    sub_train_set = set(X_sub_train.astype(str).str.strip())
    val_set = set(X_val.astype(str).str.strip())
    overlap_count = len(sub_train_set.intersection(val_set))

    meta = {
        "random_state": random_state,
        "val_size": val_size,
        "total_train_records": len(X_train),
        "sub_train_records": len(X_sub_train),
        "val_records": len(X_val),
        "sub_train_safe": int((y_sub_train == NEGATIVE_LABEL).sum()),
        "sub_train_phish": int((y_sub_train == POSITIVE_LABEL).sum()),
        "val_safe": int((y_val == NEGATIVE_LABEL).sum()),
        "val_phish": int((y_val == POSITIVE_LABEL).sum()),
        "val_phish_percent": round((int((y_val == POSITIVE_LABEL).sum()) / len(X_val)) * 100, 2),
        "exact_text_overlap_count": overlap_count,
    }

    logger.info(
        "Validation Split: %d sub-train, %d validation (Safe: %d, Phish: %d [%.2f%%])",
        meta["sub_train_records"],
        meta["val_records"],
        meta["val_safe"],
        meta["val_phish"],
        meta["val_phish_percent"],
    )

    return X_sub_train, X_val, y_sub_train, y_val, meta


def evaluate_predictions_at_thresholds(
    y_true: Union[pd.Series, np.ndarray, list],
    y_probs: Union[pd.Series, np.ndarray, list],
    thresholds: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Evaluate classification metrics across multiple probability thresholds.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_probs: Predicted probabilities for the positive class (Phishing, class 1).
        thresholds: List of decision thresholds (defaults to [0.20, ..., 0.90]).

    Returns:
        List of dictionaries containing threshold and corresponding classification metrics.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    y_true_arr = np.asarray(y_true, dtype=int)
    y_probs_arr = np.asarray(y_probs, dtype=float)

    if len(y_true_arr) != len(y_probs_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} entries, "
            f"y_probs has {len(y_probs_arr)} entries."
        )

    for th in thresholds:
        if not (0.0 <= th <= 1.0):
            raise ValueError(f"Threshold must be between 0.0 and 1.0. Got: {th}")

    results = []
    for th in thresholds:
        y_pred = (y_probs_arr >= th).astype(int)
        metrics = calculate_classification_metrics(y_true=y_true_arr, y_pred=y_pred)
        results.append({
            "threshold": round(float(th), 2),
            "metrics": metrics,
        })

    return results


def select_optimal_threshold(
    threshold_evaluations: List[Dict[str, Any]],
    min_recall: float = DEFAULT_MIN_RECALL,
    target_metric: str = "false_positive_rate",
) -> Dict[str, Any]:
    """Select the optimal decision threshold based on validation performance.

    Primary Objective:
    Reduce False Positives (minimize False Positive Rate) while ensuring that
    threat detection Recall stays at or above the safety threshold (default >= 90%).

    If no threshold satisfies min_recall, selects the threshold with highest Recall.

    Args:
        threshold_evaluations: List of evaluations from evaluate_predictions_at_thresholds.
        min_recall: Minimum required Recall for positive class (default 0.90).
        target_metric: Metric to optimize (default 'false_positive_rate' minimized).

    Returns:
        Dictionary with selected threshold, reason, and corresponding metrics.
    """
    if not threshold_evaluations:
        raise ValueError("Cannot select from empty threshold evaluations.")

    # Filter candidates meeting the minimum recall requirement
    eligible = [
        item for item in threshold_evaluations
        if item["metrics"]["recall"] >= min_recall
    ]

    if eligible:
        # Sort eligible thresholds by lowest FPR.
        # If FPR is tied, prefer higher F1-score, then higher Recall.
        eligible.sort(
            key=lambda x: (
                x["metrics"]["false_positive_rate"],
                -x["metrics"]["f1_score"],
                -x["metrics"]["recall"],
            )
        )
        selected = eligible[0]
        reason = (
            f"Minimizes False Positive Rate ({selected['metrics']['false_positive_rate']:.4f}) "
            f"while maintaining Recall >= {min_recall:.0%} "
            f"(Validation Recall: {selected['metrics']['recall']:.4f})."
        )
    else:
        # Fallback if no threshold reaches min_recall: maximize recall
        logger.warning(
            "No threshold achieved the required min_recall of %.2f. "
            "Falling back to threshold with highest Recall.",
            min_recall,
        )
        threshold_evaluations_sorted = sorted(
            threshold_evaluations,
            key=lambda x: -x["metrics"]["recall"],
        )
        selected = threshold_evaluations_sorted[0]
        reason = (
            f"Fallback: Highest achievable Recall ({selected['metrics']['recall']:.4f}) "
            f"as no threshold reached min_recall={min_recall:.0%}."
        )

    return {
        "selected_threshold": selected["threshold"],
        "selection_reason": reason,
        "min_recall_constraint": min_recall,
        "metrics": selected["metrics"],
    }


def run_threshold_tuning_experiment(
    data_path: str = DEFAULT_DATA_PATH,
    models_dir: str = DEFAULT_MODELS_DIR,
    results_filename: str = DEFAULT_TUNING_RESULTS_FILE,
    thresholds: Optional[List[float]] = None,
    random_state: int = DEFAULT_RANDOM_STATE,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    min_recall: float = DEFAULT_MIN_RECALL,
) -> Dict[str, Any]:
    """Execute complete decision threshold tuning experiment with strict leakage protection.

    Workflow:
    1. Stratified 80/20 split into X_train_full (14,015) and held-out X_test (3,504).
       -> Held-out test set is SEALED and untouched during tuning.
    2. Sub-split X_train_full into sub-train (80%, 11,212) and validation (20%, 2,803).
    3. Fit candidate models on sub-train only.
    4. Predict class 1 probabilities on validation set.
    5. Evaluate all candidate thresholds [0.20, ..., 0.90] on validation set.
    6. Select the optimal threshold for each model (minimizing FPR subject to Recall >= min_recall).
    7. Load or fit the production model (trained on full X_train_full) and evaluate ONCE
       on held-out X_test at:
       - Baseline threshold (0.50)
       - Validation-selected optimal threshold
    8. Serialize results to JSON and display summary table.

    Args:
        data_path: Path to cleaned dataset CSV.
        models_dir: Directory where model artifacts and results are stored.
        results_filename: JSON filename for saving threshold tuning outputs.
        thresholds: List of candidate thresholds.
        random_state: Fixed random seed.
        test_size: Held-out test split fraction (0.20).
        val_size: Validation split fraction from training partition (0.20).
        min_recall: Safety threshold for minimum recall (0.90).

    Returns:
        Structured dictionary containing full validation and held-out test evaluations.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    os.makedirs(models_dir, exist_ok=True)

    # 1. Stratified Train/Test Split (Held-out test set sealed)
    X_train_full, X_test, y_train_full, y_test, split_meta = create_reproducible_split(
        data_input=data_path,
        test_size=test_size,
        random_state=random_state,
    )

    # 2. Carve Validation set out of training partition ONLY
    X_sub_train, X_val, y_sub_train, y_val, val_meta = create_validation_split(
        X_train=X_train_full,
        y_train=y_train_full,
        val_size=val_size,
        random_state=random_state,
    )

    experiment_results: Dict[str, Any] = {
        "metadata": {
            "random_state": random_state,
            "test_size": test_size,
            "val_size": val_size,
            "min_recall_constraint": min_recall,
            "candidate_thresholds": thresholds,
            "full_train_records": len(X_train_full),
            "sub_train_records": len(X_sub_train),
            "val_records": len(X_val),
            "held_out_test_records": len(X_test),
        },
        "validation_evaluations": {},
        "selected_thresholds": {},
        "test_evaluations": {},
    }

    model_configs = {
        "Logistic_Regression": {
            "classifier": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
            "artifact_name": "logistic_regression_pipeline.joblib",
        },
        "Random_Forest": {
            "classifier": RandomForestClassifier(**RANDOM_FOREST_PARAMS),
            "artifact_name": "random_forest_pipeline.joblib",
        },
    }

    # 3 & 4. Train on sub-train and evaluate on validation across thresholds
    for model_name, cfg in model_configs.items():
        logger.info("============================================================")
        logger.info("Tuning thresholds on Validation Set for: %s", model_name)

        # Build validation pipeline and fit on sub-train
        val_pipeline = build_classification_pipeline(
            classifier=cfg["classifier"],
            max_features=DEFAULT_TFIDF_PARAMS["max_features"],
            ngram_range=DEFAULT_TFIDF_PARAMS["ngram_range"],
            min_df=DEFAULT_TFIDF_PARAMS["min_df"],
            max_df=DEFAULT_TFIDF_PARAMS["max_df"],
        )

        logger.info("Fitting validation pipeline on %d sub-train samples...", len(X_sub_train))
        val_pipeline.fit(X_sub_train, y_sub_train)

        # Predict probabilities on validation set (class 1: Phishing)
        logger.info("Predicting probabilities on %d validation samples...", len(X_val))
        val_probs = val_pipeline.predict_proba(X_val)[:, 1]

        # Evaluate across candidate thresholds
        val_eval_list = evaluate_predictions_at_thresholds(
            y_true=y_val,
            y_probs=val_probs,
            thresholds=thresholds,
        )
        experiment_results["validation_evaluations"][model_name] = val_eval_list

        # Select optimal threshold based on validation set
        optimal_choice = select_optimal_threshold(
            threshold_evaluations=val_eval_list,
            min_recall=min_recall,
        )
        experiment_results["selected_thresholds"][model_name] = optimal_choice

        opt_th = optimal_choice["selected_threshold"]
        logger.info(
            "Selected optimal threshold for %s: %.2f (%s)",
            model_name,
            opt_th,
            optimal_choice["selection_reason"],
        )

        # 5. Evaluate on Held-Out Test Set ONCE using full-trained model
        # Try loading pre-trained artifact or fit on X_train_full
        artifact_path = os.path.join(models_dir, cfg["artifact_name"])
        if os.path.exists(artifact_path):
            logger.info("Loading full production model from: %s", artifact_path)
            full_pipeline = joblib.load(artifact_path)
        else:
            logger.info("Fitting full production model on %d train samples...", len(X_train_full))
            full_pipeline = build_classification_pipeline(
                classifier=cfg["classifier"],
                max_features=DEFAULT_TFIDF_PARAMS["max_features"],
                ngram_range=DEFAULT_TFIDF_PARAMS["ngram_range"],
                min_df=DEFAULT_TFIDF_PARAMS["min_df"],
                max_df=DEFAULT_TFIDF_PARAMS["max_df"],
            )
            full_pipeline.fit(X_train_full, y_train_full)
            joblib.dump(full_pipeline, artifact_path)

        # Predict probabilities on held-out test set
        test_probs = full_pipeline.predict_proba(X_test)[:, 1]

        # Evaluate at default 0.50 baseline
        pred_base = (test_probs >= 0.50).astype(int)
        metrics_base = calculate_classification_metrics(y_test, pred_base)

        # Evaluate at chosen optimal threshold
        pred_opt = (test_probs >= opt_th).astype(int)
        metrics_opt = calculate_classification_metrics(y_test, pred_opt)

        experiment_results["test_evaluations"][model_name] = {
            "baseline_threshold_0_50": {
                "threshold": 0.50,
                "metrics": metrics_base,
            },
            "selected_optimal_threshold": {
                "threshold": opt_th,
                "metrics": metrics_opt,
                "relative_fp_reduction_percent": round(
                    ((metrics_base["false_positives"] - metrics_opt["false_positives"])
                     / max(metrics_base["false_positives"], 1)) * 100,
                    2,
                ),
            },
        }

    # 6. Save results to JSON
    results_path = os.path.join(models_dir, results_filename)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    logger.info("Saved threshold tuning experiment results to '%s'", results_path)

    # 7. Print summary comparison report
    print_tuning_report(experiment_results)

    return experiment_results


def print_tuning_report(results: Dict[str, Any]) -> None:
    """Print readable terminal report comparing threshold tuning results."""
    meta = results.get("metadata", {})
    val_evals = results.get("validation_evaluations", {})
    selected = results.get("selected_thresholds", {})
    test_evals = results.get("test_evaluations", {})

    print("\n" + "=" * 90)
    print("             DECISION THRESHOLD TUNING & EVALUATION AUDIT REPORT")
    print("=" * 90)
    print(f"Data Partitioning:")
    print(f"  * Total Clean Dataset:        {meta.get('full_train_records', 0) + meta.get('held_out_test_records', 0):,} emails")
    print(f"  * Training Partition:         {meta.get('full_train_records', 0):,} emails (80%)")
    print(f"    - Sub-Training Set:         {meta.get('sub_train_records', 0):,} emails (used for model fitting)")
    print(f"    - Validation Set:           {meta.get('val_records', 0):,} emails (used strictly for threshold search)")
    print(f"  * Held-Out Test Set:          {meta.get('held_out_test_records', 0):,} emails (SEALED, evaluated ONCE)")
    print(f"  * Minimum Recall Constraint:  {meta.get('min_recall_constraint', 0.9):.0%}")
    print("=" * 90)

    for model_name, evals in val_evals.items():
        print(f"\n--- [1] VALIDATION SET PERFORMANCE ACROSS THRESHOLDS: {model_name} ---")
        print(
            f"{'Threshold':<10} | {'FPR':<10} | {'FP':<6} | {'Recall':<10} | {'FN':<6} | "
            f"{'Precision':<10} | {'F1-Score':<10} | {'Accuracy':<10}"
        )
        print("-" * 85)
        for row in evals:
            th = row["threshold"]
            m = row["metrics"]
            is_sel = "*" if th == selected.get(model_name, {}).get("selected_threshold") else " "
            print(
                f"{th:<8.2f} {is_sel} | {m['false_positive_rate']:<10.4f} | {m['false_positives']:<6} | "
                f"{m['recall']:<10.4f} | {m['false_negatives']:<6} | {m['precision']:<10.4f} | "
                f"{m['f1_score']:<10.4f} | {m['accuracy']:<10.4f}"
            )
        opt = selected.get(model_name, {})
        print(f"  -> Selected Threshold: {opt.get('selected_threshold')} ({opt.get('selection_reason')})")

    print("\n" + "=" * 90)
    print("      [2] HELD-OUT TEST SET EVALUATION: BASELINE (0.50) VS. SELECTED THRESHOLD")
    print("=" * 90)

    for model_name, test_data in test_evals.items():
        base = test_data["baseline_threshold_0_50"]
        opt = test_data["selected_optimal_threshold"]
        bm = base["metrics"]
        om = opt["metrics"]

        print(f"\nModel: {model_name}")
        print(f"{'Metric':<25} | {'Baseline (0.50)':<18} | {'Selected (' + str(opt['threshold']) + ')':<18} | {'Difference / Impact':<20}")
        print("-" * 85)

        fp_diff = om["false_positives"] - bm["false_positives"]
        fpr_diff = om["false_positive_rate"] - bm["false_positive_rate"]
        rec_diff = om["recall"] - bm["recall"]
        fn_diff = om["false_negatives"] - bm["false_negatives"]
        f1_diff = om["f1_score"] - bm["f1_score"]
        acc_diff = om["accuracy"] - bm["accuracy"]

        fp_impact = f"{fp_diff:+d} FPs ({opt.get('relative_fp_reduction_percent', 0):+.1f}%)"

        rows = [
            ("False Positives (FP)", f"{bm['false_positives']}", f"{om['false_positives']}", fp_impact),
            ("False Positive Rate (FPR)", f"{bm['false_positive_rate']:.4f}", f"{om['false_positive_rate']:.4f}", f"{fpr_diff:+.4f}"),
            ("False Negatives (FN)", f"{bm['false_negatives']}", f"{om['false_negatives']}", f"{fn_diff:+d} missed phish"),
            ("Recall (Sensitivity)", f"{bm['recall']:.4f}", f"{om['recall']:.4f}", f"{rec_diff:+.4f}"),
            ("Precision", f"{bm['precision']:.4f}", f"{om['precision']:.4f}", f"{om['precision'] - bm['precision']:+.4f}"),
            ("F1-Score", f"{bm['f1_score']:.4f}", f"{om['f1_score']:.4f}", f"{f1_diff:+.4f}"),
            ("Accuracy", f"{bm['accuracy']:.4f}", f"{om['accuracy']:.4f}", f"{acc_diff:+.4f}"),
        ]

        for label, b_val, o_val, diff in rows:
            print(f"{label:<25} | {b_val:<18} | {o_val:<18} | {diff:<20}")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_threshold_tuning_experiment()
