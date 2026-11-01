"""Unit and integration tests for decision threshold tuning module.

Tests cover:
1. Stratified validation splitting without text leakage.
2. Metrics computation across varying decision thresholds.
3. Monotonic trade-off properties (FPR decrease vs. FN increase as threshold rises).
4. Boundary condition validation (0.0, 1.0, out-of-range, empty inputs).
5. Optimal threshold selection criteria (FPR minimization subject to Recall constraint).
6. End-to-end execution on isolated toy datasets with JSON artifact creation.
"""

import json
import os
import numpy as np
import pandas as pd
import pytest

from src.tune_threshold import (
    DEFAULT_THRESHOLDS,
    create_validation_split,
    evaluate_predictions_at_thresholds,
    run_threshold_tuning_experiment,
    select_optimal_threshold,
)


@pytest.fixture
def synthetic_toy_csv(tmp_path):
    """Create a minimal synthetic cleaned dataset CSV for fast, isolated test execution."""
    # 50 samples: 30 Safe (0), 20 Phishing (1)
    data = {
        "email_text": [
            "Weekly payroll department update and benefits summary",
            "Urgent: your account has been temporarily locked click here",
            "Quarterly strategic planning meeting scheduled for tomorrow",
            "Security notice: confirm your corporate credentials immediately",
            "Cafeteria menu update for the upcoming week",
        ] * 10,
        "label": [0, 1, 0, 1, 0] * 10,
    }
    csv_file = tmp_path / "toy_clean_emails.csv"
    pd.DataFrame(data).to_csv(csv_file, index=False)
    return str(csv_file)


def test_create_validation_split_stratification_and_isolation():
    """Verify that validation split maintains class balance and has zero text overlap."""
    X = pd.Series([f"Unique message number {i}" for i in range(100)])
    y = pd.Series([0] * 65 + [1] * 35)  # 65% Safe, 35% Phishing

    X_sub, X_val, y_sub, y_val, meta = create_validation_split(
        X_train=X, y_train=y, val_size=0.20, random_state=42
    )

    # 1. Row counts check
    assert len(X_sub) == 80
    assert len(X_val) == 20
    assert len(X_sub) + len(X_val) == 100

    # 2. Stratification proportion check (both partitions should be ~35% phishing)
    assert abs(y_sub.mean() - 0.35) < 0.05
    assert abs(y_val.mean() - 0.35) < 0.05
    assert meta["val_phish_percent"] == 35.0

    # 3. Text isolation check
    sub_texts = set(X_sub.str.strip())
    val_texts = set(X_val.str.strip())
    assert len(sub_texts.intersection(val_texts)) == 0
    assert meta["exact_text_overlap_count"] == 0


def test_create_validation_split_error_handling():
    """Verify error handling on invalid inputs for create_validation_split."""
    empty_series = pd.Series([], dtype=str)
    with pytest.raises(ValueError, match="Cannot split empty"):
        create_validation_split(empty_series, empty_series)

    # Length mismatch
    with pytest.raises(ValueError, match="Length mismatch"):
        create_validation_split(pd.Series(["a", "b"]), pd.Series([1]))


def test_evaluate_predictions_at_thresholds_known_values():
    """Verify metric calculations against manually known ground truth and probabilities."""
    # 5 safe emails (0), 5 phishing emails (1)
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    # Probabilities designed with clear separation and overlaps
    y_probs = [0.05, 0.15, 0.45, 0.55, 0.75, 0.35, 0.52, 0.68, 0.82, 0.95]

    evals = evaluate_predictions_at_thresholds(y_true, y_probs, thresholds=[0.30, 0.50, 0.70])

    assert len(evals) == 3

    # At threshold 0.50:
    # Safe emails with prob >= 0.50: 0.55, 0.75 -> 2 False Positives
    # Safe emails with prob < 0.50: 0.05, 0.15, 0.45 -> 3 True Negatives
    # Phishing with prob >= 0.50: 0.52, 0.68, 0.82, 0.95 -> 4 True Positives
    # Phishing with prob < 0.50: 0.35 -> 1 False Negative
    eval_50 = next(e for e in evals if e["threshold"] == 0.50)
    m50 = eval_50["metrics"]
    assert m50["false_positives"] == 2
    assert m50["true_negatives"] == 3
    assert m50["true_positives"] == 4
    assert m50["false_negatives"] == 1
    assert m50["false_positive_rate"] == 2 / 5  # 0.40
    assert m50["recall"] == 4 / 5  # 0.80

    # At threshold 0.70 (higher threshold):
    # Safe with prob >= 0.70: 0.75 -> 1 False Positive (down from 2)
    # Phishing with prob >= 0.70: 0.82, 0.95 -> 2 True Positives
    # Phishing with prob < 0.70: 0.35, 0.52, 0.68 -> 3 False Negatives
    eval_70 = next(e for e in evals if e["threshold"] == 0.70)
    m70 = eval_70["metrics"]
    assert m70["false_positives"] == 1
    assert m70["false_negatives"] == 3
    assert m70["false_positive_rate"] == 1 / 5  # 0.20 (lower FPR!)
    assert m70["recall"] == 2 / 5  # 0.40 (lower Recall!)


def test_monotonic_trade_off_property():
    """Verify that increasing threshold monotonically reduces or preserves FP and FPR."""
    y_true = [0] * 50 + [1] * 50
    np.random.seed(42)
    y_probs = np.random.uniform(0.0, 1.0, 100)

    thresholds = [0.10, 0.25, 0.50, 0.75, 0.90]
    evals = evaluate_predictions_at_thresholds(y_true, y_probs, thresholds=thresholds)

    prev_fp = float("inf")
    prev_fn = -1

    for row in evals:
        fp = row["metrics"]["false_positives"]
        fn = row["metrics"]["false_negatives"]
        # As threshold increases, False Positives must NOT increase
        assert fp <= prev_fp
        # As threshold increases, False Negatives must NOT decrease
        assert fn >= prev_fn
        prev_fp = fp
        prev_fn = fn


def test_boundary_thresholds_zero_and_one():
    """Verify behavior at extreme boundary thresholds: 0.0 and 1.0."""
    y_true = [0, 0, 1, 1]
    y_probs = [0.2, 0.4, 0.6, 0.8]

    # At 0.0 threshold: everything is classified as Positive (Phishing)
    eval_0 = evaluate_predictions_at_thresholds(y_true, y_probs, thresholds=[0.0])[0]
    assert eval_0["metrics"]["true_positives"] == 2
    assert eval_0["metrics"]["false_positives"] == 2
    assert eval_0["metrics"]["true_negatives"] == 0
    assert eval_0["metrics"]["false_negatives"] == 0
    assert eval_0["metrics"]["recall"] == 1.0
    assert eval_0["metrics"]["false_positive_rate"] == 1.0

    # At 1.0 threshold: nothing is classified as Positive (since none are >= 1.0)
    eval_1 = evaluate_predictions_at_thresholds(y_true, y_probs, thresholds=[1.0])[0]
    assert eval_1["metrics"]["true_positives"] == 0
    assert eval_1["metrics"]["false_positives"] == 0
    assert eval_1["metrics"]["true_negatives"] == 2
    assert eval_1["metrics"]["false_negatives"] == 2
    assert eval_1["metrics"]["recall"] == 0.0
    assert eval_1["metrics"]["false_positive_rate"] == 0.0


def test_invalid_thresholds_raise_errors():
    """Verify that thresholds outside [0.0, 1.0] raise ValueError."""
    y_true = [0, 1]
    y_probs = [0.2, 0.8]

    with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
        evaluate_predictions_at_thresholds(y_true, y_probs, thresholds=[-0.1])

    with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
        evaluate_predictions_at_thresholds(y_true, y_probs, thresholds=[1.05])

    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_predictions_at_thresholds([0, 1], [0.5])


def test_select_optimal_threshold_decision_logic():
    """Verify that optimal threshold selection respects the min_recall constraint."""
    # Synthetic evaluations table
    mock_evals = [
        {
            "threshold": 0.40,
            "metrics": {"recall": 0.98, "false_positive_rate": 0.03, "f1_score": 0.94},
        },
        {
            "threshold": 0.50,
            "metrics": {"recall": 0.95, "false_positive_rate": 0.02, "f1_score": 0.96},
        },
        {
            "threshold": 0.60,
            "metrics": {"recall": 0.92, "false_positive_rate": 0.01, "f1_score": 0.95},
        },
        {
            "threshold": 0.70,
            "metrics": {"recall": 0.85, "false_positive_rate": 0.005, "f1_score": 0.91},
        },
    ]

    # With min_recall = 0.90:
    # Thresholds 0.40, 0.50, 0.60 are eligible.
    # Among them, 0.60 has the lowest FPR (0.01) while having Recall 0.92 >= 0.90.
    choice = select_optimal_threshold(mock_evals, min_recall=0.90)
    assert choice["selected_threshold"] == 0.60
    assert choice["metrics"]["false_positive_rate"] == 0.01
    assert choice["metrics"]["recall"] == 0.92

    # With higher min_recall = 0.96:
    # Only 0.40 is eligible (recall 0.98 >= 0.96)
    choice_strict = select_optimal_threshold(mock_evals, min_recall=0.96)
    assert choice_strict["selected_threshold"] == 0.40

    # Fallback case: if min_recall = 0.99 (none achieve it), fallback picks highest recall (0.40 with 0.98)
    choice_fallback = select_optimal_threshold(mock_evals, min_recall=0.99)
    assert choice_fallback["selected_threshold"] == 0.40
    assert "Fallback" in choice_fallback["selection_reason"]


def test_run_threshold_tuning_experiment_e2e(synthetic_toy_csv, tmp_path):
    """Verify full threshold tuning execution on an isolated synthetic dataset."""
    models_dir = tmp_path / "test_models"
    results = run_threshold_tuning_experiment(
        data_path=synthetic_toy_csv,
        models_dir=str(models_dir),
        results_filename="test_tuning.json",
        thresholds=[0.30, 0.50, 0.70],
        random_state=42,
        test_size=0.20,
        val_size=0.25,
        min_recall=0.80,
    )

    # 1. Verify structure
    assert "metadata" in results
    assert "validation_evaluations" in results
    assert "selected_thresholds" in results
    assert "test_evaluations" in results

    # 2. Check models present
    for model_name in ["Logistic_Regression", "Random_Forest"]:
        assert model_name in results["validation_evaluations"]
        assert model_name in results["selected_thresholds"]
        assert model_name in results["test_evaluations"]

        test_data = results["test_evaluations"][model_name]
        assert "baseline_threshold_0_50" in test_data
        assert "selected_optimal_threshold" in test_data

    # 3. Verify JSON artifact exists
    json_path = models_dir / "test_tuning.json"
    assert os.path.exists(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    assert saved_data["metadata"]["random_state"] == 42
