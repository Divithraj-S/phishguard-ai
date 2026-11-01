"""Unit tests for reproducible data-splitting, pipeline leakage prevention, and metrics.

Tests run offline on isolated DataFrames without external network calls.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.evaluation import (
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    audit_near_duplicates,
    build_classification_pipeline,
    calculate_classification_metrics,
    create_reproducible_split,
)


def test_reproducible_splitting_consistency():
    """Verify that using the same random_state produces identical train/test splits."""
    df = pd.DataFrame({
        "email_text": [f"Sample email text number {i}" for i in range(50)],
        "label": [0] * 30 + [1] * 20,
    })

    X_train1, X_test1, y_train1, y_test1, meta1 = create_reproducible_split(
        df, test_size=0.2, random_state=42
    )
    X_train2, X_test2, y_train2, y_test2, meta2 = create_reproducible_split(
        df, test_size=0.2, random_state=42
    )

    pd.testing.assert_series_equal(X_train1, X_train2)
    pd.testing.assert_series_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)
    assert meta1 == meta2


def test_stratified_class_proportions():
    """Verify that stratified splitting maintains matching class ratios in train and test sets."""
    df = pd.DataFrame({
        "email_text": [f"Email text {i}" for i in range(100)],
        "label": [0] * 70 + [1] * 30,  # 70% Safe (0), 30% Phishing (1)
    })

    X_train, X_test, y_train, y_test, meta = create_reproducible_split(
        df, test_size=0.2, random_state=42
    )

    train_pct = y_train.mean()
    test_pct = y_test.mean()

    # Both train and test should have ~30% positive class
    assert abs(train_pct - 0.30) < 0.05
    assert abs(test_pct - 0.30) < 0.05
    assert meta["train_phish_percent"] == 30.0
    assert meta["test_phish_percent"] == 30.0


def test_no_exact_duplicate_texts_across_train_and_test():
    """Verify that zero identical text strings overlap between train and test partitions."""
    df = pd.DataFrame({
        "email_text": [f"Unique message body {i}" for i in range(60)],
        "label": [0] * 40 + [1] * 20,
    })

    X_train, X_test, y_train, y_test, meta = create_reproducible_split(
        df, test_size=0.25, random_state=42
    )

    train_set = set(X_train.str.strip())
    test_set = set(X_test.str.strip())
    assert len(train_set.intersection(test_set)) == 0
    assert meta["exact_text_overlap_count"] == 0


def test_tfidf_pipeline_fits_only_on_training_data():
    """Verify that TF-IDF vocabulary is learned exclusively from training texts.

    Test texts containing unique, unseen words must NOT appear in the fitted vocabulary.
    """
    train_texts = pd.Series([
        "Alpha beta gamma company update",
        "Delta epsilon internal team notes",
    ])
    train_labels = pd.Series([0, 1])

    test_texts = pd.Series([
        "Zeta theta unfamiliar terminology",  # 'zeta' and 'theta' are test-only words
    ])

    # Build pipeline with dummy classifier (zero ML training needed)
    pipeline = build_classification_pipeline(DummyClassifier(strategy="most_frequent"), min_df=1)
    pipeline.fit(train_texts, train_labels)

    tfidf_vocab = pipeline.named_steps["tfidf"].vocabulary_

    # Training words MUST be in vocabulary
    assert "alpha" in tfidf_vocab
    assert "beta" in tfidf_vocab

    # Test-only words MUST NOT be in vocabulary
    assert "zeta" not in tfidf_vocab
    assert "theta" not in tfidf_vocab


def test_correct_false_positive_rate_calculation():
    """Verify FPR = FP / (FP + TN) calculation with known ground truth and predictions."""
    # 8 Safe emails (0), 4 Phishing emails (1)
    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
    # Predictions: 6 TN, 2 FP (safe flagged as phish), 1 FN, 3 TP
    y_pred = [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1]

    metrics = calculate_classification_metrics(y_true, y_pred)

    assert metrics["true_negatives"] == 6
    assert metrics["false_positives"] == 2
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 3

    # FPR = FP / (FP + TN) = 2 / (2 + 6) = 2 / 8 = 0.25
    assert metrics["false_positive_rate"] == 0.25
    # Precision = TP / (TP + FP) = 3 / (3 + 2) = 0.60
    assert metrics["precision"] == 0.60
    # Recall = TP / (TP + FN) = 3 / (3 + 1) = 0.75
    assert metrics["recall"] == 0.75
    assert metrics["zero_denominator_fpr"] is False


def test_metrics_zero_denominator_handling():
    """Verify safe handling when classes or predictions are absent."""
    # Edge Case 1: Evaluation set has NO true negatives (all are positive)
    y_true_all_pos = [1, 1, 1, 1]
    y_pred_all_pos = [1, 1, 0, 0]
    metrics1 = calculate_classification_metrics(y_true_all_pos, y_pred_all_pos)
    assert metrics1["false_positive_rate"] == 0.0
    assert metrics1["zero_denominator_fpr"] is True

    # Edge Case 2: Model predicts 0 positives (precision denominator is 0)
    y_true_mixed = [0, 1, 0, 1]
    y_pred_all_neg = [0, 0, 0, 0]
    metrics2 = calculate_classification_metrics(y_true_mixed, y_pred_all_neg)
    assert metrics2["precision"] == 0.0
    assert metrics2["recall"] == 0.0


def test_invalid_and_empty_inputs_raise_errors(tmp_path):
    """Verify that invalid inputs, empty sets, or missing columns raise errors."""
    # 1. Empty DataFrame
    empty_df = pd.DataFrame(columns=["email_text", "label"])
    with pytest.raises(ValueError, match="Cannot split an empty dataset"):
        create_reproducible_split(empty_df)

    # 2. Missing columns
    bad_df = pd.DataFrame({"random_col": [1, 2, 3]})
    with pytest.raises(ValueError, match="Schema validation failed! Missing required column"):
        create_reproducible_split(bad_df)

    # 3. Only 1 class present
    single_class_df = pd.DataFrame({
        "email_text": ["A", "B", "C"],
        "label": [0, 0, 0],
    })
    with pytest.raises(ValueError, match="requires at least 2 distinct classes"):
        create_reproducible_split(single_class_df)

    # 4. Empty arrays in metrics
    with pytest.raises(ValueError, match="Cannot calculate metrics on empty input arrays"):
        calculate_classification_metrics([], [])


def test_audit_near_duplicates_sample():
    """Verify near-duplicate audit execution on small synthetic collections."""
    train_texts = [
        "Dear customer, your bank account access has been restricted immediately.",
        "Team weekly review meeting scheduled on Thursday morning at ten.",
        "Monthly company newsletter with updates from executive leadership.",
    ]
    test_texts = [
        # Near duplicate of train text 1 (only 1 word altered)
        "Dear customer, your bank account access has been restricted today.",
        # Completely distinct text
        "Random completely different sentence about cooking recipes and food.",
    ]

    audit_result = audit_near_duplicates(
        train_texts, test_texts, sample_size=2, similarity_threshold=0.80
    )

    assert audit_result["sample_size_audited"] == 2
    assert audit_result["near_duplicate_matches"] >= 1
    assert audit_result["estimated_near_duplicate_pct"] >= 50.0
