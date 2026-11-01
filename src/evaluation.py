"""Reproducible data-splitting, pipeline building, and evaluation metrics module.

Provides:
- Stratified 80/20 train/test splitting with fixed random seeds.
- Verification of zero exact duplicate leakage across partitions.
- Empirical near-duplicate audit sampling using Jaccard word similarity.
- Construction of scikit-learn Pipelines ensuring TF-IDF vectorization fits strictly on training data.
- Safe calculation of classification metrics:
  Accuracy, Precision, Recall, F1, Confusion Matrix, and False Positive Rate (FPR).
- Safe zero-denominator handling for edge-case evaluation partitions.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.20
DEFAULT_TEXT_COL = "email_text"
DEFAULT_LABEL_COL = "label"

# Positive class definition: Phishing Email is the positive class of interest
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0


def create_reproducible_split(
    data_input: Union[str, pd.DataFrame],
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = DEFAULT_RANDOM_STATE,
    text_col: str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    """Create a stratified, reproducible train/test partition without data leakage.

    Args:
        data_input: Path to cleaned CSV or existing DataFrame.
        test_size: Proportion of the dataset allocated to the test set (default 0.20).
        random_state: Fixed random seed for complete reproducibility.
        text_col: Name of column containing email text.
        label_col: Name of column containing binary labels (0 or 1).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, metadata_dict).

    Raises:
        ValueError: If input is empty, has missing columns, or has fewer than 2 classes.
        FileNotFoundError: If data_input file path does not exist.
    """
    if isinstance(data_input, str):
        if not os.path.exists(data_input):
            raise FileNotFoundError(f"Dataset file not found at: '{data_input}'")
        df = pd.read_csv(data_input)
    elif isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
    else:
        raise TypeError("data_input must be a file path string or pandas DataFrame.")

    if len(df) == 0:
        raise ValueError("Cannot split an empty dataset (0 rows).")

    missing_cols = [col for col in (text_col, label_col) if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Schema validation failed! Missing required column(s): {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Validate classes
    unique_labels = df[label_col].dropna().unique()
    if len(unique_labels) < 2:
        raise ValueError(
            f"Stratified splitting requires at least 2 distinct classes. Found: {unique_labels}"
        )

    X = df[text_col].astype(str)
    y = df[label_col].astype(int)

    # Perform stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Check for exact duplicate normalized texts across train and test sets
    train_normalized = set(X_train.str.strip())
    test_normalized = set(X_test.str.strip())
    exact_leak_overlap = len(train_normalized.intersection(test_normalized))

    if exact_leak_overlap > 0:
        logger.warning(
            "Found %d exact duplicate texts overlapping between train and test partitions!",
            exact_leak_overlap,
        )

    # Class distributions
    train_safe = int((y_train == NEGATIVE_LABEL).sum())
    train_phish = int((y_train == POSITIVE_LABEL).sum())
    test_safe = int((y_test == NEGATIVE_LABEL).sum())
    test_phish = int((y_test == POSITIVE_LABEL).sum())

    metadata = {
        "random_state": random_state,
        "test_size": test_size,
        "total_records": len(df),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "train_safe_count": train_safe,
        "train_phish_count": train_phish,
        "train_phish_percent": round((train_phish / len(X_train)) * 100, 2),
        "test_safe_count": test_safe,
        "test_phish_count": test_phish,
        "test_phish_percent": round((test_phish / len(X_test)) * 100, 2),
        "exact_text_overlap_count": exact_leak_overlap,
    }

    logger.info("================ REPRODUCIBLE SPLIT SUMMARY ================")
    logger.info("Total rows split:                 %d", metadata["total_records"])
    logger.info(
        "Training set:                     %d rows (Safe: %d, Phish: %d [%.2f%%])",
        metadata["train_records"],
        train_safe,
        train_phish,
        metadata["train_phish_percent"],
    )
    logger.info(
        "Testing set:                      %d rows (Safe: %d, Phish: %d [%.2f%%])",
        metadata["test_records"],
        test_safe,
        test_phish,
        metadata["test_phish_percent"],
    )
    logger.info("Exact normalized text overlap:    %d", metadata["exact_text_overlap_count"])
    logger.info("============================================================")

    return X_train, X_test, y_train, y_test, metadata


def audit_near_duplicates(
    train_texts: Union[pd.Series, list],
    test_texts: Union[pd.Series, list],
    sample_size: int = 200,
    similarity_threshold: float = 0.85,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Dict[str, Any]:
    """Audit near-duplicate leakage between train and test sets using Jaccard word similarity.

    Uses length-binned comparison on a representative sample of test emails to evaluate
    approximate template overlap without requiring full quadratic pair computations.

    Args:
        train_texts: Collection of training email texts.
        test_texts: Collection of test email texts.
        sample_size: Number of test emails to sample for the audit.
        similarity_threshold: Word-set Jaccard threshold (default 0.85).
        random_state: Random seed for sampling.

    Returns:
        Audit summary dictionary with sample count, match count, and estimated percentage.
    """
    train_series = pd.Series(train_texts).astype(str)
    test_series = pd.Series(test_texts).astype(str)

    actual_sample_size = min(sample_size, len(test_series))
    test_sample = test_series.sample(n=actual_sample_size, random_state=random_state)

    # Pre-index train texts by character length and word sets
    train_records = []
    for t in train_series:
        words = set(t.lower().split())
        if len(words) > 3:  # Only index texts with enough words for meaningful comparison
            train_records.append((len(t), words))

    near_dup_count = 0

    for _, test_text in test_sample.items():
        test_len = len(test_text)
        test_words = set(test_text.lower().split())
        if len(test_words) <= 3:
            continue

        min_len = test_len * 0.80
        max_len = test_len * 1.20
        matched = False

        for t_len, t_words in train_records:
            if min_len <= t_len <= max_len:
                intersection = len(test_words.intersection(t_words))
                union = len(test_words) + len(t_words) - intersection
                if union > 0 and (intersection / union) >= similarity_threshold:
                    matched = True
                    break

        if matched:
            near_dup_count += 1

    pct = round((near_dup_count / actual_sample_size) * 100, 2) if actual_sample_size > 0 else 0.0

    return {
        "sample_size_audited": actual_sample_size,
        "similarity_threshold": similarity_threshold,
        "near_duplicate_matches": near_dup_count,
        "estimated_near_duplicate_pct": pct,
    }


def calculate_classification_metrics(
    y_true: Union[pd.Series, np.ndarray, list],
    y_pred: Union[pd.Series, np.ndarray, list],
    positive_label: int = POSITIVE_LABEL,
    negative_label: int = NEGATIVE_LABEL,
) -> Dict[str, Any]:
    """Calculate comprehensive classification metrics including False Positive Rate (FPR).

    Safely handles zero-division edge cases (e.g. absent classes, zero predicted positives).

    Positive class: 1 (Phishing Email)
    Negative class: 0 (Safe Email)

    Confusion Matrix Layout:
      [[TN, FP],
       [FN, TP]]

    False Positive Rate:
      FPR = FP / (FP + TN)

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        positive_label: Label integer representing the positive class (default 1).
        negative_label: Label integer representing the negative class (default 0).

    Returns:
        Dictionary of formatted evaluation metrics.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} entries, "
            f"y_pred has {len(y_pred_arr)} entries."
        )

    if len(y_true_arr) == 0:
        raise ValueError("Cannot calculate metrics on empty input arrays.")

    # 1. Standard Metrics with safe zero division
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, pos_label=positive_label, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, pos_label=positive_label, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, pos_label=positive_label, zero_division=0))

    # 2. Confusion Matrix explicitly indexed with [negative_label, positive_label]
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[negative_label, positive_label])
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    # 3. False Positive Rate: FP / (FP + TN)
    actual_negatives = fp + tn
    if actual_negatives > 0:
        fpr = float(fp / actual_negatives)
        zero_denominator_fpr = False
    else:
        fpr = 0.0
        zero_denominator_fpr = True
        logger.warning(
            "True negative instances (Safe Emails) are absent from evaluation set. "
            "False Positive Rate denominator is zero; set to 0.0."
        )

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "confusion_matrix": cm.tolist(),
        "positive_class": f"{positive_label} (Phishing Email)",
        "negative_class": f"{negative_label} (Safe Email)",
        "zero_denominator_fpr": zero_denominator_fpr,
    }


def build_classification_pipeline(
    classifier: Any,
    max_features: int = 10000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> Pipeline:
    """Construct an end-to-end scikit-learn Pipeline with TF-IDF vectorization.

    Enforces data leakage prevention: when pipeline.fit(X_train, y_train) is called,
    the TF-IDF vectorizer learns vocabulary strictly from X_train.
    During pipeline.predict(X_test), X_test is transformed using training vocabulary.

    Args:
        classifier: Initialized scikit-learn classifier (e.g. LogisticRegression).
        max_features: Maximum number of TF-IDF vocabulary features.
        ngram_range: Lower and upper boundary of n-gram values (default: unigrams and bigrams).
        min_df: Minimum document frequency threshold.
        max_df: Maximum document frequency threshold to ignore corpus-wide stopwords.

    Returns:
        Unfitted sklearn.pipeline.Pipeline object.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
    )
    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("classifier", classifier),
    ])
    return pipeline
