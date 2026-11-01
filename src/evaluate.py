"""Evaluation and metrics interface for phishing email detection models.

Exposes core functions from src.evaluation:
- create_reproducible_split
- calculate_classification_metrics
- build_classification_pipeline
- audit_near_duplicates
"""

from src.evaluation import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    audit_near_duplicates,
    build_classification_pipeline,
    calculate_classification_metrics,
    create_reproducible_split,
)

__all__ = [
    "create_reproducible_split",
    "calculate_classification_metrics",
    "build_classification_pipeline",
    "audit_near_duplicates",
    "POSITIVE_LABEL",
    "NEGATIVE_LABEL",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_TEST_SIZE",
]

if __name__ == "__main__":
    import os
    import pandas as pd

    processed_file = os.path.join("data", "processed", "cleaned_emails.csv")
    if os.path.exists(processed_file):
        print(f"Loading cleaned dataset: {processed_file}")
        X_train, X_test, y_train, y_test, meta = create_reproducible_split(processed_file)
        print("Reproducible split verified successfully.")
        print("Running near-duplicate audit sample (200 test samples)...")
        near_dup_info = audit_near_duplicates(X_train, X_test, sample_size=200)
        print(
            f"Near-duplicate matches (Jaccard >= 0.85): "
            f"{near_dup_info['near_duplicate_matches']} / {near_dup_info['sample_size_audited']} "
            f"({near_dup_info['estimated_near_duplicate_pct']}%)"
        )
    else:
        print(f"Processed file not found at '{processed_file}'. Please run preprocessing first.")
