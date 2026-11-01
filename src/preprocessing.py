"""Data preprocessing and hygiene pipeline for email text classification.

Implements rigorous data cleaning:
- Schema and label validation
- Blank/null text removal
- Whitespace normalization
- Conflicting-label text detection and exclusion
- Deduplication of identical messages
- Configurable extreme-length text handling (truncation vs filtering)
- Removal of non-feature index columns
- Full step-by-step reporting to prevent silent data discarding
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TEXT_COL = "Email Text"
DEFAULT_LABEL_COL = "Email Type"
DEFAULT_LABEL_MAPPING = {
    "Safe Email": 0,
    "Phishing Email": 1,
}
DEFAULT_MAX_CHAR_LENGTH = 50_000  # Covers >99.8% of emails while capping 17M char outlier


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for duplicate matching without altering text semantics.

    Trims leading/trailing whitespace and collapses internal multiple spaces/tabs.

    Args:
        text: Raw input text.

    Returns:
        Clean whitespace-normalized text.
    """
    if not isinstance(text, str):
        return ""
    # Strip leading/trailing whitespace
    trimmed = text.strip()
    # Normalize internal runs of spaces, tabs, and newlines for canonical comparison
    return re.sub(r"[ \t\r\f\v]+", " ", trimmed)


def clean_and_preprocess_dataset(
    data_input: Union[str, pd.DataFrame],
    text_col: str = DEFAULT_TEXT_COL,
    label_col: str = DEFAULT_LABEL_COL,
    label_mapping: Optional[Dict[str, int]] = None,
    max_char_length: Optional[int] = DEFAULT_MAX_CHAR_LENGTH,
    long_text_strategy: str = "truncate",
    output_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute complete data preprocessing pipeline with full auditable reporting.

    Args:
        data_input: File path to raw CSV or an existing pd.DataFrame.
        text_col: Name of column containing email text.
        label_col: Name of column containing target labels.
        label_mapping: Dictionary mapping label names to binary integers.
        max_char_length: Maximum character threshold for handling extreme-length texts.
        long_text_strategy: How to handle texts > max_char_length:
                            'truncate' (default: cap text at max length),
                            'drop' (remove row entirely),
                            'none' (keep original length).
        output_path: Optional file path to save the cleaned CSV (e.g. data/processed/...).

    Returns:
        A tuple of (clean_df, audit_report_dict).

    Raises:
        ValueError: If input is empty, missing required columns, or contains unknown labels.
        FileNotFoundError: If data_input path does not exist.
    """
    if label_mapping is None:
        label_mapping = DEFAULT_LABEL_MAPPING

    # 1. Load Data
    if isinstance(data_input, str):
        if not os.path.exists(data_input):
            raise FileNotFoundError(f"Input file not found at: '{data_input}'")
        df_raw = pd.read_csv(data_input)
    elif isinstance(data_input, pd.DataFrame):
        df_raw = data_input.copy()
    else:
        raise TypeError("data_input must be a file path string or a pandas DataFrame.")

    initial_rows = len(df_raw)
    initial_cols = list(df_raw.columns)

    if initial_rows == 0:
        raise ValueError("Input dataset is empty (0 rows).")

    # 2. Validate Required Columns
    missing_cols = []
    if text_col not in df_raw.columns:
        missing_cols.append(text_col)
    if label_col not in df_raw.columns:
        missing_cols.append(label_col)

    if missing_cols:
        raise ValueError(
            f"Schema validation failed! Missing required column(s): {missing_cols}. "
            f"Columns found in input: {initial_cols}"
        )

    # 3. Validate Labels (Check for unknown categories)
    valid_labels = set(label_mapping.keys())
    present_labels = set(df_raw[label_col].dropna().unique())
    unknown_labels = present_labels - valid_labels
    if unknown_labels:
        raise ValueError(
            f"Label validation failed! Found unrecognized labels: {unknown_labels}. "
            f"Allowed labels are: {list(valid_labels)}"
        )

    # 4. Filter Missing, Null, and Whitespace-Only Email Texts or Null Labels
    null_or_blank_text = df_raw[text_col].isna() | (df_raw[text_col].astype(str).str.strip() == "")
    null_label = df_raw[label_col].isna()
    invalid_mask = null_or_blank_text | null_label
    blank_dropped_count = int(invalid_mask.sum())

    df_valid = df_raw[~invalid_mask].copy()

    if len(df_valid) == 0:
        raise ValueError("Dataset is empty after dropping missing or blank rows.")

    # 5. Normalize Whitespace for Canonical Matching
    df_valid["_norm_text"] = df_valid[text_col].astype(str).apply(normalize_whitespace)

    # 6. Detect and Exclude Conflicting Labels (Same text with multiple classes)
    label_counts_per_text = df_valid.groupby("_norm_text")[label_col].nunique()
    conflicting_texts = set(label_counts_per_text[label_counts_per_text > 1].index)
    conflict_mask = df_valid["_norm_text"].isin(conflicting_texts)
    conflicting_rows_dropped = int(conflict_mask.sum())
    unique_conflicting_texts = len(conflicting_texts)

    df_non_conflicting = df_valid[~conflict_mask].copy()

    # 7. Deduplicate Exact Normalized Texts (Keep First)
    is_duplicate = df_non_conflicting["_norm_text"].duplicated(keep="first")
    duplicate_rows_dropped = int(is_duplicate.sum())

    df_dedup = df_non_conflicting[~is_duplicate].copy()

    # 8. Handle Extreme-Length Messages
    truncated_count = 0
    length_dropped_count = 0

    if max_char_length is not None and max_char_length > 0:
        text_lens = df_dedup[text_col].astype(str).str.len()
        extreme_mask = text_lens > max_char_length
        extreme_count = int(extreme_mask.sum())

        if long_text_strategy == "truncate":
            if extreme_count > 0:
                truncated_count = extreme_count
                df_dedup.loc[extreme_mask, text_col] = (
                    df_dedup.loc[extreme_mask, text_col].astype(str).str.slice(0, max_char_length)
                )
        elif long_text_strategy == "drop":
            if extreme_count > 0:
                length_dropped_count = extreme_count
                df_dedup = df_dedup[~extreme_mask].copy()
        elif long_text_strategy != "none":
            raise ValueError(
                f"Unknown long_text_strategy '{long_text_strategy}'. "
                "Must be 'truncate', 'drop', or 'none'."
            )

    # 9. Format Clean Output DataFrame (Drop index columns like 'Unnamed: 0')
    df_clean = pd.DataFrame()
    df_clean["email_text"] = df_dedup[text_col].astype(str)
    df_clean["label"] = df_dedup[label_col].map(label_mapping).astype(int)
    df_clean = df_clean.reset_index(drop=True)

    clean_total = len(df_clean)
    if clean_total == 0:
        raise ValueError("Dataset is empty after completing preprocessing pipeline.")

    # 10. Compute Final Statistics & Audit Report
    safe_count = int((df_clean["label"] == 0).sum())
    phishing_count = int((df_clean["label"] == 1).sum())
    safe_pct = (safe_count / clean_total) * 100
    phishing_pct = (phishing_count / clean_total) * 100

    audit_report = {
        "initial_rows": initial_rows,
        "blank_or_null_dropped": blank_dropped_count,
        "conflicting_texts_found": unique_conflicting_texts,
        "conflicting_rows_dropped": conflicting_rows_dropped,
        "duplicate_rows_dropped": duplicate_rows_dropped,
        "extreme_length_threshold": max_char_length,
        "long_text_strategy": long_text_strategy,
        "extreme_texts_truncated": truncated_count,
        "extreme_texts_dropped": length_dropped_count,
        "final_clean_rows": clean_total,
        "safe_count": safe_count,
        "safe_percent": round(safe_pct, 2),
        "phishing_count": phishing_count,
        "phishing_percent": round(phishing_pct, 2),
        "output_path": output_path,
    }

    # 11. Optionally Save Cleaned Dataset Safely
    if output_path:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        audit_report["output_file_size_bytes"] = os.path.getsize(output_path)
        logger.info(
            "Saved cleaned dataset to '%s' (%d rows, %.2f MB).",
            output_path,
            clean_total,
            audit_report["output_file_size_bytes"] / (1024 * 1024),
        )

    # Print Log Summary
    logger.info("================ PREPROCESSING AUDIT REPORT ================")
    logger.info("Initial records loaded:           %d", audit_report["initial_rows"])
    logger.info("Blank / null rows dropped:        %d", audit_report["blank_or_null_dropped"])
    logger.info(
        "Conflicting-label rows excluded:  %d (across %d unique text keys)",
        audit_report["conflicting_rows_dropped"],
        audit_report["conflicting_texts_found"],
    )
    logger.info("Exact duplicate rows dropped:     %d", audit_report["duplicate_rows_dropped"])
    if truncated_count > 0:
        logger.info("Texts truncated to %d chars:   %d", max_char_length, truncated_count)
    if length_dropped_count > 0:
        logger.info("Texts dropped for exceeding len:  %d", length_dropped_count)
    logger.info("Final modeling records retained:  %d", audit_report["final_clean_rows"])
    logger.info("  - Safe Email (0):               %d (%.2f%%)", safe_count, safe_pct)
    logger.info("  - Phishing Email (1):           %d (%.2f%%)", phishing_count, phishing_pct)
    logger.info("============================================================")

    return df_clean, audit_report


if __name__ == "__main__":
    import sys

    raw_csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "raw", "Phishing_Email.csv")
    processed_csv = os.path.join("data", "processed", "cleaned_emails.csv")

    print(f"Executing preprocessing pipeline on: {raw_csv}")
    clean_df, report = clean_and_preprocess_dataset(
        data_input=raw_csv,
        output_path=processed_csv,
        max_char_length=DEFAULT_MAX_CHAR_LENGTH,
        long_text_strategy="truncate",
    )
    print("Preprocessing completed successfully.")
