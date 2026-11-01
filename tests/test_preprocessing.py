"""Unit tests for the data preprocessing and hygiene pipeline.

Tests run offline using small temporary DataFrames and sample files,
verifying blank removal, conflict exclusion, deduplication, length truncation,
schema validation, and preservation of raw input files.
"""

import hashlib
import os
import pandas as pd
import pytest

from src.preprocessing import (
    normalize_whitespace,
    clean_and_preprocess_dataset,
    DEFAULT_TEXT_COL,
    DEFAULT_LABEL_COL,
)


def test_normalize_whitespace():
    """Verify that leading, trailing, and redundant internal whitespace is normalized."""
    raw = "   Subject: Urgent    Action \t Required  \n\n  "
    norm = normalize_whitespace(raw)
    assert norm == "Subject: Urgent Action Required"
    # Ensure non-string values return empty string safely
    assert normalize_whitespace(None) == ""


def test_empty_input_and_missing_columns_raise_errors(tmp_path):
    """Verify that empty inputs and schema discrepancies trigger descriptive errors."""
    # 1. Empty DataFrame
    empty_df = pd.DataFrame(columns=[DEFAULT_TEXT_COL, DEFAULT_LABEL_COL])
    with pytest.raises(ValueError, match="Input dataset is empty"):
        clean_and_preprocess_dataset(empty_df)

    # 2. Missing required columns
    bad_cols_df = pd.DataFrame({
        "Subject": ["Hello"],
        "Target": ["Safe Email"],
    })
    with pytest.raises(ValueError, match="Schema validation failed! Missing required column"):
        clean_and_preprocess_dataset(bad_cols_df)

    # 3. Non-existent file path
    non_existent = str(tmp_path / "non_existent.csv")
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        clean_and_preprocess_dataset(non_existent)


def test_unknown_labels_raise_error():
    """Verify that unrecognized class labels are rejected."""
    df = pd.DataFrame({
        DEFAULT_TEXT_COL: ["Promotional offer for cheap travel"],
        DEFAULT_LABEL_COL: ["Unsolicited Commercial Spam"],
    })
    with pytest.raises(ValueError, match="Label validation failed! Found unrecognized labels"):
        clean_and_preprocess_dataset(df)


def test_missing_and_blank_texts_removed():
    """Verify that empty, blank, and null text entries are dropped."""
    df = pd.DataFrame({
        DEFAULT_TEXT_COL: [
            "Valid legitimate email body",
            "",               # empty string
            "   \t  \n ",     # whitespace only
            None,             # null
            "Valid phishing email text",
        ],
        DEFAULT_LABEL_COL: [
            "Safe Email",
            "Safe Email",
            "Phishing Email",
            "Safe Email",
            "Phishing Email",
        ],
    })
    clean_df, report = clean_and_preprocess_dataset(df)

    assert len(clean_df) == 2
    assert report["blank_or_null_dropped"] == 3
    assert report["final_clean_rows"] == 2
    assert list(clean_df["label"]) == [0, 1]


def test_duplicate_texts_same_label_removed():
    """Verify that exact duplicate texts sharing the same label are deduplicated."""
    df = pd.DataFrame({
        DEFAULT_TEXT_COL: [
            "Project deadline on Friday.",
            "Project deadline on Friday.",        # Exact duplicate
            "  Project deadline on Friday.  \t",  # Duplicate with whitespace variance
            "Urgent: click link to verify bank.",
        ],
        DEFAULT_LABEL_COL: [
            "Safe Email",
            "Safe Email",
            "Safe Email",
            "Phishing Email",
        ],
    })
    clean_df, report = clean_and_preprocess_dataset(df)

    assert len(clean_df) == 2
    assert report["duplicate_rows_dropped"] == 2
    assert report["final_clean_rows"] == 2


def test_conflicting_labels_excluded():
    """Verify that texts appearing with conflicting labels are entirely excluded."""
    df = pd.DataFrame({
        DEFAULT_TEXT_COL: [
            "Account verification notice",  # Row 1 (Safe)
            "Account verification notice",  # Row 2 (Phishing) -> CONFLICT!
            "Meeting agenda notes",         # Row 3 (Safe)
            "Fake lottery claim prize",     # Row 4 (Phishing)
        ],
        DEFAULT_LABEL_COL: [
            "Safe Email",
            "Phishing Email",
            "Safe Email",
            "Phishing Email",
        ],
    })
    clean_df, report = clean_and_preprocess_dataset(df)

    # Both conflicting rows (Row 1 and Row 2) must be removed
    assert len(clean_df) == 2
    assert report["conflicting_rows_dropped"] == 2
    assert report["conflicting_texts_found"] == 1
    # Ensure the conflicting text is completely absent from the cleaned output
    remaining_texts = list(clean_df["email_text"])
    assert "Account verification notice" not in remaining_texts
    assert "Meeting agenda notes" in remaining_texts
    assert "Fake lottery claim prize" in remaining_texts


def test_extreme_length_truncation_and_filtering():
    """Verify handling of extreme-length messages via truncation and dropping."""
    short_text = "Short legitimate text."
    long_text = "A" * 150  # 150 characters

    df = pd.DataFrame({
        DEFAULT_TEXT_COL: [short_text, long_text],
        DEFAULT_LABEL_COL: ["Safe Email", "Phishing Email"],
    })

    # Strategy 1: Truncate (default)
    clean_trunc, rep_trunc = clean_and_preprocess_dataset(
        df, max_char_length=100, long_text_strategy="truncate"
    )
    assert len(clean_trunc) == 2
    assert rep_trunc["extreme_texts_truncated"] == 1
    assert len(clean_trunc.iloc[1]["email_text"]) == 100

    # Strategy 2: Drop / Filter
    clean_drop, rep_drop = clean_and_preprocess_dataset(
        df, max_char_length=100, long_text_strategy="drop"
    )
    assert len(clean_drop) == 1
    assert rep_drop["extreme_texts_dropped"] == 1
    assert clean_drop.iloc[0]["email_text"] == short_text


def test_preservation_of_raw_file(tmp_path):
    """Verify that running preprocessing never modifies the raw input file."""
    raw_content = (
        "Unnamed: 0,Email Text,Email Type\n"
        "0,Quarterly status report,Safe Email\n"
        "1,Urgent: password reset,Phishing Email\n"
        "2,,Safe Email\n"
        "3,Quarterly status report,Safe Email\n"
    )
    raw_file = tmp_path / "raw_input.csv"
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(raw_content)

    # Compute SHA-256 before preprocessing
    hasher_before = hashlib.sha256()
    with open(raw_file, "rb") as f:
        hasher_before.update(f.read())
    hash_before = hasher_before.hexdigest()

    # Run preprocessing with output path
    out_file = tmp_path / "processed" / "clean.csv"
    clean_df, report = clean_and_preprocess_dataset(
        str(raw_file), output_path=str(out_file)
    )

    # Compute SHA-256 after preprocessing
    hasher_after = hashlib.sha256()
    with open(raw_file, "rb") as f:
        hasher_after.update(f.read())
    hash_after = hasher_after.hexdigest()

    # Raw file must remain bit-for-bit identical
    assert hash_before == hash_after

    # Processed file must exist and contain only modeling columns
    assert os.path.exists(out_file)
    assert list(clean_df.columns) == ["email_text", "label"]
    assert "Unnamed: 0" not in clean_df.columns
    assert len(clean_df) == 2  # 1 blank dropped, 1 duplicate dropped
