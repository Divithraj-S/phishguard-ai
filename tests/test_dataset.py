"""Unit tests for dataset loading, validation, hashing, and deduplication logic.

Tests use temporary in-memory/isolated files to verify validation logic
offline without requiring network access or external downloads.
"""

import hashlib
import os
import pandas as pd
import pytest

from src.dataset import (
    load_and_validate_dataset,
    compute_file_sha256,
    EXPECTED_TEXT_COLUMN,
    EXPECTED_LABEL_COLUMN,
)


def test_load_and_validate_success(tmp_path):
    """Verify that a valid CSV is loaded, validated, and correctly mapped."""
    sample_data = {
        EXPECTED_TEXT_COLUMN: [
            "Team meeting tomorrow at 10 AM.",
            "Urgent: Your account is suspended. Click here to verify password.",
            "Please review the attached project report.",
            "You have won a lottery! Claim your cash prize immediately.",
        ],
        EXPECTED_LABEL_COLUMN: [
            "Safe Email",
            "Phishing Email",
            "Safe Email",
            "Phishing Email",
        ],
    }
    csv_file = tmp_path / "sample_valid.csv"
    pd.DataFrame(sample_data).to_csv(csv_file, index=False)

    df_clean, stats = load_and_validate_dataset(str(csv_file))

    assert len(df_clean) == 4
    assert stats["initial_rows"] == 4
    assert stats["dropped_rows"] == 0
    assert stats["clean_rows"] == 4
    assert stats["safe_count"] == 2
    assert stats["phishing_count"] == 2
    assert stats["safe_percent"] == 50.0
    assert stats["phishing_percent"] == 50.0

    # Ensure labels are binary integers: Safe Email -> 0, Phishing Email -> 1
    assert list(df_clean["label"]) == [0, 1, 0, 1]
    assert list(df_clean.columns) == ["email_text", "label"]


def test_missing_file_raises_error(tmp_path):
    """Verify that attempting to load a non-existent file raises FileNotFoundError."""
    non_existent = str(tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_and_validate_dataset(non_existent)


def test_missing_column_raises_error(tmp_path):
    """Verify that missing required columns triggers a descriptive ValueError."""
    sample_data = {
        "RandomText": ["Hello world"],
        EXPECTED_LABEL_COLUMN: ["Safe Email"],
    }
    csv_file = tmp_path / "sample_missing_col.csv"
    pd.DataFrame(sample_data).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Schema validation failed! Missing required column"):
        load_and_validate_dataset(str(csv_file))


def test_handles_null_values_transparently(tmp_path):
    """Verify that missing/empty rows are detected, reported, and dropped."""
    sample_data = {
        EXPECTED_TEXT_COLUMN: [
            "Valid legitimate email text",
            "",                      # empty string
            None,                    # null text
            "Valid phishing email text",
        ],
        EXPECTED_LABEL_COLUMN: [
            "Safe Email",
            "Safe Email",
            "Phishing Email",
            None,                    # null label
        ],
    }
    csv_file = tmp_path / "sample_with_nulls.csv"
    pd.DataFrame(sample_data).to_csv(csv_file, index=False)

    df_clean, stats = load_and_validate_dataset(str(csv_file))

    # Only the first row is completely valid
    assert len(df_clean) == 1
    assert stats["initial_rows"] == 4
    assert stats["dropped_rows"] == 3
    assert stats["clean_rows"] == 1
    assert stats["safe_count"] == 1
    assert stats["phishing_count"] == 0
    assert df_clean.iloc[0]["label"] == 0


def test_unexpected_labels_raise_error(tmp_path):
    """Verify that unrecognized category labels trigger a descriptive error."""
    sample_data = {
        EXPECTED_TEXT_COLUMN: [
            "Special discounted sunglasses for sale!",
        ],
        EXPECTED_LABEL_COLUMN: [
            "Commercial Spam",  # Not recognized in binary Safe/Phishing schema
        ],
    }
    csv_file = tmp_path / "sample_bad_label.csv"
    pd.DataFrame(sample_data).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Found unexpected label values"):
        load_and_validate_dataset(str(csv_file))


def test_empty_dataset_raises_error(tmp_path):
    """Verify that an empty dataset (headers only) raises a ValueError."""
    sample_data = {
        EXPECTED_TEXT_COLUMN: [],
        EXPECTED_LABEL_COLUMN: [],
    }
    csv_file = tmp_path / "sample_empty.csv"
    pd.DataFrame(sample_data).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Dataset is empty after removing missing values"):
        load_and_validate_dataset(str(csv_file))


def test_deduplication_removes_duplicate_texts(tmp_path):
    """Verify that deduplicate=True removes repeated email bodies."""
    sample_data = {
        EXPECTED_TEXT_COLUMN: [
            "Meeting at noon",
            "Meeting at noon",  # exact duplicate
            "Meeting at noon ", # duplicate with trailing whitespace
            "Security alert: verify account",
        ],
        EXPECTED_LABEL_COLUMN: [
            "Safe Email",
            "Safe Email",
            "Safe Email",
            "Phishing Email",
        ],
    }
    csv_file = tmp_path / "sample_duplicates.csv"
    pd.DataFrame(sample_data).to_csv(csv_file, index=False)

    # Without deduplication
    df_no_dedup, stats_no_dedup = load_and_validate_dataset(str(csv_file), deduplicate=False)
    assert len(df_no_dedup) == 4
    assert stats_no_dedup["duplicate_rows_dropped"] == 0

    # With deduplication
    df_dedup, stats_dedup = load_and_validate_dataset(str(csv_file), deduplicate=True)
    assert len(df_dedup) == 2
    assert stats_dedup["duplicate_rows_dropped"] == 2
    assert stats_dedup["clean_rows"] == 2


def test_compute_file_sha256(tmp_path):
    """Verify cryptographic SHA-256 checksum computation."""
    test_content = b"Antigravity Phishing Project Test Content 123"
    test_file = tmp_path / "hash_test.txt"
    with open(test_file, "wb") as f:
        f.write(test_content)

    expected_hash = hashlib.sha256(test_content).hexdigest()
    computed_hash = compute_file_sha256(str(test_file))

    assert computed_hash == expected_hash
