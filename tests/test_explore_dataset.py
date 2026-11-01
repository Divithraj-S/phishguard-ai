"""Unit tests for dataset exploration and hygiene checks.

Tests run offline using small temporary sample DataFrames or CSV files.
"""

import os
import pandas as pd
import pytest

from src.explore_dataset import (
    redact_sensitive_text,
    check_empty_or_whitespace,
    find_duplicates,
    compute_text_statistics,
    check_label_leakage,
    run_exploratory_analysis,
)


def test_redact_sensitive_text_emails_and_phones():
    """Verify that emails and phone numbers are redacted and text is truncated."""
    raw_text = (
        "Hello, reach out to security-alert@finance-bank.com or call +1 555-019-2834 "
        "immediately regarding your pending bank transfer authorization."
    )
    redacted = redact_sensitive_text(raw_text, max_length=100)

    assert "[EMAIL REDACTED]" in redacted
    assert "security-alert@finance-bank.com" not in redacted
    assert "[PHONE REDACTED]" in redacted
    assert "555-019-2834" not in redacted
    assert redacted.endswith("...")
    assert len(redacted) <= 103  # 100 + "..."


def test_check_empty_or_whitespace():
    """Verify detection of blank, whitespace, and NaN email entries."""
    df = pd.DataFrame({
        "Email Text": [
            "Valid text here",
            "   ",           # whitespace only
            "",              # empty string
            None,            # null
            "Another valid text",
        ]
    })
    count = check_empty_or_whitespace(df, text_col="Email Text")
    assert count == 3


def test_find_duplicates_clean_dataset():
    """Verify zero duplicates reported when all texts are unique."""
    df = pd.DataFrame({
        "Email Text": ["Email A", "Email B", "Email C"],
        "Email Type": ["Safe Email", "Safe Email", "Phishing Email"],
    })
    dup_stats = find_duplicates(df, text_col="Email Text", label_col="Email Type")

    assert dup_stats["total_duplicate_rows"] == 0
    assert dup_stats["unique_duplicated_texts"] == 0
    assert dup_stats["conflicting_duplicate_texts"] == 0
    assert len(dup_stats["conflicts"]) == 0


def test_find_duplicates_with_exact_and_conflicting_labels():
    """Verify accurate detection of duplicate texts and conflicting class labels."""
    df = pd.DataFrame({
        "Email Text": [
            "Click here for update",      # 1
            "Click here for update",      # 2 (duplicate with different label!)
            "Meeting agenda attached",    # 3
            "Meeting agenda attached",    # 4 (duplicate with same label)
            "Unique email body",          # 5
        ],
        "Email Type": [
            "Safe Email",
            "Phishing Email",
            "Safe Email",
            "Safe Email",
            "Phishing Email",
        ],
    })
    dup_stats = find_duplicates(df, text_col="Email Text", label_col="Email Type")

    assert dup_stats["total_duplicate_rows"] == 2
    assert dup_stats["unique_duplicated_texts"] == 2
    assert dup_stats["conflicting_duplicate_texts"] == 1
    assert len(dup_stats["conflicts"]) == 1
    assert "Safe Email" in dup_stats["conflicts"][0]["conflicting_labels"]
    assert "Phishing Email" in dup_stats["conflicts"][0]["conflicting_labels"]


def test_compute_text_statistics():
    """Verify computation of character and word length statistics."""
    df = pd.DataFrame({
        "Email Text": [
            "One two three",       # 13 chars, 3 words
            "One two three four",  # 18 chars, 4 words
        ],
        "Email Type": ["Safe Email", "Phishing Email"],
    })
    stats = compute_text_statistics(df, text_col="Email Text", label_col="Email Type")

    assert stats["overall"]["word_stats"]["min"] == 3
    assert stats["overall"]["word_stats"]["max"] == 4
    assert stats["overall"]["word_stats"]["mean"] == 3.5
    assert stats["overall"]["char_stats"]["min"] == 13
    assert stats["overall"]["char_stats"]["max"] == 18


def test_check_label_leakage():
    """Verify that label-revealing tokens inside text are identified and tallied."""
    df = pd.DataFrame({
        "Email Text": [
            "Notice: This is a verified phishing email test.",
            "Normal corporate memo without sensitive tokens.",
            "Warning regarding phishing campaign reported by HR.",
        ],
        "Email Type": ["Phishing Email", "Safe Email", "Safe Email"],
    })
    leakage = check_label_leakage(df, text_col="Email Text", label_col="Email Type")

    phish_entry = next((item for item in leakage if item["term"] == "phishing"), None)
    assert phish_entry is not None
    assert phish_entry["total_occurrences"] == 2
    assert phish_entry["label_breakdown"].get("Phishing Email") == 1
    assert phish_entry["label_breakdown"].get("Safe Email") == 1


def test_run_exploratory_analysis_end_to_end(tmp_path):
    """Verify complete analysis execution on a temporary CSV file."""
    sample_csv = tmp_path / "test_eda.csv"
    df = pd.DataFrame({
        "Email Text": [
            "Quarterly financial briefing",
            "Urgent account reset required",
        ],
        "Email Type": [
            "Safe Email",
            "Phishing Email",
        ],
    })
    df.to_csv(sample_csv, index=False)

    report = run_exploratory_analysis(str(sample_csv))

    assert report["rows"] == 2
    assert report["columns"] == 2
    assert report["label_counts"]["Safe Email"] == 1
    assert report["label_counts"]["Phishing Email"] == 1
    assert report["empty_or_whitespace_count"] == 0
    assert report["duplicates"]["total_duplicate_rows"] == 0
