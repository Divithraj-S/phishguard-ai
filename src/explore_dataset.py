"""Exploratory Data Analysis (EDA) module for email phishing dataset.

Analyzes dataset hygiene, schema integrity, class distributions, text length statistics,
exact duplicates, conflicting labels, and potential label leakage signals.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RAW_CSV = os.path.join("data", "raw", "Phishing_validation_emails.csv")


def redact_sensitive_text(text: str, max_length: int = 120) -> str:
    """Redact email addresses, phone numbers, and truncate long text.

    Args:
        text: Raw email text string.
        max_length: Maximum characters to display before truncating.

    Returns:
        Sanitized and truncated text string.
    """
    if not isinstance(text, str):
        return ""

    # Replace email addresses
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    sanitized = re.sub(email_pattern, "[EMAIL REDACTED]", text)

    # Replace phone numbers / long phone-like digit sequences
    phone_pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    sanitized = re.sub(phone_pattern, "[PHONE REDACTED]", sanitized)

    # Collapse excess internal whitespace
    sanitized = " ".join(sanitized.split())

    if len(sanitized) > max_length:
        return sanitized[:max_length].rstrip() + "..."
    return sanitized


def check_empty_or_whitespace(df: pd.DataFrame, text_col: str = "Email Text") -> int:
    """Count empty or whitespace-only rows in the email text column."""
    if text_col not in df.columns:
        return 0
    mask = df[text_col].isna() | (df[text_col].astype(str).str.strip() == "")
    return int(mask.sum())


def find_duplicates(
    df: pd.DataFrame,
    text_col: str = "Email Text",
    label_col: str = "Email Type",
) -> Dict[str, Any]:
    """Identify exact duplicate email texts and check for conflicting labels.

    Args:
        df: Input DataFrame.
        text_col: Name of column containing email text.
        label_col: Name of column containing class labels.

    Returns:
        Dictionary containing counts of duplicates, unique duplicated texts,
        and details of any conflicting duplicates (same text, different labels).
    """
    if text_col not in df.columns:
        return {
            "total_duplicate_rows": 0,
            "unique_duplicated_texts": 0,
            "conflicting_duplicate_texts": 0,
            "conflicts": [],
        }

    # Normalize text to detect duplicates ignoring leading/trailing whitespace
    normalized_texts = df[text_col].astype(str).str.strip()

    # Exact duplicate rows (excluding the first occurrence)
    is_duplicate_row = normalized_texts.duplicated(keep="first")
    total_duplicate_rows = int(is_duplicate_row.sum())

    # Find texts that appear more than once
    counts = normalized_texts.value_counts()
    duplicated_texts = counts[counts > 1].index.tolist()
    unique_duplicated_texts = len(duplicated_texts)

    # Check for conflicting labels among duplicates
    conflicts: List[Dict[str, Any]] = []
    if label_col in df.columns:
        # Group by normalized text and check unique labels per group
        grouped = df.groupby(normalized_texts)[label_col].unique()
        conflicting_groups = grouped[grouped.apply(len) > 1]

        for text_sample, labels in conflicting_groups.items():
            conflicts.append({
                "sample_text": redact_sensitive_text(str(text_sample), max_length=100),
                "conflicting_labels": [str(lbl) for lbl in labels],
            })

    return {
        "total_duplicate_rows": total_duplicate_rows,
        "unique_duplicated_texts": unique_duplicated_texts,
        "conflicting_duplicate_texts": len(conflicts),
        "conflicts": conflicts,
    }


def compute_text_statistics(
    df: pd.DataFrame,
    text_col: str = "Email Text",
    label_col: Optional[str] = "Email Type",
) -> Dict[str, Any]:
    """Calculate character and word length statistics overall and per class."""
    if text_col not in df.columns or len(df) == 0:
        return {}

    series = df[text_col].astype(str)
    char_lens = series.str.len()
    word_lens = series.apply(lambda s: len(s.split()))

    def get_summary_stats(vals: pd.Series) -> Dict[str, float]:
        return {
            "min": int(vals.min()),
            "max": int(vals.max()),
            "mean": round(float(vals.mean()), 1),
            "median": round(float(vals.median()), 1),
        }

    overall = {
        "char_stats": get_summary_stats(char_lens),
        "word_stats": get_summary_stats(word_lens),
    }

    by_class: Dict[str, Dict[str, Any]] = {}
    if label_col and label_col in df.columns:
        for lbl in df[label_col].dropna().unique():
            sub = df[df[label_col] == lbl][text_col].astype(str)
            if len(sub) > 0:
                sub_chars = sub.str.len()
                sub_words = sub.apply(lambda s: len(s.split()))
                by_class[str(lbl)] = {
                    "count": len(sub),
                    "char_stats": get_summary_stats(sub_chars),
                    "word_stats": get_summary_stats(sub_words),
                }

    return {
        "overall": overall,
        "by_class": by_class,
    }


def check_label_leakage(
    df: pd.DataFrame,
    text_col: str = "Email Text",
    label_col: str = "Email Type",
    suspicious_terms: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Check for obvious label leakage or dataset artifacts inside email text.

    Searches for terms like 'phishing', 'safe email', 'phish', 'spamassassin', etc.,
    and analyzes if their presence directly reveals the label.
    """
    if text_col not in df.columns or label_col not in df.columns:
        return []

    if suspicious_terms is None:
        suspicious_terms = [
            "phishing",
            "phish",
            "safe email",
            "phishing email",
            "spamassassin",
            "nazario",
            "enron",
            "legitimate email",
        ]

    results: List[Dict[str, Any]] = []
    text_lower = df[text_col].astype(str).str.lower()

    for term in suspicious_terms:
        # Match as whole word or phrase
        pattern = rf"\b{re.escape(term)}\b"
        matches_mask = text_lower.str.contains(pattern, regex=True, na=False)
        match_count = int(matches_mask.sum())

        if match_count > 0:
            sub = df[matches_mask]
            label_breakdown = sub[label_col].value_counts().to_dict()
            results.append({
                "term": term,
                "total_occurrences": match_count,
                "label_breakdown": {str(k): int(v) for k, v in label_breakdown.items()},
            })

    return results


def run_exploratory_analysis(csv_path: str = DEFAULT_RAW_CSV) -> Dict[str, Any]:
    """Execute complete exploratory data analysis on the specified dataset.

    Args:
        csv_path: File path to the raw dataset CSV.

    Returns:
        Structured dictionary containing all computed metrics.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found at '{csv_path}'. Please ensure the dataset is present."
        )

    df = pd.read_csv(csv_path)

    # 1. Basic Dimensions
    rows, cols = df.shape
    column_info = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # 2. Label Distribution
    text_col = "Email Text" if "Email Text" in df.columns else df.columns[0]
    label_col = "Email Type" if "Email Type" in df.columns else (df.columns[1] if cols > 1 else None)

    label_counts: Dict[str, int] = {}
    label_pcts: Dict[str, float] = {}
    if label_col and label_col in df.columns:
        counts = df[label_col].value_counts()
        for k, v in counts.items():
            label_counts[str(k)] = int(v)
            label_pcts[str(k)] = round((v / rows) * 100, 2)

    # 3. Data Hygiene
    empty_count = check_empty_or_whitespace(df, text_col=text_col)
    dup_info = find_duplicates(df, text_col=text_col, label_col=label_col or "")

    # 4. Length Statistics
    length_stats = compute_text_statistics(df, text_col=text_col, label_col=label_col)

    # 5. Label Leakage Check
    leakage_info = (
        check_label_leakage(df, text_col=text_col, label_col=label_col)
        if label_col
        else []
    )

    # 6. Sample Examples (Redacted & Truncated)
    samples: Dict[str, List[str]] = {}
    if label_col and label_col in df.columns:
        for lbl in df[label_col].dropna().unique():
            examples = (
                df[df[label_col] == lbl][text_col]
                .dropna()
                .head(3)
                .apply(lambda s: redact_sensitive_text(str(s), max_length=120))
                .tolist()
            )
            samples[str(lbl)] = examples

    report = {
        "csv_path": csv_path,
        "rows": rows,
        "columns": cols,
        "column_info": column_info,
        "label_counts": label_counts,
        "label_percentages": label_pcts,
        "empty_or_whitespace_count": empty_count,
        "duplicates": dup_info,
        "length_stats": length_stats,
        "leakage_findings": leakage_info,
        "samples": samples,
    }

    return report


def print_exploration_report(report: Dict[str, Any]) -> None:
    """Print a clean, beginner-friendly formatted report of EDA findings."""
    print("\n" + "=" * 65)
    print("       EXPLORATORY DATA ANALYSIS (EDA) REPORT")
    print("=" * 65)
    print(f"Dataset File:       {report['csv_path']}")
    print(f"Total Rows:         {report['rows']:,}")
    print(f"Total Columns:      {report['columns']}")
    print("Column Schema:")
    for col, dtype in report["column_info"].items():
        print(f"  - {col}: {dtype}")

    print("\n--- 1. CLASS BALANCE & LABEL DISTRIBUTION ---")
    for lbl, count in report["label_counts"].items():
        pct = report["label_percentages"].get(lbl, 0.0)
        print(f"  * {lbl}: {count:,} emails ({pct:.2f}%)")

    print("\n--- 2. DATA HYGIENE & DUPLICATE CHECKS ---")
    print(f"  * Empty or whitespace-only emails: {report['empty_or_whitespace_count']}")
    dup = report["duplicates"]
    print(f"  * Duplicate email rows:            {dup['total_duplicate_rows']}")
    print(f"  * Unique texts that repeat:        {dup['unique_duplicated_texts']}")
    print(f"  * Conflicting label duplicates:    {dup['conflicting_duplicate_texts']}")

    if dup["conflicting_duplicate_texts"] > 0:
        print("    [WARNING] Found texts appearing with both Safe and Phishing labels:")
        for conflict in dup["conflicts"][:3]:
            print(f"      - Labels {conflict['conflicting_labels']}: \"{conflict['sample_text']}\"")

    print("\n--- 3. EMAIL LENGTH STATISTICS ---")
    overall = report["length_stats"].get("overall", {})
    char_st = overall.get("char_stats", {})
    word_st = overall.get("word_stats", {})
    print(
        f"  * Overall Character Length: Min={char_st.get('min')}, "
        f"Max={char_st.get('max')}, Median={char_st.get('median')}, Mean={char_st.get('mean')}"
    )
    print(
        f"  * Overall Word Count:       Min={word_st.get('min')}, "
        f"Max={word_st.get('max')}, Median={word_st.get('median')}, Mean={word_st.get('mean')}"
    )

    by_class = report["length_stats"].get("by_class", {})
    for lbl, st in by_class.items():
        c_st = st.get("char_stats", {})
        w_st = st.get("word_stats", {})
        print(
            f"  * [{lbl}] Word Count: Median={w_st.get('median')}, Mean={w_st.get('mean')} "
            f"(Char Median={c_st.get('median')})"
        )

    print("\n--- 4. LABEL LEAKAGE & SUSPICIOUS MARKERS CHECK ---")
    leakages = report["leakage_findings"]
    if not leakages:
        print("  * No suspicious label-revealing keywords ('phishing', 'safe email', etc.) detected in email bodies.")
    else:
        print("  * Detected potential keyword overlap in email bodies:")
        for item in leakages:
            print(
                f"    - Term '{item['term']}': {item['total_occurrences']} occurrences. "
                f"Breakdown: {item['label_breakdown']}"
            )

    print("\n--- 5. REPRESENTATIVE EMAIL SNIPPETS (REDACTED) ---")
    for lbl, examples in report["samples"].items():
        print(f"  [{lbl} Examples]:")
        for i, snippet in enumerate(examples, 1):
            print(f"    {i}. \"{snippet}\"")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    import sys

    target_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW_CSV
    rep = run_exploratory_analysis(target_path)
    print_exploration_report(rep)
