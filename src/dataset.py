"""Dataset acquisition, validation, and integrity module.

Supports acquiring and validating phishing email corpora, including:
1. The Hugging Face / Kaggle Phishing Email Benchmark (Phishing_Email.csv, ~18.6k rows).
2. The Zenodo Phishing Validation Dataset (Phishing_validation_emails.csv, 2k rows).

Ensures data integrity, checks column existence, handles nulls transparently,
supports pre-split deduplication to prevent data leakage, and computes cryptographic hashes.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.request
from typing import Optional, Tuple

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Primary Hugging Face / Kaggle mirror URL
DEFAULT_HF_URL = (
    "https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/main/Phishing_Email.csv"
)
DEFAULT_HF_FILENAME = "Phishing_Email.csv"

# Legacy Zenodo Record URL (preserved for reproducibility)
DEFAULT_ZENODO_URL = (
    "https://zenodo.org/records/13474746/files/Phishing_validation_emails.csv?download=1"
)
DEFAULT_ZENODO_FILENAME = "Phishing_validation_emails.csv"

DEFAULT_RAW_DATA_DIR = os.path.join("data", "raw")

# Expected column names and target label mapping
EXPECTED_TEXT_COLUMN = "Email Text"
EXPECTED_LABEL_COLUMN = "Email Type"

LABEL_MAPPING = {
    "Safe Email": 0,
    "Phishing Email": 1,
}


def compute_file_sha256(file_path: str, chunk_size: int = 65536) -> str:
    """Compute the SHA-256 cryptographic hash of a file.

    Args:
        file_path: Path to the target file.
        chunk_size: Byte chunk size for reading large files in memory.

    Returns:
        Hexadecimal SHA-256 string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot compute hash: file not found at '{file_path}'")

    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_dataset(
    url: str = DEFAULT_HF_URL,
    dest_folder: str = DEFAULT_RAW_DATA_DIR,
    dest_filename: str = DEFAULT_HF_FILENAME,
    timeout: int = 60,
    force_download: bool = False,
) -> str:
    """Download a dataset CSV from an official remote repository.

    Args:
        url: Direct download link (e.g. Hugging Face or Zenodo).
        dest_folder: Destination folder for raw CSV.
        dest_filename: Target filename for the CSV.
        timeout: Network timeout in seconds.
        force_download: If True, re-download even if file already exists.

    Returns:
        The local file path to the downloaded CSV.

    Raises:
        RuntimeError: If download fails due to network, HTTP, or filesystem errors.
    """
    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, dest_filename)

    if os.path.exists(dest_path) and not force_download:
        logger.info(
            "Dataset file already exists at '%s'. Skipping download. (Use force_download=True to re-download).",
            dest_path,
        )
        return dest_path

    logger.info("Initiating dataset download from: %s", url)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PhishingDetectionProject/1.0"}

    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status_code = getattr(response, "status", 200)
            if status_code != 200:
                raise RuntimeError(
                    f"Unexpected HTTP status code {status_code} received from repository."
                )
            content = response.read()

        with open(dest_path, "wb") as f:
            f.write(content)

        file_size_mb = len(content) / (1024 * 1024)
        logger.info("Download completed successfully! Saved to '%s' (%.2f MB).", dest_path, file_size_mb)
        return dest_path

    except urllib.error.HTTPError as http_err:
        raise RuntimeError(
            f"HTTP error during dataset download: {http_err.code} {http_err.reason}. "
            "Please check repository URL availability or access permissions."
        ) from http_err
    except urllib.error.URLError as url_err:
        raise RuntimeError(
            f"Network connection failed: {url_err.reason}. "
            "Please verify your internet connection or proxy settings."
        ) from url_err
    except TimeoutError as timeout_err:
        raise RuntimeError(
            f"Download timed out after {timeout} seconds. "
            "Repository servers may be responding slowly. Please retry."
        ) from timeout_err
    except Exception as exc:
        raise RuntimeError(f"An unexpected error occurred during download: {exc}") from exc


def load_and_validate_dataset(
    csv_path: str,
    text_col: str = EXPECTED_TEXT_COLUMN,
    label_col: str = EXPECTED_LABEL_COLUMN,
    deduplicate: bool = False,
) -> Tuple[pd.DataFrame, dict]:
    """Load, validate, inspect, and optionally deduplicate the email dataset CSV.

    Validates column presence, reports and drops null or blank rows,
    maps text labels to binary targets (0: Safe Email, 1: Phishing Email),
    optionally drops exact duplicate texts to prevent train/test leakage,
    and returns dynamic summary statistics.

    Args:
        csv_path: Path to the CSV file.
        text_col: Name of the column containing email text.
        label_col: Name of the column containing category labels.
        deduplicate: If True, drops duplicate texts (keeping first occurrence)
                     to ensure clean partitioning for training.

    Returns:
        A tuple of:
          - Cleaned pd.DataFrame with columns ['email_text', 'label']
          - Dictionary of calculated summary statistics

    Raises:
        FileNotFoundError: If the specified CSV file does not exist.
        ValueError: If required columns are missing or if no valid rows remain.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset file not found at '{csv_path}'. "
            "Please download the dataset first using download_dataset()."
        )

    # 1. Read the CSV file
    df_raw = pd.read_csv(csv_path)
    initial_rows = len(df_raw)
    actual_columns = list(df_raw.columns)

    # 2. Verify column existence
    missing_columns = []
    if text_col not in df_raw.columns:
        missing_columns.append(text_col)
    if label_col not in df_raw.columns:
        missing_columns.append(label_col)

    if missing_columns:
        raise ValueError(
            f"Schema validation failed! Missing required column(s): {missing_columns}. "
            f"Actual columns found in CSV: {actual_columns}"
        )

    # 3. Explicitly identify and drop null or blank text rows
    null_text_mask = df_raw[text_col].isna() | (df_raw[text_col].astype(str).str.strip() == "")
    null_label_mask = df_raw[label_col].isna()
    invalid_mask = null_text_mask | null_label_mask

    invalid_count = int(invalid_mask.sum())
    if invalid_count > 0:
        logger.warning(
            "Found %d row(s) with missing or empty text/labels. "
            "These rows will be explicitly dropped to protect model integrity.",
            invalid_count,
        )
        df_clean = df_raw[~invalid_mask].copy()
    else:
        df_clean = df_raw.copy()

    # 4. Optional Deduplication
    duplicate_texts_dropped = 0
    if deduplicate:
        normalized_series = df_clean[text_col].astype(str).str.strip()
        dup_mask = normalized_series.duplicated(keep="first")
        duplicate_texts_dropped = int(dup_mask.sum())
        if duplicate_texts_dropped > 0:
            logger.info(
                "Deduplicating dataset: dropped %d duplicate email text rows.",
                duplicate_texts_dropped,
            )
            df_clean = df_clean[~dup_mask].copy()

    # 5. Validate label values
    unique_labels = df_clean[label_col].unique().tolist()
    unexpected_labels = [lbl for lbl in unique_labels if lbl not in LABEL_MAPPING]

    if unexpected_labels:
        raise ValueError(
            f"Validation error: Found unexpected label values: {unexpected_labels}. "
            f"Expected labels are: {list(LABEL_MAPPING.keys())}"
        )

    # 6. Map labels to binary values (0: Safe Email, 1: Phishing Email)
    df_clean["label"] = df_clean[label_col].map(LABEL_MAPPING).astype(int)
    df_clean["email_text"] = df_clean[text_col].astype(str)

    # Standardize output DataFrame
    df_output = df_clean[["email_text", "label"]].reset_index(drop=True)
    clean_rows = len(df_output)

    if clean_rows == 0:
        raise ValueError("Dataset is empty after removing missing values and duplicates!")

    # 7. Calculate dynamic statistics
    safe_count = int((df_output["label"] == 0).sum())
    phishing_count = int((df_output["label"] == 1).sum())
    safe_pct = (safe_count / clean_rows) * 100
    phishing_pct = (phishing_count / clean_rows) * 100

    stats = {
        "initial_rows": initial_rows,
        "dropped_rows": invalid_count,
        "duplicate_rows_dropped": duplicate_texts_dropped,
        "clean_rows": clean_rows,
        "columns_found": actual_columns,
        "safe_count": safe_count,
        "safe_percent": round(safe_pct, 2),
        "phishing_count": phishing_count,
        "phishing_percent": round(phishing_pct, 2),
    }

    logger.info("================ DATASET VALIDATION SUMMARY ================")
    logger.info("Total rows loaded:           %d", stats["initial_rows"])
    logger.info("Missing/blank rows dropped:  %d", stats["dropped_rows"])
    if deduplicate:
        logger.info("Exact duplicates dropped:    %d", stats["duplicate_rows_dropped"])
    logger.info("Valid rows retained:         %d", stats["clean_rows"])
    logger.info("Class 0 ('Safe Email'):       %d (%.2f%%)", stats["safe_count"], stats["safe_percent"])
    logger.info("Class 1 ('Phishing Email'):   %d (%.2f%%)", stats["phishing_count"], stats["phishing_percent"])
    logger.info("============================================================")

    return df_output, stats


if __name__ == "__main__":
    import sys

    target_file = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DEFAULT_RAW_DATA_DIR, DEFAULT_HF_FILENAME)
    print(f"Validating dataset at: {target_file}")
    if os.path.exists(target_file):
        df, s = load_and_validate_dataset(target_file, deduplicate=True)
        print("Validation and deduplication successful.")
    else:
        print(f"File not found. Run download_dataset() first.")
