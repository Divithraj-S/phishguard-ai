# Dataset Documentation: Phishing Email Detection Benchmark

## 1. Primary Sources & Provenance

### Active Dataset (`Phishing_Email.csv`)
* **Repository (Download Mirror):** [Hugging Face: `zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)
* **Direct Download URL:** `https://huggingface.co/datasets/zefang-liu/phishing-email-dataset/resolve/main/Phishing_Email.csv`
* **Original Kaggle Source:** [Kaggle: `subhajournal/phishingemails`](https://www.kaggle.com/datasets/subhajournal/phishingemails) (Uploaded by Subhadeep Chakraborty)
* **Associated Academic Publication:**
  * Al-Subaiey, A., Al-Thani, M., Alam, N. A., Antora, K. F., Khandakar, A., & Zaman, S. A. U. (May 2024). *"Novel Interpretable and Robust Web-based AI Platform for Phishing Email Detection"*, arXiv:2405.11619.
* **Underlying Constituent Corpora:**
  * *Legitimate / Safe Emails:* Sourced primarily from the **Enron Email Corpus** (FERC public record) and **Ling-Spam / Apache SpamAssassin** ham archives.
  * *Phishing / Malicious Emails:* Sourced from the **Jose Nazario Phishing Corpus**, **CEAS 2008**, **Nigerian Fraud letters**, and **SpamAssassin** threat feeds.

### Preserved Historical File (`Phishing_validation_emails.csv`)
* **Source:** [Zenodo Record 13474746](https://zenodo.org/records/13474746) (Miltchev, Rangelov, & Genchev, 2024).
* **Status:** Preserved in `data/raw/` for audit reference. Not used for training due to finding only 100 unique sentences padded to 2,000 rows.

---

## 2. Verified File Verification & Cryptographic Integrity

* **Raw File Location:** `data/raw/Phishing_Email.csv`
* **Acquisition Date:** 2026-10-28
* **File Size:** `52,034,604 bytes (49.62 MB)`
* **SHA-256 Checksum:** `18ef4fff1acb8986f0ab01e83ede6025420c829656af5a377560fce17e153b97` (Verified bit-for-bit unchanged after preprocessing).
* **Processed File Location:** `data/processed/cleaned_emails.csv` (`32,610,642 bytes / 31.10 MB`, 17,519 rows).
* **Git Status:** Both `data/raw/` and `data/processed/` are excluded from Git tracking via `.gitignore`.

---

## 3. Preprocessing Pipeline Rules & Audit Report

The preprocessing module ([src/preprocessing.py](file:///E:/Phishing-Email-Detection/src/preprocessing.py)) executes auditable, non-destructive cleaning steps:

| Preprocessing Step | Action Taken | Records Affected | Output Records Remaining |
| :--- | :--- | :--- | :--- |
| **1. Input Load & Validation** | Validated schema (`Email Text`, `Email Type`) and labels. Dropped index column (`Unnamed: 0`). | 0 invalid columns | 18,650 rows |
| **2. Blank & Null Text Removal** | Removed rows where `Email Text` was NaN or whitespace-only, or label was missing. | **19 rows dropped** | 18,631 rows |
| **3. Conflicting-Label Exclusion** | Grouped by normalized text. Identified text `"empty"` assigned conflicting labels (335 Phishing, 198 Safe). Excluded all conflicting rows. | **533 rows excluded** (1 unique text key) | 18,098 rows |
| **4. Exact Text Deduplication** | Collapsed duplicate messages using canonical whitespace matching, retaining first occurrence. | **579 duplicate rows dropped** | 17,519 rows |
| **5. Extreme-Length Handling** | Configured `max_char_length = 50,000` with `long_text_strategy = "truncate"`. Outliers capped without dropping records. | **25 texts truncated** (0 dropped) | **17,519 final rows** |

### Clean Modeling Dataset Distribution
* **Total Valid Records:** **17,519 emails**
* **Class 0 (`Safe Email`):** 10,978 emails (**62.66%**)
* **Class 1 (`Phishing Email`):** 6,541 emails (**37.34%**)
* **Clean File Columns:** `email_text` (*string*), `label` (*int64: 0 or 1*).

---

## 4. Reproducible Splitting & Leakage Audit

### Stratified 80/20 Partition (`random_state=42`)
* **Total Clean Rows:** 17,519
* **Training Set (80%):** 14,015 rows (Safe: 8,782 [62.66%], Phishing: 5,233 [37.34%])
* **Test Set (20%):** 3,504 rows (Safe: 2,196 [62.67%], Phishing: 1,308 [37.33%])
* **Exact Text Overlap:** **0 rows (0.0%)** — No exact duplicate messages exist across the train and test partitions.
* **Leakage Prevention Architecture:** All text vectorization is encapsulated inside a `sklearn.pipeline.Pipeline`, ensuring that TF-IDF vocabulary and IDF frequency weights are fitted strictly on `X_train` and merely transform `X_test`.

### Near-Duplicate Audit Findings
* **Methodology:** We audited a random sample of 200 test emails against training emails within a $\pm 20\%$ character length window, computing word-set Jaccard similarity.
* **Empirical Finding:** **17 out of 200 sampled test emails (8.5%)** exhibited Jaccard similarity $\ge 0.85$ with training messages.
* **Interpretation:** These near-duplicates correspond to repeating campaign templates (e.g. mass phishing alerts or corporate newsletters where only dates, recipient names, or transaction IDs differ). We do **not** claim near-duplicates have been eliminated.

---

## 5. Evaluation Metrics & False Positive Rate Formulation

In our security evaluation, **Class 1 (`Phishing Email`) is the positive target class**, and **Class 0 (`Safe Email`) is the negative class**.

### Metric Formulations:
1. **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$ (Overall correctness).
2. **Precision:** $\frac{TP}{TP + FP}$ (When an alarm sounds, what percentage is actually phishing?).
3. **Recall / Sensitivity:** $\frac{TP}{TP + FN}$ (What percentage of actual phishing attacks did we catch?).
4. **F1-Score:** $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ (Harmonic balance).
5. **False Positive Rate (FPR):**
   $$\text{FPR} = \frac{FP}{FP + TN}$$
   *Meaning:* The percentage of truly safe emails that are mistakenly quarantined or flagged as phishing.
6. **Zero-Denominator Handling:** If an evaluation subset contains zero true negative instances ($FP + TN == 0$), $FPR$ safely defaults to `0.0` with an explicit flag (`zero_denominator_fpr = True`) rather than crashing.

---

## 6. Methodological Limitations & Source Biases

### Absence of Source Metadata
* The dataset schema does not contain an `origin` column (e.g. Enron vs. Nazario vs. SpamAssassin). A source-based split is therefore impossible without heuristic guessing.

### The "Enron" Corporate Bias
* The word `"enron"` appears **2,299 times in `Safe Email` and only 1 time in `Phishing Email`**. The model faces the risk of learning corporate energy terminology as a marker of safety.

### Spam vs. Phishing Contamination
* The `Phishing Email` class contains general commercial spam, adult spam, and penny-stock scams alongside true credential-theft phishing.

---

## 7. Privacy & Security Notices

### Personal Identifiable Information (PII)
* The raw emails contain historical personal names, telephone numbers, and email addresses.
* **Mandatory Control:** All project displays, test logs, and Streamlit interfaces route email text through `redact_sensitive_text()`. Raw and processed CSVs are ignored by Git.

### Security Non-Guarantee Disclaimer
> **DISCLAIMER:** Machine learning classification models are probabilistic analytical aids. A classification of "Safe Email" or a low risk score does not guarantee that an email is 100% benign. Users must always verify sender authenticity, inspect URLs directly, and adhere to organizational multi-factor authentication policies.
