# Reducing False Positives in Phishing Email Detection

An end-to-end machine learning project and interactive Streamlit web application designed to classify email text and evaluate decision threshold trade-offs to minimize disruptive false alarms.

---

## 1. Project Overview

In organizational email security, **False Positives** (legitimate emails incorrectly flagged as phishing) carry a heavy operational cost: they block critical business correspondence, disrupt communication with clients, and induce alert fatigue among employees.

This project implements a reproducible, tested machine learning pipeline that:
1. Cleans and deduplicates raw email corpora without data leakage.
2. Trains baseline NLP models (TF-IDF + Logistic Regression and Random Forest).
3. Systematically tunes the **decision threshold** on a dedicated validation split to cut False Positives while maintaining $\ge 90\%$ threat detection coverage.
4. Delivers a beginner-friendly, accessible **Streamlit web application** for interactive analysis.

---

## 2. Quickstart & Running the Application

### 2.1 Prerequisites
* Windows 10/11
* Python 3.12 (Virtual environment configured at `.venv`)

### 2.2 Launch the Streamlit App
Run the following command from the project root in PowerShell:
```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```
Streamlit will open the application in your default web browser (typically at `http://localhost:8501`).

### 2.3 Run Automated Tests
Execute the complete test suite (52 tests across 7 test modules):
```powershell
.venv\Scripts\python.exe -m pytest -v
```

---

## 3. How the Application Works

1. **Email Text Input:** Paste arbitrary email text (subject + body) or pick from pre-configured sample templates.
2. **Decision Threshold Selection:** Adjust the threshold slider (`0.20` to `0.90`, step `0.05`, default `0.60`) or click one of the quick presets:
   * **More sensitive (`0.40`):** Catches more borderline attacks, but causes more false alarms on safe emails.
   * **Balanced (`0.50`):** Standard default machine learning cutoff.
   * **Fewer false alarms (`0.60`):** Recommended operating point; cuts false alarms by ~42% while retaining 93.7% recall.
3. **Analyze:** The app applies the pre-trained Logistic Regression pipeline, computing estimated probabilities:
   $$\text{If } P(\text{Phishing}) \ge \text{Threshold} \implies \text{Classified as Phishing Email}$$
   $$\text{If } P(\text{Phishing}) < \text{Threshold} \implies \text{Classified as Safe Email}$$
4. **Transparent Results:** Displays the predicted category, exact percentage scores, visual probability spectrum, and a pedagogical explanation of why the active threshold produced that result.

---

## 4. Model Architecture & Training

* **Feature Vectorizer:** Scikit-learn `TfidfVectorizer`
  * Vocabulary: Top 10,000 unigrams and bigrams (`ngram_range=(1, 2)`)
  * Sublinear term frequency scaling (`sublinear_tf=True`)
  * Document frequency bounds: `min_df=2`, `max_df=0.95`
  * Learned strictly from training data inside an encapsulated `Pipeline`.
* **Primary Classifier:** `LogisticRegression`
  * Regularization: $C=1.0$, L-BFGS solver, `max_iter=1000`
  * Lightweight artifact (~0.4 MB), fast CPU inference, superior calibration.

---

## 5. Dataset Provenance & Preprocessing

* **Raw Source:** 18,650 emails from the curated Hugging Face / Kaggle dataset (`Phishing_Email.csv`).
* **Cleaning Pipeline (`src/preprocessing.py`):**
  * Dropped 19 missing/null rows.
  * Excluded 533 conflicting rows where text `"empty"` appeared with contradictory labels.
  * Dropped 579 duplicate texts with identical labels.
  * Truncated 25 extreme outliers (>50,000 characters).
* **Final Clean Dataset (`data/processed/cleaned_emails.csv`):**
  * **17,519 total rows:** 10,978 Safe emails (62.66%), 6,541 Phishing emails (37.34%).
  * Stratified 80/20 train/test split (`random_state=42`):
    * Training partition: 14,015 emails
    * Held-out test set: 3,504 emails
    * Exact duplicate text overlap between partitions: **0**

---

## 6. Evaluation Audit & Threshold Tuning Results

The held-out test set (3,504 emails: 2,196 Safe, 1,308 Phishing) remained completely sealed during threshold tuning and was evaluated strictly once:

### Held-Out Test Set Performance (Logistic Regression)

| Metric | Baseline (Threshold = 0.50) | Selected (Threshold = 0.60) | Absolute Difference | Operational Impact |
| :--- | :---: | :---: | :---: | :--- |
| **False Positives (FP)** | **26** | **15** | **-11** | **-42.31% false alarms!** |
| **False Positive Rate (FPR)** | **0.0118 (1.18%)** | **0.0068 (0.68%)** | **-0.0050** | **42% lower chance of false alarm** |
| **False Negatives (FN)** | 47 | 83 | +36 | +36 missed threats |
| **Recall (Sensitivity)** | 0.9641 (96.41%) | 0.9365 (93.65%) | -0.0276 | Still catches 93.7% of phishing |
| **Precision** | 0.9798 | 0.9879 | +0.0081 | 98.8% confidence when alert fires |
| **F1-Score** | 0.9719 | 0.9615 | -0.0104 | Strong overall balance (> 0.96) |
| **Accuracy** | 0.9792 | 0.9720 | -0.0072 | Marginal decrease |

*Full audit documentation is available in [`reports/threshold_tuning_report.md`](reports/threshold_tuning_report.md).*

---

## 7. Security Disclaimers & Known Limitations

> [!IMPORTANT]
> **Prototype Disclaimer:** This is an experimental text classifier. A low phishing score does not guarantee that an email is safe. Verify suspicious requests through trusted channels.

1. **Text Pattern Scope:** The application evaluates text phrasing and vocabulary only. It **does not** scan attachments, inspect embedded images, visit destination URLs, or check sender SPF/DKIM authentication headers.
2. **Enron Source Bias:** Legitimate emails in the dataset originate predominantly from the 2001 Enron corporate dataset (the word `"enron"` appears 2,299 times in safe emails and only once in phishing).
3. **Spam Contamination:** The phishing class contains general commercial spam and pump-and-dump scams in addition to spear-phishing.
4. **Template Overlap:** An empirical similarity audit revealed that ~8.5% of test emails share $\ge 85\%$ Jaccard word similarity with training emails due to repeating spam templates.

---

## 8. Repository Structure & Git Hygiene

```
Phishing-Email-Detection/
├── app.py                             # Streamlit interactive application
├── requirements.txt                   # Project dependencies
├── README.md                          # Documentation and setup instructions
├── .gitignore                         # Excludes datasets, virtualenv, and binaries
├── src/
│   ├── dataset.py                     # Dataset validation and checksum helper
│   ├── explore_dataset.py             # Exploratory data analysis module
│   ├── preprocessing.py               # Data cleaning and deduplication pipeline
│   ├── evaluation.py                  # Reproducible split and classification metrics
│   ├── train.py                       # Pipeline training and artifact serialization
│   ├── tune_threshold.py              # Validation threshold tuning engine
│   └── predict.py                     # Model inference and probability scoring
├── tests/
│   ├── test_dataset.py                # Dataset acquisition tests
│   ├── test_explore_dataset.py        # EDA module tests
│   ├── test_preprocessing.py          # Data cleaning pipeline tests
│   ├── test_evaluation.py             # Splitting and metrics tests
│   ├── test_model_pipeline.py         # Pipeline and baseline training tests
│   ├── test_threshold_tuning.py       # Threshold tuning and monotonic property tests
│   └── test_app.py                    # Streamlit app and integration tests
├── models/
│   ├── evaluation_results.json        # Machine-readable baseline metrics
│   ├── threshold_tuning_results.json  # Machine-readable threshold tuning sweeps
│   ├── logistic_regression_pipeline.joblib  # Serialized production artifact
│   └── random_forest_pipeline.joblib        # Serialized benchmark artifact
└── reports/
    └── threshold_tuning_report.md     # In-depth threshold tuning & audit report
```

*All raw datasets (`data/raw/`), processed datasets (`data/processed/`), virtual environments (`.venv/`), and binary models (`*.joblib`) are excluded from Git version control via `.gitignore`.*
