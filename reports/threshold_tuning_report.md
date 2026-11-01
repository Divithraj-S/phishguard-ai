# Decision Threshold Tuning & Evaluation Audit Report

**Project Title:** Reducing False Positives in Phishing Email Detection  
**Workspace:** `E:\Phishing-Email-Detection`  
**Dataset:** Cleaned Phishing Email Dataset (`data/processed/cleaned_emails.csv`, 17,519 rows)  
**Date:** October 2026  
**Audience:** Technical & beginner-friendly walkthrough  

---

## 1. Executive Summary

In email security systems, **False Positives** (legitimate emails incorrectly flagged as phishing) cause severe disruptions: important client communications are blocked, urgent internal memos are quarantined, and end-users experience alert fatigue and lose trust in security tools.

This phase had two major objectives:
1. **Implementation & Leakage Audit:** Rigorously verify that all data splitting, TF-IDF feature extraction, and metric calculations (especially False Positive Rate, $FPR = \frac{FP}{FP + TN}$) are mathematically sound and completely free from data leakage.
2. **Decision Threshold Tuning:** Move away from the naive default threshold of $p \ge 0.50$ by tuning the decision threshold on a dedicated **Validation partition** (carved strictly from the training data). The goal was to systematically reduce False Positives while maintaining a high safety bar for threat detection ($\ge 90\%$ Recall).

### Key Empirical Findings

* **Logistic Regression:** Raising the decision threshold from **0.50** to **0.60** reduced False Positives on the unseen held-out test set from **26 down to 15**—a **42.31% relative reduction in false alarms**—while maintaining a strong threat detection Recall of **93.65%** (above our 90% minimum constraint) and boosting Precision to **98.79%**.
* **Random Forest:** Tuning showed that Random Forest's recall dropped below our 90% safety requirement at threshold 0.60 (validation recall: 89.88%), making **0.50** its optimal operating point.
* **Leakage Integrity:** The held-out test partition (3,504 emails) remained 100% sealed during threshold search and was evaluated strictly once at the end.

---

## 2. What is a Decision Threshold and Why Tune It?

### 2.1 How Probabilistic Classifiers Decide
When our machine learning models evaluate an incoming email, they do not simply output "Phishing" or "Safe". Instead, they calculate an estimated probability:
$$P(\text{Phishing} \mid \text{email text}) \in [0.0, 1.0]$$

By default, almost all machine learning libraries apply a default cut-off:
$$\text{If } P(\text{Phishing}) \ge 0.50 \implies \text{Flag as Phishing}$$
$$\text{If } P(\text{Phishing}) < 0.50 \implies \text{Allow as Safe}$$

### 2.2 The Asymmetric Cost of Errors
In the real world, the two types of classification errors have very different consequences:

| Error Type | What Happened | Real-World Impact |
| :--- | :--- | :--- |
| **False Positive (FP)** | A safe email is flagged as phishing | Disrupts business operations, delays urgent contracts or invoices, frustrates employees, damages trust in IT security. |
| **False Negative (FN)** | A phishing email is delivered as safe | Potential credential theft, malware infection, or financial wire fraud. |

### 2.3 Why 0.50 is Rarely Optimal
A default threshold of 0.50 assumes that both errors have equal severity. However:
* In corporate email environments where legitimate emails vastly outnumber phishing emails, even a small 1% False Positive Rate can generate hundreds of quarantined emails every week.
* By adjusting the threshold upward (e.g. to 0.60), we demand **higher model certainty** before sounding the alarm. This weeds out borderline safe emails that merely had slightly formal or persuasive wording, without requiring changes to the model architecture.

---

## 3. Evaluation Audit Findings

Prior to running threshold tuning, we conducted a systematic audit of the existing codebase (`src/train.py`, `src/predict.py`, `src/evaluation.py`, and `tests/`).

### 3.1 Verification of Data Splits
* **Single Source of Truth:** Both models utilize `create_reproducible_split()` from `src/evaluation.py` with `random_state=42` and `test_size=0.20`.
* **Exact Partition Sizes:** Full dataset = 17,519 rows $\rightarrow$ Training set = 14,015 rows (80%), Held-out test set = 3,504 rows (20%).
* **Stratification:** Class balance was preserved identically in both partitions:
  * Full dataset: 62.66% Safe (10,978), 37.34% Phishing (6,541).
  * Training partition: 62.66% Safe (8,782), 37.34% Phishing (5,233).
  * Testing partition: 62.67% Safe (2,196), 37.33% Phishing (1,308).
* **Exact Duplicate Overlap:** 0 exact duplicate texts overlap between train and test partitions.

### 3.2 Verification of Data Leakage Controls
* **Feature Extraction:** TF-IDF vectorization is encapsulated inside a scikit-learn `Pipeline`.
* **Fitting Verification:** The vectorizer vocabulary is learned strictly when `pipeline.fit(X_train, y_train)` is executed. Test data is only transformed during inference. Unseen words in the test set never enter the model's vocabulary (verified by `test_tfidf_pipeline_fits_only_on_training_data`).

### 3.3 Verification of Confusion Matrix and Metric Calculations
* **Confusion Matrix Layout:** Explicitly verified with `labels=[0, 1]`:
  $$\begin{bmatrix} TN & FP \\ FN & TP \end{bmatrix}$$
  where Negative class ($0$) = Safe Email and Positive class ($1$) = Phishing Email.
* **Metric Formulas:**
  * $\text{FPR} = \frac{FP}{FP + TN} = \frac{FP}{\text{Actual Legitimate Emails}}$
  * $\text{Precision} = \frac{TP}{TP + FP}$
  * $\text{Recall (Sensitivity)} = \frac{TP}{TP + FN}$
  * $\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
* **Edge Case Handling:** Safe zero-division handling was confirmed for cases where true negatives or positive predictions are zero.

### 3.4 Discrepancy Found and Resolved in `src/predict.py`
* **Finding:** In `src/predict.py`, the threshold validation check was written as:
  ```python
  if not (0.0 < threshold < 1.0):
      raise ValueError(...)
  ```
  This strictly prohibited threshold values of exactly `0.0` or `1.0`.
* **Fix Applied:** Changed to inclusive boundaries `0.0 <= threshold <= 1.0` to permit boundary testing and extreme operating profiles (e.g., maximum permissiveness vs. total quarantine).

---

## 4. Methodology & Data Leakage Safeguards for Threshold Tuning

### 4.1 The Three-Way Partition Design
A common amateur mistake is tuning decision thresholds directly on the test set. Doing so causes **optimism bias** (data leakage), because the threshold is fitted to the random noise of the test partition.

To prevent this, we implemented a three-tier partitioning scheme:
```
Clean Dataset (17,519 rows)
│
├── Training Partition (14,015 rows, 80%)
│   ├── Sub-Training Set (11,212 rows, 80% of train) ──> Used to fit Pipeline (TF-IDF + Model)
│   └── Validation Set    (2,803 rows, 20% of train) ──> Used EXCLUSIVELY to tune & select threshold
│
└── Held-Out Test Set     (3,504 rows, 20% of total) ──> SEALED; evaluated strictly ONCE at the end
```

### 4.2 Candidate Thresholds Evaluated
We tested 8 candidate decision thresholds:
$$\theta \in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]$$

### 4.3 Predefined Selection Rule
To eliminate subjective cherry-picking, the selection rule was programmatically defined prior to evaluation:
$$\text{Objective: Minimize } FPR \quad \text{subject to } \text{Recall} \ge 90.0\%$$
* If multiple thresholds satisfy $\text{Recall} \ge 90\%$, choose the one with the lowest False Positive Rate.
* If FPR is tied, break the tie using highest F1-Score.

---

## 5. Validation Tuning Results

Both models were fitted on the **Sub-Training Set** (11,212 emails) and evaluated across all candidate thresholds on the **Validation Set** (2,803 emails: 1,756 Safe, 1,047 Phishing).

### 5.1 Logistic Regression (Validation Set)

| Threshold | FPR | False Positives | Recall | False Negatives | Precision | F1-Score | Accuracy | Meets $\text{Rec} \ge 90\%$? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.20 | 0.0957 | 168 | **0.9981** | 2 | 0.8615 | 0.9248 | 0.9394 | Yes |
| 0.30 | 0.0484 | 85 | 0.9914 | 9 | 0.9243 | 0.9567 | 0.9665 | Yes |
| 0.40 | 0.0171 | 30 | 0.9876 | 13 | 0.9718 | **0.9796** | **0.9847** | Yes |
| 0.50 | 0.0097 | 17 | 0.9723 | 29 | 0.9836 | 0.9779 | 0.9836 | Yes |
| **0.60** ⭐ | **0.0051** | **9** | **0.9398** | **63** | **0.9909** | **0.9647** | **0.9743** | **Yes (Selected)** |
| 0.70 | 0.0034 | 6 | 0.8863 | 119 | 0.9936 | 0.9369 | 0.9554 | No (Rec < 90%) |
| 0.80 | 0.0011 | 2 | 0.7708 | 240 | 0.9975 | 0.8696 | 0.9137 | No (Rec < 90%) |
| 0.90 | 0.0006 | 1 | 0.5559 | 465 | 0.9983 | 0.7141 | 0.8337 | No (Rec < 90%) |

> **Selected for Logistic Regression:** Threshold **0.60**.  
> At 0.60, False Positives dropped from 17 down to 9 (a 47% drop on validation) while threat detection remained at 93.98%. Moving to 0.70 was rejected because Recall dropped to 88.63%, violating the 90% safety constraint.

---

### 5.2 Random Forest (Validation Set)

| Threshold | FPR | False Positives | Recall | False Negatives | Precision | F1-Score | Accuracy | Meets $\text{Rec} \ge 90\%$? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.20 | 0.2158 | 379 | **0.9990** | 1 | 0.7340 | 0.8463 | 0.8644 | Yes |
| 0.30 | 0.0974 | 171 | 0.9952 | 5 | 0.8590 | 0.9221 | 0.9372 | Yes |
| 0.40 | 0.0530 | 93 | 0.9828 | 18 | 0.9171 | 0.9488 | 0.9604 | Yes |
| **0.50** ⭐ | **0.0239** | **42** | **0.9599** | **42** | **0.9599** | **0.9599** | **0.9700** | **Yes (Selected)** |
| 0.60 | 0.0091 | 16 | 0.8988 | 106 | 0.9833 | 0.9391 | 0.9565 | No (Rec: 89.88% < 90%) |
| 0.70 | 0.0023 | 4 | 0.7851 | 225 | 0.9952 | 0.8777 | 0.9183 | No (Rec < 90%) |
| 0.80 | 0.0006 | 1 | 0.6294 | 388 | 0.9985 | 0.7721 | 0.8612 | No (Rec < 90%) |
| 0.90 | 0.0000 | 0 | 0.4002 | 628 | 1.0000 | 0.5716 | 0.7760 | No (Rec < 90%) |

> **Selected for Random Forest:** Threshold **0.50**.  
> At threshold 0.60, Random Forest achieved Recall of 89.88%, which narrowly failed the 90.0% constraint. Therefore, 0.50 is the optimal threshold for Random Forest under this policy.

---

## 6. Held-Out Test Set Verification

Once thresholds were locked in based strictly on the Validation set, the production models (trained on the full 14,015 training emails) were evaluated **once** on the untouched **Held-Out Test Set** (3,504 emails: 2,196 Safe, 1,308 Phishing).

### 6.1 Logistic Regression: Baseline (0.50) vs. Selected Threshold (0.60)

```
Test Set Confusion Matrices:
Baseline (p >= 0.50):           Selected Threshold (p >= 0.60):
[[TN=2170,  FP=26],             [[TN=2181,  FP=15],
 [FN=47,    TP=1261]]            [FN=83,    TP=1225]]
```

| Metric | Baseline (0.50) | Selected (0.60) | Absolute Difference | Relative Impact |
| :--- | :---: | :---: | :---: | :--- |
| **False Positives (FP)** | **26** | **15** | **-11** | **-42.31% false alarms!** |
| **False Positive Rate (FPR)** | **0.0118 (1.18%)** | **0.0068 (0.68%)** | **-0.0050** | **42% lower chance of false alarm** |
| **False Negatives (FN)** | 47 | 83 | +36 | +36 missed threats |
| **Recall (Sensitivity)** | 0.9641 (96.41%) | 0.9365 (93.65%) | -0.0276 | Catches 93.7% of all phishing |
| **Precision** | 0.9798 | 0.9879 | +0.0081 | Higher confidence when alert fires |
| **F1-Score** | 0.9719 | 0.9615 | -0.0104 | Balanced score remains > 0.96 |
| **Accuracy** | 0.9792 | 0.9720 | -0.0072 | Slight drop due to missed threats |

### 6.2 Key Takeaways from the Test Set
1. **False Positives cut almost in half:** In a production mailbox processing thousands of emails, dropping the False Positive Rate from 1.18% to 0.68% prevents 5 false alarms for every 1,000 legitimate emails handled.
2. **Safe Threat Catch Rate:** Even with the higher threshold, the model catches **1,225 out of 1,308 phishing attacks (93.65%)**, well exceeding our 90% requirement.
3. **Random Forest Comparison:** Random Forest at threshold 0.50 yielded 50 False Positives on the test set (FPR = 2.28%), making Logistic Regression at 0.60 vastly superior (15 False Positives, FPR = 0.68%).

---

## 7. The Operational Security Trade-Off

Threshold selection is not merely a mathematical exercise; it reflects an **organizational policy choice**:

```
<--- LOWER THRESHOLD (e.g. 0.30)           HIGHER THRESHOLD (e.g. 0.70) --->
     Max Threat Paranoia                         Max Business Continuity
     - Fewest missed phishing (FN ↓)             - Fewest false alarms (FP ↓)
     - High false alarms (FP ↑)                  - More missed phishing (FN ↑)
     - Blocks legitimate business                - Slips into inboxes
```

* **Aggressive / High Security (Threshold = 0.30 - 0.40):** Best suited for high-risk targets (e.g., finance departments handling wire transfers). Every suspicious email is blocked, even if it creates false alarms.
* **Balanced / Standard Corporate (Threshold = 0.50):** The traditional default trade-off.
* **Low False-Positive Focus (Threshold = 0.60):** **Recommended for general enterprise mailboxes.** Cuts false positive disruptions by 42% while preserving 93.7% threat detection.
* **Permissive (Threshold = 0.70+):** Too risky; missed attacks rise above 10%.

---

## 8. Dataset Limitations & Caveats

To ensure responsible machine learning practices, these empirical findings must be interpreted alongside verified dataset biases:

1. **Enron Legitimate Bias:** The safe emails originate predominantly from the 2001 Enron corporate dataset (the word `"enron"` appeared 2,299 times in safe emails and only once in phishing). Words specific to corporate oil trading and 2000s business culture may artificially inflate the model's confidence on modern business emails.
2. **Spam Contamination:** The phishing class includes general non-targeted commercial spam and scams alongside credential harvesting phishing.
3. **Template Overlap:** Our empirical Jaccard similarity audit found that 8.5% of test emails share $\ge 85\%$ word similarity with training emails due to syndicated spam campaign templates.

**Production Disclaimer:** These models are research prototypes and probabilistic indicators. A low risk score does NOT guarantee that an email is safe.

---

## 9. Recommendations for Phase 9 (Streamlit Application)

Based on these findings, we recommend the following design for the interactive Streamlit user interface:

1. **Default Model:** Make **Logistic Regression** the primary recommended model (lighter artifact, faster inference, and superior FPR: 0.68% vs 2.28%).
2. **Interactive Threshold Slider:** Provide a user slider for the decision threshold from `0.20` to `0.90` (defaulting to `0.60` with a label *"Recommended: Low False Positives"*).
3. **Threshold Presets:**
   * *"Strict Anti-Phishing"* ($0.40$) — for high-risk accounts.
   * *"Balanced Default"* ($0.50$) — standard ML baseline.
   * *"Reduced False Positives"* ($0.60$) — recommended by Phase 8 audit.
4. **Transparent Risk Explanation:** Display both the estimated probability and the current decision threshold, along with the permanent security disclaimer.
