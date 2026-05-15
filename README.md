# Bengali NLP: Sentiment & Emotion Analysis

## 📌 Project Overview

End-to-end Bengali NLP project for **Sentiment** and **Emotion** classification using Classical ML — with Deep Learning and Transformer models planned for upcoming phases.

The project covers the full pipeline: raw data ingestion → EDA → text preprocessing → feature extraction → model training & evaluation.

---

## 📊 Dataset

- **Source file:** `Advancing Bengali NLP for Sentiment and Emotion Dataset.xlsx`
- **Raw rows:** 34,812 total observations
- **After deduplication:** 5,917 unique Bengali texts
- **After outlier removal (>300 chars):** 5,635 final samples
- **Columns:** `Text`, `Sentiment`, `Emotion`
- **No missing values** in the dataset

### Label Classes

| Task | Classes |
|---|---|
| **Sentiment** (5) | Negative, Strongly Negative, Neutral, Positive, Strongly Positive |
| **Emotion** (5) | Creepy, Bullying, Unbiased, Joyful, Surprise |

### Class Distribution

**Sentiment** (relatively balanced):
- Negative: 1,384 (23.4%)
- Positive: 1,254 (21.2%)
- Strongly Negative: 1,215 (20.5%)
- Neutral: 1,071 (18.1%)
- Strongly Positive: 993 (16.8%)

**Emotion** (imbalanced):
- Creepy: 1,915 (32.4%)
- Joyful: 1,655 (28.0%)
- Unbiased: 1,351 (22.8%)
- Bullying: 598 (10.1%)
- Surprise: 398 (6.7%)

---

## 🗂️ Project Structure

```
Notebooks/
├── Bengali_NLP_Sentiment_Emotion_EDA.ipynb       ← EDA
└── Bengali_NLP_Preprocessing_Model.ipynb         ← Preprocessing + Classical ML Models
```

---

## 🔬 Notebooks Summary

### 1. `Bengali_NLP_Sentiment_Emotion_EDA.ipynb` — EDA

Key steps performed:
- Dataset loading and initial inspection
- Duplicate removal: 34,812 → 5,917 rows
- Label standardization (whitespace trimming, typo fix: `Joyfull` → `Joyful`)
- Text length & word count distribution analysis (most texts < 100 chars, < 10 words)
- Outlier removal: texts > 300 characters removed (5,917 → 5,635 rows)
- Sentiment vs Emotion cross-tabulation & heatmap analysis
- Bengali Word Cloud visualization (NotoSerifBengali font)

**Key EDA Insight:** Strong semantic alignment between labels — Strongly Negative → Creepy/Bullying; Positive/Strongly Positive → Joyful; Neutral → Unbiased/Surprise.

---

### 2. `Bengali_NLP_Preprocessing_Model.ipynb` — Preprocessing + Classical ML

**Text Preprocessing (`clean_text` function):**
- URL removal (`http`, `www`)
- Emoji removal (Unicode range `\U0001F300–\U0001FFFF`)
- Retain only Bengali Unicode characters (`\u0980–\u09FF`)
- Remove Bengali punctuation (`।`, `৷`) and Bengali numerals (`০–৯`)
- Remove English digits and extra whitespace

**Feature Extraction:**
- `TfidfVectorizer` with `max_features=5000`, `min_df=2`, `ngram_range=(1,2)`
- Output shape: sparse matrix over Bengali bigrams/unigrams

**Models Trained:**

| Task | Model | Notes |
|---|---|---|
| Sentiment | Logistic Regression | `max_iter=1000` |
| Sentiment | LinearSVC | `max_iter=1000` |
| Emotion | Logistic Regression | `class_weight='balanced'` |
| Emotion | LinearSVC | `class_weight='balanced'` |

**Evaluation:**
- Classification report (precision, recall, F1) for all models
- Confusion matrices (heatmaps: Blues for LR, Oranges for SVM)
- Multi-class ROC curves with per-class AUC scores (LR uses `predict_proba`; SVM uses `decision_function`)

---

## 🛠️ Libraries Used

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |
| NLP / Features | `scikit-learn` (TfidfVectorizer, LabelEncoder) |
| ML Models | `scikit-learn` (LogisticRegression, LinearSVC) |
| Evaluation | `scikit-learn` (classification_report, confusion_matrix, roc_curve, auc) |
| Font | NotoSerifBengali (for Bengali Word Cloud) |

---

## 🔜 Upcoming

- [ ] Deep Learning Models (LSTM, BiLSTM)
- [ ] Transformer Models (BanglaBERT)
- [ ] Bengali Stopword Removal
- [ ] Hyperparameter Tuning
- [ ] Model Comparison Dashboard
