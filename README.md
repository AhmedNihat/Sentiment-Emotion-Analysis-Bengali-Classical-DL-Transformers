# Bengali NLP: Sentiment & Emotion Analysis

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Classical%20ML-orange)
![Status](https://img.shields.io/badge/Status-Phase%202%20Complete-green)
![Next](https://img.shields.io/badge/Next-BanglaBERT-purple)

## 📌 Project Overview

End-to-end Bengali NLP pipeline for **Sentiment** (5-class) and **Emotion** (5-class) classification.  
Built on a real-world Bengali social media dataset with full EDA, text preprocessing, feature extraction, and classical ML modeling.

**Current Phase:** Classical ML baseline complete (Logistic Regression + LinearSVC)  
**Next Phase:** Deep Learning & BanglaBERT fine-tuning

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | `Advancing Bengali NLP for Sentiment and Emotion Dataset.xlsx` |
| Raw rows | 34,812 |
| After deduplication | 5,917 |
| After outlier removal (>300 chars) | **5,635 (final)** |
| Columns | `Text`, `Sentiment`, `Emotion` |
| Missing values | None |

### Class Distribution

**Sentiment** — relatively balanced:

| Class | Count | % |
|---|---|---|
| Negative | 1,384 | 23.4% |
| Positive | 1,254 | 21.2% |
| Strongly Negative | 1,215 | 20.5% |
| Neutral | 1,071 | 18.1% |
| Strongly Positive | 993 | 16.8% |

**Emotion** — imbalanced (requires special handling):

| Class | Count | % |
|---|---|---|
| Creepy | 1,915 | 32.4% |
| Joyful | 1,655 | 28.0% |
| Unbiased | 1,351 | 22.8% |
| Bullying | 598 | 10.1% |
| Surprise | 398 | 6.7% ⚠️ |

---

## 🗂️ Project Structure

```
Bengali-NLP-Sentiment-Emotion/
│
├── Notebooks/
│   ├── Bengali_NLP_Sentiment_Emotion_EDA.ipynb        ← Phase 1: EDA
│   └── Bengali_NLP_Preprocessing_Model.ipynb          ← Phase 2: Preprocessing + Classical ML
│
├── data/
│   └── Advancing Bengali NLP for Sentiment and Emotion Dataset.xlsx
│
└── README.md
```

---

## 🔬 Pipeline

### Phase 1 — EDA (`Bengali_NLP_Sentiment_Emotion_EDA.ipynb`)

- Dataset loading & structure inspection
- Duplicate removal: 34,812 → 5,917 rows
- Label standardization (`Joyfull` → `Joyful`, whitespace cleanup)
- Text length & word count distribution (most texts < 100 chars, < 10 words)
- Outlier removal: texts > 300 characters (5,917 → 5,635)
- Sentiment vs Emotion cross-tabulation & heatmap
- Bengali Word Cloud (NotoSerifBengali font)

**Key Finding:** Strong semantic alignment — Strongly Negative → Creepy/Bullying; Positive → Joyful; Neutral → Unbiased/Surprise.

---

### Phase 2 — Preprocessing + Classical ML (`Bengali_NLP_Preprocessing_Model.ipynb`)

**Text Cleaning (`clean_text`):**
- Remove URLs, emojis, English characters/digits
- Retain only Bengali Unicode (`\u0980–\u09FF`)
- Remove Bengali punctuation (`।`, `৷`) and Bengali numerals (`০–৯`)
- Normalize whitespace

**Feature Extraction:**
- `TfidfVectorizer`: `max_features=5000`, `min_df=2`, `ngram_range=(1,2)`

**Models:** Logistic Regression, LinearSVC (both with `class_weight='balanced'`)

---

## 📈 Results

### Emotion Classification

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Logistic Regression | 0.64 | **0.58** | 0.65 |
| LinearSVC | 0.64 | 0.57 | 0.64 |

**Per-class F1 — Emotion:**

| Emotion | Support | LR F1 | SVM F1 |
|---|---|---|---|
| Creepy | 359 | **0.78** | 0.77 |
| Joyful | 323 | 0.70 | **0.72** |
| Unbiased | 252 | **0.55** | 0.54 |
| Bullying | 116 | **0.45** | 0.42 |
| Surprise | 77 | **0.41** | 0.40 |

**ROC-AUC — Emotion (Logistic Regression):**

| Emotion | AUC |
|---|---|
| Creepy | 0.92 |
| Joyful | 0.90 |
| Bullying | 0.84 |
| Surprise | 0.83 |
| Unbiased | 0.81 |

> **Note:** Macro F1 = 0.58 is consistent with Bengali NLP literature for classical ML baselines on multi-class emotion tasks. Minority classes (Surprise, Bullying) suffer from data scarcity — a known limitation of TF-IDF + classical ML on low-resource languages.

---

## 🛠️ Libraries Used

| Category | Libraries |
|---|---|
| Data | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |
| Features | `scikit-learn` — TfidfVectorizer, LabelEncoder |
| Models | `scikit-learn` — LogisticRegression, LinearSVC |
| Evaluation | `scikit-learn` — classification_report, confusion_matrix, roc_curve, auc |

---

## 🔜 Next Steps (Planned)

### Phase 3 — Stronger Classical ML Baseline
- [ ] **Bengali stopword removal** — stopwords currently dominate TF-IDF features, hurting Unbiased class most
- [ ] **SMOTE / oversampling** — fix Surprise (6.7%) and Bullying (10.1%) minority class imbalance
- [ ] **Add Random Forest + XGBoost** — complete the classical ML comparison
- [ ] **5-fold cross-validation** — replace single 80/20 split for reliable evaluation
- [ ] **Hyperparameter tuning** — GridSearchCV on C, ngram_range, max_features

### Phase 4 — Deep Learning
- [ ] **Bengali fastText embeddings** — richer word representations than TF-IDF
- [ ] **BiLSTM** — capture sequential context in Bengali sentences
- [ ] **CNN + BiLSTM hybrid** — local + sequential feature extraction

### Phase 5 — Transformer (Main Goal)
- [ ] **BanglaBERT fine-tuning** — expected to push Macro F1 from 0.58 → 0.75+
- [ ] **Multi-task learning** — train Sentiment + Emotion jointly (semantically correlated)
- [ ] **Full benchmark table** — unified comparison across all phases

---

## ⚠️ Known Limitations

- No Bengali stopword removal applied before modeling (planned Phase 3)
- Single train/test split used — results may vary with cross-validation
- Minority classes (Surprise F1=0.41, Bullying F1=0.45) need oversampling
- TF-IDF cannot capture morphological richness of Bengali — transformer needed for ceiling performance
