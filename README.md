# Bengali NLP - Sentiment & Emotion Classification

A classical ML pipeline for multi-class sentiment and emotion classification on Bengali social media text. The project covers raw data inspection, duplicate-aware cleaning, Unicode-correct preprocessing, TF-IDF feature extraction with a custom Bengali tokenizer, and baseline classifier evaluation.

---

## Repository Structure

```
.
├── Notebooks/
│   ├── Bengali_NLP_Sentiment_Emotion_EDA.ipynb       # Data cleaning & EDA
│   └── Bengali_NLP_Preprocessing_Model.ipynb         # Preprocessing & Baseline Models
├── data/
│   └── Advancing Bengali NLP for Sentiment and Emotion Dataset.xlsx
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Raw rows | 34,812 |
| Columns | `Text`, `Sentiment`, `Emotion` |
| Missing values | None |
| After deduplication | 5,917 |
| After outlier removal (> 300 chars) | **5,635** |

**Sentiment distribution** (relatively balanced):

| Class | Count | Share |
|---|---|---|
| Negative | 1,384 | 23.4% |
| Positive | 1,254 | 21.2% |
| Strongly Negative | 1,215 | 20.5% |
| Neutral | 1,071 | 18.1% |
| Strongly Positive | 993 | 16.8% |

**Emotion distribution** (imbalanced — handled via `class_weight='balanced'`):

| Class | Count | Share |
|---|---|---|
| Creepy | 1,915 | 32.4% |
| Joyful | 1,655 | 28.0% |
| Unbiased | 1,351 | 22.8% |
| Bullying | 598 | 10.1% |
| Surprise | 398 | 6.7% |

---

## Methodology

### Notebook 1 : Data Cleaning & EDA

| Step | Action |
|---|---|
| Deduplication | Removed 28,895 duplicate texts; retained 5,917 unique rows |
| Label fix | Corrected `"Joyfull"` → `"Joyful"`; stripped whitespace from all labels |
| Outlier removal | Texts > 300 characters removed (95th percentile ≈ 288 chars); 282 rows dropped |
| Distribution analysis | Bar charts + pie charts for sentiment and emotion class frequencies |
| Sentiment–Emotion heatmap | Cross-tabulation reveals strong label alignment (e.g., Strongly Negative → Creepy 81%) |
| Word cloud | Bengali word cloud using NotoSerifBengali font; negation token `না` retained intentionally |
| Export | Cleaned data saved as `clean_data.csv` for the modelling notebook |

### Notebook 2 : Preprocessing & Baseline Models

| Step | Action |
|---|---|
| Noise audit | Detected URLs, emojis, and non-Bengali symbols before cleaning |
| Text cleaning | Removed URLs, emojis, non-Bengali characters, punctuation, and digits |
| Custom tokenizer | Regex-based Bengali Unicode tokenizer — bypasses scikit-learn's ASCII `\b` boundary limitation |
| TF-IDF | 8,000 features · `min_df=2` · unigrams + bigrams |
| Stopword audit | Negation tokens (`না`, `নয়`, `নেই`) confirmed present and retained — critical sentiment signals |
| Train/test split | Stratified 80/20 · `random_state=42` |
| Cross-validation | 5-fold Stratified K-Fold on training set |
| Models | Logistic Regression · LinearSVC |
| Evaluation | Weighted F1 · Accuracy · Confusion Matrix · ROC-AUC (OvR) |

---

## Results

### Sentiment Classification (5-class)

| Model | CV Mean F1 | CV Std | Test F1 | Test Accuracy |
|---|---|---|---|---|
| **Logistic Regression** | **0.5937** | 0.0146 | **0.60** | **0.60** |
| LinearSVC | 0.5849 | 0.0154 | 0.57 | 0.57 |

**Best model: Logistic Regression**

- `Strongly Negative` — best class (F1: 0.77, AUC: 0.94); distinct vocabulary makes it linearly separable.
- `Neutral` — hardest class (F1: 0.48); lacks strong lexical signal.
- Intensity-level confusion (`Positive` vs `Strongly Positive`) is expected — TF-IDF cannot encode degree of affect.

### Emotion Classification (5-class)

| Model | CV Mean F1 | CV Std | Test F1 | Test Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.6682 | **0.0065** | 0.66 | 0.65 |
| **LinearSVC** | **0.6741** | 0.0124 | **0.66** | **0.66** |

**Best model: LinearSVC** (marginal; LR is more stable — lower CV std)

- `Creepy` — best class (AUC: 0.92); largest class with most training data.
- `Bullying` / `Surprise` — weakest (F1: 0.45); limited by data volume, not model capacity.
- `Surprise` AUC (0.87) vs F1 (0.45) gap indicates ranking ability without reliable calibration.

---

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| Custom Bengali tokenizer | scikit-learn's default `\b` boundary is undefined for Unicode scripts; fails silently on Bengali |
| Negation tokens retained | `না`, `নয়`, `নেই` are primary sentiment polarity signals — stopword removal would degrade performance |
| 300-char outlier threshold | Set at 95th percentile (≈ 288 chars); removes structurally atypical texts with < 5% data loss |
| `class_weight='balanced'` on emotion only | Sentiment classes are near-uniform (16–23%); emotion is skewed (6–32%) — weighting only where justified |

---

## Limitations & Future Work

- TF-IDF cannot capture word order or semantic intensity — the primary source of intensity-class confusion.
- `Surprise` and `Bullying` performance is data-volume constrained, not model constrained.
- Mixed-script (code-switched) tokens are discarded by the current cleaning function.

**Recommended next steps:**

- Fine-tune `sagorsarker/bangla-bert-base` or `csebuetnlp/banglabert` for contextual embeddings.
- Multi-task learning with a shared encoder and dual classification heads (sentiment + emotion jointly).
- Back-translation or paraphrasing for minority class augmentation (`Surprise`, `Bullying`).
- Morphological stemming via `bnlp_toolkit` to reduce vocabulary fragmentation.

---

## Dependencies

`pandas` · `numpy` · `matplotlib` · `seaborn` · `scikit-learn` · `bnlp_toolkit` · `wordcloud` · `nltk`

> Notebooks were developed in Google Colab. Run `Bengali_NLP_Sentiment_Emotion_EDA.ipynb` first — it produces `clean_data.csv` required by the modelling notebook.
