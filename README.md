# 📰 Fake News Detector

A complete AI/ML project to **train, validate, and test** a fake news classifier, with a **Streamlit web app** for live predictions.

---

## 📁 Project Structure

```
fake-news-detector/
├── requirements.txt      # Python dependencies
├── model.py              # Model architecture (TF-IDF + Logistic Regression)
├── train.py              # Training & cross-validation script
├── test.py               # Test set evaluation script
├── app.py                # Streamlit web app
├── model/
│   └── fake_news_model.pkl   # Trained model (generated after training)
├── dataset/
│   ├── Fake.csv              # ← KAGGLE FAKE NEWS FILE
│   └── True.csv              # ← KAGGLE REAL NEWS FILE
└── results/
    ├── training_validation_results.png  # Training plots
    ├── test_evaluation_results.png      # Test plots
    ├── test_report.txt                  # Test metrics report
    └── test_data.pkl                    # Saved test data
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
cd fake-news-detector
pip install -r requirements.txt
```

### Step 2: Adding Kaggle Dataset

Place Kaggle dataset files in the `dataset/` folder:

```
fake-news-detector/dataset/
├── Fake.csv    ← From Kaggle (fake news articles)
└── True.csv    ← From Kaggle (real news articles)
```

Your CSV should have these columns:
| Column | Description |
|--------|-------------|
| `title` | Article headline |
| `text` | Full article body |
| `subject` | Topic/category (optional) |
| `date` | Publication date (optional) |

> The code automatically combines `title + text` into a single `content` column and adds `label` (1=Fake, 0=Real).

### Step 3: Train the Model

```bash
python train.py
```

This will:
- ✅ Loading Fake.csv + True.csv
- 📊 Split: 60% Train / 20% Validation / 20% Test
- 🧠 Train TF-IDF + Logistic Regression
- 🔄 Run 5-Fold Cross-Validation
- 📈 Generate training visualizations
- 💾 Save model to `model/fake_news_model.pkl`

### Step 4: Evaluate on Test Set

```bash
python test.py
```

This will:
- 🧪 Evaluate on the held-out test set
- 📊 Generate comprehensive test visualizations
- 📝 Save detailed report to `results/test_report.txt`

### Step 5: Launch Streamlit App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`
(Navigate to :
Local URL: http://localhost:8501
  Network URL: http://192.168.98.57:8501 )

---

## 📊 What Each Script Shows

### train.py — Training & Validation
| Output | Description |
|--------|-------------|
| Dataset loading | Shows Fake & Real article counts |
| Data split | Train/Val/Test sample counts |
| Accuracy, Precision, Recall, F1 | On validation set |
| Classification Report | Per-class metrics |
| 5-Fold CV Scores | Bar chart with mean ± std |
| Confusion Matrix | Heatmap of predictions |
| ROC Curve | AUC score visualization |
| Training Plot | `results/training_validation_results.png` |

### test.py — Test Set Evaluation
| Output | Description |
|--------|-------------|
| Full Metrics Table | Accuracy, Precision, Recall, F1, ROC AUC, AP |
| Confusion Matrix | Raw counts + row-normalized % |
| ROC Curve | Test set AUC |
| Precision-Recall Curve | Average Precision |
| Confidence Distribution | Histogram of prediction probabilities |
| Test Plot | `results/test_evaluation_results.png` |
| Text Report | `results/test_report.txt` |

### app.py — Streamlit Web App
| Feature | Description |
|---------|-------------|
| Live Prediction | Paste any text → instant Fake/Real verdict |
| Confidence Bars | Visual bars for Real vs Fake probability |
| Probability Chart | Matplotlib bar chart |
| History | Tracks recent predictions |
| Quick Examples | Pre-loaded fake & real sample texts |

---

## 🔬 Model Architecture

```
Input Text (title + text)
    ↓
[Text Cleaning]
  - Lowercase, remove URLs, HTML, special chars
    ↓
[TF-IDF Vectorizer]
  - Max 50,000 features
  - Unigrams + Bigrams (ngram_range=(1,2))
  - Sublinear TF scaling
  - min_df=2, max_df=0.95
    ↓
[Logistic Regression]
  - C=1.0, balanced class weights
  - LBFGS solver, 1000 max iterations
    ↓
Output: Real (0) or Fake (1) + probability scores
```

---

## 📈 Expected Results (with real Kaggle dataset ~44K articles)

| Metric | Validation | Test |
|--------|------------|------|
| Accuracy | ~97-99% | ~97-99% |
| Precision | ~97-99% | ~97-99% |
| Recall | ~97-99% | ~97-99% |
| F1-Score | ~97-99% | ~97-99% |
| ROC AUC | ~99%+ | ~99%+ |

---

## 🎓 Demo Flow (for showing in class)

```
1. python train.py
   → Shows: data loading → train/val split → training → CV → plots

2. python test.py
   → Shows: full test metrics → confusion matrix → ROC curve → report

3. streamlit run app.py
   → Shows: live web interface with real-time predictions

4. In Streamlit:
   → Paste fake news text → see FAKE prediction + confidence
   → Paste real news text → see REAL prediction + confidence
   → Try "Quick Examples" buttons in sidebar
```

---

## ⚙️ Customizing

### Different column names?
Edit `train.py` → `load_dataset()` → change `df['title']` and `df['text']` to your column names.

### Use a single combined CSV (instead of 2 files)?
Edit `train.py` → `load_dataset()` to read one CSV with a `label` column (0=Real, 1=Fake).

### Switch model?
Edit `model.py` → `_create_pipeline()`:
```python
# SVM:
('clf', SVC(probability=True, class_weight='balanced'))

# Random Forest:
('clf', RandomForestClassifier(n_estimators=100, class_weight='balanced'))

# Neural Network:
# Uncomment TensorFlow in requirements.txt, use Keras/TensorFlow
```

### Tune hyperparameters?
Edit `model.py` → `_create_pipeline()`:
```python
# TF-IDF: max_features, ngram_range, min_df, max_df
# Logistic Regression: C, class_weight, solver
```

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "Dataset not found" | Place Fake.csv + True.csv in `dataset/` folder |
| "No module named 'streamlit'" | `pip install streamlit` |
| Out of memory | Reduce `max_features` in `model.py` (e.g., 10000) |
| Low accuracy | Try more `max_features` or use a neural model |
| Kaggle download fails | Use dropbox/google drive links for the ISOT dataset |

---

## 📝 Dataset Sources

- **ISOT Fake News Dataset** (recommended): [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
  - ~23,000 Fake articles + ~21,000 Real articles
  - Columns: title, text, subject, date
