# 🔍 Fake News Detector — NLP Project

> Binary classification of news articles as **Real** or **Fake** using traditional ML and deep learning techniques.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-99.93%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Overview

This project builds a complete end-to-end fake news detection pipeline across **5 phases**, progressing from classical machine learning to state-of-the-art transformer models.

**Dataset:** [ISOT Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
**Total Articles:** 44,898 (21,417 Real + 23,481 Fake)  
**Split:** 70% Train / 15% Validation / 15% Test (stratified)

---

## 🏆 Results Summary

| Phase | Model | Accuracy | F1 Score | ROC-AUC |
|-------|-------|----------|----------|---------|
| Phase 1 | TF-IDF Pipeline | — | — | — |
| Phase 2 | Logistic Regression | 99.14% | 99.10% | 99.94% |
| Phase 3 | XGBoost ⭐ | 99.79% | 99.78% | 99.99% |
| Phase 4 | DistilBERT 🏆 | **99.93%** | **99.92%** | **100%** |

---

## 📁 Project Structure

```
fake_news_detector/
├── data/
│   ├── raw/              ← ISOT dataset (True.csv + Fake.csv)
│   └── processed/        ← Cleaned and combined CSVs
├── src/
│   ├── config.py         ← All settings in one place
│   ├── data/
│   │   ├── ingest.py     ← Load and merge raw CSVs
│   │   ├── preprocess.py ← Clean and normalise text
│   │   └── split_vectorize.py ← Train/Val/Test + TF-IDF
│   ├── models/
│   │   ├── train.py      ← Logistic Regression training
│   │   ├── compare.py    ← Model comparison (RF, XGBoost)
│   │   └── distilbert_model.py ← DistilBERT fine-tuning
│   └── utils/
│       ├── eda.py        ← EDA plots
│       ├── plots.py      ← Evaluation plots
│       └── compare_plots.py ← Model comparison plots
├── outputs/              ← Saved plots and reports
├── pipeline.py           ← Phase 1: Data pipeline
├── pipeline_phase2.py    ← Phase 2: Logistic Regression
├── pipeline_phase3.py    ← Phase 3: Model comparison
├── pipeline_phase4.py    ← Phase 4: DistilBERT
├── app.py                ← Phase 5: Streamlit web app
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/labikakumari2-cmyk/fake-news-detection-nlp.git
cd fake-news-detection-nlp
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
```bash
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset -p data/raw --unzip
```

### 5. Run the pipeline
```bash
# Phase 1: Data pipeline
python pipeline.py

# Phase 2: Logistic Regression
python pipeline_phase2.py

# Phase 3: Model comparison
python pipeline_phase3.py

# Phase 4: DistilBERT (requires GPU)
python pipeline_phase4.py

# Phase 5: Web app
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.13
- **ML:** Scikit-learn, XGBoost
- **Deep Learning:** PyTorch, HuggingFace Transformers (DistilBERT)
- **NLP:** NLTK, TF-IDF
- **Web App:** Streamlit
- **Visualization:** Matplotlib, Seaborn

---

## 📊 Phase Details

### Phase 1 — Data Pipeline
- Loaded and merged ISOT dataset (Real + Fake CSVs)
- Text cleaning: URL removal, HTML stripping, stopword filtering, lemmatization
- TF-IDF vectorization (50,000 vocab, unigrams + bigrams)
- Stratified 70/15/15 train/val/test split

### Phase 2 — Logistic Regression
- Trained on TF-IDF features
- **99.14% test accuracy**, F1: 99.10%, AUC: 99.94%
- Generated confusion matrix, ROC curve, top feature plots

### Phase 3 — Model Comparison
- Compared Logistic Regression, Random Forest, XGBoost
- 5-fold cross-validation
- **XGBoost won** with 99.79% accuracy and 99.99% AUC

### Phase 4 — DistilBERT Fine-Tuning
- Fine-tuned `distilbert-base-uncased` on GPU (Google Colab T4)
- 3 epochs, batch size 32, learning rate 2e-5
- **Best result: 99.93% test accuracy, 100% AUC**

### Phase 5 — Streamlit Web App
- Real-time fake news detection
- Paste any article and get instant Real/Fake prediction
- Confidence scores with visual progress bars

---

## 👩‍💻 Author

**Labika**  
Built as a portfolio project demonstrating end-to-end NLP pipeline development.

---

## 📄 License

This project is licensed under the MIT License.
