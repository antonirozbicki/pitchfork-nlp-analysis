# Pitchfork Music Reviews - NLP Analysis and Score Prediction

A complete NLP pipeline for analyzing Pitchfork music reviews: from information extraction and keyword analysis to score prediction and model explainability.

## Overview

This project investigates whether the text of a music review can predict the numerical score (0-10) assigned by the reviewer. The pipeline combines traditional NLP methods with transformer-based sentiment analysis and applies Explainable AI (XAI) techniques to interpret the model's predictions.

### Key components

1. **Text preprocessing** - HTML cleanup, tokenization, lemmatization, stopword removal
2. **Information Extraction (CRF)** - Sequence labelling for ARTIST, GENRE, INSTRUMENT, TECHNIQUE, DATE entities using silver annotations and a Conditional Random Field model
3. **Keyword Extraction (YAKE)** - Unsupervised keyword extraction for topical analysis
4. **Sentiment Analysis** - Zero-shot transfer using DistilBERT (SST-2) as a predictive feature
5. **Score Prediction**
   - Binary classification (positive >= 7 vs negative): Ridge Logistic Regression, Linear SVM, Random Forest, LASSO Logistic Regression
   - Regression (exact score): LASSO vs Ridge
6. **Explainability (XAI)** - LASSO coefficient analysis revealing which tokens drive predictions

### Results

**Binary classification** (positive >= 7 vs negative):

| Model | Accuracy | F1-score |
|-------|----------|----------|
| Ridge Logistic Regression | 76.3% | 81.2% |
| Linear SVM | 75.7% | 80.9% |
| LASSO Logistic Regression | 72.1% | 60.8% |
| Random Forest | 69.5% | 76.2% |

**Regression** (exact score prediction, 0-10 scale):

| Model | MAE | RMSE | Accuracy +/-1.0 | Non-zero features |
|-------|-----|------|-----------------|-------------------|
| Ridge | 0.737 | 1.010 | 74.0% | 3,030,929 |
| LASSO (alpha=0.01) | 0.780 | 1.073 | 72.2% | 279 |

## Dataset

Pitchfork music reviews from the public Kaggle repository:
https://www.kaggle.com/datasets/nolanbconaway/pitchfork-data

Download `database.sqlite` and place it in the project root directory.

## Setup

```bash
pip install -r requirements.txt
```

Then open and run `Pitchfork_music_reviews.ipynb` from top to bottom.

**Note:** Sentiment inference with DistilBERT runs on CPU by default and takes approximately 10-15 minutes on the full dataset. LASSO regression (alpha=0.01) with 3M+ features may take several minutes to converge.

## Project structure

```
.
├── Pitchfork_music_reviews.ipynb   # Main notebook (all analysis)
├── database.sqlite                  # Pitchfork reviews SQLite database (not tracked)
├── requirements.txt
├── .gitignore
└── README.md
```

## Author

Antoni Rozbicki
