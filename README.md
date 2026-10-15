# MasakhaNEWS Kiswahili News Topic Classification

**Capstone Project – Machine Learning Techniques I**  
**Task**: Text Classification for an African Language (Kiswahili)  
**Dataset**: MasakhaNEWS (Kiswahili / `swa` split only)  
**Author**: Francis Mutabazi  
**Date**: October 2026

---

## Project Overview

This project builds an end-to-end **Kiswahili news topic classifier**.  
Given a Kiswahili news article (headline + body), the system automatically assigns one of seven topics:

| Topic | Description |
|-------|-------------|
| business | Economy, markets, companies, finance |
| entertainment | Music, film, celebrity, culture |
| health | Medicine, public health, disease, hospitals |
| politics | Government, elections, policy, parties |
| religion | Churches, mosques, faith events |
| sports | Football, athletics, competitions |
| technology | Gadgets, internet, telecom, innovation |

The work uses only real, naturally occurring data from the MasakhaNEWS benchmark. No synthetic text was generated.

---

## Live Demo

**Permanent public application**  
→ https://kiswahili-news-classifier.streamlit.app/

Paste any Kiswahili news text and receive the predicted topic together with decision scores.

---

## Key Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| TF-IDF + Logistic Regression (Baseline) | 0.8277 | **0.8025** | 0.8301 |
| **TF-IDF + Linear SVM** (deployed) | 0.8340 | 0.7961 | **0.8343** |
| Afro-XLMR-mini (Fine-tuned) | **0.8403** | 0.7929 | 0.8318 |

**Selected model for deployment**: TF-IDF + Linear SVM  
Reason: highest weighted F1, near-best accuracy, extremely fast training and inference, and low memory footprint — ideal for a practical web service.

---

## Dataset at a Glance

- **Source**: Masakhane – MasakhaNEWS (Hugging Face)
- **Language**: Kiswahili only (`swa` configuration)
- **Size**: 2 371 articles  
  - Train: 1 658  
  - Validation: 237  
  - Test: 476
- **Class distribution (train)**: sports / health / politics (350 each), business (221), religion (204), technology (115), entertainment (68)

---

## How to Run Locally

```bash
# Create environment (Python ≥ 3.10 recommended)
python3.10 -m venv .venv310
source .venv310/bin/activate          # Windows: .venv310\Scripts\activate

# Install lightweight dependencies
pip install -r requirements-streamlit.txt

# Launch the web app
streamlit run streamlit_app.py
```

The original Gradio version is also available:

```bash
cd app
python app.py
```

---

## Project Structure

```
NLP_Summative-Project/
├── streamlit_app.py                 # Main web interface (Streamlit)
├── requirements-streamlit.txt       # Dependencies for the live app
├── requirements.txt                 # Full stack (for re-running the notebook)
├── notebooks/
│   └── MasakhaNEWS_Kiswahili_Classification.ipynb
├── models/
│   └── kiswahili_news_classifier_svm/
│       ├── svm_pipeline.joblib      # Best model
│       └── label_names.json
├── app/                             # Optional Gradio version
├── src/utils.py
├── report/                          # Academic report materials
└── README.md
```

---

## Reproducing the Experiments

Open the notebook on Kaggle, Colab, or locally. It is fully executed and contains:

1. Data loading (Kiswahili split only)
2. Exploratory analysis and class distribution
3. Baseline: TF-IDF + Logistic Regression
4. Experiment: TF-IDF + Linear SVM
5. Experiment: fine-tuned Afro-XLMR-mini
6. Learning curves, confusion matrices, per-class F1
7. Error analysis with concrete examples
8. Model serialisation for deployment

---

## Links

- **GitHub Repository**: https://github.com/francis-collab/NLP_Summative-Project
- **Live System**: https://kiswahili-news-classifier.streamlit.app/
- **Demo Video**: *[insert your video link]*

---

**Author**: Francis Mutabazi  
Summative Project – Machine Learning Techniques I  
October 2026
