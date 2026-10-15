# MasakhaNEWS Kiswahili News Topic Classification

**Capstone Project – Machine Learning Techniques I**  
**Task**: Text Classification for an African Language (Kiswahili)  
**Dataset**: MasakhaNEWS (Kiswahili / swa split only)  
**Author**: Francis Mutabazi  
**Date**: October 2026

---

## Key Results

| Model                                | Accuracy | Macro F1 | Weighted F1 |
|--------------------------------------|----------|----------|-------------|
| TF-IDF + Logistic Regression (Baseline) | 0.8277   | 0.8025   | 0.8301      |
| **TF-IDF + Linear SVM** (deployed)   | 0.8340   | 0.7961   | **0.8343**  |
| Afro-XLMR-mini (Fine-tuned)          | 0.8403   | 0.7929   | 0.8318      |

Deployed model = **TF-IDF + Linear SVM**

---

## Fastest way to get a permanent live link (FREE)

### 1. Put this project on GitHub
- Create a new **public** repository on GitHub.
- Upload the whole `Project` folder (or push it with git).

### 2. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **New app**
4. Choose:
   - Repository = your repo
   - Branch = `main`
   - Main file path = `streamlit_app.py`
5. Click **Deploy**

In 1–2 minutes you get a permanent public URL that looks like:
```
https://YOUR-USERNAME-....streamlit.app
```

That is the link you put in your report and demo video.

### Alternative local test
```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

---

## Project Structure

```
Project/
├── streamlit_app.py                 ← ready for Streamlit Community Cloud
├── requirements-streamlit.txt       ← use this for the web app
├── requirements.txt                 ← full ML stack (for the notebook)
├── notebooks/
│   └── MasakhaNEWS_Kiswahili_Classification.ipynb
├── models/
│   └── kiswahili_news_classifier_svm/
│       ├── svm_pipeline.joblib
│       └── label_names.json
├── app/                             ← original Gradio version (optional)
├── report/
│   └── Final_Report.md
├── src/utils.py
└── README.md
```

---

## Links (fill after deployment)

- **GitHub Repository**: 
- **Live Demo**: 
- **Demo Video**: 

## Citation
Adelani et al. (2023). MasakhaNEWS: News Topic Classification for African languages. https://arxiv.org/abs/2304.09972
