# MasakhaNEWS Kiswahili News Topic Classification

**Capstone Project – Machine Learning Techniques I**  
**Project Type**: Text Classification for an African Language  
**Author**: Francis Mutabazi  
**Date**: October 2026

---

## Project Links

- **GitHub Repository**: *[Insert your GitHub repository URL here]*
- **Demo Video**: *[Insert link to 10–15 minute demonstration video]*
- **Live System**: *[Insert Gradio / Hugging Face Spaces / public URL if available]*  

If the live system cannot be kept online indefinitely due to hosting limits, the application can be launched locally with:

```bash
cd app
python app.py
```

---

## 1. Abstract

This project develops a news topic classification system for Kiswahili, one of the most widely spoken languages in East Africa. Using the MasakhaNEWS dataset (Kiswahili split only), we train and compare classical machine-learning models (TF-IDF + Logistic Regression, TF-IDF + Linear SVM) and a fine-tuned Africa-centric transformer (Afro-XLMR-mini). The Linear SVM achieves the highest weighted F1-score (0.8343) and strong accuracy (0.8340), while the transformer obtains the highest accuracy (0.8403). After systematic experiments and error analysis, the TF-IDF + Linear SVM model is selected for deployment because of its competitive performance, simplicity, and low computational cost. A Gradio web interface allows users to paste Kiswahili news text and receive the predicted topic together with decision scores. The work demonstrates that carefully engineered classical models remain highly effective for medium-sized African-language classification tasks and provides a fully reproducible pipeline from data preparation to web deployment.

## 2. Introduction

African languages remain under-represented in natural language processing research, largely because of limited high-quality labelled datasets (Adelani et al., 2023). Kiswahili (Swahili) is a major lingua franca spoken by more than 100 million people across East and Central Africa, yet publicly available NLP resources for the language are still modest compared with high-resource languages.

News topic classification is a practical and high-impact task: media organisations, content aggregators, and civic platforms need to automatically organise large volumes of Kiswahili news into categories such as politics, health, sports or business. An accurate classifier can improve information retrieval, personalised news feeds, and content moderation for East African audiences.

**Objective.** The goal of this project is to:

1. Investigate real Kiswahili news topic classification using a publicly available, naturally occurring dataset.
2. Establish a strong classical baseline and conduct controlled experiments with both classical and transformer-based models.
3. Perform quantitative evaluation and qualitative error analysis.
4. Deploy the best-performing practical model through a simple web interface so that end users can interact with it.

We deliberately focus on a single African language (Kiswahili) and a clearly defined multi-class classification problem, satisfying the course requirement for a meaningful machine-learning component rather than a simple API wrapper.

## 3. Related Work

MasakhaNEWS (Adelani et al., 2023) is the largest publicly available news-topic classification dataset covering 16 African languages, including Kiswahili. The authors release classical baselines (logistic regression, SVMs) and fine-tuned multilingual language models, showing that supervised approaches work well even for lower-resource African languages.

Africa-centric pretrained models such as AfroXLMR (Alabi et al., 2022) and AfriBERTa (Ogueji et al., 2021) have been shown to improve downstream performance on African languages by continuing pre-training or adapting the vocabulary of massively multilingual models (XLM-R). For news classification specifically, several recent fine-tunes of AfroXLMR and AfriBERTa on MasakhaNEWS have been published on the Hugging Face Hub, confirming that the dataset is a standard benchmark.

Despite these advances, many practical deployments still rely on classical TF-IDF + linear models because they require far less compute, train in seconds, and often match or exceed transformer performance on medium-sized datasets. This project therefore compares both families of approaches on the Kiswahili split and selects the most practical model for deployment.

## 4. Dataset and Data Preparation

### 4.1 Source and Characteristics

We use the **MasakhaNEWS** dataset (Adelani et al., 2023), specifically the Kiswahili (`swa`) configuration available on the Hugging Face Hub:

```python
from datasets import load_dataset
dataset = load_dataset("masakhane/masakhanews", "swa")
```

| Split      | Number of articles |
|------------|--------------------|
| Train      | 1 658              |
| Validation | 237                |
| Test       | 476                |
| **Total**  | **2 371**          |

Each example contains:

- `headline` – news title
- `text` – article body
- `headline_text` – concatenation of headline and body (used as primary input)
- `label` – integer class index
- `category` – string label
- `url` – original source URL (mostly BBC Swahili)

The seven categories are identical across all MasakhaNEWS languages:

`business`, `entertainment`, `health`, `politics`, `religion`, `sports`, `technology`.

### 4.2 Class Distribution (Training Set)

| Category       | Count |
|----------------|-------|
| sports         | 350   |
| health         | 350   |
| politics       | 350   |
| business       | 221   |
| religion       | 204   |
| technology     | 115   |
| entertainment  | 68    |

The dataset exhibits mild class imbalance; entertainment and technology are the minority classes. All texts are real, naturally occurring news articles, satisfying the course requirement to avoid synthetic LLM-generated data.

### 4.3 Preprocessing

We apply light cleaning that preserves Kiswahili orthography and morphology:

- strip leading/trailing whitespace
- normalise multiple spaces / newlines to a single space
- keep all original characters (no aggressive stemming or stop-word removal, which can harm agglutinative languages)

The cleaned `headline_text` field is used for all classical models. For the transformer we tokenise with the model’s own tokenizer (maximum length 256 tokens).

### 4.4 Challenges

- Limited size of minority classes (especially entertainment).
- Occasional multi-topic articles that receive only a single gold label.
- Spelling variation and code-switching common in real East African news.

## 5. Methodology

### 5.1 Baseline – TF-IDF + Logistic Regression

- Vectoriser: `TfidfVectorizer(max_features=15 000, ngram_range=(1,2), min_df=2)`
- Classifier: `LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`
- Pipeline trained on the full training set and evaluated on the held-out test set.

### 5.2 Experiment 1 – TF-IDF + Linear SVM

Identical TF-IDF features, replaced the classifier with `LinearSVC(class_weight="balanced", max_iter=2000, random_state=42)`. Linear SVMs are known to perform well on high-dimensional sparse text data.

### 5.3 Experiment 2 – Fine-tuned Afro-XLMR-mini

- Base model: `Davlan/afro-xlmr-mini` (Africa-centric adaptation of XLM-R)
- Task head: sequence classification with 7 labels
- Training: Hugging Face `Trainer`, 4 epochs, learning rate 2e-5, batch size 8, weight decay 0.01
- Evaluation metric during training: macro F1

All experiments use a fixed random seed (42) for reproducibility.

## 6. Experiments and Results

| Model                              | Accuracy | Macro F1 | Weighted F1 |
|------------------------------------|----------|----------|-------------|
| TF-IDF + Logistic Regression (Baseline) | 0.8277   | **0.8025** | 0.8301      |
| TF-IDF + Linear SVM                | 0.8340   | 0.7961   | **0.8343**  |
| Afro-XLMR-mini (Fine-tuned)        | **0.8403** | 0.7929 | 0.8318      |

**Key observations**

- The classical baseline is already strong (>82 % accuracy).
- Linear SVM improves accuracy and weighted F1 over logistic regression while remaining extremely fast to train.
- The transformer obtains the highest raw accuracy but lower macro F1, indicating that gains are concentrated on the majority classes while minority-class recall suffers.
- On this dataset size, a well-tuned TF-IDF + linear model remains highly competitive with a fine-tuned transformer and is far cheaper to train and serve.

## 7. Error Analysis and Discussion

Representative correct predictions include clear sports reports and health policy articles, where lexical cues (team names, “afya”, “chanjo”, etc.) are strong.

Common error patterns:

1. **Semantically close categories** – politics ↔ business (economic policy stories) and health ↔ religion (faith-based health campaigns).
2. **Short or ambiguous headlines** – limited context leads to confusion.
3. **Multi-topic articles** – an article that mixes politics and technology receives only one gold label.
4. **Minority classes** – entertainment and technology have fewer training examples, resulting in lower recall for those classes, especially for the transformer.

These findings align with the observations reported by Adelani et al. (2023) on the same dataset and highlight the remaining challenges of single-label topic classification for African languages.

## 8. Deployment

The final model (TF-IDF + Linear SVM pipeline) is serialised with `joblib` and served through a Gradio web interface located in `app/app.py`. Users paste Kiswahili news text and receive:

- the predicted topic (markdown)
- decision-function scores for all seven classes (approximate confidence)

The interface includes three example texts (health, sports, business) for quick testing. The application can be launched locally or deployed on Hugging Face Spaces / Streamlit Cloud.

## 9. Limitations and Future Work

**Limitations**

- Dataset size remains modest, especially for entertainment and technology.
- Single-label formulation cannot capture multi-topic articles.
- Performance may degrade on highly informal or dialectal Kiswahili not represented in BBC-style news.
- The transformer experiment used a small model and limited epochs; larger models or longer training might close the gap.

**Future Work**

- Collect additional labelled Kiswahili news (or use data-augmentation techniques that preserve linguistic validity).
- Explore multi-label classification.
- Apply parameter-efficient fine-tuning (LoRA / QLoRA) to larger Afrocentric models.
- Continuously monitor the deployed system and retrain when distribution shift is detected.

## 10. Conclusion

We successfully built an end-to-end Kiswahili news topic classifier using real MasakhaNEWS data. A simple TF-IDF + Linear SVM model achieves strong quantitative results (accuracy 0.8340, weighted F1 0.8343) while remaining computationally inexpensive and easy to deploy. The fine-tuned transformer yields the highest accuracy but does not dominate on balanced metrics, illustrating that classical methods are still highly relevant for medium-sized African-language tasks. Error analysis revealed the expected difficulties of overlapping topics and class imbalance. The complete pipeline—from data loading through experimentation to a usable web application—is fully reproducible and satisfies all requirements of the ML Techniques I capstone.

## 11. References

Adelani, D. I., Masiak, M., Azime, I. A., Alabi, J. O., Tonja, A. L., Mwase, C., … Stenetorp, P. (2023). *MasakhaNEWS: News Topic Classification for African languages*. arXiv:2304.09972. https://arxiv.org/abs/2304.09972

Alabi, J. O., Adelani, D. I., Mosquera, M., & Stenetorp, P. (2022). *Adapting Pre-trained Language Models to African Languages via Multilingual Adaptive Fine-Tuning*. In *Proceedings of the 29th International Conference on Computational Linguistics* (COLING 2022). https://aclanthology.org/2022.coling-1.433/

Ogueji, K., Zhu, Y., & Lin, J. (2021). *Small Data? No Problem! Exploring the Viability of Pretrained Multilingual Language Models for Low-resourced Languages*. In *Proceedings of the 1st Workshop on Multilingual Representation Learning*. https://aclanthology.org/2021.mrl-1.11/

Masakhane. (2023). *MasakhaNEWS Dataset*. Hugging Face. https://huggingface.co/datasets/masakhane/masakhanews

Davlan. (2022). *Afro-XLMR-mini*. Hugging Face. https://huggingface.co/Davlan/afro-xlmr-mini

Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations*.

---

*All source code, the trained model, the notebook, and this report are provided in the accompanying GitHub repository.*
