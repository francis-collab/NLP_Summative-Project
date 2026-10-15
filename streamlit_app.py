"""
Kiswahili News Topic Classifier – Streamlit Web Interface
Best model: TF-IDF + Linear SVM (trained on MasakhaNEWS)
"""

import streamlit as st
import joblib
import json
import os

st.set_page_config(
    page_title="Kiswahili News Topic Classifier",
    page_icon="🌍",
    layout="centered"
)

# -------------------------------------------------
# Locate the saved model
# -------------------------------------------------
_possible_dirs = [
    os.path.join("models", "kiswahili_news_classifier_svm"),
    "kiswahili_news_classifier_svm",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "kiswahili_news_classifier_svm"),
]

MODEL_DIR = None
for d in _possible_dirs:
    if os.path.isdir(d) and os.path.isfile(os.path.join(d, "svm_pipeline.joblib")):
        MODEL_DIR = d
        break

if MODEL_DIR is None:
    st.error("Could not find the model folder 'kiswahili_news_classifier_svm'.")
    st.stop()

@st.cache_resource
def load_model():
    pipeline = joblib.load(os.path.join(MODEL_DIR, "svm_pipeline.joblib"))
    with open(os.path.join(MODEL_DIR, "label_names.json"), encoding="utf-8") as f:
        labels = json.load(f)
    return pipeline, labels

pipeline, label_names = load_model()

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("🌍 Kiswahili News Topic Classifier")
st.markdown(
    "Paste any **Kiswahili** news text below. "
    "The model will classify it into one of seven topics:\n\n"
    "**business · entertainment · health · politics · religion · sports · technology**"
)

text = st.text_area(
    "Kiswahili News Text",
    height=220,
    placeholder="Weka habari ya Kiswahili hapa... (Paste Kiswahili news text here)"
)

col1, col2 = st.columns([1, 3])
with col1:
    classify_btn = st.button("Classify", type="primary", use_container_width=True)

if classify_btn:
    if not text or not text.strip():
        st.warning("Please enter some Kiswahili news text.")
    else:
        pred_id = int(pipeline.predict([text.strip()])[0])
        pred_label = label_names[pred_id]

        st.success(f"**Predicted Topic:** {pred_label.upper()}")

        # Decision scores
        try:
            scores = pipeline.decision_function([text.strip()])[0]
            score_dict = {label_names[i]: float(s) for i, s in enumerate(scores)}
            min_s = min(score_dict.values())
            score_dict = {k: round(v - min_s + 0.1, 4) for k, v in score_dict.items()}
            st.subheader("Decision Scores")
            st.caption("Higher bar = model is more confident for that topic")
            st.bar_chart(score_dict)
        except Exception:
            pass

st.divider()
st.caption(
    "Model: TF-IDF + Linear SVM · Dataset: MasakhaNEWS (Kiswahili) · "
    "Capstone Project – ML Techniques I (October 2026)"
)

# Example buttons
with st.expander("Try an example"):
    examples = {
        "Health": "Serikali imezindua mpango mpya wa kuboresha huduma za afya vijijini na kuongeza idadi ya madaktari.",
        "Sports": "Timu ya Taifa ya Kandanda imafuzu fainali za Kombe la Afrika baada ya kushinda mechi muhimu.",
        "Business": "Bei ya mafuta duniani imepanda kwa kasi wiki hii kutokana na mvutano wa kijiografia.",
        "Religion": "Kanisa limetangaza mkutano mkubwa wa maombi utakaofanyika Ijumaa ijayo.",
    }
    for name, ex in examples.items():
        if st.button(f"Example: {name}"):
            st.session_state["example_text"] = ex
            st.rerun()

if "example_text" in st.session_state:
    st.text_area("Selected example (copy it into the box above)", value=st.session_state["example_text"], height=100)
