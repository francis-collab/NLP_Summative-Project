"""
Kiswahili News Topic Classifier – Gradio Web Interface
Best model: TF-IDF + Linear SVM (trained on MasakhaNEWS)
"""

import gradio as gr
import joblib
import json
import os
import sys

# -------------------------------------------------
# Locate the saved model (works in multiple layouts)
# -------------------------------------------------
_possible_dirs = [
    # When app.py sits next to the model folder (HF Spaces style)
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "kiswahili_news_classifier_svm"),
    # When model is in ../models/ (local project layout)
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "kiswahili_news_classifier_svm"),
    os.path.join("models", "kiswahili_news_classifier_svm"),
    "./kiswahili_news_classifier_svm",
    os.path.join("..", "models", "kiswahili_news_classifier_svm"),
]

MODEL_DIR = None
for d in _possible_dirs:
    if os.path.isdir(d) and os.path.isfile(os.path.join(d, "svm_pipeline.joblib")):
        MODEL_DIR = d
        break

if MODEL_DIR is None:
    raise FileNotFoundError(
        "Could not locate 'kiswahili_news_classifier_svm'. "
        "Expected folder containing svm_pipeline.joblib and label_names.json."
    )

print(f"[INFO] Loading model from: {MODEL_DIR}")
svm_pipeline = joblib.load(os.path.join(MODEL_DIR, "svm_pipeline.joblib"))

with open(os.path.join(MODEL_DIR, "label_names.json"), encoding="utf-8") as f:
    label_names = json.load(f)

# -------------------------------------------------
# Prediction function
# -------------------------------------------------
def predict_topic(text: str):
    if not text or not str(text).strip():
        return "Please enter some Kiswahili news text.", {}

    text = str(text).strip()

    # Prediction
    pred_id = int(svm_pipeline.predict([text])[0])
    pred_label = label_names[pred_id]

    # Decision scores → simple positive scores for display
    try:
        decision_scores = svm_pipeline.decision_function([text])[0]
        scores = {label_names[i]: float(score) for i, score in enumerate(decision_scores)}
        min_score = min(scores.values())
        scores = {k: round(v - min_score + 0.1, 4) for k, v in scores.items()}
    except Exception:
        scores = {pred_label: 1.0}

    summary = f"**Predicted Topic:** {pred_label.upper()}"
    return summary, scores


# -------------------------------------------------
# Gradio Interface
# -------------------------------------------------
demo = gr.Interface(
    fn=predict_topic,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Weka habari ya Kiswahili hapa... (Paste Kiswahili news text here)",
        label="Kiswahili News Text"
    ),
    outputs=[
        gr.Markdown(label="Prediction"),
        gr.Label(label="Decision Scores (higher = more confident)")
    ],
    title="🌍 Kiswahili News Topic Classifier",
    description=(
        "Classify Kiswahili news articles into one of seven topics:\n"
        "**business · entertainment · health · politics · religion · sports · technology**\n\n"
        "Model: **TF-IDF + Linear SVM** trained on the MasakhaNEWS (swa) dataset.\n"
        "Capstone Project – ML Techniques I (October 2026)"
    ),
    examples=[
        ["Serikali imezindua mpango mpya wa kuboresha huduma za afya vijijini na kuongeza idadi ya madaktari."],
        ["Timu ya Taifa ya Kandanda imafuzu fainali za Kombe la Afrika baada ya kushinda mechi muhimu."],
        ["Bei ya mafuta duniani imepanda kwa kasi wiki hii kutokana na mvutano wa kijiografia."],
        ["Wanafunzi wa shule za msingi wameanza mitihani ya kitaifa wiki hii nchini Tanzania."],
        ["Kanisa limetangaza mkutano mkubwa wa maombi utakaofanyika Ijumaa ijayo."]
    ],
    allow_flagging="never",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    # share=True creates a temporary public link (useful for demos)
    demo.launch(share=True, server_name="0.0.0.0")
