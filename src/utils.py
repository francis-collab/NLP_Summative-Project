"""
Helper functions for the Kiswahili News Topic Classification project.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

LABEL_NAMES = [
    "business",
    "entertainment",
    "health",
    "politics",
    "religion",
    "sports",
    "technology"
]

def compute_metrics_dict(y_true, y_pred):
    """Return a dictionary of standard classification metrics."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4)
    }

def clean_text(text: str) -> str:
    """Light cleaning for Kiswahili news text."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())
