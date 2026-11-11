"""Evaluation metrics for classification."""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_classification_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Calculate accuracy, F1, precision, and recall."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    prec_macro = precision_score(labels, predictions, average="macro", zero_division=0)
    rec_macro = recall_score(labels, predictions, average="macro", zero_division=0)

    return {
        "eval_accuracy": float(acc),
        "eval_f1_macro": float(f1_macro),
        "eval_f1_weighted": float(f1_weighted),
        "eval_precision_macro": float(prec_macro),
        "eval_recall_macro": float(rec_macro),
    }
