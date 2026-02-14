from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, top_k_accuracy_score


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> dict:
    y_pred = logits.argmax(axis=1)
    return {
        "top1": float((y_pred == y_true).mean()),
        "top3": float(top_k_accuracy_score(y_true, logits, k=min(3, logits.shape[1]), labels=np.arange(logits.shape[1]))),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "cm": confusion_matrix(y_true, y_pred),
    }
