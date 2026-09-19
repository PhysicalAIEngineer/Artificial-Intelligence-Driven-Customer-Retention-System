from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

def classification_metrics(y_true, probabilities, threshold: float):
    pred=(np.asarray(probabilities)>=threshold).astype(int)
    return {
        "accuracy":float(accuracy_score(y_true,pred)),
        "precision":float(precision_score(y_true,pred,zero_division=0)),
        "recall":float(recall_score(y_true,pred,zero_division=0)),
        "f1":float(f1_score(y_true,pred,zero_division=0)),
        "roc_auc":float(roc_auc_score(y_true,probabilities)),
        "pr_auc":float(average_precision_score(y_true,probabilities)),
    }

def optimize_threshold(y_true, probabilities, min_recall=0.80):
    candidates=np.linspace(0.05,0.95,181); best=None
    for t in candidates:
        m=classification_metrics(y_true,probabilities,float(t))
        if m["recall"]>=min_recall:
            key=(m["f1"],m["precision"],float(t))
            if best is None or key>best["key"]:
                best={"threshold":float(t),"metrics":m,"key":key}
    if best:
        best.pop("key"); return best
    return max(
        ({"threshold":float(t),"metrics":classification_metrics(y_true,probabilities,float(t))} for t in candidates),
        key=lambda x:(x["metrics"]["recall"],x["metrics"]["precision"]),
    )
