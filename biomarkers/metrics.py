"""Patient-level metrics and transparent handling of single-class cohorts."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from biomarkers import TASKS


def aggregate_patients(slide_predictions):
    for task in TASKS:
        known = slide_predictions.loc[slide_predictions[task] >= 0]
        if (known.groupby("patient_id")[task].nunique() > 1).any():
            raise ValueError(f"Inconsistent patient-level {task} labels")
    grouped = slide_predictions.groupby("patient_id", sort=True)
    result = grouped[[f"{t}_prob" for t in TASKS]].mean()
    result["n_slides"] = grouped.size()
    for task in TASKS:
        known = slide_predictions.loc[slide_predictions[task] >= 0].groupby("patient_id")[task].first()
        result[task] = known.reindex(result.index).fillna(-1).astype(int)
    return result.reset_index()


def compute_metrics(predictions, threshold=0.5):
    result = {"n_units": len(predictions), "threshold": float(threshold), "tasks": {}}
    aucs, losses = [], []
    for task in TASKS:
        labeled = predictions.loc[predictions[task] >= 0]
        y = labeled[task].to_numpy(dtype=int)
        p = labeled[f"{task}_prob"].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"Invalid {task} prediction probabilities")
        if not len(y):
            result["tasks"][task] = {"n": 0, "n_positive": 0, "n_negative": 0,
                                     "auroc": None, "auprc": None, "log_loss": None,
                                     "reason": "No observed labels"}
            continue
        predicted = p >= threshold
        tp = int(((y == 1) & predicted).sum())
        tn = int(((y == 0) & ~predicted).sum())
        fp = int(((y == 0) & predicted).sum())
        fn = int(((y == 1) & ~predicted).sum())
        both_classes = len(np.unique(y)) == 2
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        log_loss = float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean())
        losses.append(log_loss)
        auc = float(roc_auc_score(y, p)) if both_classes else None
        if auc is not None:
            aucs.append(auc)
        result["tasks"][task] = {
            "n": len(y), "n_positive": int(y.sum()), "n_negative": int(len(y) - y.sum()),
            "auroc": auc,
            "auprc": float(average_precision_score(y, p)) if both_classes else None,
            "accuracy": float((predicted == y).mean()),
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
            "ppv": tp / (tp + fp) if tp + fp else None,
            "npv": tn / (tn + fn) if tn + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "brier_score": float(((p - y) ** 2).mean()), "log_loss": log_loss,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "reason": None if both_classes else "Only one label class; AUROC/AUPRC undefined",
        }
    result["macro_auroc"] = float(np.mean(aucs)) if aucs else None
    result["macro_auroc_n_tasks"] = len(aucs)
    result["macro_log_loss"] = float(np.mean(losses)) if losses else None
    return result
