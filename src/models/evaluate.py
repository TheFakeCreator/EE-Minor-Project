"""
Evaluation utilities and power-system specific metrics for grid stability prediction.
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, brier_score_loss
)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, Any]:
    """
    Computes standard ML metrics along with power-systems specific reliability metrics.
    Label convention: 1 = Stable, 0 = Unstable.
    """
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    # Precision, recall, F1 per class
    # 0 = Unstable, 1 = Stable
    prec, rec, f1, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    
    # Confusion matrix:
    # [[TN, FP],
    #  [FN, TP]] where 0 is negative (unstable) and 1 is positive (stable)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # Power System Critical Metric:
    # Missed Instability (False Positives where ground truth was Unstable 0, but predicted Stable 1)
    # This is catastrophic for grid security because an operator wouldn't take remedial action!
    unstable_total = int(tn + fp)
    missed_instability_count = int(fp)
    missed_instability_rate = float(fp / unstable_total) if unstable_total > 0 else 0.0
    
    metrics = {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'unstable_recall': float(rec[0]),  # Detecting unstable states
        'unstable_precision': float(prec[0]),
        'unstable_f1': float(f1[0]),
        'stable_recall': float(rec[1]),
        'stable_precision': float(prec[1]),
        'stable_f1': float(f1[1]),
        'macro_f1': float((f1[0] + f1[1]) / 2.0),
        'missed_instability_count': missed_instability_count,
        'missed_instability_rate': missed_instability_rate,
        'confusion_matrix': cm.tolist()
    }
    
    if y_prob is not None:
        try:
            # probability of class 1 (stable)
            prob_stable = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            metrics['roc_auc'] = float(roc_auc_score(y_true, prob_stable))
            metrics['brier_score'] = float(brier_score_loss(y_true, prob_stable))
        except Exception:
            metrics['roc_auc'] = None
            metrics['brier_score'] = None
            
    return metrics
