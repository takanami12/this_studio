"""
Evaluation Metrics for PhaBERT-CNN.

Standard binary classification metrics:
- Sensitivity (sn) = TP / (TP + FN)  -- ability to identify virulent phages
- Specificity (sp) = TN / (TN + FP)  -- ability to identify temperate phages
- Accuracy (acc) = (TP + TN) / (TP + FN + TN + FP)

Convention:
- Positive class (1) = Virulent
- Negative class (0) = Temperate
"""

import numpy as np
from typing import Dict
from sklearn.metrics import confusion_matrix


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute sensitivity, specificity, and accuracy.
    
    Args:
        y_true: Ground truth labels (0=temperate, 1=virulent)
        y_pred: Predicted labels
        
    Returns:
        Dict with 'sensitivity', 'specificity', 'accuracy' keys
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return {
        'sensitivity': sensitivity * 100,  # as percentage
        'specificity': specificity * 100,
        'accuracy': accuracy * 100,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics."""
    print(f"{prefix}Sensitivity (sn): {metrics['sensitivity']:.2f}%")
    print(f"{prefix}Specificity (sp): {metrics['specificity']:.2f}%")
    print(f"{prefix}Accuracy   (acc): {metrics['accuracy']:.2f}%")


def aggregate_fold_metrics(fold_metrics: list) -> Dict[str, float]:
    """
    Aggregate metrics across k folds (mean ± std).
    
    Args:
        fold_metrics: List of metric dicts from each fold
        
    Returns:
        Dict with mean and std for each metric
    """
    result = {}
    for key in ['sensitivity', 'specificity', 'accuracy']:
        values = [m[key] for m in fold_metrics]
        result[f'{key}_mean'] = np.mean(values)
        result[f'{key}_std'] = np.std(values)
    return result
