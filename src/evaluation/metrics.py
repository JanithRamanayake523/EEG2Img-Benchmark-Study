"""
Evaluation metrics for EEG classification models.

Provides comprehensive metrics including:
- Accuracy, precision, recall, F1-score
- Multi-class AUC (one-vs-rest and one-vs-one)
- Confusion matrices
- Per-class performance metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix as sklearn_confusion_matrix,
    classification_report,
    cohen_kappa_score,
    matthews_corrcoef
)
from typing import Dict, List, Optional, Tuple, Union
import json
from pathlib import Path


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    average: str = 'macro',
    labels: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        y_proba: Predicted probabilities, shape (n_samples, n_classes), optional
        average: Averaging strategy for multi-class ('macro', 'weighted', 'micro')
        labels: List of label indices to include, optional

    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - precision: Precision score
            - recall: Recall score
            - f1: F1-score
            - kappa: Cohen's kappa coefficient
            - mcc: Matthews correlation coefficient
            - auc: Multi-class AUC (if y_proba provided)
            - auc_ovr: One-vs-rest AUC (if y_proba provided)
            - auc_ovo: One-vs-one AUC (if y_proba provided)

    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 2, 2, 0, 1, 1])
        >>> y_proba = np.random.rand(6, 3)
        >>> metrics = compute_metrics(y_true, y_pred, y_proba)
        >>> print(f"Accuracy: {metrics['accuracy']:.4f}")
    """
    metrics = {}

    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, labels=labels, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, labels=labels, zero_division=0)

    # Agreement metrics
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred, labels=labels)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    # AUC metrics (if probabilities provided)
    if y_proba is not None:
        try:
            # One-vs-rest AUC
            metrics['auc_ovr'] = roc_auc_score(
                y_true, y_proba,
                average=average,
                multi_class='ovr',
                labels=labels
            )

            # One-vs-one AUC
            metrics['auc_ovo'] = roc_auc_score(
                y_true, y_proba,
                average=average,
                multi_class='ovo',
                labels=labels
            )

            # Use OVR as default AUC
            metrics['auc'] = metrics['auc_ovr']

        except ValueError as e:
            # Handle cases where AUC cannot be computed (e.g., single class)
            metrics['auc'] = np.nan
            metrics['auc_ovr'] = np.nan
            metrics['auc_ovo'] = np.nan

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    normalize: Optional[str] = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label indices, optional
        normalize: Normalization mode ('true', 'pred', 'all', or None)

    Returns:
        Confusion matrix, shape (n_classes, n_classes)

    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 2, 2, 0, 1, 1])
        >>> cm = compute_confusion_matrix(y_true, y_pred)
        >>> print(cm)
    """
    cm = sklearn_confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return cm


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[int]] = None,
    target_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class precision, recall, and F1-score.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label indices, optional
        target_names: List of class names, optional

    Returns:
        Dictionary mapping class names/indices to their metrics

    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_pred = np.array([0, 2, 2, 0, 1, 1])
        >>> metrics = compute_per_class_metrics(y_true, y_pred,
        ...                                      target_names=['left', 'right', 'rest'])
        >>> print(metrics['left'])
    """
    # Get unique classes
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))

    if target_names is None:
        target_names = [str(label) for label in labels]

    # Compute per-class metrics
    precision = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    # Create per-class dictionary
    per_class = {}
    for i, (label, name) in enumerate(zip(labels, target_names)):
        per_class[name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(np.sum(y_true == label))
        }

    return per_class


def compute_multiclass_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    average: str = 'macro',
    multi_class: str = 'ovr'
) -> float:
    """
    Compute multi-class AUC score.

    Args:
        y_true: True labels, shape (n_samples,)
        y_proba: Predicted probabilities, shape (n_samples, n_classes)
        average: Averaging strategy ('macro', 'weighted', 'micro')
        multi_class: Multi-class strategy ('ovr' = one-vs-rest, 'ovo' = one-vs-one)

    Returns:
        AUC score

    Example:
        >>> y_true = np.array([0, 1, 2, 0, 1, 2])
        >>> y_proba = np.random.rand(6, 3)
        >>> y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize
        >>> auc = compute_multiclass_auc(y_true, y_proba)
        >>> print(f"AUC: {auc:.4f}")
    """
    try:
        auc = roc_auc_score(y_true, y_proba, average=average, multi_class=multi_class)
        return float(auc)
    except ValueError:
        return np.nan


class MetricsTracker:
    """
    Track and aggregate metrics across multiple folds/runs.

    Useful for cross-validation and repeated experiments.

    Example:
        >>> tracker = MetricsTracker()
        >>> for fold in range(5):
        ...     # Train and evaluate model
        ...     metrics = compute_metrics(y_true, y_pred, y_proba)
        ...     tracker.update(metrics)
        >>> summary = tracker.summary()
        >>> print(f"Mean accuracy: {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")
    """

    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics_list: List[Dict[str, float]] = []
        self.confusion_matrices: List[np.ndarray] = []

    def update(self, metrics: Dict[str, float], confusion_matrix: Optional[np.ndarray] = None):
        """
        Add metrics from a single fold/run.

        Args:
            metrics: Dictionary of metric values
            confusion_matrix: Confusion matrix, optional
        """
        self.metrics_list.append(metrics)
        if confusion_matrix is not None:
            self.confusion_matrices.append(confusion_matrix)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics (mean, std, min, max) across all folds.

        Returns:
            Dictionary mapping metric names to their statistics
        """
        if not self.metrics_list:
            return {}

        summary = {}
        metric_names = self.metrics_list[0].keys()

        for metric_name in metric_names:
            values = [m[metric_name] for m in self.metrics_list if not np.isnan(m[metric_name])]

            if values:
                summary[metric_name] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'values': values
                }
            else:
                summary[metric_name] = {
                    'mean': np.nan,
                    'std': np.nan,
                    'min': np.nan,
                    'max': np.nan,
                    'values': []
                }

        return summary

    def mean_confusion_matrix(self) -> Optional[np.ndarray]:
        """
        Compute mean confusion matrix across all folds.

        Returns:
            Mean confusion matrix, or None if no matrices stored
        """
        if not self.confusion_matrices:
            return None

        return np.mean(self.confusion_matrices, axis=0)

    def get_best_fold(self, metric: str = 'accuracy', mode: str = 'max') -> Tuple[int, float]:
        """
        Get the fold with best performance for a given metric.

        Args:
            metric: Metric name to optimize
            mode: 'max' or 'min'

        Returns:
            Tuple of (fold_index, metric_value)
        """
        if not self.metrics_list:
            return -1, np.nan

        values = [m[metric] for m in self.metrics_list if not np.isnan(m[metric])]

        if not values:
            return -1, np.nan

        if mode == 'max':
            best_idx = int(np.argmax(values))
        else:
            best_idx = int(np.argmin(values))

        return best_idx, values[best_idx]

    def save(self, filepath: Union[str, Path]):
        """
        Save metrics to JSON file.

        Args:
            filepath: Path to save file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'metrics_list': self.metrics_list,
            'summary': self.summary(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: Union[str, Path]):
        """
        Load metrics from JSON file.

        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.metrics_list = data['metrics_list']

    def __len__(self) -> int:
        """Return number of folds/runs tracked."""
        return len(self.metrics_list)

    def __repr__(self) -> str:
        """String representation."""
        if not self.metrics_list:
            return "MetricsTracker(empty)"

        summary = self.summary()
        acc = summary.get('accuracy', {})
        return f"MetricsTracker(n={len(self)}, accuracy={acc.get('mean', np.nan):.4f}±{acc.get('std', np.nan):.4f})"
