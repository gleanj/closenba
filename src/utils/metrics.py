"""Shared evaluation metrics for model assessment."""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    classification_report
)
import logging

logger = logging.getLogger(__name__)


def calculate_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    include_report: bool = False
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics.

    This function provides a standardized way to evaluate binary classification
    models across the codebase, ensuring consistent metrics calculation.

    Args:
        y_true: True labels (binary)
        y_pred: Predicted labels (binary)
        y_prob: Predicted probabilities (optional, for ROC-AUC and log loss)
        include_report: Whether to include detailed classification report

    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score
        - roc_auc: ROC-AUC score (if y_prob provided)
        - log_loss: Log loss (if y_prob provided)
        - confusion_matrix: 2x2 confusion matrix as list
        - tn, fp, fn, tp: Individual confusion matrix values
        - specificity: True negative rate
        - report: Classification report text (if include_report=True)

    Example:
        >>> y_true = np.array([0, 1, 1, 0, 1])
        >>> y_pred = np.array([0, 1, 0, 0, 1])
        >>> y_prob = np.array([0.2, 0.8, 0.4, 0.1, 0.9])
        >>> metrics = calculate_classification_metrics(y_true, y_pred, y_prob)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
        Accuracy: 0.800
    """
    metrics = {}

    try:
        # Basic metrics (always calculated)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Extract individual confusion matrix values
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            metrics['tp'] = int(tp)

            # Calculate specificity (true negative rate)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Probability-based metrics (if probabilities provided)
        if y_prob is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except ValueError as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                metrics['roc_auc'] = None

            try:
                metrics['log_loss'] = log_loss(y_true, y_prob)
            except ValueError as e:
                logger.warning(f"Could not calculate log loss: {e}")
                metrics['log_loss'] = None

        # Detailed classification report (optional)
        if include_report:
            metrics['report'] = classification_report(y_true, y_pred)

    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise

    return metrics


def format_metrics(metrics: Dict[str, Any], decimal_places: int = 3) -> str:
    """
    Format metrics dictionary as readable string.

    Args:
        metrics: Dictionary of metrics from calculate_classification_metrics()
        decimal_places: Number of decimal places to display

    Returns:
        Formatted string representation of metrics

    Example:
        >>> metrics = {'accuracy': 0.85, 'precision': 0.82, 'f1': 0.83}
        >>> print(format_metrics(metrics))
        Accuracy: 0.850
        Precision: 0.820
        F1: 0.830
    """
    lines = []

    # Key metrics first
    key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for metric in key_metrics:
        if metric in metrics and metrics[metric] is not None:
            value = metrics[metric]
            if isinstance(value, (int, float)):
                lines.append(f"{metric.replace('_', ' ').title()}: {value:.{decimal_places}f}")

    # Confusion matrix
    if 'tp' in metrics:
        lines.append("")
        lines.append("Confusion Matrix:")
        lines.append(f"  TN: {metrics['tn']}  FP: {metrics['fp']}")
        lines.append(f"  FN: {metrics['fn']}  TP: {metrics['tp']}")

        if 'specificity' in metrics:
            lines.append(f"  Specificity: {metrics['specificity']:.{decimal_places}f}")

    return "\n".join(lines)


def compare_model_metrics(
    metrics_dict: Dict[str, Dict[str, Any]],
    primary_metric: str = 'accuracy'
) -> str:
    """
    Compare metrics across multiple models.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        primary_metric: Metric to highlight for comparison (default: 'accuracy')

    Returns:
        Formatted comparison string

    Example:
        >>> metrics = {
        ...     'Logistic Regression': {'accuracy': 0.85, 'f1': 0.82},
        ...     'Random Forest': {'accuracy': 0.88, 'f1': 0.86}
        ... }
        >>> print(compare_model_metrics(metrics))
    """
    if not metrics_dict:
        return "No models to compare"

    lines = ["Model Performance Comparison", "=" * 50]

    # Find best model by primary metric
    best_model = None
    best_score = -1

    for model_name, metrics in metrics_dict.items():
        if primary_metric in metrics and metrics[primary_metric] is not None:
            score = metrics[primary_metric]
            if score > best_score:
                best_score = score
                best_model = model_name

    # Display metrics for each model
    for model_name, metrics in metrics_dict.items():
        is_best = (model_name == best_model)
        marker = " ⭐ BEST" if is_best else ""

        lines.append(f"\n{model_name}{marker}")
        lines.append("-" * 40)

        # Show key metrics
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if metric in metrics and metrics[metric] is not None:
                value = metrics[metric]
                lines.append(f"  {metric.replace('_', ' ').title()}: {value:.3f}")

    return "\n".join(lines)


def calculate_betting_roi(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    confidence_threshold: float = 0.6,
    odds: float = -110,
    stake: float = 100.0
) -> Dict[str, float]:
    """
    Calculate betting ROI metrics.

    Simulates betting performance by only placing bets when model confidence
    exceeds threshold.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        confidence_threshold: Only bet when prob > threshold
        odds: American odds (default -110 means risk $110 to win $100)
        stake: Amount to bet per game

    Returns:
        Dictionary with:
        - total_bets: Number of bets placed
        - wins: Number of winning bets
        - losses: Number of losing bets
        - total_wagered: Total amount bet
        - total_profit: Net profit/loss
        - roi: Return on investment (%)
        - win_rate: Percentage of bets won

    Example:
        >>> y_true = np.array([1, 0, 1, 1, 0])
        >>> y_pred = np.array([1, 0, 1, 0, 0])
        >>> y_prob = np.array([0.8, 0.7, 0.9, 0.5, 0.6])
        >>> roi_metrics = calculate_betting_roi(y_true, y_pred, y_prob)
    """
    # Filter predictions by confidence threshold
    confident_bets = y_prob > confidence_threshold

    if not np.any(confident_bets):
        return {
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'total_wagered': 0.0,
            'total_profit': 0.0,
            'roi': 0.0,
            'win_rate': 0.0
        }

    # Get confident predictions
    y_true_bet = y_true[confident_bets]
    y_pred_bet = y_pred[confident_bets]

    # Calculate wins and losses
    wins = np.sum(y_true_bet == y_pred_bet)
    total_bets = len(y_true_bet)
    losses = total_bets - wins

    # Calculate profit
    # For American odds: if odds < 0, profit = stake * (100 / abs(odds))
    if odds < 0:
        win_profit = stake * (100 / abs(odds))
    else:
        win_profit = stake * (odds / 100)

    total_profit = (wins * win_profit) - (losses * stake)
    total_wagered = total_bets * stake
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0.0
    win_rate = (wins / total_bets * 100) if total_bets > 0 else 0.0

    return {
        'total_bets': total_bets,
        'wins': wins,
        'losses': losses,
        'total_wagered': total_wagered,
        'total_profit': total_profit,
        'roi': roi,
        'win_rate': win_rate
    }
