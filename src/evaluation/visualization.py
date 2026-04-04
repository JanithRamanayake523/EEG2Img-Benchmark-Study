"""
Visualization utilities for evaluation results.

Provides plotting functions for:
- Confusion matrices
- Training curves (loss/accuracy over epochs)
- Model comparison bar charts and box plots
- Robustness curves (performance vs noise/dropout)
- Statistical comparison plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import pandas as pd


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = 'Confusion Matrix',
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix, shape (n_classes, n_classes)
        class_names: List of class names for axis labels
        normalize: If True, normalize by true label counts
        title: Plot title
        cmap: Colormap name
        figsize: Figure size (width, height)
        save_path: Path to save figure, optional
        show: If True, display the plot

    Example:
        >>> from sklearn.metrics import confusion_matrix
        >>> y_true = [0, 1, 2, 0, 1, 2]
        >>> y_pred = [0, 2, 2, 0, 1, 1]
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> plot_confusion_matrix(cm, class_names=['Class A', 'Class B', 'Class C'])
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap=cmap,
        square=True,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        xticklabels=class_names if class_names else range(cm.shape[1]),
        yticklabels=class_names if class_names else range(cm.shape[0]),
        ax=ax
    )

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_training_curves(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with training history
                 e.g., {'train_loss': [...], 'val_loss': [...],
                        'train_acc': [...], 'val_acc': [...]}
        metrics: List of metrics to plot (e.g., ['loss', 'acc'])
                If None, plot all available metrics
        figsize: Figure size
        save_path: Path to save figure
        show: If True, display plot

    Example:
        >>> history = {
        ...     'train_loss': [1.2, 0.8, 0.5, 0.3],
        ...     'val_loss': [1.3, 0.9, 0.6, 0.4],
        ...     'train_acc': [0.5, 0.7, 0.8, 0.9],
        ...     'val_acc': [0.48, 0.68, 0.78, 0.85]
        ... }
        >>> plot_training_curves(history, metrics=['loss', 'acc'])
    """
    # Determine metrics to plot
    if metrics is None:
        # Extract unique metric names
        metrics = set()
        for key in history.keys():
            if key.startswith('train_'):
                metrics.add(key.replace('train_', ''))
            elif key.startswith('val_'):
                metrics.add(key.replace('val_', ''))
        metrics = sorted(list(metrics))

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'

        # Plot training curve
        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], 'b-', label='Training', linewidth=2)

        # Plot validation curve
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], 'r-', label='Validation', linewidth=2)

        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric.capitalize(), fontsize=10)
        ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_metrics_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    plot_type: str = 'bar',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot comparison of metric values across models/methods.

    Args:
        results: Nested dictionary of results
                e.g., {'Model A': {'mean': 0.85, 'std': 0.03, 'values': [...]},
                       'Model B': {'mean': 0.87, 'std': 0.02, 'values': [...]}}
        metric: Metric name (used in title)
        plot_type: Type of plot ('bar' or 'box')
        figsize: Figure size
        save_path: Path to save figure
        show: If True, display plot

    Example:
        >>> results = {
        ...     'ResNet': {'mean': 0.85, 'std': 0.03, 'values': [0.82, 0.85, 0.87]},
        ...     'ViT': {'mean': 0.88, 'std': 0.02, 'values': [0.86, 0.88, 0.90]},
        ...     'Baseline': {'mean': 0.75, 'std': 0.04, 'values': [0.71, 0.75, 0.79]}
        ... }
        >>> plot_metrics_comparison(results, metric='Accuracy', plot_type='bar')
    """
    fig, ax = plt.subplots(figsize=figsize)

    model_names = list(results.keys())

    if plot_type == 'bar':
        # Bar chart with error bars
        means = [results[name]['mean'] for name in model_names]
        stds = [results[name]['std'] for name in model_names]

        x = np.arange(len(model_names))
        bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black')

        # Color bars
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    elif plot_type == 'box':
        # Box plot
        data = [results[name]['values'] for name in model_names]
        bp = ax.boxplot(data, labels=model_names, patch_artist=True, showmeans=True)

        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_robustness_curves(
    results: Dict[str, Dict[float, Dict[str, float]]],
    test_type: str,
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot robustness curves showing performance degradation.

    Args:
        results: Nested dictionary of robustness results
                 e.g., {'Model A': {20: {'accuracy': 0.85}, 10: {'accuracy': 0.80}, ...}}
        test_type: Type of robustness test ('noise', 'dropout', 'shift')
        metric: Metric to plot
        figsize: Figure size
        save_path: Path to save figure
        show: If True, display plot

    Example:
        >>> results = {
        ...     'ResNet': {20: {'accuracy': 0.85}, 10: {'accuracy': 0.80}, 0: {'accuracy': 0.70}},
        ...     'ViT': {20: {'accuracy': 0.88}, 10: {'accuracy': 0.84}, 0: {'accuracy': 0.75}}
        ... }
        >>> plot_robustness_curves(results, test_type='noise', metric='accuracy')
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot curves for each model
    for model_name, model_results in results.items():
        # Sort by x values
        x_values = sorted(model_results.keys())
        y_values = [model_results[x][metric] for x in x_values]

        # Check if std is available
        has_std = 'std' in list(model_results.values())[0]
        if has_std:
            y_std = [model_results[x]['std'] for x in x_values]
            ax.errorbar(x_values, y_values, yerr=y_std, marker='o', label=model_name,
                       linewidth=2, capsize=5, markersize=8)
        else:
            ax.plot(x_values, y_values, marker='o', label=model_name,
                   linewidth=2, markersize=8)

    # Customize based on test type
    if test_type == 'noise':
        ax.set_xlabel('SNR (dB)', fontsize=12)
        title = f'{metric.capitalize()} vs Noise Level'
        ax.invert_xaxis()  # Higher SNR (less noise) on left
    elif test_type == 'dropout':
        ax.set_xlabel('Channel Dropout Rate', fontsize=12)
        title = f'{metric.capitalize()} vs Channel Dropout'
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x*100:.0f}%'))
    elif test_type == 'shift':
        ax.set_xlabel('Max Temporal Shift (ms)', fontsize=12)
        title = f'{metric.capitalize()} vs Temporal Shift'
    else:
        ax.set_xlabel('Perturbation Level', fontsize=12)
        title = f'{metric.capitalize()} vs Perturbation'

    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_statistical_comparison(
    posthoc_df: pd.DataFrame,
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    Plot pairwise statistical comparison results.

    Args:
        posthoc_df: DataFrame from posthoc_tests function
        metric: Metric name for title
        figsize: Figure size
        save_path: Path to save figure
        show: If True, display plot

    Example:
        >>> from src.evaluation.statistical import posthoc_tests
        >>> results = {...}
        >>> posthoc_df = posthoc_tests(results)
        >>> plot_statistical_comparison(posthoc_df, metric='Accuracy')
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create comparison labels
    comparisons = [f"{row['model_a']} vs\n{row['model_b']}"
                  for _, row in posthoc_df.iterrows()]

    # Get p-values and significance
    p_values = posthoc_df['p_corrected'].values
    significant = posthoc_df['significant'].values

    # Create bar chart
    colors = ['red' if sig else 'gray' for sig in significant]
    bars = ax.barh(comparisons, -np.log10(p_values), color=colors, alpha=0.7, edgecolor='black')

    # Add significance threshold line
    ax.axvline(-np.log10(0.05), color='black', linestyle='--', linewidth=2, label='α = 0.05')

    ax.set_xlabel('-log10(p-value)', fontsize=12)
    ax.set_ylabel('Comparison', fontsize=12)
    ax.set_title(f'Statistical Significance of {metric.capitalize()} Differences',
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    # Annotate bars with actual p-values
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        width = bar.get_width()
        label = f'p={p_val:.4f}'
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
               label, va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_all_results(
    metrics_summary: Dict,
    confusion_matrices: Dict[str, np.ndarray],
    robustness_results: Dict,
    class_names: Optional[List[str]] = None,
    save_dir: Optional[Union[str, Path]] = None
):
    """
    Create comprehensive visualization of all results.

    Args:
        metrics_summary: Summary of metrics across models
        confusion_matrices: Dictionary mapping model names to confusion matrices
        robustness_results: Dictionary of robustness test results per model
        class_names: List of class names
        save_dir: Directory to save all figures

    Example:
        >>> metrics_summary = {...}
        >>> confusion_matrices = {'ResNet': cm1, 'ViT': cm2}
        >>> robustness_results = {'ResNet': {...}, 'ViT': {...}}
        >>> plot_all_results(metrics_summary, confusion_matrices, robustness_results,
        ...                  save_dir='results/figures')
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot metrics comparison
    plot_metrics_comparison(
        metrics_summary,
        metric='accuracy',
        plot_type='bar',
        save_path=save_dir / 'accuracy_comparison.png' if save_dir else None,
        show=False
    )

    plot_metrics_comparison(
        metrics_summary,
        metric='f1',
        plot_type='box',
        save_path=save_dir / 'f1_distribution.png' if save_dir else None,
        show=False
    )

    # 2. Plot confusion matrices
    for model_name, cm in confusion_matrices.items():
        plot_confusion_matrix(
            cm,
            class_names=class_names,
            normalize=True,
            title=f'Confusion Matrix - {model_name}',
            save_path=save_dir / f'cm_{model_name.lower().replace(" ", "_")}.png' if save_dir else None,
            show=False
        )

    # 3. Plot robustness curves
    if 'noise' in list(robustness_results.values())[0]:
        noise_results = {name: res['noise'] for name, res in robustness_results.items()}
        plot_robustness_curves(
            noise_results,
            test_type='noise',
            save_path=save_dir / 'robustness_noise.png' if save_dir else None,
            show=False
        )

    if 'dropout' in list(robustness_results.values())[0]:
        dropout_results = {name: res['dropout'] for name, res in robustness_results.items()}
        plot_robustness_curves(
            dropout_results,
            test_type='dropout',
            save_path=save_dir / 'robustness_dropout.png' if save_dir else None,
            show=False
        )

    if 'shift' in list(robustness_results.values())[0]:
        shift_results = {name: res['shift'] for name, res in robustness_results.items()}
        plot_robustness_curves(
            shift_results,
            test_type='shift',
            save_path=save_dir / 'robustness_shift.png' if save_dir else None,
            show=False
        )

    print(f"All plots saved to {save_dir}")
