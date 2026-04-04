"""
Evaluation and analysis module for EEG classification models.

This module provides comprehensive evaluation tools including:
- Performance metrics (accuracy, F1, AUC, confusion matrix)
- Statistical significance testing (Wilcoxon, ANOVA)
- Robustness testing (noise injection, channel dropout)
- Visualization utilities
"""

from .metrics import (
    compute_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    compute_multiclass_auc,
    MetricsTracker
)

from .statistical import (
    wilcoxon_test,
    paired_t_test,
    anova_test,
    posthoc_tests,
    multiple_comparison_correction,
    compare_models,
    effect_size_cohens_d
)

from .robustness import (
    add_gaussian_noise,
    add_noise_and_evaluate,
    channel_dropout_test,
    temporal_shift_test,
    evaluate_robustness,
    channel_dropout,
    temporal_shift
)

from .visualization import (
    plot_confusion_matrix,
    plot_training_curves,
    plot_metrics_comparison,
    plot_robustness_curves,
    plot_statistical_comparison,
    plot_all_results
)

__all__ = [
    # Metrics
    'compute_metrics',
    'compute_confusion_matrix',
    'compute_per_class_metrics',
    'compute_multiclass_auc',
    'MetricsTracker',

    # Statistical tests
    'wilcoxon_test',
    'paired_t_test',
    'anova_test',
    'posthoc_tests',
    'multiple_comparison_correction',
    'compare_models',
    'effect_size_cohens_d',

    # Robustness testing
    'add_gaussian_noise',
    'add_noise_and_evaluate',
    'channel_dropout_test',
    'temporal_shift_test',
    'evaluate_robustness',
    'channel_dropout',
    'temporal_shift',

    # Visualization
    'plot_confusion_matrix',
    'plot_training_curves',
    'plot_metrics_comparison',
    'plot_robustness_curves',
    'plot_statistical_comparison',
    'plot_all_results',
]
