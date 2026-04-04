"""
Test script for Phase 6 evaluation components.

Tests metrics, statistical tests, robustness testing, and visualization.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.nn as nn
from src.evaluation import (
    # Metrics
    compute_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    compute_multiclass_auc,
    MetricsTracker,
    # Statistical
    wilcoxon_test,
    paired_t_test,
    anova_test,
    posthoc_tests,
    compare_models,
    effect_size_cohens_d,
    # Robustness
    add_gaussian_noise,
    channel_dropout,
    temporal_shift,
    # Visualization
    plot_confusion_matrix,
    plot_training_curves,
    plot_metrics_comparison
)
from src.models import get_model


def test_metrics():
    """Test metrics computation."""
    print("\n" + "="*60)
    print("Testing Metrics")
    print("="*60)

    # Generate dummy predictions
    np.random.seed(42)
    n_samples = 100
    n_classes = 4

    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.random.randint(0, n_classes, n_samples)
    y_proba = np.random.rand(n_samples, n_classes)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)  # Normalize

    # Test compute_metrics
    print("\n--- Testing compute_metrics ---")
    metrics = compute_metrics(y_true, y_pred, y_proba)
    print(f"  [OK] Computed {len(metrics)} metrics")
    print(f"      Accuracy: {metrics['accuracy']:.4f}")
    print(f"      F1: {metrics['f1']:.4f}")
    print(f"      AUC: {metrics['auc']:.4f}")

    # Test confusion matrix
    print("\n--- Testing compute_confusion_matrix ---")
    cm = compute_confusion_matrix(y_true, y_pred)
    print(f"  [OK] Confusion matrix shape: {cm.shape}")

    # Test per-class metrics
    print("\n--- Testing compute_per_class_metrics ---")
    per_class = compute_per_class_metrics(y_true, y_pred,
                                          target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    print(f"  [OK] Computed metrics for {len(per_class)} classes")

    # Test MetricsTracker
    print("\n--- Testing MetricsTracker ---")
    tracker = MetricsTracker()
    for fold in range(5):
        # Simulate fold results
        fold_y_true = np.random.randint(0, n_classes, 50)
        fold_y_pred = np.random.randint(0, n_classes, 50)
        fold_y_proba = np.random.rand(50, n_classes)
        fold_y_proba = fold_y_proba / fold_y_proba.sum(axis=1, keepdims=True)

        fold_metrics = compute_metrics(fold_y_true, fold_y_pred, fold_y_proba)
        tracker.update(fold_metrics)

    summary = tracker.summary()
    print(f"  [OK] Tracked {len(tracker)} folds")
    print(f"      Mean accuracy: {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")

    return True


def test_statistical():
    """Test statistical tests."""
    print("\n" + "="*60)
    print("Testing Statistical Tests")
    print("="*60)

    # Generate dummy results
    np.random.seed(42)
    results_a = [0.85, 0.87, 0.82, 0.89, 0.84]
    results_b = [0.82, 0.84, 0.80, 0.86, 0.81]
    results_c = [0.75, 0.77, 0.73, 0.78, 0.76]

    # Test Wilcoxon
    print("\n--- Testing wilcoxon_test ---")
    stat, p = wilcoxon_test(results_a, results_b)
    print(f"  [OK] Wilcoxon test: statistic={stat:.4f}, p-value={p:.4f}")

    # Test paired t-test
    print("\n--- Testing paired_t_test ---")
    stat, p = paired_t_test(results_a, results_b)
    print(f"  [OK] Paired t-test: t-statistic={stat:.4f}, p-value={p:.4f}")

    # Test ANOVA
    print("\n--- Testing anova_test ---")
    results_dict = {
        'Model A': results_a,
        'Model B': results_b,
        'Model C': results_c
    }
    anova_result = anova_test(results_dict, repeated_measures=True)
    print(f"  [OK] ANOVA: F={anova_result['F']:.4f}, p-value={anova_result['p_value']:.4f}")

    # Test post-hoc tests
    print("\n--- Testing posthoc_tests ---")
    posthoc_df = posthoc_tests(results_dict, test='wilcoxon', correction='bonferroni')
    print(f"  [OK] Post-hoc tests: {len(posthoc_df)} comparisons")
    for _, row in posthoc_df.iterrows():
        print(f"      {row['model_a']} vs {row['model_b']}: p={row['p_corrected']:.4f}")

    # Test effect size
    print("\n--- Testing effect_size_cohens_d ---")
    d = effect_size_cohens_d(results_a, results_b)
    print(f"  [OK] Cohen's d: {d:.4f}")

    return True


def test_robustness():
    """Test robustness utilities."""
    print("\n" + "="*60)
    print("Testing Robustness Utilities")
    print("="*60)

    # Generate dummy EEG data
    np.random.seed(42)
    n_samples = 50
    n_channels = 25
    n_timepoints = 751

    eeg_data = np.random.randn(n_samples, n_channels, n_timepoints)

    # Test Gaussian noise
    print("\n--- Testing add_gaussian_noise ---")
    noisy_data = add_gaussian_noise(eeg_data, snr_db=10)
    print(f"  [OK] Added noise at SNR=10 dB")
    print(f"      Original shape: {eeg_data.shape}")
    print(f"      Noisy shape: {noisy_data.shape}")

    # Test channel dropout
    print("\n--- Testing channel_dropout ---")
    dropped_data = channel_dropout(eeg_data, dropout_rate=0.2, channel_axis=1)
    print(f"  [OK] Applied 20% channel dropout")
    print(f"      Dropped channels shape: {dropped_data.shape}")

    # Test temporal shift
    print("\n--- Testing temporal_shift ---")
    shifted_data = temporal_shift(eeg_data, max_shift_samples=50, time_axis=-1)
    print(f"  [OK] Applied temporal shift (±50 samples)")
    print(f"      Shifted shape: {shifted_data.shape}")

    return True


def test_visualization():
    """Test visualization utilities."""
    print("\n" + "="*60)
    print("Testing Visualization")
    print("="*60)

    # Generate dummy data
    np.random.seed(42)

    # Test confusion matrix plot
    print("\n--- Testing plot_confusion_matrix ---")
    cm = np.array([[50, 2, 0, 3],
                   [5, 45, 2, 1],
                   [0, 3, 48, 2],
                   [2, 1, 1, 49]])
    try:
        plot_confusion_matrix(cm, class_names=['Left', 'Right', 'Feet', 'Tongue'],
                             normalize=True, show=False)
        print(f"  [OK] Confusion matrix plot created")
    except Exception as e:
        print(f"  [FAIL] Confusion matrix plot failed: {e}")
        return False

    # Test training curves plot
    print("\n--- Testing plot_training_curves ---")
    history = {
        'train_loss': [1.2, 0.8, 0.5, 0.3, 0.2],
        'val_loss': [1.3, 0.9, 0.6, 0.4, 0.3],
        'train_acc': [0.5, 0.7, 0.8, 0.9, 0.95],
        'val_acc': [0.48, 0.68, 0.78, 0.85, 0.88]
    }
    try:
        plot_training_curves(history, metrics=['loss', 'acc'], show=False)
        print(f"  [OK] Training curves plot created")
    except Exception as e:
        print(f"  [FAIL] Training curves plot failed: {e}")
        return False

    # Test metrics comparison plot
    print("\n--- Testing plot_metrics_comparison ---")
    results = {
        'ResNet': {'mean': 0.85, 'std': 0.03, 'values': [0.82, 0.85, 0.87, 0.84, 0.86]},
        'ViT': {'mean': 0.88, 'std': 0.02, 'values': [0.86, 0.88, 0.90, 0.87, 0.89]},
        'Baseline': {'mean': 0.75, 'std': 0.04, 'values': [0.71, 0.75, 0.79, 0.73, 0.77]}
    }
    try:
        plot_metrics_comparison(results, metric='Accuracy', plot_type='bar', show=False)
        print(f"  [OK] Metrics comparison plot created")
    except Exception as e:
        print(f"  [FAIL] Metrics comparison plot failed: {e}")
        return False

    return True


def test_integration():
    """Test end-to-end evaluation workflow."""
    print("\n" + "="*60)
    print("Testing Integration (End-to-End)")
    print("="*60)

    # Create a simple model
    print("\n--- Creating model ---")
    model = get_model('lightweight_cnn', num_classes=4, in_channels=25, pretrained=False)
    model.eval()
    print(f"  [OK] Model created")

    # Generate dummy test data
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 100
    test_data = torch.randn(n_samples, 25, 64, 64)
    test_labels = torch.randint(0, 4, (n_samples,))

    # Get predictions
    print("\n--- Getting predictions ---")
    with torch.no_grad():
        outputs = model(test_data)
        probs = torch.softmax(outputs, dim=1).numpy()
        preds = outputs.argmax(dim=1).numpy()

    print(f"  [OK] Predictions shape: {preds.shape}")

    # Compute metrics
    print("\n--- Computing metrics ---")
    metrics = compute_metrics(test_labels.numpy(), preds, probs)
    print(f"  [OK] Accuracy: {metrics['accuracy']:.4f}")
    print(f"      F1: {metrics['f1']:.4f}")
    print(f"      AUC: {metrics['auc']:.4f}")

    # Compute confusion matrix
    print("\n--- Computing confusion matrix ---")
    cm = compute_confusion_matrix(test_labels.numpy(), preds)
    print(f"  [OK] Confusion matrix:\n{cm}")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHASE 6 EVALUATION INFRASTRUCTURE VALIDATION")
    print("="*80)

    # Track results
    results = {}

    # Test metrics
    results['metrics'] = test_metrics()

    # Test statistical tests
    results['statistical'] = test_statistical()

    # Test robustness
    results['robustness'] = test_robustness()

    # Test visualization
    results['visualization'] = test_visualization()

    # Test integration
    results['integration'] = test_integration()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "[OK] PASSED" if passed else "[FAIL] FAILED"
        print(f"  {test_name}: {status}")

    print("\n" + "="*80)
    if all_passed:
        print("[OK] ALL TESTS PASSED - Phase 6 Evaluation Infrastructure Validated")
    else:
        print("[FAIL] SOME TESTS FAILED - Review errors above")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
