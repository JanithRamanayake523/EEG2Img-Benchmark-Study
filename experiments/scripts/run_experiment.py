"""
Script to run a single experiment from configuration file.

Usage:
    python run_experiment.py --config configs/experiment_baseline.yaml
    python run_experiment.py --config configs/experiment_model_comparison.yaml
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.experiments import ExperimentConfig, ExperimentRunner, load_config


def load_data(filepath: str, split_ratio: float = 0.8, validation_split: float = 0.1):
    """
    Load data from HDF5 file.

    Args:
        filepath: Path to HDF5 file
        split_ratio: Train/test split ratio
        validation_split: Validation split from training data

    Returns:
        Tuple of (train_data, train_labels, val_data, val_labels, test_data, test_labels)
    """
    print(f"Loading data from {filepath}...")

    with h5py.File(filepath, 'r') as f:
        # Load images and labels
        if 'images' in f:
            data = f['images'][:]
        else:
            data = f['data'][:]

        labels = f['labels'][:]

    print(f"Loaded data: {data.shape}, labels: {labels.shape}")

    # Split data
    n_samples = len(data)
    train_size = int(n_samples * split_ratio)
    val_size = int(train_size * validation_split)
    train_size = train_size - val_size

    # Shuffle indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_data = data[train_indices]
    train_labels = labels[train_indices]

    val_data = data[val_indices]
    val_labels = labels[val_indices]

    test_data = data[test_indices]
    test_labels = labels[test_indices]

    print(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def main():
    """Run experiment from configuration."""
    parser = argparse.ArgumentParser(description='Run EEG experiment from configuration')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to experiment configuration file (YAML or JSON)')
    parser.add_argument('--data', type=str, default='data/BCI_IV_2a.hdf5',
                       help='Path to data file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Load configuration
    print("="*80)
    print("Loading experiment configuration...")
    print("="*80)

    config = load_config(args.config)
    print(f"Configuration: {config}")
    print()

    # Load data
    print("="*80)
    print("Loading data...")
    print("="*80)

    try:
        train_data, train_labels, val_data, val_labels, test_data, test_labels = \
            load_data(args.data,
                     split_ratio=config.dataset.split_ratio,
                     validation_split=config.dataset.validation_split)
    except FileNotFoundError:
        print(f"Error: Data file not found: {args.data}")
        print("Using dummy data for demonstration...")

        # Create dummy data for testing
        np.random.seed(config.seed)
        n_samples = 200
        train_data = np.random.randn(n_samples, 25, 64, 64).astype(np.float32)
        train_labels = np.random.randint(0, 4, n_samples)

        val_data = np.random.randn(50, 25, 64, 64).astype(np.float32)
        val_labels = np.random.randint(0, 4, 50)

        test_data = np.random.randn(50, 25, 64, 64).astype(np.float32)
        test_labels = np.random.randint(0, 4, 50)

    print()

    # Create runner
    print("="*80)
    print("Initializing experiment runner...")
    print("="*80)

    runner = ExperimentRunner(config, output_dir=args.output)
    print()

    # Run experiment
    print("="*80)
    print("Running experiment...")
    print("="*80)

    results = runner.run(
        train_data, train_labels,
        val_data, val_labels,
        test_data, test_labels
    )

    # Summary
    print("\n" + "="*80)
    print("Experiment Summary")
    print("="*80)

    for model_name, model_results in results['models'].items():
        print(f"\n{model_name}:")
        if 'test_metrics' in model_results:
            metrics = model_results['test_metrics']
            print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
            print(f"  F1-score: {metrics.get('f1', 'N/A'):.4f}")
            print(f"  AUC: {metrics.get('auc', 'N/A'):.4f}")
        else:
            print("  (No test metrics available)")

    print("\n" + "="*80)
    print(f"Results saved to: {runner.output_dir}")
    print("="*80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
