"""
Script to run grid search experiments.

Usage:
    python run_grid_search.py --type baseline
    python run_grid_search.py --type augmentation
    python run_grid_search.py --type hyperparameter
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.experiments import (
    ExperimentRunner,
    create_baseline_experiments,
    create_augmentation_experiments,
    create_hyperparameter_tuning_experiments
)


def load_data(filepath: str):
    """Load data from HDF5 file."""
    print(f"Loading data from {filepath}...")

    try:
        with h5py.File(filepath, 'r') as f:
            if 'images' in f:
                data = f['images'][:]
            else:
                data = f['data'][:]
            labels = f['labels'][:]

        print(f"Loaded data: {data.shape}, labels: {labels.shape}")
        return data, labels

    except FileNotFoundError:
        print(f"Warning: Data file not found: {filepath}")
        print("Using dummy data for demonstration...")

        # Create dummy data
        n_samples = 200
        data = np.random.randn(n_samples, 25, 64, 64).astype(np.float32)
        labels = np.random.randint(0, 4, n_samples)

        print(f"Created dummy data: {data.shape}, labels: {labels.shape}")
        return data, labels


def split_data(data, labels, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split data into train/val/test sets."""
    n_samples = len(data)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    return (data[train_idx], labels[train_idx],
            data[val_idx], labels[val_idx],
            data[test_idx], labels[test_idx])


def run_experiments(configs, data, labels, output_dir='results'):
    """Run multiple experiments."""
    print(f"\nRunning {len(configs)} experiments...")
    print("="*80)

    results_summary = {}

    for i, config in enumerate(tqdm(configs, desc='Experiments')):
        print(f"\n[{i+1}/{len(configs)}] Running: {config.name}")

        # Split data
        train_data, train_labels, val_data, val_labels, test_data, test_labels = \
            split_data(data, labels,
                      train_ratio=config.dataset.split_ratio,
                      val_ratio=config.dataset.validation_split)

        # Create and run runner
        runner = ExperimentRunner(config, output_dir=output_dir)

        try:
            results = runner.run(
                train_data, train_labels,
                val_data, val_labels,
                test_data, test_labels
            )

            # Extract summary
            results_summary[config.name] = {
                'status': 'success',
                'models': {}
            }

            for model_name, model_results in results['models'].items():
                if 'test_metrics' in model_results:
                    metrics = model_results['test_metrics']
                    results_summary[config.name]['models'][model_name] = {
                        'accuracy': metrics.get('accuracy', 0.0),
                        'f1': metrics.get('f1', 0.0),
                        'auc': metrics.get('auc', 0.0)
                    }

        except Exception as e:
            print(f"Error: {e}")
            results_summary[config.name] = {
                'status': 'failed',
                'error': str(e)
            }

    return results_summary


def print_summary(results_summary):
    """Print experiment summary."""
    print("\n" + "="*80)
    print("Grid Search Summary")
    print("="*80)

    for config_name, config_results in results_summary.items():
        print(f"\n{config_name}: {config_results['status']}")

        if config_results['status'] == 'success':
            for model_name, metrics in config_results['models'].items():
                print(f"  {model_name}:")
                print(f"    Accuracy: {metrics.get('accuracy', 0.0):.4f}")
                print(f"    F1: {metrics.get('f1', 0.0):.4f}")
                print(f"    AUC: {metrics.get('auc', 0.0):.4f}")
        else:
            print(f"  Error: {config_results.get('error', 'Unknown')}")

    print("\n" + "="*80)


def main():
    """Run grid search."""
    parser = argparse.ArgumentParser(description='Run grid search experiments')
    parser.add_argument('--type', type=str, default='baseline',
                       choices=['baseline', 'augmentation', 'hyperparameter'],
                       help='Type of grid search to run')
    parser.add_argument('--data', type=str, default='data/BCI_IV_2a.hdf5',
                       help='Path to data file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Create experiment configurations
    print("="*80)
    print(f"Creating {args.type} experiments...")
    print("="*80)

    if args.type == 'baseline':
        configs = create_baseline_experiments()
        print(f"Created {len(configs)} baseline configurations")

    elif args.type == 'augmentation':
        configs = create_augmentation_experiments()
        print(f"Created {len(configs)} augmentation configurations")

    elif args.type == 'hyperparameter':
        configs = create_hyperparameter_tuning_experiments()
        print(f"Created {len(configs)} hyperparameter configurations")

    else:
        print(f"Unknown type: {args.type}")
        return 1

    print()

    # Load data
    print("="*80)
    print("Loading data...")
    print("="*80)

    data, labels = load_data(args.data)
    print()

    # Run experiments
    results_summary = run_experiments(configs, data, labels, output_dir=args.output)

    # Print summary
    print_summary(results_summary)

    return 0


if __name__ == '__main__':
    sys.exit(main())
