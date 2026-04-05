"""
Test script for Phase 7 experiment orchestration components.

Tests configuration loading, experiment running, and grid search.
"""

import sys
import json
from pathlib import Path
import tempfile
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.experiments import (
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    AugmentationConfig,
    EvaluationConfig,
    load_config,
    save_config,
    ExperimentRunner,
    GridSearch,
    RandomSearch,
    create_baseline_experiments,
    create_augmentation_experiments,
    create_hyperparameter_tuning_experiments
)


def test_config_loading():
    """Test configuration loading and saving."""
    print("\n" + "="*60)
    print("Testing Configuration Management")
    print("="*60)

    # Test loading from YAML
    print("\n--- Testing YAML configuration loading ---")
    try:
        config = load_config('configs/experiment_baseline.yaml')
        print(f"  [OK] Loaded config: {config.name}")
        print(f"      Models: {len(config.models)}")
        print(f"      Epochs: {config.training.epochs}")
    except Exception as e:
        print(f"  [FAIL] Failed to load config: {e}")
        return False

    # Test saving to JSON
    print("\n--- Testing configuration saving ---")
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_config.json'
            config.to_json(save_path)
            print(f"  [OK] Saved config to {save_path}")

            # Load back
            loaded_config = ExperimentConfig.from_json(save_path)
            print(f"  [OK] Reloaded config: {loaded_config.name}")
    except Exception as e:
        print(f"  [FAIL] Failed to save/load config: {e}")
        return False

    # Test config dict conversion
    print("\n--- Testing configuration to_dict ---")
    try:
        config_dict = config.to_dict()
        print(f"  [OK] Config dict created with {len(config_dict)} keys")
        assert 'name' in config_dict
        assert 'models' in config_dict
        assert 'training' in config_dict
    except Exception as e:
        print(f"  [FAIL] Failed to convert config to dict: {e}")
        return False

    return True


def test_experiment_config():
    """Test experiment configuration building."""
    print("\n" + "="*60)
    print("Testing Experiment Configuration")
    print("="*60)

    print("\n--- Creating experiment config programmatically ---")
    try:
        config = ExperimentConfig(
            name='test_experiment',
            description='Test configuration',
            seed=42,
            device='cuda',
            save_dir='results',
            dataset={
                'name': 'BCI-IV-2a',
                'file_path': 'data/test.hdf5',
                'split_ratio': 0.8,
                'validation_split': 0.1
            },
            augmentation={
                'enabled': True,
                'transforms': [
                    {'name': 'rotation', 'enabled': True, 'degrees': 15}
                ],
                'mixup_alpha': 1.0
            },
            models=[
                {
                    'name': 'resnet18',
                    'architecture': 'resnet18',
                    'num_classes': 4,
                    'in_channels': 25
                }
            ],
            optimizer={
                'name': 'adam',
                'learning_rate': 0.001,
                'weight_decay': 1e-4
            },
            training={
                'epochs': 100,
                'batch_size': 32,
                'early_stopping_patience': 10
            },
            evaluation={
                'metrics': ['accuracy', 'f1', 'auc'],
                'robustness_tests': True
            }
        )

        print(f"  [OK] Created config: {config}")
        print(f"      Dataset: {config.dataset.name}")
        print(f"      Models: {[m.name for m in config.models]}")
        print(f"      Augmentation enabled: {config.augmentation.enabled}")

    except Exception as e:
        print(f"  [FAIL] Failed to create config: {e}")
        return False

    return True


def test_grid_search():
    """Test grid search utilities."""
    print("\n" + "="*60)
    print("Testing Grid Search")
    print("="*60)

    print("\n--- Testing GridSearch ---")
    try:
        param_grid = {
            'models': ['resnet18', 'vit_tiny'],
            'batch_sizes': [32, 64],
            'learning_rates': [0.001, 0.0001]
        }

        gs = GridSearch(param_grid)
        print(f"  [OK] Created GridSearch with {len(gs)} combinations")

        # Test iteration
        combos = list(gs)
        print(f"  [OK] Generated {len(combos)} combinations")
        print(f"      First: {combos[0]}")
        print(f"      Last: {combos[-1]}")

    except Exception as e:
        print(f"  [FAIL] GridSearch failed: {e}")
        return False

    print("\n--- Testing RandomSearch ---")
    try:
        rs = RandomSearch(param_grid, n_iter=5, random_state=42)
        print(f"  [OK] Created RandomSearch with {len(rs)} iterations")

        combos = list(rs)
        print(f"  [OK] Generated {len(combos)} random combinations")

    except Exception as e:
        print(f"  [FAIL] RandomSearch failed: {e}")
        return False

    return True


def test_experiment_factories():
    """Test experiment factory functions."""
    print("\n" + "="*60)
    print("Testing Experiment Factories")
    print("="*60)

    print("\n--- Testing create_baseline_experiments ---")
    try:
        configs = create_baseline_experiments()
        print(f"  [OK] Created {len(configs)} baseline configurations")
        print(f"      Models: {[c.models[0].architecture for c in configs[:3]]}...")
    except Exception as e:
        print(f"  [FAIL] create_baseline_experiments failed: {e}")
        return False

    print("\n--- Testing create_augmentation_experiments ---")
    try:
        configs = create_augmentation_experiments()
        print(f"  [OK] Created {len(configs)} augmentation configurations")
        print(f"      First: {configs[0].name}")
        print(f"      Last: {configs[-1].name}")
    except Exception as e:
        print(f"  [FAIL] create_augmentation_experiments failed: {e}")
        return False

    print("\n--- Testing create_hyperparameter_tuning_experiments ---")
    try:
        configs = create_hyperparameter_tuning_experiments()
        print(f"  [OK] Created {len(configs)} hyperparameter configurations")
    except Exception as e:
        print(f"  [FAIL] create_hyperparameter_tuning_experiments failed: {e}")
        return False

    return True


def test_experiment_runner():
    """Test experiment runner."""
    print("\n" + "="*60)
    print("Testing Experiment Runner")
    print("="*60)

    print("\n--- Creating dummy data ---")
    try:
        np.random.seed(42)
        train_data = np.random.randn(50, 25, 64, 64).astype(np.float32)
        train_labels = np.random.randint(0, 4, 50)

        val_data = np.random.randn(20, 25, 64, 64).astype(np.float32)
        val_labels = np.random.randint(0, 4, 20)

        test_data = np.random.randn(20, 25, 64, 64).astype(np.float32)
        test_labels = np.random.randint(0, 4, 20)

        print(f"  [OK] Created dummy data")
        print(f"      Train: {train_data.shape}")
        print(f"      Val: {val_data.shape}")
        print(f"      Test: {test_data.shape}")

    except Exception as e:
        print(f"  [FAIL] Failed to create dummy data: {e}")
        return False

    print("\n--- Testing ExperimentRunner initialization ---")
    output_dir = None
    try:
        config = ExperimentConfig(
            name='test_runner',
            device='cpu',  # Use CPU for testing
            save_dir='results',
            dataset={'name': 'test', 'file_path': 'test.hdf5'},
            models=[
                {'name': 'lightweight_cnn', 'architecture': 'lightweight_cnn',
                 'num_classes': 4, 'in_channels': 25}
            ],
            training={'epochs': 1, 'batch_size': 16},  # 1 epoch for quick test
        )

        output_dir = tempfile.mkdtemp()
        runner = ExperimentRunner(config, output_dir=output_dir)
        print(f"  [OK] Created ExperimentRunner")
        print(f"      Output dir: {runner.output_dir}")

    except Exception as e:
        print(f"  [FAIL] Failed to create runner: {e}")
        return False

    print("\n--- Testing experiment execution (quick run) ---")
    try:
        # Quick test with minimal data
        results = runner.run(
            train_data, train_labels,
            val_data, val_labels,
            test_data, test_labels
        )

        print(f"  [OK] Experiment executed")
        print(f"      Models trained: {len(results['models'])}")
        if results['models']:
            first_model = list(results['models'].keys())[0]
            print(f"      First model: {first_model}")

    except Exception as e:
        print(f"  [FAIL] Experiment execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if output_dir:
            import shutil
            try:
                shutil.rmtree(output_dir, ignore_errors=True)
            except:
                pass

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHASE 7 EXPERIMENT ORCHESTRATION VALIDATION")
    print("="*80)

    results = {}

    # Run tests
    results['config_loading'] = test_config_loading()
    results['experiment_config'] = test_experiment_config()
    results['grid_search'] = test_grid_search()
    results['experiment_factories'] = test_experiment_factories()
    results['experiment_runner'] = test_experiment_runner()

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
        print("[OK] ALL TESTS PASSED - Phase 7 Experiment Orchestration Validated")
    else:
        print("[FAIL] SOME TESTS FAILED - Review errors above")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
