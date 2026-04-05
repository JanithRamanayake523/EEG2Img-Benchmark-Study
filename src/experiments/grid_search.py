"""
Grid search utilities for hyperparameter tuning.

Provides grid search over model architectures, transforms, and hyperparameters.
"""

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Iterator, Tuple
from .config import (
    ExperimentConfig, ModelConfig, OptimizerConfig, TrainingConfig,
    AugmentationConfig, TransformConfig
)


class GridSearch:
    """
    Grid search over experiment configurations.

    Generates all combinations of specified parameter values.

    Example:
        >>> param_grid = {
        ...     'models': ['resnet18', 'vit_tiny'],
        ...     'batch_sizes': [32, 64],
        ...     'learning_rates': [0.001, 0.0001]
        ... }
        >>> gs = GridSearch(param_grid)
        >>> for config in gs:
        ...     print(config)
    """

    def __init__(self, param_grid: Dict[str, List[Any]]):
        """
        Initialize grid search.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
                       Keys can be:
                       - 'models': list of model architectures
                       - 'batch_sizes': list of batch sizes
                       - 'learning_rates': list of learning rates
                       - 'optimizers': list of optimizer names
                       - 'transforms': list of transform names
                       - 'epochs': list of epoch counts
                       - Custom combinations of above
        """
        self.param_grid = param_grid

        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]

        self.combinations = list(itertools.product(*param_values))
        self.param_names = param_names

    def __len__(self) -> int:
        """Return number of combinations."""
        return len(self.combinations)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over all combinations."""
        for combo in self.combinations:
            yield dict(zip(self.param_names, combo))

    def __repr__(self) -> str:
        """String representation."""
        return f"GridSearch({len(self)} combinations)"

    def to_configs(self, base_config: ExperimentConfig) -> List[ExperimentConfig]:
        """
        Convert grid search to list of experiment configs.

        Args:
            base_config: Base experiment configuration to modify

        Returns:
            List of ExperimentConfig instances

        Example:
            >>> base_config = ExperimentConfig.from_yaml('configs/base.yaml')
            >>> param_grid = {
            ...     'models': ['resnet18', 'vit_tiny'],
            ...     'batch_sizes': [32, 64]
            ... }
            >>> gs = GridSearch(param_grid)
            >>> configs = gs.to_configs(base_config)
            >>> print(f"Generated {len(configs)} configs")
        """
        configs = []

        for params in self:
            # Create copy of base config
            config_dict = base_config.to_dict()

            # Update parameters
            if 'models' in params and params['models']:
                # Update model configs
                models = params['models']
                if isinstance(models, str):
                    models = [models]
                elif not isinstance(models, list):
                    models = [models]

                config_dict['models'] = [
                    {
                        'name': m,
                        'architecture': m,
                        'num_classes': base_config.models[0].num_classes if base_config.models else 4,
                        'in_channels': base_config.models[0].in_channels if base_config.models else 25,
                    }
                    for m in models
                ]

            if 'batch_sizes' in params:
                if 'training' not in config_dict or config_dict['training'] is None:
                    config_dict['training'] = {}
                config_dict['training']['batch_size'] = params['batch_sizes']

            if 'learning_rates' in params:
                if 'optimizer' not in config_dict or config_dict['optimizer'] is None:
                    config_dict['optimizer'] = {}
                config_dict['optimizer']['learning_rate'] = params['learning_rates']

            if 'optimizers' in params:
                if 'optimizer' not in config_dict or config_dict['optimizer'] is None:
                    config_dict['optimizer'] = {}
                config_dict['optimizer']['name'] = params['optimizers']

            if 'epochs' in params:
                if 'training' not in config_dict or config_dict['training'] is None:
                    config_dict['training'] = {}
                config_dict['training']['epochs'] = params['epochs']

            if 'transforms' in params:
                if 'augmentation' not in config_dict or config_dict['augmentation'] is None:
                    config_dict['augmentation'] = {'enabled': True}
                config_dict['augmentation']['transforms'] = [
                    {'name': t, 'enabled': True}
                    for t in (params['transforms'] if isinstance(params['transforms'], list) else [params['transforms']])
                ]

            # Create config
            config = ExperimentConfig(**config_dict)

            # Update name to include parameters
            param_suffix = '_'.join(
                f"{k}={str(v)[:20]}"
                for k, v in params.items()
            )
            config.name = f"{base_config.name}_{param_suffix}"

            configs.append(config)

        return configs


class RandomSearch:
    """
    Random search over experiment configurations.

    Samples random combinations of parameters.
    """

    def __init__(self, param_grid: Dict[str, List[Any]], n_iter: int = 10,
                 random_state: int = 42):
        """
        Initialize random search.

        Args:
            param_grid: Dictionary mapping parameter names to lists of values
            n_iter: Number of random combinations to generate
            random_state: Random seed
        """
        import random

        self.param_grid = param_grid
        self.n_iter = n_iter
        self.random_state = random_state

        # Set random seed
        random.seed(random_state)

        # Generate random combinations
        param_names = list(param_grid.keys())
        self.combinations = []

        for _ in range(n_iter):
            combo = {
                name: random.choice(param_grid[name])
                for name in param_names
            }
            self.combinations.append(combo)

    def __len__(self) -> int:
        """Return number of iterations."""
        return len(self.combinations)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over random combinations."""
        return iter(self.combinations)

    def __repr__(self) -> str:
        """String representation."""
        return f"RandomSearch({len(self)} combinations)"

    def to_configs(self, base_config: ExperimentConfig) -> List[ExperimentConfig]:
        """
        Convert random search to list of experiment configs.

        Args:
            base_config: Base experiment configuration to modify

        Returns:
            List of ExperimentConfig instances
        """
        configs = []

        for i, params in enumerate(self):
            # Create copy of base config
            config_dict = base_config.to_dict()

            # Update parameters (same as GridSearch)
            if 'models' in params:
                config_dict['models'] = [
                    {
                        'name': params['models'],
                        'architecture': params['models'],
                        'num_classes': base_config.models[0].num_classes if base_config.models else 4,
                        'in_channels': base_config.models[0].in_channels if base_config.models else 25,
                    }
                ]

            if 'batch_sizes' in params:
                if 'training' not in config_dict or config_dict['training'] is None:
                    config_dict['training'] = {}
                config_dict['training']['batch_size'] = params['batch_sizes']

            if 'learning_rates' in params:
                if 'optimizer' not in config_dict or config_dict['optimizer'] is None:
                    config_dict['optimizer'] = {}
                config_dict['optimizer']['learning_rate'] = params['learning_rates']

            if 'optimizers' in params:
                if 'optimizer' not in config_dict or config_dict['optimizer'] is None:
                    config_dict['optimizer'] = {}
                config_dict['optimizer']['name'] = params['optimizers']

            if 'epochs' in params:
                if 'training' not in config_dict or config_dict['training'] is None:
                    config_dict['training'] = {}
                config_dict['training']['epochs'] = params['epochs']

            # Create config
            config = ExperimentConfig(**config_dict)
            config.name = f"{base_config.name}_random_{i:03d}"

            configs.append(config)

        return configs


def create_baseline_experiments() -> List[ExperimentConfig]:
    """
    Create baseline experiment configurations.

    Returns:
        List of baseline experiment configs for all models

    Example:
        >>> configs = create_baseline_experiments()
        >>> print(f"Created {len(configs)} baseline configs")
    """
    base_config = ExperimentConfig(
        name='baseline',
        description='Baseline experiments with default parameters',
        seed=42,
        device='cuda',
        save_dir='results',
        dataset={
            'name': 'BCI-IV-2a',
            'file_path': 'data/BCI_IV_2a.hdf5',
            'split_ratio': 0.8,
            'validation_split': 0.1
        },
        augmentation={
            'enabled': False,  # No augmentation for baseline
            'transforms': []
        },
        optimizer={
            'name': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 0.0
        },
        training={
            'epochs': 100,
            'batch_size': 32,
            'num_workers': 0,
            'early_stopping_patience': 10,
            'mixed_precision': False,
            'gradient_accumulation_steps': 1
        },
        evaluation={
            'metrics': ['accuracy', 'f1', 'auc'],
            'robustness_tests': False,
            'statistical_tests': False,
            'visualization': True
        }
    )

    # Models to evaluate
    models = [
        'resnet18',
        'resnet50',
        'vit_tiny',
        'vit_small',
        'lightweight_cnn',
        'cnn_1d',
        'lstm',
        'transformer',
        'eegnet'
    ]

    configs = []
    for model in models:
        config_dict = base_config.to_dict()
        config_dict['name'] = f"baseline_{model}"
        config_dict['models'] = [
            {
                'name': model,
                'architecture': model,
                'num_classes': 4,
                'in_channels': 25
            }
        ]

        config = ExperimentConfig(**config_dict)
        configs.append(config)

    return configs


def create_augmentation_experiments() -> List[ExperimentConfig]:
    """
    Create augmentation study experiment configurations.

    Returns:
        List of experiment configs testing different augmentation strategies
    """
    base_config = ExperimentConfig(
        name='augmentation_study',
        description='Evaluate different augmentation strategies',
        seed=42,
        device='cuda',
        save_dir='results',
        dataset={
            'name': 'BCI-IV-2a',
            'file_path': 'data/BCI_IV_2a.hdf5',
            'split_ratio': 0.8,
            'validation_split': 0.1
        },
        optimizer={
            'name': 'adam',
            'learning_rate': 0.001,
            'weight_decay': 1e-4
        },
        training={
            'epochs': 100,
            'batch_size': 32,
            'num_workers': 0,
            'early_stopping_patience': 10,
            'mixed_precision': False,
            'gradient_accumulation_steps': 1
        },
        evaluation={
            'metrics': ['accuracy', 'f1', 'auc'],
            'robustness_tests': True,
            'statistical_tests': True,
            'visualization': True
        }
    )

    # Augmentation strategies
    strategies = [
        {
            'name': 'no_augmentation',
            'transforms': []
        },
        {
            'name': 'basic_transforms',
            'transforms': [
                {'name': 'horizontal_flip', 'enabled': True, 'p': 0.5},
                {'name': 'rotation', 'enabled': True, 'degrees': 15}
            ]
        },
        {
            'name': 'with_mixup',
            'transforms': [
                {'name': 'horizontal_flip', 'enabled': True, 'p': 0.5},
                {'name': 'rotation', 'enabled': True, 'degrees': 15}
            ],
            'mixup_alpha': 1.0
        },
        {
            'name': 'with_cutmix',
            'transforms': [
                {'name': 'horizontal_flip', 'enabled': True, 'p': 0.5},
                {'name': 'rotation', 'enabled': True, 'degrees': 15}
            ],
            'cutmix_alpha': 1.0
        },
        {
            'name': 'full_augmentation',
            'transforms': [
                {'name': 'horizontal_flip', 'enabled': True, 'p': 0.5},
                {'name': 'rotation', 'enabled': True, 'degrees': 20},
                {'name': 'brightness', 'enabled': True, 'factor': 0.2},
                {'name': 'contrast', 'enabled': True, 'factor': 0.2}
            ],
            'mixup_alpha': 1.0,
            'cutmix_alpha': 1.0
        }
    ]

    configs = []
    for strategy in strategies:
        config_dict = base_config.to_dict()
        config_dict['name'] = f"augmentation_{strategy['name']}"
        config_dict['augmentation'] = {k: v for k, v in strategy.items() if k != 'name'}
        config_dict['augmentation']['enabled'] = len(config_dict['augmentation']['transforms']) > 0 or \
                                                 'mixup_alpha' in config_dict['augmentation']
        config_dict['models'] = [
            {
                'name': 'resnet18',
                'architecture': 'resnet18',
                'num_classes': 4,
                'in_channels': 25
            }
        ]

        config = ExperimentConfig(**config_dict)
        configs.append(config)

    return configs


def create_hyperparameter_tuning_experiments() -> List[ExperimentConfig]:
    """
    Create hyperparameter tuning experiment configurations.

    Returns:
        List of experiment configs for hyperparameter tuning
    """
    base_config = ExperimentConfig(
        name='hyperparameter_tuning',
        description='Hyperparameter tuning for optimal performance',
        seed=42,
        device='cuda',
        save_dir='results',
        dataset={
            'name': 'BCI-IV-2a',
            'file_path': 'data/BCI_IV_2a.hdf5',
            'split_ratio': 0.8,
            'validation_split': 0.1
        },
        augmentation={
            'enabled': True,
            'transforms': [
                {'name': 'horizontal_flip', 'enabled': True, 'p': 0.5},
                {'name': 'rotation', 'enabled': True, 'degrees': 15}
            ],
            'mixup_alpha': 1.0
        },
        training={
            'epochs': 100,
            'batch_size': 32,
            'num_workers': 0,
            'early_stopping_patience': 10,
            'mixed_precision': False,
            'gradient_accumulation_steps': 1
        },
        evaluation={
            'metrics': ['accuracy', 'f1', 'auc'],
            'robustness_tests': True,
            'statistical_tests': True,
            'visualization': True
        }
    )

    # Hyperparameter grid
    param_grid = {
        'batch_sizes': [16, 32, 64],
        'learning_rates': [0.001, 0.0001, 0.00001],
        'optimizers': ['adam', 'sgd']
    }

    gs = GridSearch(param_grid)
    configs = gs.to_configs(base_config)

    return configs
