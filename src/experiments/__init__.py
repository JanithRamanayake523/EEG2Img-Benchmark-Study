"""
Experiment orchestration module.

Provides configuration management, experiment running, and grid search utilities.
"""

from .config import (
    DatasetConfig,
    TransformConfig,
    AugmentationConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
    EvaluationConfig,
    ExperimentConfig,
    load_config,
    save_config
)

from .runner import ExperimentRunner, ExperimentLogger

from .grid_search import (
    GridSearch,
    RandomSearch,
    create_baseline_experiments,
    create_augmentation_experiments,
    create_hyperparameter_tuning_experiments
)

__all__ = [
    # Configuration classes
    'DatasetConfig',
    'TransformConfig',
    'AugmentationConfig',
    'ModelConfig',
    'OptimizerConfig',
    'TrainingConfig',
    'EvaluationConfig',
    'ExperimentConfig',
    'load_config',
    'save_config',

    # Runner
    'ExperimentRunner',
    'ExperimentLogger',

    # Grid search
    'GridSearch',
    'RandomSearch',
    'create_baseline_experiments',
    'create_augmentation_experiments',
    'create_hyperparameter_tuning_experiments',
]
