"""
Experiment configuration management.

Loads and manages experiment configurations from YAML files.
Provides type-safe configuration objects with validation.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


class DatasetConfig:
    """Configuration for dataset."""

    def __init__(self, name: str, file_path: str, split_ratio: float = 0.8,
                 validation_split: float = 0.1, **kwargs):
        """
        Initialize dataset configuration.

        Args:
            name: Dataset name
            file_path: Path to dataset file
            split_ratio: Train/test split ratio
            validation_split: Validation split from training data
            **kwargs: Additional dataset-specific options
        """
        self.name = name
        self.file_path = file_path
        self.split_ratio = split_ratio
        self.validation_split = validation_split
        self.options = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'file_path': self.file_path,
            'split_ratio': self.split_ratio,
            'validation_split': self.validation_split,
            **self.options
        }

    @classmethod
    def from_dict(cls, config: Dict) -> 'DatasetConfig':
        """Create from dictionary."""
        return cls(**config)


class TransformConfig:
    """Configuration for data transforms."""

    def __init__(self, name: str, enabled: bool = True, **params):
        """
        Initialize transform configuration.

        Args:
            name: Transform name (e.g., 'horizontal_flip', 'rotation')
            enabled: Whether to apply this transform
            **params: Transform-specific parameters
        """
        self.name = name
        self.enabled = enabled
        self.params = params

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'enabled': self.enabled,
            **self.params
        }

    @classmethod
    def from_dict(cls, config: Dict) -> 'TransformConfig':
        """Create from dictionary."""
        return cls(**config)


class AugmentationConfig:
    """Configuration for data augmentation."""

    def __init__(self, enabled: bool = True, transforms: Optional[List[Dict]] = None,
                 mixup_alpha: float = 1.0, cutmix_alpha: float = 1.0, **kwargs):
        """
        Initialize augmentation configuration.

        Args:
            enabled: Whether to apply augmentation
            transforms: List of transform configs
            mixup_alpha: Alpha parameter for MixUp
            cutmix_alpha: Alpha parameter for CutMix
            **kwargs: Additional augmentation options
        """
        self.enabled = enabled
        self.transforms = [TransformConfig.from_dict(t) if isinstance(t, dict) else t
                          for t in (transforms or [])]
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.options = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'enabled': self.enabled,
            'transforms': [t.to_dict() for t in self.transforms],
            'mixup_alpha': self.mixup_alpha,
            'cutmix_alpha': self.cutmix_alpha,
            **self.options
        }

    @classmethod
    def from_dict(cls, config: Dict) -> 'AugmentationConfig':
        """Create from dictionary."""
        return cls(**config)


class ModelConfig:
    """Configuration for model architecture."""

    def __init__(self, name: str, architecture: str, num_classes: int = 4,
                 in_channels: int = 25, pretrained: bool = False, **kwargs):
        """
        Initialize model configuration.

        Args:
            name: Identifier for this model config
            architecture: Architecture name (e.g., 'resnet18', 'vit_tiny')
            num_classes: Number of output classes
            in_channels: Number of input channels
            pretrained: Whether to use pretrained weights
            **kwargs: Model-specific parameters
        """
        self.name = name
        self.architecture = architecture
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.options = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'architecture': self.architecture,
            'num_classes': self.num_classes,
            'in_channels': self.in_channels,
            'pretrained': self.pretrained,
            **self.options
        }

    @classmethod
    def from_dict(cls, config: Dict) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**config)


class OptimizerConfig:
    """Configuration for optimizer."""

    def __init__(self, name: str, learning_rate: float = 0.001, weight_decay: float = 0.0,
                 momentum: float = 0.9, **kwargs):
        """
        Initialize optimizer configuration.

        Args:
            name: Optimizer name (e.g., 'adam', 'sgd')
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            momentum: Momentum for SGD
            **kwargs: Additional optimizer parameters
        """
        self.name = name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.options = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            **self.options
        }

    @classmethod
    def from_dict(cls, config: Dict) -> 'OptimizerConfig':
        """Create from dictionary."""
        return cls(**config)


class TrainingConfig:
    """Configuration for training."""

    def __init__(self, epochs: int = 100, batch_size: int = 32, num_workers: int = 0,
                 early_stopping_patience: int = 10, mixed_precision: bool = False,
                 gradient_accumulation_steps: int = 1, **kwargs):
        """
        Initialize training configuration.

        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            num_workers: Number of data loading workers
            early_stopping_patience: Early stopping patience
            mixed_precision: Whether to use mixed precision training
            gradient_accumulation_steps: Gradient accumulation steps
            **kwargs: Additional training options
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.early_stopping_patience = early_stopping_patience
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.options = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'early_stopping_patience': self.early_stopping_patience,
            'mixed_precision': self.mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            **self.options
        }

    @classmethod
    def from_dict(cls, config: Dict) -> 'TrainingConfig':
        """Create from dictionary."""
        return cls(**config)


class EvaluationConfig:
    """Configuration for evaluation."""

    def __init__(self, metrics: Optional[List[str]] = None, robustness_tests: bool = True,
                 statistical_tests: bool = True, visualization: bool = True, **kwargs):
        """
        Initialize evaluation configuration.

        Args:
            metrics: List of metrics to compute
            robustness_tests: Whether to run robustness tests
            statistical_tests: Whether to run statistical tests
            visualization: Whether to generate visualizations
            **kwargs: Additional evaluation options
        """
        self.metrics = metrics or ['accuracy', 'f1', 'auc']
        self.robustness_tests = robustness_tests
        self.statistical_tests = statistical_tests
        self.visualization = visualization
        self.options = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'metrics': self.metrics,
            'robustness_tests': self.robustness_tests,
            'statistical_tests': self.statistical_tests,
            'visualization': self.visualization,
            **self.options
        }

    @classmethod
    def from_dict(cls, config: Dict) -> 'EvaluationConfig':
        """Create from dictionary."""
        return cls(**config)


class ExperimentConfig:
    """Complete experiment configuration."""

    def __init__(self, name: str, description: str = "", seed: int = 42,
                 device: str = 'cuda', save_dir: str = 'results',
                 dataset: Optional[Union[Dict, DatasetConfig]] = None,
                 augmentation: Optional[Union[Dict, AugmentationConfig]] = None,
                 models: Optional[List[Union[Dict, ModelConfig]]] = None,
                 optimizer: Optional[Union[Dict, OptimizerConfig]] = None,
                 training: Optional[Union[Dict, TrainingConfig]] = None,
                 evaluation: Optional[Union[Dict, EvaluationConfig]] = None,
                 **kwargs):
        """
        Initialize experiment configuration.

        Args:
            name: Experiment name
            description: Experiment description
            seed: Random seed
            device: Device to use (cuda/cpu)
            save_dir: Directory to save results
            dataset: Dataset configuration
            augmentation: Augmentation configuration
            models: List of model configurations
            optimizer: Optimizer configuration
            training: Training configuration
            evaluation: Evaluation configuration
            **kwargs: Additional options
        """
        self.name = name
        self.description = description
        self.seed = seed
        self.device = device
        self.save_dir = save_dir

        # Parse configurations
        self.dataset = DatasetConfig.from_dict(dataset) if isinstance(dataset, dict) else dataset
        self.augmentation = AugmentationConfig.from_dict(augmentation) if isinstance(augmentation, dict) else augmentation
        self.models = [ModelConfig.from_dict(m) if isinstance(m, dict) else m for m in (models or [])]
        self.optimizer = OptimizerConfig.from_dict(optimizer) if isinstance(optimizer, dict) else optimizer
        self.training = TrainingConfig.from_dict(training) if isinstance(training, dict) else training
        self.evaluation = EvaluationConfig.from_dict(evaluation) if isinstance(evaluation, dict) else evaluation
        self.options = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'seed': self.seed,
            'device': self.device,
            'save_dir': self.save_dir,
            'dataset': self.dataset.to_dict() if self.dataset else None,
            'augmentation': self.augmentation.to_dict() if self.augmentation else None,
            'models': [m.to_dict() for m in self.models],
            'optimizer': self.optimizer.to_dict() if self.optimizer else None,
            'training': self.training.to_dict() if self.training else None,
            'evaluation': self.evaluation.to_dict() if self.evaluation else None,
            **self.options
        }

    def to_json(self, filepath: Union[str, Path]):
        """Save configuration to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_yaml(self, filepath: Union[str, Path]):
        """Save configuration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from YAML file.

        Args:
            filepath: Path to YAML configuration file

        Returns:
            ExperimentConfig instance

        Example:
            >>> config = ExperimentConfig.from_yaml('configs/experiment_baseline.yaml')
            >>> print(config.name)
        """
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def __repr__(self) -> str:
        """String representation."""
        return (f"ExperimentConfig(name={self.name}, models={len(self.models)}, "
                f"epochs={self.training.epochs if self.training else 'N/A'}, "
                f"batch_size={self.training.batch_size if self.training else 'N/A'})")


def load_config(filepath: Union[str, Path]) -> ExperimentConfig:
    """
    Load configuration from file (auto-detects format).

    Args:
        filepath: Path to YAML or JSON config file

    Returns:
        ExperimentConfig instance

    Example:
        >>> config = load_config('configs/experiment.yaml')
        >>> print(f"Loaded experiment: {config.name}")
    """
    filepath = Path(filepath)

    if filepath.suffix == '.yaml' or filepath.suffix == '.yml':
        return ExperimentConfig.from_yaml(filepath)
    elif filepath.suffix == '.json':
        return ExperimentConfig.from_json(filepath)
    else:
        raise ValueError(f"Unsupported format: {filepath.suffix}")


def save_config(config: ExperimentConfig, filepath: Union[str, Path], format: str = 'yaml'):
    """
    Save configuration to file.

    Args:
        config: ExperimentConfig instance
        filepath: Output file path
        format: Output format ('yaml' or 'json')

    Example:
        >>> config = ExperimentConfig(name='test')
        >>> save_config(config, 'results/config.yaml')
    """
    if format.lower() in ['yaml', 'yml']:
        config.to_yaml(filepath)
    elif format.lower() == 'json':
        config.to_json(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
