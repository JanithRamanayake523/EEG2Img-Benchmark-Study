"""
Models module for EEG classification.

This package provides various neural network architectures for EEG classification:
- CNNs: ResNet variants and lightweight custom CNN for image-based classification
- Vision Transformers: ViT models for transformer-based image classification
- Baselines: Raw-signal models (1D CNN, LSTM, Transformer, EEGNet)

All models follow a consistent interface for easy experimentation.
"""

from .cnn import (
    ResNet18EEG,
    ResNet50EEG,
    LightweightCNN,
    get_cnn_model
)

from .vit import (
    ViTEEG,
    ViTBase16,
    ViTSmall16,
    ViTTiny16,
    get_vit_model
)

from .baselines import (
    CNN1D,
    LSTMClassifier,
    TransformerClassifier,
    EEGNet,
    get_baseline_model
)

import torch.nn as nn
from typing import Literal, Optional, Dict, Any


# Unified model registry
MODEL_REGISTRY = {
    # CNN models (for images)
    'resnet18': lambda **kwargs: get_cnn_model('resnet18', **kwargs),
    'resnet50': lambda **kwargs: get_cnn_model('resnet50', **kwargs),
    'lightweight_cnn': lambda **kwargs: get_cnn_model('lightweight', **kwargs),

    # Vision Transformers (for images)
    'vit_base': lambda **kwargs: get_vit_model('vit_base', **kwargs),
    'vit_small': lambda **kwargs: get_vit_model('vit_small', **kwargs),
    'vit_tiny': lambda **kwargs: get_vit_model('vit_tiny', **kwargs),

    # Baseline models (for raw time-series)
    'cnn1d': lambda **kwargs: get_baseline_model('cnn1d', **kwargs),
    'lstm': lambda **kwargs: get_baseline_model('lstm', **kwargs),
    'bilstm': lambda **kwargs: get_baseline_model('bilstm', **kwargs),
    'transformer': lambda **kwargs: get_baseline_model('transformer', **kwargs),
    'eegnet': lambda **kwargs: get_baseline_model('eegnet', **kwargs),
}


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Unified factory function to get any model by name.

    Args:
        model_name: Model identifier
        **kwargs: Model-specific arguments (num_classes, in_channels, etc.)

    Returns:
        Model instance

    Example:
        >>> # CNN for images
        >>> model = get_model('resnet18', num_classes=4, in_channels=25, pretrained=True)
        >>>
        >>> # ViT for images
        >>> model = get_model('vit_small', num_classes=4, in_channels=25, img_size=224)
        >>>
        >>> # Baseline for raw signals
        >>> model = get_model('cnn1d', num_classes=4, in_channels=25, seq_length=751)

    Raises:
        ValueError: If model_name not found in registry
    """
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {available}"
        )

    return MODEL_REGISTRY[model_name](**kwargs)


def list_models(category: Optional[Literal['cnn', 'vit', 'baseline', 'all']] = 'all') -> list:
    """
    List available models by category.

    Args:
        category: Model category to list
            - 'cnn': CNN models for images
            - 'vit': Vision Transformer models for images
            - 'baseline': Raw-signal baseline models
            - 'all': All models

    Returns:
        List of model names

    Example:
        >>> cnn_models = list_models('cnn')
        >>> print(cnn_models)  # ['resnet18', 'resnet50', 'lightweight_cnn']
    """
    cnn_models = ['resnet18', 'resnet50', 'lightweight_cnn']
    vit_models = ['vit_base', 'vit_small', 'vit_tiny']
    baseline_models = ['cnn1d', 'lstm', 'bilstm', 'transformer', 'eegnet']

    if category == 'cnn':
        return cnn_models
    elif category == 'vit':
        return vit_models
    elif category == 'baseline':
        return baseline_models
    elif category == 'all':
        return cnn_models + vit_models + baseline_models
    else:
        raise ValueError(f"Unknown category: {category}. Use 'cnn', 'vit', 'baseline', or 'all'")


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with total and trainable parameter counts

    Example:
        >>> model = get_model('resnet18', num_classes=4, in_channels=25)
        >>> params = count_parameters(model)
        >>> print(f"Total: {params['total']:,}, Trainable: {params['trainable']:,}")
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive model information.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model info (name, parameters, etc.)

    Example:
        >>> model = get_model('vit_tiny', num_classes=4, in_channels=25)
        >>> info = get_model_info(model)
        >>> print(info)
    """
    params = count_parameters(model)

    return {
        'class_name': model.__class__.__name__,
        'total_parameters': params['total'],
        'trainable_parameters': params['trainable'],
        'frozen_parameters': params['frozen'],
        'num_classes': getattr(model, 'num_classes', None),
        'in_channels': getattr(model, 'in_channels', None),
    }


__all__ = [
    # CNN models
    'ResNet18EEG',
    'ResNet50EEG',
    'LightweightCNN',
    'get_cnn_model',

    # ViT models
    'ViTEEG',
    'ViTBase16',
    'ViTSmall16',
    'ViTTiny16',
    'get_vit_model',

    # Baseline models
    'CNN1D',
    'LSTMClassifier',
    'TransformerClassifier',
    'EEGNet',
    'get_baseline_model',

    # Registry
    'MODEL_REGISTRY',
    'get_model',
    'list_models',
    'count_parameters',
    'get_model_info',
]
