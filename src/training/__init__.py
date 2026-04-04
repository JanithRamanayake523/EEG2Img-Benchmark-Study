"""
Training module for EEG classification.

This package provides training infrastructure including:
- Trainer: Complete training loop with validation
- Callbacks: Early stopping, checkpointing, LR scheduling
- Augmentation: Data augmentation for images and time-series
- DataLoaders: Dataset classes for EEG data

All components work together for end-to-end model training.
"""

from .trainer import Trainer, EEGDataset, create_data_loaders
from .callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    History,
    ProgressBar,
    CallbackList
)
from .augmentation import (
    ImageAugmentation,
    TimeSeriesAugmentation,
    MixUp,
    CutMix,
    get_augmentation
)

__all__ = [
    # Trainer
    'Trainer',
    'EEGDataset',
    'create_data_loaders',

    # Callbacks
    'Callback',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateScheduler',
    'History',
    'ProgressBar',
    'CallbackList',

    # Augmentation
    'ImageAugmentation',
    'TimeSeriesAugmentation',
    'MixUp',
    'CutMix',
    'get_augmentation',
]
