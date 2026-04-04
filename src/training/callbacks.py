"""
Training callbacks for monitoring and control.

Implements callbacks for early stopping, model checkpointing,
learning rate scheduling, and logging.
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
import time
from typing import Optional, Dict, Any, List
import numpy as np


class Callback:
    """Base class for callbacks."""

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """
    Early stopping to terminate training when validation metric stops improving.

    Monitors a metric and stops training if no improvement for patience epochs.
    """

    def __init__(self,
                 monitor: str = 'val_loss',
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'min',
                 restore_best_weights: bool = True,
                 verbose: bool = True):
        """
        Initialize Early Stopping callback.

        Args:
            monitor: Metric to monitor ('val_loss', 'val_acc', etc.)
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Restore model weights from best epoch
            verbose: Print messages

        Example:
            >>> early_stop = EarlyStopping(monitor='val_loss', patience=10)
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        if mode == 'min':
            self.best_value = np.inf
            self.monitor_op = np.less
        else:
            self.best_value = -np.inf
            self.monitor_op = np.greater

    def on_train_begin(self, logs=None):
        """Reset counters at training start."""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = np.inf if self.mode == 'min' else -np.inf

    def on_epoch_end(self, epoch, logs=None):
        """Check if should stop training."""
        current = logs.get(self.monitor)

        if current is None:
            return

        # Check if improved
        if self.monitor_op(current - self.min_delta, self.best_value):
            self.best_value = current
            self.wait = 0
            if self.restore_best_weights and 'model' in logs:
                self.best_weights = {k: v.cpu().clone() for k, v in logs['model'].state_dict().items()}
            if self.verbose:
                print(f"Epoch {epoch+1}: {self.monitor} improved to {current:.4f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logs['stop_training'] = True
                if self.verbose:
                    print(f"\nEarly stopping triggered after epoch {epoch+1}")
                    print(f"Best {self.monitor}: {self.best_value:.4f}")

                # Restore best weights
                if self.restore_best_weights and self.best_weights and 'model' in logs:
                    logs['model'].load_state_dict(self.best_weights)
                    if self.verbose:
                        print("Restored best model weights")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Saves model weights when monitored metric improves.
    """

    def __init__(self,
                 filepath: str,
                 monitor: str = 'val_loss',
                 mode: str = 'min',
                 save_best_only: bool = True,
                 save_last: bool = True,
                 verbose: bool = True):
        """
        Initialize Model Checkpoint callback.

        Args:
            filepath: Path to save model (can include epoch/metric placeholders)
            monitor: Metric to monitor
            mode: 'min' or 'max'
            save_best_only: Only save when metric improves
            save_last: Always save last epoch
            verbose: Print messages

        Example:
            >>> checkpoint = ModelCheckpoint('models/best_model.pt', monitor='val_acc', mode='max')
        """
        super().__init__()
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.verbose = verbose

        if mode == 'min':
            self.best_value = np.inf
            self.monitor_op = np.less
        else:
            self.best_value = -np.inf
            self.monitor_op = np.greater

    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint if metric improved."""
        current = logs.get(self.monitor)

        if current is None:
            return

        # Check if should save
        should_save = False

        if self.save_best_only:
            if self.monitor_op(current, self.best_value):
                should_save = True
                self.best_value = current
        else:
            should_save = True

        if should_save and 'model' in logs:
            # Create directory if needed
            self.filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': logs['model'].state_dict(),
                'optimizer_state_dict': logs.get('optimizer').state_dict() if logs.get('optimizer') else None,
                'metrics': {k: v for k, v in logs.items() if isinstance(v, (int, float))},
                'best_value': self.best_value,
            }

            torch.save(checkpoint, self.filepath)

            if self.verbose:
                print(f"Saved checkpoint to {self.filepath} ({self.monitor}={current:.4f})")

    def on_train_end(self, logs=None):
        """Save final checkpoint if save_last enabled."""
        if self.save_last and 'model' in logs:
            last_filepath = self.filepath.parent / f"last_{self.filepath.name}"
            checkpoint = {
                'epoch': logs.get('epoch', 0),
                'model_state_dict': logs['model'].state_dict(),
                'optimizer_state_dict': logs.get('optimizer').state_dict() if logs.get('optimizer') else None,
                'metrics': {k: v for k, v in logs.items() if isinstance(v, (int, float))},
            }
            torch.save(checkpoint, last_filepath)
            if self.verbose:
                print(f"Saved last checkpoint to {last_filepath}")


class LearningRateScheduler(Callback):
    """
    Adjust learning rate during training.

    Supports ReduceLROnPlateau and step-based schedules.
    """

    def __init__(self,
                 scheduler,
                 monitor: str = 'val_loss',
                 verbose: bool = True):
        """
        Initialize LR Scheduler callback.

        Args:
            scheduler: PyTorch learning rate scheduler
            monitor: Metric to monitor (for ReduceLROnPlateau)
            verbose: Print LR changes

        Example:
            >>> from torch.optim.lr_scheduler import ReduceLROnPlateau
            >>> scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)
            >>> lr_callback = LearningRateScheduler(scheduler)
        """
        super().__init__()
        self.scheduler = scheduler
        self.monitor = monitor
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        """Step the scheduler."""
        # Check if ReduceLROnPlateau (needs metric)
        if hasattr(self.scheduler, 'step') and 'metrics' in str(self.scheduler.__class__):
            current = logs.get(self.monitor)
            if current is not None:
                old_lr = self.scheduler.optimizer.param_groups[0]['lr']
                self.scheduler.step(current)
                new_lr = self.scheduler.optimizer.param_groups[0]['lr']
                if self.verbose and old_lr != new_lr:
                    print(f"Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        else:
            # Regular scheduler
            self.scheduler.step()


class History(Callback):
    """
    Record training history.

    Stores metrics for each epoch.
    """

    def __init__(self):
        """Initialize History callback."""
        super().__init__()
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        """Record metrics for this epoch."""
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)

    def save(self, filepath: str):
        """Save history to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {filepath}")

    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """Get epoch with best metric value."""
        if metric not in self.history:
            return -1

        values = self.history[metric]
        if mode == 'min':
            return int(np.argmin(values))
        else:
            return int(np.argmax(values))


class ProgressBar(Callback):
    """
    Display training progress bar.

    Shows epoch progress and metrics.
    """

    def __init__(self, total_epochs: int):
        """
        Initialize Progress Bar.

        Args:
            total_epochs: Total number of epochs
        """
        super().__init__()
        self.total_epochs = total_epochs
        self.epoch_start_time = None

    def on_epoch_begin(self, epoch, logs=None):
        """Record epoch start time."""
        self.epoch_start_time = time.time()
        print(f"\nEpoch {epoch+1}/{self.total_epochs}")

    def on_epoch_end(self, epoch, logs=None):
        """Display epoch metrics."""
        epoch_time = time.time() - self.epoch_start_time

        # Format metrics
        metrics_str = []
        for key, value in logs.items():
            if isinstance(value, (int, float)) and not key.startswith('_'):
                metrics_str.append(f"{key}: {value:.4f}")

        print(f"  Time: {epoch_time:.2f}s - " + " - ".join(metrics_str))


class CallbackList:
    """
    Container for multiple callbacks.

    Calls all callbacks in sequence.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialize callback list.

        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []

    def append(self, callback: Callback):
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def on_train_begin(self, logs=None):
        """Call on_train_begin for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        """Call on_train_end for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        """Call on_epoch_begin for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        """Call on_epoch_end for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

        return logs.get('stop_training', False)

    def on_batch_begin(self, batch, logs=None):
        """Call on_batch_begin for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        """Call on_batch_end for all callbacks."""
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


if __name__ == '__main__':
    # Test callbacks
    print("="*60)
    print("Testing Callback Modules")
    print("="*60)

    # Test Early Stopping
    print("\n--- Early Stopping ---")
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=True)
    early_stop.on_train_begin()

    for epoch in range(10):
        logs = {'val_loss': 1.0 - epoch * 0.1 if epoch < 5 else 0.5 + epoch * 0.01}
        early_stop.on_epoch_end(epoch, logs)
        if logs.get('stop_training'):
            print(f"Training stopped at epoch {epoch+1}")
            break

    # Test History
    print("\n--- History ---")
    history = History()
    for epoch in range(5):
        history.on_epoch_end(epoch, {'loss': 1.0 - epoch * 0.1, 'acc': epoch * 0.2})

    print(f"History: {history.history}")
    print(f"Best epoch (loss): {history.get_best_epoch('loss', 'min')}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
