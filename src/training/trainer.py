"""
Training loop and utilities for EEG classification models.

Implements complete training pipeline with support for:
- Image-based and time-series models
- Cross-validation
- Mixed precision training
- Gradient accumulation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import h5py
from tqdm import tqdm

from .callbacks import CallbackList, Callback
from .augmentation import get_augmentation


class EEGDataset(Dataset):
    """
    Dataset for loading preprocessed EEG data from HDF5 files.

    Supports both image-transformed and raw time-series data.
    """

    def __init__(self,
                 file_path: str,
                 data_type: str = 'image',
                 transform=None,
                 target_transform=None):
        """
        Initialize EEG Dataset.

        Args:
            file_path: Path to HDF5 file containing data
            data_type: 'image' or 'timeseries'
            transform: Data augmentation transform
            target_transform: Label transform

        Example:
            >>> dataset = EEGDataset('data/images/gaf/A01T_gaf.h5', data_type='image')
        """
        self.file_path = Path(file_path)
        self.data_type = data_type
        self.transform = transform
        self.target_transform = target_transform

        # Load data
        with h5py.File(self.file_path, 'r') as f:
            self.data = f['images'][:] if 'images' in f else f['data'][:]
            self.labels = f['labels'][:]

            # Store metadata
            self.metadata = {key: f.attrs[key] for key in f.attrs.keys()}

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get data and label at index."""
        data = self.data[idx]
        label = self.labels[idx]

        # Convert to tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.long)

        # Apply transforms
        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            label = self.target_transform(label)

        return data, label

    def get_metadata(self) -> Dict:
        """Get dataset metadata."""
        return self.metadata


class Trainer:
    """
    Training engine for EEG classification models.

    Handles training loop, validation, and callbacks.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 callbacks: Optional[List[Callback]] = None,
                 mixed_precision: bool = False,
                 gradient_accumulation_steps: int = 1):
        """
        Initialize Trainer.

        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on ('cuda' or 'cpu')
            callbacks: List of callbacks
            mixed_precision: Use automatic mixed precision
            gradient_accumulation_steps: Accumulate gradients over N batches

        Example:
            >>> model = get_model('resnet18', num_classes=4, in_channels=25)
            >>> criterion = nn.CrossEntropyLoss()
            >>> optimizer = optim.Adam(model.parameters(), lr=0.001)
            >>> trainer = Trainer(model, criterion, optimizer)
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.callbacks = CallbackList(callbacks or [])
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        self.optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)

            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss / self.gradient_accumulation_steps
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # Statistics
            running_loss += loss.item() * self.gradient_accumulation_steps * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return {'train_loss': epoch_loss, 'train_acc': epoch_acc}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item() * data.size(0)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total

        return {'val_loss': val_loss, 'val_acc': val_acc}

    def fit(self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train

        Returns:
            Training history

        Example:
            >>> history = trainer.fit(train_loader, val_loader, epochs=50)
        """
        # Prepare logs
        logs = {
            'model': self.model,
            'optimizer': self.optimizer,
        }

        # Callbacks
        self.callbacks.on_train_begin(logs)

        try:
            for epoch in range(epochs):
                # Epoch begin
                self.callbacks.on_epoch_begin(epoch, logs)

                # Train
                train_metrics = self.train_epoch(train_loader)

                # Validate
                if val_loader:
                    val_metrics = self.validate(val_loader)
                else:
                    val_metrics = {}

                # Update logs
                logs.update(train_metrics)
                logs.update(val_metrics)
                logs['epoch'] = epoch

                # Update history
                for key, value in {**train_metrics, **val_metrics}.items():
                    if key not in self.history:
                        self.history[key] = []
                    self.history[key].append(value)

                # Epoch end
                should_stop = self.callbacks.on_epoch_end(epoch, logs)

                # Check early stopping
                if should_stop:
                    print(f"\nTraining stopped early at epoch {epoch+1}")
                    break

        finally:
            # Training end
            self.callbacks.on_train_end(logs)

        return self.history

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on data.

        Args:
            data_loader: Data loader

        Returns:
            predictions: Predicted class labels
            probabilities: Class probabilities
        """
        self.model.eval()

        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data, _ in tqdm(data_loader, desc="Predicting"):
                data = data.to(self.device)
                output = self.model(data)
                probs = torch.softmax(output, dim=1)

                _, predicted = torch.max(output, 1)

                all_preds.append(predicted.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        predictions = np.concatenate(all_preds)
        probabilities = np.concatenate(all_probs)

        return predictions, probabilities

    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        print(f"Loaded checkpoint from {filepath}")


def create_data_loaders(
    train_file: str,
    val_file: Optional[str] = None,
    batch_size: int = 32,
    data_type: str = 'image',
    num_workers: int = 0,
    augment_train: bool = True,
    **aug_kwargs
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation data loaders.

    Args:
        train_file: Path to training HDF5 file
        val_file: Path to validation HDF5 file (optional)
        batch_size: Batch size
        data_type: 'image' or 'timeseries'
        num_workers: Number of data loading workers
        augment_train: Apply augmentation to training data
        **aug_kwargs: Arguments for augmentation

    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader (or None)

    Example:
        >>> train_loader, val_loader = create_data_loaders(
        ...     'data/images/gaf/A01T_gaf.h5',
        ...     'data/images/gaf/A01E_gaf.h5',
        ...     batch_size=32
        ... )
    """
    # Training dataset
    train_transform = get_augmentation(data_type, **aug_kwargs) if augment_train else None
    train_dataset = EEGDataset(train_file, data_type=data_type, transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Validation dataset
    val_loader = None
    if val_file:
        val_dataset = EEGDataset(val_file, data_type=data_type, transform=None)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )

    return train_loader, val_loader


if __name__ == '__main__':
    # Test trainer components
    print("="*60)
    print("Testing Trainer Components")
    print("="*60)

    # Create dummy data
    print("\n--- Creating Dummy Data ---")
    dummy_images = torch.randn(100, 25, 64, 64)
    dummy_labels = torch.randint(0, 4, (100,))
    dummy_dataset = TensorDataset(dummy_images, dummy_labels)
    dummy_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=True)
    print(f"Created dataset with {len(dummy_dataset)} samples")

    # Create model
    print("\n--- Creating Model ---")
    from src.models import get_model
    model = get_model('lightweight_cnn', num_classes=4, in_channels=25, pretrained=False)
    print(f"Model: {model.__class__.__name__}")

    # Create trainer
    print("\n--- Creating Trainer ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    trainer = Trainer(model, criterion, optimizer, device='cpu')
    print("Trainer initialized")

    # Train for 2 epochs
    print("\n--- Training for 2 Epochs ---")
    history = trainer.fit(dummy_loader, val_loader=None, epochs=2)
    print(f"Training complete. Final loss: {history['train_loss'][-1]:.4f}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
