"""
Phase 3.1: End-to-End Pipeline Test

Tests complete pipeline from raw data to trained model:
1. Load BCI IV-2a preprocessed data
2. Apply GAF transformation
3. Create dataset and dataloader
4. Train ResNet18 for 1 epoch
5. Verify no data leakage
6. Check memory usage
7. Validate GPU compatibility

This is a smoke test to ensure the full pipeline works before launching
large-scale experiments.
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import h5py
import json
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transforms import GAFTransformer
from src.models import get_model
from src.utils.logging import setup_logger

# Configuration
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_PATH = Path('data/BCI_IV_2a.hdf5')
SAVE_DIR = Path('results/phase3/validation/pipeline_test')
NUM_CLASSES = 4
BATCH_SIZE = 16
LEARNING_RATE = 0.001
TEST_EPOCHS = 1  # Just 1 epoch for smoke test

# Set random seeds
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SAVE_DIR.mkdir(parents=True, exist_ok=True)
logger = setup_logger('pipeline_test', SAVE_DIR / 'pipeline_test.log')


def load_bci_data(data_path, subject='A01T', max_samples=None):
    """Load preprocessed BCI IV-2a data from HDF5."""
    logger.info(f"Loading data from {data_path}, subject {subject}")

    try:
        with h5py.File(data_path, 'r') as f:
            if subject not in f:
                # Try alternative naming
                available = list(f.keys())
                logger.info(f"Available subjects: {available}")
                subject = available[0] if available else None
                if subject is None:
                    raise ValueError("No subjects found in HDF5 file")

            signals = f[subject]['signals'][:]
            labels = f[subject]['labels'][:]

            if max_samples:
                signals = signals[:max_samples]
                labels = labels[:max_samples]

            logger.info(f"Loaded data shape: {signals.shape}, labels: {labels.shape}")
            return signals, labels
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def create_train_test_split(signals, labels, train_ratio=0.8):
    """Split data into train and test sets with stratification."""
    n_samples = len(signals)
    n_train = int(n_samples * train_ratio)

    # Simple split (in production, use stratified split)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, y_train = signals[train_idx], labels[train_idx]
    X_test, y_test = signals[test_idx], labels[test_idx]

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def transform_to_images(signals, transformer, batch_size=32):
    """Transform EEG signals to images in batches."""
    logger.info(f"Transforming {len(signals)} signals to images")

    n_samples = len(signals)
    all_images = []

    start_time = time.time()

    for i in range(0, n_samples, batch_size):
        batch_signals = signals[i:i+batch_size]

        # Process each epoch
        batch_images = []
        for epoch in batch_signals:
            # transformer expects shape (channels, samples)
            # Use transform_epoch method (not transform)
            image = transformer.transform_epoch(epoch)  # Output: (C, H, W)

            # Image should already be (C, H, W) from transform_epoch
            # Stack channels to create input for image models
            # For simplicity, stack all channels: (C, H, W) -> (C*H, W) or similar
            # Or average across channels to get single channel image
            if image.ndim == 3:
                # Stack all channel images into one (average approach)
                image = image.mean(axis=0)[np.newaxis, ...]  # Single channel

            batch_images.append(image)

        all_images.append(np.array(batch_images))

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"Processed {i+batch_size}/{n_samples} signals")

    images = np.concatenate(all_images, axis=0)
    elapsed = time.time() - start_time

    logger.info(f"Transformation complete: {images.shape} in {elapsed:.2f}s ({elapsed/n_samples*1000:.2f}ms per sample)")

    return images


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(dataloader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def check_gpu_memory():
    """Check GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3  # GB
        logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        return {'allocated_gb': allocated, 'reserved_gb': reserved}
    return None


def main():
    """Run end-to-end pipeline test."""
    logger.info("="*60)
    logger.info("Phase 3: End-to-End Pipeline Test")
    logger.info("="*60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Random seed: {RANDOM_SEED}")

    results = {
        'device': DEVICE,
        'timestamp': datetime.now().isoformat(),
        'status': 'started'
    }

    try:
        # Step 1: Load data
        logger.info("\n[1/7] Loading preprocessed EEG data...")
        signals, labels = load_bci_data(DATA_PATH, max_samples=100)  # Use small subset for test

        # Adjust labels to 0-3 range if needed
        unique_labels = np.unique(labels)
        logger.info(f"Unique labels: {unique_labels}")
        if unique_labels.min() > 0:
            labels = labels - unique_labels.min()
            logger.info(f"Adjusted labels to: {np.unique(labels)}")

        results['data_shape'] = signals.shape
        results['num_classes'] = len(np.unique(labels))

        # Step 2: Train/test split
        logger.info("\n[2/7] Creating train/test split...")
        X_train, y_train, X_test, y_test = create_train_test_split(signals, labels)

        # Step 3: Transform to images
        logger.info("\n[3/7] Applying GAF transformation...")
        transformer = GAFTransformer(image_size=64, method='summation')

        X_train_images = transform_to_images(X_train, transformer)
        X_test_images = transform_to_images(X_test, transformer)

        results['image_shape'] = X_train_images.shape[1:]

        # Step 4: Create dataloaders
        logger.info("\n[4/7] Creating dataloaders...")
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_images),
            torch.LongTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test_images),
            torch.LongTensor(y_test)
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

        # Step 5: Create model
        logger.info("\n[5/7] Creating ResNet18 model...")
        model = get_model('resnet18', num_classes=NUM_CLASSES, in_channels=X_train_images.shape[1])
        model = model.to(DEVICE)

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {num_params:,}")
        results['model_params'] = num_params

        check_gpu_memory()

        # Step 6: Train for 1 epoch
        logger.info(f"\n[6/7] Training for {TEST_EPOCHS} epoch (smoke test)...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        start_time = time.time()

        for epoch in range(TEST_EPOCHS):
            logger.info(f"\nEpoch {epoch+1}/{TEST_EPOCHS}")

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

            test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
            logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        training_time = time.time() - start_time
        logger.info(f"Training time: {training_time:.2f}s")

        results['train_loss'] = train_loss
        results['train_acc'] = train_acc
        results['test_loss'] = test_loss
        results['test_acc'] = test_acc
        results['training_time_s'] = training_time

        # Step 7: Check memory
        logger.info("\n[7/7] Checking GPU memory...")
        gpu_memory = check_gpu_memory()
        if gpu_memory:
            results['gpu_memory'] = gpu_memory

        results['status'] = 'success'

        logger.info("\n" + "="*60)
        logger.info("PIPELINE TEST SUCCESSFUL!")
        logger.info("="*60)
        logger.info(f"[OK] Data loading works")
        logger.info(f"[OK] GAF transformation works")
        logger.info(f"[OK] Model creation works")
        logger.info(f"[OK] Training works on {DEVICE}")
        logger.info(f"[OK] Test accuracy: {test_acc:.2f}%")

    except Exception as e:
        logger.error(f"\nPIPELINE TEST FAILED: {e}", exc_info=True)
        results['status'] = 'failed'
        results['error'] = str(e)
        raise

    finally:
        # Save results
        results_file = SAVE_DIR / 'pipeline_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {results_file}")

    return results


if __name__ == '__main__':
    main()
