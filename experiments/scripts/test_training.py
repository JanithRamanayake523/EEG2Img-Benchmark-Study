"""
Test script for Phase 5 training components.

Tests trainer, callbacks, and augmentation on dummy data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.training import (
    Trainer,
    EEGDataset,
    create_data_loaders,
    EarlyStopping,
    ModelCheckpoint,
    History,
    ProgressBar,
    ImageAugmentation,
    TimeSeriesAugmentation,
    MixUp,
    CutMix
)
from src.models import get_model


def test_augmentation():
    """Test augmentation modules."""
    print("\n" + "="*60)
    print("Testing Augmentation")
    print("="*60)

    # Test Image Augmentation
    print("\n--- Image Augmentation ---")
    img_aug = ImageAugmentation(rotation_degrees=10, noise_std=0.01)
    x_img = torch.randn(3, 64, 64)  # (C, H, W)
    x_img_aug = img_aug(x_img)
    print(f"  [OK] Input: {x_img.shape} -> Output: {x_img_aug.shape}")

    # Test Time-Series Augmentation
    print("\n--- Time-Series Augmentation ---")
    ts_aug = TimeSeriesAugmentation(jitter_std=0.03, scaling_range=(0.9, 1.1))
    x_ts = torch.randn(8, 25, 751)  # (B, C, T)
    x_ts_aug = ts_aug(x_ts)
    print(f"  [OK] Input: {x_ts.shape} -> Output: {x_ts_aug.shape}")

    # Test MixUp
    print("\n--- MixUp ---")
    mixup = MixUp(alpha=0.2)
    x1, y1 = torch.randn(25, 64, 64), torch.tensor(0)
    x2, y2 = torch.randn(25, 64, 64), torch.tensor(1)
    mixed_x, mixed_y = mixup(x1, y1, x2, y2)
    print(f"  [OK] Mixed shape: {mixed_x.shape}")

    # Test CutMix
    print("\n--- CutMix ---")
    cutmix = CutMix(alpha=1.0)
    mixed_x, mixed_y = cutmix(x1, y1, x2, y2)
    print(f"  [OK] Mixed shape: {mixed_x.shape}")

    return True


def test_callbacks():
    """Test callback modules."""
    print("\n" + "="*60)
    print("Testing Callbacks")
    print("="*60)

    # Test Early Stopping
    print("\n--- Early Stopping ---")
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=False)
    early_stop.on_train_begin()

    stopped = False
    for epoch in range(10):
        logs = {'val_loss': 1.0 - epoch * 0.1 if epoch < 5 else 0.5 + epoch * 0.01}
        early_stop.on_epoch_end(epoch, logs)
        if logs.get('stop_training'):
            print(f"  [OK] Training stopped at epoch {epoch+1}")
            stopped = True
            break

    if not stopped:
        print(f"  [FAIL] Early stopping did not trigger")
        return False

    # Test History
    print("\n--- History ---")
    history = History()
    for epoch in range(5):
        history.on_epoch_end(epoch, {'loss': 1.0 - epoch * 0.1, 'acc': epoch * 0.2})

    print(f"  [OK] Recorded {len(history.history['loss'])} epochs")
    print(f"  [OK] Best epoch (loss): {history.get_best_epoch('loss', 'min')}")

    return True


def test_trainer():
    """Test training loop."""
    print("\n" + "="*60)
    print("Testing Trainer")
    print("="*60)

    # Create dummy data
    print("\n--- Creating Dummy Data ---")
    dummy_images = torch.randn(100, 25, 64, 64)
    dummy_labels = torch.randint(0, 4, (100,))

    train_dataset = TensorDataset(dummy_images[:80], dummy_labels[:80])
    val_dataset = TensorDataset(dummy_images[80:], dummy_labels[80:])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    print(f"  [OK] Train: {len(train_dataset)} samples")
    print(f"  [OK] Val: {len(val_dataset)} samples")

    # Create model
    print("\n--- Creating Model ---")
    model = get_model('lightweight_cnn', num_classes=4, in_channels=25, pretrained=False)
    print(f"  [OK] Model: {model.__class__.__name__}")

    # Create trainer with callbacks
    print("\n--- Creating Trainer with Callbacks ---")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=False),
        History()
    ]

    trainer = Trainer(
        model,
        criterion,
        optimizer,
        device='cpu',
        callbacks=callbacks
    )
    print("  [OK] Trainer initialized")

    # Train for a few epochs
    print("\n--- Training for 5 Epochs ---")
    history = trainer.fit(train_loader, val_loader, epochs=5)

    print(f"  [OK] Training complete")
    print(f"      Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"      Final train acc: {history['train_acc'][-1]:.4f}")
    print(f"      Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"      Final val acc: {history['val_acc'][-1]:.4f}")

    # Check that loss decreased
    if history['train_loss'][-1] >= history['train_loss'][0]:
        print(f"  [FAIL] Training loss did not decrease")
        return False

    # Test prediction
    print("\n--- Testing Prediction ---")
    predictions, probabilities = trainer.predict(val_loader)
    print(f"  [OK] Predictions shape: {predictions.shape}")
    print(f"  [OK] Probabilities shape: {probabilities.shape}")

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHASE 5 TRAINING INFRASTRUCTURE VALIDATION")
    print("="*80)

    # Track results
    results = {}

    # Test augmentation
    results['augmentation'] = test_augmentation()

    # Test callbacks
    results['callbacks'] = test_callbacks()

    # Test trainer
    results['trainer'] = test_trainer()

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
        print("[OK] ALL TESTS PASSED - Phase 5 Training Infrastructure Validated")
    else:
        print("[FAIL] SOME TESTS FAILED - Review errors above")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
