"""
Phase 3.3: Raw-Signal Baseline Experiments

Train baseline models on raw time-series EEG without image transformation:
- 1D CNN (reference CNN for time-series)
- BiLSTM (temporal dependencies)
- Transformer (attention on time-series)

These establish the performance ceiling that can be achieved without
image transformation, providing crucial comparison baseline.

Usage:
    python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model cnn1d --device cuda:0
    python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model bilstm --device cuda:0
    python experiments/phase_3_benchmark_experiments/04_train_baselines.py --model transformer --device cuda:0
"""

import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import h5py
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, cohen_kappa_score
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.baselines import CNN1D, LSTMClassifier, TransformerClassifier
from src.utils.logging import setup_logger

# Configuration
RANDOM_SEED = 42
NUM_CLASSES = 4
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 15
K_FOLDS = 5


def parse_args():
    parser = argparse.ArgumentParser(description='Train baseline models on raw EEG signals')
    parser.add_argument('--model', type=str, default='cnn1d',
                        choices=['cnn1d', 'bilstm', 'transformer'],
                        help='Baseline model type')
    parser.add_argument('--data_path', type=str, default='data/BCI_IV_2a_EEG_only.hdf5',
                        help='Path to preprocessed EEG-only data (22 channels)')
    parser.add_argument('--subject', type=str, default='all',
                        help='Subject to train on (e.g., A01T, A02T, or "all" to pool all subjects)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda:0, cuda:1, or cpu)')
    parser.add_argument('--output_dir', type=str, default='results/phase3/baselines',
                        help='Output directory')
    parser.add_argument('--k_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data(data_path, subject):
    """Load preprocessed EEG-only data from HDF5.

    Args:
        data_path: Path to HDF5 file (containing 22 EEG channels only, no EOG)
        subject: Subject ID (e.g., 'A01T') or 'all' to pool all subjects

    Returns:
        signals, labels (pooled across subjects if subject='all')
        - signals shape: (n_trials, 22, 751) for pooled data
        - labels shape: (n_trials,) with values 0-3
    """
    with h5py.File(data_path, 'r') as f:
        available_subjects = list(f.keys())

        if subject == 'all':
            # Pool all subjects
            all_signals = []
            all_labels = []

            for subj in sorted(available_subjects):
                signals = f[subj]['signals'][:]
                labels = f[subj]['labels'][:]
                all_signals.append(signals)
                all_labels.append(labels)
                print(f"Loaded {subj}: {signals.shape}")

            signals = np.concatenate(all_signals, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            print(f"Total pooled data: {signals.shape}")
        else:
            # Load single subject
            if subject not in f:
                print(f"Subject {subject} not found. Available: {available_subjects}")
                subject = available_subjects[0]
                print(f"Using {subject}")

            signals = f[subject]['signals'][:]
            labels = f[subject]['labels'][:]

    # Adjust labels to 0-based indexing
    unique_labels = np.unique(labels)
    if unique_labels.min() > 0:
        labels = labels - unique_labels.min()

    return signals, labels


def create_model(model_type, n_channels, n_samples, num_classes, device):
    """Create baseline model."""
    if model_type == 'cnn1d':
        model = CNN1D(
            num_classes=num_classes,
            in_channels=n_channels,
            seq_length=n_samples,
            base_filters=64,
            dropout=0.5
        )
    elif model_type == 'bilstm':
        model = LSTMClassifier(
            num_classes=num_classes,
            in_channels=n_channels,
            seq_length=n_samples,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            dropout=0.5
        )
    elif model_type == 'transformer':
        model = TransformerClassifier(
            num_classes=num_classes,
            in_channels=n_channels,
            seq_length=n_samples,
            d_model=128,
            nhead=8,
            num_layers=4,
            dim_feedforward=512,
            dropout=0.1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for signals, labels in dataloader:
        signals = signals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate model and compute loss, accuracy, F1, and Kappa."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for signals, labels in dataloader:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(dataloader)
    test_acc = 100. * correct / total
    test_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    test_kappa = cohen_kappa_score(all_labels, all_preds)

    return test_loss, test_acc, test_f1, test_kappa, all_preds, all_labels


def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer,
                                device, max_epochs, patience, logger, warmup_epochs=50):
    """Train with early stopping based on Kappa (disabled for first warmup_epochs).

    Args:
        warmup_epochs: Number of epochs to train before early stopping kicks in (default: 50)
    """
    best_val_kappa = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_kappa': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_kappa': []
    }

    for epoch in range(max_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate with metrics
        val_loss, val_acc, val_f1, val_kappa, _, _ = evaluate(model, val_loader, criterion, device)

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_kappa'].append(val_kappa)

        logger.info(f"Epoch {epoch+1}/{max_epochs} - "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1: {val_f1:.4f}, Val Kappa: {val_kappa:.4f}")

        # Early stopping based on Kappa (only after warmup_epochs)
        if epoch >= warmup_epochs:
            if val_kappa > best_val_kappa:
                best_val_kappa = val_kappa
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}. Best Val Kappa: {best_val_kappa:.4f} (Acc: {best_val_acc:.2f}%) at epoch {best_epoch}")
                break
        else:
            # Warmup phase - just track best kappa
            if val_kappa > best_val_kappa:
                best_val_kappa = val_kappa
                best_val_acc = val_acc
                best_epoch = epoch + 1

    return history, best_val_kappa, best_val_acc, best_epoch


def run_kfold_cv(signals, labels, model_type, device, k_folds, max_epochs,
                 patience, output_dir, logger):
    """Run K-fold cross-validation."""
    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    n_channels = signals.shape[1]
    n_samples = signals.shape[2]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(signals, labels)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold+1}/{k_folds}")
        logger.info(f"{'='*60}")

        # Split data
        X_train, y_train = signals[train_idx], labels[train_idx]
        X_val, y_val = signals[val_idx], labels[val_idx]

        # Create dataloaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Create model
        model = create_model(model_type, n_channels, n_samples, NUM_CLASSES, device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        # Train
        start_time = time.time()
        history, best_val_kappa, best_val_acc, best_epoch = train_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer,
            device, max_epochs, patience, logger, warmup_epochs=50
        )
        training_time = time.time() - start_time

        # Store results
        fold_result = {
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'best_val_kappa': best_val_kappa,
            'best_epoch': best_epoch,
            'training_time_s': training_time,
            'history': history
        }
        fold_results.append(fold_result)

        logger.info(f"Fold {fold+1} - Best Val Kappa: {best_val_kappa:.4f}, Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")

    # Aggregate results
    val_accs = [r['best_val_acc'] for r in fold_results]
    val_kappas = [r['best_val_kappa'] for r in fold_results]
    mean_acc = np.mean(val_accs)
    std_acc = np.std(val_accs)
    mean_kappa = np.mean(val_kappas)
    std_kappa = np.std(val_kappas)

    logger.info(f"\n{'='*60}")
    logger.info(f"K-Fold CV Results ({k_folds} folds)")
    logger.info(f"{'='*60}")
    logger.info(f"Mean Accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
    logger.info(f"Mean Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")
    logger.info(f"Individual Accuracy: {[f'{acc:.2f}%' for acc in val_accs]}")
    logger.info(f"Individual Kappa: {[f'{kappa:.4f}' for kappa in val_kappas]}")

    return fold_results, mean_acc, std_acc, mean_kappa, std_kappa


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup output directory
    subject_label = args.subject if args.subject != 'all' else 'all_subjects'
    output_dir = Path(args.output_dir) / args.model / subject_label
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(f'{args.model}_{subject_label}',
                          output_dir / f'{args.model}_{subject_label}.log')

    logger.info("="*60)
    logger.info(f"Phase 3: Baseline Training - {args.model.upper()}")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    data_desc = "All subjects (pooled)" if args.subject == 'all' else f"Subject: {args.subject}"
    logger.info(f"Data: {data_desc}")
    logger.info(f"Device: {args.device}")
    logger.info(f"K-Folds: {args.k_folds}")
    logger.info(f"Max Epochs: {MAX_EPOCHS}")
    logger.info(f"Early Stop Patience: {EARLY_STOP_PATIENCE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Optimizer: AdamW")

    # Load data
    logger.info(f"\nLoading data from {args.data_path}...")
    signals, labels = load_data(args.data_path, args.subject)
    logger.info(f"Data shape: {signals.shape}, Labels: {labels.shape}")
    logger.info(f"Classes: {np.unique(labels)}")

    # Run K-fold CV
    fold_results, mean_acc, std_acc, mean_kappa, std_kappa = run_kfold_cv(
        signals, labels, args.model, args.device, args.k_folds,
        MAX_EPOCHS, EARLY_STOP_PATIENCE, output_dir, logger
    )

    # Save results
    results = {
        'model': args.model,
        'subject': args.subject,
        'data': 'all_subjects_pooled' if args.subject == 'all' else args.subject,
        'device': args.device,
        'k_folds': args.k_folds,
        'optimizer': 'AdamW',
        'loss_function': 'CrossEntropyLoss',
        'early_stopping_metric': 'Cohen Kappa',
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'mean_kappa': float(mean_kappa),
        'std_kappa': float(std_kappa),
        'fold_results': fold_results,
        'config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'max_epochs': MAX_EPOCHS,
            'patience': EARLY_STOP_PATIENCE,
            'warmup_epochs': 50,
            'seed': args.seed
        },
        'timestamp': datetime.now().isoformat()
    }

    results_file = output_dir / f'{args.model}_{subject_label}_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")
    logger.info(f"\n{'='*60}")
    logger.info(f"TRAINING COMPLETE")
    logger.info(f"Final Result - Accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
    logger.info(f"Final Result - Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
