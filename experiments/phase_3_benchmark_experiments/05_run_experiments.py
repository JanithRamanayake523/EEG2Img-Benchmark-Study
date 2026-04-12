"""
Phase 3.4: Main Experiment Runner

Orchestrates complete benchmark experiments:
1. Load configuration
2. Load and preprocess data
3. Apply image transformation
4. Initialize model
5. Run k-fold or LOSO validation
6. Save metrics and checkpoints
7. Optional: Log to Weights & Biases

Usage:
    python experiments/phase_3_benchmark_experiments/05_run_experiments.py \\
        --config experiments/configs/phase3/example_experiment.yaml \\
        --device cuda:0

    python experiments/phase_3_benchmark_experiments/05_run_experiments.py \\
        --transform gaf_summation \\
        --model resnet18 \\
        --subject A01T \\
        --device cuda:0
"""

import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import h5py
import json
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, cohen_kappa_score
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transforms import get_transformer
from src.models import get_model
from src.utils.logging import setup_logger

# Default configuration
DEFAULT_CONFIG = {
    'data_path': 'data/BCI_IV_2a_EEG_only.hdf5',
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'max_epochs': 100,
    'patience': 15,
    'k_folds': 5,
    'num_classes': 4,
    'seed': 42
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run Phase 3 benchmark experiments')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--transform', type=str, help='Transform type (overrides config)')
    parser.add_argument('--model', type=str, help='Model type (overrides config)')
    parser.add_argument('--subject', type=str, default='all',
                       help='Subject ID (e.g., A01T, A02T) or "all" to pool all subjects')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='results/phase3/experiments')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(file_config, args):
    """Merge file config with command-line arguments."""
    config = DEFAULT_CONFIG.copy()

    if file_config:
        config.update(file_config)

    # Command-line overrides
    if args.transform:
        config['transform'] = args.transform
    if args.model:
        config['model'] = args.model
    if args.subject:
        config['subject'] = args.subject
    if args.device:
        config['device'] = args.device
    if args.seed:
        config['seed'] = args.seed

    return config


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_eeg_data(data_path, subject):
    """Load preprocessed EEG-only data.

    Args:
        data_path: Path to HDF5 file (containing 22 EEG channels only, no EOG)
        subject: Subject ID (e.g., 'A01T') or 'all' to pool all subjects

    Returns:
        signals, labels (pooled across subjects if subject='all')
        - signals shape: (n_trials, 22, 751) for pooled data
        - labels shape: (n_trials,) with values 0-3
    """
    with h5py.File(data_path, 'r') as f:
        available_subjects = [s for s in sorted(f.keys()) if 'subject_' in s or s.startswith('A')]

        if subject == 'all':
            # Pool all subjects
            all_signals = []
            all_labels = []

            for subj in sorted(available_subjects):
                try:
                    signals = f[subj]['signals'][:]
                    labels = f[subj]['labels'][:]
                    all_signals.append(signals)
                    all_labels.append(labels)
                    print(f"Loaded {subj}: {signals.shape}")
                except Exception as e:
                    print(f"Warning: Could not load {subj}: {e}")
                    continue

            signals = np.concatenate(all_signals, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            print(f"Total pooled data: {signals.shape}")
        else:
            # Load single subject
            if subject not in f:
                # Try with 'subject_' prefix
                prefixed_subject = f'subject_{subject}'
                if prefixed_subject in f:
                    subject = prefixed_subject
                else:
                    print(f"Subject {subject} not found. Available: {available_subjects}")
                    subject = available_subjects[0]
                    print(f"Using {subject}")

            signals = f[subject]['signals'][:]
            labels = f[subject]['labels'][:]

    # Normalize labels to 0-based
    unique_labels = np.unique(labels)
    if unique_labels.min() > 0:
        labels = labels - unique_labels.min()

    return signals, labels


def transform_signals_to_images(signals, transform_config, logger):
    """Transform EEG signals to images."""
    logger.info(f"Transforming {len(signals)} signals to images...")

    # Get transformer
    transform_type = transform_config.get('type', transform_config.get('transform', 'gaf_summation'))
    # Remove 'type' and 'transform' keys before passing to get_transformer
    transformer_kwargs = {k: v for k, v in transform_config.items() if k not in ['type', 'transform']}
    transformer = get_transformer(transform_type, **transformer_kwargs)

    n_samples = len(signals)
    batch_size = 32
    all_images = []

    start_time = time.time()

    for i in range(0, n_samples, batch_size):
        batch_signals = signals[i:i+batch_size]
        batch_images = []

        for epoch in batch_signals:
            image = transformer.transform_epoch(epoch)

            # Ensure (C, H, W) format
            if image.ndim == 2:
                image = image[np.newaxis, ...]
            elif image.ndim == 3:
                # Average across channels if multi-channel
                image = image.mean(axis=0)[np.newaxis, ...]

            batch_images.append(image)

        all_images.append(np.array(batch_images))

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  Processed {min(i+batch_size, n_samples)}/{n_samples}")

    images = np.concatenate(all_images, axis=0)
    elapsed = time.time() - start_time

    logger.info(f"Transformation complete: {images.shape} in {elapsed:.2f}s "
               f"({elapsed/n_samples*1000:.2f}ms per sample)")

    return images


def create_model_from_config(model_config, num_classes, in_channels, device):
    """Create model from configuration."""
    model_type = model_config.get('architecture', model_config.get('model', 'resnet18'))

    model = get_model(
        model_type,
        num_classes=num_classes,
        in_channels=in_channels,
        **{k: v for k, v in model_config.items() if k not in ['architecture', 'model', 'type']}
    )

    return model.to(device)


def train_epoch(model, dataloader, criterion, optimizer, device, logger=None):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
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
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
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

    return test_loss, test_acc, test_f1, test_kappa, np.array(all_preds), np.array(all_labels)


def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer,
                               scheduler, device, max_epochs, patience, logger, warmup_epochs=50):
    """Train with early stopping based on Kappa (disabled for first warmup_epochs).

    Args:
        warmup_epochs: Number of epochs to train before early stopping kicks in (default: 50)
    """
    best_val_kappa = -1.0
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_kappa': []
    }

    for epoch in range(max_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, logger)

        # Validate with metrics
        val_loss, val_acc, val_f1, val_kappa, _, _ = evaluate(model, val_loader, criterion, device)

        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Store history
        history['train_loss'].append(float(train_loss))
        history['train_acc'].append(float(train_acc))
        history['val_loss'].append(float(val_loss))
        history['val_acc'].append(float(val_acc))
        history['val_f1'].append(float(val_f1))
        history['val_kappa'].append(float(val_kappa))

        logger.info(f"Epoch {epoch+1}/{max_epochs} - "
                   f"Train: {train_loss:.4f}/{train_acc:.2f}% - "
                   f"Val: {val_loss:.4f}/{val_acc:.2f}%, F1: {val_f1:.4f}, Kappa: {val_kappa:.4f}")

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


def run_kfold_experiment(images, labels, config, logger):
    """Run K-fold cross-validation experiment."""
    k_folds = config.get('k_folds', 5)
    device = config['device']
    num_classes = config['num_classes']
    in_channels = images.shape[1]

    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=config['seed'])
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(images, labels)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold+1}/{k_folds}")
        logger.info(f"{'='*60}")

        # Split data
        X_train, y_train = images[train_idx], labels[train_idx]
        X_val, y_val = images[val_idx], labels[val_idx]

        # Create dataloaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True if device != 'cpu' else False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0,
            pin_memory=True if device != 'cpu' else False
        )

        # Create model
        model = create_model_from_config(
            config.get('model_config', {}),
            num_classes,
            in_channels,
            device
        )

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.0001)
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('max_epochs', 100)
        )

        # Train
        start_time = time.time()
        history, best_val_kappa, best_val_acc, best_epoch = train_with_early_stopping(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            device, config.get('max_epochs', 100), config.get('patience', 15), logger,
            warmup_epochs=50
        )
        training_time = time.time() - start_time

        # Final evaluation
        final_loss, final_acc, final_f1, final_kappa, preds, true_labels = evaluate(model, val_loader, criterion, device)

        # Store results
        fold_result = {
            'fold': fold + 1,
            'best_val_acc': float(best_val_acc),
            'best_val_kappa': float(best_val_kappa),
            'final_val_acc': float(final_acc),
            'final_val_f1': float(final_f1),
            'final_val_kappa': float(final_kappa),
            'best_epoch': int(best_epoch),
            'training_time_s': float(training_time),
            'history': history,
            'predictions': preds.tolist(),
            'true_labels': true_labels.tolist()
        }
        fold_results.append(fold_result)

        logger.info(f"Fold {fold+1} completed - Best Kappa: {best_val_kappa:.4f} (Acc: {best_val_acc:.2f}%), Final Kappa: {final_kappa:.4f} (Acc: {final_acc:.2f}%)")

    # Aggregate results
    val_accs = [r['best_val_acc'] for r in fold_results]
    val_kappas = [r['best_val_kappa'] for r in fold_results]
    mean_acc = np.mean(val_accs)
    std_acc = np.std(val_accs)
    mean_kappa = np.mean(val_kappas)
    std_kappa = np.std(val_kappas)

    logger.info(f"\n{'='*60}")
    logger.info(f"K-Fold Results")
    logger.info(f"{'='*60}")
    logger.info(f"Mean Accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
    logger.info(f"Mean Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")
    logger.info(f"Individual Accuracy: {[f'{acc:.2f}%' for acc in val_accs]}")
    logger.info(f"Individual Kappa: {[f'{kappa:.4f}' for kappa in val_kappas]}")
    logger.info(f"{'='*60}")

    return fold_results, mean_acc, std_acc, mean_kappa, std_kappa


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load configuration
    file_config = load_config(args.config) if args.config else {}
    config = merge_configs(file_config, args)

    # Setup output directory
    exp_name = f"{config.get('transform', 'unknown')}_{config.get('model', 'unknown')}_{config['subject']}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger(exp_name, output_dir / f'{exp_name}.log')

    logger.info("="*60)
    logger.info("Phase 3: Benchmark Experiment")
    logger.info("="*60)
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Transform: {config.get('transform')}")
    logger.info(f"Model: {config.get('model')}")
    logger.info(f"Subject: {config['subject']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"K-Folds: {config.get('k_folds', 5)}")
    logger.info(f"Optimizer: AdamW")
    logger.info(f"Early Stopping Metric: Cohen Kappa")

    # Load data
    logger.info(f"\nLoading EEG data...")
    signals, labels = load_eeg_data(config['data_path'], config['subject'])
    logger.info(f"Data shape: {signals.shape}")

    # Transform to images
    logger.info(f"\nApplying transformation...")
    transform_config = config.get('transform_config', {'type': config.get('transform', 'gaf_summation')})
    images = transform_signals_to_images(signals, transform_config, logger)
    logger.info(f"Image shape: {images.shape}")

    # Run experiment
    logger.info(f"\nRunning K-fold cross-validation...")
    fold_results, mean_acc, std_acc, mean_kappa, std_kappa = run_kfold_experiment(images, labels, config, logger)

    # Save results
    results = {
        'experiment': exp_name,
        'optimizer': 'AdamW',
        'loss_function': 'CrossEntropyLoss',
        'early_stopping_metric': 'Cohen Kappa',
        'config': {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict)) else v
                   for k, v in config.items()},
        'mean_accuracy': float(mean_acc),
        'std_accuracy': float(std_acc),
        'mean_kappa': float(mean_kappa),
        'std_kappa': float(std_kappa),
        'fold_results': fold_results,
        'timestamp': datetime.now().isoformat()
    }

    results_file = output_dir / f'{exp_name}_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")
    logger.info(f"\nFINAL RESULT - Accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
    logger.info(f"FINAL RESULT - Kappa: {mean_kappa:.4f} ± {std_kappa:.4f}")


if __name__ == '__main__':
    main()
