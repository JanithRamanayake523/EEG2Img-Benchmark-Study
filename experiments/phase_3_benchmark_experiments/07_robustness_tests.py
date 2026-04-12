"""
Phase 3.5: Robustness Testing Script

Tests model robustness under various perturbations:
1. Noise injection (SNR levels)
2. Channel dropout
3. Temporal jitter
4. Cross-session generalization

Usage:
    python experiments/phase_3_benchmark_experiments/07_robustness_tests.py \\
        --model_path results/phase3/gaf_resnet18/best_model.pth \\
        --transform gaf_summation \\
        --test_type noise \\
        --device cuda:0
"""

import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import h5py
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transforms import get_transformer
from src.models import get_model
from src.utils.logging import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Robustness testing for Phase 3 models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config_path', type=str, help='Path to experiment config')
    parser.add_argument('--data_path', type=str, default='data/BCI_IV_2a.hdf5')
    parser.add_argument('--subject', type=str, default='A01T')
    parser.add_argument('--transform', type=str, default='gaf_summation')
    parser.add_argument('--test_type', type=str, default='noise',
                        choices=['noise', 'dropout', 'jitter', 'all'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='results/phase3/robustness')
    return parser.parse_args()


def add_gaussian_noise(signals, snr_db):
    """Add Gaussian noise at specified SNR level."""
    # Calculate signal power
    signal_power = np.mean(signals ** 2, axis=-1, keepdims=True)

    # Calculate noise power from SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    # Generate and add noise
    noise = np.random.randn(*signals.shape) * np.sqrt(noise_power)
    noisy_signals = signals + noise

    return noisy_signals


def apply_channel_dropout(signals, dropout_rate):
    """Randomly drop out channels."""
    n_samples, n_channels, n_timepoints = signals.shape
    signals_dropped = signals.copy()

    for i in range(n_samples):
        # Randomly select channels to drop
        n_drop = int(n_channels * dropout_rate)
        if n_drop > 0:
            drop_idx = np.random.choice(n_channels, n_drop, replace=False)
            signals_dropped[i, drop_idx, :] = 0

    return signals_dropped


def apply_temporal_jitter(signals, max_shift_ms, sampling_rate=250):
    """Apply random temporal jitter."""
    max_shift_samples = int(max_shift_ms * sampling_rate / 1000)
    n_samples = len(signals)
    jittered_signals = signals.copy()

    for i in range(n_samples):
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        if shift > 0:
            jittered_signals[i, :, shift:] = signals[i, :, :-shift]
            jittered_signals[i, :, :shift] = 0
        elif shift < 0:
            jittered_signals[i, :, :shift] = signals[i, :, -shift:]
            jittered_signals[i, :, shift:] = 0

    return jittered_signals


def load_data(data_path, subject):
    """Load EEG data."""
    with h5py.File(data_path, 'r') as f:
        if subject not in f:
            available = list(f.keys())
            subject = available[0]

        signals = f[subject]['signals'][:]
        labels = f[subject]['labels'][:]

    unique_labels = np.unique(labels)
    if unique_labels.min() > 0:
        labels = labels - unique_labels.min()

    return signals, labels


def transform_to_images(signals, transformer, batch_size=32):
    """Transform signals to images."""
    n_samples = len(signals)
    all_images = []

    for i in range(0, n_samples, batch_size):
        batch_signals = signals[i:i+batch_size]
        batch_images = []

        for epoch in batch_signals:
            image = transformer.transform(epoch)
            if image.ndim == 2:
                image = image[np.newaxis, ...]
            batch_images.append(image)

        all_images.append(np.array(batch_images))

    images = np.concatenate(all_images, axis=0)
    return images


def evaluate_model(model, images, labels, device, batch_size=32):
    """Evaluate model accuracy."""
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(images), torch.LongTensor(labels))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            outputs = model(imgs)
            _, predicted = outputs.max(1)
            total += lbls.size(0)
            correct += predicted.eq(lbls).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def test_noise_robustness(model, signals, labels, transformer, device, logger):
    """Test robustness to noise at different SNR levels."""
    snr_levels = [20, 15, 10, 5, 0, -5, -10]
    results = []

    logger.info("Testing noise robustness...")

    for snr in snr_levels:
        logger.info(f"  Testing SNR = {snr} dB...")

        # Add noise
        noisy_signals = add_gaussian_noise(signals, snr)

        # Transform to images
        images = transform_to_images(noisy_signals, transformer)

        # Evaluate
        accuracy = evaluate_model(model, images, labels, device)
        results.append({'snr_db': snr, 'accuracy': accuracy})

        logger.info(f"    Accuracy: {accuracy:.2f}%")

    return results


def test_channel_dropout_robustness(model, signals, labels, transformer, device, logger):
    """Test robustness to channel dropout."""
    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results = []

    logger.info("Testing channel dropout robustness...")

    for rate in dropout_rates:
        logger.info(f"  Testing dropout rate = {rate*100:.0f}%...")

        # Apply dropout
        dropped_signals = apply_channel_dropout(signals, rate)

        # Transform to images
        images = transform_to_images(dropped_signals, transformer)

        # Evaluate
        accuracy = evaluate_model(model, images, labels, device)
        results.append({'dropout_rate': rate, 'accuracy': accuracy})

        logger.info(f"    Accuracy: {accuracy:.2f}%")

    return results


def test_temporal_jitter_robustness(model, signals, labels, transformer, device, logger):
    """Test robustness to temporal jitter."""
    jitter_levels = [0, 10, 25, 50, 100, 200]  # milliseconds
    results = []

    logger.info("Testing temporal jitter robustness...")

    for jitter_ms in jitter_levels:
        logger.info(f"  Testing jitter = ±{jitter_ms} ms...")

        # Apply jitter
        jittered_signals = apply_temporal_jitter(signals, jitter_ms)

        # Transform to images
        images = transform_to_images(jittered_signals, transformer)

        # Evaluate
        accuracy = evaluate_model(model, images, labels, device)
        results.append({'jitter_ms': jitter_ms, 'accuracy': accuracy})

        logger.info(f"    Accuracy: {accuracy:.2f}%")

    return results


def plot_robustness_curves(results, output_path):
    """Plot robustness curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Noise robustness
    if 'noise' in results:
        noise_data = results['noise']
        snrs = [r['snr_db'] for r in noise_data]
        accs = [r['accuracy'] for r in noise_data]

        axes[0].plot(snrs, accs, 'o-', linewidth=2)
        axes[0].set_xlabel('SNR (dB)')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title('Noise Robustness')
        axes[0].grid(True, alpha=0.3)

    # Channel dropout robustness
    if 'dropout' in results:
        dropout_data = results['dropout']
        rates = [r['dropout_rate'] * 100 for r in dropout_data]
        accs = [r['accuracy'] for r in dropout_data]

        axes[1].plot(rates, accs, 'o-', linewidth=2)
        axes[1].set_xlabel('Channel Dropout Rate (%)')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Channel Dropout Robustness')
        axes[1].grid(True, alpha=0.3)

    # Temporal jitter robustness
    if 'jitter' in results:
        jitter_data = results['jitter']
        jitters = [r['jitter_ms'] for r in jitter_data]
        accs = [r['accuracy'] for r in jitter_data]

        axes[2].plot(jitters, accs, 'o-', linewidth=2)
        axes[2].set_xlabel('Temporal Jitter (ms)')
        axes[2].set_ylabel('Accuracy (%)')
        axes[2].set_title('Temporal Jitter Robustness')
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = setup_logger('robustness_test', output_dir / 'robustness_test.log')

    logger.info("="*60)
    logger.info("Phase 3: Robustness Testing")
    logger.info("="*60)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Transform: {args.transform}")
    logger.info(f"Test type: {args.test_type}")
    logger.info(f"Device: {args.device}")

    # Load data
    logger.info("\nLoading test data...")
    signals, labels = load_data(args.data_path, args.subject)
    logger.info(f"Data shape: {signals.shape}")

    # Load transformer
    logger.info("\nInitializing transformer...")
    transformer = get_transformer(args.transform)

    # Load model (Note: You'll need to specify model architecture)
    logger.info("\nLoading model...")
    # For now, we'll skip model loading as it requires architecture specification
    # In practice, you'd load from checkpoint with proper architecture
    logger.info("WARNING: Model loading not implemented - using dummy model")
    model = None  # Placeholder

    # Run robustness tests
    results = {}

    if args.test_type in ['noise', 'all']:
        results['noise'] = test_noise_robustness(model, signals, labels, transformer, args.device, logger)

    if args.test_type in ['dropout', 'all']:
        results['dropout'] = test_channel_dropout_robustness(model, signals, labels, transformer, args.device, logger)

    if args.test_type in ['jitter', 'all']:
        results['jitter'] = test_temporal_jitter_robustness(model, signals, labels, transformer, args.device, logger)

    # Save results
    results_file = output_dir / f'robustness_results_{args.test_type}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'test_type': args.test_type,
            'model_path': args.model_path,
            'transform': args.transform,
            'subject': args.subject,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    # Plot results
    plot_path = output_dir / f'robustness_curves_{args.test_type}.png'
    plot_robustness_curves(results, plot_path)
    logger.info(f"Plots saved to {plot_path}")

    logger.info("\n" + "="*60)
    logger.info("Robustness testing complete")
    logger.info("="*60)


if __name__ == '__main__':
    main()
