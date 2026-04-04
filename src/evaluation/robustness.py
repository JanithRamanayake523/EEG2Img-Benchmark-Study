"""
Robustness testing for EEG classification models.

Tests model performance under various perturbations:
- Gaussian noise injection (varying SNR levels)
- Channel dropout (simulating sensor failures)
- Temporal shifts (testing invariance to timing)
- Cross-session/subject generalization
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple, Union, Callable
from tqdm import tqdm


def add_gaussian_noise(
    data: np.ndarray,
    snr_db: float,
    signal_axis: Optional[int] = None
) -> np.ndarray:
    """
    Add Gaussian noise to data at specified signal-to-noise ratio.

    Args:
        data: Input data, shape (batch, channels, ...) or (batch, ...)
        snr_db: Signal-to-noise ratio in decibels
               Higher values = less noise
               Examples: 20 dB (clean), 10 dB (moderate), 0 dB (high noise)
        signal_axis: Axis along which to compute signal power
                    If None, use all axes

    Returns:
        Noisy data with same shape as input

    Example:
        >>> clean_eeg = np.random.randn(100, 25, 751)  # (samples, channels, time)
        >>> noisy_eeg = add_gaussian_noise(clean_eeg, snr_db=10)
        >>> print(f"Added noise at 10 dB SNR")
    """
    # Compute signal power
    if signal_axis is None:
        signal_power = np.mean(data ** 2)
    else:
        signal_power = np.mean(data ** 2, axis=signal_axis, keepdims=True)

    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)

    # Compute noise power needed for desired SNR
    noise_power = signal_power / snr_linear

    # Generate Gaussian noise
    noise = np.random.randn(*data.shape) * np.sqrt(noise_power)

    # Add noise to signal
    noisy_data = data + noise

    return noisy_data


def channel_dropout(
    data: np.ndarray,
    dropout_rate: float,
    channel_axis: int = 1
) -> np.ndarray:
    """
    Randomly zero out entire channels (simulating sensor failures).

    Args:
        data: Input data, shape (batch, channels, ...)
        dropout_rate: Fraction of channels to drop (0.0 to 1.0)
        channel_axis: Axis corresponding to channels

    Returns:
        Data with randomly dropped channels

    Example:
        >>> eeg = np.random.randn(100, 25, 751)  # (samples, 25 channels, time)
        >>> eeg_dropped = channel_dropout(eeg, dropout_rate=0.2)  # Drop 20% of channels
        >>> print(f"Dropped ~5 channels")
    """
    data_copy = data.copy()

    # Get number of channels
    n_channels = data.shape[channel_axis]
    n_drop = int(n_channels * dropout_rate)

    if n_drop == 0:
        return data_copy

    # For each sample, randomly drop channels
    for i in range(data.shape[0]):
        # Randomly select channels to drop
        drop_indices = np.random.choice(n_channels, size=n_drop, replace=False)

        # Create indexing tuple to zero out selected channels
        idx = [slice(None)] * data.ndim
        idx[0] = i
        idx[channel_axis] = drop_indices

        data_copy[tuple(idx)] = 0

    return data_copy


def temporal_shift(
    data: np.ndarray,
    max_shift_samples: int,
    time_axis: int = -1
) -> np.ndarray:
    """
    Apply random temporal shifts to each sample.

    Args:
        data: Input data, shape (batch, ..., time)
        max_shift_samples: Maximum shift in samples (positive or negative)
        time_axis: Axis corresponding to time dimension

    Returns:
        Temporally shifted data

    Example:
        >>> eeg = np.random.randn(100, 25, 751)
        >>> eeg_shifted = temporal_shift(eeg, max_shift_samples=50)
        >>> print(f"Applied random shifts up to ±50 samples")
    """
    data_copy = data.copy()

    for i in range(data.shape[0]):
        # Random shift for this sample
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)

        if shift == 0:
            continue

        # Roll along time axis
        idx = [slice(None)] * data.ndim
        idx[0] = i

        data_copy[tuple(idx)] = np.roll(data[tuple(idx)], shift, axis=time_axis)

    return data_copy


def add_noise_and_evaluate(
    model: nn.Module,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    snr_db_levels: List[float],
    batch_size: int = 32,
    device: str = 'cuda',
    return_predictions: bool = False
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate model performance across different noise levels.

    Args:
        model: Trained PyTorch model
        test_data: Clean test data, shape (n_samples, ...)
        test_labels: Test labels, shape (n_samples,)
        snr_db_levels: List of SNR levels to test (in dB)
                      e.g., [20, 15, 10, 5, 0, -5]
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        return_predictions: If True, include predictions in output

    Returns:
        Dictionary mapping SNR levels to metrics:
        {snr_db: {'accuracy': ..., 'loss': ..., 'predictions': ...}}

    Example:
        >>> model = torch.load('model.pt')
        >>> test_data = np.random.randn(1000, 25, 64, 64)
        >>> test_labels = np.random.randint(0, 4, 1000)
        >>> results = add_noise_and_evaluate(
        ...     model, test_data, test_labels,
        ...     snr_db_levels=[20, 15, 10, 5, 0]
        ... )
        >>> for snr, metrics in results.items():
        ...     print(f"SNR {snr} dB: Accuracy = {metrics['accuracy']:.4f}")
    """
    model.eval()
    model.to(device)

    results = {}

    for snr_db in snr_db_levels:
        # Add noise at this SNR level
        noisy_data = add_gaussian_noise(test_data, snr_db=snr_db)

        # Convert to tensors
        tensor_data = torch.FloatTensor(noisy_data)
        tensor_labels = torch.LongTensor(test_labels)

        # Create data loader
        dataset = TensorDataset(tensor_data, tensor_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Evaluate
        all_preds = []
        all_labels = []
        total_loss = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, labels in loader:
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * data.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()

                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        # Aggregate results
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        accuracy = np.mean(all_preds == all_labels)
        avg_loss = total_loss / len(test_data)

        results[snr_db] = {
            'accuracy': float(accuracy),
            'loss': float(avg_loss),
        }

        if return_predictions:
            results[snr_db]['predictions'] = all_preds

    return results


def channel_dropout_test(
    model: nn.Module,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    dropout_rates: List[float],
    batch_size: int = 32,
    device: str = 'cuda',
    channel_axis: int = 1,
    n_trials: int = 5
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate model robustness to channel dropouts.

    Args:
        model: Trained PyTorch model
        test_data: Test data, shape (n_samples, channels, ...)
        test_labels: Test labels
        dropout_rates: List of dropout rates to test (0.0 to 1.0)
                      e.g., [0.0, 0.1, 0.2, 0.3, 0.5]
        batch_size: Batch size for evaluation
        device: Device to run on
        channel_axis: Axis corresponding to channels
        n_trials: Number of random dropout trials per rate (for averaging)

    Returns:
        Dictionary mapping dropout rates to metrics

    Example:
        >>> results = channel_dropout_test(
        ...     model, test_data, test_labels,
        ...     dropout_rates=[0.0, 0.1, 0.2, 0.3, 0.5],
        ...     n_trials=5
        ... )
        >>> for rate, metrics in results.items():
        ...     print(f"Dropout {rate*100:.0f}%: Accuracy = {metrics['accuracy']:.4f}")
    """
    model.eval()
    model.to(device)

    results = {}

    for dropout_rate in dropout_rates:
        trial_accuracies = []

        for trial in range(n_trials):
            # Apply channel dropout
            if dropout_rate > 0:
                dropped_data = channel_dropout(test_data, dropout_rate, channel_axis)
            else:
                dropped_data = test_data

            # Convert to tensors
            tensor_data = torch.FloatTensor(dropped_data)
            tensor_labels = torch.LongTensor(test_labels)

            # Create data loader
            dataset = TensorDataset(tensor_data, tensor_labels)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Evaluate
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for data, labels in loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    preds = outputs.argmax(dim=1).cpu().numpy()

                    all_preds.append(preds)
                    all_labels.append(labels.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            accuracy = np.mean(all_preds == all_labels)
            trial_accuracies.append(accuracy)

        # Average across trials
        results[dropout_rate] = {
            'accuracy': float(np.mean(trial_accuracies)),
            'std': float(np.std(trial_accuracies)),
            'trials': trial_accuracies
        }

    return results


def temporal_shift_test(
    model: nn.Module,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    max_shift_ms: List[float],
    sampling_rate: float,
    batch_size: int = 32,
    device: str = 'cuda',
    time_axis: int = -1,
    n_trials: int = 5
) -> Dict[float, Dict[str, float]]:
    """
    Evaluate model robustness to temporal shifts.

    Args:
        model: Trained PyTorch model
        test_data: Test data
        test_labels: Test labels
        max_shift_ms: List of maximum shift amounts in milliseconds
                     e.g., [0, 50, 100, 200]
        sampling_rate: Sampling rate in Hz (e.g., 250 Hz)
        batch_size: Batch size
        device: Device
        time_axis: Time axis in data
        n_trials: Number of random shift trials per max_shift

    Returns:
        Dictionary mapping shift amounts to metrics

    Example:
        >>> results = temporal_shift_test(
        ...     model, test_data, test_labels,
        ...     max_shift_ms=[0, 50, 100, 200],
        ...     sampling_rate=250
        ... )
        >>> for shift_ms, metrics in results.items():
        ...     print(f"Shift ±{shift_ms}ms: Accuracy = {metrics['accuracy']:.4f}")
    """
    model.eval()
    model.to(device)

    results = {}

    for shift_ms in max_shift_ms:
        # Convert ms to samples
        max_shift_samples = int(shift_ms * sampling_rate / 1000)

        trial_accuracies = []

        for trial in range(n_trials):
            # Apply temporal shift
            if max_shift_samples > 0:
                shifted_data = temporal_shift(test_data, max_shift_samples, time_axis)
            else:
                shifted_data = test_data

            # Convert to tensors
            tensor_data = torch.FloatTensor(shifted_data)
            tensor_labels = torch.LongTensor(test_labels)

            # Create data loader
            dataset = TensorDataset(tensor_data, tensor_labels)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            # Evaluate
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for data, labels in loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = model(data)
                    preds = outputs.argmax(dim=1).cpu().numpy()

                    all_preds.append(preds)
                    all_labels.append(labels.cpu().numpy())

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            accuracy = np.mean(all_preds == all_labels)
            trial_accuracies.append(accuracy)

        results[shift_ms] = {
            'accuracy': float(np.mean(trial_accuracies)),
            'std': float(np.std(trial_accuracies)),
            'trials': trial_accuracies
        }

    return results


def evaluate_robustness(
    model: nn.Module,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    sampling_rate: float = 250,
    batch_size: int = 32,
    device: str = 'cuda',
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Comprehensive robustness evaluation.

    Tests model under:
    - Gaussian noise (varying SNR)
    - Channel dropouts
    - Temporal shifts

    Args:
        model: Trained model
        test_data: Test data
        test_labels: Test labels
        sampling_rate: Sampling rate in Hz
        batch_size: Batch size
        device: Device
        verbose: Print progress

    Returns:
        Dictionary with all robustness test results

    Example:
        >>> model = torch.load('model.pt')
        >>> test_data = np.load('test_data.npy')
        >>> test_labels = np.load('test_labels.npy')
        >>> results = evaluate_robustness(model, test_data, test_labels)
        >>> print(f"Noise robustness: {results['noise']}")
        >>> print(f"Dropout robustness: {results['dropout']}")
        >>> print(f"Shift robustness: {results['shift']}")
    """
    results = {}

    if verbose:
        print("\n" + "="*60)
        print("Robustness Evaluation")
        print("="*60)

    # Test noise robustness
    if verbose:
        print("\n[1/3] Testing noise robustness...")

    snr_levels = [20, 15, 10, 5, 0, -5]
    results['noise'] = add_noise_and_evaluate(
        model, test_data, test_labels,
        snr_db_levels=snr_levels,
        batch_size=batch_size,
        device=device
    )

    if verbose:
        for snr, metrics in results['noise'].items():
            print(f"  SNR {snr:>3} dB: Accuracy = {metrics['accuracy']:.4f}")

    # Test channel dropout robustness
    if verbose:
        print("\n[2/3] Testing channel dropout robustness...")

    dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
    results['dropout'] = channel_dropout_test(
        model, test_data, test_labels,
        dropout_rates=dropout_rates,
        batch_size=batch_size,
        device=device,
        n_trials=5
    )

    if verbose:
        for rate, metrics in results['dropout'].items():
            print(f"  Dropout {rate*100:>3.0f}%: Accuracy = {metrics['accuracy']:.4f} ± {metrics['std']:.4f}")

    # Test temporal shift robustness
    if verbose:
        print("\n[3/3] Testing temporal shift robustness...")

    max_shifts = [0, 50, 100, 200]
    results['shift'] = temporal_shift_test(
        model, test_data, test_labels,
        max_shift_ms=max_shifts,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        device=device,
        n_trials=5
    )

    if verbose:
        for shift_ms, metrics in results['shift'].items():
            print(f"  Shift ±{shift_ms:>3.0f}ms: Accuracy = {metrics['accuracy']:.4f} ± {metrics['std']:.4f}")

    if verbose:
        print("\n" + "="*60)
        print("Robustness evaluation complete")
        print("="*60 + "\n")

    return results
