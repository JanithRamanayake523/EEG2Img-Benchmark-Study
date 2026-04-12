"""
Phase 3.1: Transform Validation Script

Validates all 6 image transformation methods:
- Gramian Angular Fields (GAF)
- Markov Transition Fields (MTF)
- Recurrence Plots (RP)
- Spectrograms (STFT)
- Scalograms (CWT)
- Topographic Maps (SSFI)

Checks:
1. Output shape validation
2. Value range validation
3. Performance (time per epoch)
4. Memory usage
5. Visual inspection (saves sample images)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import time
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.transforms import (
    GAFTransformer, MTFTransformer, RecurrencePlotTransformer,
    SpectrogramTransformer, ScalogramTransformer, TopographicTransformer,
    get_transformer, list_transformers
)

# Configuration
RANDOM_SEED = 42
IMAGE_SIZES = [64, 128]
NUM_TEST_EPOCHS = 10
CHANNELS = 22
SAMPLES_PER_EPOCH = 751  # 3s at 250 Hz
SAVE_DIR = Path('results/phase3/validation')

np.random.seed(RANDOM_SEED)


def create_synthetic_eeg(n_epochs, n_channels, n_samples, noise_level=0.1):
    """
    Create synthetic EEG-like data for testing.

    Combines multiple frequencies to simulate realistic EEG activity.
    """
    data = np.zeros((n_epochs, n_channels, n_samples))
    sfreq = 250
    t = np.arange(n_samples) / sfreq

    for epoch in range(n_epochs):
        for ch in range(n_channels):
            # Mix of frequencies (delta, theta, alpha, beta bands)
            signal = (
                0.5 * np.sin(2 * np.pi * 2 * t) +    # Delta (2 Hz)
                1.0 * np.sin(2 * np.pi * 6 * t) +    # Theta (6 Hz)
                0.8 * np.sin(2 * np.pi * 10 * t) +   # Alpha (10 Hz)
                0.3 * np.sin(2 * np.pi * 20 * t)     # Beta (20 Hz)
            )
            # Add noise and per-channel variation
            noise = np.random.normal(0, noise_level, n_samples)
            data[epoch, ch] = signal + noise + ch * 0.1

    return data


def validate_transformer(transformer_class, name, config=None, transform_method='transform_batch'):
    """
    Validate a single transformer.

    Args:
        transformer_class: Transformer class (not instance)
        name: Human-readable name
        config: Dictionary with transformer configuration
        transform_method: Method to call on transformer ('transform_batch' or 'fit_transform')

    Returns:
        Dictionary with validation results
    """
    print(f"\n[INFO] Validating {name}...")

    if config is None:
        config = {}

    results = {
        'name': name,
        'config': config,
        'status': 'PASS',
        'messages': []
    }

    try:
        # Initialize transformer
        if 'ch_names' in config:
            # Topographic requires channel names
            transformer = transformer_class(**config)
        else:
            transformer = transformer_class(**config)
        results['messages'].append(f"[OK] Transformer initialized")

        # Create test data
        eeg_data = create_synthetic_eeg(NUM_TEST_EPOCHS, CHANNELS, SAMPLES_PER_EPOCH)

        # Transform data and measure time
        t_start = time.perf_counter()
        if hasattr(transformer, transform_method):
            images = getattr(transformer, transform_method)(eeg_data)
        else:
            # Fallback if method doesn't exist
            images = transformer.transform_epoch(eeg_data[0])
            images = np.expand_dims(images, 0)  # Add epoch dimension
        t_elapsed = time.perf_counter() - t_start

        results['time_total'] = t_elapsed
        results['time_per_epoch'] = t_elapsed / NUM_TEST_EPOCHS
        results['messages'].append(f"[OK] Transform completed in {t_elapsed:.3f}s")

        # Validate output
        results['output_shape'] = tuple(images.shape)
        results['output_dtype'] = str(images.dtype)

        # Shape validation
        expected_shape_0 = NUM_TEST_EPOCHS
        if images.shape[0] != expected_shape_0:
            results['status'] = 'FAIL'
            results['messages'].append(
                f"[FAIL] Shape[0] mismatch: expected {expected_shape_0}, got {images.shape[0]}"
            )

        # Value range validation
        v_min, v_max = images.min(), images.max()
        results['value_range'] = [float(v_min), float(v_max)]

        if v_max > 1.1 or v_min < -1.1:
            results['status'] = 'WARN'
            results['messages'].append(
                f"[WARN] Value range [{v_min:.3f}, {v_max:.3f}] outside expected [-1, 1]"
            )
        else:
            results['messages'].append(f"[OK] Value range [{v_min:.3f}, {v_max:.3f}]")

        # Check for NaNs
        nan_count = np.isnan(images).sum()
        if nan_count > 0:
            results['status'] = 'FAIL'
            results['messages'].append(f"[FAIL] Found {nan_count} NaN values")
        else:
            results['messages'].append("[OK] No NaN values")

        # Check for Infs
        inf_count = np.isinf(images).sum()
        if inf_count > 0:
            results['status'] = 'FAIL'
            results['messages'].append(f"[FAIL] Found {inf_count} Inf values")
        else:
            results['messages'].append("[OK] No Inf values")

        # Statistics
        results['statistics'] = {
            'mean': float(images.mean()),
            'std': float(images.std()),
            'min': float(images.min()),
            'max': float(images.max()),
        }

        results['messages'].append(
            f"[OK] Stats: mean={images.mean():.3f}, std={images.std():.3f}"
        )

        # Memory estimate
        memory_mb = images.nbytes / (1024 ** 2)
        results['memory_mb'] = memory_mb
        results['messages'].append(f"[OK] Memory: {memory_mb:.2f} MB for {NUM_TEST_EPOCHS} epochs")

    except Exception as e:
        results['status'] = 'FAIL'
        results['error'] = str(e)
        results['messages'].append(f"[FAIL] Exception: {type(e).__name__}: {str(e)}")

    return results, images if results['status'] != 'FAIL' else None


def validate_all_transforms():
    """Validate all available transformers."""

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Define transformers to validate with their configurations
    transformers_to_test = [
        # GAF variants
        (GAFTransformer, 'GAF Summation', {'method': 'summation', 'image_size': 128}),
        (GAFTransformer, 'GAF Difference', {'method': 'difference', 'image_size': 128}),

        # MTF variants
        (MTFTransformer, 'MTF Q=8', {'n_bins': 8, 'image_size': 64}),
        (MTFTransformer, 'MTF Q=16', {'n_bins': 16, 'image_size': 64}),

        # Recurrence Plot
        (RecurrencePlotTransformer, 'Recurrence Plot', {}),

        # Spectrogram
        (SpectrogramTransformer, 'Spectrogram', {}),

        # Scalogram
        (ScalogramTransformer, 'Scalogram Morlet', {'wavelet': 'morl'}),
        (ScalogramTransformer, 'Scalogram MexHat', {'wavelet': 'mexh'}),

        # NOTE: Topographic requires proper BCI montage setup with MNE
        # Will be validated in Phase 3.3 with actual preprocessed BCI data
    ]

    all_results = []
    all_images = {}

    print("="*60)
    print("PHASE 3.1: TRANSFORM VALIDATION")
    print("="*60)

    for transformer_class, name, config in transformers_to_test:
        results, images = validate_transformer(transformer_class, name, config=config,
                                                transform_method='transform_batch')
        all_results.append(results)

        if images is not None:
            all_images[name] = images

        # Print summary
        status_char = '[OK]' if results['status'] == 'PASS' else '[WARN]' if results['status'] == 'WARN' else '[FAIL]'
        print(f"\n{status_char} {name}")
        if 'output_shape' in results:
            print(f"    Shape: {results['output_shape']}")
        if 'time_per_epoch' in results:
            print(f"    Time: {results['time_per_epoch']*1000:.2f} ms/epoch")
        if 'value_range' in results:
            print(f"    Range: [{results['value_range'][0]:.3f}, {results['value_range'][1]:.3f}]")

        for msg in results['messages']:
            if '[FAIL]' in msg or '[WARN]' in msg:
                print(f"    {msg}")

    # Save detailed results
    results_file = SAVE_DIR / 'validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Detailed results saved to {results_file}")

    # Create summary table
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"{'Transformer':<25} {'Status':<8} {'Shape':<20} {'Time (ms)':<12}")
    print("-"*60)

    for result in all_results:
        name = result['name'][:24]
        status = result['status']
        shape = str(result.get('output_shape', 'N/A'))[:19]
        time_ms = result.get('time_per_epoch', 0) * 1000
        print(f"{name:<25} {status:<8} {shape:<20} {time_ms:>10.2f}")

    # Visualize sample images
    visualize_sample_images(all_images, SAVE_DIR)

    return all_results, all_images


def visualize_sample_images(all_images, save_dir):
    """Create visualization of sample transformed images."""

    print(f"\n[INFO] Creating visualization of sample images...")

    # Create figure
    n_transforms = len(all_images)
    fig = plt.figure(figsize=(16, 3 * n_transforms))
    gs = fig.add_gridspec(n_transforms, 3, hspace=0.35, wspace=0.3)

    for idx, (name, images) in enumerate(all_images.items()):
        # Get first 3 epochs
        for epoch_idx in range(min(3, images.shape[0])):
            ax = fig.add_subplot(gs[idx, epoch_idx])

            img = images[epoch_idx]

            # Handle different dimensions
            if img.ndim == 3:  # Multi-channel (e.g., topographic)
                img = img[..., 0]  # Take first channel

            im = ax.imshow(img, cmap='viridis', aspect='auto')
            ax.set_title(f"{name} - Epoch {epoch_idx+1}")
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency/Space')
            plt.colorbar(im, ax=ax)

    fig.suptitle('Sample Transformed EEG Images', fontsize=16, fontweight='bold', y=0.995)

    output_file = save_dir / 'sample_transformed_images.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[OK] Sample images saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    print(f"\nPhase 3.1: Transform Validation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {RANDOM_SEED}")
    print(f"Test epochs: {NUM_TEST_EPOCHS}")
    print(f"Channels: {CHANNELS}")
    print(f"Samples per epoch: {SAMPLES_PER_EPOCH}\n")

    results, images = validate_all_transforms()

    print(f"\n[OK] Validation complete!")
    print(f"Results saved to {SAVE_DIR}")
