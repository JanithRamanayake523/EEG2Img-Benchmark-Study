"""
Phase 3.1: Model Architecture Validation Script

Validates all model architectures:
- CNN models: ResNet-18, ResNet-50, Lightweight CNN
- ViT models: Vision Transformer Base, Vision Transformer Small, Vision Transformer Tiny
- Baseline models: 1D CNN, LSTM, BiLSTM, Transformer, EEGNet (raw time-series)

Checks:
1. Forward pass validity
2. Parameter count
3. Output shape correctness
4. GPU/CPU compatibility
5. Memory usage
6. Inference speed
"""

import torch
import torch.nn as nn
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import get_model, list_models

# Configuration
RANDOM_SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 4
BATCH_SIZE = 8
SEQ_LENGTH = 751  # For time-series models (3.004 sec at 250 Hz)
EEG_CHANNELS = 22  # Number of EEG channels (EOG excluded)
IMAGE_CHANNELS = 1  # For transformed images (averaged across EEG channels)
IMAGE_SIZE = 128  # Image size for transformed EEG signals
SAVE_DIR = Path('results/phase3/validation')

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def count_parameters(model):
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def validate_model(model_name, architecture, model_type, num_classes=NUM_CLASSES,
                   input_shape=None, **kwargs):
    """
    Validate a single model architecture.

    Args:
        model_name: Human-readable name
        architecture: Architecture type (e.g., 'resnet18', 'vit_base', 'cnn1d')
        model_type: Type of model ('image' or 'timeseries')
        num_classes: Number of output classes
        input_shape: Shape of input tensor
        **kwargs: Additional model-specific arguments

    Returns:
        Dictionary with validation results
    """
    print(f"\n[INFO] Validating {model_name}...")

    # Default input shapes
    if input_shape is None:
        if model_type == 'image':
            input_shape = (BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        else:  # timeseries
            input_shape = (BATCH_SIZE, EEG_CHANNELS, SEQ_LENGTH)

    results = {
        'name': model_name,
        'architecture': architecture,
        'type': model_type,
        'num_classes': num_classes,
        'input_shape': input_shape,
        'status': 'PASS',
        'messages': []
    }

    try:
        # Set appropriate channels for image models
        if model_type == 'image':
            kwargs['in_channels'] = IMAGE_CHANNELS
        elif model_type == 'timeseries':
            kwargs['in_channels'] = EEG_CHANNELS
            kwargs['seq_length'] = SEQ_LENGTH

            # EEGNet expects (batch, 1, channels, time)
            if architecture == 'eegnet':
                input_shape = (BATCH_SIZE, 1, EEG_CHANNELS, SEQ_LENGTH)

        # Initialize model
        model = get_model(architecture, num_classes=num_classes, **kwargs)
        model = model.to(DEVICE)
        model.eval()
        results['messages'].append("[OK] Model initialized and moved to device")

        # Count parameters
        trainable_params, total_params = count_parameters(model)
        results['trainable_params'] = int(trainable_params)
        results['total_params'] = int(total_params)
        results['messages'].append(
            f"[OK] Parameters: {trainable_params:,} trainable, {total_params:,} total"
        )

        # Create dummy input
        dummy_input = torch.randn(input_shape, device=DEVICE)

        # Forward pass and measure time
        with torch.no_grad():
            t_start = time.perf_counter()
            output = model(dummy_input)
            t_elapsed = time.perf_counter() - t_start

        results['forward_time_ms'] = t_elapsed * 1000
        results['throughput_samples_sec'] = BATCH_SIZE / t_elapsed
        results['messages'].append(
            f"[OK] Forward pass: {t_elapsed*1000:.2f} ms ({BATCH_SIZE/t_elapsed:.1f} samples/sec)"
        )

        # Validate output shape
        expected_output_shape = (BATCH_SIZE, num_classes)
        actual_output_shape = tuple(output.shape)

        if actual_output_shape != expected_output_shape:
            results['status'] = 'FAIL'
            results['messages'].append(
                f"[FAIL] Output shape mismatch: expected {expected_output_shape}, got {actual_output_shape}"
            )
        else:
            results['output_shape'] = actual_output_shape
            results['messages'].append(f"[OK] Output shape: {actual_output_shape}")

        # Check for NaN/Inf
        if torch.isnan(output).any():
            results['status'] = 'FAIL'
            results['messages'].append("[FAIL] Output contains NaN values")
        elif torch.isinf(output).any():
            results['status'] = 'FAIL'
            results['messages'].append("[FAIL] Output contains Inf values")
        else:
            results['messages'].append("[OK] No NaN/Inf in output")

        # Output statistics
        results['output_stats'] = {
            'min': float(output.min().cpu()),
            'max': float(output.max().cpu()),
            'mean': float(output.mean().cpu()),
            'std': float(output.std().cpu()),
        }

        # Memory usage
        if DEVICE == 'cuda':
            memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            results['memory_mb'] = memory_mb
            results['messages'].append(f"[OK] GPU memory: {memory_mb:.2f} MB")

        # Test backward pass (gradient flow)
        # CUDNN RNNs require training mode for backward pass
        try:
            model.train()  # Switch to training mode for backward pass
            dummy_input_grad = torch.randn(input_shape, device=DEVICE, requires_grad=True)
            output_grad = model(dummy_input_grad)
            loss = output_grad.sum()
            loss.backward()

            # Check if gradients computed
            if dummy_input_grad.grad is not None:
                results['messages'].append("[OK] Gradient flow verified")
            else:
                results['messages'].append("[WARN] No gradients computed")

        except Exception as e:
            results['status'] = 'FAIL'
            results['messages'].append(f"[FAIL] Backward pass error: {str(e)}")

        finally:
            model.eval()  # Switch back to eval mode

    except Exception as e:
        results['status'] = 'FAIL'
        results['error'] = str(e)
        results['messages'].append(f"[FAIL] Exception: {type(e).__name__}: {str(e)}")

    return results


def validate_all_models():
    """Validate all available model architectures."""

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Define models to validate
    # Format: (model_name, architecture, type, kwargs)
    models_to_test = [
        # Image-based CNN models
        ("ResNet-18", "resnet18", "image", {'pretrained': False}),
        ("ResNet-50", "resnet50", "image", {'pretrained': False}),
        ("Lightweight CNN", "lightweight_cnn", "image", {}),

        # Image-based Vision Transformers
        ("ViT Base", "vit_base", "image", {'pretrained': False}),
        ("ViT Small", "vit_small", "image", {'pretrained': False}),
        ("ViT Tiny", "vit_tiny", "image", {'pretrained': False}),

        # Raw time-series baseline models
        ("1D CNN", "cnn1d", "timeseries", {}),
        ("LSTM", "lstm", "timeseries", {}),
        ("BiLSTM", "bilstm", "timeseries", {}),
        ("Transformer", "transformer", "timeseries", {}),
        ("EEGNet", "eegnet", "timeseries", {}),
    ]

    all_results = []

    print("="*70)
    print("PHASE 3.1: MODEL ARCHITECTURE VALIDATION")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Num Classes: {NUM_CLASSES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print()

    for model_name, architecture, model_type, kwargs in models_to_test:
        results = validate_model(model_name, architecture, model_type, NUM_CLASSES, **kwargs)
        all_results.append(results)

        # Print summary
        status_char = '[OK]' if results['status'] == 'PASS' else '[FAIL]'
        print(f"{status_char} {model_name} ({architecture})")

        if 'output_shape' in results:
            print(f"    Output Shape: {results['output_shape']}")

        if 'total_params' in results:
            params_m = results['total_params'] / 1e6
            print(f"    Parameters: {params_m:.1f}M")

        if 'forward_time_ms' in results:
            print(f"    Forward Time: {results['forward_time_ms']:.2f} ms")

        if 'throughput_samples_sec' in results:
            print(f"    Throughput: {results['throughput_samples_sec']:.1f} samples/sec")

        for msg in results['messages']:
            if '[FAIL]' in msg or '[WARN]' in msg:
                print(f"    {msg}")

    # Save detailed results
    results_file = SAVE_DIR / 'model_validation_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[OK] Detailed results saved to {results_file}")

    # Create summary table
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Type':<12} {'Status':<8} {'Parameters':<15} {'Time (ms)':<10}")
    print("-"*70)

    for result in all_results:
        name = result['name'][:19]
        model_type = result['type'][:11]
        status = result['status']
        params = f"{result.get('total_params', 0) / 1e6:.1f}M" if 'total_params' in result else 'N/A'
        time_ms = result.get('forward_time_ms', 0)
        print(f"{name:<20} {model_type:<12} {status:<8} {params:<15} {time_ms:>9.2f}")

    return all_results


if __name__ == '__main__':
    print(f"\nPhase 3.1: Model Architecture Validation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print(f"Random Seed: {RANDOM_SEED}\n")

    results = validate_all_models()

    # Count passes/fails
    n_pass = sum(1 for r in results if r['status'] == 'PASS')
    n_fail = sum(1 for r in results if r['status'] == 'FAIL')

    print(f"\n" + "="*70)
    print(f"FINAL RESULT: {n_pass} PASSED, {n_fail} FAILED")
    print("="*70)

    if n_fail == 0:
        print("[OK] All models validated successfully!")
        print(f"Results saved to {SAVE_DIR}")
    else:
        print(f"[WARN] {n_fail} model(s) failed validation")
        print("Check results file for details")
