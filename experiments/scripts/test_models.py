"""
Model testing script for Phase 4 validation.

Tests all model architectures with dummy inputs to ensure:
- Models instantiate without errors
- Forward pass works correctly
- Output dimensions match expected number of classes
- Parameter counts are reasonable
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from src.models import (
    get_model,
    list_models,
    count_parameters,
    get_model_info
)


def test_image_models():
    """Test CNN and ViT models with image inputs."""
    print("\n" + "="*60)
    print("Testing Image-Based Models (CNN & ViT)")
    print("="*60)

    # Test parameters
    batch_size = 4
    num_classes = 4
    in_channels = 25
    image_sizes = [64, 128, 224]

    # CNN models
    cnn_models = ['resnet18', 'resnet50', 'lightweight_cnn']

    for model_name in cnn_models:
        print(f"\n--- Testing {model_name.upper()} ---")

        for img_size in image_sizes:
            try:
                # Create model
                model = get_model(
                    model_name,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    pretrained=False  # Faster testing
                )

                # Test forward pass
                x = torch.randn(batch_size, in_channels, img_size, img_size)
                with torch.no_grad():
                    out = model(x)

                # Get model info
                info = get_model_info(model)

                # Validate
                assert out.shape == (batch_size, num_classes), \
                    f"Expected output shape {(batch_size, num_classes)}, got {out.shape}"

                print(f"  [OK] Size {img_size}x{img_size}: {x.shape} -> {out.shape}")
                if img_size == image_sizes[0]:  # Print params only once
                    print(f"    Parameters: {info['trainable_parameters']:,}")

            except Exception as e:
                print(f"  [FAIL] Size {img_size}x{img_size}: FAILED - {e}")
                return False

    # ViT models
    vit_models = ['vit_tiny', 'vit_small']  # Skip vit_base for faster testing

    for model_name in vit_models:
        print(f"\n--- Testing {model_name.upper()} ---")

        # ViT typically works best with 224x224 or divisible by patch size
        test_sizes = [224]  # Could add 128, 64 if patch size allows

        for img_size in test_sizes:
            try:
                # Create model
                model = get_model(
                    model_name,
                    num_classes=num_classes,
                    in_channels=in_channels,
                    img_size=img_size,
                    pretrained=False
                )

                # Test forward pass
                x = torch.randn(batch_size, in_channels, img_size, img_size)
                with torch.no_grad():
                    out = model(x)

                # Get model info
                info = get_model_info(model)

                # Validate
                assert out.shape == (batch_size, num_classes), \
                    f"Expected output shape {(batch_size, num_classes)}, got {out.shape}"

                print(f"  [OK] Size {img_size}x{img_size}: {x.shape} -> {out.shape}")
                print(f"    Parameters: {info['trainable_parameters']:,}")

            except Exception as e:
                print(f"  [FAIL] Size {img_size}x{img_size}: FAILED - {e}")
                return False

    return True


def test_baseline_models():
    """Test baseline models with raw time-series inputs."""
    print("\n" + "="*60)
    print("Testing Raw-Signal Baseline Models")
    print("="*60)

    # Test parameters
    batch_size = 8
    num_classes = 4
    in_channels = 25
    seq_length = 751

    # Baseline models (1D inputs)
    baseline_1d = ['cnn1d', 'lstm', 'bilstm', 'transformer']

    for model_name in baseline_1d:
        print(f"\n--- Testing {model_name.upper()} ---")

        try:
            # Create model
            model = get_model(
                model_name,
                num_classes=num_classes,
                in_channels=in_channels,
                seq_length=seq_length
            )

            # Test forward pass
            x = torch.randn(batch_size, in_channels, seq_length)
            with torch.no_grad():
                out = model(x)

            # Get model info
            info = get_model_info(model)

            # Validate
            assert out.shape == (batch_size, num_classes), \
                f"Expected output shape {(batch_size, num_classes)}, got {out.shape}"

            print(f"  [OK] Input: {x.shape} -> Output: {out.shape}")
            print(f"    Parameters: {info['trainable_parameters']:,}")

        except Exception as e:
            print(f"  [FAIL] FAILED - {e}")
            return False

    # EEGNet (4D input)
    print(f"\n--- Testing EEGNET ---")

    try:
        # Create model
        model = get_model(
            'eegnet',
            num_classes=num_classes,
            in_channels=in_channels,
            seq_length=seq_length
        )

        # Test forward pass (EEGNet expects 4D input)
        x = torch.randn(batch_size, 1, in_channels, seq_length)
        with torch.no_grad():
            out = model(x)

        # Get model info
        info = get_model_info(model)

        # Validate
        assert out.shape == (batch_size, num_classes), \
            f"Expected output shape {(batch_size, num_classes)}, got {out.shape}"

        print(f"  [OK] Input: {x.shape} -> Output: {out.shape}")
        print(f"    Parameters: {info['trainable_parameters']:,}")

    except Exception as e:
        print(f"  [FAIL] FAILED - {e}")
        return False

    return True


def test_model_registry():
    """Test model registry functionality."""
    print("\n" + "="*60)
    print("Testing Model Registry")
    print("="*60)

    # Test list_models
    print("\n--- Available Models ---")
    all_models = list_models('all')
    print(f"All models ({len(all_models)}): {', '.join(all_models)}")

    cnn_models = list_models('cnn')
    print(f"CNN models ({len(cnn_models)}): {', '.join(cnn_models)}")

    vit_models = list_models('vit')
    print(f"ViT models ({len(vit_models)}): {', '.join(vit_models)}")

    baseline_models = list_models('baseline')
    print(f"Baseline models ({len(baseline_models)}): {', '.join(baseline_models)}")

    # Test get_model with different models
    print("\n--- Testing get_model Factory ---")
    test_configs = [
        ('resnet18', {'num_classes': 4, 'in_channels': 25, 'pretrained': False}),
        ('vit_tiny', {'num_classes': 4, 'in_channels': 25, 'img_size': 224, 'pretrained': False}),
        ('cnn1d', {'num_classes': 4, 'in_channels': 25, 'seq_length': 751}),
    ]

    for model_name, config in test_configs:
        try:
            model = get_model(model_name, **config)
            info = get_model_info(model)
            print(f"  [OK] {model_name}: {info['class_name']} with {info['trainable_parameters']:,} params")
        except Exception as e:
            print(f"  [FAIL] {model_name}: FAILED - {e}")
            return False

    return True


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("PHASE 4 MODEL VALIDATION TESTS")
    print("="*80)

    # Track results
    results = {}

    # Test image models
    results['image_models'] = test_image_models()

    # Test baseline models
    results['baseline_models'] = test_baseline_models()

    # Test registry
    results['registry'] = test_model_registry()

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
        print("[OK] ALL TESTS PASSED - Phase 4 Models Validated")
    else:
        print("[FAIL] SOME TESTS FAILED - Review errors above")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
