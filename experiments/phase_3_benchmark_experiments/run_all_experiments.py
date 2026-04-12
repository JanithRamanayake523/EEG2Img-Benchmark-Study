#!/usr/bin/env python3
"""
Phase 3: Complete Benchmark Experiments Runner

Runs all transform + model combinations across all subjects.
Provides progress tracking and error handling.

Usage:
    python experiments/phase_3_benchmark_experiments/run_all_experiments.py --device cuda:0
    python experiments/phase_3_benchmark_experiments/run_all_experiments.py --device cpu
"""

import subprocess
import sys
from pathlib import Path

# Configuration
TRANSFORMS = [
    "gaf_summation",
    "gaf_difference",
    "mtf_q8",
    "mtf_q16",
    "recurrence_plot",
    "spectrogram",
    "scalogram_morlet",
    "scalogram_mexican",
    "topographic",
]
MODELS = [
    "resnet18",
    "resnet50",
    "lightweight_cnn",
    "vit_base",
    "vit_small",
]

DEVICE = "cuda:0"  # Default device
SCRIPT_PATH = Path(__file__).parent / "05_run_experiments.py"


def run_experiment(transform, model, device):
    """Run a single experiment on all pooled subjects."""
    cmd = [
        "python",
        str(SCRIPT_PATH),
        "--transform", transform,
        "--model", model,
        "--subject", "all",  # Run on all pooled subjects
        "--device", device,
    ]

    try:
        result = subprocess.run(cmd, capture_output=False, timeout=7200)  # 2-hour timeout
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {transform} + {model}")
        return False
    except Exception as e:
        print(f"  [ERROR] {transform} + {model}: {str(e)}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run all Phase 3 benchmark experiments on pooled subjects')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:0, cuda:1, or cpu)')
    parser.add_argument('--transforms', type=str, nargs='+', default=None,
                       help='Specific transforms to run (default: all)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Specific models to run (default: all)')

    args = parser.parse_args()

    # Use specified or default values
    transforms = args.transforms if args.transforms else TRANSFORMS
    models = args.models if args.models else MODELS
    device = args.device

    # Calculate totals
    total = len(transforms) * len(models)
    completed = 0
    successful = 0
    failed = 0
    failed_experiments = []

    # Print header
    print("=" * 60)
    print("Phase 3: Complete Benchmark Experiments")
    print("Running on all subjects pooled together")
    print("=" * 60)
    print(f"Transforms: {len(transforms)}")
    print(f"Models: {len(models)}")
    print(f"Total combinations: {total}")
    print(f"Device: {device}")
    print("=" * 60)
    print()

    # Run all experiments on pooled subjects
    for transform in transforms:
        for model in models:
            completed += 1
            progress = (completed / total) * 100

            print(f"[{completed:3d}/{total}] ({progress:5.1f}%) ", end="", flush=True)
            print(f"{transform:20s} + {model:15s}...", end=" ", flush=True)

            if run_experiment(transform, model, device):
                print("[OK]")
                successful += 1
            else:
                print("[FAIL]")
                failed += 1
                failed_experiments.append((transform, model))

    # Print summary
    print()
    print("=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total:       {total}")
    print(f"Successful:  {successful}")
    print(f"Failed:      {failed}")
    print(f"Success Rate: {(successful/total)*100:.1f}%")

    if failed_experiments:
        print("\nFailed experiments:")
        for transform, model in failed_experiments:
            print(f"  - {transform} + {model}")

    print()
    print("Results saved to: results/phase3/experiments/")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
