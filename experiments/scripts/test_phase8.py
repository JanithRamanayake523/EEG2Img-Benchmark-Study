"""
Test script for Phase 8: Results Analysis & Reporting.

Tests result aggregation, analysis, and reporting functionality.
"""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation import ResultsAggregator


def test_results_aggregator():
    """Test results aggregation functionality."""
    print("\n" + "="*60)
    print("Testing Results Aggregation")
    print("="*60)

    # Create dummy results for testing
    print("\n--- Creating test results ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create experiment result structure
        exp_dir = tmpdir / "experiment_1"
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy results JSON
        results = {
            "config": {
                "name": "test_experiment_1",
                "dataset": {"name": "BCI-IV-2a"},
                "models": [
                    {"architecture": "resnet18", "name": "resnet18"}
                ],
                "training": {"epochs": 100, "batch_size": 32},
                "optimizer": {"learning_rate": 0.001},
                "augmentation": {"enabled": True}
            },
            "models": {
                "resnet18": {
                    "test_metrics": {
                        "accuracy": 0.85,
                        "f1": 0.84,
                        "auc": 0.92,
                        "kappa": 0.80,
                        "mcc": 0.81
                    }
                }
            }
        }

        result_file = exp_dir / "results.json"
        with open(result_file, 'w') as f:
            json.dump(results, f)

        print(f"  [OK] Created test results at {result_file}")

        # Test aggregator
        print("\n--- Testing ResultsAggregator initialization ---")
        try:
            aggregator = ResultsAggregator(tmpdir)
            print(f"  [OK] Aggregator initialized")
        except Exception as e:
            print(f"  [FAIL] Failed to initialize aggregator: {e}")
            return False

        print("\n--- Testing load_results ---")
        try:
            df = aggregator.load_results()
            print(f"  [OK] Loaded results: {len(df)} rows")
            print(f"      Columns: {len(df.columns)} columns")
        except Exception as e:
            print(f"  [FAIL] Failed to load results: {e}")
            return False

        print("\n--- Testing summarize_by_model ---")
        try:
            summary = aggregator.summarize_by_model()
            print(f"  [OK] Generated model summary: {len(summary)} models")
            for model, stats in summary.items():
                if 'accuracy' in stats:
                    acc = stats['accuracy']
                    print(f"      {model}: {acc['mean']:.4f} ± {acc['std']:.4f}")
        except Exception as e:
            print(f"  [FAIL] Failed to summarize by model: {e}")
            return False

        print("\n--- Testing get_top_models ---")
        try:
            top_models = aggregator.get_top_models('accuracy', n=3)
            print(f"  [OK] Got top models: {len(top_models)} models")
            for i, (model, acc) in enumerate(top_models.items(), 1):
                print(f"      {i}. {model}: {acc:.4f}")
        except Exception as e:
            print(f"  [FAIL] Failed to get top models: {e}")
            return False

        print("\n--- Testing create_comparison_table ---")
        try:
            table = aggregator.create_comparison_table('accuracy')
            print(f"  [OK] Created comparison table: {len(table)} rows")
        except Exception as e:
            print(f"  [FAIL] Failed to create comparison table: {e}")
            return False

        print("\n--- Testing export_csv ---")
        try:
            export_dir = tmpdir / "export"
            aggregator.export_csv(export_dir)
            csv_files = list(export_dir.glob("*.csv"))
            print(f"  [OK] Exported CSV: {len(csv_files)} files")
        except Exception as e:
            print(f"  [FAIL] Failed to export CSV: {e}")
            return False

        print("\n--- Testing export_json ---")
        try:
            export_dir = tmpdir / "export"
            aggregator.export_json(export_dir)
            json_files = list(export_dir.glob("*.json"))
            print(f"  [OK] Exported JSON: {len(json_files)} files")
        except Exception as e:
            print(f"  [FAIL] Failed to export JSON: {e}")
            return False

        print("\n--- Testing create_summary_report ---")
        try:
            export_dir = tmpdir / "export"
            aggregator.create_summary_report(export_dir / "report.txt")
            if (export_dir / "report.txt").exists():
                print(f"  [OK] Created summary report")
            else:
                print(f"  [FAIL] Report file not created")
                return False
        except Exception as e:
            print(f"  [FAIL] Failed to create report: {e}")
            return False

    return True


def test_notebook_imports():
    """Test that notebook dependencies are available."""
    print("\n" + "="*60)
    print("Testing Notebook Dependencies")
    print("="*60)

    try:
        print("\n--- Testing imports ---")
        import pandas as pd
        print(f"  [OK] pandas {pd.__version__}")

        import numpy as np
        print(f"  [OK] numpy {np.__version__}")

        import matplotlib.pyplot as plt
        print(f"  [OK] matplotlib {plt.matplotlib.__version__}")

        import seaborn as sns
        print(f"  [OK] seaborn {sns.__version__}")

        return True
    except ImportError as e:
        print(f"  [FAIL] Missing import: {e}")
        return False


def main():
    """Run all Phase 8 tests."""
    print("\n" + "="*80)
    print("PHASE 8 RESULTS ANALYSIS & REPORTING VALIDATION")
    print("="*80)

    results = {}

    # Run tests
    results['results_aggregator'] = test_results_aggregator()
    results['notebook_imports'] = test_notebook_imports()

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
        print("[OK] ALL TESTS PASSED - Phase 8 Results Analysis Validated")
    else:
        print("[FAIL] SOME TESTS FAILED - Review errors above")
    print("="*80 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
