"""
Master Script: Generate All Phase 2 Publication Figures
Runs all figure generation scripts in sequence
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import all figure generation modules
from experiments.figures_phase2 import (
    fig1_dataset_overview,
    fig2_preprocessing_pipeline,
    fig3_ica_components,
    fig4_epoch_analysis,
    fig5_normalization,
    fig6_summary_dashboard,
)

def generate_all_figures():
    """Generate all Phase 2 publication-ready figures"""

    print("=" * 80)
    print("PHASE 2: GENERATING ALL PUBLICATION FIGURES")
    print("=" * 80)
    print()

    figures = [
        ("Figure 1: Dataset Overview", fig1_dataset_overview.create_dataset_overview_figure),
        ("Figure 2: Preprocessing Pipeline", fig2_preprocessing_pipeline.create_preprocessing_pipeline_figure),
        ("Figure 3: ICA Components", fig3_ica_components.create_ica_components_figure),
        ("Figure 4: Epoch Analysis", fig4_epoch_analysis.create_epoch_analysis_figure),
        ("Figure 5: Normalization", fig5_normalization.create_normalization_figure),
        ("Figure 6: Summary Dashboard", fig6_summary_dashboard.create_summary_dashboard),
    ]

    total_figures = len(figures)
    successful = 0
    failed = []

    for idx, (name, func) in enumerate(figures, 1):
        print(f"\n[{idx}/{total_figures}] Generating {name}...")
        print("-" * 80)

        start_time = time.time()

        try:
            func()
            elapsed = time.time() - start_time
            print(f"[OK] {name} completed in {elapsed:.2f} seconds")
            successful += 1
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[FAIL] {name} FAILED after {elapsed:.2f} seconds")
            print(f"  Error: {str(e)}")
            failed.append((name, str(e)))

        print("-" * 80)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total figures: {total_figures}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed figures:")
        for name, error in failed:
            print(f"  - {name}")
            print(f"    Error: {error}")
    else:
        print("\n[OK] All figures generated successfully!")

    print("\nOutput directory: results/figures/phase2/")
    print("=" * 80)

    return successful, failed

if __name__ == '__main__':
    print("\n")
    print("=" * 80)
    print("PHASE 2: PUBLICATION FIGURE GENERATION")
    print("EEG Time-Series-to-Image Benchmark Study")
    print("=" * 80)
    print("\n")

    total_start = time.time()
    successful, failed = generate_all_figures()
    total_elapsed = time.time() - total_start

    print(f"\nTotal execution time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")

    if len(failed) == 0:
        print("\n[OK] SUCCESS: All Phase 2 figures are ready for publication!")
        sys.exit(0)
    else:
        print(f"\n[WARNING] {len(failed)} figure(s) failed to generate")
        sys.exit(1)
