"""
Phase 2 Publication Figure Generation Package

This package contains scripts to generate publication-ready figures for Phase 2
(Data Preprocessing) of the EEG Time-Series-to-Image Benchmark Study.

Figures:
- Figure 1: Dataset Overview and Characteristics
- Figure 2: Preprocessing Pipeline Effects
- Figure 3: ICA Component Analysis
- Figure 4: Epoch Analysis and Artifact Rejection
- Figure 5: Z-Score Normalization Effects
- Figure 6: Summary Dashboard

Usage:
    # Generate all figures
    python experiments/figures_phase2/generate_all_figures.py

    # Generate individual figures
    python experiments/figures_phase2/fig1_dataset_overview.py
    python experiments/figures_phase2/fig2_preprocessing_pipeline.py
    # ... etc

Output:
    All figures are saved to: results/figures/phase2/
    - PNG format (300 DPI)
    - PDF format (vector graphics for publication)
"""

__version__ = '1.0.0'
__author__ = 'EEG2Img Benchmark Study'

# Package metadata
FIGURES = {
    'fig1': 'Dataset Overview and Characteristics',
    'fig2': 'Preprocessing Pipeline Effects',
    'fig3': 'ICA Component Analysis',
    'fig4': 'Epoch Analysis and Artifact Rejection',
    'fig5': 'Z-Score Normalization Effects',
    'fig6': 'Summary Dashboard',
}

OUTPUT_DIR = 'results/figures/phase2'
