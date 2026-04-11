"""
Figure 3: ICA Component Analysis
Shows ICA components and artifact identification
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mne
from mne.preprocessing import ICA
from scipy import stats

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def create_ica_components_figure():
    """Generate ICA component analysis figure"""

    # Load raw data
    raw_data_path = Path('data/raw/bci_iv_2a')
    filename = raw_data_path / 'A01T.gdf'

    if not filename.exists():
        print(f"ERROR: Data file not found at {filename}")
        return

    print("Loading and preprocessing data...")
    raw = mne.io.read_raw_gdf(filename, preload=True, verbose=False)

    # Get EEG channels only
    eeg_channels = [ch for ch in raw.ch_names if not ch.startswith('EOG')]
    raw_eeg = raw.copy().pick_channels(eeg_channels)

    # Apply filtering
    raw_filtered = raw_eeg.copy()
    raw_filtered.filter(l_freq=0.5, h_freq=40.0, verbose=False)
    raw_filtered.notch_filter(freqs=50, verbose=False)

    # Apply ICA
    print("Fitting ICA...")
    ica = ICA(n_components=20, random_state=42, max_iter=500, verbose=False)
    ica.fit(raw_filtered, verbose=False)

    # Get ICA sources
    ica_sources = ica.get_sources(raw_filtered)
    source_data = ica_sources.get_data()

    # Calculate variance and kurtosis for each component
    variances = np.var(source_data, axis=1)
    kurtosis_values = stats.kurtosis(source_data, axis=1)

    # Detect artifacts
    var_threshold = np.percentile(variances, 75)
    kurt_threshold = np.percentile(np.abs(kurtosis_values), 75)

    artifact_candidates = []
    for comp_idx in range(len(variances)):
        if variances[comp_idx] > var_threshold or np.abs(kurtosis_values[comp_idx]) > kurt_threshold:
            artifact_candidates.append(comp_idx)

    ica.exclude = sorted(artifact_candidates, key=lambda x: variances[x], reverse=True)[:2]

    # Create figure
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.35)

    # Plot first 10 ICA components (Top 3 rows)
    time_window = [0, 30]  # 30 second window
    time_mask = (ica_sources.times >= time_window[0]) & (ica_sources.times <= time_window[1])

    component_colors = []
    for comp_idx in range(10):
        row = comp_idx // 3
        col = comp_idx % 3
        ax = fig.add_subplot(gs[row, col])

        # Determine if artifact
        is_artifact = comp_idx in ica.exclude
        color = '#d62728' if is_artifact else '#1f77b4'  # Red for artifacts, blue for brain
        component_colors.append(color)

        # Plot component time series
        data = source_data[comp_idx, time_mask] * 1e6  # Convert to µV
        ax.plot(ica_sources.times[time_mask], data, linewidth=0.6, color=color, alpha=0.8)

        # Style
        title_str = f'IC {comp_idx}'
        if is_artifact:
            title_str += ' ⚠ ARTIFACT'
            ax.set_facecolor('#ffe6e6')  # Light red background

        ax.set_title(title_str, fontsize=11, fontweight='bold' if is_artifact else 'normal')
        ax.set_ylabel('Amplitude (µV)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_window)

        # Add statistics
        var = variances[comp_idx]
        kurt = kurtosis_values[comp_idx]
        ax.text(0.02, 0.98, f'Var: {var:.2f}\nKurt: {kurt:.1f}',
               transform=ax.transAxes, fontsize=8,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, pad=0.3))

        # X-label only on bottom row
        if row == 2:
            ax.set_xlabel('Time (seconds)', fontsize=9)
        else:
            ax.set_xlabel('')

    # Bottom row: Component statistics visualization
    # Left: Variance distribution
    ax_var = fig.add_subplot(gs[3, 0])
    bars = ax_var.bar(range(10), variances[:10], color=component_colors[:10], alpha=0.7, edgecolor='black')
    ax_var.axhline(var_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({var_threshold:.2f})')
    ax_var.set_xlabel('Component Index', fontsize=10)
    ax_var.set_ylabel('Variance', fontsize=10)
    ax_var.set_title('(K) Component Variance', fontsize=11, fontweight='bold', loc='left')
    ax_var.legend(fontsize=8)
    ax_var.grid(True, alpha=0.3, axis='y')
    ax_var.set_xticks(range(10))

    # Middle: Kurtosis distribution
    ax_kurt = fig.add_subplot(gs[3, 1])
    bars = ax_kurt.bar(range(10), np.abs(kurtosis_values[:10]), color=component_colors[:10], alpha=0.7, edgecolor='black')
    ax_kurt.axhline(kurt_threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({kurt_threshold:.1f})')
    ax_kurt.set_xlabel('Component Index', fontsize=10)
    ax_kurt.set_ylabel('|Kurtosis|', fontsize=10)
    ax_kurt.set_title('(L) Component Kurtosis (Absolute)', fontsize=11, fontweight='bold', loc='left')
    ax_kurt.legend(fontsize=8)
    ax_kurt.grid(True, alpha=0.3, axis='y')
    ax_kurt.set_xticks(range(10))

    # Right: Explained variance
    ax_expl = fig.add_subplot(gs[3, 2])
    explained_var = ica.pca_explained_variance_[:10]
    cumulative_var = np.cumsum(explained_var)

    ax_expl.bar(range(10), explained_var, alpha=0.6, label='Individual', color='steelblue', edgecolor='black')
    ax_expl.plot(range(10), cumulative_var, 'ro-', linewidth=2, markersize=6, label='Cumulative')
    ax_expl.set_xlabel('Component Index', fontsize=10)
    ax_expl.set_ylabel('Explained Variance (%)', fontsize=10)
    ax_expl.set_title('(M) PCA Explained Variance', fontsize=11, fontweight='bold', loc='left')
    ax_expl.legend(fontsize=8)
    ax_expl.grid(True, alpha=0.3, axis='y')
    ax_expl.set_xticks(range(10))

    # Main title
    fig.suptitle('Figure 3: Independent Component Analysis (ICA) - Component Characterization',
                fontsize=16, fontweight='bold', y=0.995)

    # Add legend explanation
    fig.text(0.5, 0.005, 'Red components = Artifacts (high variance/kurtosis) | Blue components = Brain signals',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Save figure
    output_dir = Path('results/figures/phase2')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'fig3_ica_components.png'

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"OK Saved: {output_file}")

    output_file_pdf = output_dir / 'fig3_ica_components.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"OK Saved: {output_file_pdf}")

    plt.close()

    print(f"\nICA Summary:")
    print(f"  Total components analyzed: 10 (out of {ica.n_components_})")
    print(f"  Artifact components identified: {ica.exclude}")
    print(f"  Variance threshold: {var_threshold:.2f}")
    print(f"  Kurtosis threshold: {kurt_threshold:.2f}")

if __name__ == '__main__':
    print("Generating Figure 3: ICA Components...")
    create_ica_components_figure()
    print("Done!")
