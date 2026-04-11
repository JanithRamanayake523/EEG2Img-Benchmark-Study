"""
Figure 2: Preprocessing Pipeline Effects
Shows before/after comparisons for filtering and ICA artifact removal
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
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def create_preprocessing_pipeline_figure():
    """Generate comprehensive preprocessing pipeline visualization"""

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

    # Detect artifacts
    ica_sources = ica.get_sources(raw_filtered)
    source_data = ica_sources.get_data()
    variances = np.var(source_data, axis=1)
    kurtosis_values = stats.kurtosis(source_data, axis=1)

    var_threshold = np.percentile(variances, 75)
    kurt_threshold = np.percentile(np.abs(kurtosis_values), 75)

    artifact_candidates = []
    for comp_idx in range(len(variances)):
        if variances[comp_idx] > var_threshold or np.abs(kurtosis_values[comp_idx]) > kurt_threshold:
            artifact_candidates.append(comp_idx)

    ica.exclude = sorted(artifact_candidates, key=lambda x: variances[x], reverse=True)[:2]

    # Apply ICA
    raw_ica = raw_filtered.copy()
    ica.apply(raw_ica, verbose=False)

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 1, 1.2])

    # Find channels to plot
    available_channels = raw_eeg.ch_names
    channels_to_plot = []
    for target in ['C3', 'Cz', 'C4']:
        for ch in available_channels:
            if target in ch:
                channels_to_plot.append(ch)
                break

    if len(channels_to_plot) < 3:
        channels_to_plot = [ch for ch in available_channels if 'EEG' in ch][:3]

    # 1. Raw vs Filtered Signal (Top Row)
    time_window = [50, 55]  # 5 second window

    # Left: Raw Signal
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, ch_name in enumerate(channels_to_plot):
        ch_idx = raw_eeg.ch_names.index(ch_name)
        data, times = raw_eeg[ch_idx, :]
        time_mask = (times >= time_window[0]) & (times <= time_window[1])

        offset = idx * 60
        ax1.plot(times[time_mask], data[0, time_mask] * 1e6 + offset,
                linewidth=0.8, label=ch_name, alpha=0.8)

    ax1.set_ylabel('Amplitude (µV, offset)', fontsize=11)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_title('(A) Raw Signal', fontsize=13, fontweight='bold', loc='left')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(time_window)

    # Right: Filtered Signal
    ax2 = fig.add_subplot(gs[0, 1])
    for idx, ch_name in enumerate(channels_to_plot):
        ch_idx = raw_filtered.ch_names.index(ch_name)
        data, times = raw_filtered[ch_idx, :]
        time_mask = (times >= time_window[0]) & (times <= time_window[1])

        offset = idx * 60
        ax2.plot(times[time_mask], data[0, time_mask] * 1e6 + offset,
                linewidth=0.8, label=ch_name, alpha=0.8)

    ax2.set_ylabel('Amplitude (µV, offset)', fontsize=11)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_title('(B) Filtered Signal (0.5-40 Hz + Notch 50 Hz)', fontsize=13, fontweight='bold', loc='left')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(time_window)

    # 2. Power Spectral Density Comparison (Middle Row)
    # Find a good channel for PSD
    ch_name = channels_to_plot[1] if len(channels_to_plot) > 1 else channels_to_plot[0]

    # Left: Raw PSD
    ax3 = fig.add_subplot(gs[1, 0])
    raw_eeg.compute_psd(fmax=100, picks=[ch_name], verbose=False).plot(
        picks=[ch_name], axes=ax3, show=False, average=True, amplitude=False
    )
    ax3.axvline(50, color='red', linestyle='--', linewidth=2, label='50 Hz (power line)', alpha=0.7)
    ax3.set_title(f'(C) Raw Signal PSD - {ch_name}', fontsize=13, fontweight='bold', loc='left')
    ax3.set_ylim([-40, 40])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Right: Filtered PSD
    ax4 = fig.add_subplot(gs[1, 1])
    raw_filtered.compute_psd(fmax=100, picks=[ch_name], verbose=False).plot(
        picks=[ch_name], axes=ax4, show=False, average=True, amplitude=False
    )
    ax4.axvline(50, color='red', linestyle='--', linewidth=2, alpha=0.7, label='50 Hz removed')
    ax4.axvline(0.5, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Low cutoff')
    ax4.axvline(40, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='High cutoff')
    ax4.set_title(f'(D) Filtered Signal PSD - {ch_name}', fontsize=13, fontweight='bold', loc='left')
    ax4.set_ylim([-40, 40])
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 3. ICA Before/After Comparison (Bottom Row - Full Width)
    time_window_ica = [100, 110]

    # Left: Before ICA
    ax5 = fig.add_subplot(gs[2, 0])
    for idx, ch_name in enumerate(channels_to_plot):
        ch_idx = raw_filtered.ch_names.index(ch_name)
        data, times = raw_filtered[ch_idx, :]
        time_mask = (times >= time_window_ica[0]) & (times <= time_window_ica[1])

        offset = idx * 60
        ax5.plot(times[time_mask], data[0, time_mask] * 1e6 + offset,
                linewidth=0.8, label=ch_name, alpha=0.8, color='orange')

    ax5.set_ylabel('Amplitude (µV, offset)', fontsize=11)
    ax5.set_xlabel('Time (seconds)', fontsize=11)
    ax5.set_title('(E) Before ICA (Artifacts Present)', fontsize=13, fontweight='bold', loc='left')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(time_window_ica)

    # Right: After ICA
    ax6 = fig.add_subplot(gs[2, 1])
    for idx, ch_name in enumerate(channels_to_plot):
        ch_idx = raw_ica.ch_names.index(ch_name)
        data, times = raw_ica[ch_idx, :]
        time_mask = (times >= time_window_ica[0]) & (times <= time_window_ica[1])

        offset = idx * 60
        ax6.plot(times[time_mask], data[0, time_mask] * 1e6 + offset,
                linewidth=0.8, label=ch_name, alpha=0.8, color='green')

    ax6.set_ylabel('Amplitude (µV, offset)', fontsize=11)
    ax6.set_xlabel('Time (seconds)', fontsize=11)
    ax6.set_title(f'(F) After ICA ({len(ica.exclude)} Components Removed)', fontsize=13, fontweight='bold', loc='left')
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(time_window_ica)

    # Main title
    fig.suptitle('Figure 2: Preprocessing Pipeline - Signal Transformation Effects',
                fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    output_dir = Path('results/figures/phase2')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'fig2_preprocessing_pipeline.png'

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"OK Saved: {output_file}")

    output_file_pdf = output_dir / 'fig2_preprocessing_pipeline.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"OK Saved: {output_file_pdf}")

    plt.close()

if __name__ == '__main__':
    print("Generating Figure 2: Preprocessing Pipeline...")
    create_preprocessing_pipeline_figure()
    print("Done!")
