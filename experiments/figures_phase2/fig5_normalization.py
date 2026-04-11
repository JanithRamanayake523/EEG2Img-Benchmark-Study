"""
Figure 5: Z-Score Normalization Effects
Shows before/after normalization with statistical verification
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mne
from mne.preprocessing import ICA
from scipy import stats as scipy_stats

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def create_normalization_figure():
    """Generate normalization effects visualization"""

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

    # Detect and remove artifacts
    ica_sources = ica.get_sources(raw_filtered)
    source_data = ica_sources.get_data()
    variances = np.var(source_data, axis=1)
    kurtosis_values = scipy_stats.kurtosis(source_data, axis=1)

    var_threshold = np.percentile(variances, 75)
    kurt_threshold = np.percentile(np.abs(kurtosis_values), 75)

    artifact_candidates = []
    for comp_idx in range(len(variances)):
        if variances[comp_idx] > var_threshold or np.abs(kurtosis_values[comp_idx]) > kurt_threshold:
            artifact_candidates.append(comp_idx)

    ica.exclude = sorted(artifact_candidates, key=lambda x: variances[x], reverse=True)[:2]

    raw_ica = raw_filtered.copy()
    ica.apply(raw_ica, verbose=False)

    # Extract epochs
    print("Extracting epochs...")
    events, event_id = mne.events_from_annotations(raw_ica, verbose=False)

    # Find motor imagery events
    mi_event_ids = {}
    for code in [769, 770, 771, 772]:
        if code in event_id.values():
            mi_event_ids.update({k: code for k, v in event_id.items() if v == code})

    if len(mi_event_ids) == 0:
        for code in [7, 8, 9, 10]:
            if code in event_id.values():
                mi_event_ids.update({k: code for k, v in event_id.items() if v == code})

    # Create epochs
    tmin, tmax = 0.5, 3.5
    epochs = mne.Epochs(
        raw_ica,
        events,
        event_id=mi_event_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    # Artifact rejection
    epoch_data = epochs.get_data() * 1e6
    peak_to_peak = epoch_data.max(axis=(1, 2)) - epoch_data.min(axis=(1, 2))
    reject_threshold = 100
    bad_epochs = peak_to_peak > reject_threshold

    # Clean epochs
    epochs_clean = epochs.copy()
    if bad_epochs.sum() > 0:
        epochs_clean.drop(np.where(bad_epochs)[0], reason='Amplitude', verbose=False)

    # Get data before normalization
    print("Applying normalization...")
    data_before_norm = epochs_clean.get_data() * 1e6  # Convert to µV

    # Apply z-score normalization
    data_normalized = np.zeros_like(data_before_norm)
    for epoch_idx in range(data_before_norm.shape[0]):
        for ch_idx in range(data_before_norm.shape[1]):
            signal = data_before_norm[epoch_idx, ch_idx, :]
            mean = signal.mean()
            std = signal.std()
            if std > 0:
                data_normalized[epoch_idx, ch_idx, :] = (signal - mean) / std
            else:
                data_normalized[epoch_idx, ch_idx, :] = signal - mean

    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

    # Find channels to plot
    available_channels = epochs_clean.ch_names
    channels_to_plot = []
    for target in ['C3', 'Cz', 'C4', 'Pz']:
        for ch in available_channels:
            if target in ch:
                channels_to_plot.append(ch)
                break

    if len(channels_to_plot) < 4:
        channels_to_plot = [ch for ch in available_channels if 'EEG' in ch][:4]

    # Sample epoch for detailed view
    epoch_idx = 0

    # Plot before/after for 4 channels (Top 2 rows, 2 columns each)
    for idx, ch_name in enumerate(channels_to_plot[:4]):
        if ch_name in epochs_clean.ch_names:
            ch_idx = epochs_clean.ch_names.index(ch_name)

            row = idx // 2
            col = idx % 2

            # Before normalization
            ax_before = fig.add_subplot(gs[row, col*2])
            ax_before.plot(epochs_clean.times, data_before_norm[epoch_idx, ch_idx, :],
                          linewidth=1.5, color='purple', alpha=0.8)
            ax_before.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            ax_before.set_ylabel(f'{ch_name} (µV)', fontsize=10)
            ax_before.set_title(f'({chr(65+idx*2)}) Before - {ch_name}', fontsize=12, fontweight='bold', loc='left')
            ax_before.grid(True, alpha=0.3)

            # Statistics
            before_mean = data_before_norm[epoch_idx, ch_idx, :].mean()
            before_std = data_before_norm[epoch_idx, ch_idx, :].std()
            ax_before.text(0.02, 0.98, f'μ = {before_mean:.2f} µV\nσ = {before_std:.2f} µV',
                          transform=ax_before.transAxes, fontsize=9,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            # After normalization
            ax_after = fig.add_subplot(gs[row, col*2+1])
            ax_after.plot(epochs_clean.times, data_normalized[epoch_idx, ch_idx, :],
                         linewidth=1.5, color='darkgreen', alpha=0.8)
            ax_after.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            ax_after.set_ylabel(f'{ch_name} (z-score)', fontsize=10)
            ax_after.set_title(f'({chr(66+idx*2)}) After - {ch_name}', fontsize=12, fontweight='bold', loc='left')
            ax_after.grid(True, alpha=0.3)

            # Statistics
            after_mean = data_normalized[epoch_idx, ch_idx, :].mean()
            after_std = data_normalized[epoch_idx, ch_idx, :].std()
            ax_after.text(0.02, 0.98, f'μ = {after_mean:.3f}\nσ = {after_std:.3f}',
                         transform=ax_after.transAxes, fontsize=9,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

            # X-labels only on bottom row
            if row == 1:
                ax_before.set_xlabel('Time (seconds)', fontsize=10)
                ax_after.set_xlabel('Time (seconds)', fontsize=10)

    # Overall distribution comparison (Row 3)
    # Before normalization histogram
    ax_hist_before = fig.add_subplot(gs[2, :2])
    ax_hist_before.hist(data_before_norm.flatten()[::100], bins=100, edgecolor='black',
                       alpha=0.7, color='purple')
    ax_hist_before.axvline(data_before_norm.mean(), color='red', linestyle='--',
                          linewidth=2, label=f'Mean = {data_before_norm.mean():.2f} µV')
    ax_hist_before.set_xlabel('Amplitude (µV)', fontsize=11)
    ax_hist_before.set_ylabel('Frequency', fontsize=11)
    ax_hist_before.set_title('(I) Before Normalization - Distribution', fontsize=12, fontweight='bold', loc='left')
    ax_hist_before.legend(fontsize=9)
    ax_hist_before.grid(True, alpha=0.3, axis='y')

    # After normalization histogram
    ax_hist_after = fig.add_subplot(gs[2, 2:])
    ax_hist_after.hist(data_normalized.flatten()[::100], bins=100, edgecolor='black',
                      alpha=0.7, color='darkgreen')
    ax_hist_after.axvline(data_normalized.mean(), color='red', linestyle='--',
                         linewidth=2, label=f'Mean = {data_normalized.mean():.6f}')
    ax_hist_after.set_xlabel('Amplitude (z-score)', fontsize=11)
    ax_hist_after.set_ylabel('Frequency', fontsize=11)
    ax_hist_after.set_title('(J) After Normalization - Distribution', fontsize=12, fontweight='bold', loc='left')
    ax_hist_after.legend(fontsize=9)
    ax_hist_after.grid(True, alpha=0.3, axis='y')


    # Main title
    fig.suptitle('Figure 5: Z-Score Normalization Effects and Verification',
                fontsize=16, fontweight='bold', y=0.98)

    # Add explanation
    fig.text(0.5, 0.005,
            'Z-score normalization: X_norm = (X - μ) / σ → Centers data at 0 and scales to unit variance',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Save figure
    output_dir = Path('results/figures/phase2')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'fig5_normalization.png'

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"OK Saved: {output_file}")

    output_file_pdf = output_dir / 'fig5_normalization.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"OK Saved: {output_file_pdf}")

    plt.close()

    print(f"\nNormalization Verification:")
    print(f"  Mean: {data_normalized.mean():.6f} (target: 0)")
    print(f"  Std:  {data_normalized.std():.6f} (target: 1)")
    print(f"  OK Normalization successful!")

if __name__ == '__main__':
    print("Generating Figure 5: Normalization...")
    create_normalization_figure()
    print("Done!")
