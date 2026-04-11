"""
Figure 4: Epoch Extraction and Motor Imagery Class Analysis
Shows epoch extraction, class-specific patterns, and artifact rejection
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

def create_epoch_analysis_figure():
    """Generate epoch extraction and class analysis figure"""

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
    kurtosis_values = stats.kurtosis(source_data, axis=1)

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

    # Create figure
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3, height_ratios=[1, 1.2, 1, 1])

    # Class mapping
    class_mapping = {769: 'Left Hand', 770: 'Right Hand', 771: 'Feet', 772: 'Tongue',
                    7: 'Left Hand', 8: 'Right Hand', 9: 'Feet', 10: 'Tongue'}

    # Find channel for visualization
    available_channels = epochs.ch_names
    channel_to_plot = None
    for target in ['C3', 'Cz', 'C4']:
        for ch in available_channels:
            if target in ch:
                channel_to_plot = ch
                break
        if channel_to_plot:
            break

    if channel_to_plot is None:
        channel_to_plot = [ch for ch in available_channels if 'EEG' in ch][0]

    ch_idx = epochs.ch_names.index(channel_to_plot)

    # Plot motor imagery epochs for each class (Top 2 rows, 4 subplots)
    class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    plot_idx = 0
    for event_key, event_code in sorted(mi_event_ids.items(), key=lambda x: x[1]):
        if plot_idx < 4:
            row = plot_idx // 2
            col = plot_idx % 2
            ax = fig.add_subplot(gs[row, col])

            # Get epochs for this class
            event_mask = epochs.events[:, 2] == event_code
            class_epochs = epochs[event_mask]

            if len(class_epochs) > 0:
                # Get data
                data = class_epochs.get_data()[:, ch_idx, :] * 1e6

                # Plot individual trials (light)
                n_trials_to_plot = min(30, len(data))
                for trial in data[:n_trials_to_plot]:
                    ax.plot(epochs.times, trial, alpha=0.15, color='gray', linewidth=0.5)

                # Plot average (bold)
                mean_signal = data.mean(axis=0)
                std_signal = data.std(axis=0)
                ax.plot(epochs.times, mean_signal, linewidth=2.5, color=class_colors[plot_idx],
                       label=f'Average ({len(class_epochs)} trials)')

                # Add standard deviation band
                ax.fill_between(epochs.times, mean_signal - std_signal, mean_signal + std_signal,
                               alpha=0.25, color=class_colors[plot_idx])

                # Cue onset marker
                ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Cue onset')

                # Style
                class_name = class_mapping.get(event_code, f'Event {event_code}')
                ax.set_title(f'({chr(65+plot_idx)}) {class_name}', fontsize=13, fontweight='bold', loc='left')
                ax.set_ylabel(f'{channel_to_plot} (µV)', fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9, loc='upper right')

                # X-label only on bottom row
                if row == 1:
                    ax.set_xlabel('Time (seconds)', fontsize=11)

                plot_idx += 1

    # Artifact rejection analysis (Row 3, Left)
    ax5 = fig.add_subplot(gs[2, 0])

    # Histogram
    n, bins, patches = ax5.hist(peak_to_peak, bins=40, edgecolor='black', alpha=0.7, color='steelblue')

    # Color bad epochs
    for i, patch in enumerate(patches):
        if bins[i] >= reject_threshold:
            patch.set_facecolor('#d62728')
            patch.set_alpha(0.7)

    ax5.axvline(reject_threshold, color='red', linestyle='--', linewidth=2.5,
               label=f'Threshold ({reject_threshold} µV)')
    ax5.set_xlabel('Peak-to-Peak Amplitude (µV)', fontsize=11)
    ax5.set_ylabel('Number of Epochs', fontsize=11)
    ax5.set_title('(E) Artifact Rejection - Amplitude Distribution', fontsize=13, fontweight='bold', loc='left')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    rejection_rate = (bad_epochs.sum() / len(epochs)) * 100
    stats_text = f'Total epochs: {len(epochs)}\n'
    stats_text += f'Bad epochs: {bad_epochs.sum()} ({rejection_rate:.1f}%)\n'
    stats_text += f'Good epochs: {(~bad_epochs).sum()} ({100-rejection_rate:.1f}%)'

    ax5.text(0.98, 0.97, stats_text, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Box plot (Row 3, Right)
    ax6 = fig.add_subplot(gs[2, 1])

    bp = ax6.boxplot(peak_to_peak, vert=True, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    ax6.axhline(reject_threshold, color='red', linestyle='--', linewidth=2.5,
               label=f'Rejection threshold')
    ax6.set_ylabel('Peak-to-Peak Amplitude (µV)', fontsize=11)
    ax6.set_title('(F) Amplitude Distribution (Box Plot)', fontsize=13, fontweight='bold', loc='left')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_xticks([])

    # Class distribution after rejection (Row 4, Left)
    ax7 = fig.add_subplot(gs[3, 0])

    # Clean epochs
    epochs_clean = epochs.copy()
    if bad_epochs.sum() > 0:
        epochs_clean.drop(np.where(bad_epochs)[0], reason='Amplitude', verbose=False)

    # Count trials per class after rejection
    labels = epochs_clean.events[:, 2]
    unique_events = np.unique(labels)
    label_mapping = {event: idx for idx, event in enumerate(sorted(unique_events))}
    class_labels = np.array([label_mapping[label] for label in labels])

    class_counts_clean = [np.sum(class_labels == i) for i in range(4)]
    class_names = [class_mapping[code] for code in sorted(mi_event_ids.values())]

    bars = ax7.bar(class_names, class_counts_clean, color=class_colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, count in zip(bars, class_counts_clean):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax7.set_ylabel('Number of Epochs', fontsize=11)
    ax7.set_title('(G) Class Distribution After Artifact Rejection', fontsize=13, fontweight='bold', loc='left')
    ax7.grid(True, alpha=0.3, axis='y')
    ax7.set_ylim(0, max(class_counts_clean) * 1.15)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # Pie chart (Row 4, Right)
    ax8 = fig.add_subplot(gs[3, 1])

    wedges, texts, autotexts = ax8.pie(class_counts_clean, labels=class_names,
                                        autopct='%1.1f%%', startangle=90,
                                        colors=class_colors, explode=[0.05]*4)

    # Style percentages
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax8.set_title('(H) Class Balance', fontsize=13, fontweight='bold', loc='left')

    # Main title
    fig.suptitle('Figure 4: Motor Imagery Epoch Analysis and Artifact Rejection',
                fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    output_dir = Path('results/figures/phase2')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'fig4_epoch_analysis.png'

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"OK Saved: {output_file}")

    output_file_pdf = output_dir / 'fig4_epoch_analysis.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"OK Saved: {output_file_pdf}")

    plt.close()

    print(f"\nEpoch Analysis Summary:")
    print(f"  Total epochs extracted: {len(epochs)}")
    print(f"  Epochs rejected: {bad_epochs.sum()} ({rejection_rate:.1f}%)")
    print(f"  Clean epochs: {len(epochs_clean)}")
    print(f"  Class distribution: {class_counts_clean}")

if __name__ == '__main__':
    print("Generating Figure 4: Epoch Analysis...")
    create_epoch_analysis_figure()
    print("Done!")
