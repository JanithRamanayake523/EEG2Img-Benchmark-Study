"""
Figure 1: Dataset Overview and Characteristics
Generates publication-ready visualization showing BCI IV-2a dataset structure
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import mne

# Set publication style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

def create_dataset_overview_figure():
    """Generate comprehensive dataset overview figure"""

    # Load raw data
    raw_data_path = Path('data/raw/bci_iv_2a')
    filename = raw_data_path / 'A01T.gdf'

    if not filename.exists():
        print(f"ERROR: Data file not found at {filename}")
        return

    print("Loading raw data...")
    raw = mne.io.read_raw_gdf(filename, preload=True, verbose=False)
    events, event_id = mne.events_from_annotations(raw, verbose=False)

    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # 1. Dataset Structure Diagram (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    structure_text = [
        'BCI Competition IV-2a Dataset',
        '',
        '• Subjects: 9 healthy participants',
        '• Sessions: 2 per subject (T, E)',
        '• Channels: 22 EEG + 3 auxiliary',
        '• Sampling Rate: 250 Hz',
        '• Classes: 4 motor imagery tasks',
        '  - Left Hand',
        '  - Right Hand',
        '  - Feet',
        '  - Tongue',
        '• Trials: 72 per class (288 total)',
    ]
    ax1.text(0.05, 0.95, '\n'.join(structure_text),
             transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax1.axis('off')
    ax1.set_title('(A) Dataset Characteristics', fontsize=13, fontweight='bold', loc='left')

    # 2. Motor Imagery Task Timeline (Top Right)
    ax2 = fig.add_subplot(gs[0, 1])
    # Draw trial timeline
    timeline_y = 0.5

    # Background
    ax2.barh(timeline_y, 8, left=0, height=0.3, color='lightgray', alpha=0.3, label='Trial Duration')

    # Phases
    ax2.barh(timeline_y, 2, left=0, height=0.3, color='yellow', alpha=0.5, label='Fixation (2s)')
    ax2.barh(timeline_y, 1.25, left=2, height=0.3, color='orange', alpha=0.5, label='Cue (1.25s)')
    ax2.barh(timeline_y, 4, left=3.25, height=0.3, color='green', alpha=0.5, label='Motor Imagery (4s)')
    ax2.barh(timeline_y, 0.75, left=7.25, height=0.3, color='lightgray', alpha=0.5, label='Rest (0.75s)')

    # Markers
    ax2.axvline(2, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(2, 0.9, 'Cue Onset', ha='center', fontsize=10, color='red')
    ax2.axvline(3.25, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(3.25, 0.9, 'MI Start', ha='center', fontsize=10, color='darkgreen')

    # Epoch extraction window
    ax2.axvspan(2.5, 5.5, alpha=0.2, color='blue', label='Epoch Window (3s)')

    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_yticks([])
    ax2.legend(loc='upper left', fontsize=8, ncol=2)
    ax2.set_title('(B) Motor Imagery Trial Timeline', fontsize=13, fontweight='bold', loc='left')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Channel Layout (Middle Left)
    ax3 = fig.add_subplot(gs[1, 0])
    # Get EEG channels only
    eeg_channels = [ch for ch in raw.ch_names if not ch.startswith('EOG')]

    # Create montage for electrode positions
    montage = mne.channels.make_standard_montage('standard_1020')

    # Get positions for available channels
    try:
        raw_eeg = raw.copy().pick_channels(eeg_channels)
        raw_eeg.set_montage(montage, on_missing='ignore')

        # Plot topomap with channel positions
        mne.viz.plot_sensors(raw_eeg.info, kind='topomap', show_names=True,
                            show=False, axes=ax3)
        ax3.set_title('(C) EEG Channel Layout (10-20 System)', fontsize=13, fontweight='bold', loc='left')
    except Exception as e:
        print(f"Could not plot channel layout: {e}")
        ax3.text(0.5, 0.5, 'Channel layout\nnot available',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('(C) EEG Channel Layout', fontsize=13, fontweight='bold', loc='left')

    # 4. Trial Distribution per Class (Middle Right)
    ax4 = fig.add_subplot(gs[1, 1])

    # Find motor imagery events
    mi_event_ids = {}
    for code in [769, 770, 771, 772]:
        if code in event_id.values():
            mi_event_ids.update({k: code for k, v in event_id.items() if v == code})

    if len(mi_event_ids) == 0:
        for code in [7, 8, 9, 10]:
            if code in event_id.values():
                mi_event_ids.update({k: code for k, v in event_id.items() if v == code})

    # Count trials per class
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    class_counts = []
    for event_code in sorted(mi_event_ids.values()):
        count = np.sum(events[:, 2] == event_code)
        class_counts.append(count)

    # Bar plot
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax4.bar(class_names, class_counts, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax4.set_ylabel('Number of Trials', fontsize=11)
    ax4.set_title('(D) Trial Distribution by Class', fontsize=13, fontweight='bold', loc='left')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, max(class_counts) * 1.15)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')

    # 5. Raw Signal Sample (Bottom - Full Width)
    ax5 = fig.add_subplot(gs[2, :])

    # Plot sample of raw data from key channels
    channels_to_plot = []
    for target in ['C3', 'Cz', 'C4']:
        for ch in raw.ch_names:
            if target in ch:
                channels_to_plot.append(ch)
                break

    if len(channels_to_plot) < 3:
        channels_to_plot = [ch for ch in raw.ch_names if 'EEG' in ch][:3]

    time_window = [10, 15]  # 5 second window
    colors_ch = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, ch_name in enumerate(channels_to_plot):
        ch_idx = raw.ch_names.index(ch_name)
        data, times = raw[ch_idx, :]
        time_mask = (times >= time_window[0]) & (times <= time_window[1])

        # Offset for visibility
        offset = idx * 50
        ax5.plot(times[time_mask], data[0, time_mask] * 1e6 + offset,
                linewidth=1, label=ch_name, color=colors_ch[idx], alpha=0.8)

    ax5.set_xlabel('Time (seconds)', fontsize=11)
    ax5.set_ylabel('Amplitude (µV, offset)', fontsize=11)
    ax5.set_title('(E) Representative Raw EEG Signals', fontsize=13, fontweight='bold', loc='left')
    ax5.legend(loc='upper right', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(time_window)

    # Main title
    fig.suptitle('Figure 1: BCI Competition IV-2a Dataset Overview',
                fontsize=16, fontweight='bold', y=0.98)

    # Save figure
    output_dir = Path('results/figures/phase2')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'fig1_dataset_overview.png'

    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"OK Saved: {output_file}")

    # Also save as PDF for publication
    output_file_pdf = output_dir / 'fig1_dataset_overview.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"OK Saved: {output_file_pdf}")

    plt.close()

if __name__ == '__main__':
    print("Generating Figure 1: Dataset Overview...")
    create_dataset_overview_figure()
    print("Done!")
