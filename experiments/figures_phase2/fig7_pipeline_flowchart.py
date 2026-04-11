"""
Figure 7: Preprocessing Pipeline Flowchart
Publication-quality flowchart showing the complete preprocessing workflow
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set publication style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11

def create_pipeline_flowchart():
    """Generate preprocessing pipeline flowchart"""

    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis('off')

    # Colors
    input_color = '#E3F2FD'  # Light blue
    process_color = '#FFF3E0'  # Light orange
    output_color = '#E8F5E9'  # Light green
    decision_color = '#FCE4EC'  # Light pink
    arrow_color = '#424242'

    # Box dimensions
    box_width = 5
    box_height = 0.8
    x_center = 7

    # Helper function to draw boxes
    def draw_box(x, y, width, height, color, text, style='round'):
        if style == 'round':
            box = mpatches.FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                facecolor=color, edgecolor='black', linewidth=1.5
            )
        elif style == 'diamond':
            # Draw diamond shape
            diamond = plt.Polygon(
                [(x, y + height/2), (x + width/3, y), (x, y - height/2), (x - width/3, y)],
                facecolor=color, edgecolor='black', linewidth=1.5
            )
            ax.add_patch(diamond)
            ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold')
            return
        else:
            box = mpatches.Rectangle(
                (x - width/2, y - height/2), width, height,
                facecolor=color, edgecolor='black', linewidth=1.5
            )
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=10, fontweight='bold', wrap=True)

    # Helper function to draw arrows
    def draw_arrow(x1, y1, x2, y2, text=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
        if text:
            mid_x = (x1 + x2) / 2 + 0.3
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y, text, fontsize=8, style='italic', color='gray')

    # Title
    ax.text(x_center, 15.5, 'Phase 2: EEG Data Preprocessing Pipeline',
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(x_center, 15.0, 'BCI Competition IV-2a Dataset',
            ha='center', va='center', fontsize=12, style='italic', color='gray')

    # Pipeline boxes (top to bottom)
    y_positions = [14, 12.8, 11.6, 10.4, 9.2, 8.0, 6.8, 5.6, 4.4, 3.2, 2.0]

    # Step 1: Input
    draw_box(x_center, y_positions[0], box_width, box_height, input_color,
             'Raw GDF File (BCI IV-2a)')
    ax.text(x_center + 3, y_positions[0], '25 channels, 250 Hz\n672,528 samples',
            fontsize=8, color='gray', va='center')

    draw_arrow(x_center, y_positions[0] - box_height/2, x_center, y_positions[1] + box_height/2)

    # Step 2: Load Data
    draw_box(x_center, y_positions[1], box_width, box_height, process_color,
             'Load Raw EEG Data (MNE)')

    draw_arrow(x_center, y_positions[1] - box_height/2, x_center, y_positions[2] + box_height/2)

    # Step 3: Channel Selection
    draw_box(x_center, y_positions[2], box_width, box_height, process_color,
             'Select EEG Channels')
    ax.text(x_center + 3, y_positions[2], '22 EEG channels\n(exclude EOG)',
            fontsize=8, color='gray', va='center')

    draw_arrow(x_center, y_positions[2] - box_height/2, x_center, y_positions[3] + box_height/2)

    # Step 4: Band-pass Filter
    draw_box(x_center, y_positions[3], box_width, box_height, process_color,
             'Band-pass Filter (0.5-40 Hz)')
    ax.text(x_center + 3, y_positions[3], 'Remove DC drift\n& high-freq noise',
            fontsize=8, color='gray', va='center')

    draw_arrow(x_center, y_positions[3] - box_height/2, x_center, y_positions[4] + box_height/2)

    # Step 5: Notch Filter
    draw_box(x_center, y_positions[4], box_width, box_height, process_color,
             'Notch Filter (50 Hz)')
    ax.text(x_center + 3, y_positions[4], 'Remove power\nline interference',
            fontsize=8, color='gray', va='center')

    draw_arrow(x_center, y_positions[4] - box_height/2, x_center, y_positions[5] + box_height/2)

    # Step 6: ICA
    draw_box(x_center, y_positions[5], box_width, box_height, process_color,
             'ICA Decomposition (20 comp.)')
    ax.text(x_center + 3, y_positions[5], 'FastICA algorithm\nmax 500 iterations',
            fontsize=8, color='gray', va='center')

    draw_arrow(x_center, y_positions[5] - box_height/2, x_center, y_positions[6] + box_height/2)

    # Step 7: Artifact Detection
    draw_box(x_center, y_positions[6], box_width, box_height, decision_color,
             'Artifact Detection')
    ax.text(x_center + 3, y_positions[6], 'Variance + Kurtosis\nthreshold (75th %ile)',
            fontsize=8, color='gray', va='center')

    draw_arrow(x_center, y_positions[6] - box_height/2, x_center, y_positions[7] + box_height/2,
               '2 components removed')

    # Step 8: Epoch Extraction
    draw_box(x_center, y_positions[7], box_width, box_height, process_color,
             'Epoch Extraction (0.5-3.5s)')
    ax.text(x_center + 3, y_positions[7], '288 trials extracted\n4 MI classes',
            fontsize=8, color='gray', va='center')

    draw_arrow(x_center, y_positions[7] - box_height/2, x_center, y_positions[8] + box_height/2)

    # Step 9: Artifact Rejection
    draw_box(x_center, y_positions[8], box_width, box_height, decision_color,
             'Amplitude Rejection (>100 uV)')
    ax.text(x_center + 3, y_positions[8], '3 epochs rejected\n(1.0% rejection rate)',
            fontsize=8, color='gray', va='center')

    draw_arrow(x_center, y_positions[8] - box_height/2, x_center, y_positions[9] + box_height/2,
               '285 clean epochs')

    # Step 10: Normalization
    draw_box(x_center, y_positions[9], box_width, box_height, process_color,
             'Z-Score Normalization')
    ax.text(x_center + 3, y_positions[9], 'Per-channel, per-epoch\nmean=0, std=1',
            fontsize=8, color='gray', va='center')

    draw_arrow(x_center, y_positions[9] - box_height/2, x_center, y_positions[10] + box_height/2)

    # Step 11: Output
    draw_box(x_center, y_positions[10], box_width, box_height, output_color,
             'Preprocessed Data (HDF5)')
    ax.text(x_center + 3, y_positions[10], 'Shape: (285, 22, 751)\nReady for Phase 3',
            fontsize=8, color='gray', va='center')

    # Add legend
    legend_y = 1.0
    legend_x = 1.5
    legend_items = [
        (input_color, 'Input/Output'),
        (process_color, 'Processing Step'),
        (decision_color, 'Quality Control'),
    ]

    for i, (color, label) in enumerate(legend_items):
        box = mpatches.Rectangle((legend_x + i*3.5, legend_y - 0.15), 0.4, 0.3,
                                  facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(legend_x + i*3.5 + 0.6, legend_y, label, fontsize=9, va='center')

    # Add side panel with key parameters
    param_x = 11.5
    param_y = 10.5
    params_text = """Key Parameters
-----------------
Sampling Rate: 250 Hz
Channels: 22 EEG
Epoch Duration: 3.0 s
Filter Band: 0.5-40 Hz
Notch: 50 Hz
ICA Components: 20
Rejection: 100 uV
Classes: 4 MI tasks"""

    ax.text(param_x, param_y, params_text, fontsize=9, fontfamily='monospace',
            va='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8))

    # Save figure
    output_dir = Path('results/figures/phase2')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'fig7_pipeline_flowchart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"OK Saved: {output_file}")

    output_file_pdf = output_dir / 'fig7_pipeline_flowchart.pdf'
    plt.savefig(output_file_pdf, bbox_inches='tight', facecolor='white')
    print(f"OK Saved: {output_file_pdf}")

    plt.close()

if __name__ == '__main__':
    print("Generating Figure 7: Pipeline Flowchart...")
    create_pipeline_flowchart()
    print("Done!")
