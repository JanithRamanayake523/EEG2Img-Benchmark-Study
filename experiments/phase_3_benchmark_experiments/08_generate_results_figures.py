"""
Phase 3.5: Results Visualization Script

Generates publication-quality figures:
1. Accuracy comparison (bar charts, box plots)
2. Statistical significance heatmaps
3. Confusion matrices
4. Transform comparison plots
5. Model comparison plots
6. Robustness curves

Usage:
    python experiments/phase_3_benchmark_experiments/08_generate_results_figures.py \\
        --results_csv results/phase3/aggregated_results.csv \\
        --output_dir results/phase3/figures
"""

import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'DejaVu Sans'


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Phase 3 result figures')
    parser.add_argument('--results_csv', type=str, required=True,
                        help='Path to aggregated results CSV')
    parser.add_argument('--output_dir', type=str, default='results/phase3/figures',
                        help='Output directory for figures')
    parser.add_argument('--format', type=str, default='png',
                        choices=['png', 'pdf', 'svg'],
                        help='Figure format')
    return parser.parse_args()


def plot_accuracy_comparison(df, output_dir, fmt='png'):
    """Plot accuracy comparison across transforms and models."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by transform and model
    pivot = df.pivot_table(values='mean_accuracy', index='transform', columns='model', aggfunc='mean')

    # Create grouped bar plot
    pivot.plot(kind='bar', ax=ax, width=0.8)

    ax.set_xlabel('Transform Method')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Classification Accuracy by Transform and Model')
    ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / f'accuracy_comparison.{fmt}', bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: accuracy_comparison.{fmt}")


def plot_boxplots(df, output_dir, fmt='png'):
    """Plot box plots for accuracy distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By transform
    transforms = sorted(df['transform'].unique())
    transform_data = [df[df['transform'] == t]['mean_accuracy'].values for t in transforms]

    bp1 = axes[0].boxplot(transform_data, labels=transforms, patch_artist=True)
    axes[0].set_xlabel('Transform Method')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy Distribution by Transform')
    axes[0].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # By model
    models = sorted(df['model'].unique())
    model_data = [df[df['model'] == m]['mean_accuracy'].values for m in models]

    bp2 = axes[1].boxplot(model_data, labels=models, patch_artist=True)
    axes[1].set_xlabel('Model Architecture')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Distribution by Model')
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Color boxes
    colors = sns.color_palette("husl", max(len(transforms), len(models)))
    for patch, color in zip(bp1['boxes'], colors[:len(transforms)]):
        patch.set_facecolor(color)
    for patch, color in zip(bp2['boxes'], colors[:len(models)]):
        patch.set_facecolor(color)

    plt.tight_layout()
    plt.savefig(output_dir / f'accuracy_boxplots.{fmt}', bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: accuracy_boxplots.{fmt}")


def plot_heatmap(df, output_dir, fmt='png'):
    """Plot heatmap of mean accuracies."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create pivot table
    pivot = df.pivot_table(values='mean_accuracy', index='transform', columns='model', aggfunc='mean')

    # Plot heatmap
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn', center=75,
                vmin=60, vmax=90, cbar_kws={'label': 'Accuracy (%)'}, ax=ax)

    ax.set_xlabel('Model Architecture')
    ax.set_ylabel('Transform Method')
    ax.set_title('Mean Classification Accuracy Heatmap')

    plt.tight_layout()
    plt.savefig(output_dir / f'accuracy_heatmap.{fmt}', bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: accuracy_heatmap.{fmt}")


def plot_statistical_significance(df, output_dir, fmt='png'):
    """Plot statistical significance matrix."""
    # Create method combinations
    df['method'] = df['transform'] + '_' + df['model']
    methods = sorted(df['method'].unique())
    n_methods = len(methods)

    # Compute p-values (Wilcoxon signed-rank test)
    p_matrix = np.ones((n_methods, n_methods))

    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i != j:
                acc1 = df[df['method'] == method1]['mean_accuracy'].values
                acc2 = df[df['method'] == method2]['mean_accuracy'].values

                min_len = min(len(acc1), len(acc2))
                if min_len > 1:
                    try:
                        _, p = stats.wilcoxon(acc1[:min_len], acc2[:min_len])
                        p_matrix[i, j] = p
                    except:
                        pass

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(p_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                xticklabels=methods, yticklabels=methods,
                vmin=0, vmax=0.05, center=0.025,
                cbar_kws={'label': 'p-value'}, ax=ax)

    ax.set_title('Statistical Significance Matrix (Wilcoxon Test)')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / f'statistical_significance.{fmt}', bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: statistical_significance.{fmt}")


def plot_top_performers(df, output_dir, n=10, fmt='png'):
    """Plot top N performing methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get top N
    top_df = df.nlargest(n, 'mean_accuracy').copy()
    top_df['method'] = top_df['transform'] + '\n' + top_df['model']

    # Plot
    bars = ax.barh(range(len(top_df)), top_df['mean_accuracy'].values)
    ax.set_yticks(range(len(top_df)))
    ax.set_yticklabels(top_df['method'].values)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title(f'Top {n} Performing Methods')
    ax.grid(True, alpha=0.3, axis='x')

    # Color bars
    colors = sns.color_palette("RdYlGn", len(top_df))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels
    for i, (idx, row) in enumerate(top_df.iterrows()):
        ax.text(row['mean_accuracy'] + 0.5, i,
                f"{row['mean_accuracy']:.2f}%",
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / f'top_{n}_performers.{fmt}', bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: top_{n}_performers.{fmt}")


def plot_summary_stats(df, output_dir, fmt='png'):
    """Plot summary statistics."""
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Overall distribution
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(df['mean_accuracy'], bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(df['mean_accuracy'].mean(), color='r', linestyle='--',
                label=f'Mean: {df["mean_accuracy"].mean():.2f}%')
    ax1.axvline(df['mean_accuracy'].median(), color='g', linestyle='--',
                label=f'Median: {df["mean_accuracy"].median():.2f}%')
    ax1.set_xlabel('Accuracy (%)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Overall Accuracy Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Mean by transform
    ax2 = fig.add_subplot(gs[1, 0])
    transform_means = df.groupby('transform')['mean_accuracy'].mean().sort_values(ascending=False)
    transform_means.plot(kind='barh', ax=ax2)
    ax2.set_xlabel('Mean Accuracy (%)')
    ax2.set_title('Mean Accuracy by Transform')
    ax2.grid(True, alpha=0.3, axis='x')

    # 3. Mean by model
    ax3 = fig.add_subplot(gs[1, 1])
    model_means = df.groupby('model')['mean_accuracy'].mean().sort_values(ascending=False)
    model_means.plot(kind='barh', ax=ax3)
    ax3.set_xlabel('Mean Accuracy (%)')
    ax3.set_title('Mean Accuracy by Model')
    ax3.grid(True, alpha=0.3, axis='x')

    # 4. Std by transform
    ax4 = fig.add_subplot(gs[2, 0])
    transform_std = df.groupby('transform')['std_accuracy'].mean().sort_values()
    transform_std.plot(kind='barh', ax=ax4, color='orange')
    ax4.set_xlabel('Mean Std Dev (%)')
    ax4.set_title('Variability by Transform')
    ax4.grid(True, alpha=0.3, axis='x')

    # 5. Std by model
    ax5 = fig.add_subplot(gs[2, 1])
    model_std = df.groupby('model')['std_accuracy'].mean().sort_values()
    model_std.plot(kind='barh', ax=ax5, color='orange')
    ax5.set_xlabel('Mean Std Dev (%)')
    ax5.set_title('Variability by Model')
    ax5.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Phase 3: Summary Statistics', fontsize=14, y=0.995)

    plt.savefig(output_dir / f'summary_statistics.{fmt}', bbox_inches='tight')
    plt.close()

    print(f"✓ Generated: summary_statistics.{fmt}")


def generate_all_figures(df, output_dir, fmt='png'):
    """Generate all figures."""
    print("\nGenerating figures...")
    print("="*60)

    plot_accuracy_comparison(df, output_dir, fmt)
    plot_boxplots(df, output_dir, fmt)
    plot_heatmap(df, output_dir, fmt)
    plot_statistical_significance(df, output_dir, fmt)
    plot_top_performers(df, output_dir, n=10, fmt=fmt)
    plot_summary_stats(df, output_dir, fmt)

    print("="*60)
    print(f"All figures generated in {output_dir}")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Phase 3: Results Visualization")
    print("="*60)
    print(f"Input: {args.results_csv}")
    print(f"Output: {output_dir}")
    print(f"Format: {args.format}")

    # Load results
    print("\nLoading results...")
    df = pd.read_csv(args.results_csv)
    print(f"Loaded {len(df)} experiments")

    # Generate figures
    generate_all_figures(df, output_dir, args.format)

    print("\n" + "="*60)
    print("Visualization complete!")
    print("="*60)


if __name__ == '__main__':
    main()
