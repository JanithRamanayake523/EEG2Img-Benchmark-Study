"""
Phase 3.5: Results Aggregation Script

Collects all experiment results from Phase 3:
1. Scan results directory for all JSON files
2. Parse and aggregate metrics
3. Create master DataFrame
4. Compute statistics (mean, std, SEM, CI)
5. Perform statistical tests
6. Generate summary tables
7. Export to CSV and Excel

Usage:
    python experiments/phase_3_benchmark_experiments/06_aggregate_results.py \\
        --results_dir results/phase3 \\
        --output results/phase3/aggregated_results.csv
"""

import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def parse_args():
    parser = argparse.ArgumentParser(description='Aggregate Phase 3 experiment results')
    parser.add_argument('--results_dir', type=str, default='results/phase3',
                        help='Directory containing experiment results')
    parser.add_argument('--output', type=str, default='results/phase3/aggregated_results.csv',
                        help='Output CSV file path')
    parser.add_argument('--excel', action='store_true',
                        help='Also save as Excel file')
    parser.add_argument('--include_baselines', action='store_true',
                        help='Include baseline results')
    return parser.parse_args()


def find_result_files(results_dir, pattern='*_results.json'):
    """Find all result JSON files recursively."""
    results_path = Path(results_dir)
    json_files = list(results_path.rglob(pattern))
    print(f"Found {len(json_files)} result files")
    return json_files


def parse_result_file(json_file):
    """Parse a single result JSON file."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Extract metadata
        result = {
            'file_path': str(json_file),
            'timestamp': data.get('timestamp', ''),
        }

        # Try to extract experiment info
        if 'experiment' in data:
            parts = data['experiment'].split('_')
            if len(parts) >= 3:
                result['transform'] = parts[0]
                result['model'] = parts[1]
                result['subject'] = '_'.join(parts[2:])
        elif 'model' in data:
            # Baseline result format
            result['transform'] = 'raw_signal'
            result['model'] = data.get('model', 'unknown')
            result['subject'] = data.get('subject', 'unknown')

        # Extract performance metrics
        result['mean_accuracy'] = data.get('mean_accuracy', np.nan)
        result['std_accuracy'] = data.get('std_accuracy', np.nan)
        result['k_folds'] = data.get('k_folds', len(data.get('fold_results', [])))

        # Extract fold-level results if available
        if 'fold_results' in data:
            fold_accs = [fold.get('best_val_acc', fold.get('final_val_acc', np.nan))
                        for fold in data['fold_results']]
            result['fold_accuracies'] = fold_accs
            result['min_accuracy'] = np.min(fold_accs)
            result['max_accuracy'] = np.max(fold_accs)
            result['median_accuracy'] = np.median(fold_accs)

            # Compute confidence interval (95%)
            n = len(fold_accs)
            sem = np.std(fold_accs, ddof=1) / np.sqrt(n)
            ci = stats.t.ppf(0.975, n-1) * sem
            result['sem'] = sem
            result['ci_95'] = ci
            result['ci_lower'] = result['mean_accuracy'] - ci
            result['ci_upper'] = result['mean_accuracy'] + ci

        # Extract training time if available
        if 'fold_results' in data:
            training_times = [fold.get('training_time_s', np.nan)
                             for fold in data['fold_results']]
            result['mean_training_time_s'] = np.mean([t for t in training_times if not np.isnan(t)])

        return result

    except Exception as e:
        print(f"Error parsing {json_file}: {e}")
        return None


def aggregate_results(result_files):
    """Aggregate all result files into a DataFrame."""
    results = []

    for json_file in result_files:
        result = parse_result_file(json_file)
        if result is not None:
            results.append(result)

    df = pd.DataFrame(results)
    return df


def compute_summary_statistics(df):
    """Compute summary statistics grouped by transform and model."""
    summary = df.groupby(['transform', 'model']).agg({
        'mean_accuracy': ['mean', 'std', 'min', 'max', 'count'],
        'std_accuracy': 'mean',
        'mean_training_time_s': 'mean'
    }).round(2)

    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary = summary.reset_index()

    return summary


def perform_statistical_tests(df):
    """Perform pairwise statistical tests between methods."""
    print("\nPerforming statistical tests...")

    # Get unique transform-model combinations
    df['method'] = df['transform'] + '_' + df['model']
    methods = df['method'].unique()

    # Initialize p-value matrix
    n_methods = len(methods)
    p_values = np.ones((n_methods, n_methods))

    # Perform pairwise Wilcoxon signed-rank tests
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if i < j:
                # Get fold accuracies for both methods
                acc1_list = df[df['method'] == method1]['fold_accuracies'].tolist()
                acc2_list = df[df['method'] == method2]['fold_accuracies'].tolist()

                if acc1_list and acc2_list:
                    # Flatten lists
                    acc1 = [item for sublist in acc1_list for item in sublist]
                    acc2 = [item for sublist in acc2_list for item in sublist]

                    # Ensure same length (use minimum)
                    min_len = min(len(acc1), len(acc2))
                    acc1, acc2 = acc1[:min_len], acc2[:min_len]

                    if len(acc1) > 1:
                        try:
                            _, p_value = stats.wilcoxon(acc1, acc2)
                            p_values[i, j] = p_value
                            p_values[j, i] = p_value
                        except:
                            pass

    # Create p-value DataFrame
    p_df = pd.DataFrame(p_values, index=methods, columns=methods)

    return p_df


def export_results(df, summary, p_values, output_path, excel=False):
    """Export results to CSV and optionally Excel."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save main results to CSV
    df_export = df.drop(columns=['fold_accuracies'], errors='ignore')
    df_export.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Save summary
    summary_path = output_path.parent / (output_path.stem + '_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    # Save p-values
    if p_values is not None:
        p_path = output_path.parent / (output_path.stem + '_pvalues.csv')
        p_values.to_csv(p_path)
        print(f"P-values saved to {p_path}")

    # Excel export
    if excel:
        excel_path = output_path.with_suffix('.xlsx')
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_export.to_excel(writer, sheet_name='All Results', index=False)
            summary.to_excel(writer, sheet_name='Summary', index=False)
            if p_values is not None:
                p_values.to_excel(writer, sheet_name='P-Values')
        print(f"Excel file saved to {excel_path}")


def print_top_performers(df, n=10):
    """Print top N performing methods."""
    print(f"\n{'='*60}")
    print(f"Top {n} Performing Methods")
    print(f"{'='*60}")

    top = df.nlargest(n, 'mean_accuracy')[['transform', 'model', 'subject', 'mean_accuracy', 'std_accuracy']]
    print(top.to_string(index=False))


def generate_summary_report(df, summary, output_dir):
    """Generate a text summary report."""
    report_path = Path(output_dir) / 'summary_report.txt'

    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Phase 3: Experiment Results Summary\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Unique transforms: {df['transform'].nunique()}\n")
        f.write(f"Unique models: {df['model'].nunique()}\n")
        f.write(f"Subjects: {df['subject'].nunique()}\n\n")

        f.write("="*60 + "\n")
        f.write("Summary by Transform and Model\n")
        f.write("="*60 + "\n")
        f.write(summary.to_string(index=False) + "\n\n")

        # Best overall
        best_row = df.loc[df['mean_accuracy'].idxmax()]
        f.write("="*60 + "\n")
        f.write("Best Overall Performance\n")
        f.write("="*60 + "\n")
        f.write(f"Transform: {best_row['transform']}\n")
        f.write(f"Model: {best_row['model']}\n")
        f.write(f"Subject: {best_row['subject']}\n")
        f.write(f"Accuracy: {best_row['mean_accuracy']:.2f} ± {best_row['std_accuracy']:.2f}%\n\n")

        # Per transform
        f.write("="*60 + "\n")
        f.write("Best Performance per Transform\n")
        f.write("="*60 + "\n")
        for transform in df['transform'].unique():
            transform_df = df[df['transform'] == transform]
            best = transform_df.loc[transform_df['mean_accuracy'].idxmax()]
            f.write(f"{transform:20s}: {best['mean_accuracy']:.2f}% ({best['model']})\n")

        f.write("\n")

        # Per model
        f.write("="*60 + "\n")
        f.write("Best Performance per Model\n")
        f.write("="*60 + "\n")
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            best = model_df.loc[model_df['mean_accuracy'].idxmax()]
            f.write(f"{model:20s}: {best['mean_accuracy']:.2f}% ({best['transform']})\n")

    print(f"\nSummary report saved to {report_path}")


def main():
    args = parse_args()

    print("="*60)
    print("Phase 3: Results Aggregation")
    print("="*60)

    # Find result files
    print(f"\nScanning {args.results_dir} for result files...")
    result_files = find_result_files(args.results_dir)

    if not result_files:
        print("No result files found!")
        return

    # Aggregate results
    print("\nAggregating results...")
    df = aggregate_results(result_files)

    print(f"\nLoaded {len(df)} experiments")
    print(f"Transforms: {sorted(df['transform'].unique())}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Subjects: {sorted(df['subject'].unique())}")

    # Compute summary
    print("\nComputing summary statistics...")
    summary = compute_summary_statistics(df)

    # Statistical tests
    p_values = None
    if len(df) > 1:
        try:
            p_values = perform_statistical_tests(df)
        except Exception as e:
            print(f"Warning: Could not perform statistical tests: {e}")

    # Print top performers
    print_top_performers(df, n=min(10, len(df)))

    # Export results
    print("\nExporting results...")
    export_results(df, summary, p_values, args.output, args.excel)

    # Generate summary report
    output_dir = Path(args.output).parent
    generate_summary_report(df, summary, output_dir)

    print("\n" + "="*60)
    print("Aggregation Complete")
    print("="*60)


if __name__ == '__main__':
    main()
