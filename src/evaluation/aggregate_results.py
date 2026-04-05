"""
Results aggregation module for benchmark studies.

Aggregates results from multiple experiment runs, computes summary statistics,
and generates comparison tables for reporting.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime


class ResultsAggregator:
    """
    Aggregate and analyze results from multiple experiment runs.

    Example:
        >>> aggregator = ResultsAggregator('results/')
        >>> df = aggregator.load_results()
        >>> summary = aggregator.summarize_by_model()
        >>> aggregator.export_summary('results/analysis/')
    """

    def __init__(self, results_dir: Union[str, Path]):
        """
        Initialize results aggregator.

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = Path(results_dir)
        self.results = []
        self.df = None

    def load_results(self, pattern: str = '**/results.json') -> pd.DataFrame:
        """
        Load all results from JSON files.

        Args:
            pattern: Glob pattern for result files

        Returns:
            DataFrame with all results

        Example:
            >>> aggregator = ResultsAggregator('results/')
            >>> df = aggregator.load_results()
            >>> print(df.shape)
        """
        result_files = list(self.results_dir.glob(pattern))

        if not result_files:
            raise FileNotFoundError(f"No results found in {self.results_dir}")

        all_results = []

        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)

                # Flatten nested structure
                config = data.get('config', {})
                models = data.get('models', {})

                for model_name, model_results in models.items():
                    row = {
                        'experiment': config.get('name', 'unknown'),
                        'model': model_name,
                        'model_architecture': config.get('models', [{}])[0].get('architecture', 'unknown'),
                        'dataset': config.get('dataset', {}).get('name', 'unknown'),
                        'batch_size': config.get('training', {}).get('batch_size', 32),
                        'learning_rate': config.get('optimizer', {}).get('learning_rate', 0.001),
                        'epochs': config.get('training', {}).get('epochs', 100),
                        'augmentation_enabled': config.get('augmentation', {}).get('enabled', False),
                    }

                    # Add test metrics if available
                    if 'test_metrics' in model_results:
                        metrics = model_results['test_metrics']
                        row.update({
                            'accuracy': metrics.get('accuracy', np.nan),
                            'precision': metrics.get('precision', np.nan),
                            'recall': metrics.get('recall', np.nan),
                            'f1': metrics.get('f1', np.nan),
                            'auc': metrics.get('auc', np.nan),
                            'kappa': metrics.get('kappa', np.nan),
                            'mcc': metrics.get('mcc', np.nan),
                        })

                    all_results.append(row)

            except Exception as e:
                print(f"Warning: Failed to load {result_file}: {e}")
                continue

        self.df = pd.DataFrame(all_results)
        return self.df

    def summarize_by_model(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics per model.

        Returns:
            Dictionary mapping model names to summary statistics

        Example:
            >>> summary = aggregator.summarize_by_model()
            >>> for model, stats in summary.items():
            ...     print(f"{model}: {stats['accuracy']['mean']:.4f}")
        """
        if self.df is None:
            raise ValueError("Results not loaded. Call load_results() first.")

        summary = {}
        metrics = ['accuracy', 'f1', 'auc', 'kappa', 'mcc']

        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]

            summary[model] = {}

            for metric in metrics:
                if metric in model_data.columns:
                    values = model_data[metric].dropna()

                    if len(values) > 0:
                        summary[model][metric] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'n': len(values)
                        }

        return summary

    def summarize_by_architecture(self) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics per model architecture.

        Returns:
            Dictionary mapping architectures to summary statistics
        """
        if self.df is None:
            raise ValueError("Results not loaded. Call load_results() first.")

        summary = {}
        metrics = ['accuracy', 'f1', 'auc', 'kappa', 'mcc']

        for arch in self.df['model_architecture'].unique():
            arch_data = self.df[self.df['model_architecture'] == arch]

            summary[arch] = {}

            for metric in metrics:
                if metric in arch_data.columns:
                    values = arch_data[metric].dropna()

                    if len(values) > 0:
                        summary[arch][metric] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'min': float(values.min()),
                            'max': float(values.max()),
                            'n': len(values)
                        }

        return summary

    def summarize_by_augmentation(self) -> Dict[str, Dict[str, float]]:
        """
        Compare results with and without augmentation.

        Returns:
            Dictionary with augmentation impact analysis
        """
        if self.df is None:
            raise ValueError("Results not loaded. Call load_results() first.")

        summary = {}
        metrics = ['accuracy', 'f1', 'auc', 'kappa', 'mcc']

        for aug_status in [True, False]:
            aug_data = self.df[self.df['augmentation_enabled'] == aug_status]

            summary[f'augmentation_{aug_status}'] = {}

            for metric in metrics:
                if metric in aug_data.columns:
                    values = aug_data[metric].dropna()

                    if len(values) > 0:
                        summary[f'augmentation_{aug_status}'][metric] = {
                            'mean': float(values.mean()),
                            'std': float(values.std()),
                            'n': len(values)
                        }

        return summary

    def get_top_models(self, metric: str = 'accuracy', n: int = 5) -> pd.DataFrame:
        """
        Get top-performing models by metric.

        Args:
            metric: Metric to sort by
            n: Number of top models to return

        Returns:
            DataFrame with top models

        Example:
            >>> top_models = aggregator.get_top_models('accuracy', n=5)
            >>> print(top_models)
        """
        if self.df is None:
            raise ValueError("Results not loaded. Call load_results() first.")

        # Group by model and compute mean metric
        grouped = self.df.groupby('model')[metric].mean().sort_values(ascending=False)

        return grouped.head(n)

    def create_comparison_table(self, metric: str = 'accuracy') -> pd.DataFrame:
        """
        Create model comparison table for publication.

        Args:
            metric: Primary metric for comparison

        Returns:
            Formatted comparison DataFrame

        Example:
            >>> table = aggregator.create_comparison_table('accuracy')
            >>> print(table.to_string())
        """
        if self.df is None:
            raise ValueError("Results not loaded. Call load_results() first.")

        summary = self.summarize_by_model()

        rows = []
        for model, stats in summary.items():
            if metric in stats:
                m_stats = stats[metric]
                rows.append({
                    'Model': model,
                    'Mean': f"{m_stats['mean']:.4f}",
                    'Std': f"{m_stats['std']:.4f}",
                    'Min': f"{m_stats['min']:.4f}",
                    'Max': f"{m_stats['max']:.4f}",
                    'N': m_stats['n']
                })

        df = pd.DataFrame(rows)
        df = df.sort_values('Mean', ascending=False)

        return df

    def export_csv(self, output_dir: Union[str, Path]):
        """
        Export all results to CSV.

        Args:
            output_dir: Output directory for CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.df is not None:
            # Full results
            self.df.to_csv(output_dir / 'all_results.csv', index=False)

            # Summary by model
            summary = self.summarize_by_model()
            summary_df = pd.DataFrame([
                {
                    'Model': model,
                    'Accuracy_Mean': stats.get('accuracy', {}).get('mean', np.nan),
                    'Accuracy_Std': stats.get('accuracy', {}).get('std', np.nan),
                    'F1_Mean': stats.get('f1', {}).get('mean', np.nan),
                    'F1_Std': stats.get('f1', {}).get('std', np.nan),
                    'AUC_Mean': stats.get('auc', {}).get('mean', np.nan),
                    'AUC_Std': stats.get('auc', {}).get('std', np.nan),
                }
                for model, stats in summary.items()
            ])
            summary_df = summary_df.sort_values('Accuracy_Mean', ascending=False)
            summary_df.to_csv(output_dir / 'model_summary.csv', index=False)

            print(f"Exported results to {output_dir}")

    def export_json(self, output_dir: Union[str, Path]):
        """
        Export aggregated results to JSON.

        Args:
            output_dir: Output directory for JSON files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'by_model': self.summarize_by_model(),
            'by_architecture': self.summarize_by_architecture(),
            'by_augmentation': self.summarize_by_augmentation(),
            'total_experiments': len(self.df) if self.df is not None else 0
        }

        with open(output_dir / 'aggregated_results.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Exported JSON to {output_dir}")

    def create_summary_report(self, output_file: Union[str, Path]):
        """
        Create a text summary report.

        Args:
            output_file: Output file path
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RESULTS AGGREGATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Experiments: {len(self.df) if self.df is not None else 0}\n\n")

            # Model summary
            f.write("=" * 80 + "\n")
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            summary = self.summarize_by_model()
            for model, stats in sorted(summary.items()):
                f.write(f"{model}:\n")
                for metric, values in stats.items():
                    f.write(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f} "
                           f"(n={values['n']})\n")
                f.write("\n")

            # Architecture summary
            f.write("=" * 80 + "\n")
            f.write("ARCHITECTURE PERFORMANCE SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            arch_summary = self.summarize_by_architecture()
            for arch, stats in sorted(arch_summary.items()):
                f.write(f"{arch}:\n")
                for metric, values in stats.items():
                    f.write(f"  {metric}: {values['mean']:.4f} ± {values['std']:.4f} "
                           f"(n={values['n']})\n")
                f.write("\n")

            # Top models
            f.write("=" * 80 + "\n")
            f.write("TOP 5 MODELS BY ACCURACY\n")
            f.write("=" * 80 + "\n\n")

            top_models = self.get_top_models('accuracy', n=5)
            for i, (model, acc) in enumerate(top_models.items(), 1):
                f.write(f"{i}. {model}: {acc:.4f}\n")

        print(f"Report saved to {output_file}")

    def print_summary(self):
        """Print summary statistics to console."""
        if self.df is None:
            print("No results loaded.")
            return

        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        print(f"\nTotal experiments: {len(self.df)}")
        print(f"Unique models: {self.df['model'].nunique()}")
        print(f"Unique architectures: {self.df['model_architecture'].nunique()}")

        print("\nTop 5 Models by Accuracy:")
        top_models = self.get_top_models('accuracy', n=5)
        for i, (model, acc) in enumerate(top_models.items(), 1):
            print(f"  {i}. {model}: {acc:.4f}")

        print("\nAccuracy by Architecture:")
        arch_summary = self.summarize_by_architecture()
        for arch in sorted(arch_summary.keys()):
            stats = arch_summary[arch]
            if 'accuracy' in stats:
                acc = stats['accuracy']
                print(f"  {arch}: {acc['mean']:.4f} ± {acc['std']:.4f}")

        print("\nAugmentation Impact:")
        aug_summary = self.summarize_by_augmentation()
        if 'augmentation_True' in aug_summary and 'augmentation_False' in aug_summary:
            with_aug = aug_summary['augmentation_True'].get('accuracy', {})
            without_aug = aug_summary['augmentation_False'].get('accuracy', {})
            if with_aug and without_aug:
                diff = with_aug['mean'] - without_aug['mean']
                print(f"  With augmentation: {with_aug['mean']:.4f}")
                print(f"  Without augmentation: {without_aug['mean']:.4f}")
                print(f"  Difference: {diff:+.4f}")

        print("\n" + "=" * 80 + "\n")


def aggregate_and_report(results_dir: Union[str, Path], output_dir: Union[str, Path]):
    """
    Convenience function to aggregate results and generate reports.

    Args:
        results_dir: Directory containing experiment results
        output_dir: Output directory for aggregated results and reports

    Example:
        >>> aggregate_and_report('results/', 'results/analysis/')
    """
    aggregator = ResultsAggregator(results_dir)
    aggregator.load_results()
    aggregator.print_summary()
    aggregator.export_csv(output_dir)
    aggregator.export_json(output_dir)
    aggregator.create_summary_report(output_dir / 'summary_report.txt')
