"""
Statistical significance testing for model comparison.

Provides statistical tests including:
- Wilcoxon signed-rank test for paired comparisons
- Paired t-test for parametric comparisons
- ANOVA for comparing multiple models
- Post-hoc tests with multiple comparison correction
"""

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd


def wilcoxon_test(
    results_a: Union[List[float], np.ndarray],
    results_b: Union[List[float], np.ndarray],
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Perform Wilcoxon signed-rank test for paired samples.

    Non-parametric test for comparing two related samples.
    Suitable for cross-validation results where models are tested on same folds.

    Args:
        results_a: Results from model A across folds, shape (n_folds,)
        results_b: Results from model B across folds, shape (n_folds,)
        alternative: Alternative hypothesis ('two-sided', 'greater', 'less')

    Returns:
        Tuple of (statistic, p-value)

    Example:
        >>> results_a = [0.85, 0.87, 0.82, 0.89, 0.84]  # Model A accuracies
        >>> results_b = [0.82, 0.84, 0.80, 0.86, 0.81]  # Model B accuracies
        >>> stat, p = wilcoxon_test(results_a, results_b)
        >>> print(f"p-value: {p:.4f}, significant: {p < 0.05}")
    """
    results_a = np.array(results_a)
    results_b = np.array(results_b)

    if len(results_a) != len(results_b):
        raise ValueError("Results must have same length")

    if len(results_a) < 3:
        raise ValueError("Need at least 3 samples for Wilcoxon test")

    # Perform Wilcoxon signed-rank test
    statistic, p_value = stats.wilcoxon(results_a, results_b, alternative=alternative)

    return float(statistic), float(p_value)


def paired_t_test(
    results_a: Union[List[float], np.ndarray],
    results_b: Union[List[float], np.ndarray],
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Perform paired t-test for comparing two models.

    Parametric test assuming normality. Use when results are normally distributed.

    Args:
        results_a: Results from model A across folds
        results_b: Results from model B across folds
        alternative: Alternative hypothesis ('two-sided', 'greater', 'less')

    Returns:
        Tuple of (statistic, p-value)

    Example:
        >>> results_a = [0.85, 0.87, 0.82, 0.89, 0.84]
        >>> results_b = [0.82, 0.84, 0.80, 0.86, 0.81]
        >>> stat, p = paired_t_test(results_a, results_b)
        >>> print(f"t-statistic: {stat:.4f}, p-value: {p:.4f}")
    """
    results_a = np.array(results_a)
    results_b = np.array(results_b)

    if len(results_a) != len(results_b):
        raise ValueError("Results must have same length")

    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(results_a, results_b, alternative=alternative)

    return float(statistic), float(p_value)


def anova_test(
    results_dict: Dict[str, Union[List[float], np.ndarray]],
    repeated_measures: bool = False,
    subject_ids: Optional[List] = None
) -> Dict[str, float]:
    """
    Perform ANOVA to compare multiple models.

    Args:
        results_dict: Dictionary mapping model names to their results
                     e.g., {'model_a': [0.85, 0.87, ...], 'model_b': [0.82, 0.84, ...]}
        repeated_measures: If True, use repeated-measures ANOVA (for paired data)
        subject_ids: Subject/fold identifiers for repeated-measures ANOVA

    Returns:
        Dictionary containing F-statistic and p-value

    Example:
        >>> results = {
        ...     'ResNet': [0.85, 0.87, 0.82, 0.89, 0.84],
        ...     'ViT': [0.88, 0.90, 0.85, 0.91, 0.87],
        ...     'Baseline': [0.75, 0.77, 0.73, 0.78, 0.76]
        ... }
        >>> result = anova_test(results)
        >>> print(f"F-statistic: {result['F']:.4f}, p-value: {result['p_value']:.4f}")
    """
    if repeated_measures:
        # Prepare data for repeated-measures ANOVA
        data = []
        model_names = list(results_dict.keys())
        n_folds = len(results_dict[model_names[0]])

        if subject_ids is None:
            subject_ids = list(range(n_folds))

        for model_name, results in results_dict.items():
            for fold_id, result in zip(subject_ids, results):
                data.append({
                    'subject': fold_id,
                    'model': model_name,
                    'score': result
                })

        df = pd.DataFrame(data)

        # Perform repeated-measures ANOVA
        aovrm = AnovaRM(df, 'score', 'subject', within=['model'])
        res = aovrm.fit()

        return {
            'F': float(res.anova_table['F Value']['model']),
            'p_value': float(res.anova_table['Pr > F']['model']),
            'table': res.anova_table
        }

    else:
        # One-way ANOVA (independent groups)
        groups = [np.array(results) for results in results_dict.values()]
        f_stat, p_value = stats.f_oneway(*groups)

        return {
            'F': float(f_stat),
            'p_value': float(p_value)
        }


def posthoc_tests(
    results_dict: Dict[str, Union[List[float], np.ndarray]],
    test: str = 'wilcoxon',
    correction: str = 'bonferroni'
) -> pd.DataFrame:
    """
    Perform pairwise post-hoc tests between all models.

    Args:
        results_dict: Dictionary mapping model names to their results
        test: Test to use ('wilcoxon' or 'ttest')
        correction: Multiple comparison correction method
                   ('bonferroni', 'fdr_bh', 'holm', 'none')

    Returns:
        DataFrame with pairwise comparison results

    Example:
        >>> results = {
        ...     'ResNet': [0.85, 0.87, 0.82, 0.89, 0.84],
        ...     'ViT': [0.88, 0.90, 0.85, 0.91, 0.87],
        ...     'Baseline': [0.75, 0.77, 0.73, 0.78, 0.76]
        ... }
        >>> df = posthoc_tests(results, test='wilcoxon', correction='bonferroni')
        >>> print(df)
    """
    model_names = list(results_dict.keys())
    n_models = len(model_names)

    comparisons = []

    # Perform all pairwise comparisons
    for i in range(n_models):
        for j in range(i + 1, n_models):
            model_a = model_names[i]
            model_b = model_names[j]

            results_a = np.array(results_dict[model_a])
            results_b = np.array(results_dict[model_b])

            # Perform test
            if test == 'wilcoxon':
                stat, p_value = wilcoxon_test(results_a, results_b)
            elif test == 'ttest':
                stat, p_value = paired_t_test(results_a, results_b)
            else:
                raise ValueError(f"Unknown test: {test}")

            comparisons.append({
                'model_a': model_a,
                'model_b': model_b,
                'mean_a': float(np.mean(results_a)),
                'mean_b': float(np.mean(results_b)),
                'diff': float(np.mean(results_a) - np.mean(results_b)),
                'statistic': stat,
                'p_value': p_value
            })

    df = pd.DataFrame(comparisons)

    # Apply multiple comparison correction
    if correction != 'none' and len(comparisons) > 0:
        p_values = df['p_value'].values
        reject, p_corrected, _, _ = multipletests(p_values, method=correction)

        df['p_corrected'] = p_corrected
        df['significant'] = reject
    else:
        df['p_corrected'] = df['p_value']
        df['significant'] = df['p_value'] < 0.05

    return df


def multiple_comparison_correction(
    p_values: Union[List[float], np.ndarray],
    method: str = 'bonferroni',
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply multiple comparison correction to p-values.

    Args:
        p_values: Array of p-values from multiple tests
        method: Correction method:
               - 'bonferroni': Bonferroni correction (most conservative)
               - 'fdr_bh': Benjamini-Hochberg FDR control
               - 'holm': Holm-Bonferroni method
               - 'sidak': Sidak correction
        alpha: Family-wise error rate

    Returns:
        Tuple of (reject_array, corrected_p_values)
        - reject_array: Boolean array indicating which null hypotheses to reject
        - corrected_p_values: Adjusted p-values

    Example:
        >>> p_values = [0.01, 0.04, 0.03, 0.08, 0.02]
        >>> reject, p_corrected = multiple_comparison_correction(p_values, method='bonferroni')
        >>> print(f"Reject: {reject}")
        >>> print(f"Corrected p-values: {p_corrected}")
    """
    p_values = np.array(p_values)

    reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)

    return reject, p_corrected


def compare_models(
    model_results: Dict[str, Dict[str, Union[List[float], np.ndarray]]],
    metric: str = 'accuracy',
    test: str = 'wilcoxon',
    correction: str = 'bonferroni',
    verbose: bool = True
) -> Dict:
    """
    Comprehensive statistical comparison of multiple models.

    Performs ANOVA followed by pairwise post-hoc tests.

    Args:
        model_results: Nested dictionary:
                      {'model_name': {'accuracy': [...], 'f1': [...], ...}}
        metric: Metric to compare (e.g., 'accuracy', 'f1')
        test: Pairwise test ('wilcoxon' or 'ttest')
        correction: Multiple comparison correction method
        verbose: If True, print results

    Returns:
        Dictionary containing:
            - anova: ANOVA results
            - posthoc: DataFrame of pairwise comparisons
            - summary: Summary statistics per model

    Example:
        >>> results = {
        ...     'ResNet': {'accuracy': [0.85, 0.87, 0.82, 0.89, 0.84]},
        ...     'ViT': {'accuracy': [0.88, 0.90, 0.85, 0.91, 0.87]},
        ...     'Baseline': {'accuracy': [0.75, 0.77, 0.73, 0.78, 0.76]}
        ... }
        >>> comparison = compare_models(results, metric='accuracy')
    """
    # Extract metric values for each model
    metric_results = {
        model_name: results[metric]
        for model_name, results in model_results.items()
    }

    # Perform ANOVA
    anova_results = anova_test(metric_results, repeated_measures=True)

    # Perform post-hoc tests
    posthoc_results = posthoc_tests(metric_results, test=test, correction=correction)

    # Compute summary statistics
    summary = {}
    for model_name, values in metric_results.items():
        values = np.array(values)
        summary[model_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values))
        }

    if verbose:
        print(f"\n{'='*60}")
        print(f"Statistical Comparison: {metric}")
        print(f"{'='*60}\n")

        print("Summary Statistics:")
        for model_name, stats in summary.items():
            print(f"  {model_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")

        print(f"\nANOVA Results:")
        print(f"  F-statistic: {anova_results['F']:.4f}")
        print(f"  p-value: {anova_results['p_value']:.4f}")
        print(f"  Significant: {anova_results['p_value'] < 0.05}")

        print(f"\nPairwise Comparisons ({test} test, {correction} correction):")
        for _, row in posthoc_results.iterrows():
            sig_marker = "***" if row['significant'] else ""
            print(f"  {row['model_a']} vs {row['model_b']}: "
                  f"p={row['p_corrected']:.4f} {sig_marker}")

    return {
        'anova': anova_results,
        'posthoc': posthoc_results,
        'summary': summary
    }


def effect_size_cohens_d(
    results_a: Union[List[float], np.ndarray],
    results_b: Union[List[float], np.ndarray]
) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    Cohen's d measures the standardized difference between two means.
    Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large

    Args:
        results_a: Results from model A
        results_b: Results from model B

    Returns:
        Cohen's d effect size

    Example:
        >>> results_a = [0.85, 0.87, 0.82, 0.89, 0.84]
        >>> results_b = [0.82, 0.84, 0.80, 0.86, 0.81]
        >>> d = effect_size_cohens_d(results_a, results_b)
        >>> print(f"Cohen's d: {d:.4f}")
    """
    results_a = np.array(results_a)
    results_b = np.array(results_b)

    # Compute means
    mean_a = np.mean(results_a)
    mean_b = np.mean(results_b)

    # Compute pooled standard deviation
    std_a = np.std(results_a, ddof=1)
    std_b = np.std(results_b, ddof=1)
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)

    # Compute Cohen's d
    d = (mean_a - mean_b) / pooled_std

    return float(d)
