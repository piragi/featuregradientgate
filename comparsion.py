#!/usr/bin/env python3
"""
Comprehensive comparison of best performer configurations vs vanilla baselines.
Analyzes SaCo, Faithfulness Correlation, and PixelFlipping metrics with statistical analysis.
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

warnings.filterwarnings('ignore')


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def load_experiment_data(experiment_path: Path) -> Dict[str, Any]:
    """Load all data for a single experiment."""
    results_file = experiment_path / "results.json"
    config_file = experiment_path / "experiment_config.json"

    # Find faithfulness stats file
    test_dir = experiment_path / "test"
    faithfulness_files = list(test_dir.glob("faithfulness_stats_*.json"))

    if not faithfulness_files:
        raise FileNotFoundError(f"No faithfulness stats found in {test_dir}")

    faithfulness_file = faithfulness_files[0]

    # Load all files
    with open(results_file) as f:
        results = json.load(f)

    with open(config_file) as f:
        config = json.load(f)

    with open(faithfulness_file) as f:
        faithfulness = json.load(f)

    return {'results': results, 'config': config, 'faithfulness': faithfulness, 'experiment_name': experiment_path.name}


def extract_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from experiment data."""
    metrics = {}

    # SaCo metrics
    saco = data['results']['saco_results']
    metrics['saco_mean'] = saco['mean']
    metrics['saco_std'] = saco['std']
    metrics['saco_n'] = saco['n_samples']

    # Faithfulness Correlation
    faith_corr = data['faithfulness']['metrics']['FaithfulnessCorrelation']['overall']
    metrics['faithfulness_correlation_mean'] = faith_corr['mean']
    metrics['faithfulness_correlation_std'] = faith_corr['std']
    metrics['faithfulness_correlation_n'] = faith_corr['count']

    # PixelFlipping
    pixel_flip = data['faithfulness']['metrics']['PixelFlipping']['overall']
    metrics['pixelflipping_mean'] = pixel_flip['mean']
    metrics['pixelflipping_std'] = pixel_flip['std']
    metrics['pixelflipping_n'] = pixel_flip['count']

    return metrics


def get_experiment_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract experiment configuration info."""
    config = data['config']
    return {
        'dataset': data['results']['dataset'],
        'gate_construction': config['experiment_params']['gate_construction'],
        'use_feature_gradients': config['experiment_params']['use_feature_gradients'],
        'feature_gradient_layers': config['experiment_params'].get('feature_gradient_layers', []),
        'kappa': config['experiment_params']['kappa'],
        'experiment_name': data['experiment_name']
    }


def load_all_experiments(sweep_dirs: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all experiments from sweep folders and separate vanilla from treatment experiments."""

    sweep_data = []
    vanilla_data = []

    # Load all experiments from all sweep directories
    for sweep_dir_str in sweep_dirs:
        sweep_dir = Path(sweep_dir_str)
        print(f"Loading experiments from {sweep_dir}...")

        # Load all experiments from sweep directory structure
        for dataset_dir in sweep_dir.iterdir():
            if dataset_dir.is_dir():
                # Load all experiment configurations for this dataset
                for exp_dir in dataset_dir.iterdir():
                    if exp_dir.is_dir():
                        try:
                            data = load_experiment_data(exp_dir)
                            metrics = extract_metrics(data)
                            info = get_experiment_info(data)

                            # Add experiment path info for identification
                            combined = {**metrics, **info, 'experiment_path': str(exp_dir)}

                            # Separate vanilla from treatment experiments
                            if exp_dir.name == 'vanilla':
                                combined['experiment_type'] = 'vanilla'
                                vanilla_data.append(combined)
                            else:
                                combined['experiment_type'] = 'treatment'
                                sweep_data.append(combined)

                        except Exception as e:
                            print(f"Error loading {exp_dir}: {e}")

    sweep_df = pd.DataFrame(sweep_data)
    vanilla_df = pd.DataFrame(vanilla_data)

    return sweep_df, vanilla_df


def calculate_statistical_comparison(sweep_df: pd.DataFrame, vanilla_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive statistical comparisons between best performers and vanilla."""

    metrics = ['saco', 'faithfulness_correlation', 'pixelflipping']
    comparisons = []

    # Get unique datasets from the data
    datasets = sweep_df['dataset'].unique()

    for dataset in datasets:
        # Get vanilla baseline for this dataset
        vanilla_subset = vanilla_df[vanilla_df['dataset'] == dataset]
        if len(vanilla_subset) == 0:
            print(f"Warning: No vanilla baseline found for {dataset}")
            continue

        vanilla_row = vanilla_subset.iloc[0]

        # Get all treatment experiments for this dataset
        treatment_subset = sweep_df[sweep_df['dataset'] == dataset]

        if len(treatment_subset) == 0:
            print(f"Warning: No treatment experiments found for {dataset}")
            continue

        # Find best performers for each metric
        for metric in metrics:
            # For pixelflipping, lower is better; for others, higher is better
            if metric == 'pixelflipping':
                best_performer = treatment_subset.loc[treatment_subset[f'{metric}_mean'].idxmin()]
            else:
                best_performer = treatment_subset.loc[treatment_subset[f'{metric}_mean'].idxmax()]

        # Compare all experiments against vanilla
        for _, treatment_row in treatment_subset.iterrows():
            comparison = {
                'dataset': dataset,
                'experiment_config': treatment_row['experiment_name'],
                'gate_construction': treatment_row['gate_construction'],
                'use_feature_gradients': treatment_row['use_feature_gradients'],
                'kappa': treatment_row['kappa'],
                'experiment_path': treatment_row['experiment_path']
            }

            for metric in metrics:
                # Extract values for comparison
                treatment_mean = treatment_row[f'{metric}_mean']
                treatment_std = treatment_row[f'{metric}_std']
                treatment_n = treatment_row[f'{metric}_n']

                vanilla_mean = vanilla_row[f'{metric}_mean']
                vanilla_std = vanilla_row[f'{metric}_std']
                vanilla_n = vanilla_row[f'{metric}_n']

                # Calculate effect size (Cohen's d)
                # Since we don't have raw data, approximate using means and stds
                pooled_std = np.sqrt(((treatment_n - 1) * treatment_std**2 + (vanilla_n - 1) * vanilla_std**2) /
                                     (treatment_n + vanilla_n - 2))

                # For PixelFlipping, lower is better, so we flip the comparison
                if metric == 'pixelflipping':
                    cohens_d_value = (vanilla_mean - treatment_mean) / pooled_std if pooled_std > 0 else 0
                    difference = vanilla_mean - treatment_mean  # Positive means improvement (reduction)
                    percent_improvement = ((vanilla_mean - treatment_mean) / vanilla_mean *
                                           100) if vanilla_mean != 0 else 0
                    t_stat_direction = (vanilla_mean - treatment_mean)
                else:
                    cohens_d_value = (treatment_mean - vanilla_mean) / pooled_std if pooled_std > 0 else 0
                    difference = treatment_mean - vanilla_mean
                    percent_improvement = ((treatment_mean - vanilla_mean) / vanilla_mean *
                                           100) if vanilla_mean != 0 else 0
                    t_stat_direction = (treatment_mean - vanilla_mean)

                # Calculate standard error for difference
                se_diff = np.sqrt((treatment_std**2 / treatment_n) + (vanilla_std**2 / vanilla_n))

                # Calculate t-statistic for two-sample t-test (approximation)
                t_stat = t_stat_direction / se_diff if se_diff > 0 else 0
                df_welch = ((treatment_std**2 / treatment_n + vanilla_std**2 / vanilla_n)**
                            2) / ((treatment_std**2 / treatment_n)**2 / (treatment_n - 1) +
                                  (vanilla_std**2 / vanilla_n)**2 / (vanilla_n - 1))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df_welch)) if se_diff > 0 else 1.0

                # 95% confidence interval for difference
                critical_t = stats.t.ppf(0.975, df_welch)
                if metric == 'pixelflipping':
                    ci_lower = (vanilla_mean - treatment_mean) - critical_t * se_diff
                    ci_upper = (vanilla_mean - treatment_mean) + critical_t * se_diff
                else:
                    ci_lower = (treatment_mean - vanilla_mean) - critical_t * se_diff
                    ci_upper = (treatment_mean - vanilla_mean) + critical_t * se_diff

                comparison.update({
                    f'{metric}_treatment_mean': treatment_mean,
                    f'{metric}_treatment_std': treatment_std,
                    f'{metric}_treatment_se': treatment_std / np.sqrt(treatment_n),
                    f'{metric}_vanilla_mean': vanilla_mean,
                    f'{metric}_vanilla_std': vanilla_std,
                    f'{metric}_vanilla_se': vanilla_std / np.sqrt(vanilla_n),
                    f'{metric}_difference': difference,
                    f'{metric}_percent_improvement': percent_improvement,
                    f'{metric}_cohens_d': cohens_d_value,
                    f'{metric}_p_value': p_value,
                    f'{metric}_ci_lower': ci_lower,
                    f'{metric}_ci_upper': ci_upper,
                    f'{metric}_significant': p_value < 0.05
                })

            comparisons.append(comparison)

    return pd.DataFrame(comparisons)


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def print_detailed_results(comparison_df: pd.DataFrame):
    """Print detailed comparison results."""

    print("=" * 100)
    print("COMPREHENSIVE COMPARISON: BEST PERFORMERS vs VANILLA BASELINES")
    print("=" * 100)

    datasets = comparison_df['dataset'].unique()
    metrics = ['saco', 'faithfulness_correlation', 'pixelflipping']

    for dataset in datasets:
        print(f"\n{'='*20} {dataset.upper()} {'='*20}")

        dataset_df = comparison_df[comparison_df['dataset'] == dataset]

        for _, row in dataset_df.iterrows():
            print(f"\nConfiguration: {row['experiment_config']}")
            print(f"Gate Construction: {row['gate_construction']}")
            print(f"Feature Gradients: {row['use_feature_gradients']}")
            print(f"Kappa: {row['kappa']}")
            print(f"Path: {row['experiment_path']}")
            print("-" * 60)

            for metric in metrics:
                metric_name = metric.replace('_', ' ').title()

                treatment_mean = row[f'{metric}_treatment_mean']
                treatment_se = row[f'{metric}_treatment_se']
                vanilla_mean = row[f'{metric}_vanilla_mean']
                vanilla_se = row[f'{metric}_vanilla_se']

                difference = row[f'{metric}_difference']
                percent_improvement = row[f'{metric}_percent_improvement']
                cohens_d_val = row[f'{metric}_cohens_d']
                p_value = row[f'{metric}_p_value']
                ci_lower = row[f'{metric}_ci_lower']
                ci_upper = row[f'{metric}_ci_upper']

                effect_interpretation = interpret_effect_size(cohens_d_val)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

                print(f"{metric_name}:")
                print(f"  Treatment:      {treatment_mean:.4f} ± {treatment_se:.4f}")
                print(f"  Vanilla:        {vanilla_mean:.4f} ± {vanilla_se:.4f}")
                if metric == 'pixelflipping':
                    print(f"  Improvement:    {difference:.4f} ({percent_improvement:+.2f}%) [lower is better]")
                else:
                    print(f"  Difference:     {difference:.4f} ({percent_improvement:+.2f}%)")
                print(f"  95% CI:         [{ci_lower:.4f}, {ci_upper:.4f}]")
                print(f"  Cohen's d:      {cohens_d_val:.4f} ({effect_interpretation})")
                print(f"  p-value:        {p_value:.4f} {significance}")
                print()


def create_summary_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Create a summary table of all comparisons."""

    summary_data = []
    metrics = ['saco', 'faithfulness_correlation', 'pixelflipping']

    for _, row in comparison_df.iterrows():
        for metric in metrics:
            summary_data.append({
                'Dataset': row['dataset'],
                'Configuration': row['experiment_config'],
                'Metric': metric.replace('_', ' ').title(),
                'Treatment': f"{row[f'{metric}_treatment_mean']:.4f} ± {row[f'{metric}_treatment_se']:.4f}",
                'Vanilla': f"{row[f'{metric}_vanilla_mean']:.4f} ± {row[f'{metric}_vanilla_se']:.4f}",
                'Improvement (%)': f"{row[f'{metric}_percent_improvement']:+.2f}%",
                'Cohen\'s d': f"{row[f'{metric}_cohens_d']:.4f}",
                'Effect Size': interpret_effect_size(row[f'{metric}_cohens_d']),
                'p-value': f"{row[f'{metric}_p_value']:.4f}",
                'Significant': "Yes" if row[f'{metric}_significant'] else "No"
            })

    return pd.DataFrame(summary_data)


def identify_best_performers(comparison_df: pd.DataFrame):
    """Identify and print the best performing configurations for each metric."""

    print("\n" + "=" * 60)
    print("BEST PERFORMERS BY METRIC")
    print("=" * 60)

    datasets = comparison_df['dataset'].unique()
    metrics = ['saco', 'faithfulness_correlation', 'pixelflipping']

    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        dataset_df = comparison_df[comparison_df['dataset'] == dataset]

        for metric in metrics:
            # For pixelflipping, lower is better; for others, higher is better
            if metric == 'pixelflipping':
                best_row = dataset_df.loc[dataset_df[f'{metric}_treatment_mean'].idxmin()]
                direction = "lowest"
            else:
                best_row = dataset_df.loc[dataset_df[f'{metric}_treatment_mean'].idxmax()]
                direction = "highest"

            metric_name = metric.replace('_', ' ').title()
            treatment_mean = best_row[f'{metric}_treatment_mean']
            p_value = best_row[f'{metric}_p_value']
            percent_improvement = best_row[f'{metric}_percent_improvement']
            significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            print(f"  {metric_name} ({direction}): {best_row['experiment_config']}")
            print(f"    Value: {treatment_mean:.4f} ({percent_improvement:+.2f}% vs vanilla)")
            print(f"    p-value: {p_value:.4f} {significant}")


def classify_layer_type(feature_gradient_layers: List[str]) -> str:
    """Classify whether the configuration uses single or multi-layer."""
    if not feature_gradient_layers or len(feature_gradient_layers) == 0:
        return 'none'
    elif len(feature_gradient_layers) == 1:
        return 'single'
    else:
        return 'multi'


def calculate_composite_improvement(row: pd.Series, metrics: List[str] = None) -> float:
    """
    Calculate composite improvement across all three metrics.

    For each metric, we normalize the improvement and then take the average.
    For pixel flipping, positive improvement means reduction (better).
    For saco and faithfulness, positive improvement means increase (better).
    """
    if metrics is None:
        metrics = ['saco', 'faithfulness_correlation', 'pixelflipping']

    improvements = []
    for metric in metrics:
        percent_key = f'{metric}_percent_improvement'
        if percent_key in row and not pd.isna(row[percent_key]):
            improvements.append(row[percent_key])

    if not improvements:
        return np.nan

    return np.mean(improvements)


def identify_best_overall_performers(comparison_df: pd.DataFrame, sweep_df: pd.DataFrame):
    """
    Identify best single layer and best multi-layer across all three metrics.
    Also identify best single layer for each method type (if present).
    """

    print("\n" + "=" * 80)
    print("BEST OVERALL PERFORMERS ACROSS ALL THREE METRICS")
    print("=" * 80)

    # Add layer type classification to comparison_df
    layer_types = []
    for _, row in comparison_df.iterrows():
        # Find corresponding row in sweep_df to get feature_gradient_layers
        sweep_row = sweep_df[sweep_df['experiment_path'] == row['experiment_path']]
        if len(sweep_row) > 0:
            layers = sweep_row.iloc[0]['feature_gradient_layers']
            layer_types.append(classify_layer_type(layers))
        else:
            layer_types.append('unknown')

    comparison_df['layer_type'] = layer_types

    # Calculate composite improvement for each configuration
    comparison_df['composite_improvement'] = comparison_df.apply(calculate_composite_improvement, axis=1)

    datasets = comparison_df['dataset'].unique()

    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        print("-" * 60)

        dataset_df = comparison_df[comparison_df['dataset'] == dataset].copy()

        if len(dataset_df) == 0:
            print(f"  No experiments found for {dataset}")
            continue

        # Best single layer overall - Top 3
        single_layer_df = dataset_df[dataset_df['layer_type'] == 'single']
        if len(single_layer_df) > 0:
            top_single = single_layer_df.nlargest(3, 'composite_improvement')
            print(f"\n  Top 3 Single Layers (across all metrics):")
            for rank, (_, best_single) in enumerate(top_single.iterrows(), 1):
                print(f"\n    #{rank}:")
                print(f"      Configuration: {best_single['experiment_config']}")
                print(f"      Method: {best_single['gate_construction']}")
                print(f"      Kappa: {best_single['kappa']}")
                print(f"      Composite Improvement: {best_single['composite_improvement']:.2f}%")
                print(f"      Individual Improvements:")
                print(
                    f"        SaCo: {best_single['saco_treatment_mean']:.4f} vs {best_single['saco_vanilla_mean']:.4f} ({best_single['saco_percent_improvement']:+.2f}%, p={best_single['saco_p_value']:.4f})"
                )
                print(
                    f"        Faithfulness Correlation: {best_single['faithfulness_correlation_treatment_mean']:.4f} vs {best_single['faithfulness_correlation_vanilla_mean']:.4f} ({best_single['faithfulness_correlation_percent_improvement']:+.2f}%, p={best_single['faithfulness_correlation_p_value']:.4f})"
                )
                print(
                    f"        Pixel Flipping: {best_single['pixelflipping_treatment_mean']:.4f} vs {best_single['pixelflipping_vanilla_mean']:.4f} ({best_single['pixelflipping_percent_improvement']:+.2f}%, p={best_single['pixelflipping_p_value']:.4f})"
                )
        else:
            print(f"\n  No single-layer configurations found")

        # Best multi-layer overall - Top 3
        multi_layer_df = dataset_df[dataset_df['layer_type'] == 'multi']
        if len(multi_layer_df) > 0:
            top_multi = multi_layer_df.nlargest(3, 'composite_improvement')
            print(f"\n  Top 3 Multi-Layers (across all metrics):")
            for rank, (_, best_multi) in enumerate(top_multi.iterrows(), 1):
                print(f"\n    #{rank}:")
                print(f"      Configuration: {best_multi['experiment_config']}")
                print(f"      Method: {best_multi['gate_construction']}")
                print(f"      Kappa: {best_multi['kappa']}")
                print(f"      Composite Improvement: {best_multi['composite_improvement']:.2f}%")
                print(f"      Individual Improvements:")
                print(
                    f"        SaCo: {best_multi['saco_treatment_mean']:.4f} vs {best_multi['saco_vanilla_mean']:.4f} ({best_multi['saco_percent_improvement']:+.2f}%, p={best_multi['saco_p_value']:.4f})"
                )
                print(
                    f"        Faithfulness Correlation: {best_multi['faithfulness_correlation_treatment_mean']:.4f} vs {best_multi['faithfulness_correlation_vanilla_mean']:.4f} ({best_multi['faithfulness_correlation_percent_improvement']:+.2f}%, p={best_multi['faithfulness_correlation_p_value']:.4f})"
                )
                print(
                    f"        Pixel Flipping: {best_multi['pixelflipping_treatment_mean']:.4f} vs {best_multi['pixelflipping_vanilla_mean']:.4f} ({best_multi['pixelflipping_percent_improvement']:+.2f}%, p={best_multi['pixelflipping_p_value']:.4f})"
                )
        else:
            print(f"\n  No multi-layer configurations found")

        # Best single layer for each method type (if present) - Top 3 per method
        if len(single_layer_df) > 0:
            print(f"\n  Top 3 Single Layers by Method Type:")
            methods = single_layer_df['gate_construction'].unique()

            for method in methods:
                method_df = single_layer_df[single_layer_df['gate_construction'] == method]
                if len(method_df) > 0:
                    top_method = method_df.nlargest(3, 'composite_improvement')
                    print(f"\n    {method}:")
                    for rank, (_, best_method) in enumerate(top_method.iterrows(), 1):
                        print(f"\n      #{rank}:")
                        print(f"        Configuration: {best_method['experiment_config']}")
                        print(f"        Kappa: {best_method['kappa']}")
                        print(f"        Composite Improvement: {best_method['composite_improvement']:.2f}%")
                        print(f"        Individual Improvements:")
                        print(
                            f"          SaCo: {best_method['saco_treatment_mean']:.4f} vs {best_method['saco_vanilla_mean']:.4f} ({best_method['saco_percent_improvement']:+.2f}%, p={best_method['saco_p_value']:.4f})"
                        )
                        print(
                            f"          Faithfulness Correlation: {best_method['faithfulness_correlation_treatment_mean']:.4f} vs {best_method['faithfulness_correlation_vanilla_mean']:.4f} ({best_method['faithfulness_correlation_percent_improvement']:+.2f}%, p={best_method['faithfulness_correlation_p_value']:.4f})"
                        )
                        print(
                            f"          Pixel Flipping: {best_method['pixelflipping_treatment_mean']:.4f} vs {best_method['pixelflipping_vanilla_mean']:.4f} ({best_method['pixelflipping_percent_improvement']:+.2f}%, p={best_method['pixelflipping_p_value']:.4f})"
                        )

    return comparison_df


def save_results(comparison_df: pd.DataFrame, summary_df: pd.DataFrame):
    """Save results to CSV files."""

    comparison_df.to_csv('detailed_sweep_comparison.csv', index=False)
    summary_df.to_csv('sweep_summary_table.csv', index=False)

    print(f"\nResults saved to:")
    print(f"  - detailed_sweep_comparison.csv")
    print(f"  - sweep_summary_table.csv")


def main(sweep_dirs: List[str]):
    """Main analysis function."""

    print("Loading experiment data...")
    sweep_df, vanilla_df = load_all_experiments(sweep_dirs)

    print(f"Loaded {len(sweep_df)} treatment experiments")
    print(f"Loaded {len(vanilla_df)} vanilla baseline experiments")

    print("\nCalculating statistical comparisons...")
    comparison_df = calculate_statistical_comparison(sweep_df, vanilla_df)

    print("\nIdentifying best performers...")
    identify_best_performers(comparison_df)

    print("\nIdentifying best overall performers across all three metrics...")
    comparison_df = identify_best_overall_performers(comparison_df, sweep_df)

    print("\nGenerating detailed results...")
    print_detailed_results(comparison_df)

    print("\nCreating summary table...")
    summary_df = create_summary_table(comparison_df)

    print("\nSUMMARY TABLE:")
    print(summary_df.to_string(index=False))

    save_results(comparison_df, summary_df)


if __name__ == "__main__":
    # Specific sweep directories to analyze
    sweep_dirs = [

        # Test set
        'experiments/feature_gradient_sweep_20260128_114325',
        'experiments/feature_gradient_sweep_20260128_220546/', 
        'experiments/feature_gradient_sweep_20260129_100435/', 

        # Val set
        # 'experiments/feature_gradient_sweep_20260129_103937',
        # 'experiments/feature_gradient_sweep_20260129_173906',

    ]

    print(f"Analyzing sweep directories: {sweep_dirs}")
    main(sweep_dirs)
