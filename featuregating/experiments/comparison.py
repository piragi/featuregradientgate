#!/usr/bin/env python3
"""
Comprehensive comparison of best performer configurations vs vanilla baselines.
Analyzes SaCo, Faithfulness Correlation, and PixelFlipping metrics with statistical analysis.

Script usage:
    uv run python -m featuregating.experiments.comparison
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

warnings.filterwarnings('ignore')


def _discover_default_sweep_dirs(base_dir: Path = Path("data/runs")) -> List[str]:
    """Discover recent sweep directories under ``data/runs``."""
    if not base_dir.exists():
        return []

    candidates = sorted(
        [p for p in base_dir.glob("feature_gradient_sweep_*") if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    return [str(p) for p in candidates[:3]]


def _validate_sweep_dirs(sweep_dirs: List[str]) -> List[Path]:
    """Validate sweep directory inputs and provide actionable errors."""
    if not sweep_dirs:
        raise ValueError(
            "No sweep directories provided.\n"
            "Provide at least one directory under data/runs/feature_gradient_sweep_<timestamp>/."
        )

    resolved: List[Path] = []
    missing: List[str] = []
    not_dirs: List[str] = []

    for raw in sweep_dirs:
        p = Path(raw)
        if not p.exists():
            missing.append(str(p))
            continue
        if not p.is_dir():
            not_dirs.append(str(p))
            continue
        resolved.append(p)

    if missing:
        raise FileNotFoundError(
            "Some sweep directories do not exist:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nRun a sweep first via `uv run python -m featuregating.experiments.sweep`."
        )
    if not_dirs:
        raise NotADirectoryError(
            "Some sweep paths are not directories:\n"
            + "\n".join(f"  - {m}" for m in not_dirs)
        )

    return resolved


def load_experiment_data(experiment_path: Path) -> Dict[str, Any]:
    """Load results and config for a single experiment."""
    results_file = experiment_path / "results.json"
    config_file = experiment_path / "experiment_config.json"

    with open(results_file) as f:
        results = json.load(f)

    with open(config_file) as f:
        config = json.load(f)

    return {'results': results, 'config': config, 'experiment_name': experiment_path.name}


def extract_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from experiment data.

    Reads all 3 metrics from the unified ``results.json`` ``metrics`` key.
    """
    metrics = {}
    results_metrics = data['results']['metrics']

    # SaCo
    saco = results_metrics['SaCo']
    metrics['saco_mean'] = saco['mean']
    metrics['saco_std'] = saco['std']
    metrics['saco_n'] = saco['n_samples']

    # FaithfulnessCorrelation and PixelFlipping
    for metric_key, prefix in [('FaithfulnessCorrelation', 'faithfulness_correlation'),
                                ('PixelFlipping', 'pixelflipping')]:
        source = results_metrics[metric_key]
        metrics[f'{prefix}_mean'] = source['mean']
        metrics[f'{prefix}_std'] = source['std']
        metrics[f'{prefix}_n'] = source['n_samples']

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

    sweep_paths = _validate_sweep_dirs(sweep_dirs)

    sweep_data = []
    vanilla_data = []

    # Load all experiments from all sweep directories
    for sweep_dir in sweep_paths:
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


def _average_percent_improvement(row: pd.Series, metrics: List[str] = None) -> float:
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


def _print_metric_improvements(row: pd.Series, indent: str = "        "):
    """Print per-metric treatment vs vanilla comparison for a single experiment row."""
    metrics = [
        ('SaCo', 'saco'),
        ('Faithfulness Correlation', 'faithfulness_correlation'),
        ('Pixel Flipping', 'pixelflipping'),
    ]
    for label, key in metrics:
        print(
            f"{indent}{label}: {row[f'{key}_treatment_mean']:.4f} vs "
            f"{row[f'{key}_vanilla_mean']:.4f} "
            f"({row[f'{key}_percent_improvement']:+.2f}%, "
            f"p={row[f'{key}_p_value']:.4f})"
        )


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
    comparison_df['composite_improvement'] = comparison_df.apply(_average_percent_improvement, axis=1)

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
                _print_metric_improvements(best_single)
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
                _print_metric_improvements(best_multi)
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
                        _print_metric_improvements(best_method, indent="          ")

    return comparison_df


def save_results(comparison_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path = Path(".")):
    """Save results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_dir / 'detailed_sweep_comparison.csv', index=False)
    summary_df.to_csv(output_dir / 'sweep_summary_table.csv', index=False)

    print(f"\nResults saved to:")
    print(f"  - {output_dir / 'detailed_sweep_comparison.csv'}")
    print(f"  - {output_dir / 'sweep_summary_table.csv'}")


def main(sweep_dirs: List[str]):
    """Main analysis function for comparison outputs."""
    print("=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)
    print("Expected structure per experiment:")
    print("  <sweep>/<dataset>/<experiment>/results.json")
    print("  <sweep>/<dataset>/<experiment>/experiment_config.json")
    print("Will write:")
    print("  ./detailed_sweep_comparison.csv")
    print("  ./sweep_summary_table.csv")
    print()

    print("Loading experiment data...")
    sweep_df, vanilla_df = load_all_experiments(sweep_dirs)

    print(f"Loaded {len(sweep_df)} treatment experiments")
    print(f"Loaded {len(vanilla_df)} vanilla baseline experiments")

    if sweep_df.empty and vanilla_df.empty:
        raise ValueError(
            "No experiments were loaded from the provided sweep directories.\n"
            "Check that each sweep folder contains per-dataset experiment subdirectories."
        )
    if vanilla_df.empty:
        raise ValueError(
            "No vanilla baseline experiments were found.\n"
            "Expected at least one '<sweep>/<dataset>/vanilla/' directory."
        )

    print("\nCalculating statistical comparisons...")
    comparison_df = calculate_statistical_comparison(sweep_df, vanilla_df)
    if comparison_df.empty:
        raise ValueError(
            "No comparable rows were produced.\n"
            "Check that treatment experiments and vanilla baselines exist for matching datasets."
        )

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
    # Config-first runner: edit this list if you want explicit sweep folders.
    sweep_dirs = _discover_default_sweep_dirs()
    if not sweep_dirs:
        raise FileNotFoundError(
            "No default sweep directories found under data/runs/.\n"
            "Run `uv run python -m featuregating.experiments.sweep` first, or edit "
            "`sweep_dirs` in featuregating/experiments/comparison.py."
        )

    print(f"Analyzing sweep directories: {sweep_dirs}\n")
    main(sweep_dirs)
