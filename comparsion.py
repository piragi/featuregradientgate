"""Compatibility wrapper — canonical source is gradcamfaith.experiments.comparison"""
from gradcamfaith.experiments.comparison import *  # noqa: F401,F403
from gradcamfaith.experiments.comparison import (  # noqa: F401
    cohens_d,
    load_experiment_data,
    extract_metrics,
    get_experiment_info,
    load_all_experiments,
    calculate_statistical_comparison,
    interpret_effect_size,
    print_detailed_results,
    create_summary_table,
    identify_best_performers,
    classify_layer_type,
    calculate_composite_improvement,
    identify_best_overall_performers,
    save_results,
    main,
)


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
