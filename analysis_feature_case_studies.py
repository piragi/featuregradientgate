"""Compatibility wrapper — canonical source is gradcamfaith.experiments.case_studies"""
from gradcamfaith.experiments.case_studies import *  # noqa: F401,F403
from gradcamfaith.experiments.case_studies import (  # noqa: F401
    _extract_sae_activations,
    extract_sae_activations_if_needed,
    load_and_preprocess_image,
    load_faithfulness_results,
    load_debug_data,
    load_activation_data,
    build_feature_activation_index,
    compute_composite_improvement,
    find_dominant_features_in_image,
    extract_case_studies,
    visualize_case_study,
    save_case_study_individual_images,
    get_image_path,
    load_attribution,
    run_case_study_analysis,
)
from pathlib import Path


if __name__ == "__main__":
    # Configuration
    experiment_path = Path("./experiments/feature_gradient_sweep_20260128_220546/") #TODO: Fill out the experiment folder
    experiment_config = "layers_2_3_4_kappa_None_combined_clamp_None"
    layers = [2,3,4]

    # Extract validation set activations for prototypes if not already done
    # This will skip extraction if files already exist
    validation_activations_path = extract_sae_activations_if_needed(
        dataset_name='covidquex',
        layers=layers,
        split='val',
        output_dir=Path("./sae_activations/covidquex_val"),
        subset_size=None,  # Process all validation images
        use_clip=True
    )

    # Run analysis for improved images
    case_studies_improved = run_case_study_analysis(
        experiment_path=experiment_path,
        experiment_config=experiment_config,
        layers=layers,
        n_top_images=500,
        n_patches_per_image=10,
        n_case_visualizations=1000,
        n_prototypes=20,
        validation_activations_path=validation_activations_path,
        mode='improved',
        max_prototype_images=20000  # Limit to save RAM (layer 8 is 5.9GB for full 50k)
    )

    # Run analysis for degraded images
    case_studies_degraded = run_case_study_analysis(
        experiment_path=experiment_path,
        experiment_config=experiment_config,
        layers=layers,
        n_top_images=100,
        n_patches_per_image=5,
        n_case_visualizations=500,
        n_prototypes=20,
        validation_activations_path=validation_activations_path,
        mode='degraded',
        max_prototype_images=5000  # Limit to save RAM (layer 8 is 5.9GB for full 50k)
    )
