"""
Unified Pipeline — experiment orchestrator.

Prepares data, loops over images, accumulates debug info, saves CSV results,
runs faithfulness evaluation, and runs SaCo attribution analysis.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from gradcamfaith.core.config import PipelineConfig
from gradcamfaith.core.types import ClassificationResult
from gradcamfaith.data import io_utils
from gradcamfaith.data.dataloader import create_dataloader
from gradcamfaith.data.dataset_config import get_dataset_config
from gradcamfaith.data.setup import prepare_dataset_if_needed
from gradcamfaith.experiments.classify import classify_explain_single_image
from gradcamfaith.experiments.faithfulness import evaluate_and_report_faithfulness
from gradcamfaith.experiments.saco import extract_saco_summary, run_binned_attribution_analysis

# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)


def run_unified_pipeline(
    config: PipelineConfig,
    dataset_name: str,
    source_data_path: Path,
    model: torch.nn.Module,
    steering_resources: Dict[int, Dict[str, Any]],
    clip_classifier: Optional[Any] = None,
    prepared_data_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    force_prepare: bool = False,
    subset_size: Optional[int] = None,
    random_seed: Optional[int] = None
) -> Tuple[List[ClassificationResult], Dict[str, Any]]:
    """
    Run the unified pipeline for any supported dataset.

    Args:
        config: Pipeline configuration
        dataset_name: Name of the dataset ('covidquex' or 'hyperkvasir')
        source_data_path: Path to the source dataset
        model: Pre-loaded model
        steering_resources: Pre-loaded SAE resources
        clip_classifier: Pre-loaded CLIP classifier (None for non-CLIP models)
        prepared_data_path: Path for prepared data (default: ./data/{dataset_name}_unified)
        device: Device to use
        force_prepare: Force re-preparation of dataset
        subset_size: If specified, only use this many random images
        random_seed: Random seed for reproducible subset selection

    Returns:
        List of classification results
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get dataset configuration
    dataset_config = get_dataset_config(dataset_name)
    print(f"Loading configuration for {dataset_name} dataset")

    # Ensure all required directories exist
    io_utils.ensure_directories(config.directories)

    # Prepare dataset if needed
    if prepared_data_path is None:
        prepared_data_path = Path(f"./data/{dataset_name}_unified")

    prepared_path = prepare_dataset_if_needed(
        dataset_name=dataset_name,
        source_path=source_data_path,
        prepared_path=prepared_data_path,
        force_prepare=force_prepare
    )

    # Create dataloader
    print(f"Creating dataloader from {prepared_path}")
    dataset_loader = create_dataloader(dataset_name=dataset_name, data_path=prepared_path)

    # Use pre-loaded model and steering resources
    print(f"\nUsing pre-loaded model for {dataset_name}")
    print(f"Using pre-loaded SAE resources for {dataset_name}")

    # Process images from the specified split (mode)
    split_to_use = config.file.current_mode if config.file.current_mode in ['train', 'val', 'test', 'dev'] else 'test'

    image_data = list(dataset_loader.get_numeric_samples(split_to_use))
    total_samples = len(image_data)

    # Apply subset if requested
    if subset_size is not None and subset_size < total_samples:
        import random
        if random_seed is not None:
            random.seed(random_seed)
        image_data = random.sample(image_data, subset_size)
        print(f"\nProcessing {len(image_data)} randomly selected {split_to_use} images (subset of {total_samples})")
    else:
        print(f"\nProcessing {total_samples} {split_to_use} images")

    # Use pre-loaded CLIP classifier (created once per dataset)
    if clip_classifier is not None:
        print("Using pre-loaded CLIP classifier")

    # Classify and explain
    results = []

    # Initialize debug data accumulators if debug mode is enabled
    debug_mode = config.classify.boosting.debug_mode
    debug_data_per_layer = {}  # {layer_idx: {key: list of values per image}}

    for _, (image_path, true_label_idx) in enumerate(tqdm(image_data, desc="Classifying & Explaining")):
        try:
            # Convert label index to label name (handle unlabeled = -1)
            true_label = dataset_config.idx_to_class.get(true_label_idx)

            result, debug_info = classify_explain_single_image(
                config=config,
                dataset_config=dataset_config,
                image_path=image_path,
                model=model,
                device=device,
                steering_resources=steering_resources,
                true_label=true_label,
                clip_classifier=clip_classifier  # Pass pre-created classifier
            )
            results.append(result)

            # Accumulate debug data
            if debug_mode and debug_info:
                for layer_idx, layer_debug in debug_info.items():
                    # Extract feature_gating nested dict
                    feature_gating = layer_debug.get('feature_gating', {})

                    if layer_idx not in debug_data_per_layer:
                        debug_data_per_layer[layer_idx] = {
                            'gate_values': [],
                            'patch_attribution_deltas': [],
                            'contribution_sum': [],
                            'total_contribution_magnitude': [],
                        }
                    # Skip sparse feature accumulation to save RAM
                    # (uncomment if needed for detailed per-feature analysis)
                    # debug_data_per_layer[layer_idx]['sparse_indices'].append(
                    #     feature_gating.get('sparse_features_indices', [])
                    # )
                    # debug_data_per_layer[layer_idx]['sparse_activations'].append(
                    #     feature_gating.get('sparse_features_activations', [])
                    # )
                    # debug_data_per_layer[layer_idx]['sparse_gradients'].append(
                    #     feature_gating.get('sparse_features_gradients', [])
                    # )
                    # debug_data_per_layer[layer_idx]['sparse_contributions'].append(
                    #     feature_gating.get('sparse_features_contributions', [])
                    # )
                    debug_data_per_layer[layer_idx]['gate_values'].append(feature_gating.get('gate_values', np.array([])))
                    debug_data_per_layer[layer_idx]['patch_attribution_deltas'].append(
                        layer_debug.get('patch_attribution_deltas', np.array([]))
                    )
                    debug_data_per_layer[layer_idx]['contribution_sum'].append(
                        feature_gating.get('contribution_sum', np.array([]))
                    )
                    debug_data_per_layer[layer_idx]['total_contribution_magnitude'].append(
                        feature_gating.get('total_contribution_magnitude', np.array([]))
                    )

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue

    # Save results
    if results:
        csv_path = config.file.output_dir / f"results_{dataset_name}_unified.csv"
        io_utils.save_classification_results_to_csv(results, csv_path)
        print(f"Results saved to {csv_path}")

    # Save debug data if collected
    if debug_mode and debug_data_per_layer:
        debug_dir = config.file.output_dir / "debug_data"
        debug_dir.mkdir(exist_ok=True, parents=True)

        for layer_idx, layer_data in debug_data_per_layer.items():
            debug_file = debug_dir / f"layer_{layer_idx}_debug.npz"

            # Convert lists to numpy arrays
            gate_values_array = np.array(layer_data['gate_values'])
            patch_attribution_deltas_array = np.array(layer_data['patch_attribution_deltas'])
            contribution_sum_array = np.array(layer_data['contribution_sum'])
            total_contribution_magnitude_array = np.array(layer_data['total_contribution_magnitude'])

            # Save with numpy - sparse data skipped to save RAM
            np.savez_compressed(
                debug_file,
                gate_values=gate_values_array,
                patch_attribution_deltas=patch_attribution_deltas_array,
                contribution_sum=contribution_sum_array,
                total_contribution_magnitude=total_contribution_magnitude_array
            )

    # For CLIP models, wrap the classifier for both faithfulness and attribution analysis
    model_for_analysis = model
    if clip_classifier is not None:
        from gradcamfaith.models.clip_classifier import CLIPModelWrapper
        model_for_analysis = CLIPModelWrapper(clip_classifier)
        print("Using CLIP wrapper for analysis")

    # Run faithfulness evaluation if configured
    if config.classify.analysis:
        print("Running faithfulness evaluation...")
        try:
            evaluate_and_report_faithfulness(
                config, model_for_analysis, device, results
            )
        except Exception as e:
            print(f"Error in faithfulness evaluation: {e}")

    # Run attribution analysis
    # n_bins is automatically determined from config based on model architecture (B-16 vs B-32)
    print("Running SaCo attribution analysis...")
    saco_analysis = run_binned_attribution_analysis(config, model_for_analysis, results, device)

    # Extract SaCo summary statistics
    saco_results = extract_saco_summary(saco_analysis)

    print("\nPipeline complete!")

    return results, saco_results
