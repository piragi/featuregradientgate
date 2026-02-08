"""
Unified Pipeline Module

This is the updated pipeline that uses the unified dataloader system.
It can work with any dataset that has been converted to the standard format.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

from gradcamfaith.data import io_utils
from gradcamfaith.core.config import FileConfig, PipelineConfig
from gradcamfaith.core.types import (AttributionDataBundle, AttributionOutputPaths, ClassificationPrediction, ClassificationResult)
from gradcamfaith.data.dataset_config import DatasetConfig, get_dataset_config
from gradcamfaith.experiments.faithfulness import evaluate_and_report_faithfulness
from gradcamfaith.experiments.saco import run_binned_attribution_analysis
from gradcamfaith.data.setup import convert_dataset
from gradcamfaith.core.attribution import compute_attribution
from gradcamfaith.data.dataloader import create_dataloader, get_single_image_loader

# Compatibility re-exports — canonical source is now in gradcamfaith.models.*
from gradcamfaith.models.load import load_model_for_dataset  # noqa: F401
from gradcamfaith.models.sae_resources import load_steering_resources  # noqa: F401


def prepare_dataset_if_needed(
    dataset_name: str, source_path: Path, prepared_path: Path, force_prepare: bool = False, **converter_kwargs
) -> Path:
    """
    Prepare dataset if not already prepared.

    Args:
        dataset_name: Name of the dataset
        source_path: Path to raw dataset
        prepared_path: Path where prepared dataset should be
        force_prepare: If True, force re-preparation even if exists
        **converter_kwargs: Additional arguments for converter

    Returns:
        Path to prepared dataset
    """
    metadata_file = prepared_path / "dataset_metadata.json"

    if not force_prepare and metadata_file.exists():
        print(f"Dataset already prepared at {prepared_path}")
        return prepared_path

    print(f"Preparing {dataset_name} dataset...")
    print("Images will be preprocessed to 224x224")
    convert_dataset(dataset_name=dataset_name, source_path=source_path, output_path=prepared_path, **converter_kwargs)

    return prepared_path


def classify_single_image(
    config: PipelineConfig,
    dataset_config: DatasetConfig,
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    true_label: Optional[str] = None
) -> ClassificationResult:
    """
    Classify a single image using the unified system.
    """
    cache_path = io_utils.build_cache_path(config.file.cache_dir, image_path, f"_classification_{dataset_config.name}")

    # Try to load from cache
    loaded_result = io_utils.try_load_from_cache(cache_path)
    if config.file.use_cached_perturbed and loaded_result:
        return loaded_result

    # Load and preprocess image
    # Check if we're using CLIP (config might be None in some cases)
    use_clip = config and config.classify.use_clip
    input_tensor = get_single_image_loader(image_path, dataset_config, use_clip=use_clip)
    input_tensor = input_tensor.to(device)

    # Get prediction
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        predicted_idx = int(torch.argmax(probabilities, dim=-1).item())

    current_prediction = ClassificationPrediction(
        predicted_class_label=dataset_config.idx_to_class[predicted_idx],
        predicted_class_idx=predicted_idx,
        confidence=float(probabilities[0, predicted_idx].item()),
        probabilities=probabilities[0].tolist()
    )

    result = ClassificationResult(
        image_path=image_path, prediction=current_prediction, true_label=true_label, attribution_paths=None
    )

    # Cache the result
    io_utils.save_to_cache(cache_path, result)

    return result


def save_attribution_bundle_to_files(
    image_stem: str, attribution_bundle: AttributionDataBundle, file_config: FileConfig
) -> AttributionOutputPaths:
    """Save attribution bundle contents to .npy files."""

    # Ensure attribution directory exists
    io_utils.ensure_directories([file_config.attribution_dir])

    attribution_path = file_config.attribution_dir / f"{image_stem}_attribution.npy"
    raw_attribution_path = file_config.attribution_dir / f"{image_stem}_raw_attribution.npy"

    # Save positive attribution
    np.save(attribution_path, attribution_bundle.positive_attribution)
    # Save raw attribution
    np.save(raw_attribution_path, attribution_bundle.raw_attribution)

    return AttributionOutputPaths(
        attribution_path=attribution_path,
        raw_attribution_path=raw_attribution_path,
    )


def classify_explain_single_image(
    config: PipelineConfig,
    dataset_config: DatasetConfig,
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    steering_resources: Optional[Dict[int, Dict[str, Any]]],
    true_label: Optional[str] = None,
    clip_classifier: Optional[Any] = None,  # Pre-created CLIP classifier
) -> Tuple[ClassificationResult, Dict[int, Dict[str, Any]]]:
    """
    Classify a single image AND generate explanations using unified system.
    Returns classification result and debug info (if debug mode is enabled).
    """
    cache_path = io_utils.build_cache_path(
        config.file.cache_dir, image_path, f"_classification_explained_{dataset_config.name}"
    )

    # Try to load from cache
    loaded_result = io_utils.try_load_from_cache(cache_path)
    if config.file.use_cached_original and loaded_result:
        if loaded_result.attribution_paths is not None:
            return loaded_result, {}  # No debug info from cache

    # Load and preprocess image
    # Check if we're using CLIP (config might be None in some cases)
    use_clip = config and config.classify.use_clip
    input_tensor = get_single_image_loader(image_path, dataset_config, use_clip=use_clip)
    input_tensor = input_tensor.to(device)

    raw_attribution_result_dict = compute_attribution(
        model_prisma=model,
        input_tensor=input_tensor,
        config=config,
        idx_to_class=dataset_config.idx_to_class,  # Pass dataset-specific class mapping
        device=device,
        steering_resources=steering_resources,
        enable_feature_gradients=config.classify.boosting.enable_feature_gradients,
        feature_gradient_layers=config.classify.boosting.feature_gradient_layers
        if config.classify.boosting.enable_feature_gradients else [],
        clip_classifier=clip_classifier,
        debug=getattr(config.classify.boosting, 'debug_mode', False),
    )

    # Extract raw attribution and debug info
    raw_attr = raw_attribution_result_dict.get("raw_attribution", np.array([]))
    debug_info = raw_attribution_result_dict.get("debug_info", {})

    # Create prediction
    prediction_data = raw_attribution_result_dict["predictions"]
    current_prediction = ClassificationPrediction(
        predicted_class_label=dataset_config.idx_to_class[prediction_data["predicted_class_idx"]],
        predicted_class_idx=prediction_data["predicted_class_idx"],
        confidence=float(prediction_data["probabilities"][prediction_data["predicted_class_idx"]]),
        probabilities=prediction_data["probabilities"]
    )

    # Create attribution bundle
    attribution_bundle = AttributionDataBundle(
        positive_attribution=raw_attribution_result_dict["attribution_positive"],
        raw_attribution=raw_attr,
    )

    # Save attribution bundle
    saved_attribution_paths = save_attribution_bundle_to_files(image_path.stem, attribution_bundle, config.file)

    # Create final result with cached tensors for efficient faithfulness evaluation
    final_result = ClassificationResult(
        image_path=image_path,
        prediction=current_prediction,
        true_label=true_label,
        attribution_paths=saved_attribution_paths,
        _cached_tensor=input_tensor.cpu().numpy()[0],  # Cache preprocessed tensor (C, H, W)
        _cached_raw_attribution=raw_attr  # Cache raw attribution
    )

    # Cache the result
    io_utils.save_to_cache(cache_path, final_result)

    return final_result, debug_info


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
    # Check if we're using CLIP for this dataset
    config.classify.use_clip if hasattr(config.classify, 'use_clip') else False
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

    # Extract SaCo scores (overall and per-class)
    saco_results = {}
    if saco_analysis and "faithfulness_correctness" in saco_analysis:
        fc_df = saco_analysis["faithfulness_correctness"]

        # Overall statistics
        saco_results['mean'] = fc_df["saco_score"].mean()
        saco_results['std'] = fc_df["saco_score"].std()
        saco_results['n_samples'] = len(fc_df)

        # Per-class breakdown
        per_class_stats = fc_df.groupby('true_class')['saco_score'].agg(['mean', 'std', 'count'])
        saco_results['per_class'] = per_class_stats.to_dict('index')

        # Also include correctness breakdown
        correctness_stats = fc_df.groupby('is_correct')['saco_score'].agg(['mean', 'std', 'count'])
        saco_results['by_correctness'] = correctness_stats.to_dict('index')

    print("\nPipeline complete!")

    return results, saco_results
