"""
Unified Pipeline — experiment orchestrator.

Prepares data, classifies+explains each image, then runs faithfulness and
SaCo analysis.  Debug I/O is isolated in private helpers at the bottom.
"""

import json
import logging
from datetime import datetime
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
from gradcamfaith.experiments.faithfulness import compute_faithfulness
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

    # Classify and explain each image
    results = []
    debug_mode = config.classify.boosting.debug_mode
    debug_data_per_layer = {}

    for _, (image_path, true_label_idx) in enumerate(tqdm(image_data, desc="Classifying & Explaining")):
        try:
            true_label = dataset_config.idx_to_class.get(true_label_idx)
            result, debug_info = classify_explain_single_image(
                config=config,
                dataset_config=dataset_config,
                image_path=image_path,
                model=model,
                device=device,
                steering_resources=steering_resources,
                true_label=true_label,
                clip_classifier=clip_classifier,
            )
            results.append(result)
            if debug_mode and debug_info:
                _accumulate_debug_info(debug_data_per_layer, debug_info)
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue

    # Save debug outputs (classification CSV + gate/attribution .npz)
    _save_debug_outputs(config, results, debug_data_per_layer)

    # For CLIP models, wrap the classifier for both faithfulness and attribution analysis
    model_for_analysis = model
    if clip_classifier is not None:
        from gradcamfaith.models.clip_classifier import CLIPModelWrapper
        model_for_analysis = CLIPModelWrapper(clip_classifier)
        print("Using CLIP wrapper for analysis")

    # Run faithfulness evaluation (compute only — we save a unified file later)
    faithfulness_data = {}
    if config.classify.analysis:
        print("Running faithfulness evaluation...")
        try:
            faithfulness_data = compute_faithfulness(
                config, model_for_analysis, device, results
            )
        except Exception as e:
            print(f"Error in faithfulness evaluation: {e}")

    # Run SaCo attribution analysis
    print("Running SaCo attribution analysis...")
    saco_analysis = run_binned_attribution_analysis(config, model_for_analysis, results, device)

    # Build unified metrics and save
    unified_metrics = _build_unified_metrics(faithfulness_data, saco_analysis)
    _save_unified_faithfulness_stats(config, faithfulness_data, saco_analysis)

    print("\nPipeline complete!")
    return results, unified_metrics


# ---------------------------------------------------------------------------
# Unified metrics helpers
# ---------------------------------------------------------------------------

def _build_unified_metrics(faithfulness_data, saco_analysis):
    """Build unified metric summary dict for results.json.

    Combines SaCo summary with FC/PF summaries into a single ``metrics`` dict.
    """
    saco_summary = extract_saco_summary(saco_analysis)

    metrics: Dict[str, Any] = {'SaCo': saco_summary}

    # Add FC and PF summaries from faithfulness data
    for metric_name in ('FaithfulnessCorrelation', 'PixelFlipping'):
        metric_data = faithfulness_data.get('metrics', {}).get(metric_name, {})
        overall = metric_data.get('overall', {})
        if overall:
            metrics[metric_name] = {
                'mean': overall.get('mean', 0.0),
                'std': overall.get('std', 0.0),
                'n_samples': overall.get('count', 0),
                'median': overall.get('median', 0.0),
                'min': overall.get('min', 0.0),
                'max': overall.get('max', 0.0),
            }

    return metrics


def _save_unified_faithfulness_stats(config, faithfulness_data, saco_analysis):
    """Save unified faithfulness stats with all 3 metrics + per-image metadata."""
    if not saco_analysis and not faithfulness_data:
        return

    # Start from faithfulness data (has FC + PF with mean_scores, overall, by_class)
    unified = dict(faithfulness_data) if faithfulness_data else {'dataset': config.file.dataset_name, 'metrics': {}}

    # Inject SaCo per-image scores into metrics
    fc_df = saco_analysis.get('faithfulness_correctness')
    if fc_df is not None and not fc_df.empty:
        saco_scores = fc_df['saco_score'].values
        unified.setdefault('metrics', {})['SaCo'] = {
            'overall': {
                'count': len(saco_scores),
                'mean': float(np.nanmean(saco_scores)),
                'median': float(np.nanmedian(saco_scores)),
                'std': float(np.nanstd(saco_scores)),
                'min': float(np.nanmin(saco_scores)) if len(saco_scores) > 0 else 0.0,
                'max': float(np.nanmax(saco_scores)) if len(saco_scores) > 0 else 0.0,
            },
            'mean_scores': saco_scores.tolist(),
            'n_trials': 1,
        }

        # Add per-image classification metadata
        unified['images'] = fc_df[
            ['filename', 'predicted_class', 'predicted_idx', 'true_class', 'is_correct', 'confidence']
        ].to_dict('records')

    # Remove class_labels (replaced by richer 'images' array)
    unified.pop('class_labels', None)

    # Save
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    json_path = config.file.output_dir / f"faithfulness_stats{config.file.output_suffix}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(unified, f, indent=2)
    print(f"\nUnified faithfulness statistics saved to {json_path}")


# ---------------------------------------------------------------------------
# Debug I/O helpers (all output goes to output_dir/debug/)
# ---------------------------------------------------------------------------

def _accumulate_debug_info(debug_data_per_layer, debug_info):
    """Accumulate per-image debug data into per-layer buffers.

    Collects only what case_studies.py needs:
    - patch_attribution_deltas (from outer debug_info)
    - sparse_indices, sparse_activations, sparse_contributions (from feature_gating)
    """
    for layer_idx, layer_debug in debug_info.items():
        feature_gating = layer_debug.get('feature_gating', {})
        if layer_idx not in debug_data_per_layer:
            debug_data_per_layer[layer_idx] = {
                'patch_attribution_deltas': [],
                'sparse_indices': [],
                'sparse_activations': [],
                'sparse_contributions': [],
            }
        buf = debug_data_per_layer[layer_idx]
        buf['patch_attribution_deltas'].append(
            layer_debug.get('patch_attribution_deltas', np.array([])))
        buf['sparse_indices'].append(
            feature_gating.get('sparse_features_indices', []))
        buf['sparse_activations'].append(
            feature_gating.get('sparse_features_activations', []))
        buf['sparse_contributions'].append(
            feature_gating.get('sparse_features_contributions', []))


def _save_debug_outputs(config, results, debug_data_per_layer):
    """Save all debug outputs to output_dir/debug/. Gated by debug_mode."""
    if not config.classify.boosting.debug_mode:
        return
    if not results and not debug_data_per_layer:
        return

    debug_dir = config.file.output_dir / "debug"
    debug_dir.mkdir(exist_ok=True, parents=True)

    if results:
        csv_path = debug_dir / "classification_results.csv"
        io_utils.save_classification_results_to_csv(results, csv_path)

    for layer_idx, layer_data in debug_data_per_layer.items():
        np.savez_compressed(
            debug_dir / f"layer_{layer_idx}_debug.npz",
            patch_attribution_deltas=np.array(layer_data['patch_attribution_deltas']),
            sparse_indices=np.array(layer_data['sparse_indices'], dtype=object),
            sparse_activations=np.array(layer_data['sparse_activations'], dtype=object),
            sparse_contributions=np.array(layer_data['sparse_contributions'], dtype=object),
        )
