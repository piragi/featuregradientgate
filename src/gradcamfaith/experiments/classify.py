"""Per-image classification and attribution.

Contains the per-image classify+explain logic and attribution file I/O.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from gradcamfaith.core.attribution import compute_attribution
from gradcamfaith.core.config import FileConfig, PipelineConfig
from gradcamfaith.core.types import (
    AttributionDataBundle,
    AttributionOutputPaths,
    ClassificationPrediction,
    ClassificationResult,
)
from gradcamfaith.data import io_utils
from gradcamfaith.data.dataloader import get_single_image_loader
from gradcamfaith.data.dataset_config import DatasetConfig


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
