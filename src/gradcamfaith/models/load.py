"""
Model loading for the attribution pipeline.

Handles loading ViT models (both standard and CLIP-based) with
appropriate weight conversion and classifier setup.
"""

from pathlib import Path
from typing import Optional

import torch

from gradcamfaith.core.config import PipelineConfig
from gradcamfaith.data.dataset_config import DatasetConfig


def _validate_model_interface(model, model_type: str):
    """Validate that the loaded model has the expected HookedViT interface."""
    for attr_path in ("cfg", "cfg.n_layers", "cfg.patch_size"):
        obj = model
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                raise ValueError(
                    f"{model_type} model missing expected attribute '{attr_path}'. "
                    f"Got type {type(model).__name__}."
                )


def load_model_for_dataset(
    dataset_config: DatasetConfig, device: torch.device, config: Optional[PipelineConfig] = None
):
    """
    Load the appropriate model and CLIP classifier for a given dataset configuration.

    Args:
        dataset_config: Configuration for the dataset
        device: Device to load the model on
        config: Optional pipeline config for CLIP settings

    Returns:
        Tuple of (model, clip_classifier) where clip_classifier is None for non-CLIP models
    """
    # Check if we should use CLIP for this dataset
    use_clip = (config and config.classify.use_clip) or dataset_config.name == "waterbirds"

    if use_clip:
        from vit_prisma.models.model_loader import load_hooked_model

        print(f"Loading CLIP as HookedViT for {dataset_config.name}")

        # Use vit_prisma's load_hooked_model to get CLIP as HookedViT
        # This automatically converts CLIP weights to HookedViT format
        clip_model_name = config.classify.clip_model_name if config else "openai/clip-vit-base-patch32"

        model = load_hooked_model(clip_model_name, dtype=torch.float32, device=str(device))
        # Ensure model is on the correct device
        model = model.to(device)
        model.eval()

        print(f"CLIP loaded as HookedViT")

        # Create CLIP classifier for this model
        from gradcamfaith.models.clip_classifier import create_clip_classifier_for_waterbirds
        print("Creating CLIP classifier...")
        clip_classifier = create_clip_classifier_for_waterbirds(
            vision_model=model,
            device=device,
            clip_model_name=clip_model_name,
            custom_prompts=config.classify.clip_text_prompts if config.classify.clip_text_prompts else None
        )

        _validate_model_interface(model, "CLIP")
        return model, clip_classifier

    # Original ViT loading code
    from vit_prisma.models.base_vit import HookedSAEViT
    from vit_prisma.models.weight_conversion import convert_timm_weights

    # Create model with correct number of classes (don't load ImageNet weights)
    model = HookedSAEViT.from_pretrained("vit_base_patch16_224", load_pretrained_model=False)

    # Update the config and recreate the head with the correct number of classes
    model.cfg.n_classes = dataset_config.num_classes
    from vit_prisma.models.layers.head import Head
    model.head = Head(model.cfg)

    # Load checkpoint if available
    checkpoint_path = Path(dataset_config.model_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at '{checkpoint_path}'. ")

    print(f"Loading model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict'].copy()
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict'].copy()
    else:
        state_dict = checkpoint

    # Rename linear head if needed for compatibility
    if 'lin_head.weight' in state_dict:
        state_dict['head.weight'] = state_dict.pop('lin_head.weight')
    if 'lin_head.bias' in state_dict:
        state_dict['head.bias'] = state_dict.pop('lin_head.bias')

    # Convert weights to the correct format for the model and load them
    converted_weights = convert_timm_weights(state_dict, model.cfg)
    model.load_state_dict(converted_weights)

    model.to(device).eval()
    _validate_model_interface(model, "ViT")
    return model, None  # No CLIP classifier for regular ViT models
