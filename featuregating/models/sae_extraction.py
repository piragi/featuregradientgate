"""
SAE activation extraction for feature prototype analysis.

Runs forward passes through a model with hooks to capture residual stream
activations, encodes them with SAEs, and saves sparse activations to disk.
Supports checkpointing for large datasets.
"""

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm


def extract_sae_activations_if_needed(
    dataset_name: str,
    layers: List[int],
    split: str = 'val',
    output_dir: Optional[Path] = None,
    subset_size: Optional[int] = None,
    use_clip: bool = True
) -> Path:
    """
    Extract SAE activations if they don't already exist.

    Returns path to the activations directory (whether extracted or already existing).
    """
    if output_dir is None:
        output_dir = Path(f"./data/sae_activations/{dataset_name}_{split}")

    # Check if activations already exist
    debug_dir = output_dir / "debug"
    metadata_file = output_dir / "extraction_metadata.json"

    if debug_dir.exists() and metadata_file.exists():
        all_exist = all((debug_dir / f"layer_{layer_idx}_activations.npz").exists() for layer_idx in layers)

        if all_exist:
            print(f"Activation files already exist in {output_dir}")
            print("Skipping extraction. Delete the directory to force re-extraction.")
            return output_dir

    # Files don't exist, run extraction
    print(f"Activation files not found. Extracting from {split} split...")
    return _extract_sae_activations(
        dataset_name=dataset_name,
        layers=layers,
        split=split,
        output_dir=output_dir,
        subset_size=subset_size,
        use_clip=use_clip
    )


def _extract_sae_activations(
    dataset_name: str, layers: List[int], split: str, output_dir: Path, subset_size: Optional[int], use_clip: bool
) -> Path:
    """
    Extract SAE activations by running forward passes with residual hooks.

    Loads the model and SAEs, processes each image, encodes residuals into
    sparse feature activations, and saves to NPZ with periodic checkpointing.
    """
    import featuregating.core.config as config
    from featuregating.datasets.dataset_config import get_dataset_config
    from featuregating.models.load import load_model_for_dataset
    from featuregating.models.sae_resources import load_steering_resources
    from featuregating.datasets.dataloader import create_dataloader, get_single_image_loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset config and model
    dataset_config = get_dataset_config(dataset_name)
    print(f"Dataset: {dataset_name} ({dataset_config.num_classes} classes)")

    temp_config = config.PipelineConfig()
    temp_config.file = config.FileConfig.for_dataset(dataset_name)
    if use_clip:
        temp_config.classify.use_clip = True
        temp_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
        temp_config.classify.clip_text_prompts = [
            f"a photo of a {cls}" for cls in dataset_config.class_names
        ]

    print(f"Loading model for {dataset_name}...")
    model, clip_classifier = load_model_for_dataset(dataset_config, device, temp_config)
    model.eval()

    # Load SAEs
    print(f"Loading SAE resources for layers: {layers}")
    steering_resources = load_steering_resources(layers, dataset_name=dataset_name)

    # Create dataloader
    print(f"Creating dataloader for {split} split...")
    prepared_path = Path(f"./data/prepared/{dataset_name}")
    dataset_loader = create_dataloader(dataset_name=dataset_name, data_path=prepared_path)

    # Get image list
    image_data = list(dataset_loader.get_numeric_samples(split))
    total_samples = len(image_data)

    if subset_size is not None and subset_size < total_samples:
        import random
        random.seed(42)
        image_data = random.sample(image_data, subset_size)
        print(f"Processing {len(image_data)} randomly selected images (subset of {total_samples})")
    else:
        print(f"Processing all {total_samples} images")

    # Initialize storage and checkpointing
    checkpoint_interval = 10000
    layer_data = {layer_idx: {'sparse_indices': [], 'sparse_activations': []} for layer_idx in layers}
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nExtracting SAE activations (saving every {checkpoint_interval} images)...")

    # Process images
    for img_idx, (image_path, _label) in enumerate(tqdm(image_data, desc="Processing images")):
        try:
            input_tensor = get_single_image_loader(image_path, dataset_config)
            input_tensor = input_tensor.to(device)

            # Setup hooks to capture residuals
            residuals = {}

            def save_resid_hook(tensor, hook):
                layer_idx = int(hook.name.split('.')[1])
                if layer_idx in layers:
                    residuals[layer_idx] = tensor.detach().cpu()
                return tensor

            fwd_hooks = [(f"blocks.{layer_idx}.hook_resid_post", save_resid_hook) for layer_idx in layers]

            # Forward pass
            with torch.no_grad():
                with model.hooks(fwd_hooks=fwd_hooks, reset_hooks_end=True):
                    _ = model(input_tensor)

            # Encode residuals with SAE
            for layer_idx in layers:
                if layer_idx not in residuals:
                    continue

                resid = residuals[layer_idx].to(device)
                sae = steering_resources[layer_idx]['sae']

                with torch.no_grad():
                    _, codes = sae.encode(resid)

                # Remove batch and CLS token
                codes = codes.cpu()
                if codes.dim() == 3:
                    codes = codes[0]
                codes = codes[1:]  # Remove CLS

                # Convert to sparse format
                active_threshold = config.BoostingConfig.active_feature_threshold
                sparse_indices_per_patch = []
                sparse_activations_per_patch = []

                for patch_idx in range(codes.shape[0]):
                    patch_codes = codes[patch_idx]
                    active_mask = patch_codes > active_threshold
                    sparse_indices_per_patch.append(torch.where(active_mask)[0].numpy())
                    sparse_activations_per_patch.append(patch_codes[active_mask].numpy())

                layer_data[layer_idx]['sparse_indices'].append(sparse_indices_per_patch)
                layer_data[layer_idx]['sparse_activations'].append(sparse_activations_per_patch)

        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
            continue

        # Save checkpoints
        if (img_idx + 1) % checkpoint_interval == 0 or (img_idx + 1) == len(image_data):
            for layer_idx in layers:
                checkpoint_file = debug_dir / f"layer_{layer_idx}_checkpoint_{img_idx + 1}.npz"
                np.savez_compressed(
                    checkpoint_file,
                    sparse_indices=np.array(layer_data[layer_idx]['sparse_indices'], dtype=object),
                    sparse_activations=np.array(layer_data[layer_idx]['sparse_activations'], dtype=object)
                )
            layer_data = {layer_idx: {'sparse_indices': [], 'sparse_activations': []} for layer_idx in layers}
            torch.cuda.empty_cache()

    # Merge checkpoints
    print("\nMerging checkpoint files...")
    for layer_idx in layers:
        checkpoint_files = sorted(debug_dir.glob(f"layer_{layer_idx}_checkpoint_*.npz"))
        all_indices, all_activations = [], []

        for checkpoint_file in checkpoint_files:
            data = np.load(checkpoint_file, allow_pickle=True)
            all_indices.extend(data['sparse_indices'])
            all_activations.extend(data['sparse_activations'])

        output_file = debug_dir / f"layer_{layer_idx}_activations.npz"
        np.savez_compressed(
            output_file,
            sparse_indices=np.array(all_indices, dtype=object),
            sparse_activations=np.array(all_activations, dtype=object)
        )
        print(f"  Layer {layer_idx}: Merged {len(all_indices)} images")

        # Cleanup checkpoints
        for checkpoint_file in checkpoint_files:
            checkpoint_file.unlink()

    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'split': split,
        'layers': layers,
        'n_images': len(image_data),
        'use_clip': use_clip,
        'image_paths': {
            idx: str(image_path)
            for idx, (image_path, _label) in enumerate(image_data)
        }
    }

    with open(output_dir / "extraction_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nExtraction complete! Saved to {output_dir}")

    # Cleanup
    del model
    if clip_classifier is not None:
        del clip_classifier
    for layer_idx, resources in steering_resources.items():
        if 'sae' in resources:
            del resources['sae']
    torch.cuda.empty_cache()

    return output_dir
