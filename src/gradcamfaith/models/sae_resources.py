"""
SAE resource loading for feature gradient gating.

Loads sparse autoencoders for specified layers, supporting both
dataset-specific trained SAEs and CLIP Vanilla SAEs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from vit_prisma.sae import SparseAutoencoder


def load_steering_resources(layers: List[int], dataset_name: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """
    Loads SAEs for the specified layers for feature gradient gating.

    Args:
        layers: List of layer indices to load
        dataset_name: Name of the dataset ('covidquex', 'hyperkvasir', 'waterbirds', etc.)
    """
    resources = {}

    for layer_idx in layers:
        try:
            if dataset_name in ["waterbirds", "imagenet"]:
                # Use CLIP Vanilla B-32 SAE for waterbirds
                sae_path = Path(f"data/sae_clip_vanilla_b32/layer_{layer_idx}/weights.pt")
                if not sae_path.exists():
                    print(f"Warning: CLIP Vanilla SAE not found at {sae_path}")
                    continue
                print(f"Loading CLIP Vanilla SAE from {sae_path}")
                sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
                sae.cuda().eval()
            else:
                # Load SAE for other datasets
                sae_dir = Path("data") / f"sae_{dataset_name}" / f"layer_{layer_idx}"
                sae_files = list(sae_dir.glob("**/n_images_*.pt"))
                # Filter out log_feature_sparsity files
                sae_files = [f for f in sae_files if 'log_feature_sparsity' not in str(f)]

                if not sae_files:
                    print(f"Warning: No SAE found for {dataset_name} layer {layer_idx} in {sae_dir}")
                    continue

                # Use the most recent SAE file
                sae_path = sorted(sae_files)[-1]
                print(f"Loading SAE from {sae_path}")

                sae = SparseAutoencoder.load_from_pretrained(str(sae_path))
                sae.cuda().eval()

            resources[layer_idx] = {"sae": sae}

        except Exception as e:
            print(f"Error loading SAE for {dataset_name} layer {layer_idx}: {e}")

    return resources
