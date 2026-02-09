"""
Dataset and model download utilities.

Handles downloading datasets (HyperKvasir, ImageNet, CovidQueX),
model checkpoints, and SAE weights from various sources.
"""
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path

import gdown
import requests
from huggingface_hub import hf_hub_download


def download_with_progress(url: str, filename: Path) -> None:
    """Download file with progress bar using requests."""
    try:
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(filename, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rDownloading: {percent:.1f}%", end='')
        print()  # New line after download
    except Exception:
        raise


def download_from_gdrive(file_id: str, output_path: Path) -> None:
    """Download file from Google Drive using gdown."""
    if output_path.exists():
        return

    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        # fuzzy=True handles virus scan confirmation for large files
        gdown.download(url, str(output_path), quiet=False, fuzzy=True)
    except Exception:
        raise


def extract_zip(zip_path: Path, extract_to: Path, remove_after: bool = True) -> None:
    """Extract zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        if remove_after:
            zip_path.unlink()
    except Exception:
        raise


def extract_tar_gz(tar_path: Path, extract_to: Path, remove_after: bool = True) -> None:
    """Extract tar.gz file."""
    try:
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            # Python 3.14 defaults to safer extraction filtering; pass it
            # explicitly when available while staying compatible with 3.10+.
            try:
                tar_ref.extractall(extract_to, filter="data")
            except TypeError:
                tar_ref.extractall(extract_to)
        if remove_after:
            tar_path.unlink()
    except Exception:
        raise


def download_hyperkvasir(data_dir: Path, models_dir: Path) -> None:
    """Download Hyperkvasir dataset and model."""
    print("\nDownloading Hyperkvasir...")

    # Create hyperkvasir subdirectories
    hk_data_dir = data_dir / "hyperkvasir"
    hk_models_dir = models_dir / "hyperkvasir"
    hk_data_dir.mkdir(exist_ok=True, parents=True)
    hk_models_dir.mkdir(exist_ok=True, parents=True)

    extracted_dir = hk_data_dir / "labeled-images"
    dataset_ready = extracted_dir.exists() and any(extracted_dir.rglob("*"))

    if dataset_ready:
        print(f"Hyperkvasir dataset already present at {extracted_dir}; skipping dataset download.")
    else:
        dataset_url = "https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-labeled-images.zip"
        dataset_path = hk_data_dir / "hyper-kvasir-labeled-images.zip"

        if not dataset_path.exists():
            try:
                # Try wget first (faster for large files) - disable cert check for Simula server
                subprocess.run(["wget", "--no-check-certificate", "-O", str(dataset_path), dataset_url], check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to requests if wget is not available
                download_with_progress(dataset_url, dataset_path)

        extract_zip(dataset_path, hk_data_dir)

    # Download Hyperkvasir model
    model_info = {
        "name": "hyperkvasir_vit_model.pth",
        "id": "1gT4Z0qD09ClPOcfgXo0rMAsjvoD-tpDE",
        "description": "ViT model for Hyperkvasir"
    }

    output_path = hk_models_dir / model_info["name"]
    download_from_gdrive(model_info["id"], output_path)


def download_imagenet(data_dir: Path) -> None:
    """
    Download ImageNet-1k validation and test splits from Hugging Face parquet files.
    Avoids training shards entirely by targeting parquet files directly.
    """
    print("Downloading ImageNet-1k validation and test splits...")
    print("Note: Requires HF login & access to ILSVRC/imagenet-1k.")
    print("Visit https://huggingface.co/datasets/ILSVRC/imagenet-1k to request access.")

    # Create imagenet subdirectories
    in_data_dir = data_dir / "imagenet"
    raw_dir = in_data_dir / "raw"
    val_dir = raw_dir / "val"
    test_dir = raw_dir / "test"

    # Fast path if both splits already exist with content
    if val_dir.exists() and any(val_dir.glob("class_*/*")) and test_dir.exists() and any(test_dir.glob("class_*/*")):
        print(f"ImageNet raw splits already exist at {raw_dir}")
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("⚠ Error: datasets library not installed. Install with: pip install datasets")
        return

    def _save_split(split_name: str, pattern: str, out_dir: Path):
        try:
            print(f"Loading {split_name} parquet files...")
            ds = load_dataset(
                "parquet",
                data_files=pattern,
                split="train",
            )

            # Save class names from validation split
            if split_name == "validation" and 'label' in ds.features and hasattr(ds.features['label'], 'names'):
                class_names = ds.features['label'].names
                print(f"Found {len(class_names)} ImageNet class names")
                class_names_file = raw_dir / "class_names.json"
                import json
                with open(class_names_file, 'w') as f:
                    json.dump(class_names, f, indent=2)
                print(f"✓ Saved class names to {class_names_file}")

            print(f"Saving {len(ds)} {split_name} images...")
            from tqdm import tqdm
            saved = 0
            warned_unlabeled = False
            for idx, item in enumerate(tqdm(ds, desc=f"Saving {split_name} images")):
                image = item.get("image")
                label = item.get("label")
                # Robust unlabeled handling:
                # - validation: skip unlabeled samples, keep iterating
                # - test: put unlabeled samples under class_-1
                if label is None or (isinstance(label, int) and label < 0):
                    if split_name.lower().startswith("test"):
                        label = -1
                    else:
                        if not warned_unlabeled:
                            print(f"⚠ {split_name} sample without label encountered — skipping.")
                            warned_unlabeled = True
                        continue

                class_dir = out_dir / f"class_{label}"
                class_dir.mkdir(exist_ok=True)
                img_path = class_dir / f"img_{idx:06d}.JPEG"
                if hasattr(image, "save"):
                    image.save(img_path)
                    saved += 1
            print(f"✓ {split_name.capitalize()} set saved to {out_dir} ({saved} files)")
        except Exception as e:
            import traceback
            print(f"⚠ Error downloading {split_name} split: {e}")
            print("Full traceback:")
            traceback.print_exc()

    # Download/prepare validation
    if not any(val_dir.glob("class_*/*")):
        _save_split("validation", "hf://datasets/ILSVRC/imagenet-1k/data/validation-*.parquet", val_dir)
    else:
        print(f"Validation already present at {val_dir}")

    # Download/prepare test
    if not any(test_dir.glob("class_*/*")):
        _save_split("test", "hf://datasets/ILSVRC/imagenet-1k/data/test-*.parquet", test_dir)
    else:
        print(f"Test already present at {test_dir}")

    print(f"✓ ImageNet raw download complete at {raw_dir}")


def download_covidquex(data_dir: Path, models_dir: Path) -> None:
    """Download CovidQueX dataset and model."""
    print("\nDownloading CovidQueX...")

    # Create covidquex subdirectories
    cq_data_dir = data_dir / "covidquex"
    cq_models_dir = models_dir / "covidquex"
    cq_data_dir.mkdir(exist_ok=True, parents=True)
    cq_models_dir.mkdir(exist_ok=True, parents=True)

    # Download CovidQueX dataset
    dataset_info = {
        "name": "covidquex_data.tar.gz",
        "id": "1XrCWP3ICQvurchnJjyVweYy2jHQM0BHO&confirm=t",
        "description": "CovidQueX dataset (tar.gz)"
    }

    dataset_path = cq_data_dir / dataset_info["name"]
    download_from_gdrive(dataset_info["id"], dataset_path)

    # Extract dataset if it's a tar.gz file
    if dataset_path.suffix == '.gz' and dataset_path.exists():
        extracted_dir = cq_data_dir / "extracted"
        if not extracted_dir.exists():
            extract_tar_gz(dataset_path, cq_data_dir)

    # Download CovidQueX model
    model_info = {
        "name": "covidquex_model.pth",
        "id": "1JZM5ZRncaV3iFX9L6NFT1P0-APyHbBV0&confirm=t",
        "description": "CovidQueX model"
    }

    output_path = cq_models_dir / model_info["name"]
    download_from_gdrive(model_info["id"], output_path)

    # Check if the downloaded model is a tar.gz archive and extract if needed
    if output_path.exists():
        import shutil

        # Check if file is a tar/gzip archive
        try:
            with open(output_path, 'rb') as f:
                magic = f.read(2)
                if magic == b'\x1f\x8b':  # gzip magic number - it's a tar.gz file
                    print(f"Extracting compressed model archive: {output_path}")

                    # First extract the tar.gz file
                    with tarfile.open(output_path, 'r:gz') as tar:
                        tar.extractall(cq_models_dir)

                    # Now extract model_best.pth.tar
                    model_tar_path = cq_models_dir / "results_model" / "model_best.pth.tar"
                    if model_tar_path.exists():
                        print(f"Extracting model from: {model_tar_path}")

                        # Extract the actual model from the .pth.tar file
                        # The .pth.tar file contains the actual PyTorch model state dict
                        final_model_path = cq_models_dir / "covidquex_model.pth"

                        # Copy the .pth.tar file directly as the model
                        # (PyTorch can load .pth.tar files directly)
                        shutil.copy(str(model_tar_path), str(final_model_path))
                        print(f"Model copied to: {final_model_path}")

                        # Clean up - remove the original tar.gz file and extracted directories
                        results_dir = cq_models_dir / "results_model"
                        if results_dir.exists():
                            shutil.rmtree(results_dir)
                    else:
                        print(f"Warning: Expected model file not found at {model_tar_path}")
        except Exception as e:
            print(f"Warning: Could not check/extract model file: {e}")


def download_thesis_saes(data_dir: Path) -> None:
    """
    Download thesis-trained SAE checkpoints from Google Drive zips.
    """
    print("\n" + "=" * 50)
    print("Downloading Thesis SAE Checkpoints (Zip Archives)")
    print("=" * 50)

    # Map: 'Folder Name' -> 'Google Drive File ID'
    # Added &confirm=t to ensure large file download works smoothly
    sae_zips = {
        "sae_covidquex": "1Kncxk-tfQQFdLG_mFL5fJ1I-rDdcqLGy&confirm=t",
        "sae_hyperkvasir": "1nVAvXoJxOKNy7ROMA2XLfAG3diUw5XFf&confirm=t"
    }

    for folder_name, file_id in sae_zips.items():
        target_dir = data_dir / folder_name

        # 1. Check if the folder already exists
        if target_dir.exists():
            print(f"✓ {folder_name} already exists. Skipping download.")
            continue

        print(f"Processing {folder_name}...")

        # 2. Download the zip file
        zip_filename = f"{folder_name}.zip"
        zip_path = data_dir / zip_filename

        download_from_gdrive(
            file_id=file_id,
            output_path=zip_path,
        )

        # 3. Extract and cleanup
        # Assumes the zip was created by zipping the *folder*, not the files inside.
        # If the zip contains just files, change extract_to=data_dir to extract_to=target_dir
        if zip_path.exists():
            print(f"Extracting {zip_filename}...")
            extract_zip(zip_path, data_dir, remove_after=True)

            if target_dir.exists():
                print(f"✓ {folder_name} installed successfully.")
            else:
                print(f"⚠ Warning: Extraction finished but {target_dir} was not found.")
        else:
            print(f"✗ Failed to download {zip_filename}")

    print("✓ Thesis SAE setup complete.")

def download_sae_checkpoints(data_dir: Path) -> None:
    """Download SAE checkpoints from HuggingFace for all layers."""
    print("\n" + "=" * 50)
    print("Downloading CLIP Vanilla SAE Checkpoints from HuggingFace")
    print("=" * 50)

    # CLIP Vanilla SAEs (resid_post only) - all patches (commented out)
    layer_repo_map = {
        0: "prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-0-hook_resid_post-l1-1e-05",
        1: "prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-1-hook_resid_post-l1-1e-05",
        2: "prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-2-hook_resid_post-l1-5e-05",
        3: "prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-3-hook_resid_post-l1-1e-05",
        4: "prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-4-hook_resid_post-l1-1e-05",
        5: "prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-5-hook_resid_post-l1-1e-05",
        6: "prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-6-hook_resid_post-l1-1e-05",
        7: "prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-7-hook_resid_post-l1-1e-05",
        8: "prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-8-hook_resid_post-l1-1e-05",
        9: "prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-9-hook_resid_post-l1-1e-05",
        10: "prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-10-hook_resid_post-l1-1e-05",
        11: "prisma-multimodal/sparse-autoencoder-clip-b-32-sae-vanilla-x64-layer-11-hook_resid_post-l1-1e-05"
    }

    sae_base_dir = data_dir / "sae_clip_vanilla_b32"
    successful_layers = []

    for layer_num, repo_id in layer_repo_map.items():
        print(f"Downloading Layer {layer_num} SAE checkpoint...")

        try:
            # List files in the repository to find the .pt file
            from huggingface_hub import list_repo_files
            files = list_repo_files(repo_id)
            pt_files = [f for f in files if f.endswith(".pt")]

            if not pt_files:
                print(f"✗ Layer {layer_num} failed: No .pt file found in repository")
                continue

            # Download the .pt file
            pt_filename = pt_files[0]

            downloaded_path = hf_hub_download(repo_id=repo_id, filename=pt_filename, cache_dir="./hf_cache")
            target_dir = sae_base_dir / f"layer_{layer_num}"
            target_dir.mkdir(parents=True, exist_ok=True)

            # Save as weights.pt for compatibility
            shutil.copy(downloaded_path, target_dir / "weights.pt")

            # Download config if available
            try:
                config_path = hf_hub_download(repo_id=repo_id, filename="config.json", cache_dir="./hf_cache")
                shutil.copy(config_path, target_dir / "config.json")
            except Exception:
                pass

            successful_layers.append(layer_num)
            print(f"✓ Layer {layer_num} complete")

        except Exception as e:
            print(f"✗ Layer {layer_num} failed: {str(e)}")

    print(f"\nDownloaded {len(successful_layers)}/12 SAE layers to {sae_base_dir}")
