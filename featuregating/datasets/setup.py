"""Data setup: download orchestration and dataset preparation."""
import shutil
import sys
from pathlib import Path

from featuregating.datasets.download import (
    download_covidquex,
    download_hyperkvasir,
    download_imagenet,
    download_sae_checkpoints,
    download_thesis_saes,
)
from featuregating.datasets.prepare import convert_dataset, print_summary  # noqa: F401


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
        **converter_kwargs: Additional arguments for converter.
            Special option: cleanup_source=True removes the raw source directory
            after successful conversion.

    Returns:
        Path to prepared dataset
    """
    metadata_file = prepared_path / "dataset_metadata.json"

    if not force_prepare and metadata_file.exists():
        print(f"Dataset already prepared at {prepared_path}")
        return prepared_path

    print(f"Preparing {dataset_name} dataset...")
    print("Images will be preprocessed to 224x224")
    cleanup_source = bool(converter_kwargs.pop("cleanup_source", False))

    convert_dataset(dataset_name=dataset_name, source_path=source_path, output_path=prepared_path, **converter_kwargs)

    if cleanup_source:
        _cleanup_source_after_prepare(source_path=source_path, prepared_path=prepared_path)

    return prepared_path


def _cleanup_source_after_prepare(source_path: Path, prepared_path: Path):
    """Safely remove raw source directory after successful preparation."""
    source = source_path.resolve()
    prepared = prepared_path.resolve()

    if not source.exists():
        return

    if source == prepared or prepared.is_relative_to(source):
        raise ValueError(
            f"Refusing to clean source path {source} because prepared path {prepared} is inside it."
        )

    print(f"Cleaning up raw source directory: {source}")
    shutil.rmtree(source)


def main():
    """Main function to orchestrate all downloads."""
    data_dir, models_dir = Path("./data"), Path("./data/models")
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True, parents=True)

    print("Dataset & Model Setup")
    print("=" * 50)

    try:
        download_hyperkvasir(data_dir, models_dir)
        download_covidquex(data_dir, models_dir)
        download_imagenet(data_dir)

        from featuregating.datasets.dataset_config import refresh_imagenet_config
        refresh_imagenet_config()

        download_thesis_saes(data_dir)
        download_sae_checkpoints(data_dir)
        print_summary(data_dir, models_dir)
        print("\nSetup completed successfully.")

    except KeyboardInterrupt:
        print("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        import gdown  # noqa: F401
        import requests  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401
    except ImportError:
        print("Missing required package. Install: uv add gdown requests huggingface-hub")
        sys.exit(1)

    main()
