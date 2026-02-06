"""Compatibility wrapper — canonical sources are gradcamfaith.data.download and gradcamfaith.data.prepare"""
import sys
from pathlib import Path

# Re-export download functions
from gradcamfaith.data.download import (  # noqa: F401
    download_with_progress,
    download_from_gdrive,
    extract_zip,
    extract_tar_gz,
    download_hyperkvasir,
    download_imagenet,
    download_covidquex,
    download_thesis_saes,
    download_sae_checkpoints,
)

# Re-export prepare/convert functions
from gradcamfaith.data.prepare import (  # noqa: F401
    convert_dataset,
    print_summary,
    prepare_covidquex,
    prepare_hyperkvasir,
    prepare_waterbirds,
    prepare_imagenet,
    split_ids,
)


def main():
    """Main function to orchestrate all downloads."""
    data_dir, models_dir = Path("./data"), Path("./models")
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    print("Dataset & Model Setup")
    print("=" * 50)

    try:
        # Download datasets (uncomment as needed)
        download_hyperkvasir(data_dir, models_dir)
        download_covidquex(data_dir, models_dir)
        download_imagenet(data_dir, models_dir)

        from dataset_config import refresh_imagenet_config
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
    # Check for required packages
    try:
        import gdown
        import requests
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Missing required package. Install: pip install gdown requests huggingface-hub")
        sys.exit(1)

    main()
