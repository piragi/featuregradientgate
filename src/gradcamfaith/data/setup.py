"""Data setup: download orchestration and dataset preparation."""
import sys
from pathlib import Path

from gradcamfaith.data.download import (
    download_covidquex,
    download_hyperkvasir,
    download_imagenet,
    download_sae_checkpoints,
    download_thesis_saes,
)
from gradcamfaith.data.prepare import convert_dataset, print_summary  # noqa: F401


def main():
    """Main function to orchestrate all downloads."""
    data_dir, models_dir = Path("./data"), Path("./models")
    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    print("Dataset & Model Setup")
    print("=" * 50)

    try:
        download_hyperkvasir(data_dir, models_dir)
        download_covidquex(data_dir, models_dir)
        download_imagenet(data_dir, models_dir)

        from gradcamfaith.data.dataset_config import refresh_imagenet_config
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
