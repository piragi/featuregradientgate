"""
Dataset preparation and conversion utilities.

Converts raw datasets (HyperKvasir, ImageNet, CovidQueX)
into a prepared directory format: train/val/test/class_<idx>/.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from gradcamfaith.data.dataset_config import COVIDQUEX_CONFIG, HYPERKVASIR_CONFIG, DatasetConfig


def _create_output_structure(output_path: Path, num_classes: int) -> None:
    """Create standard output directory structure for all datasets."""
    for split in ['train', 'val', 'test']:
        for class_idx in range(num_classes):
            (output_path / split / f"class_{class_idx}").mkdir(parents=True, exist_ok=True)


def _create_conversion_stats(dataset_name: str, config: DatasetConfig) -> Dict:
    """Create initial conversion statistics dictionary."""
    return {
        'dataset': dataset_name,
        'total_images': 0,
        'splits': {
            'train': 0,
            'val': 0,
            'test': 0
        },
        'classes': {
            name: 0
            for name in config.class_names
        },
        'class_mapping': config.class_to_idx
    }


def _save_metadata(output_path: Path, stats: Dict) -> None:
    """Save conversion metadata to JSON file."""
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nConversion complete!")
    print(f"Total images: {stats['total_images']}")
    print(f"Splits: {stats['splits']}")


def _process_image(
    img_path: Path, dest_path: Path, stats: Dict, split: str, class_name: str, copy_only: bool = False
) -> bool:
    """Process and save a single image, updating statistics. Returns True if successful."""
    try:
        if img_path.stat().st_size == 0:
            print(f"Warning: Skipping empty file: {img_path}")
            return False

        # Ensure destination directory exists (covers special cases like class_-1)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        if copy_only:
            shutil.copy2(img_path, dest_path)
        else:
            img = Image.open(img_path).convert('RGB')
            img.save(dest_path, 'PNG')

        stats['total_images'] += 1
        stats['splits'][split] += 1
        if class_name not in stats['classes']:
            stats['classes'][class_name] = 0
        stats['classes'][class_name] += 1
        return True
    except Exception as e:
        print(f"Warning: Failed to process {img_path}: {e}")
        return False


def split_ids(len_ids):
    """Reproduce the exact same split as in the SSL4GIE paper."""
    int(round((80 / 100) * len_ids))
    int(round((10 / 100) * len_ids))
    test_size = int(round((10 / 100) * len_ids))
    train_indices, test_indices = train_test_split(
        np.linspace(0, len_ids - 1, len_ids).astype("int"),
        test_size=test_size,
        random_state=42,
    )
    train_indices, val_indices = train_test_split(train_indices, test_size=test_size, random_state=42)
    return train_indices, test_indices, val_indices


def prepare_covidquex(source_path: Path, output_path: Path, config: DatasetConfig = COVIDQUEX_CONFIG) -> Dict:
    """Convert CovidQUEX dataset to prepared format."""
    output_path, source_path = Path(output_path), Path(source_path)

    _create_output_structure(output_path, config.num_classes)
    conversion_stats = _create_conversion_stats('covidquex', config)

    split_mapping = {'Train': 'train', 'Val': 'val', 'Test': 'test'}

    for source_split, target_split in split_mapping.items():
        split_dir = source_path / source_split
        if not split_dir.exists():
            raise ValueError(f"Expected split directory not found: {split_dir}")

        for class_name, class_idx in config.class_to_idx.items():
            images_dir = split_dir / class_name / "images"
            if not images_dir.exists():
                print(f"Warning: Images directory {images_dir} not found")
                continue

            images = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
            if not images:
                continue

            print(f"Found {len(images)} images in {source_split}/{class_name}")

            for idx, img_path in enumerate(tqdm(images, desc=f"{source_split}/{class_name}")):
                new_name = f"img_{class_idx:02d}_{target_split}_{idx:05d}.png"
                dest_path = output_path / target_split / f"class_{class_idx}" / new_name
                _process_image(img_path, dest_path, conversion_stats, target_split, class_name)

    _save_metadata(output_path, conversion_stats)
    return conversion_stats


def prepare_hyperkvasir(
    source_path: Path,
    output_path: Path,
    config: DatasetConfig = HYPERKVASIR_CONFIG,
    csv_path: Optional[Path] = None
) -> Dict:
    """Convert HyperKvasir dataset to prepared format using CSV metadata."""
    output_path, source_path = Path(output_path), Path(source_path)

    _create_output_structure(output_path, config.num_classes)
    conversion_stats = _create_conversion_stats('hyperkvasir', config)

    # Load CSV metadata
    csv_path = csv_path or source_path / "image-labels.csv"
    if not csv_path.exists():
        raise ValueError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    df_filtered = df[(df['Classification'] == 'anatomical-landmarks') & (df['Finding'].isin(config.class_names))].copy()

    print(f"Found {len(df_filtered)} images for classes: {config.class_names}")

    # Create splits using SSL4GIE method
    train_idx, test_idx, val_idx = split_ids(len(df_filtered))
    split_indices = {'train': train_idx, 'val': val_idx, 'test': test_idx}

    for split, indices in split_indices.items():
        split_df = df_filtered.iloc[indices]

        for _, row in tqdm(split_df.iterrows(), desc=f"Processing {split}", total=len(split_df)):
            video_file, finding = row['Video file'], row['Finding']
            class_idx = config.class_to_idx[finding]

            # Find the actual image file
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png']:
                if (potential_path := source_path / f"{video_file}{ext}").exists():
                    img_path = potential_path
                    break
                if matches := list(source_path.rglob(f"{video_file}{ext}")):
                    img_path = matches[0]
                    break

            if img_path is None:
                print(f"Warning: Could not find image for {video_file}")
                continue

            img_count = conversion_stats['splits'][split]
            new_name = f"img_{class_idx:02d}_{split}_{img_count:05d}.png"
            dest_path = output_path / split / f"class_{class_idx}" / new_name
            _process_image(img_path, dest_path, conversion_stats, split, finding)

    _save_metadata(output_path, conversion_stats)
    return conversion_stats


def prepare_imagenet(source_path: Path, output_path: Path, config: Optional['DatasetConfig'] = None) -> Dict:
    """
    Convert ImageNet dataset to prepared format.
    If raw/test exists, copy val and test directly. Otherwise, fall back to splitting val in half.
    """
    output_path, source_path = Path(output_path), Path(source_path)

    if config is None:
        from gradcamfaith.data.dataset_config import get_imagenet_config
        config = get_imagenet_config()

    # Optional guard (nice safety net)
    if "class_0" in config.class_names[0]:
        print("⚠ Using placeholder ImageNet class names. Did you call refresh_imagenet_config() after download?")

    _create_output_structure(output_path, config.num_classes)
    conversion_stats = _create_conversion_stats('imagenet', config)

    val_source = source_path / "val"
    test_source = source_path / "test"

    def _copy_split(split_name: str, split_dir: Path):
        print(f"Processing {split_name} split from {split_dir}")
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir() or not class_dir.name.startswith("class_"):
                continue
            try:
                class_idx = int(class_dir.name.split("_")[1])
            except Exception:
                continue
            class_name = config.idx_to_class.get(class_idx, f"class_{class_idx}")
            images = list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            images = sorted(images)
            for img_path in images:
                img_count = conversion_stats['splits'][split_name]
                ext = img_path.suffix.lower()
                if ext not in [".jpeg", ".jpg", ".png"]:
                    ext = ".jpeg"
                new_name = f"img_{class_idx:03d}_{split_name}_{img_count:06d}{ext}"
                dest_path = output_path / split_name / f"class_{class_idx}" / new_name
                _process_image(img_path, dest_path, conversion_stats, split_name, class_name, copy_only=True)
        print(f"✓ {split_name.capitalize()} copied: {conversion_stats['splits'][split_name]} images")

    processed_test = False
    if val_source.exists() and any(val_source.glob("class_*/*")):
        _copy_split('val', val_source)
    if test_source.exists() and any(test_source.glob("class_*/*")):
        _copy_split('test', test_source)
        processed_test = True

    # If test wasn't processed but we do have validation, derive test from second half of val per class
    if not processed_test and val_source.exists():
        print(f"Processing validation split from {val_source}")
        print("Will split 50k images into 25k val + 25k test")
        images_by_class = {}
        for class_dir in sorted(val_source.iterdir()):
            if not class_dir.is_dir() or not class_dir.name.startswith("class_"):
                continue
            class_idx = int(class_dir.name.split("_")[1])
            images = list(class_dir.glob("*.JPEG")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            images_by_class[class_idx] = sorted(images)
        from tqdm import tqdm
        print("Splitting images per class (first half → val, second half → test)...")
        for class_idx, images in tqdm(images_by_class.items(), desc="Processing classes"):
            class_name = config.idx_to_class[class_idx]
            mid_point = len(images) // 2
            for img_path in images[:mid_point]:
                img_count = conversion_stats['splits']['val']
                new_name = f"img_{class_idx:03d}_val_{img_count:06d}.jpeg"
                dest_path = output_path / "val" / f"class_{class_idx}" / new_name
                _process_image(img_path, dest_path, conversion_stats, 'val', class_name, copy_only=True)
            for img_path in images[mid_point:]:
                img_count = conversion_stats['splits']['test']
                new_name = f"img_{class_idx:03d}_test_{img_count:06d}.jpeg"
                dest_path = output_path / "test" / f"class_{class_idx}" / new_name
                _process_image(img_path, dest_path, conversion_stats, 'test', class_name, copy_only=True)
        print(f"✓ Split complete: {conversion_stats['splits']['val']} val, {conversion_stats['splits']['test']} test")

    _save_metadata(output_path, conversion_stats)
    return conversion_stats


def convert_dataset(dataset_name: str, source_path: Path, output_path: Path, **kwargs) -> Dict:
    """Main entry point for dataset conversion."""
    converters = {
        'covidquex': prepare_covidquex,
        'hyperkvasir': prepare_hyperkvasir,
        'imagenet': prepare_imagenet,
    }

    if dataset_name.lower() not in converters:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(converters.keys())}")

    converter_func = converters[dataset_name.lower()]
    return converter_func(source_path, output_path, **kwargs)


def print_summary(data_dir: Path, models_dir: Path) -> None:
    """Print summary of downloaded files."""
    print("\nSummary:")

    # Count files
    total_files = 0
    for path in [data_dir, models_dir]:
        if path.exists():
            total_files += sum(1 for _ in path.rglob('*') if _.is_file())

    print(f"  Total files downloaded: {total_files}")
    print(f"  Data directory: {data_dir.absolute()}")
    print(f"  Models directory: {models_dir.absolute()}")
