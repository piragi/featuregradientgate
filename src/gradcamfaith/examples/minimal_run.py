"""
Minimal end-to-end example for gradcamfaith.

Demonstrates SAE feature-gradient gating on a tiny image subset.

Usage modes:
  explore  — edit ExampleConfig fields below, then run:
             uv run python -m gradcamfaith.examples.minimal_run
  paper    — use a frozen config from configs/paper/v1/ (placeholder path,
             to be populated after paper submission)

Dry-run (no data/models required):
  uv run python -m gradcamfaith.examples.minimal_run --dry-run
"""

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExampleConfig:
    """Typed config for the minimal example run."""

    # Dataset
    dataset_name: str = "hyperkvasir"
    source_path: Path = field(default_factory=lambda: Path("./data/hyperkvasir/labeled-images/"))

    # Feature gradient gating
    layers: List[int] = field(default_factory=lambda: [6, 9, 10])
    kappa: float = 0.5
    gate_construction: str = "combined"

    # Run settings
    subset_size: int = 5
    random_seed: int = 42
    current_mode: str = "test"
    output_dir: Optional[Path] = None

    # Dry-run: validate config and manifest without requiring data/models
    dry_run: bool = False


def _git_sha() -> str:
    """Get current git commit SHA, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _write_manifest(output_dir: Path, config: ExampleConfig) -> Path:
    """Write a reproducibility manifest for this run."""
    manifest = {
        "resolved_config": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(config).items()
        },
        "seed": config.random_seed,
        "timestamp": datetime.now().isoformat(),
        "git_sha": _git_sha(),
        "environment_lock": "uv.lock",
    }
    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def run_example(config: ExampleConfig) -> Dict[str, Any]:
    """
    Run the minimal example pipeline.

    Steps:
      1. Resolve output directory and write run manifest.
      2. Load model and SAE resources.
      3. Run a single experiment (vanilla baseline + gated).
      4. Print summary.

    Returns:
        Dictionary with experiment results and manifest path.
    """
    # Resolve output directory
    if config.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.output_dir = Path(f"./experiments/example_run_{timestamp}")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Write manifest early (before any heavy imports)
    manifest_path = _write_manifest(config.output_dir, config)
    print(f"Run manifest: {manifest_path}")

    if config.dry_run:
        print("\n[dry-run] Config validated and manifest written.")
        print(f"[dry-run] Output directory: {config.output_dir}")
        print(f"[dry-run] Resolved config:\n{json.dumps(asdict(config), indent=2, default=str)}")
        return {"status": "dry-run", "manifest_path": str(manifest_path)}

    # Heavy imports deferred so dry-run stays fast
    import torch

    from gradcamfaith.core.config import PipelineConfig
    from gradcamfaith.data import get_dataset_config
    from gradcamfaith.models import run_unified_pipeline
    from gradcamfaith.models.load import load_model_for_dataset
    from gradcamfaith.models.sae_resources import load_steering_resources

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset config and model
    dataset_config = get_dataset_config(config.dataset_name)
    print(f"Dataset: {config.dataset_name} ({dataset_config.num_classes} classes)")

    pipeline_config = PipelineConfig()
    pipeline_config.file.set_dataset(config.dataset_name)
    pipeline_config.file.current_mode = config.current_mode
    pipeline_config.file.base_pipeline_dir = config.output_dir

    model, clip_classifier = load_model_for_dataset(dataset_config, device, pipeline_config)

    # Load SAE resources
    print(f"Loading SAE resources for layers {config.layers}...")
    steering_resources = load_steering_resources(config.layers, dataset_name=config.dataset_name)

    results = {}

    # --- Vanilla baseline (no gating) ---
    print("\n--- Vanilla TransLRP (baseline) ---")
    pipeline_config_vanilla = PipelineConfig()
    pipeline_config_vanilla.file.set_dataset(config.dataset_name)
    pipeline_config_vanilla.file.current_mode = config.current_mode
    pipeline_config_vanilla.file.base_pipeline_dir = config.output_dir / "vanilla"
    pipeline_config_vanilla.classify.analysis = True
    pipeline_config_vanilla.classify.boosting.enable_feature_gradients = False

    vanilla_results, vanilla_saco = run_unified_pipeline(
        config=pipeline_config_vanilla,
        dataset_name=config.dataset_name,
        source_data_path=config.source_path,
        model=model,
        steering_resources=steering_resources,
        clip_classifier=clip_classifier,
        prepared_data_path=Path(f"./data/{config.dataset_name}_unified/"),
        subset_size=config.subset_size,
        random_seed=config.random_seed,
    )
    results["vanilla"] = {"n_images": len(vanilla_results), "saco": vanilla_saco}
    print(f"  Processed {len(vanilla_results)} images")
    print(f"  SaCo mean: {vanilla_saco.get('mean', 'N/A')}")

    # --- Feature-gradient gated ---
    print(f"\n--- Feature-Gradient Gated (kappa={config.kappa}, layers={config.layers}) ---")
    pipeline_config_gated = PipelineConfig()
    pipeline_config_gated.file.set_dataset(config.dataset_name)
    pipeline_config_gated.file.current_mode = config.current_mode
    pipeline_config_gated.file.base_pipeline_dir = config.output_dir / "gated"
    pipeline_config_gated.classify.analysis = True
    pipeline_config_gated.classify.boosting.enable_feature_gradients = True
    pipeline_config_gated.classify.boosting.feature_gradient_layers = config.layers
    pipeline_config_gated.classify.boosting.kappa = config.kappa
    pipeline_config_gated.classify.boosting.gate_construction = config.gate_construction

    gated_results, gated_saco = run_unified_pipeline(
        config=pipeline_config_gated,
        dataset_name=config.dataset_name,
        source_data_path=config.source_path,
        model=model,
        steering_resources=steering_resources,
        clip_classifier=clip_classifier,
        prepared_data_path=Path(f"./data/{config.dataset_name}_unified/"),
        subset_size=config.subset_size,
        random_seed=config.random_seed,
    )
    results["gated"] = {"n_images": len(gated_results), "saco": gated_saco}
    print(f"  Processed {len(gated_results)} images")
    print(f"  SaCo mean: {gated_saco.get('mean', 'N/A')}")

    # --- Summary ---
    print(f"\n{'='*50}")
    print("EXAMPLE RUN SUMMARY")
    print(f"{'='*50}")
    print(f"Dataset:       {config.dataset_name}")
    print(f"Subset size:   {config.subset_size}")
    print(f"Layers:        {config.layers}")
    print(f"Kappa:         {config.kappa}")
    v_mean = vanilla_saco.get("mean", 0)
    g_mean = gated_saco.get("mean", 0)
    print(f"SaCo vanilla:  {v_mean:.4f}")
    print(f"SaCo gated:    {g_mean:.4f}")
    if v_mean:
        print(f"SaCo delta:    {g_mean - v_mean:+.4f} ({(g_mean - v_mean) / abs(v_mean) * 100:+.1f}%)")
    print(f"Output:        {config.output_dir}")
    print(f"Manifest:      {manifest_path}")

    results["manifest_path"] = str(manifest_path)
    results["status"] = "success"
    return results


def main():
    """Thin runner: load config and execute."""
    config = ExampleConfig()

    # Minimal CLI: --dry-run flag
    if "--dry-run" in sys.argv:
        config.dry_run = True

    run_example(config)


if __name__ == "__main__":
    main()
