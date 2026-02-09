"""
Feature Gradient Gating Sweep Experiments

Compares vanilla TransLRP vs feature gradient gating across different configurations.

Flow:
    run_parameter_sweep  (outer loop: per-dataset)
      _load_dataset_resources   → model, CLIP, SAE
      _build_experiment_grid    → [(name, params), ...]
      for each experiment:
          run_single_experiment
            _build_pipeline_config  → PipelineConfig
            run_unified_pipeline    → results
            _gpu_cleanup
      _release_dataset_resources
"""

import gc
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import gradcamfaith.core.config as config
from gradcamfaith.data.dataset_config import get_dataset_config
from gradcamfaith.experiments.pipeline import run_unified_pipeline
from gradcamfaith.models.load import load_model_for_dataset
from gradcamfaith.models.sae_resources import load_steering_resources


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """Typed configuration for a parameter sweep experiment.

    Matches the parameter names of ``run_parameter_sweep`` so that
    ``run_parameter_sweep(**dataclasses.asdict(cfg))`` works directly.
    """
    datasets: List[Tuple[str, Path]]
    layer_combinations: List[List[int]]
    kappa_values: List[float]
    gate_constructions: List[str] = field(default_factory=lambda: ["combined"])
    shuffle_decoder_options: List[bool] = field(default_factory=lambda: [False])
    clamp_max_values: List[float] = field(default_factory=lambda: [10.0])
    current_mode: str = "val"
    debug_mode: bool = False
    output_base_dir: Optional[Path] = None
    subset_size: Optional[int] = None
    random_seed: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gpu_cleanup(aggressive: bool = False) -> None:
    """Centralized GPU memory cleanup.

    Args:
        aggressive: If True, runs more GC rounds and synchronizes CUDA.
                    Use after releasing large resources (model, SAE).
    """
    rounds = 5 if aggressive else 3
    for _ in range(rounds):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.synchronize()
            for _ in range(2):
                gc.collect()
            torch.cuda.empty_cache()


def _print_gpu_memory(label: str) -> None:
    """Print current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        print(
            f"GPU Memory {label}: {torch.cuda.memory_allocated()/1024**2:.1f} MB allocated, "
            f"{torch.cuda.memory_reserved()/1024**2:.1f} MB reserved"
        )


def _build_pipeline_config(
    dataset_name: str,
    experiment_params: Dict[str, Any],
    output_dir: Path,
    current_mode: str,
    debug_mode: bool,
) -> config.PipelineConfig:
    """Build a fully-configured PipelineConfig for one experiment run.

    Consolidates all PipelineConfig construction that was previously inline
    in run_single_experiment. Handles ImageNet CLIP setup.
    """
    pipeline_config = config.PipelineConfig()

    pipeline_config.file.use_cached_original = False
    pipeline_config.file.current_mode = current_mode
    pipeline_config.classify.analysis = True
    pipeline_config.classify.boosting.debug_mode = debug_mode
    pipeline_config.file.set_dataset(dataset_name)
    pipeline_config.file.base_pipeline_dir = output_dir

    if dataset_name == "imagenet":
        pipeline_config.classify.use_clip = True
        pipeline_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
        dataset_cfg = get_dataset_config(dataset_name)
        pipeline_config.classify.clip_text_prompts = [f"a photo of a {cls}" for cls in dataset_cfg.class_names]

    pipeline_config.classify.boosting.enable_feature_gradients = experiment_params['use_feature_gradients']
    pipeline_config.classify.boosting.feature_gradient_layers = experiment_params.get('feature_gradient_layers', [])
    pipeline_config.classify.boosting.kappa = experiment_params.get('kappa', 50.0)
    pipeline_config.classify.boosting.gate_construction = experiment_params.get('gate_construction', 'combined')
    pipeline_config.classify.boosting.shuffle_decoder = experiment_params.get('shuffle_decoder', False)
    pipeline_config.classify.boosting.steering_layers = []

    return pipeline_config


def _build_experiment_grid(
    layer_combinations: List[List[int]],
    kappa_values: List[float],
    gate_constructions: List[str],
    shuffle_decoder_options: List[bool],
    clamp_max_values: List[float],
) -> List[Tuple[str, Dict[str, Any]]]:
    """Generate the full experiment list: vanilla baseline + all gated combinations.

    Returns:
        List of (experiment_name, experiment_params) tuples.
        First entry is always the vanilla baseline.
    """
    experiments: List[Tuple[str, Dict[str, Any]]] = []

    experiments.append(("vanilla", {
        'use_feature_gradients': False,
        'feature_gradient_layers': [],
        'kappa': 0,
        'gate_construction': 'combined',
        'shuffle_decoder': False,
    }))

    for layers, kappa, gate_construction, shuffle_decoder, clamp_max in product(
        layer_combinations, kappa_values, gate_constructions, shuffle_decoder_options, clamp_max_values
    ):
        layers_str = '_'.join(map(str, layers))
        shuffle_suffix = "_shuffled" if shuffle_decoder else ""
        exp_name = f"layers_{layers_str}_kappa_{kappa}_{gate_construction}_clamp_{clamp_max}{shuffle_suffix}"

        experiments.append((exp_name, {
            'use_feature_gradients': True,
            'feature_gradient_layers': layers,
            'kappa': kappa,
            'gate_construction': gate_construction,
            'shuffle_decoder': shuffle_decoder,
            'clamp_max': clamp_max,
        }))

    return experiments


def _build_imagenet_clip_prompts() -> List[str]:
    """Build article-aware CLIP prompts for ImageNet class names.

    Used during model loading to configure the CLIP classifier with
    grammatically correct prompts (e.g. "a photo of a tench",
    "a photo of an orange").
    """
    def _first_synonym(name: str) -> str:
        return name.split(",")[0].strip()

    def _needs_article(s: str) -> bool:
        return not re.match(r"^(a|an|the)\b", s, flags=re.I)

    def _article(s: str) -> str:
        return "an" if re.match(r"^[aeiou]", s, flags=re.I) else "a"

    imagenet_cfg = get_dataset_config("imagenet")
    cleaned = [_first_synonym(n) for n in imagenet_cfg.class_names]
    return [
        f"a photo of {(_article(n) + ' ') if _needs_article(n) else ''}{n}" for n in cleaned
    ]


def _load_dataset_resources(
    dataset_name: str,
    layer_combinations: List[List[int]],
    device: torch.device,
) -> Tuple[torch.nn.Module, Optional[Any], Dict[int, Dict[str, Any]]]:
    """Load model, CLIP classifier, and SAE resources for one dataset.

    Returns:
        (model, clip_classifier, steering_resources)
    """
    dataset_cfg = get_dataset_config(dataset_name)

    temp_config = config.PipelineConfig()
    temp_config.file.set_dataset(dataset_name)
    if dataset_name == "imagenet":
        temp_config.classify.use_clip = True
        temp_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
        temp_config.classify.clip_text_prompts = _build_imagenet_clip_prompts()

    print(f"Loading model for {dataset_name}...")
    model, clip_classifier = load_model_for_dataset(dataset_cfg, device, temp_config)
    model = model.to(device)

    all_layers_needed = set()
    for layers in layer_combinations:
        all_layers_needed.update(layers)

    print(f"Loading SAE resources for layers: {sorted(all_layers_needed)}")
    steering_resources = load_steering_resources(list(all_layers_needed), dataset_name=dataset_name)

    return model, clip_classifier, steering_resources


def _release_dataset_resources(
    model: torch.nn.Module,
    clip_classifier: Optional[Any],
    steering_resources: Dict[int, Dict[str, Any]],
    dataset_name: str,
) -> None:
    """Move model/CLIP/SAE to CPU and free GPU memory."""
    print(f"Cleaning up model and SAE resources for {dataset_name}...")

    if hasattr(model, 'to'):
        model.to("cpu")
    del model

    if clip_classifier is not None:
        if hasattr(clip_classifier, 'text_model') and clip_classifier.text_model is not None:
            if hasattr(clip_classifier.text_model, 'to'):
                clip_classifier.text_model.to("cpu")
            del clip_classifier.text_model
        del clip_classifier

    for layer_idx, resources in steering_resources.items():
        if 'sae' in resources:
            if hasattr(resources['sae'], 'to'):
                resources['sae'].to("cpu")
            del resources['sae']
    del steering_resources

    _gpu_cleanup(aggressive=True)
    _print_gpu_memory("after cleanup")


def _summarize_result(exp_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract minimal summary from a full experiment result dict."""
    return {
        'name': exp_name,
        'status': result.get('status'),
        'n_images': result.get('n_images', 0),
        'error': result.get('error') if result.get('status') == 'error' else None,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_single_experiment(
    dataset_name: str,
    source_path: Path,
    experiment_params: Dict[str, Any],
    output_dir: Path,
    model: torch.nn.Module,
    steering_resources: Dict[int, Dict[str, Any]],
    current_mode: str = "test",
    debug_mode: bool = False,
    clip_classifier: Optional[Any] = None,
    subset_size: Optional[int] = None,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Run a single experiment with specified parameters.

    Args:
        dataset_name: Name of the dataset
        source_path: Path to dataset source
        experiment_params: Dictionary containing:
            - use_feature_gradients: bool
            - feature_gradient_layers: List[int]
            - kappa: float (gating strength parameter)
            - gate_construction: str ("activation_only", "gradient_only", or "combined")
            - shuffle_decoder: bool (whether to shuffle decoder columns)
            - clamp_max: float (maximum gate value, range: [1/clamp_max, clamp_max])
        output_dir: Where to save results
        model: Pre-loaded model to use
        steering_resources: Pre-loaded SAE resources
        clip_classifier: Pre-loaded CLIP classifier (None for non-CLIP models)
        subset_size: Number of images to process (None for all)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with results and metadata
    """
    pipeline_config = _build_pipeline_config(
        dataset_name, experiment_params, output_dir, current_mode, debug_mode,
    )

    config_dict = {
        'dataset': dataset_name,
        'timestamp': datetime.now().isoformat(),
        'experiment_params': experiment_params,
        'subset_size': subset_size,
        'random_seed': random_seed,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    try:
        results, saco_results = run_unified_pipeline(
            config=pipeline_config,
            dataset_name=dataset_name,
            source_data_path=source_path,
            prepared_data_path=Path(f"./data/{dataset_name}_unified/"),
            force_prepare=False,
            subset_size=subset_size,
            random_seed=random_seed,
            model=model,
            steering_resources=steering_resources,
            clip_classifier=clip_classifier
        )

        config_dict['status'] = 'success'
        config_dict['n_images'] = len(results)
        config_dict['saco_results'] = saco_results

    except Exception as e:
        print(f"Error in experiment: {e}")
        config_dict['status'] = 'error'
        config_dict['error'] = str(e)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    _gpu_cleanup()

    return config_dict


def run_parameter_sweep(
    datasets: List[Tuple[str, Path]],
    layer_combinations: List[List[int]],
    kappa_values: List[float],
    gate_constructions: List[str] = ["combined"],
    shuffle_decoder_options: List[bool] = [False],
    clamp_max_values: List[float] = [5.0],
    current_mode: str = "test",
    debug_mode: bool = False,
    output_base_dir: Optional[Path] = None,
    subset_size: Optional[int] = None,
    random_seed: int = 42
) -> Dict[str, List[Dict]]:
    """
    Run a parameter sweep comparing vanilla TransLRP with feature gradient gating.

    Args:
        datasets: List of (dataset_name, source_path) tuples
        layer_combinations: List of layer combinations to test (e.g., [[4], [9], [4,9]])
        kappa_values: List of kappa values to test (gating strength)
        gate_constructions: List of gate construction types to test
        shuffle_decoder_options: List of shuffle decoder options (True/False)
        clamp_max_values: List of maximum gate values (gate range: [1/clamp_max, clamp_max])
        current_mode: Dataset split to use ("train", "val", "test", "dev")
        debug_mode: If True, collect sparse features, gradients, and gate values per image
        output_base_dir: Base directory for output (auto-generated if None)
        subset_size: Number of images per dataset (None for all)
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping dataset names to lists of experiment results
    """
    if output_base_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_dir = Path(f"./experiments/feature_gradient_sweep_{timestamp}")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    sweep_config = {
        'datasets': [d[0] for d in datasets],
        'layer_combinations': layer_combinations,
        'kappa_values': kappa_values,
        'gate_constructions': gate_constructions,
        'shuffle_decoder_options': shuffle_decoder_options,
        'clamp_max_values': clamp_max_values,
        'current_mode': current_mode,
        'debug_mode': debug_mode,
        'subset_size': subset_size,
        'random_seed': random_seed,
        'timestamp': datetime.now().isoformat()
    }
    with open(output_base_dir / 'sweep_config.json', 'w') as f:
        json.dump(sweep_config, f, indent=2)

    experiments = _build_experiment_grid(
        layer_combinations, kappa_values, gate_constructions,
        shuffle_decoder_options, clamp_max_values,
    )

    all_results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name, source_path in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        _print_gpu_memory("initial")

        model, clip_classifier, steering_resources = _load_dataset_resources(
            dataset_name, layer_combinations, device,
        )

        dataset_results = []

        for exp_name, exp_params in experiments:
            print(f"\nRunning {exp_name}...")

            exp_dir = output_base_dir / dataset_name / exp_name
            result = run_single_experiment(
                dataset_name=dataset_name,
                source_path=source_path,
                experiment_params=exp_params,
                output_dir=exp_dir,
                model=model,
                steering_resources=steering_resources,
                current_mode=current_mode,
                debug_mode=debug_mode,
                clip_classifier=clip_classifier,
                subset_size=subset_size,
                random_seed=random_seed,
            )

            dataset_results.append(_summarize_result(exp_name, result))
            del result
            _print_gpu_memory(f"after {exp_name}")

        all_results[dataset_name] = dataset_results
        _release_dataset_resources(model, clip_classifier, steering_resources, dataset_name)

    with open(output_base_dir / 'sweep_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Sweep completed! Results saved to: {output_base_dir}")
    print(f"Total experiments: {sum(len(r) for r in all_results.values())}")

    return all_results


def main():
    """
    Main entry point for running feature gradient gating experiments.
    """
    from dataclasses import asdict

    cfg = SweepConfig(
        datasets=[
            # ("hyperkvasir", Path("./data/hyperkvasir/labeled-images/")),
            ("imagenet", Path("./data/imagenet/raw")),
            # ("covidquex", Path("./data/covidquex/data/lung/")),
        ],
        layer_combinations=[[3]],
        kappa_values=[0.5],
        clamp_max_values=[10.0],
        subset_size=500,
        random_seed=123,
    )

    results = run_parameter_sweep(**asdict(cfg))

    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)

    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        successful = sum(1 for r in dataset_results if r.get('status') == 'success')
        failed = sum(1 for r in dataset_results if r.get('status') == 'error')
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")


if __name__ == "__main__":
    main()
    # run_best_performers(subset_size=500)
