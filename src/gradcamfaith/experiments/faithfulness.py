"""
Faithfulness evaluation framework.

Shared perturbation infrastructure (patch masks, baseline perturbation, model
inference, attribution normalization) used by both faithfulness metrics and SaCo.

Orchestration layer that runs PatchPixelFlipping and FaithfulnessCorrelation,
computes statistics, and saves results.
"""

import gc
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from gradcamfaith.core.config import PipelineConfig
from gradcamfaith.core.types import ClassificationResult
from gradcamfaith.data.dataset_config import get_dataset_config


# ---------------------------------------------------------------------------
# Shared perturbation infrastructure
# ---------------------------------------------------------------------------

def create_patch_mask(patch_indices, image_hw, n_patches, patch_size):
    """Create a (H, W) boolean mask for the given patch indices.

    Args:
        patch_indices: Indices of patches to mask.
        image_hw: Tuple (H, W) of the image.
        n_patches: Total number of patches.
        patch_size: Size of each patch in pixels.

    Returns:
        Boolean numpy array of shape (H, W).
    """
    H, W = image_hw
    grid_size = int(np.sqrt(n_patches))
    mask = np.zeros((H, W), dtype=bool)

    for patch_idx in patch_indices:
        if patch_idx >= n_patches:
            continue
        row, col = patch_idx // grid_size, patch_idx % grid_size
        r0, r1 = row * patch_size, min((row + 1) * patch_size, H)
        c0, c1 = col * patch_size, min((col + 1) * patch_size, W)
        mask[r0:r1, c0:c1] = True

    return mask


def apply_baseline_perturbation(image, mask, perturb_baseline):
    """Apply baseline perturbation to image where *mask* is True.

    *mask* can be (H, W) or (C, H, W) — it is broadcast as needed.

    Args:
        image: Image array (C, H, W) to perturb.
        mask: Boolean mask.
        perturb_baseline: ``"black"``, ``"white"``, ``"mean"``, ``"uniform"``, or numeric.

    Returns:
        Perturbed image (copy).
    """
    arr = image.copy()
    if perturb_baseline == "black":
        baseline_value = 0.0
    elif perturb_baseline == "white":
        baseline_value = arr.max()
    elif perturb_baseline == "mean":
        baseline_value = arr.mean()
    elif perturb_baseline == "uniform":
        baseline_value = np.random.uniform(0.0, 1.0, size=arr.shape)
    elif isinstance(perturb_baseline, (int, float)):
        baseline_value = float(perturb_baseline)
    else:
        baseline_value = 0.0
    return np.where(mask, baseline_value, arr)


def predict_on_batch(model, x_batch, y_batch, device=None, use_softmax=False):
    """Run inference and return predictions for target classes.

    Args:
        model: PyTorch model.
        x_batch: Input images as numpy (N, C, H, W).
        y_batch: Target class indices as numpy (N,).
        device: Device string or None.
        use_softmax: If True return probabilities, else raw logits.

    Returns:
        Numpy array (N,) of target-class scores.
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(x_batch).float()
        if device:
            x_tensor = x_tensor.to(device)
        outputs = model(x_tensor)
        if use_softmax and outputs.shape[-1] > 1:
            outputs = torch.softmax(outputs, dim=-1)
        preds = outputs[torch.arange(len(y_batch)), y_batch].cpu().numpy()
    return preds


def normalize_patch_attribution(attribution, n_patches=196):
    """Convert an attribution map to a flat patch vector of length *n_patches*.

    Handles (3, 224, 224), (224, 224), (grid, grid), and already-flat inputs.
    Returns None if the result cannot match *n_patches*.
    """
    patch_size = 32 if n_patches == 49 else 16
    grid_size = int(np.sqrt(n_patches))

    attr = np.asarray(attribution)

    # Drop redundant channel dimension
    if attr.ndim == 3 and attr.shape[0] == 3:
        attr = attr[0]

    # Downsample spatial attributions to patch level
    if attr.shape == (224, 224):
        attr = attr.reshape(grid_size, patch_size, grid_size, patch_size).mean(axis=(1, 3))

    if attr.ndim == 2 and attr.shape != (grid_size, grid_size):
        if attr.shape[0] * attr.shape[1] == n_patches:
            attr = attr.reshape(grid_size, grid_size)

    attr = attr.reshape(-1)[:n_patches]

    if attr.shape[0] != n_patches:
        print(f"Warning: Expected {n_patches} features, got {attr.shape[0]}")
        return None

    return attr.astype(np.float32)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def calc_faithfulness(model, x_batch, y_batch, a_batch_expl, device, config, n_patches=196):
    """Run all faithfulness estimators with multi-trial statistical robustness."""
    from gradcamfaith.experiments.faithfulness_correlation import FaithfulnessCorrelation
    from gradcamfaith.experiments.pixel_flipping import PatchPixelFlipping

    n_trials = config.faithfulness.n_trials
    nr_runs = config.faithfulness.nr_runs
    gpu_batch_size = config.faithfulness.gpu_batch_size
    subset_size = config.faithfulness.subset_size_b32 if n_patches == 49 else config.faithfulness.subset_size

    print(
        f'Settings - n_trials: {n_trials}, nr_runs: {nr_runs}, '
        f'subset_size: {subset_size}, patches: {n_patches}, gpu_batch_size: {gpu_batch_size}'
    )

    estimators = [
        (
            "FaithfulnessCorrelation",
            n_trials,
            lambda: FaithfulnessCorrelation(
                n_patches=n_patches, subset_size=subset_size,
                nr_runs=nr_runs, perturb_baseline=config.faithfulness.perturb_baseline,
            ),
        ),
        (
            "PixelFlipping",
            1,
            lambda: PatchPixelFlipping(
                n_patches=n_patches, features_in_step=config.faithfulness.features_in_step,
                perturb_baseline=config.faithfulness.perturb_baseline,
            ),
        ),
    ]

    results_by_estimator = {}
    for name, trials, make_estimator in estimators:
        print(f"Running estimator: {name}")
        result = _run_estimator_trials(
            name, trials, make_estimator, model, x_batch, y_batch,
            a_batch_expl, device, gpu_batch_size,
        )
        if result:
            results_by_estimator[name] = result

    return results_by_estimator


def _run_estimator_trials(name, n_trials, make_estimator, model, x_batch, y_batch,
                          a_batch_expl, device, gpu_batch_size):
    """Run multiple trials for a single estimator and aggregate statistics."""
    all_results = []

    for trial in range(n_trials):
        original_state = np.random.get_state()
        np.random.seed(42 + trial)
        try:
            estimator = make_estimator()
            output = estimator(
                model=model, x_batch=x_batch, y_batch=y_batch,
                a_batch=a_batch_expl, device=str(device), batch_size=gpu_batch_size,
            )
            # Normalize output to 1-D scores array
            if isinstance(output, dict):
                scores = np.array(output.get('auc', [0]))
            else:
                scores = np.array(output)
            if scores.ndim > 1 and scores.shape[0] != len(y_batch):
                scores = np.mean(scores, axis=tuple(range(1, scores.ndim)))
            all_results.append(scores)
        except Exception as e:
            import traceback
            print(f"Error in trial {trial} for {name}: {e}")
            print(f"Full traceback:\n{traceback.format_exc()}")
            if "CUDA out of memory" in str(e) or "RuntimeError" in type(e).__name__:
                print("CUDA memory issue detected. Consider reducing gpu_batch_size or nr_runs.")
                torch.cuda.empty_cache()
                gc.collect()
            all_results.append(np.full(len(y_batch), np.nan))
        finally:
            np.random.set_state(original_state)

    if not all_results:
        return {}

    all_results = np.nan_to_num(np.stack(all_results, axis=0), nan=0.0)
    return {
        "mean_scores": np.mean(all_results, axis=0),
        "std_scores": np.std(all_results, axis=0),
        "all_trials": all_results,
        "n_trials": n_trials,
    }


def evaluate_faithfulness_for_results(config, model, device, classification_results):
    """Prepare batch data from classification results and run faithfulness evaluation."""
    # Detect patch configuration
    is_patch32 = False
    if hasattr(config.classify, 'clip_model_name') and config.classify.clip_model_name:
        model_name = config.classify.clip_model_name.lower()
        is_patch32 = "patch32" in model_name or "b-32" in model_name or "b32" in model_name

    n_patches = 49 if is_patch32 else 196
    print(f"Processing {len(classification_results)} samples (batching at GPU level only)")

    batch_data = _prepare_batch_data(config, classification_results, n_patches)
    if not batch_data:
        print("No valid samples found")
        return {}, np.array([])

    x_batch, y_batch, a_batch = batch_data

    batch_faithfulness = calc_faithfulness(
        model=model, x_batch=x_batch, y_batch=y_batch,
        a_batch_expl=a_batch, device=device, config=config, n_patches=n_patches,
    )

    final_results = {}
    for estimator_name, data in batch_faithfulness.items():
        mean_scores = np.array(data['mean_scores'])
        std_scores = np.array(data['std_scores'])
        if len(mean_scores) == 0:
            continue
        final_results[estimator_name] = {
            'mean_scores': mean_scores.tolist(),
            'std_scores': std_scores.tolist(),
            'n_trials': data.get('n_trials', 3),
        }

    print(f"\nProcessed {len(y_batch)} total samples")
    return final_results, y_batch


def _prepare_batch_data(config, batch_results, n_patches=196):
    """Prepare batch arrays using cached tensors — no disk I/O or retransforming."""
    x_list, y_list, a_list = [], [], []

    for result in batch_results:
        try:
            if result._cached_tensor is not None:
                img_array = result._cached_tensor
            else:
                ds_cfg = get_dataset_config(config.file.dataset_name)
                transform = ds_cfg.get_transforms('test')
                img = Image.open(result.image_path).convert('RGB')
                img_array = transform(img).cpu().numpy()

            if result._cached_raw_attribution is not None:
                attr_map = result._cached_raw_attribution
            else:
                attr_map = np.load(result.attribution_paths.raw_attribution_path)

            converted_attr = normalize_patch_attribution(attr_map, n_patches)
            if converted_attr is None:
                continue

            x_list.append(img_array)
            y_list.append(result.prediction.predicted_class_idx)
            a_list.append(converted_attr)
        except Exception as e:
            print(f"Warning: Could not process {result.image_path.name}: {e}")
            continue

    if not x_list:
        return None
    return np.stack(x_list), np.array(y_list), np.stack(a_list)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def evaluate_and_report_faithfulness(config, model, device, classification_results):
    """Evaluate faithfulness and report statistics.  Main entry point."""
    faithfulness_results, class_labels = evaluate_faithfulness_for_results(
        config, model, device, classification_results,
    )
    dataset_config = get_dataset_config(config.file.dataset_name)

    results = _build_results_structure(config, faithfulness_results, class_labels)
    _print_faithfulness_summary(results['metrics'])
    _save_faithfulness_results(config, faithfulness_results, class_labels, results)
    return results


def _build_results_structure(config, faithfulness_results, class_labels):
    results = {'dataset': config.file.dataset_name, 'metrics': {}, 'class_labels': class_labels.tolist()}

    for name, est_results in faithfulness_results.items():
        if "mean_scores" not in est_results:
            continue
        scores = np.array(est_results["mean_scores"])
        stds = np.array(est_results.get("std_scores", np.zeros_like(scores)))

        stats = _compute_statistics_from_scores(scores, stds, class_labels)
        stats['overall']['method_params'] = {
            'n_trials': est_results.get("n_trials", 3),
        }
        results['metrics'][name] = {
            **stats,
            'mean_scores': scores.tolist(),
            'std_scores': stds.tolist(),
        }
    return results


def _compute_statistics_from_scores(scores, stds, class_labels=None):
    result = {
        'overall': {
            'count': len(scores),
            'mean': float(np.nanmean(scores)),
            'median': float(np.nanmedian(scores)),
            'min': float(np.nanmin(scores)) if len(scores) > 0 else 0,
            'max': float(np.nanmax(scores)) if len(scores) > 0 else 0,
            'std': float(np.nanstd(scores)),
            'avg_trial_std': float(np.nanmean(stds)),
        }
    }

    if class_labels is not None:
        result['by_class'] = {}
        for class_idx in np.unique(class_labels):
            mask = class_labels == class_idx
            cs, ct = scores[mask], stds[mask]
            if len(cs) > 0:
                result['by_class'][int(class_idx)] = {
                    'count': len(cs), 'mean': float(np.mean(cs)),
                    'median': float(np.median(cs)), 'min': float(np.min(cs)),
                    'max': float(np.max(cs)), 'std': float(np.std(cs)),
                    'avg_trial_std': float(np.mean(ct)),
                }
    return result


def _print_faithfulness_summary(metrics):
    for name, data in metrics.items():
        overall = data['overall']
        print(f"\n{name} faithfulness statistics:")
        print(f"  Mean: {overall['mean']:.4f}")
        print(f"  Median: {overall['median']:.4f}")
        print(f"  Count: {overall['count']}")
        print(f"  Avg trial std: {overall['avg_trial_std']:.4f}")


def _save_faithfulness_results(config, faithfulness_results, class_labels, results):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    json_path = config.file.output_dir / f"faithfulness_stats{config.file.output_suffix}_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFaithfulness statistics saved to {json_path}")

    for name, est_results in faithfulness_results.items():
        scores_path = config.file.output_dir / f"faithfulness_scores_{name}{config.file.output_suffix}"
        np.savez(scores_path, mean_scores=est_results["mean_scores"],
                 std_scores=est_results.get("std_scores", []), class_labels=class_labels)
        print(f"Raw scores for {name} saved to {scores_path}.npz")
