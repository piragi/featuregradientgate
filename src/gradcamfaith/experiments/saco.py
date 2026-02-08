"""
SaCo (Sorting Attribution Consistency) metric.

Bins patches by attribution value, perturbs each bin, and checks whether
higher-attribution bins cause proportionally larger confidence drops.

Flow:
    run_binned_attribution_analysis  (entry point, called by pipeline)
      _get_or_compute_binned_results
        calculate_binned_saco_for_image  (per-image)
          load_image_and_attributions
          create_binned_perturbations  → bins + perturbed tensors
          measure_bin_impacts          → per-bin confidence deltas
          compute_saco_from_impacts    → SaCo score + bias
      post-hoc analysis (faithfulness vs correctness, patterns, summary save)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageStat
from tqdm import tqdm

from gradcamfaith.core.types import ClassificationResult
from gradcamfaith.data.dataset_config import get_dataset_config
from gradcamfaith.experiments.faithfulness import create_patch_mask


# ---------------------------------------------------------------------------
# Model inference (returns full prediction dicts, unlike faithfulness.predict_on_batch)
# ---------------------------------------------------------------------------

def batched_model_inference(model_instance, image_batch, device, batch_size=32):
    """Run model inference on a batch of images.  Returns list of prediction dicts."""
    model_instance.eval()
    all_predictions = []
    with torch.no_grad():
        for i in range(0, len(image_batch), batch_size):
            batch_chunk = image_batch[i:i + batch_size].to(device)
            logits = model_instance(batch_chunk)
            probabilities = torch.softmax(logits, dim=1)
            predicted_indices = torch.argmax(probabilities, dim=1)
            for j in range(len(batch_chunk)):
                all_predictions.append({
                    "predicted_class_idx": predicted_indices[j].item(),
                    "probabilities": probabilities[j],
                    "confidence": probabilities[j, predicted_indices[j]].item(),
                })
    return all_predictions


# ---------------------------------------------------------------------------
# Core SaCo algorithm
# ---------------------------------------------------------------------------

def calculate_saco_vectorized_with_bias(attributions, confidence_impacts):
    """Compute overall SaCo score and per-bin attribution bias."""
    n = len(attributions)
    if n < 2:
        return 0.0, np.zeros(n)

    attr_diffs = attributions[:, None] - attributions[None, :]
    impact_diffs = confidence_impacts[:, None] - confidence_impacts[None, :]
    upper_tri = np.triu(np.ones((n, n), dtype=bool), k=1)

    violations = (impact_diffs * attr_diffs < 0) & upper_tri
    violation_weights = violations * attr_diffs

    bin_bias = np.zeros(n)
    bin_bias -= np.sum(violation_weights, axis=1)
    bin_bias += np.sum(violation_weights, axis=0)

    is_faithful = ~violations & upper_tri
    weights = np.where(is_faithful, attr_diffs, -attr_diffs)
    weights_upper = np.where(upper_tri, weights, 0)
    total_abs = np.sum(np.abs(weights_upper))
    overall_saco = np.sum(weights_upper) / total_abs if total_abs > 0 else 0.0

    if total_abs > 0:
        bin_bias = bin_bias / total_abs

    return overall_saco, bin_bias


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

@dataclass
class BinInfo:
    """Information about an attribution bin."""
    bin_id: int
    min_value: float
    max_value: float
    patch_indices: List[int]
    mean_attribution: float
    total_attribution: float
    n_patches: int


def create_attribution_bins_from_patches(raw_attributions, n_bins=20):
    """Create equal-size bins sorted by attribution value.

    Sorts patches by attribution (descending), splits into *n_bins* chunks,
    then re-orders bins so bin 0 has the lowest mean attribution.
    """
    if n_bins > len(raw_attributions):
        n_bins = len(raw_attributions)

    sorted_indices = np.argsort(raw_attributions)[::-1]
    chunks = np.array_split(sorted_indices, n_bins)

    temp = []
    for chunk in chunks:
        indices = chunk.tolist()
        if not indices:
            continue
        attrs = raw_attributions[indices]
        temp.append((indices, attrs))

    # Sort so lowest-attribution bin gets id 0
    temp.sort(key=lambda t: np.mean(t[1]))

    return [
        BinInfo(
            bin_id=i, min_value=float(np.min(a)), max_value=float(np.max(a)),
            patch_indices=idx, mean_attribution=float(np.mean(a)),
            total_attribution=float(np.sum(a)), n_patches=len(idx),
        )
        for i, (idx, a) in enumerate(temp)
    ]


# ---------------------------------------------------------------------------
# Perturbation (PIL-based, dataset-specific transforms)
# ---------------------------------------------------------------------------

def apply_binned_perturbation(original_tensor, bin_mask, perturbation_method="mean",
                              original_pil_image=None, dataset_name=None):
    """Apply perturbation to masked patches using PIL + dataset-specific transforms.

    Args:
        original_tensor: Unused when *original_pil_image* is provided (kept for fallback).
        bin_mask: Boolean numpy array (H, W).
        perturbation_method: Currently only ``"mean"`` is implemented.
        original_pil_image: Raw PIL image.
        dataset_name: Required for dataset-specific preprocessing.
    """
    if perturbation_method != "mean" or original_pil_image is None:
        if original_tensor is not None:
            return original_tensor.clone()
        raise ValueError("Either original_pil_image or original_tensor is required")

    mean_val = int(ImageStat.Stat(original_pil_image).mean[0])
    if original_pil_image.mode != 'RGB':
        original_pil_image = original_pil_image.convert('RGB')

    gray_layer = Image.new("RGB", original_pil_image.size, (mean_val, mean_val, mean_val))

    # Convert numpy (H, W) mask to PIL
    mask_uint8 = bin_mask.astype('uint8') * 255
    if (mask_uint8.shape[0], mask_uint8.shape[1]) != (original_pil_image.height, original_pil_image.width):
        pil_mask = Image.fromarray(mask_uint8, mode='L').resize(original_pil_image.size, Image.NEAREST)
    else:
        pil_mask = Image.fromarray(mask_uint8, mode='L')

    result_pil = original_pil_image.copy()
    result_pil.paste(gray_layer, (0, 0), mask=pil_mask)

    if not dataset_name:
        raise ValueError("dataset_name is required for proper preprocessing")
    processor = get_dataset_config(dataset_name).get_transforms('test')
    return processor(result_pil)


# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------

def load_image_and_attributions(classification_result):
    """Load raw PIL image and attribution data.  Returns (pil_image, raw_attributions, confidence, class_idx)."""
    pil_image = Image.open(classification_result.image_path).convert('RGB')

    if classification_result._cached_raw_attribution is not None:
        raw_attributions = classification_result._cached_raw_attribution
    else:
        attr_path = classification_result.attribution_paths.raw_attribution_path
        if attr_path is None:
            raise ValueError(f"No attribution path for {classification_result.image_path}")
        raw_attributions = np.load(attr_path)

    pred = classification_result.prediction
    return pil_image, raw_attributions, pred.confidence, pred.predicted_class_idx


def create_binned_perturbations(pil_image, raw_attributions, n_bins, perturbation_method,
                                patch_size, dataset_name):
    """Create bins and perturbed tensors.  Returns (bins, perturbed_tensors)."""
    bins = create_attribution_bins_from_patches(raw_attributions, n_bins)
    n_patches = patch_size_to_n_patches(patch_size)

    perturbed_tensors = []
    for b in bins:
        mask = create_patch_mask(b.patch_indices, (224, 224), n_patches, patch_size)
        perturbed = apply_binned_perturbation(None, mask, perturbation_method, pil_image, dataset_name)
        perturbed_tensors.append(perturbed)

    return bins, perturbed_tensors


def patch_size_to_n_patches(patch_size):
    """Convert patch_size to total patch count (224x224 images)."""
    grid = 224 // patch_size
    return grid * grid


def measure_bin_impacts(bins, perturbed_tensors, original_confidence, original_class_idx, model, device):
    """Measure confidence impact of perturbing each bin."""
    if not perturbed_tensors:
        return []

    batch_tensor = torch.stack(perturbed_tensors)
    predictions = batched_model_inference(model, batch_tensor, device, batch_size=len(batch_tensor))

    results = []
    for b, pred in zip(bins, predictions):
        delta = original_confidence - pred["confidence"]
        results.append({
            "bin_id": b.bin_id,
            "mean_attribution": b.mean_attribution,
            "total_attribution": b.total_attribution,
            "n_patches": b.n_patches,
            "confidence_delta": delta,
            "confidence_delta_abs": abs(delta),
            "class_changed": pred["predicted_class_idx"] != original_class_idx,
        })
    return results


def compute_saco_from_impacts(bin_results):
    """Compute SaCo score from bin impact results.  Returns (bin_results, saco_score, bin_biases)."""
    if len(bin_results) < 2:
        for r in bin_results:
            r["bin_attribution_bias"] = 0.0
        return bin_results, 0.0, np.zeros(len(bin_results))

    bin_results.sort(key=lambda x: x["total_attribution"], reverse=True)
    attributions = np.array([r["total_attribution"] for r in bin_results])
    impacts = np.array([r["confidence_delta"] for r in bin_results])

    saco_score, bin_biases = calculate_saco_vectorized_with_bias(attributions, impacts)
    for i, r in enumerate(bin_results):
        r["bin_attribution_bias"] = float(bin_biases[i])

    return bin_results, saco_score, bin_biases


def calculate_binned_saco_for_image(original_result, vit_model, config, device, n_bins=20, debug=False):
    """Calculate SaCo score for a single image.  Returns (saco_score, bin_results, [])."""
    try:
        dataset_name = config.file.dataset_name if hasattr(config.file, 'dataset_name') else None

        pil_image, raw_attributions, confidence, class_idx = load_image_and_attributions(original_result)

        patch_size = getattr(getattr(vit_model, 'cfg', None), 'patch_size', 16)
        if debug:
            print(f"Using patch_size={patch_size} for {dataset_name}")

        bins, perturbed_tensors = create_binned_perturbations(
            pil_image, raw_attributions, n_bins, config.perturb.method, patch_size, dataset_name,
        )

        vit_model.eval()
        bin_results = measure_bin_impacts(bins, perturbed_tensors, confidence, class_idx, vit_model, device)
        bin_results, saco_score, _ = compute_saco_from_impacts(bin_results)

        return saco_score, bin_results, []

    except Exception as e:
        print(f"Error in calculate_binned_saco_for_image: {e}")
        raise


# ---------------------------------------------------------------------------
# Dataset-level analysis (entry point)
# ---------------------------------------------------------------------------

def _get_or_compute_binned_results(config, original_results, vit_model, device, n_bins):
    """Load cached bin results or compute from scratch."""
    if config.file.use_cached_perturbed:
        cache_path = config.file.output_dir / config.file.use_cached_perturbed
        print(f"Attempting to load cached results from: {cache_path}")
        if cache_path.exists():
            print("Cache file found! Loading...")
            return pd.read_csv(cache_path)
        print("Cache file not found. Proceeding with computation.")

    print(f"Computing binned SaCo results for {len(original_results)} images...")
    all_bin_results = []
    for original_result in tqdm(original_results, desc=f"Processing with {n_bins} bins"):
        try:
            saco_score, bin_results, _ = calculate_binned_saco_for_image(
                original_result, vit_model, config, device, n_bins,
            )
            for r in bin_results:
                r["image_name"] = str(original_result.image_path)
                r["saco_score"] = saco_score
                all_bin_results.append(r)
        except Exception as e:
            import traceback
            print(f"Error processing {original_result.image_path.name}: {e}")
            if "do not match" in str(e):
                traceback.print_exc()
            continue

    return pd.DataFrame(all_bin_results)


def run_binned_attribution_analysis(config, vit_model, original_results, device, n_bins=None):
    """Run binned SaCo analysis for an entire dataset.

    Entry point called by ``experiments/pipeline.py``.  Determines *n_bins*
    from config if not specified, computes (or loads cached) per-image SaCo,
    then performs post-hoc analysis and saves results.
    """
    vit_model.to(device)
    vit_model.eval()

    # Determine n_bins from config
    if n_bins is None:
        is_patch32 = False
        if hasattr(config.classify, 'clip_model_name') and config.classify.clip_model_name:
            model_name = config.classify.clip_model_name.lower()
            is_patch32 = "patch32" in model_name or "b-32" in model_name or "b32" in model_name
        n_bins = config.faithfulness.n_bins_b32 if is_patch32 else config.faithfulness.n_bins

    print(f"=== BINNED SACO ANALYSIS (n_bins={n_bins}) ===")

    bin_results_df = _get_or_compute_binned_results(config, original_results, vit_model, device, n_bins)
    if bin_results_df.empty:
        print("No results were generated or loaded. Aborting analysis.")
        return {}

    # Post-hoc analysis
    print("\n--- Performing post-hoc analysis on results ---")
    analysis_results = {"bin_results": bin_results_df}

    saco_df = bin_results_df[['image_name', 'saco_score']].drop_duplicates().reset_index(drop=True)
    analysis_results["saco_scores"] = saco_df

    # Faithfulness vs correctness
    saco_scores_map = pd.Series(saco_df.saco_score.values, index=saco_df.image_name).to_dict()
    faithfulness_df = _analyze_faithfulness_vs_correctness(saco_scores_map, original_results)
    analysis_results["faithfulness_correctness"] = faithfulness_df

    # Attribution patterns (filter to correct predictions)
    patterns_df = faithfulness_df.dropna(subset=['saco_score'])
    if 'filename' in patterns_df.columns:
        patterns_df = patterns_df.copy()
        patterns_df['class'] = patterns_df['true_class']
    patterns_df = patterns_df[patterns_df['is_correct']]
    analysis_results["attribution_patterns"] = patterns_df

    # Summary
    if not saco_df.empty:
        print(f"\nBinned SaCo Analysis Summary:")
        print(f"  Number of images: {len(saco_df)}")
        print(f"  Number of bins: {n_bins}")
        print(f"  Average SaCo: {saco_df['saco_score'].mean():.4f}")
        print(f"  Std SaCo: {saco_df['saco_score'].std():.4f}")

    # Save
    print("\nSaving derived analysis files...")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    for name, df in analysis_results.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            if name == "bin_results":
                ds_name = config.file.output_dir.name
                save_path = config.file.output_dir / f"{ds_name}_bin_results.csv"
                df.to_csv(save_path, index=False)
                print(f"Saved bin results to {save_path}")
            save_path = config.file.output_dir / f"analysis_{name}_binned_{timestamp}.csv"
            df.to_csv(save_path, index=False)
            print(f"Saved {name} to {save_path}")

    print("=== BINNED SACO ANALYSIS COMPLETE ===")
    return analysis_results


def _analyze_faithfulness_vs_correctness(saco_scores, original_results):
    """Join SaCo scores with classification correctness.  Returns DataFrame."""
    rows = []
    for res in original_results:
        if res.prediction is None:
            continue
        p = res.prediction
        ap = res.attribution_paths
        rows.append({
            'filename': str(res.image_path),
            'saco_score': saco_scores.get(str(res.image_path)),
            'predicted_class': p.predicted_class_label,
            'predicted_idx': p.predicted_class_idx,
            'true_class': res.true_label,
            'is_correct': p.predicted_class_label == res.true_label,
            'confidence': p.confidence,
            'attribution_path': str(ap.attribution_path) if ap else None,
            'probabilities': p.probabilities,
        })
    return pd.DataFrame(rows)


def extract_saco_summary(saco_analysis):
    """Extract overall, per-class, and by-correctness SaCo statistics."""
    if not saco_analysis or "faithfulness_correctness" not in saco_analysis:
        return {}

    fc_df = saco_analysis["faithfulness_correctness"]
    result = {
        'mean': fc_df["saco_score"].mean(),
        'std': fc_df["saco_score"].std(),
        'n_samples': len(fc_df),
        'per_class': fc_df.groupby('true_class')['saco_score'].agg(['mean', 'std', 'count']).to_dict('index'),
        'by_correctness': fc_df.groupby('is_correct')['saco_score'].agg(['mean', 'std', 'count']).to_dict('index'),
    }
    return result
