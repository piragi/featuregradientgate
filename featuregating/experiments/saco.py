"""
SaCo (Sorting Attribution Consistency) metric.

Bins patches by attribution value, perturbs each bin, and checks whether
higher-attribution bins cause proportionally larger confidence drops.

Flow:
    run_binned_attribution_analysis  (entry point, called by pipeline)
      _get_or_compute_binned_results
        _saco_for_image  (per-image)
          create_attribution_bins  → BinInfo list
          _perturb_bins            → perturbed tensors
          _measure_bin_drops       → confidence drops array
          calculate_saco           → SaCo score
      _join_saco_with_correctness, save results
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from featuregating.experiments.faithfulness import (
    apply_baseline_perturbation,
    create_patch_mask,
)


# ---------------------------------------------------------------------------
# Core SaCo algorithm
# ---------------------------------------------------------------------------

def calculate_saco(attributions, confidence_drops):
    """Pairwise concordance: do higher-attribution bins cause bigger drops?

    For each pair (i, j), computes a signed weight:
    - Concordant (signs agree) or tied: weight = attr_diff (preserves sign)
    - Discordant (signs disagree): weight = -attr_diff (flips sign)

    Score = sum(weights) / sum(|weights|).  Range [-1, 1].
    """
    n = len(attributions)
    if n < 2:
        return 0.0

    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            attr_diff = attributions[i] - attributions[j]
            drop_diff = confidence_drops[i] - confidence_drops[j]
            if attr_diff * drop_diff < 0:  # discordant
                w = -attr_diff
            else:  # concordant or tied
                w = attr_diff
            numerator += w
            denominator += abs(w)

    return numerator / denominator if denominator > 0 else 0.0


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


def create_attribution_bins(raw_attributions, n_bins=20):
    """Create equal-size bins sorted by attribution value.

    Sorts patches by attribution (descending), splits into *n_bins* chunks,
    then re-orders bins so bin 0 has the lowest mean attribution.
    """
    n_bins = min(n_bins, len(raw_attributions))
    sorted_indices = np.argsort(raw_attributions)[::-1]
    chunks = np.array_split(sorted_indices, n_bins)

    temp = []
    for chunk in chunks:
        indices = chunk.tolist()
        if not indices:
            continue
        attrs = raw_attributions[indices]
        temp.append((indices, attrs))

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
# Per-image SaCo computation
# ---------------------------------------------------------------------------

def _load_tensor_and_attributions(result):
    """Load cached tensor and attribution from a ClassificationResult.

    Returns (image_tensor, raw_attributions, confidence).
    image_tensor is numpy (C, H, W).
    """
    if result._cached_tensor is not None:
        tensor = result._cached_tensor
    else:
        from featuregating.datasets.dataset_config import get_dataset_config
        ds_cfg = get_dataset_config("imagenet")  # fallback
        transform = ds_cfg.get_transforms('test')
        pil = Image.open(result.image_path).convert('RGB')
        tensor = transform(pil).cpu().numpy()

    if result._cached_raw_attribution is not None:
        raw_attr = result._cached_raw_attribution
    else:
        attr_path = result.attribution_paths.raw_attribution_path
        if attr_path is None:
            raise ValueError(f"No attribution path for {result.image_path}")
        raw_attr = np.load(attr_path)

    pred = result.prediction
    return tensor, raw_attr, pred.confidence


def _perturb_bins(bins, image_tensor, patch_size):
    """Create perturbed tensors for each bin.

    Applies mean-fill perturbation directly on the transformed tensor,
    ensuring exact alignment with the model's patch grid.
    """
    n_patches = (224 // patch_size) ** 2
    H, W = image_tensor.shape[1], image_tensor.shape[2]
    perturbed = []
    for b in bins:
        mask = create_patch_mask(b.patch_indices, (H, W), n_patches, patch_size)
        arr = apply_baseline_perturbation(image_tensor, mask, "mean")
        perturbed.append(torch.from_numpy(arr).float())
    return perturbed


def _classify_batch(model, batch_tensor, device):
    """Run inference on a batch. Returns (predicted_indices, confidences) as numpy arrays."""
    model.eval()
    with torch.no_grad():
        logits = model(batch_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        idxs = torch.argmax(probs, dim=1)
        confs = probs[torch.arange(len(idxs)), idxs]
    return idxs.cpu().numpy(), confs.cpu().numpy()


def _measure_bin_drops(perturbed_tensors, original_confidence, model, device):
    """Measure confidence drop when each bin is perturbed. Returns numpy array."""
    if not perturbed_tensors:
        return np.array([])
    batch = torch.stack(perturbed_tensors)
    _, confidences = _classify_batch(model, batch, device)
    return original_confidence - confidences


def _saco_for_image(result, model, device, n_bins=20):
    """Compute SaCo score for a single image. Returns (saco_score, bin_records)."""
    tensor, raw_attr, confidence = _load_tensor_and_attributions(result)

    patch_size = getattr(getattr(model, 'cfg', None), 'patch_size', 16)
    bins = create_attribution_bins(raw_attr, n_bins)
    perturbed = _perturb_bins(bins, tensor, patch_size)

    model.eval()
    drops = _measure_bin_drops(perturbed, confidence, model, device)

    # Sort descending by attribution so that upper-triangle attr_diffs are
    # positive for high-attr bins — gives positive SaCo for concordant data.
    order = np.argsort([-b.total_attribution for b in bins])
    attributions = np.array([bins[k].total_attribution for k in order])
    drops_sorted = drops[order]
    saco_score = calculate_saco(attributions, drops_sorted)

    bin_records = [
        {
            "bin_id": b.bin_id,
            "mean_attribution": b.mean_attribution,
            "total_attribution": b.total_attribution,
            "n_patches": b.n_patches,
            "confidence_delta": float(drops[i]),
        }
        for i, b in enumerate(bins)
    ]
    return saco_score, bin_records


# ---------------------------------------------------------------------------
# Dataset-level analysis (entry point)
# ---------------------------------------------------------------------------

def _get_or_compute_binned_results(original_results, model, device, n_bins):
    """Compute binned SaCo results for all images."""
    print(f"Computing binned SaCo results for {len(original_results)} images...")
    all_records = []
    for result in tqdm(original_results, desc=f"Processing with {n_bins} bins"):
        try:
            saco_score, bin_records = _saco_for_image(result, model, device, n_bins)
            for r in bin_records:
                r["image_name"] = str(result.image_path)
                r["saco_score"] = saco_score
                all_records.append(r)
        except Exception as e:
            print(f"Error processing {result.image_path.name}: {e}")
            continue

    return pd.DataFrame(all_records)


def run_binned_attribution_analysis(config, vit_model, original_results, device, n_bins=None):
    """Run binned SaCo analysis for an entire dataset.

    Entry point called by ``experiments/pipeline.py``.  Determines *n_bins*
    from config if not specified, computes (or loads cached) per-image SaCo,
    then joins with classification correctness and saves results.
    """
    vit_model.to(device)
    vit_model.eval()

    if n_bins is None:
        is_patch32 = False
        if hasattr(config.classify, 'clip_model_name') and config.classify.clip_model_name:
            model_name = config.classify.clip_model_name.lower()
            is_patch32 = "patch32" in model_name or "b-32" in model_name or "b32" in model_name
        n_bins = config.faithfulness.n_bins_b32 if is_patch32 else config.faithfulness.n_bins

    print(f"=== BINNED SACO ANALYSIS (n_bins={n_bins}) ===")

    bin_results_df = _get_or_compute_binned_results(original_results, vit_model, device, n_bins)
    if bin_results_df.empty:
        print("No results were generated or loaded. Aborting analysis.")
        return {}

    analysis_results = {"bin_results": bin_results_df}

    saco_df = bin_results_df[['image_name', 'saco_score']].drop_duplicates().reset_index(drop=True)
    analysis_results["saco_scores"] = saco_df

    saco_map = pd.Series(saco_df.saco_score.values, index=saco_df.image_name).to_dict()
    faithfulness_df = _join_saco_with_correctness(saco_map, original_results)
    analysis_results["faithfulness_correctness"] = faithfulness_df

    if not saco_df.empty:
        print(f"\nBinned SaCo Analysis Summary:")
        print(f"  Number of images: {len(saco_df)}")
        print(f"  Number of bins: {n_bins}")
        print(f"  Average SaCo: {saco_df['saco_score'].mean():.4f}")
        print(f"  Std SaCo: {saco_df['saco_score'].std():.4f}")

    # Save debug outputs if enabled
    if config.classify.boosting.debug_mode:
        debug_dir = config.file.output_dir / "debug"
        debug_dir.mkdir(exist_ok=True, parents=True)
        bin_results_df.to_csv(debug_dir / "saco_bin_results.csv", index=False)

    print("=== BINNED SACO ANALYSIS COMPLETE ===")
    return analysis_results


def _join_saco_with_correctness(saco_scores, original_results):
    """Join SaCo scores with classification correctness. Returns DataFrame."""
    rows = []
    for res in original_results:
        if res.prediction is None:
            continue
        p = res.prediction
        rows.append({
            'filename': str(res.image_path),
            'saco_score': saco_scores.get(str(res.image_path)),
            'predicted_class': p.predicted_class_label,
            'predicted_idx': p.predicted_class_idx,
            'true_class': res.true_label,
            'is_correct': p.predicted_class_label == res.true_label,
            'confidence': p.confidence,
        })
    return pd.DataFrame(rows)


def extract_saco_summary(saco_analysis):
    """Extract overall, per-class, and by-correctness SaCo statistics."""
    if not saco_analysis or "faithfulness_correctness" not in saco_analysis:
        return {}

    fc_df = saco_analysis["faithfulness_correctness"]
    return {
        'mean': fc_df["saco_score"].mean(),
        'std': fc_df["saco_score"].std(),
        'n_samples': len(fc_df),
        'per_class': fc_df.groupby('true_class')['saco_score'].agg(['mean', 'std', 'count']).to_dict('index'),
        'by_correctness': fc_df.groupby('is_correct')['saco_score'].agg(['mean', 'std', 'count']).to_dict('index'),
    }
