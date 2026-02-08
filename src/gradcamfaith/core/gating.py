"""
Feature Gradient Gating for TransLRP

Implements Option 4: Using SAE feature gradients to create cleaner
per-patch attribution scalars.

Public API:
    compute_feature_gradient_gate — compute per-patch gating multipliers
    apply_feature_gradient_gating  — apply gate to attention CAM
"""

from typing import Any, Dict, Optional, Tuple

import torch



def _extract_decoder(sae: Any) -> torch.Tensor:
    """Extract decoder weight matrix from SAE, handling implementation variants.

    Returns decoder as [d_model, n_features].
    """
    if hasattr(sae, 'W_dec'):
        return sae.W_dec.T  # StandardSparseAutoencoder: W_dec is [n_features, d_model]
    elif hasattr(sae, 'decoder'):
        return sae.decoder.weight.t()
    else:
        raise AttributeError(
            f"SAE of type {type(sae).__name__} has no decoder attribute (W_dec or decoder)"
        )


def _compute_patch_scores(
    gate_construction: str,
    sae_codes: torch.Tensor,
    feature_grads: torch.Tensor,
    residual_grad: torch.Tensor,
    residual: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-patch scalar scores for one of the four gate construction modes.

    Returns:
        s_t: Per-patch score [n_patches]
        contributions: Per-patch contribution tensor (for debug)
    """
    if gate_construction == "activation_only":
        contributions = sae_codes
        s_t = sae_codes.sum(dim=1)

    elif gate_construction == "gradient_only":
        contributions = feature_grads
        s_t = feature_grads.sum(dim=1)

    elif gate_construction == "combined":
        contributions = sae_codes * feature_grads
        s_t = contributions.sum(dim=1)

    elif gate_construction == "no_SAE":
        if residual is None:
            raise ValueError("residual must be provided for no_SAE gate construction")
        contributions = residual_grad * residual
        s_t = contributions.sum(dim=1)

    else:
        raise ValueError(f"Unknown gate_construction type: {gate_construction}")

    return s_t, contributions


def _collect_gate_debug_info(
    gate: torch.Tensor,
    s_t: torch.Tensor,
    contributions: torch.Tensor,
    sae_codes: torch.Tensor,
    feature_grads: torch.Tensor,
    gate_construction: str,
    active_feature_threshold: float,
) -> Dict[str, Any]:
    """Collect detailed debug information for gate analysis.

    Includes per-patch gate statistics and, for SAE-based constructions,
    sparse per-patch feature indices/activations/gradients/contributions.
    """
    total_contribution_magnitude = torch.abs(contributions).sum(dim=1)

    debug_info: Dict[str, Any] = {
        'gate_values': gate.detach().cpu().numpy(),
        'contribution_sum': s_t.detach().cpu().numpy(),
        'total_contribution_magnitude': total_contribution_magnitude.detach().cpu().numpy(),
        'mean_gate': gate.mean().item(),
        'std_gate': gate.std().item(),
    }

    # Sparse features only for SAE-based gate constructions
    # (no_SAE uses d_model-sized contributions, not n_features-sized)
    if gate_construction != "no_SAE":
        active_mask = sae_codes > active_feature_threshold
        sparse_indices = []
        sparse_activations = []
        sparse_gradients = []
        sparse_contributions = []

        for patch_idx in range(sae_codes.shape[0]):
            mask = active_mask[patch_idx]
            indices = torch.where(mask)[0]
            sparse_indices.append(indices.detach().cpu().numpy())
            sparse_activations.append(sae_codes[patch_idx, mask].detach().cpu().numpy())
            sparse_gradients.append(feature_grads[patch_idx, mask].detach().cpu().numpy())
            sparse_contributions.append(contributions[patch_idx, mask].detach().cpu().numpy())

        debug_info['sparse_features_indices'] = sparse_indices
        debug_info['sparse_features_activations'] = sparse_activations
        debug_info['sparse_features_gradients'] = sparse_gradients
        debug_info['sparse_features_contributions'] = sparse_contributions

    return debug_info


def _apply_gate_to_cam(
    cam_pos_avg: torch.Tensor,
    feature_gate: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply gate vector to attention CAM, handling CLS token dimension.

    Returns:
        gated_cam: Gate-modulated CAM
        cam_delta: Difference from original CAM
        patch_attribution_deltas: Per-patch signed sum of column deltas
    """
    feature_gate = feature_gate.to(cam_pos_avg.device)
    has_cls = cam_pos_avg.shape[0] == feature_gate.shape[0] + 1

    if has_cls:
        gated_cam = cam_pos_avg.clone()
        gated_cam[:, 1:] = gated_cam[:, 1:] * feature_gate.unsqueeze(0)
    else:
        gated_cam = cam_pos_avg * feature_gate.unsqueeze(0)

    cam_delta = gated_cam - cam_pos_avg

    if has_cls:
        patch_attribution_deltas = cam_delta[:, 1:].sum(dim=0)
    else:
        patch_attribution_deltas = cam_delta.sum(dim=0)

    return gated_cam, cam_delta, patch_attribution_deltas



def compute_feature_gradient_gate(
    residual_grad: torch.Tensor,
    residual: Optional[torch.Tensor],
    sae_codes: torch.Tensor,
    sae_decoder: torch.Tensor,
    clamp_max: float = 10.0,
    gate_construction: str = "combined",
    shuffle_decoder: bool = False,
    shuffle_decoder_seed: int = 12345,
    active_feature_threshold: float = 0.1,
    debug: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute per-patch gating multipliers using SAE feature gradients.

    This implements the feature gradient decomposition:
    1. Project gradient to feature space: h = D^T g
    2. Weight by activation: s_k = h_k * f_k
    3. Sum all features: s = Σ_k s_k
    4. Map to multiplier: w = clamp_max^(tanh(normalize(s)))

    Args:
        residual_grad: Gradient w.r.t. residual [n_patches, d_model]
        residual: Residual activations (required only for no_SAE mode)
        sae_codes: SAE feature activations [n_patches, n_features]
        sae_decoder: SAE decoder matrix [d_model, n_features]
        clamp_max: Base for exponential gate mapping (gate range: [1/clamp_max, clamp_max])
        gate_construction: "activation_only", "gradient_only", "combined", or "no_SAE"
        shuffle_decoder: Whether to shuffle decoder columns to break semantic alignment
        shuffle_decoder_seed: Random seed for decoder shuffling (for reproducibility)
        active_feature_threshold: Threshold for considering a feature "active" in debug mode
        debug: Whether to return detailed debug information

    Returns:
        gate: Per-patch multipliers [n_patches]
        debug_info: Dictionary with debug information
    """
    # Optional decoder shuffle (ablation control)
    if shuffle_decoder:
        g = torch.Generator(device=sae_decoder.device)
        g.manual_seed(shuffle_decoder_seed)
        shuffle_perm = torch.randperm(sae_decoder.shape[1], generator=g, device=sae_decoder.device)
        sae_decoder = sae_decoder[:, shuffle_perm]

    # Project gradient to feature space: h = D^T g
    feature_grads = residual_grad @ sae_decoder  # [n_patches, n_features]

    # Score construction (4 modes)
    s_t, contributions = _compute_patch_scores(
        gate_construction, sae_codes, feature_grads, residual_grad, residual
    )

    # Normalize via robust z-score (MAD) and map to gate multiplier
    s_median = s_t.median()
    s_mad = (s_t - s_median).abs().median() + 1e-8
    s_norm = (s_t - s_median) / (1.4826 * s_mad)
    gate = (clamp_max ** torch.tanh(s_norm)).detach()

    # Debug info
    if debug:
        debug_info = _collect_gate_debug_info(
            gate, s_t, contributions, sae_codes, feature_grads,
            gate_construction, active_feature_threshold,
        )
    else:
        debug_info = {
            'mean_gate': gate.mean().item(),
            'std_gate': gate.std().item(),
        }

    return gate, debug_info


def apply_feature_gradient_gating(
    cam_pos_avg: torch.Tensor,
    residual_grad: torch.Tensor,
    residual: Optional[torch.Tensor],
    sae_codes: torch.Tensor,
    sae: Any,
    config: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply feature gradient gating to attention CAM.

    Entry point: extracts decoder from SAE, computes gate via
    compute_feature_gradient_gate, applies gate to CAM.

    Args:
        cam_pos_avg: Averaged attention CAM [n_patches, n_patches]
        residual_grad: Gradient w.r.t. residual (spatial tokens only) [n_patches-1, d_model]
        residual: Residual activations (required only for no_SAE mode)
        sae_codes: SAE codes (spatial tokens only) [n_patches-1, n_features]
        sae: SAE model with decoder weight
        config: Configuration dictionary for gating parameters
        debug: Whether to collect debug information

    Returns:
        gated_cam: Modified attention CAM
        debug_info: Combined debug information
    """
    if config is None:
        config = {}

    decoder = _extract_decoder(sae)

    feature_gate, feature_debug = compute_feature_gradient_gate(
        residual_grad=residual_grad,
        residual=residual,
        sae_codes=sae_codes,
        sae_decoder=decoder,
        clamp_max=config.get('clamp_max', 10.0),
        gate_construction=config.get('gate_construction', 'combined'),
        shuffle_decoder=config.get('shuffle_decoder', False),
        shuffle_decoder_seed=config.get('shuffle_decoder_seed', 12345),
        active_feature_threshold=config.get('active_feature_threshold', 0.1),
        debug=debug,
    )

    gated_cam, cam_delta, patch_attribution_deltas = _apply_gate_to_cam(
        cam_pos_avg, feature_gate,
    )

    debug_info = {
        'feature_gating': feature_debug,
        'combined_gate': feature_gate.detach().cpu().numpy() if debug else None,
        'cam_delta': cam_delta.detach().cpu().numpy() if debug else None,
        'patch_attribution_deltas': patch_attribution_deltas.detach().cpu().numpy() if debug else None,
    }

    return gated_cam, debug_info
