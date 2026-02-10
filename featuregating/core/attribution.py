"""
Attribution module — TransLRP with optional feature-gradient gating.

Public API:
    compute_attribution  — canonical orchestrator (returns dict)

Internal helpers:
    apply_gradient_gating_to_cam  — adapter into core/gating.py
    compute_layer_attribution     — per-layer attention rollout loop
    avg_heads                     — gradient-weighted head averaging
    apply_self_attention_rules    — self-attention propagation rule
    _postprocess_attribution      — reshape/interpolate/normalize output

Model interaction (in models/inference.py):
    run_model_forward_backward    — forward/backward with hook data
    setup_hooks                   — hook wiring for vit_prisma
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from vit_prisma.models.base_vit import HookedViT

from featuregating.core.config import PipelineConfig
from featuregating.core.gating import apply_feature_gradient_gating
from featuregating.models.inference import run_model_forward_backward, setup_hooks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def apply_gradient_gating_to_cam(
    cam_pos_avg: torch.Tensor, layer_idx: int, gradients: Dict[str, torch.Tensor], residuals: Dict[int, torch.Tensor],
    steering_resources: Dict[int, Dict[str, Any]], config: PipelineConfig, debug: bool = False
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Adapter: unpack hook data and PipelineConfig, delegate to gating.apply_feature_gradient_gating."""
    resid_grad_key = f"blocks.{layer_idx}.hook_resid_post_grad"

    if resid_grad_key not in gradients or layer_idx not in residuals:
        return cam_pos_avg, {}

    # Get gradient and residual (already on GPU from hooks)
    residual_grad = gradients[resid_grad_key]
    residual_tensor = residuals[layer_idx]
    sae = steering_resources[layer_idx]["sae"]

    # Compute SAE codes on-demand
    with torch.no_grad():
        _, codes_gpu = sae.encode(residual_tensor)

    if residual_grad.dim() == 3:
        residual_grad = residual_grad[0]
    residual_grad = residual_grad[1:]  # Remove CLS

    if codes_gpu.dim() == 3:
        codes_gpu = codes_gpu[0]
    codes_gpu = codes_gpu[1:]  # Remove CLS

    # Apply feature gradient gating - get parameters from config
    gating_config = {
        'clamp_max': config.classify.boosting.clamp_max,
        'gate_construction': config.classify.boosting.gate_construction,
        'shuffle_decoder': config.classify.boosting.shuffle_decoder,
        'shuffle_decoder_seed': config.classify.boosting.shuffle_decoder_seed,
        'active_feature_threshold': config.classify.boosting.active_feature_threshold,
    }

    gated_cam, layer_debug = apply_feature_gradient_gating(
        cam_pos_avg=cam_pos_avg,
        residual_grad=residual_grad,
        residual=residual_tensor[0][1:],
        sae_codes=codes_gpu,
        sae=sae,
        config=gating_config,
        debug=debug
    )

    return gated_cam, layer_debug


def compute_layer_attribution(
    model_cfg: HookedViTConfig, activations: Dict[str, torch.Tensor], gradients: Dict[str, torch.Tensor],
    residuals: Dict[int, torch.Tensor], feature_gradient_layers: List[int],
    steering_resources: Optional[Dict[int, Dict[str, Any]]], config: PipelineConfig, device: torch.device,
    debug: bool = False
) -> Tuple[torch.Tensor, Dict[int, Dict[str, Any]]]:
    """Compute attribution by iterating through layers and applying attention rules."""
    attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(model_cfg.n_layers)]
    num_tokens = activations[attn_hook_names[0]].shape[-1]
    R_pos = torch.eye(num_tokens, num_tokens, device=device)

    debug_info_per_layer = {}

    for i in range(model_cfg.n_layers):
        hname = f"blocks.{i}.attn.hook_pattern"
        grad = gradients[hname + "_grad"]
        cam = activations[hname]
        cam_pos_avg = avg_heads(cam, grad)

        # Apply feature gradient gating if configured
        if (i in feature_gradient_layers and steering_resources is not None and i in steering_resources):
            cam_pos_avg, layer_debug = apply_gradient_gating_to_cam(
                cam_pos_avg, i, gradients, residuals, steering_resources, config, debug=debug
            )
            if debug:
                debug_info_per_layer[i] = layer_debug

        R_pos = R_pos + apply_self_attention_rules(R_pos, cam_pos_avg)

    transformer_attribution_pos = R_pos[0, 1:].clone()

    return transformer_attribution_pos, debug_info_per_layer


def avg_heads(cam: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


def apply_self_attention_rules(R_ss: torch.Tensor, cam_ss: torch.Tensor) -> torch.Tensor:
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def _postprocess_attribution(
    transformer_attribution_pos: torch.Tensor, img_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape patch attribution to spatial map, interpolate to image size, and normalize.

    Used by compute_attribution for output formatting.

    Returns:
        (attribution_map_2d, raw_patch_map_1d): normalized interpolated map and raw patch vector.
    """
    raw_patch_map = transformer_attribution_pos.detach().cpu().numpy()

    side_len = int(np.sqrt(transformer_attribution_pos.size(0)))
    attribution_reshaped = transformer_attribution_pos.reshape(1, 1, side_len, side_len)
    attribution_pos_np = F.interpolate(
        attribution_reshaped, size=(img_size, img_size), mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()

    # Min-max normalize
    val_range = np.max(attribution_pos_np) - np.min(attribution_pos_np)
    if val_range > 1e-8:
        attribution_pos_np = (attribution_pos_np - np.min(attribution_pos_np)) / (val_range + 1e-8)

    return attribution_pos_np, raw_patch_map


# ---------------------------------------------------------------------------
# Canonical public API
# ---------------------------------------------------------------------------

def compute_attribution(
    model_prisma: HookedViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],
    device: Optional[torch.device] = None,
    img_size: int = 224,
    steering_resources: Optional[Dict[int, Dict[str, Any]]] = None,
    enable_feature_gradients: bool = True,
    feature_gradient_layers: Optional[List[int]] = None,
    clip_classifier: Optional[Any] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """Orchestrator: full TransLRP attribution pipeline with optional feature-gradient gating.

    This is the single canonical entrypoint for attribution computation.
    Handles device resolution, input preparation, hook wiring, forward/backward,
    layer-by-layer attribution (with optional gating), and post-processing.

    Returns:
        dict with keys: ``predictions``, ``attribution_positive``,
        ``raw_attribution``, ``debug_info``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Default to applying feature gradients at layers 9-10 if not specified
    if feature_gradient_layers is None:
        feature_gradient_layers = [9, 10] if enable_feature_gradients else []

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    # Setup hooks for data collection
    fwd_hooks, bwd_hooks, gradients, activations, residuals = setup_hooks(model_prisma, feature_gradient_layers)

    # Run model with hooks to collect gradients and activations
    with model_prisma.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, reset_hooks_end=True):
        prediction_result, predicted_class_idx = run_model_forward_backward(
            model_prisma, input_tensor, clip_classifier, device
        )

    # Add class label to prediction result
    prediction_result["predicted_class_label"] = idx_to_class.get(predicted_class_idx, f"class_{predicted_class_idx}")

    if not gradients:
        raise RuntimeError("No gradients captured!")

    # Compute attribution
    transformer_attribution_pos, debug_info_per_layer = compute_layer_attribution(
        model_prisma.cfg, activations, gradients, residuals, feature_gradient_layers, steering_resources, config, device,
        debug=debug
    )

    # Post-process: reshape to spatial map, interpolate, normalize
    attribution_pos_np, raw_patch_map = _postprocess_attribution(transformer_attribution_pos, img_size)

    return {
        "predictions": prediction_result,
        "attribution_positive": attribution_pos_np,
        "raw_attribution": raw_patch_map,
        "debug_info": debug_info_per_layer if debug else {},
    }
