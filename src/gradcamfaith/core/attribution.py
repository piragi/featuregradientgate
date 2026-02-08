"""
Attribution module — TransLRP with optional feature-gradient gating.

Public API:
    compute_attribution  — canonical orchestrator (returns dict)

Internal helpers:
    apply_gradient_gating_to_cam  — adapter into core/gating.py
    compute_layer_attribution     — per-layer attention rollout loop
    avg_heads                     — gradient-weighted head averaging
    apply_self_attention_rules    — self-attention propagation rule
    run_model_forward_backward    — forward/backward with hook data
    setup_hooks                   — hook wiring for vit_prisma
    _postprocess_attribution      — reshape/interpolate/normalize output

Deprecated (delegate to compute_attribution):
    transmm_prisma_enhanced
    generate_attribution_prisma_enhanced
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from vit_prisma.models.base_vit import HookedSAEViT, HookedViT

from gradcamfaith.core.config import PipelineConfig
from gradcamfaith.core.gating import apply_feature_gradient_gating


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


def run_model_forward_backward(
    model_prisma: HookedViT, input_tensor: torch.Tensor, clip_classifier: Optional[Any], device: torch.device
) -> Tuple[Dict[str, Any], int]:
    """Run forward and backward pass, return predictions and predicted class."""
    if clip_classifier is not None:
        clip_result = clip_classifier.forward(input_tensor, requires_grad=True)
        logits = clip_result["logits"]
        probabilities = clip_result["probabilities"]
        predicted_class_idx = clip_result["predicted_class_idx"]
    else:
        logits = model_prisma(input_tensor)
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(probabilities, dim=-1).item()

    # Backward pass
    num_classes = logits.size(-1)
    one_hot = torch.zeros((1, num_classes), dtype=torch.float32, device=device)
    one_hot[0, int(predicted_class_idx)] = 1
    one_hot.requires_grad_(True)
    loss = torch.sum(one_hot * logits)
    loss.backward(retain_graph=False)

    prediction_result = {
        "logits":
        logits.detach().cpu().numpy(),
        "probabilities":
        probabilities.squeeze().cpu().detach().numpy().tolist()
        if isinstance(probabilities, torch.Tensor) else probabilities,
        "predicted_class_idx":
        predicted_class_idx,
    }

    return prediction_result, int(predicted_class_idx)


def setup_hooks(model_prisma: HookedViT, feature_gradient_layers: List[int]) -> Tuple[List, List, Dict, Dict, Dict]:
    """Setup forward and backward hooks, return hook lists and storage dictionaries."""
    gradients = {}
    activations = {}
    residuals = {}

    attn_hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(model_prisma.cfg.n_layers)]

    def save_activation_hook(tensor: torch.Tensor, hook: Any):
        activations[hook.name] = tensor.detach()

    def save_gradient_hook(grad: torch.Tensor, hook: Any):
        if grad is not None:
            gradients[hook.name + "_grad"] = grad.detach()

    fwd_hooks = [(name, save_activation_hook) for name in attn_hook_names]
    bwd_hooks = [(name, save_gradient_hook) for name in attn_hook_names]

    # Add residual hooks for feature gradient layers
    all_resid_layers = set(feature_gradient_layers)
    if all_resid_layers:

        def save_resid_hook(tensor, hook):
            layer_idx = int(hook.name.split('.')[1])
            if layer_idx in feature_gradient_layers:
                residuals[layer_idx] = tensor.detach()
            return tensor

        for layer_idx in all_resid_layers:
            resid_hook_name = f"blocks.{layer_idx}.hook_resid_post"
            fwd_hooks.append((resid_hook_name, save_resid_hook))
            bwd_hooks.append((resid_hook_name, save_gradient_hook))

    return fwd_hooks, bwd_hooks, gradients, activations, residuals


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


# ---------------------------------------------------------------------------
# Deprecated legacy entrypoints — delegate to compute_attribution
# ---------------------------------------------------------------------------

def transmm_prisma_enhanced(
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
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, Dict[int, Dict[str, Any]]]:
    """DEPRECATED — use ``compute_attribution`` instead.

    Thin wrapper that delegates to compute_attribution and repacks the dict
    output into the legacy 4-tuple format. Scheduled for removal in WP-07.
    """
    warnings.warn(
        "transmm_prisma_enhanced is deprecated, use compute_attribution instead",
        DeprecationWarning,
        stacklevel=2,
    )
    result = compute_attribution(
        model_prisma=model_prisma,
        input_tensor=input_tensor,
        config=config,
        idx_to_class=idx_to_class,
        device=device,
        img_size=img_size,
        steering_resources=steering_resources,
        enable_feature_gradients=enable_feature_gradients,
        feature_gradient_layers=feature_gradient_layers,
        clip_classifier=clip_classifier,
        debug=debug,
    )
    return (
        result["predictions"],
        result["attribution_positive"],
        result["raw_attribution"],
        result["debug_info"],
    )


def generate_attribution_prisma_enhanced(
    model: HookedSAEViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],
    device: Optional[torch.device] = None,
    steering_resources: Optional[Dict[int, Dict[str, Any]]] = None,
    enable_feature_gradients: bool = True,
    feature_gradient_layers: Optional[List[int]] = None,
    clip_classifier: Optional[Any] = None,
) -> Dict[str, Any]:
    """DEPRECATED — use ``compute_attribution`` instead.

    Thin wrapper that reads debug_mode from config and delegates to
    compute_attribution. Scheduled for removal in WP-07.
    """
    warnings.warn(
        "generate_attribution_prisma_enhanced is deprecated, use compute_attribution instead",
        DeprecationWarning,
        stacklevel=2,
    )
    debug = getattr(config.classify.boosting, 'debug_mode', False)
    return compute_attribution(
        model_prisma=model,
        input_tensor=input_tensor,
        config=config,
        idx_to_class=idx_to_class,
        device=device,
        steering_resources=steering_resources,
        enable_feature_gradients=enable_feature_gradients,
        feature_gradient_layers=feature_gradient_layers,
        clip_classifier=clip_classifier,
        debug=debug,
    )
