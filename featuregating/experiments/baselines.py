"""
Attribution baselines for Vision Transformers.

Implements comparison methods to benchmark against SAE-gated TransMM:

    compute_rollout_attribution  — Attention Rollout (Abnar & Zuidema, 2020)
    compute_tokentm_attribution  — TokenTM (Wu et al., CVPR 2024)
    compute_gradcam_vit          — GradCAM on the residual stream

All functions share the same signature as compute_attribution in core/attribution.py
and return the same dict format: {predictions, attribution_positive, raw_attribution}.

Hook points used (all confirmed available in vit_prisma's transformer_block.py):
    blocks.{i}.attn.hook_pattern   — attention weights  (fwd + bwd for TokenTM/TransMM)
    blocks.{i}.hook_resid_pre      — residual before attention sublayer (fwd only)
    blocks.{i}.hook_resid_mid      — residual between attention and FFN (fwd only)
    blocks.{i}.hook_mlp_out        — FFN output before residual add (fwd only)
    blocks.{i}.hook_resid_post     — full residual after FFN (fwd + bwd for GradCAM)

Weight matrices accessed directly (confirmed in vit_prisma attention.py):
    blocks.{i}.attn.W_V  — shape [n_heads, d_model, d_head]
    blocks.{i}.attn.W_O  — shape [n_heads, d_head, d_model]
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from vit_prisma.models.base_vit import HookedViT

from featuregating.core.config import PipelineConfig
from featuregating.models.inference import run_model_forward_backward


# ---------------------------------------------------------------------------
# Hook setup helpers
# ---------------------------------------------------------------------------

def _setup_rollout_hooks(
    model: HookedViT,
) -> Tuple[List, Dict]:
    """Forward-only hooks for attention patterns (no gradients needed)."""
    activations: Dict[str, torch.Tensor] = {}

    def save_fwd(tensor: torch.Tensor, hook: Any):
        activations[hook.name] = tensor.detach()

    n_layers = model.cfg.n_layers
    hook_names = [f"blocks.{i}.attn.hook_pattern" for i in range(n_layers)]
    fwd_hooks = [(name, save_fwd) for name in hook_names]

    return fwd_hooks, activations


def _setup_tokentm_hooks(
    model: HookedViT,
) -> Tuple[List, List, Dict, Dict]:
    """Forward + backward hooks for TokenTM.

    Forward: attention patterns, resid_pre (MHSA input), resid_mid (FFN input),
             mlp_out (FFN output before residual add).
    Backward: attention patterns only (needed for gradient weighting in MHSA update).
    The FFN update uses only forward activations (no gradients).
    """
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def save_fwd(tensor: torch.Tensor, hook: Any):
        activations[hook.name] = tensor.detach()

    def save_bwd(grad: torch.Tensor, hook: Any):
        if grad is not None:
            gradients[hook.name + "_grad"] = grad.detach()

    n_layers = model.cfg.n_layers
    attn_names   = [f"blocks.{i}.attn.hook_pattern" for i in range(n_layers)]
    resid_pre    = [f"blocks.{i}.hook_resid_pre"    for i in range(n_layers)]
    resid_mid    = [f"blocks.{i}.hook_resid_mid"    for i in range(n_layers)]
    mlp_out      = [f"blocks.{i}.hook_mlp_out"      for i in range(n_layers)]

    fwd_hooks = [(n, save_fwd) for n in attn_names + resid_pre + resid_mid + mlp_out]
    bwd_hooks = [(n, save_bwd) for n in attn_names]   # gradients only for attention

    return fwd_hooks, bwd_hooks, activations, gradients


def _setup_gradcam_hooks(
    model: HookedViT,
) -> Tuple[List, List, Dict, Dict]:
    """Forward + backward hooks for the second-to-last layer's residual stream.

    We use ``n_layers - 2`` because at the *last* layer's ``hook_resid_post``
    only the CLS token (position 0) receives gradient — everything downstream
    (layer-norm → CLS extraction → classifier) is position-wise.  Patch
    positions get zero gradient, producing a degenerate map.  At the
    second-to-last layer the gradient has already fanned out through the last
    layer's attention, giving non-zero gradients for all spatial positions.
    """
    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}

    def save_fwd(tensor: torch.Tensor, hook: Any):
        activations[hook.name] = tensor.detach()

    def save_bwd(grad: torch.Tensor, hook: Any):
        if grad is not None:
            gradients[hook.name + "_grad"] = grad.detach()

    target_layer = model.cfg.n_layers - 2
    hook_name = f"blocks.{target_layer}.hook_resid_post"
    fwd_hooks = [(hook_name, save_fwd)]
    bwd_hooks = [(hook_name, save_bwd)]

    return fwd_hooks, bwd_hooks, activations, gradients


# ---------------------------------------------------------------------------
# Shared post-processing (identical to core/attribution.py)
# ---------------------------------------------------------------------------

def _postprocess(
    patch_scores: torch.Tensor,
    img_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reshape 1-D patch scores to 2-D spatial map and interpolate to img_size."""
    raw = patch_scores.detach().cpu().numpy()
    side = int(np.sqrt(patch_scores.numel()))
    upsampled = F.interpolate(
        patch_scores.reshape(1, 1, side, side),
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()

    val_range = upsampled.max() - upsampled.min()
    if val_range > 1e-8:
        upsampled = (upsampled - upsampled.min()) / (val_range + 1e-8)

    return upsampled, raw


# ---------------------------------------------------------------------------
# Attention Rollout — Abnar & Zuidema (2020)
# ---------------------------------------------------------------------------

def compute_rollout_attribution(
    model_prisma: HookedViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],
    device: Optional[torch.device] = None,
    img_size: int = 224,
    clip_classifier: Optional[Any] = None,
) -> Dict[str, Any]:
    """Attention Rollout (Abnar & Zuidema, 2020).

    Class-agnostic: no gradients used.  For each layer, the attention matrix
    is averaged over heads and mixed with the identity via a 0.5/0.5 residual
    correction.  Matrices are accumulated multiplicatively; the CLS-to-patch
    row of the final product gives the attribution map.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    fwd_hooks, activations = _setup_rollout_hooks(model_prisma)

    with model_prisma.hooks(fwd_hooks=fwd_hooks, reset_hooks_end=True):
        with torch.no_grad():
            if clip_classifier is not None:
                result = clip_classifier.forward(input_tensor, requires_grad=False)
                logits       = result["logits"]
                probabilities = result["probabilities"]
                predicted_idx = result["predicted_class_idx"]
            else:
                logits = model_prisma(input_tensor)
                probabilities = F.softmax(logits, dim=-1)
                predicted_idx = int(torch.argmax(probabilities, dim=-1).item())

    n_layers = model_prisma.cfg.n_layers
    attn_names = [f"blocks.{i}.attn.hook_pattern" for i in range(n_layers)]

    # Initialise rollout matrix as identity
    first_attn = activations[attn_names[0]]  # [1, heads, N+1, N+1]
    num_tokens = first_attn.shape[-1]
    rollout = torch.eye(num_tokens, device=device)

    for name in attn_names:
        attn = activations[name]                     # [1, heads, N+1, N+1]
        A = attn[0].mean(dim=0)                      # [N+1, N+1]
        # Residual correction: account for skip connections
        A_bar = 0.5 * A + 0.5 * torch.eye(num_tokens, device=device)
        # Row-normalise so each row still sums to 1
        A_bar = A_bar / (A_bar.sum(dim=-1, keepdim=True) + 1e-8)
        rollout = A_bar @ rollout

    # CLS row → patch scores (exclude CLS token at position 0)
    patch_scores = rollout[0, 1:].clone()

    attribution_map, raw = _postprocess(patch_scores, img_size)

    prediction_result = {
        "logits": logits.detach().cpu().numpy(),
        "probabilities": (
            probabilities.squeeze().cpu().detach().numpy().tolist()
            if isinstance(probabilities, torch.Tensor) else probabilities
        ),
        "predicted_class_idx": predicted_idx,
        "predicted_class_label": idx_to_class.get(predicted_idx, f"class_{predicted_idx}"),
    }

    return {
        "predictions": prediction_result,
        "attribution_positive": attribution_map,
        "raw_attribution": raw,
        "debug_info": {},
    }


# ---------------------------------------------------------------------------
# TokenTM — Wu et al. (CVPR 2024)
# ---------------------------------------------------------------------------

def _transformation_weights(
    E: torch.Tensor,
    E_tilde: torch.Tensor,
) -> torch.Tensor:
    """TokenTM transformation weight vector for one sublayer (Eq. 12–13).

    For each token i:
        w_i = (‖Ẽ_i‖ / ‖E_i‖) · NECC(i)
        NECC(i) = softmax_i( cos(E_i, Ẽ_i) )

    Args:
        E:       [N+1, d] — sublayer input tokens
        E_tilde: [N+1, d] — position-wise transformed tokens

    Returns:
        w: [N+1]
    """
    norm_ratio = (
        E_tilde.norm(dim=-1) / E.norm(dim=-1).clamp(min=1e-8)
    )                                                        # [N+1]
    cos_sim = F.cosine_similarity(E, E_tilde, dim=-1)       # [N+1]
    necc    = F.softmax(cos_sim, dim=0)                     # [N+1]
    return norm_ratio * necc                                 # [N+1]


def _mhsa_update(
    attn: torch.Tensor,
    attn_grad: Optional[torch.Tensor],
    E: torch.Tensor,
    W_V: torch.Tensor,
    W_O: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """TokenTM MHSA update matrix U_MHSA (Eq. 14–15).

    Per head h:
        Ẽ_h  = E @ W_V[h] @ W_O[h]          (OV-circuit, position-wise)
        w_h   = transformation_weights(E, Ẽ_h)
        T_h   = A_h @ diag(w_h)
        term_h = relu(grad_A_h) ⊙ T_h

    U_MHSA = I + mean_h(term_h)

    Args:
        attn:      [1, n_heads, N+1, N+1]
        attn_grad: [1, n_heads, N+1, N+1] or None
        E:         [N+1, d_model] — MHSA input (resid_pre)
        W_V:       [n_heads, d_model, d_head]
        W_O:       [n_heads, d_head, d_model]
    """
    n_heads = attn.shape[1]
    N1      = attn.shape[-1]

    # OV-circuit per head, position-wise: E_tilde[n, h, d]
    E_val   = torch.einsum("nd,hde->nhe", E, W_V)     # [N+1, n_heads, d_head]
    E_tilde = torch.einsum("nhe,hed->nhd", E_val, W_O) # [N+1, n_heads, d_model]

    accumulated = torch.zeros(N1, N1, device=device)
    for h in range(n_heads):
        w_h = _transformation_weights(E, E_tilde[:, h, :])  # [N+1]
        W_h = torch.diag(w_h)                                # [N+1, N+1]
        A_h = attn[0, h]                                     # [N+1, N+1]
        T_h = A_h @ W_h                                      # [N+1, N+1]
        g_h = attn_grad[0, h].clamp(min=0) if attn_grad is not None else torch.ones_like(A_h)
        accumulated += g_h * T_h

    return torch.eye(N1, device=device) + accumulated / n_heads


def _ffn_update(
    E_ffn: torch.Tensor,
    mlp_out: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """TokenTM FFN update matrix U_FFN (Eq. 16–17).

    U_FFN = I + diag(w_ffn)

    where w_ffn = transformation_weights(resid_mid, mlp_out).

    Args:
        E_ffn:   [N+1, d_model] — FFN input (resid_mid)
        mlp_out: [N+1, d_model] — FFN output before residual add (hook_mlp_out)
    """
    N1    = E_ffn.shape[0]
    w_ffn = _transformation_weights(E_ffn, mlp_out)     # [N+1]
    return torch.eye(N1, device=device) + torch.diag(w_ffn)


def compute_tokentm_attribution(
    model_prisma: HookedViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],
    device: Optional[torch.device] = None,
    img_size: int = 224,
    clip_classifier: Optional[Any] = None,
) -> Dict[str, Any]:
    """TokenTM (Wu et al., CVPR 2024).

    Extends TransMM by incorporating token transformation weights derived from
    the OV circuit (MHSA) and FFN into a **multiplicative** relevance propagation:

        C^l = U_FFN^l · U_MHSA^l · C^(l-1)

    MHSA update (Eq. 14–15):
        U_MHSA = I + avg_h[ relu(∂y/∂A_h) ⊙ (A_h · diag(w_h)) ]
        w_h[i] = (‖Ẽ^h_i‖ / ‖E_i‖) · NECC_h(i)      ← norm ratio × cosine softmax
        Ẽ^h_i  = E_i @ W_V[h] @ W_O[h]                ← OV circuit, position-wise

    FFN update (Eq. 16–17):
        U_FFN = I + diag(w_ffn)
        w_ffn[i] = (‖mlp_out_i‖ / ‖resid_mid_i‖) · NECC(i)

    Initialization (Eq. 18):
        C^0 = diag(‖E^0_i‖₂)    ← L2 norms of patch embeddings before block 0

    Final attribution (Sec. 4.4):
        C^(nL)[0, 1:]            ← CLS row, patch columns
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    fwd_hooks, bwd_hooks, activations, gradients = _setup_tokentm_hooks(model_prisma)

    with model_prisma.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, reset_hooks_end=True):
        prediction_result, predicted_idx = run_model_forward_backward(
            model_prisma, input_tensor, clip_classifier, device
        )

    prediction_result["predicted_class_label"] = idx_to_class.get(
        predicted_idx, f"class_{predicted_idx}"
    )

    n_layers = model_prisma.cfg.n_layers

    # Initialise C^0 = diag(‖E_i^0‖₂) using patch embeddings before block 0
    E0     = activations["blocks.0.hook_resid_pre"][0]     # [N+1, d_model]
    norms0 = E0.norm(dim=-1)                               # [N+1]
    C      = torch.diag(norms0)                            # [N+1, N+1]

    for i in range(n_layers):
        # Activations collected by hooks
        attn     = activations[f"blocks.{i}.attn.hook_pattern"]  # [1, H, N+1, N+1]
        attn_g   = gradients.get(f"blocks.{i}.attn.hook_pattern_grad")
        E_pre    = activations[f"blocks.{i}.hook_resid_pre"][0]  # [N+1, d_model]
        E_mid    = activations.get(f"blocks.{i}.hook_resid_mid") # [1, N+1, d_model]
        mlp_raw  = activations.get(f"blocks.{i}.hook_mlp_out")   # [1, N+1, d_model]

        # Model weight matrices (direct attribute access, not hooks).
        # getattr used to satisfy the type checker; W_V/W_O are nn.Parameters
        # confirmed in vit_prisma/models/layers/attention.py.
        attn_module = getattr(model_prisma.blocks[i], "attn")
        W_V: torch.Tensor = getattr(attn_module, "W_V")  # [n_heads, d_model, d_head]
        W_O: torch.Tensor = getattr(attn_module, "W_O")  # [n_heads, d_head, d_model]

        # MHSA sublayer update
        U_mhsa = _mhsa_update(attn, attn_g, E_pre, W_V, W_O, device)

        # FFN sublayer update (only if both activations captured)
        if E_mid is not None and mlp_raw is not None:
            E_ffn   = E_mid[0]    # [N+1, d_model]
            mlp_out = mlp_raw[0]  # [N+1, d_model]
            U_ffn   = _ffn_update(E_ffn, mlp_out, device)
        else:
            N1    = C.shape[0]
            U_ffn = torch.eye(N1, device=device)

        # Multiplicative propagation: FFN after MHSA (forward pass order)
        C = U_ffn @ U_mhsa @ C

    # CLS row, patch columns
    patch_scores = C[0, 1:].detach().clone()
    attribution_map, raw = _postprocess(patch_scores, img_size)

    return {
        "predictions": prediction_result,
        "attribution_positive": attribution_map,
        "raw_attribution": raw,
        "debug_info": {},
    }


# ---------------------------------------------------------------------------
# GradCAM-ViT — gradient-weighted residual stream (second-to-last layer)
# ---------------------------------------------------------------------------

def compute_gradcam_vit(
    model_prisma: HookedViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],
    device: Optional[torch.device] = None,
    img_size: int = 224,
    clip_classifier: Optional[Any] = None,
) -> Dict[str, Any]:
    """GradCAM-style attribution for ViTs (Selvaraju et al., 2017).

    Hooks the second-to-last layer's residual stream (see ``_setup_gradcam_hooks``
    for why the last layer produces degenerate maps).  For each dimension k,
    the global-average-pooled gradient over spatial tokens gives a weight w_k.
    The attribution per token t is then:

        cam_t = relu( sum_k w_k * resid_post_{t,k} )
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_tensor = input_tensor.to(device)
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)

    fwd_hooks, bwd_hooks, activations, gradients = _setup_gradcam_hooks(model_prisma)

    with model_prisma.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks, reset_hooks_end=True):
        prediction_result, predicted_idx = run_model_forward_backward(
            model_prisma, input_tensor, clip_classifier, device
        )

    prediction_result["predicted_class_label"] = idx_to_class.get(
        predicted_idx, f"class_{predicted_idx}"
    )

    target_layer = model_prisma.cfg.n_layers - 2
    hook_name = f"blocks.{target_layer}.hook_resid_post"

    resid = activations[hook_name]                           # [1, N+1, d]
    grad  = gradients.get(hook_name + "_grad")               # [1, N+1, d] or None

    if grad is None:
        raise RuntimeError("Backward hook for resid_post did not fire; cannot compute GradCAM.")

    # Exclude CLS token; work on spatial patches only
    resid_patches = resid[0, 1:]   # [N, d]
    grad_patches  = grad[0, 1:]    # [N, d]

    # Global-average-pool the gradient over spatial tokens → channel weights
    weights = grad_patches.mean(dim=0)                       # [d]

    # Weighted sum over channels per token, ReLU-clamped
    cam = (resid_patches * weights.unsqueeze(0)).sum(dim=-1).clamp(min=0)  # [N]

    attribution_map, raw = _postprocess(cam, img_size)

    return {
        "predictions": prediction_result,
        "attribution_positive": attribution_map,
        "raw_attribution": raw,
        "debug_info": {},
    }
