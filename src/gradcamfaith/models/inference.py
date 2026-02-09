"""
Model inference utilities for the attribution pipeline.

Provides forward/backward pass execution and hook wiring for vit_prisma
HookedViT models. Used by core/attribution.py to collect gradients and
activations needed for TransLRP computation.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from vit_prisma.models.base_vit import HookedViT


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
