"""Compatibility wrapper — canonical source is gradcamfaith.core.attribution"""
from gradcamfaith.core.attribution import *  # noqa: F401,F403
from gradcamfaith.core.attribution import (  # noqa: F401
    apply_gradient_gating_to_cam,
    compute_layer_attribution,
    avg_heads,
    apply_self_attention_rules,
    run_model_forward_backward,
    setup_hooks,
    transmm_prisma_enhanced,
    generate_attribution_prisma_enhanced,
)
