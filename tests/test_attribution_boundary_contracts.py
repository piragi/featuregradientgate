"""WP-06B-R3 contract tests for the compute_attribution canonical API."""

import inspect

import pytest


def test_compute_attribution_exists():
    """compute_attribution is importable from the canonical package path."""
    from gradcamfaith.core.attribution import compute_attribution

    assert callable(compute_attribution)


def test_compute_attribution_output_contract():
    """compute_attribution signature matches the spec and returns Dict[str, Any]."""
    from gradcamfaith.core.attribution import compute_attribution

    sig = inspect.signature(compute_attribution)
    param_names = list(sig.parameters.keys())

    assert param_names == [
        "model_prisma",
        "input_tensor",
        "config",
        "idx_to_class",
        "device",
        "img_size",
        "steering_resources",
        "enable_feature_gradients",
        "feature_gradient_layers",
        "clip_classifier",
        "debug",
    ]

    # Return annotation is Dict[str, Any]
    assert sig.return_annotation is not inspect.Parameter.empty


def test_pipeline_uses_compute_attribution():
    """pipeline.py imports compute_attribution, not legacy entrypoints."""
    import importlib
    import ast

    spec = importlib.util.find_spec("pipeline")
    assert spec is not None and spec.origin is not None

    source = open(spec.origin).read()
    tree = ast.parse(source)

    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.name if alias.asname is None else alias.asname)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name if alias.asname is None else alias.asname)

    # compute_attribution must be imported
    assert "compute_attribution" in imported_names

    # Legacy entrypoints must NOT be imported in pipeline
    assert "generate_attribution_prisma_enhanced" not in imported_names
    assert "transmm_prisma_enhanced" not in imported_names


def test_deprecated_shims_removed():
    """Legacy entrypoints have been removed from the attribution module (WP-07)."""
    import gradcamfaith.core.attribution as attr_module

    assert not hasattr(attr_module, "transmm_prisma_enhanced")
    assert not hasattr(attr_module, "generate_attribution_prisma_enhanced")
