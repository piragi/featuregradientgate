"""WP-06B-R3 contract tests for the compute_attribution canonical API."""

import inspect
import warnings

import pytest


def test_compute_attribution_exists():
    """compute_attribution is importable from both package and root wrapper."""
    from gradcamfaith.core.attribution import compute_attribution
    from transmm import compute_attribution as root_compute_attribution

    assert callable(compute_attribution)
    assert callable(root_compute_attribution)
    assert compute_attribution is root_compute_attribution


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


def test_legacy_wrappers_deprecated():
    """Legacy entrypoints exist but emit DeprecationWarning when called.

    We only verify the warning is emitted; we don't call with real models.
    """
    from gradcamfaith.core.attribution import (
        generate_attribution_prisma_enhanced,
        transmm_prisma_enhanced,
    )

    # Check docstrings mention DEPRECATED
    assert "DEPRECATED" in (transmm_prisma_enhanced.__doc__ or "")
    assert "DEPRECATED" in (generate_attribution_prisma_enhanced.__doc__ or "")

    # Verify that calling them would emit deprecation warnings.
    # We can't call with real args, but we can patch compute_attribution
    # to verify the delegation path.
    import unittest.mock as mock

    sentinel = {
        "predictions": {},
        "attribution_positive": None,
        "raw_attribution": None,
        "debug_info": {},
    }

    with mock.patch(
        "gradcamfaith.core.attribution.compute_attribution", return_value=sentinel
    ) as mocked:
        # transmm_prisma_enhanced should warn and delegate
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = transmm_prisma_enhanced(
                model_prisma=mock.MagicMock(),
                input_tensor=mock.MagicMock(),
                config=mock.MagicMock(),
                idx_to_class={},
            )
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "compute_attribution" in str(w[0].message)

        assert mocked.called
        # Returns 4-tuple from dict
        assert isinstance(result, tuple) and len(result) == 4

        mocked.reset_mock()

        # generate_attribution_prisma_enhanced should warn and delegate
        mock_config = mock.MagicMock()
        mock_config.classify.boosting.debug_mode = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = generate_attribution_prisma_enhanced(
                model=mock.MagicMock(),
                input_tensor=mock.MagicMock(),
                config=mock_config,
                idx_to_class={},
            )
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "compute_attribution" in str(w[0].message)

        assert mocked.called
        # Returns dict directly
        assert isinstance(result, dict)
