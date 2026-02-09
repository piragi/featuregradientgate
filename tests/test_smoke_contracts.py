import inspect
import json

import pytest
import torch


def _param_names(func):
    return list(inspect.signature(func).parameters.keys())


def test_package_api_imports():
    """All public APIs are importable from their canonical package paths."""
    import gradcamfaith.data.setup  # noqa: F401

    from gradcamfaith.experiments.pipeline import run_unified_pipeline
    from gradcamfaith.models.load import load_model_for_dataset
    from gradcamfaith.models.sae_resources import load_steering_resources
    from gradcamfaith.data.setup import convert_dataset

    from gradcamfaith.experiments.case_studies import run_case_study_analysis
    from gradcamfaith.experiments.comparison import main as comparison_main
    from gradcamfaith.experiments.sae_train import SWEEP_CONFIG, train_single_config
    from gradcamfaith.experiments.sweep import run_parameter_sweep, run_single_experiment, SweepConfig

    assert callable(run_single_experiment)
    assert callable(run_parameter_sweep)
    assert callable(train_single_config)
    assert callable(comparison_main)
    assert callable(run_case_study_analysis)
    assert callable(convert_dataset)
    assert callable(run_unified_pipeline)
    assert callable(load_model_for_dataset)
    assert callable(load_steering_resources)
    assert isinstance(SWEEP_CONFIG, dict)
    assert callable(SweepConfig)


def test_public_signature_contracts():
    from gradcamfaith.experiments.case_studies import run_case_study_analysis
    from gradcamfaith.experiments.comparison import main as comparison_main
    from gradcamfaith.experiments.sae_train import train_single_config
    from gradcamfaith.experiments.sweep import run_parameter_sweep, run_single_experiment
    from gradcamfaith.experiments.pipeline import run_unified_pipeline
    from gradcamfaith.models.load import load_model_for_dataset
    from gradcamfaith.models.sae_resources import load_steering_resources
    from gradcamfaith.data.setup import convert_dataset

    assert _param_names(run_single_experiment) == [
        "dataset_name",
        "source_path",
        "experiment_params",
        "output_dir",
        "model",
        "steering_resources",
        "current_mode",
        "debug_mode",
        "clip_classifier",
        "subset_size",
        "random_seed",
    ]
    assert inspect.signature(run_single_experiment).parameters["random_seed"].default == 42

    assert _param_names(run_parameter_sweep) == [
        "datasets",
        "layer_combinations",
        "kappa_values",
        "gate_constructions",
        "shuffle_decoder_options",
        "clamp_max_values",
        "current_mode",
        "debug_mode",
        "output_base_dir",
        "subset_size",
        "random_seed",
    ]
    assert inspect.signature(run_parameter_sweep).parameters["random_seed"].default == 42

    assert _param_names(train_single_config) == ["dataset_name", "layer_idx", "expansion_factor", "k", "lr"]
    assert _param_names(comparison_main) == ["sweep_dirs"]

    assert _param_names(run_case_study_analysis) == [
        "experiment_path",
        "experiment_config",
        "layers",
        "n_top_images",
        "n_patches_per_image",
        "n_case_visualizations",
        "n_prototypes",
        "validation_activations_path",
        "mode",
        "max_prototype_images",
    ]

    convert_sig = inspect.signature(convert_dataset)
    assert _param_names(convert_dataset)[:3] == ["dataset_name", "source_path", "output_path"]
    assert any(param.kind == inspect.Parameter.VAR_KEYWORD for param in convert_sig.parameters.values())

    assert _param_names(run_unified_pipeline) == [
        "config",
        "dataset_name",
        "source_data_path",
        "model",
        "steering_resources",
        "clip_classifier",
        "prepared_data_path",
        "device",
        "force_prepare",
        "subset_size",
        "random_seed",
    ]

    assert _param_names(load_model_for_dataset) == ["dataset_config", "device", "config"]
    assert _param_names(load_steering_resources) == ["layers", "dataset_name"]


def test_example_dry_run_manifest_contract(tmp_path):
    from gradcamfaith.examples.minimal_run import ExampleConfig, run_example

    output_dir = tmp_path / "example"
    config = ExampleConfig(output_dir=output_dir, dry_run=True, random_seed=123)

    result = run_example(config)
    manifest_path = output_dir / "run_manifest.json"

    assert result["status"] == "dry-run"
    assert manifest_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert set(manifest) >= {"resolved_config", "seed", "timestamp", "git_sha", "environment_lock"}
    assert manifest["seed"] == 123
    assert manifest["environment_lock"] == "uv.lock"
    assert manifest["resolved_config"]["random_seed"] == 123
    assert manifest["resolved_config"]["dry_run"] is True


def test_feature_gate_is_deterministic_for_fixed_inputs():
    from gradcamfaith.core.gating import compute_feature_gradient_gate

    torch.manual_seed(0)
    residual_grad = torch.randn(12, 8)
    residual = torch.randn(12, 8)
    sae_codes = torch.randn(12, 16)
    sae_decoder = torch.randn(8, 16)

    gate_1, debug_1 = compute_feature_gradient_gate(
        residual_grad=residual_grad,
        residual=residual,
        sae_codes=sae_codes,
        sae_decoder=sae_decoder,
        clamp_max=10.0,
        gate_construction="combined",
        shuffle_decoder=False,
    )
    gate_2, debug_2 = compute_feature_gradient_gate(
        residual_grad=residual_grad,
        residual=residual,
        sae_codes=sae_codes,
        sae_decoder=sae_decoder,
        clamp_max=10.0,
        gate_construction="combined",
        shuffle_decoder=False,
    )

    assert torch.allclose(gate_1, gate_2)
    assert debug_1 == debug_2 == {}  # debug=False returns empty dict


def test_feature_gate_shuffle_is_seeded_and_bounded():
    from gradcamfaith.core.gating import compute_feature_gradient_gate

    torch.manual_seed(7)
    residual_grad = torch.randn(20, 10)
    residual = torch.randn(20, 10)
    sae_codes = torch.randn(20, 24)
    sae_decoder = torch.randn(10, 24)

    gate_seed_1a, _ = compute_feature_gradient_gate(
        residual_grad=residual_grad,
        residual=residual,
        sae_codes=sae_codes,
        sae_decoder=sae_decoder,
        clamp_max=10.0,
        gate_construction="combined",
        shuffle_decoder=True,
        shuffle_decoder_seed=123,
    )
    gate_seed_1b, _ = compute_feature_gradient_gate(
        residual_grad=residual_grad,
        residual=residual,
        sae_codes=sae_codes,
        sae_decoder=sae_decoder,
        clamp_max=10.0,
        gate_construction="combined",
        shuffle_decoder=True,
        shuffle_decoder_seed=123,
    )
    gate_seed_2, _ = compute_feature_gradient_gate(
        residual_grad=residual_grad,
        residual=residual,
        sae_codes=sae_codes,
        sae_decoder=sae_decoder,
        clamp_max=10.0,
        gate_construction="combined",
        shuffle_decoder=True,
        shuffle_decoder_seed=456,
    )

    assert torch.allclose(gate_seed_1a, gate_seed_1b)
    assert not torch.allclose(gate_seed_1a, gate_seed_2)

    eps = 1e-6
    assert torch.min(gate_seed_1a).item() >= (0.1 - eps)
    assert torch.max(gate_seed_1a).item() <= (10.0 + eps)
