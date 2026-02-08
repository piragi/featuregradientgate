import json
import os
from pathlib import Path

import pytest
import torch


RUN_FULL_STACK = os.getenv("GRADCAMFAITH_RUN_FULL_STACK") == "1"


def _skip_unless_full_stack():
    if not RUN_FULL_STACK:
        pytest.skip("Set GRADCAMFAITH_RUN_FULL_STACK=1 to run full download/sweep integration checks")


def _normalize_results(base_dir: Path):
    payload = {}
    for path in sorted(base_dir.rglob("results.json")):
        rel = str(path.relative_to(base_dir))
        data = json.loads(path.read_text())
        data.pop("timestamp", None)
        payload[rel] = data
    return payload


@pytest.mark.integration
def test_full_setup_downloads_assets():
    _skip_unless_full_stack()

    from setup import main as setup_main

    setup_main()

    expected_paths = [
        Path("./data/hyperkvasir/labeled-images"),
        Path("./models/hyperkvasir/hyperkvasir_vit_model.pth"),
        Path("./data/covidquex"),
        Path("./models/covidquex/covidquex_model.pth"),
        Path("./data/imagenet/raw/val"),
        Path("./data/imagenet/raw/test"),
        Path("./data/sae_hyperkvasir"),
        Path("./data/sae_covidquex"),
        Path("./data/sae_clip_vanilla_b32"),
    ]

    missing = [str(path) for path in expected_paths if not path.exists()]
    assert not missing, f"Missing expected downloaded assets: {missing}"


@pytest.mark.integration
def test_sample_sweep_reproducible_with_fixed_seed(tmp_path):
    _skip_unless_full_stack()

    if not torch.cuda.is_available():
        pytest.skip("Sample sweep integration currently requires CUDA because SAE loading calls .cuda()")

    from gradcamfaith.experiments.sweep import run_parameter_sweep

    source_path = Path("./data/hyperkvasir/labeled-images/")
    if not source_path.exists():
        pytest.skip(f"Required source dataset path missing: {source_path}")

    common_kwargs = {
        "datasets": [("hyperkvasir", source_path)],
        "layer_combinations": [[6]],
        "kappa_values": [0.5],
        "gate_constructions": ["combined"],
        "shuffle_decoder_options": [False],
        "clamp_max_values": [10.0],
        "subset_size": 2,
        "current_mode": "test",
        "debug_mode": False,
    }

    run_a = tmp_path / "seed_42_a"
    run_b = tmp_path / "seed_42_b"
    run_c = tmp_path / "seed_7"

    run_parameter_sweep(output_base_dir=run_a, random_seed=42, **common_kwargs)
    run_parameter_sweep(output_base_dir=run_b, random_seed=42, **common_kwargs)
    run_parameter_sweep(output_base_dir=run_c, random_seed=7, **common_kwargs)

    results_a = _normalize_results(run_a)
    results_b = _normalize_results(run_b)
    results_c = _normalize_results(run_c)

    assert results_a == results_b
    assert results_a != results_c
