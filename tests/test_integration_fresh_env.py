import json
import os
import shutil
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


@pytest.mark.integration
def test_imagenet_golden_faithfulness_values(tmp_path):
    """Golden-value regression test: imagenet val, subset=500, seed=123, combined, layer 3.

    Verifies that faithfulness metric outputs match known-good reference values.
    Catches silent behavior changes in the attribution/gating pipeline.

    Golden values recorded with: clamp_max=10.0 (default), kappa=0.5, combined gate.

    Tolerances use rel=0.5% to accommodate CUDA non-determinism
    (no torch.use_deterministic_algorithms in pipeline). Medians are
    not checked — they are volatile order statistics under float drift.
    """
    _skip_unless_full_stack()

    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA for SAE loading")

    from gradcamfaith.experiments.sweep import run_parameter_sweep

    source_path = Path("./data/imagenet/raw")
    if not source_path.exists():
        pytest.skip(f"Required source dataset path missing: {source_path}")

    output_dir = tmp_path / "golden_sweep"

    try:
        results = run_parameter_sweep(
            datasets=[("imagenet", source_path)],
            layer_combinations=[[3]],
            kappa_values=[0.5],
            gate_constructions=["combined"],
            shuffle_decoder_options=[False],
            clamp_max_values=[10.0],
            current_mode="val",
            debug_mode=False,
            output_base_dir=output_dir,
            subset_size=500,
            random_seed=123,
        )

        # Sweep completed with both vanilla + gated experiments
        assert "imagenet" in results
        imagenet_results = results["imagenet"]
        assert len(imagenet_results) == 2
        assert all(r["status"] == "success" for r in imagenet_results)

        # --- Gated experiment golden values ---
        gated_dir = output_dir / "imagenet" / "layers_3_kappa_0.5_combined_clamp_10.0"
        assert gated_dir.exists(), f"Expected experiment directory not found: {gated_dir}"

        # SaCo results (from results.json)
        results_json = json.loads((gated_dir / "results.json").read_text())
        saco = results_json["saco_results"]

        RTOL = 5e-3  # 0.5% — covers CUDA float non-determinism

        assert saco["mean"] == pytest.approx(0.26331857538045383, rel=RTOL)
        assert saco["std"] == pytest.approx(0.4043539904807494, rel=RTOL)

        # Faithfulness metrics (from faithfulness_stats*.json under val/)
        faith_files = list((gated_dir / "val").glob("faithfulness_stats_*.json"))
        assert len(faith_files) >= 1, "No faithfulness stats file found"

        faith_stats = json.loads(faith_files[0].read_text())
        metrics = faith_stats["metrics"]

        # PixelFlipping
        pf = metrics["PixelFlipping"]["overall"]
        assert pf["count"] == 500
        assert pf["mean"] == pytest.approx(8.393853450536728, rel=RTOL)

        # FaithfulnessCorrelation
        fc = metrics["FaithfulnessCorrelation"]["overall"]
        assert fc["count"] == 500
        assert fc["mean"] == pytest.approx(0.311978947368421, rel=RTOL)

    finally:
        # Explicit cleanup of experiment outputs
        if output_dir.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
