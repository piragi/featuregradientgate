import json
from pathlib import Path

import gradcamfaith.experiments.sweep as sweep


class _DummyTorchObject:
    def to(self, *_args, **_kwargs):
        return self


def _fake_load_model_for_dataset(_dataset_config, _device, _config):
    return _DummyTorchObject(), None


def _fake_load_steering_resources(layers, dataset_name=None):
    return {layer: {"sae": _DummyTorchObject(), "dataset": dataset_name} for layer in layers}


def _fake_run_unified_pipeline(
    *,
    subset_size=None,
    random_seed=None,
    **_kwargs,
):
    n_items = int(subset_size or 1)
    results = list(range(n_items))
    saco_results = {
        "mean": float(random_seed if random_seed is not None else 0.0),
        "std": 0.0,
    }
    return results, saco_results


def _load_normalized_result_payload(base_dir: Path):
    payload = {}
    for path in sorted(base_dir.rglob("results.json")):
        rel = str(path.relative_to(base_dir))
        data = json.loads(path.read_text())
        data.pop("timestamp", None)
        payload[rel] = data
    return payload


def test_run_parameter_sweep_writes_seed_and_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "load_model_for_dataset", _fake_load_model_for_dataset)
    monkeypatch.setattr(sweep, "load_steering_resources", _fake_load_steering_resources)
    monkeypatch.setattr(sweep, "run_unified_pipeline", _fake_run_unified_pipeline)
    monkeypatch.setattr(sweep.torch.cuda, "is_available", lambda: False)

    out_dir = tmp_path / "sweep"
    results = sweep.run_parameter_sweep(
        datasets=[("hyperkvasir", Path("./data/hyperkvasir/labeled-images/"))],
        layer_combinations=[[6]],
        kappa_values=[0.5],
        gate_constructions=["combined"],
        shuffle_decoder_options=[False],
        clamp_max_values=[10.0],
        output_base_dir=out_dir,
        subset_size=3,
        random_seed=42,
    )

    assert "hyperkvasir" in results
    assert len(results["hyperkvasir"]) == 2

    sweep_config = json.loads((out_dir / "sweep_config.json").read_text())
    assert sweep_config["random_seed"] == 42
    assert sweep_config["subset_size"] == 3

    experiment_results = list(out_dir.rglob("results.json"))
    assert len(experiment_results) == 2

    for result_file in experiment_results:
        data = json.loads(result_file.read_text())
        assert data["status"] == "success"
        assert data["random_seed"] == 42
        assert data["n_images"] == 3


def test_sample_sweep_is_reproducible_given_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, "load_model_for_dataset", _fake_load_model_for_dataset)
    monkeypatch.setattr(sweep, "load_steering_resources", _fake_load_steering_resources)
    monkeypatch.setattr(sweep, "run_unified_pipeline", _fake_run_unified_pipeline)
    monkeypatch.setattr(sweep.torch.cuda, "is_available", lambda: False)

    common_kwargs = {
        "datasets": [("hyperkvasir", Path("./data/hyperkvasir/labeled-images/"))],
        "layer_combinations": [[6]],
        "kappa_values": [0.5],
        "gate_constructions": ["combined"],
        "shuffle_decoder_options": [False],
        "clamp_max_values": [10.0],
        "subset_size": 2,
    }

    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    run_c = tmp_path / "run_c"

    sweep.run_parameter_sweep(output_base_dir=run_a, random_seed=42, **common_kwargs)
    sweep.run_parameter_sweep(output_base_dir=run_b, random_seed=42, **common_kwargs)
    sweep.run_parameter_sweep(output_base_dir=run_c, random_seed=7, **common_kwargs)

    payload_a = _load_normalized_result_payload(run_a)
    payload_b = _load_normalized_result_payload(run_b)
    payload_c = _load_normalized_result_payload(run_c)

    assert payload_a == payload_b
    assert payload_a != payload_c
