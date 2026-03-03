import gc
import itertools
import json
import logging
from pathlib import Path

import torch
import torchvision
from vit_prisma.sae import VisionSAETrainer
from vit_prisma.sae.config import VisionModelSAERunnerConfig

import wandb
from featuregating.datasets.dataset_config import get_dataset_config
from featuregating.models.load import load_model_for_dataset

# Suppress PIL debug logging
logging.getLogger('PIL').setLevel(logging.WARNING)

# ============ SWEEP CONFIG ============
SWEEP_CONFIG = {
    'dataset': 'hyperkvasir',  # run across hyperkvasir layers
    'layers': [2, 3, 4, 5, 6, 7, 8, 9, 10],  # full comparable layer range

    # Locked hyperparameters for the all-layer run
    'expansion_factors': [64],  # Stay with exp64 to avoid OOM on this GPU
    'l1_coefficients': [2e-6],  # Better sparsity than 1e-6 with near-identical EV
    'learning_rates': [9e-4],  # Highest EV observed on layer 3

    # Fixed parameters
    'epoch_values': [18],  # Highest EV observed on layer 3
    'batch_size': 4096,
    'num_workers': 10,  # Reduced from 16 to avoid warning
    'n_batches_in_buffer': 20,
    'lr_warm_up_steps': 200,
    'n_validation_runs': 2,  # Keep some validation signal with lower overhead
    'normalize_activations': "layer_norm",  # More stable on smaller/noisier datasets
    'wandb_log_frequency': 100,
    'wandb_project': 'vit_sae_sweep',
    'log_to_wandb': True,
}


def _gpu_cleanup():
    """Lightweight cleanup between runs while keeping shared model in memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_to_cpu(obj):
    """Move a module/object to CPU if it supports .to()."""
    if obj is None:
        return

    to_fn = getattr(obj, "to", None)
    if callable(to_fn):
        try:
            to_fn("cpu")
        except Exception:
            pass


def _clear_activation_store(store):
    """Release heavy references held by vit_prisma activation stores."""
    if store is None:
        return

    # These objects commonly live on CUDA.
    _move_to_cpu(getattr(store, "model", None))

    for tensor_attr in ("storage_buffer", "storage_buffer_out"):
        if not hasattr(store, tensor_attr):
            continue
        tensor = getattr(store, tensor_attr)
        if torch.is_tensor(tensor):
            try:
                setattr(store, tensor_attr, tensor.cpu())
            except Exception:
                pass

    # Drop iterator/dataloader references so worker/process state can be GC'd.
    for attr in (
        "dataloader",
        "image_dataloader",
        "image_dataloader_iter",
        "image_dataloader_eval",
        "image_dataloader_eval_iter",
        "dataset",
        "eval_dataset",
        "model",
    ):
        if hasattr(store, attr):
            try:
                setattr(store, attr, None)
            except Exception:
                pass

    # Best-effort close hooks for future compatibility.
    for method_name in ("close", "cleanup", "shutdown", "clear"):
        method = getattr(store, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass


def _clear_model_hooks(model):
    """Clear forward/backward hooks if the model exposes helper methods."""
    if model is None:
        return

    for method_name in ("reset_hooks", "remove_all_hook_fns", "remove_all_hooks"):
        method = getattr(model, method_name, None)
        if callable(method):
            try:
                method()
            except Exception:
                pass
            break


def _clear_trainer_refs(trainer):
    """Drop common heavy trainer references to speed up GC."""
    if trainer is None:
        return

    heavy_attrs = (
        "sae",
        "sparse_coder",
        "model",
        "optimizer",
        "scheduler",
        "dataset",
        "eval_dataset",
        "train_dataset",
        "val_dataset",
        "train_loader",
        "val_loader",
        "train_dataloader",
        "val_dataloader",
        "activation_store",
        "activations_store",
    )

    for attr in heavy_attrs:
        if not hasattr(trainer, attr):
            continue
        value = getattr(trainer, attr)
        if attr in ("activation_store", "activations_store"):
            _clear_activation_store(value)
        _move_to_cpu(value)
        try:
            setattr(trainer, attr, None)
        except Exception:
            pass


def _load_shared_resources(dataset_name):
    """Load model and datasets once for an entire sweep."""
    dataset_config = get_dataset_config(dataset_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hooked_model, _ = load_model_for_dataset(dataset_config, device)

    data_path = Path(f"data/prepared/{dataset_name}")
    train_path = data_path / "train"
    val_path = data_path / "val"

    transform = dataset_config.get_transforms('train')
    val_transform = dataset_config.get_transforms('test')

    train_dataset = torchvision.datasets.ImageFolder(train_path, transform)
    val_dataset = torchvision.datasets.ImageFolder(val_path, val_transform)

    return hooked_model, train_dataset, val_dataset


def train_single_config(
    dataset_name,
    layer_idx,
    expansion_factor,
    l1_coeff,
    lr,
    num_epochs,
    hooked_model,
    train_dataset,
    val_dataset,
):
    """Train a single SAE configuration."""

    print(f"\n{'='*60}")
    print(f"Training: expansion={expansion_factor}, l1={l1_coeff}, lr={lr}, epochs={num_epochs}")
    print(f"{'='*60}")

    trainer = None
    trained_sae = None
    save_dir = None

    try:
        # SAE config
        run_config = VisionModelSAERunnerConfig(
            model_name="vit_base_patch16_224",
            layer_subtype='hook_resid_post',
            cls_token_only=False,
            context_size=197,
            expansion_factor=expansion_factor,
            activation_fn_str="relu",
            activation_fn_kwargs={},
            l1_coefficient=l1_coeff,
            lr=lr,
            train_batch_size=SWEEP_CONFIG['batch_size'],
            num_epochs=num_epochs,
            num_workers=SWEEP_CONFIG['num_workers'],
            lr_scheduler_name="cosineannealingwarmup",
            lr_warm_up_steps=SWEEP_CONFIG['lr_warm_up_steps'],
            n_validation_runs=SWEEP_CONFIG['n_validation_runs'],
            n_batches_in_buffer=SWEEP_CONFIG['n_batches_in_buffer'],
            initialization_method="encoder_transpose_decoder",
            normalize_activations=SWEEP_CONFIG['normalize_activations'],
            dataset_name=dataset_name,
            log_to_wandb=SWEEP_CONFIG['log_to_wandb'],
            wandb_project=SWEEP_CONFIG['wandb_project'],
            wandb_log_frequency=SWEEP_CONFIG['wandb_log_frequency'],
            n_checkpoints=1,
            use_ghost_grads=True,
            dead_feature_threshold=1e-8,
            feature_sampling_window=1000,
            dead_feature_window=5000,
        )

        run_config.hook_point_layer = layer_idx
        run_config.dataset_size = len(train_dataset)

        # Set up save directory
        save_dir = Path(
            f"data/sae_sweep/{dataset_name}/layer_{layer_idx}/exp{expansion_factor}_l1{l1_coeff}_lr{lr}_ep{num_epochs}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)
        run_config.checkpoint_path = str(save_dir)

        # Add sweep info to wandb config
        if SWEEP_CONFIG['log_to_wandb']:
            wandb.init(
                project=SWEEP_CONFIG['wandb_project'],
                config={
                    'expansion_factor': expansion_factor,
                    'l1_coefficient': l1_coeff,
                    'lr': lr,
                    'epochs': num_epochs,
                    'layer': layer_idx,
                    'dataset': dataset_name,
                },
                name=f"exp{expansion_factor}_l1-{l1_coeff}_lr{lr}_ep{num_epochs}_l{layer_idx}",
                reinit=True
            )

        trainer = VisionSAETrainer(run_config, hooked_model, train_dataset, val_dataset)

        trained_sae = trainer.run()

        run_summary = wandb.run.summary if wandb.run is not None else {}

        # Extract metrics
        metrics = {
            'expansion_factor': expansion_factor,
            'l1_coefficient': l1_coeff,
            'lr': lr,
            'epochs': num_epochs,
            'layer': layer_idx,
            'explained_variance': run_summary.get('metrics/explained_variance', 0),
            'mse_loss': run_summary.get('losses/mse_loss', 0),
            'dead_features': run_summary.get('sparsity/dead_features', 0),
            'l0': run_summary.get('metrics/l0', 0),
            'val_explained_variance': run_summary.get('validation_metrics/explained_variance'),
            'val_mse_loss': run_summary.get('validation_metrics/mse_loss'),
            'val_l0': run_summary.get('validation_metrics/L0'),
        }

        # Save metrics
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"SUCCESS: Explained variance = {metrics['explained_variance']:.4f}")

        return metrics

    except Exception as e:
        print(f"ERROR: Training failed - {e}")
        return None

    finally:
        wandb.finish()

        # Per-run cleanup while keeping shared model and datasets alive.
        _clear_trainer_refs(trainer)
        _clear_model_hooks(hooked_model)
        _move_to_cpu(trained_sae)
        del trainer
        del trained_sae

        _gpu_cleanup()


def _print_best_summary(results_by_layer):
    """Print best config per layer and overall from results_by_layer."""
    if not results_by_layer:
        return

    print(f"\n{'='*60}")
    print("BEST CONFIGURATIONS PER LAYER:")
    print(f"{'='*60}")

    overall_best = None
    overall_best_variance = -float('inf')

    for layer_idx in sorted(results_by_layer.keys()):
        layer_results = results_by_layer[layer_idx]
        if not layer_results:
            continue

        best = max(layer_results, key=lambda x: x['explained_variance'])
        print(f"\nLayer {layer_idx}:")
        print(f"  Expansion Factor: {best['expansion_factor']}")
        print(f"  L1 Coefficient: {best['l1_coefficient']}")
        print(f"  Learning Rate: {best['lr']}")
        print(f"  Epochs: {best['epochs']}")
        print(f"  Explained Variance: {best['explained_variance']:.4f}")

        if best['explained_variance'] > overall_best_variance:
            overall_best = best
            overall_best_variance = best['explained_variance']

    if overall_best:
        print(f"\n{'='*60}")
        print("OVERALL BEST CONFIGURATION:")
        print(f"Layer: {overall_best['layer']}")
        print(f"Expansion Factor: {overall_best['expansion_factor']}")
        print(f"L1 Coefficient: {overall_best['l1_coefficient']}")
        print(f"Learning Rate: {overall_best['lr']}")
        print(f"Epochs: {overall_best['epochs']}")
        print(f"Explained Variance: {overall_best['explained_variance']:.4f}")
        print(f"{'='*60}")


def main():
    """Run a single sweep across selected layers."""

    dataset_name = SWEEP_CONFIG['dataset']
    layers = SWEEP_CONFIG['layers']

    hyperparams = list(
        itertools.product(
            SWEEP_CONFIG['expansion_factors'],
            SWEEP_CONFIG['l1_coefficients'],
            SWEEP_CONFIG['learning_rates'],
            SWEEP_CONFIG['epoch_values'],
        )
    )
    all_configs = list(itertools.product(layers, hyperparams))

    print(f"Running sweep with {len(all_configs)} total configurations")
    print(f"Dataset: {dataset_name}")
    print(f"Layers: {layers}")
    print(f"Hyperparameter combinations per layer: {len(hyperparams)}")

    all_results = {}
    hooked_model = None
    train_dataset = None
    val_dataset = None

    try:
        print("Loading shared model and datasets once for the full sweep...")
        hooked_model, train_dataset, val_dataset = _load_shared_resources(dataset_name)

        for i, (layer_idx, (exp_factor, l1_coeff, lr, num_epochs)) in enumerate(all_configs, 1):
            print(
                f"\n[{i}/{len(all_configs)}] Running Layer {layer_idx} "
                f"with exp_factor={exp_factor}, l1={l1_coeff}, lr={lr}, epochs={num_epochs}"
            )

            metrics = train_single_config(
                dataset_name=dataset_name,
                layer_idx=layer_idx,
                expansion_factor=exp_factor,
                l1_coeff=l1_coeff,
                lr=lr,
                num_epochs=num_epochs,
                hooked_model=hooked_model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
            )

            if not metrics:
                continue

            all_results.setdefault(layer_idx, []).append(metrics)
            with open(f'sweep_results_{dataset_name}_layer{layer_idx}.json', 'w') as f:
                json.dump(all_results[layer_idx], f, indent=2)
            with open(f'sweep_results_{dataset_name}_all_layers.json', 'w') as f:
                json.dump(all_results, f, indent=2)
    finally:
        # End-of-sweep cleanup.
        _clear_model_hooks(hooked_model)
        _move_to_cpu(hooked_model)
        del train_dataset
        del val_dataset
        del hooked_model
        _gpu_cleanup()

    _print_best_summary(all_results)
    return all_results


if __name__ == "__main__":
    results = main()
