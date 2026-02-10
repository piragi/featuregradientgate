# Feature Gradient Gating

Research code for attribution faithfulness experiments with sparse autoencoder (SAE) feature-gradient gating.

Project page: https://piragi.github.io/thesis

## Overview

Traditional attribution methods like TransMM combine attention maps with gradients to generate explanations. This project explores whether this principle extends to SAE feature space:

- **Conventional TransMM**: Combines attention patterns with attention gradients in attention space
- **Feature Gradient Gating**: Combines SAE feature activations with SAE feature gradients in interpretable feature space

The key research question: Can we leverage the interpretability of SAE features to create more faithful and semantically meaningful attribution maps?

## Environment Setup

Install/sync dependencies:

```bash
uv sync
```

For ImageNet downloads, authenticate Hugging Face:

```bash
uvx hf auth login
```

## Workflow Runbook

Use this sequence for end-to-end usage.

### 1. Download and Prepare Data/Models

```bash
uv run python -m featuregating.datasets.setup
```

Runtime paths:
- prepared datasets: `data/prepared/<dataset>/`
- model checkpoints: `data/models/<dataset>/`
- run outputs: `data/runs/<run_name>/`

### 2. Run a Sweep

```bash
uv run python -m featuregating.experiments.sweep
```

Expected artifacts:
- new sweep folder in `data/runs/feature_gradient_sweep_<timestamp>/`
- per-experiment outputs including `results.json`, `experiment_config.json`, `faithfulness_stats_*.json`

### 3. Run Cross-Config Comparison

```bash
uv run python -m featuregating.experiments.comparison
```

What it does:
- loads one or more sweep folders
- compares treatment configs against per-dataset vanilla baselines
- writes:
  - `detailed_sweep_comparison.csv`
  - `sweep_summary_table.csv`

By default it tries to auto-discover recent sweep folders in `data/runs/`. If that fails, edit the config block in `featuregating/experiments/comparison.py`.

### 4. Run Case Studies (Qualitative Analysis)

```bash
uv run python -m featuregating.experiments.case_studies
```

What it does:
- loads faithfulness + debug artifacts from a selected sweep config
- extracts dominant boosting/suppressing SAE features
- saves per-layer case outputs under:
  - `.../case_studies/<experiment_config>/layer_<layer_idx>/`
  - `.../case_studies_degraded/<experiment_config>/layer_<layer_idx>/`

By default it auto-discovers a recent run for `imagenet`; edit the config block in `featuregating/experiments/case_studies.py` for a specific dataset/run/config.

## Output Artifact Map

- `results.json`:
  - per-experiment summary metrics (`SaCo`, `FaithfulnessCorrelation`, `PixelFlipping`)
- `faithfulness_stats_*.json`:
  - detailed per-image metric arrays + image-level metadata
- `debug/layer_*_debug.npz`:
  - sparse feature debug arrays used by case study analysis
- comparison outputs:
  - `detailed_sweep_comparison.csv`, `sweep_summary_table.csv`
