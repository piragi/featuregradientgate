# Feature Gradient Gating

Research code for attribution faithfulness experiments with sparse autoencoder (SAE) feature-gradient gating.

Project page: https://piragi.github.io/thesis

## Overview

Traditional attribution methods like TransMM combine attention maps with gradients to generate explanations. This project explores whether this principle extends to SAE feature space:

- **Conventional TransMM**: Combines attention patterns with attention gradients in attention space
- **Feature Gradient Gating**: Combines SAE feature activations with SAE feature gradients in interpretable feature space

The key research question: Can we leverage the interpretability of SAE features to create more faithful and semantically meaningful attribution maps?

## Core Method

The heart of this repository is the feature-gradient gating mechanism in [featuregating/core/gating.py](featuregating/core/gating.py).

If you want to understand the method from the paper, start with:

- `compute_feature_gradient_gate(...)`: constructs the per-patch gate from SAE activations and SAE feature gradients
- `apply_feature_gradient_gating(...)`: applies that gate to the attention CAM used by the attribution pipeline

At a high level, the method implemented there is:

1. project residual gradients into SAE feature space
2. combine feature gradients with SAE activations
3. sum feature contributions into a scalar score per patch
4. map those scores into multiplicative gates and apply them to the CAM

## Code Map

If you want to read the method before running experiments, start here:

- [featuregating/core/gating.py](featuregating/core/gating.py): core feature-gradient gate construction and CAM modulation
- [featuregating/core/attribution.py](featuregating/core/attribution.py): canonical attribution entrypoint via `compute_attribution(...)`
- [featuregating/core/config.py](featuregating/core/config.py): central pipeline and gating configuration schema
- [featuregating/experiments/sweep.py](featuregating/experiments/sweep.py): default experiment driver used in the README workflow

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

Default onboarding run:
- ImageNet `val` split
- random `500`-image subset
- debug artifacts enabled for later case-study analysis

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

By default it auto-discovers the latest sweep folder in `data/runs/`. If that fails, point it to a specific run in [featuregating/experiments/comparison.py](featuregating/experiments/comparison.py).

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

Notes:
- by default it auto-discovers the latest gated ImageNet sweep config
- on first run it may build a one-time validation activation cache in `data/sae_activations/imagenet_val_subset10000/`

If auto-discovery is not what you want, point it to a specific dataset/run/config in [featuregating/experiments/case_studies.py](featuregating/experiments/case_studies.py).

## Output Artifact Map

- `results.json`:
  - per-experiment summary metrics (`SaCo`, `FaithfulnessCorrelation`, `PixelFlipping`)
- `faithfulness_stats_*.json`:
  - detailed per-image metric arrays + image-level metadata
- `debug/layer_*_debug.npz`:
  - sparse feature debug arrays used by case study analysis
- comparison outputs:
  - `detailed_sweep_comparison.csv`, `sweep_summary_table.csv`
