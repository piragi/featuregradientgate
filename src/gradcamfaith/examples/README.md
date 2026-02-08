# gradcamfaith Examples

## Minimal Run

Demonstrates SAE feature-gradient gating vs vanilla TransLRP attribution on a tiny image subset.

### Quick Start

```bash
# Install dependencies
uv sync

# Dry-run (validates config and manifest, no data required)
uv run python -m gradcamfaith.examples.minimal_run --dry-run

# Full run (requires dataset and model assets)
uv run python -m gradcamfaith.examples.minimal_run
```

### Modes

**Explore mode** (default): edit `ExampleConfig` fields directly in `minimal_run.py`, then run. This is the intended research workflow for fast iteration.

```python
from gradcamfaith.examples.minimal_run import ExampleConfig, run_example

config = ExampleConfig(
    dataset_name="hyperkvasir",
    layers=[6, 9, 10],
    subset_size=10,
)
results = run_example(config)
```

**Paper mode**: use a frozen config from `configs/paper/v1/` (placeholder path, to be populated after paper submission). This ensures exact reproducibility of reported results.

### Prerequisites

For a full run (not dry-run), you need:

1. **Prepared dataset** at `./data/<dataset>_unified/` (created by `uv run setup.py`)
2. **Model checkpoint** referenced in `dataset_config.py`
3. **SAE weights** at `./data/sae_<dataset>/layer_*/` (downloaded by `uv run setup.py`)

### Expected Output Tree

```
experiments/example_run_<timestamp>/
  run_manifest.json          # Reproducibility manifest (config, seed, SHA, timestamp)
  vanilla/
    test/
      attributions/          # Per-image attribution maps (.npy)
      analysis_*.csv         # Faithfulness/correctness analysis
      faithfulness_stats_*.json
  gated/
    test/
      attributions/
      analysis_*.csv
      faithfulness_stats_*.json
```

### Dry-Run Output

When running with `--dry-run`, only the manifest is written:

```
experiments/example_run_<timestamp>/
  run_manifest.json
```

The manifest contains:

```json
{
  "resolved_config": { ... },
  "seed": 42,
  "timestamp": "2026-02-06T...",
  "git_sha": "abc123...",
  "environment_lock": "uv.lock"
}
```
