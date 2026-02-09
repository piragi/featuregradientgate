# Fresh Environment Validation Protocol

## Goal
Establish a repeatable validation flow before logic-changing work:
- verify core APIs still load and match expected signatures
- verify sweep orchestration is deterministic under fixed seeds
- verify branch/tag workflow follows `AGENTS.md`

## Validation Tiers

### Tier 1: Fast smoke (always run)
Runs in a fresh checkout without datasets/models.

```bash
uv sync
uv run python scripts/validation/git_workflow_audit.py --json-output ./docs/validation/git_workflow_audit.json
uv run pytest tests/test_smoke_contracts.py tests/test_sweep_reproducibility.py
```

Expected outcomes:
- smoke tests pass
- sweep contracts and reproducibility checks pass in mocked mode

### Tier 2: Full stack integration (opt-in, heavy)
Runs real downloads and real sample sweep checks.

Prerequisites:
- Hugging Face access for ImageNet (`uvx hf auth login`)
- network access for Google Drive and Hugging Face
- CUDA available for the current SAE loader path (`load_steering_resources` calls `.cuda()`)

```bash
# Optional explicit setup run
uv run python -m gradcamfaith.data.setup

# Full integration tests (download + seeded sample sweep reproducibility)
GRADCAMFAITH_RUN_FULL_STACK=1 uv run pytest -m integration tests/test_integration_fresh_env.py
```

Expected outcomes:
- asset download paths exist (`data/*`, `data/models/*`, SAE folders)
- two sample sweeps with the same seed produce identical normalized `results.json` payloads
- changing seed produces a different normalized payload

## Seed Reproducibility Contract
Use `random_seed=42` as baseline in validation runs.

Contract checks implemented:
- sweep configs record `random_seed`
- reproducibility test compares normalized `results.json` payloads between two seeded runs

## Git Workflow Enforcement
Run before every handoff and before integration:

```bash
uv run python scripts/validation/git_workflow_audit.py
```

Audit checks:
- current branch is a slice branch (`wp/...`)
- branch name matches policy (`wp/WP-<NN>[slice]-<slug>`)
- not working on `main` or integration branch directly
- integration branch `feature/team-research-restructure-plan` exists (local or remote)
- working tree is clean before handoff/integration
- accepted checkpoint tags (`accepted/...`) are present (warn if none)

## Evidence To Attach To Review/Handoff
- command list executed
- pass/fail result for Tier 1 and Tier 2
- git workflow audit output (`docs/validation/git_workflow_audit.json`)
- reproducibility evidence: seed-equal run comparison + seed-different run comparison
- known blockers (for example missing HF credentials, missing CUDA)
