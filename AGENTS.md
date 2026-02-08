# AGENTS.md

## Purpose
This repository is a research environment, not a software product. The code should optimize for fast understanding, fast iteration, and reproducible exploration by new researchers.

Primary objective for the rework: reduce cognitive load without forcing more code.

## What This Code Does
- Core idea: modulate attention-based attribution using learned sparse autoencoder (SAE) features to improve faithfulness.
- Main comparison: `combined` SAE feature-gradient gating vs `no_SAE` baseline behavior.
- Key ablations: `activation_only`, `gradient_only`, and combined interaction variants.
- The codebase is intentionally ablation-heavy and iteration-oriented, i.e. classic research code for hypothesis testing.

## Ground Rules
- Use `uv` for all tooling.
- Do not introduce direct `pip`, `python`, or ad-hoc venv workflows in docs/scripts.
- Tests are intentionally deferred for now; prioritize structural clarity and reproducible execution first.
- Keep legacy behavior available through the current `main` branch compatibility path while rework proceeds.

## Research Interface Policy (Config-First)
- This is research code. Preferred interface is config-first, not CLI-first.
- Typed Python configs (dataclasses / config objects) are acceptable and encouraged for fast local sweeps.
- Large, generic CLI surfaces are discouraged unless there is a clear maintenance benefit.
- Keep thin runners only (for example: load config + execute), not flag-heavy orchestration.

For concurrent paper release:
- Maintain frozen paper configs under a versioned path (for example `configs/paper/v1/`).
- Every reported run must persist a resolved run manifest:
  - full resolved config
  - git commit SHA
  - random seed(s)
  - timestamp
  - environment lock reference (`uv.lock`)
- Provide dedicated reproduction entrypoints for paper outputs (tables/figures), separate from exploratory scripts.
- Keep two explicit modes documented:
  - `explore`: editable configs, rapid iteration
  - `paper`: frozen configs, strict reproducibility
- Minimal CLI overrides are allowed (`--config` and limited key overrides), but config files remain source of truth.

## Branching For Team Execution
- Team integration branch for this program: `feature/team-research-restructure-plan`.
- All coders branch from `feature/team-research-restructure-plan`, not from `main`.
- Branches are slice-scoped, not long-lived per whole workpackage.
- Single-slice workpackage branch naming: `wp/<WP-ID>-<short-slug>` (example: `wp/WP-03-data-setup-split`).
- Slice branch naming: `wp/<WP-ID><slice>-<short-slug>` (examples: `wp/WP-06A-core-audit-contract`, `wp/WP-06B-attribution-boundary-refactor`).
- Start each new slice branch from the latest `feature/team-research-restructure-plan` HEAD.
- Merge policy: after review, merge or cherry-pick accepted slice commit(s) into `feature/team-research-restructure-plan` immediately.
- Tag every accepted integration checkpoint on integration branch (example: `accepted/wp-06a`).
- Next slice always branches from the new integration HEAD after the accepted checkpoint is in.
- Promotion to `main` happens later as a separate integration decision after workpackage validation.

## Current Repo Snapshot (as of this plan)
- Active tracked code is flat at repo root (`main.py`, `pipeline.py`, `transmm.py`, `feature_gradient_gating.py`, `setup.py`, etc.).
- `src/gradcamfaith/core/` now contains tracked source files from WP-01:
  - `src/gradcamfaith/core/attribution.py`
  - `src/gradcamfaith/core/gating.py`
  - `src/gradcamfaith/core/config.py`
  - `src/gradcamfaith/core/types.py`
- Root compatibility wrappers are now active for migrated modules:
  - `transmm.py`
  - `feature_gradient_gating.py`
  - `config.py`
  - `data_types.py`
- Major concerns are mixed in single files:
  - Core method math + hook orchestration.
  - Experiment sweep orchestration.
  - Dataset download + dataset conversion.
  - Analysis scripts and plotting.

## Target Structure
Use a real package layout under `src/gradcamfaith` with clear boundaries:

```text
src/gradcamfaith/
  core/
    attribution.py
    gating.py
    faithfulness.py
    saco.py
    types.py
    config.py
  data/
    datasets.py
    dataloader.py
    prepare.py
    download.py
  models/
    load.py
    clip.py
    sae_resources.py
  experiments/
    sweep.py
    sae_train.py
    comparison.py
    case_studies.py
  examples/
    minimal_run.py
    README.md
  cli/
    main.py
    setup.py
```

Notes:
- Keep modules focused. Prefer small composable files over one large orchestrator.
- Keep orchestration in `cli/` and experiment entrypoints; keep method logic in `core/`.
- Add at least one runnable example (`examples/minimal_run.py`) that demonstrates the end-to-end happy path on a tiny subset.

## Migration Map (Current -> Target)
- `transmm.py` + `feature_gradient_gating.py` -> `core/attribution.py` and `core/gating.py`.
- `faithfulness.py` -> `core/faithfulness.py`.
- `saco.py` -> `core/saco.py`.
- `config.py` + `data_types.py` -> `core/config.py` + `core/types.py`.
- `dataset_config.py` + `unified_dataloader.py` -> `data/datasets.py` + `data/dataloader.py`.
- `setup.py` -> split into `data/download.py` and `data/prepare.py`; keep thin CLI wrapper.
- `pipeline.py` -> split responsibilities across `models/load.py`, `models/sae_resources.py`, and a lighter experiment/runtime orchestrator.
- `clip_classifier.py` -> `models/clip.py`.
- `main.py` -> `experiments/sweep.py` + thin CLI wrapper.
- `sae.py` -> `experiments/sae_train.py`.
- `comparsion.py` (typo) -> `experiments/comparison.py` (keep compatibility shim during migration).
- `analysis_feature_case_studies.py` -> `experiments/case_studies.py`.

## Legacy Compatibility Policy
During migration:
- Keep root entry scripts as compatibility shims that import and call new package modules.
- Do not break existing root commands immediately:
  - `uv run main.py`
  - `uv run setup.py`
  - `uv run sae.py`
  - `uv run comparsion.py`
  - `uv run analysis_feature_case_studies.py`
- Remove compatibility shims only after equivalent package entrypoints are validated and documented.

## Workflow Policy (Mandatory)
Use a feature branch for the rework.

Per execution slice:
1. Implement one scoped task.
2. Make exactly one commit for that slice.
3. Provide a concise change summary for review.
4. Stop and wait for feedback.

After feedback:
1. Implement requested changes.
2. Commit feedback updates as a new commit.
3. Provide summary.
4. Wait again for feedback.

After each accepted change:
- Integrate that accepted slice into `feature/team-research-restructure-plan` immediately.
- Create/update accepted checkpoint tag on integration branch (`accepted/<wp-slice>`).
- Start the next slice branch from the updated integration HEAD.
- Update `AGENTS.md` with any noteworthy decisions, deviations, or new constraints.
- Reevaluate this plan and define the next concrete step.
- Update the `Feature Tracker (Living)` section with what happened and what is next.

## Team Delivery Model
- This phase is executed by a team of coders; maintainers review and set next goals.
- Every workpackage must be independently shippable with one clear outcome.
- Every workpackage must include a short handoff note: what changed, why, how validated, and known risks.
- Evaluation loop: team delivers workpackage -> maintainers review outcome -> maintainers define next goals -> `AGENTS.md` is updated.
- If scope changes mid-workpackage, coders stop and request scope confirmation before continuing.

## "Noteworthy" Must Be Logged Here
Always record in `AGENTS.md`:
- Structural decisions (moved/split/merged modules).
- Naming changes and compatibility shims added/removed.
- New entrypoints and deprecated ones.
- Any change to expected researcher workflow.
- Any decision that could affect reproducibility.

## Feature Tracker (Living)
This tracker is a required, evolving log for project state and near-term execution. Update it after every successful commit.

- Program branch: `feature/team-research-restructure-plan`
- Branching mode: `slice branches + immediate integration + accepted checkpoint tags`
- Last successful commit reflected here: `golden-value regression test added to feature/team-research-restructure-plan`
- Last accepted integration checkpoint: `accepted/wp-06b`
- What happened most recently: `Added golden-value regression test (test_imagenet_golden_faithfulness_values) for imagenet val subset=500/seed=123/combined/layer3. Catches silent pipeline behavior changes.`
- Reviewer decision: `accepted`
- What should happen next: `create wp/WP-06C-gating-boundary-refactor from integration HEAD and execute gating refactor.`
- Immediate next task (concrete): `WP-06C: split compute_feature_gradient_gate into focused internal helpers (score construction, gate mapping, debug packaging) in core/gating.py. Preserve public signatures.`
- Immediate validation for that task: `public signature checks unchanged, import smokes pass, fixed-seed synthetic equivalence max_diff == 0.0 for gate outputs.`
- Known blockers/risks now: `none`
- Decision log pointer: `all accepted structural decisions must be appended in this section`

### Decision Log
- **WP-01**: Added `[build-system]` (hatchling) and `[tool.hatch.build.targets.wheel]` to pyproject.toml to make `src/gradcamfaith` an installable package. This is required for absolute imports (`from gradcamfaith.core.config import ...`) to work. `uv sync` installs the package in dev mode automatically.
- **WP-01**: Internal imports within the package use absolute paths (`from gradcamfaith.core.config import ...`), not relative imports, for clarity.
- **WP-01**: Root compatibility wrappers use `from gradcamfaith.core.<module> import *` plus explicit named re-exports to preserve all existing import patterns.
- **WP-02**: `load_model_for_dataset` moved to `src/gradcamfaith/models/load.py`, `load_steering_resources` moved to `src/gradcamfaith/models/sae_resources.py`. Both re-exported from `pipeline.py` for compatibility.
- **WP-02**: `models/load.py` imports `PipelineConfig` from `gradcamfaith.core.config` (package path) and `DatasetConfig` from root-level `dataset_config` (not yet migrated). This is intentional — root modules remain importable via sys.path.
- **WP-02**: Removed `models/` from `.gitignore` and anchored `model/` to root (`/model/`). The unanchored `models/` rule was blocking `src/gradcamfaith/models/` from being tracked. Model weights are still safely ignored by the global `*.pt` rule.
- **WP-03**: Download functions (9) moved to `src/gradcamfaith/data/download.py`, prepare/convert functions (11) moved to `src/gradcamfaith/data/prepare.py`. Root `setup.py` is now a compatibility wrapper with `main()` kept inline (it orchestrates both download and prepare calls).
- **WP-03**: `data/prepare.py` imports `DatasetConfig`, `COVIDQUEX_CONFIG`, `HYPERKVASIR_CONFIG` from root-level `dataset_config` (not yet migrated). Same sys.path pattern as WP-02.
- **WP-03 review**: Verified parity against pre-WP-03 `setup.py`: all required moved functions exist, signatures are unchanged, and function bodies match (structural move only).
- **WP-03 follow-up**: Two user-facing dependency hints still mention `pip install` (`setup.py`, `src/gradcamfaith/data/download.py`). Keep behavior unchanged for now; update wording to `uv`-compatible guidance in a later doc/tooling cleanup task.
- **WP-04**: `run_single_experiment`, `run_parameter_sweep`, `main` moved from `main.py` to `src/gradcamfaith/experiments/sweep.py`. Import changed: `import config` → `import gradcamfaith.core.config as config`.
- **WP-04**: `train_single_config`, `main`, `SWEEP_CONFIG` moved from `sae.py` to `src/gradcamfaith/experiments/sae_train.py`. No internal import changes needed.
- **WP-04**: All 14 functions moved from `comparsion.py` to `src/gradcamfaith/experiments/comparison.py` (typo-safe canonical name). Root `comparsion.py` wrapper preserved for backward compatibility. No internal import changes needed.
- **WP-04**: All 14 functions moved from `analysis_feature_case_studies.py` to `src/gradcamfaith/experiments/case_studies.py`. Import changed inside `_extract_sae_activations`: `import config` → `import gradcamfaith.core.config as config`.
- **WP-04**: Config-first workflow preserved — `SWEEP_CONFIG` dict and inline experiment configs remain editable in-file objects. No CLI orchestration introduced.
- **WP-04**: `seaborn` added as dependency (`uv add seaborn`) — required by `comparsion.py`/`comparison.py` but was missing from pyproject.toml.
- **WP-04 review**: Acceptance import checks passed for root and package experiment entrypoints; function signatures matched WP-04 requirements for `run_single_experiment`, `run_parameter_sweep`, `train_single_config`, `comparison.main`, and `run_case_study_analysis`.
- **WP-04 follow-up**: `_extract_sae_activations` exists in package module but is not re-exported from root `analysis_feature_case_studies.py` wrapper. Decide explicitly: re-export for strict compatibility or document it as intentionally private-only going forward.
- **WP-04 fix**: Re-exported `_extract_sae_activations` from `analysis_feature_case_studies.py` wrapper (underscore-prefixed names are skipped by `import *`; explicit named import required).
- **WP-05**: Added `src/gradcamfaith/examples/minimal_run.py` with `ExampleConfig` dataclass, `run_example(config)`, and `main()`. Imports exclusively from `gradcamfaith.*` package modules (not root wrappers). Supports `--dry-run` flag for validation without data/models.
- **WP-05**: Added `src/gradcamfaith/examples/README.md` with exact `uv` commands, explore/paper mode documentation, prerequisites, and expected output tree.
- **WP-05**: Run manifest emits `resolved_config`, `seed`, `timestamp`, `git_sha`, and `environment_lock` per the Research Interface Policy reproducibility requirements.
- **WP-05**: Heavy imports (torch, model loading) are deferred until after dry-run check, so `--dry-run` completes instantly without GPU or model dependencies.
- **WP-05 review**: Accepted. Two issues fixed: root-module imports in example (now uses `gradcamfaith.data` and `gradcamfaith.models` re-exports), stale AGENTS.md next steps.
- **WP-06A**: Produced `docs/refactor/wp06_core_responsibility_map.md` — complete function inventory (8 attribution, 2 gating), overlap analysis for both function pairs, unused parameter ledger (3 findings: `device`, `kappa`, `clamp_max`), slice plan (WP-06B through WP-06E), and equivalence contract with baseline signatures and lint results.
- **WP-06A**: Fixed circular import in `src/gradcamfaith/models/__init__.py` — eager `from pipeline import run_unified_pipeline` caused circular dependency when import chain passed through `pipeline.py` -> `gradcamfaith.models.load` -> `models/__init__` -> `pipeline`. Converted to lazy `__getattr__` re-export.
- **WP-06A**: Key finding: `kappa` and `clamp_max` parameters are plumbed through the full config path but have zero effect on output. Active gate formula is `10 ** tanh(s_norm)` (hardcoded); the parameterized formula `clamp_max ** tanh(kappa * s_norm)` is commented out. This is the highest-priority clarity issue.
- **WP-06A follow-up**: Maintainer decision on kappa/clamp_max: (1) `kappa` removed from `compute_feature_gradient_gate` signature and full config-to-gate chain; retained in `PipelineConfig` as sweep metadata only. (2) `clamp_max` wired into active gate formula: `clamp_max ** tanh(s_norm)` with default 10.0 (matching previous hardcoded value). Numeric equivalence verified: max_diff == 0.0. ARG001 findings reduced from 5 to 3. `kappa` removed from `ExampleConfig` in `minimal_run.py`.
- **Process update**: Delivery model switched from long-lived workpackage branches to slice-scoped branches with immediate integration into `feature/team-research-restructure-plan` and accepted checkpoint tags (`accepted/wp-06a`, `accepted/wp-06b`, ...).
- **Validation slice (fresh-env readiness)**: Added `tests/test_smoke_contracts.py` (API import/signature contracts, example manifest contract, deterministic gate checks) and `tests/test_sweep_reproducibility.py` (seeded sweep orchestration checks with mocked heavy dependencies) as baseline reproducibility guards before logic refactors.
- **Validation slice (heavy integration)**: Added `tests/test_integration_fresh_env.py` with opt-in `GRADCAMFAITH_RUN_FULL_STACK=1` checks for full asset setup and seeded sample sweep reproducibility. Documented CUDA requirement for current SAE loader path.
- **Workflow governance**: Added `scripts/validation/git_workflow_audit.py` to enforce branch naming, integration-branch visibility, worktree cleanliness, and accepted checkpoint tag visibility against AGENTS workflow policy.
- **Validation runbook**: Added `docs/validation/fresh_environment_validation.md` with Tier-1 (always-run smoke) and Tier-2 (heavy integration) `uv` command matrix plus evidence requirements.
- **Validation follow-up (setup hardening)**: `extract_tar_gz` now prefers `extractall(..., filter="data")` with compatibility fallback for runtimes that do not support the `filter` argument (keeps Python 3.10+ compatibility while addressing Python 3.14 warning).
- **Validation follow-up (idempotent downloads)**: `download_hyperkvasir` now skips 3.7GB dataset re-download when `data/hyperkvasir/labeled-images` is already present and non-empty; model checkpoint download behavior is unchanged.
- **Validation follow-up (tests)**: Added `tests/test_download_guardrails.py` to assert tar extraction fallback behavior and HyperKvasir dataset skip behavior.
- **Validation follow-up (git hygiene)**: Added root `.gitignore` entries for `/models` and `/logs` so full-stack setup/test runs do not leave untracked runtime artifacts that block clean-worktree handoff checks.
- **WP-06B v1 (attribution boundary refactor)**: Extracted `_postprocess_attribution` helper, removed unused `device` param from `apply_gradient_gating_to_cam`, added role docstrings. Review decision: rework requested — naming clarity insufficient.
- **WP-06B-R3 (clean attribution API)**: `compute_attribution` is now the single canonical orchestrator returning `{predictions, attribution_positive, raw_attribution, debug_info}`. `pipeline.py` migrated to `compute_attribution` (no longer imports legacy names). `transmm_prisma_enhanced` and `generate_attribution_prisma_enhanced` converted to thin deprecated shims with `DeprecationWarning` and explicit removal plan (WP-07). `transmm.py` root wrapper re-exports `compute_attribution`. Added `tests/test_attribution_boundary_contracts.py` (4 tests). ARG001 findings: 2 (only download.py). Gate equivalence: max_diff == 0.0.
- **Intermediate (golden-value regression test)**: Added `test_imagenet_golden_faithfulness_values` to `tests/test_integration_fresh_env.py`. Full-stack test runs imagenet val split, subset=500, seed=123, combined gate, layer [3], kappa=0.5, clamp_max=10.0 via `run_parameter_sweep`. Asserts exact golden values for SaCo (mean/std), PixelFlipping (count/mean/median), and FaithfulnessCorrelation (count/mean/median). Explicit cleanup via `shutil.rmtree` in `finally` block. Gated behind `GRADCAMFAITH_RUN_FULL_STACK=1` + CUDA.

## Tooling and Commands
Preferred command style:
- `uv sync`
- `uv run <script>.py`
- `uv run python -m <module>`
- `uvx <tool>` for one-off tooling

Keep command examples in docs using `uv` only.

## Validation While Tests Are Deferred
Full test coverage is still deferred, but baseline validation now exists. Each structural task must include:
- Tier-1 smoke tests:
  - `uv run pytest tests/test_smoke_contracts.py tests/test_sweep_reproducibility.py`
  - `uv run python -m gradcamfaith.examples.minimal_run --dry-run`
- Git workflow audit:
  - `uv run python scripts/validation/git_workflow_audit.py`
- Tier-2 heavy integration (opt-in on credentialed CUDA hosts):
  - `GRADCAMFAITH_RUN_FULL_STACK=1 uv run pytest -m integration tests/test_integration_fresh_env.py`
  - Includes golden-value regression test (`test_imagenet_golden_faithfulness_values`) that validates exact faithfulness metric reproducibility on imagenet val subset.

At minimum, structural tasks must still include minimal smoke validation:
- Module import sanity (`uv run python -c "..."`).
- At least one representative command path still executes.
- If behavior changes intentionally, document it in commit summary and `AGENTS.md`.

## Project Plan (Rough)
1. Package bootstrap: create tracked `src/gradcamfaith` modules and move pure-core files first.
2. Pipeline decomposition: split `pipeline.py` into model loading, runtime orchestration, and IO helpers.
3. Data setup split: separate download and conversion concerns from `setup.py`.
4. Experiment scripts migration: move sweep/SAE/comparison/case-study scripts under `experiments/`.
5. Example path: add one minimal runnable example under `examples/`.
6. Clarity pass: remove unused arguments/dead code and split oversized functions where this lowers cognitive load without behavior changes.
7. Compatibility cleanup: keep shims until new paths are stable, then prune gradually.

## Workpackages For Team
All workpackages below are designed for coder ownership and maintainer review.

### WP-01 Core Package Bootstrap
- Goal: establish `src/gradcamfaith/core` as live source for attribution/gating/config/types.
- In scope: migrate `transmm.py`, `feature_gradient_gating.py`, `config.py`, `data_types.py` into core package modules.
- In scope: keep root compatibility wrappers so current imports and commands still work.
- Out of scope: no behavior refactors, no algorithm changes, no CLI redesign.
- Depends on: none.
- Acceptance checks: `uv run python -c "import transmm, feature_gradient_gating, config, data_types"`.
- Acceptance checks: one representative command still starts (`uv run main.py` allowed to run until setup boundary).
- Deliverables: migrated files, wrappers, short migration map update in `AGENTS.md`.

### WP-02 Pipeline Decomposition
- Goal: break `pipeline.py` into smaller orchestration/model/data IO units.
- In scope: move model-loading logic to `models/load.py`; move SAE loader to `models/sae_resources.py`; keep thin pipeline orchestrator.
- In scope: preserve existing runtime behavior and output locations.
- Required function moves:
  - move `load_model_for_dataset` from `pipeline.py` to `src/gradcamfaith/models/load.py`
  - move `load_steering_resources` from `pipeline.py` to `src/gradcamfaith/models/sae_resources.py`
- Required API compatibility (must remain importable from `pipeline`):
  - `run_unified_pipeline`
  - `load_model_for_dataset`
  - `load_steering_resources`
- Required signature stability (no parameter or return type changes):
  - `run_unified_pipeline(...)`
  - `load_model_for_dataset(...)`
  - `load_steering_resources(...)`
- Allowed file-change surface for WP-02:
  - `pipeline.py`
  - `src/gradcamfaith/models/load.py` (new)
  - `src/gradcamfaith/models/sae_resources.py` (new)
  - minimal import wiring files needed for compatibility only
- Required behavior constraints:
  - no metric logic changes (`faithfulness.py`, `saco.py` semantics unchanged)
  - no output path format changes
  - no experiment configuration semantics changes
- Out of scope: changing metrics definitions or experiment semantics.
- Depends on: WP-01.
- Acceptance checks: `uv run python -c "from pipeline import run_unified_pipeline, load_model_for_dataset, load_steering_resources; print('api-ok')"` remains valid.
- Acceptance checks: `uv run python -c "import inspect; from pipeline import run_unified_pipeline, load_model_for_dataset, load_steering_resources; print(inspect.signature(run_unified_pipeline)); print(inspect.signature(load_model_for_dataset)); print(inspect.signature(load_steering_resources))"` and confirm signatures are unchanged vs pre-WP-02 baseline.
- Acceptance checks: if local data/model assets exist, run one subset smoke with existing pipeline path and document the exact command used and result.
- Required handoff artifacts in PR summary:
  - before/after function location map
  - compatibility import map
  - exact validation commands executed and outcomes
- Deliverables: decomposed modules, compatibility imports, and updated docs.

### WP-03 Data Setup Split
- Goal: split downloading and conversion concerns now mixed in `setup.py`.
- In scope: create `data/download.py` and `data/prepare.py`; keep root `setup.py` as compatibility CLI.
- In scope: retain current dataset support (`imagenet`, `hyperkvasir`, `covidquex`, `waterbirds`).
- Required function moves (`setup.py` -> `src/gradcamfaith/data/download.py`):
  - `download_with_progress`
  - `download_from_gdrive`
  - `extract_zip`
  - `extract_tar_gz`
  - `download_hyperkvasir`
  - `download_imagenet`
  - `download_covidquex`
  - `download_thesis_saes`
  - `download_sae_checkpoints`
- Required function moves (`setup.py` -> `src/gradcamfaith/data/prepare.py`):
  - `_create_output_structure`
  - `_create_conversion_stats`
  - `_save_metadata`
  - `_process_image`
  - `split_ids`
  - `prepare_covidquex`
  - `prepare_hyperkvasir`
  - `prepare_waterbirds`
  - `prepare_imagenet`
  - `convert_dataset`
  - `print_summary`
- Required API compatibility (must remain importable from root `setup.py`):
  - `convert_dataset` (required by `pipeline.py`)
  - `main`
  - existing public download/prepare functions used by current workflows
- Required signature stability (no parameter or return type changes):
  - all moved functions listed above
- Allowed file-change surface for WP-03:
  - `setup.py`
  - `src/gradcamfaith/data/download.py` (new)
  - `src/gradcamfaith/data/prepare.py` (new)
  - minimal import wiring files needed for compatibility only
- Required behavior constraints:
  - do not change dataset output layout (`train/val/test/class_<idx>`)
  - do not change metadata filename/shape (`dataset_metadata.json`)
  - do not change download URLs / IDs / repo IDs unless explicitly approved
  - do not change conversion split behavior or naming conventions
- Out of scope: changing dataset content logic beyond structural extraction.
- Depends on: WP-01.
- Acceptance checks: `uv run python -c "from setup import convert_dataset, main; print('api-ok')"` remains valid.
- Acceptance checks: `uv run python -c "import inspect; from setup import convert_dataset; print(inspect.signature(convert_dataset))"` and confirm signature unchanged vs pre-WP-03 baseline.
- Acceptance checks: if local assets exist, run one non-destructive conversion smoke (small/sample path) and document exact command and result.
- Required handoff artifacts in PR summary:
  - before/after function location map
  - compatibility import map (`setup.py` -> `gradcamfaith.data.*`)
  - exact validation commands executed and outcomes
- Deliverables: separated modules, unchanged compatibility API/CLI behavior, and AGENTS update noting boundaries.

### WP-04 Experiments Migration
- Goal: move experiment drivers into `src/gradcamfaith/experiments`.
- In scope: migrate `main.py`, `sae.py`, `comparsion.py`, `analysis_feature_case_studies.py`.
- In scope: fix typo path by introducing `comparison.py` while preserving `comparsion.py` compatibility.
- In scope: preserve typed in-file config patterns where they accelerate research iteration.
- Required function moves (`main.py` -> `src/gradcamfaith/experiments/sweep.py`):
  - `run_single_experiment`
  - `run_parameter_sweep`
  - `main`
- Required function moves (`sae.py` -> `src/gradcamfaith/experiments/sae_train.py`):
  - `train_single_config`
  - `main`
  - keep `SWEEP_CONFIG` as config-first editable object (or equivalent typed config object)
- Required function moves (`comparsion.py` -> `src/gradcamfaith/experiments/comparison.py`):
  - `cohens_d`
  - `load_experiment_data`
  - `extract_metrics`
  - `get_experiment_info`
  - `load_all_experiments`
  - `calculate_statistical_comparison`
  - `interpret_effect_size`
  - `print_detailed_results`
  - `create_summary_table`
  - `identify_best_performers`
  - `classify_layer_type`
  - `calculate_composite_improvement`
  - `identify_best_overall_performers`
  - `save_results`
  - `main`
- Required function moves (`analysis_feature_case_studies.py` -> `src/gradcamfaith/experiments/case_studies.py`):
  - `extract_sae_activations_if_needed`
  - `_extract_sae_activations`
  - `load_and_preprocess_image`
  - `load_faithfulness_results`
  - `load_debug_data`
  - `load_activation_data`
  - `build_feature_activation_index`
  - `compute_composite_improvement`
  - `find_dominant_features_in_image`
  - `extract_case_studies`
  - `visualize_case_study`
  - `save_case_study_individual_images`
  - `get_image_path`
  - `load_attribution`
  - `run_case_study_analysis`
- Required API compatibility (must remain importable from root modules):
  - from `main`: `run_single_experiment`, `run_parameter_sweep`, `main`
  - from `sae`: `train_single_config`, `main`, `SWEEP_CONFIG`
  - from `comparsion`: analysis/stat functions and `main`
  - from `analysis_feature_case_studies`: `run_case_study_analysis` and extraction helpers
- Required signature stability (no parameter or return type changes) for:
  - `run_single_experiment(...)`
  - `run_parameter_sweep(...)`
  - `train_single_config(...)`
  - `main(...)` in `comparsion.py`
  - `run_case_study_analysis(...)`
- Allowed file-change surface for WP-04:
  - `main.py`
  - `sae.py`
  - `comparsion.py`
  - `analysis_feature_case_studies.py`
  - `src/gradcamfaith/experiments/sweep.py` (new)
  - `src/gradcamfaith/experiments/sae_train.py` (new)
  - `src/gradcamfaith/experiments/comparison.py` (new canonical)
  - `src/gradcamfaith/experiments/case_studies.py` (new)
  - minimal import wiring files needed for compatibility only
- Required behavior constraints:
  - do not change experiment metric semantics (`SaCo`, `FaithfulnessCorrelation`, `PixelFlipping`)
  - do not change output directory/file naming conventions used by downstream analysis
  - preserve config-first workflow (editable in-file sweep/config objects remain valid)
  - do not convert this layer to flag-heavy CLI orchestration
- Out of scope: changing experiment objective functions or metrics interpretation.
- Depends on: WP-01 and WP-02.
- Acceptance checks: `uv run python -c "from main import run_single_experiment, run_parameter_sweep; from sae import train_single_config, SWEEP_CONFIG; from comparsion import main as comparison_main; from analysis_feature_case_studies import run_case_study_analysis; print('root-api-ok')"` remains valid.
- Acceptance checks: `uv run python -c "from gradcamfaith.experiments.sweep import run_single_experiment, run_parameter_sweep; from gradcamfaith.experiments.sae_train import train_single_config; from gradcamfaith.experiments.comparison import main as comparison_main; from gradcamfaith.experiments.case_studies import run_case_study_analysis; print('pkg-api-ok')"` remains valid.
- Acceptance checks: `uv run python -c "import inspect; from main import run_single_experiment, run_parameter_sweep; from sae import train_single_config; from comparsion import main as comparison_main; from analysis_feature_case_studies import run_case_study_analysis; print(inspect.signature(run_single_experiment)); print(inspect.signature(run_parameter_sweep)); print(inspect.signature(train_single_config)); print(inspect.signature(comparison_main)); print(inspect.signature(run_case_study_analysis))"` and confirm signatures unchanged vs pre-WP-04 baseline.
- Acceptance checks: if local assets exist, run one small/subset experiment path and one comparison/case-study path and document exact commands and outcomes.
- Required handoff artifacts in PR summary:
  - before/after function location map for all four migrated scripts
  - compatibility import map (root wrappers -> `gradcamfaith.experiments.*`)
  - exact validation commands executed and outcomes
- Deliverables: package experiment modules, stable root shims, typo-safe comparison path, and updated AGENTS tracker.

### WP-05 Example Path
- Goal: add one clear newcomer path for understanding and running the method.
- In scope: `src/gradcamfaith/examples/minimal_run.py` and `src/gradcamfaith/examples/README.md`.
- In scope: include exact `uv` commands, expected output artifacts, and an example config-first run.
- Required example API shape in `minimal_run.py`:
  - `ExampleConfig` (typed config object/dataclass for the example run)
  - `run_example(config: ExampleConfig)` as primary callable
  - `main()` as thin runner
- Required behavior for the example:
  - must import from package modules (`gradcamfaith.*`), not root compatibility wrappers
  - must support a tiny-subset path for quick execution
  - must emit a minimal run manifest (`resolved config`, `seed`, `timestamp`, `git SHA`) to align with reproducibility policy
  - must document both modes:
    - `explore`: editable config in file
    - `paper`: frozen config reference path (even if placeholder for now)
- Allowed file-change surface for WP-05:
  - `src/gradcamfaith/examples/minimal_run.py` (new)
  - `src/gradcamfaith/examples/README.md` (new)
  - minimal import wiring only if required
- Out of scope: full benchmark scripts.
- Depends on: WP-02 minimum.
- Acceptance checks: `uv run python -c "from gradcamfaith.examples.minimal_run import ExampleConfig, run_example, main; print('example-api-ok')"` remains valid.
- Acceptance checks: one documented command executes the tiny-subset example path (or dry-run path if data is unavailable) and writes the expected manifest/artifact locations.
- Acceptance checks: `README.md` includes exact `uv` commands and a short expected-output tree.
- Required handoff artifacts in PR summary:
  - exact command transcript used for smoke validation
  - generated artifact tree (or dry-run artifact tree) with paths
  - known limitations (e.g., required local data/model assets)
- Deliverables: runnable example module, newcomer README, and reproducibility manifest path.

### WP-06 Deep Clarity + Responsibility Refactor (No Logic Change)
- Goal: make the code substantially easier to read by clarifying responsibility boundaries and reducing unnecessary surface area, without changing algorithmic behavior.
- Hard constraints:
  - no metric semantic changes
  - no attribution/gating algorithm changes
  - no output artifact naming/path changes
  - no CLI surface expansion
- Priority modules:
  - `src/gradcamfaith/core/attribution.py`
  - `src/gradcamfaith/core/gating.py`
  - `src/gradcamfaith/experiments/sweep.py`
  - `src/gradcamfaith/experiments/case_studies.py`
  - `src/gradcamfaith/data/download.py`
- Required process:
  - do responsibility audit first, then refactor in narrow slices
  - one scoped slice per commit
  - for each slice: publish equivalence evidence + readability delta summary
- Recommended execution slices (one commit each):
  - **WP-06A (core audit contract)**: DONE (commit `8f4ae89`)
    - produced `docs/refactor/wp06_core_responsibility_map.md`
    - function inventory, overlap analysis, unused parameter ledger, equivalence contract
    - fixed circular import in `models/__init__.py` (lazy re-export)
  - **WP-06A follow-up**: DONE (commit `01925e0`)
    - removed `kappa` from gate formula chain per maintainer decision
    - wired `clamp_max` into active formula: `clamp_max ** tanh(s_norm)` with default 10.0
    - ARG001 findings reduced from 5 to 3
  - **WP-06B (attribution boundary refactor)**:
    - separate orchestration, attribution post-processing, and output-packing responsibilities in `core/attribution.py`
    - use `compute_attribution` as the single orchestrator (maintainer decision, WP-06B-R3)
    - clean API rewrite allowed: legacy `transmm_prisma_enhanced` / `generate_attribution_prisma_enhanced` do not need signature stability
    - migrate all in-repo call sites to `compute_attribution`
    - optional temporary wrappers are allowed only as deprecated shims with explicit removal plan
    - make wrapper/adapter role explicit so overlap is intentional and documented, not accidental
    - remove unused `device` parameter from internal `apply_gradient_gating_to_cam`
    - implementation contract is documented in `docs/refactor/wp06b_attribution_boundary_spec.md`
  - **WP-06C (gating boundary refactor)**:
    - split `compute_feature_gradient_gate` and `apply_feature_gradient_gating` into focused internal helpers (score construction, gate mapping, CAM application, debug packaging)
    - preserve public signatures unless explicitly approved
  - **WP-06D (experiments readability refactor)**:
    - split oversized orchestration functions in `experiments/sweep.py` and `experiments/case_studies.py` into small private helpers
    - preserve external behavior and output locations
  - **WP-06E (unused-argument + dead-surface cleanup)**:
    - clear `ARG001/ARG002` findings in touched modules (including `data/download.py`)
    - remove or annotate intentionally retained compatibility parameters
- Required output per slice:
  - before/after function map for touched module(s)
  - short rationale of responsibility improvements
  - equivalence evidence summary (numeric or structural) for touched behavior
- Out of scope:
  - algorithm redesign
  - metric formula changes
  - new experiment semantics
- Depends on: WP-04 and WP-05.
- Acceptance checks (global for WP-06):
  - `uvx ruff check src/gradcamfaith/core src/gradcamfaith/experiments src/gradcamfaith/data --select ARG001,ARG002`
  - public entrypoint signatures unchanged unless explicitly approved and documented
  - explicit approved exception: WP-06B-R3 clean attribution API rewrite per `docs/refactor/wp06b_attribution_boundary_spec.md`
  - import/path smokes for root and package entrypoints still pass
  - for slices touching `core/attribution.py` or `core/gating.py`, provide pre/post equivalence evidence on fixed-seed synthetic inputs (max absolute diff reported)
- Deliverables: significantly improved readability in core + experiments, explicit responsibility documentation, and preserved runtime semantics.

### WP-07 Compatibility and Cleanup
- Goal: tighten structure after migrations while preserving legacy branch compatibility promises.
- In scope: audit wrappers, imports, and deprecated paths; remove only what is explicitly approved.
- In scope: update `README.md` and `AGENTS.md` to final structure and command paths.
- Out of scope: dropping legacy wrappers without maintainer sign-off.
- Depends on: WP-01 through WP-06.
- Acceptance checks: documented command matrix remains valid.
- Deliverables: final compatibility report and cleanup summary.

## Workpackage Review Checklist
- Scope respected (`in scope` only, no silent expansion).
- Behavior parity maintained unless explicitly approved.
- `uv`-only commands and validation evidence included.
- Compatibility path preserved (root script calls still valid unless approved change).
- `AGENTS.md` updated with outcomes, decisions, and next goal proposal.
- Reviewer decision recorded: `accepted`, `accepted with follow-ups`, or `rework requested`.

## Immediate Next Steps (Concrete)
1. Review WP-06B-R3 on `wp/WP-06B-attribution-boundary-refactor` and record reviewer decision.
2. After maintainer acceptance, integrate WP-06B-R3 commit(s) into `feature/team-research-restructure-plan`, tag `accepted/wp-06b`.
3. Create branch `wp/WP-06C-gating-boundary-refactor` from integration HEAD and execute one-slice/one-commit flow with equivalence evidence.

## Done Criteria for This Rework
- Core method code is isolated from experiment orchestration.
- Experiment scripts are grouped and discoverable.
- A newcomer can run one example and one experiment with documented `uv` commands.
- `AGENTS.md` reflects current structure and next planned steps at all times.
