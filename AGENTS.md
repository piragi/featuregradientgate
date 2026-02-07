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
- Workpackage branch naming: `wp/<WP-ID>-<short-slug>` (example: `wp/WP-01-core-package-bootstrap`).
- Merge policy: all workpackage branches merge back into `feature/team-research-restructure-plan` only after review.
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

Per task:
1. Implement one scoped task.
2. Make exactly one commit for that task.
3. Provide a concise change summary for review.
4. Stop and wait for feedback.

After feedback:
1. Implement requested changes.
2. Commit feedback updates as a new commit.
3. Provide summary.
4. Wait again for feedback.

After each accepted change:
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
- Last successful commit reflected here: `WP-06A on branch wp/WP-06-clarity-cleanup`
- What happened most recently: `WP-06A completed — produced core responsibility map (docs/refactor/wp06_core_responsibility_map.md) with function inventory, overlap analysis, unused parameter ledger, slice plan (WP-06B-E), and equivalence contract with baseline signatures/lint. Fixed circular import in models/__init__.py (lazy re-export).`
- Reviewer decision: `pending review`
- What should happen next: `review WP-06A audit artifact, then proceed with WP-06B (attribution boundary refactor)`
- Immediate next task (concrete): `WP-06B: annotate adapter/core roles, extract _postprocess_attribution, remove unused device param from apply_gradient_gating_to_cam`
- Immediate validation for that task: `public signature check unchanged, import smokes pass, numeric equivalence max_diff == 0.0 on synthetic gate tensor`
- Known blockers/risks now: `kappa and clamp_max in compute_feature_gradient_gate are plumbed through config but silently ignored (active formula uses hardcoded 10 ** tanh). Requires maintainer decision: retain as reserved with comment, or re-wire into active formula.`
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

## Tooling and Commands
Preferred command style:
- `uv sync`
- `uv run <script>.py`
- `uv run python -m <module>`
- `uvx <tool>` for one-off tooling

Keep command examples in docs using `uv` only.

## Validation While Tests Are Deferred
Since tests are postponed, each structural task must include a minimal smoke validation:
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

### WP-06 Targeted Clarity Cleanup
- Goal: reduce cognitive load in migrated code via no-behavior-change simplification.
- In scope: identify and resolve unused arguments, redundant helpers, and overlong functions where decomposition improves readability.
- In scope: prioritize recently migrated modules first:
  - `src/gradcamfaith/experiments/sweep.py`
  - `src/gradcamfaith/experiments/case_studies.py`
  - `src/gradcamfaith/data/download.py`
- In scope: for intentionally reserved parameters, keep signature but mark clearly (for example `_param`) and document why retained.
- Required output:
  - short “cleanup inventory” in PR summary listing each removed/renamed unused argument and each function split
  - before/after function map for changed modules
- Required behavior constraints:
  - no metric semantic changes
  - no output artifact naming/path changes
  - no CLI surface expansion
- Out of scope: algorithmic changes, metric redesign, or dataset logic changes.
- Depends on: WP-04.
- Acceptance checks: `uvx ruff check src/gradcamfaith/experiments src/gradcamfaith/data --select ARG001,ARG002` passes for touched files.
- Acceptance checks: public entrypoint signatures remain unchanged unless explicitly approved and documented.
- Acceptance checks: representative import/path smokes still pass after cleanup.
- Deliverables: lower-complexity code in target modules, documented cleanup decisions, and unchanged runtime semantics.

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
1. Review `WP-05` on branch `wp/WP-05-example-path`, then assign `WP-06` (Targeted Clarity Cleanup) with branch name `wp/WP-06-clarity-cleanup`.
2. Require one commit for the workpackage and include validation output summary in the PR description.
3. Review against `Workpackage Review Checklist`, then update `Feature Tracker (Living)` with accepted result and next assignment.

## Done Criteria for This Rework
- Core method code is isolated from experiment orchestration.
- Experiment scripts are grouped and discoverable.
- A newcomer can run one example and one experiment with documented `uv` commands.
- `AGENTS.md` reflects current structure and next planned steps at all times.
