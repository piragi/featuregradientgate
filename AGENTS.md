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

## Current Repo Snapshot (post WP-10)
- Zero root `.py` files remain. All code lives in the package.
- Package code (`src/gradcamfaith/`) is the canonical source for all modules:
  - `core/` — attribution, gating, config, types
  - `data/` — dataset_config, dataloader, download, prepare, setup, io_utils
  - `models/` — load, sae_resources, clip_classifier
  - `experiments/` — pipeline, classify, sweep, sae_train, comparison, case_studies, faithfulness, saco
  - `examples/` — minimal_run
- `models/__init__.py` uses clean eager re-exports of `load_model_for_dataset` and `load_steering_resources` (no `__getattr__` hack).
- Dependency graph is unidirectional: `core → data → models → experiments` (no cycles).

## Target Structure
Package layout under `src/gradcamfaith` with unidirectional dependency flow: `core → data → models → experiments`.

```text
src/gradcamfaith/
  core/
    attribution.py      # TransLRP attribution + compute_attribution orchestrator
    gating.py           # feature-gradient gate computation
    types.py            # shared dataclasses (ClassificationResult, etc.)
    config.py           # PipelineConfig, FileConfig, BoostingConfig
  data/
    dataset_config.py   # DatasetConfig, transform factories, dataset registry
    dataloader.py       # UnifiedMedicalDataset, create_dataloader
    download.py         # download helpers
    prepare.py          # dataset conversion
    setup.py            # download+prepare CLI orchestrator
    io_utils.py         # cache and results I/O
  models/
    load.py             # load_model_for_dataset
    sae_resources.py    # load_steering_resources
    clip_classifier.py  # CLIPClassifier, CLIPModelWrapper
  experiments/
    pipeline.py         # run_unified_pipeline (main experiment orchestrator)
    classify.py         # per-image classification + attribution
    sweep.py            # parameter sweep orchestration + SweepConfig
    faithfulness.py     # shared perturbation infra + faithfulness orchestration
    pixel_flipping.py   # PatchPixelFlipping metric
    faithfulness_correlation.py  # FaithfulnessCorrelation metric
    saco.py             # SaCo attribution analysis
    sae_train.py        # SAE training
    comparison.py       # post-hoc experiment comparison
    case_studies.py     # qualitative case study analysis
  examples/
    minimal_run.py
    README.md
```

Notes:
- Keep modules focused. Prefer small composable files over one large orchestrator.
- No circular dependencies: experiments may import from models/data/core; models from data/core; data from core only.
- Add at least one runnable example (`examples/minimal_run.py`) that demonstrates the end-to-end happy path on a tiny subset.

## Migration Map (Current -> Target)
Completed migrations (WP-01 through WP-07):
- `transmm.py` + `feature_gradient_gating.py` → `core/attribution.py` and `core/gating.py` (done WP-01/WP-06B)
- `config.py` + `data_types.py` → `core/config.py` + `core/types.py` (done WP-01)
- `main.py` → `experiments/sweep.py` (done WP-04, root wrapper removed WP-07)
- `sae.py` → `experiments/sae_train.py` (done WP-04, root wrapper removed WP-07)
- `comparsion.py` → `experiments/comparison.py` (done WP-04, root wrapper removed WP-07)
- `analysis_feature_case_studies.py` → `experiments/case_studies.py` (done WP-04, root wrapper removed WP-07)
- All 8 thin root compatibility wrappers removed (done WP-07)

Remaining migrations (WP-08 through WP-10):
- `dataset_config.py` → `data/dataset_config.py` (WP-08)
- `unified_dataloader.py` → `data/dataloader.py` (WP-08, renamed)
- `io_utils.py` → `data/io_utils.py` (WP-08)
- `setup.py` → `data/setup.py` (WP-08)
- `clip_classifier.py` → `models/clip_classifier.py` (WP-08)
- `faithfulness.py` → `experiments/faithfulness.py` (WP-08)
- `saco.py` → `experiments/saco.py` (WP-08)
- `pipeline.py` → `experiments/pipeline.py` + `experiments/classify.py` (WP-10, decomposed)

## Legacy Compatibility Policy
- All root compatibility wrappers have been removed (WP-07). No root shim maintenance needed.
- Remaining root files (`pipeline.py`, `dataset_config.py`, etc.) are canonical code, not shims — they will be migrated into the package and deleted (WP-08 through WP-10).
- After WP-10: zero root `.py` files. All entrypoints via `gradcamfaith.*` package paths only.

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
- Last successful commit reflected here: `WP-10 integrated, accepted/wp-10 tagged`
- Last accepted integration checkpoint: `accepted/wp-10`
- WP-06B status: `done and accepted`
- WP-06C status: `done and accepted`
- WP-06D status: `done and accepted`
- WP-07 status: `done and accepted`
- WP-08 status: `done and accepted — 7 root files migrated into package, ~25 import sites updated. Only pipeline.py remains at root.`
- WP-09 status: `done and accepted`
- WP-10 status: `done and accepted`
- WP-11 status: `done and accepted`
- WP-12 status: `done and accepted`
- WP-13 status: `in progress — I/O cleanup: remove dead writes, gate debug outputs, remove SaCo CSV cache.`
- What happened most recently: `WP-13 implemented. I/O audit found ~8 write types in main pipeline, only 3 consumed downstream. Removed dead .npz, gated CSVs behind debug_mode, removed brittle use_cached_perturbed mechanism, extracted debug accumulation from pipeline.py main loop, consolidated debug_data/ → debug/ path, merged two debug flags into single debug_mode.`
- Reviewer decision: `WP-12 accepted. WP-13 pending review.`
- What should happen next: `review and integrate WP-13.`
- Immediate next task (concrete): `push WP-13 branch, integrate into feature/team-research-restructure-plan, tag accepted/wp-13.`
- Immediate validation for that task: `all 14+3 tests pass (including full-stack golden value test). Main pipeline flow reduced to 3 essential write types.`
- Known blockers/risks now: `none`
- Known follow-up (deferred from WP-06D): `sweep.py still contains resource lifecycle helpers (_load_dataset_resources, _release_dataset_resources, _gpu_cleanup, _build_imagenet_clip_prompts) that belong in models/. Extract to models/ in a future WP.`
- Decision log pointer: `all accepted structural decisions must be appended in this section`

### WP-06D Concrete Plan (`experiments/sweep.py`)

Current state: 492 lines, 3 functions (`run_single_experiment`, `run_parameter_sweep`, `main`).

Problems:
1. **`run_single_experiment` (133 lines)** mixes four concerns:
   - PipelineConfig construction from experiment_params dict (lines 57–87)
   - ImageNet-specific CLIP setup (lines 69–75) — duplicated in `run_parameter_sweep`
   - Experiment metadata save (lines 89–100)
   - Pipeline execution + result save + GPU cleanup (lines 102–152)
2. **`run_parameter_sweep` (272 lines)** is a monolith that mixes:
   - Output directory + sweep config save (lines 187–209)
   - Per-dataset resource loading with ImageNet-specific CLIP prompt construction (lines 213–263) — the prompt logic uses inline helper functions and duplicates the CLIP setup from `run_single_experiment`
   - Vanilla baseline experiment (lines 267–313)
   - Gated experiment grid loop via `itertools.product` (lines 315–370)
   - Per-experiment result summarization + GC (lines 348–370)
   - Heavy per-dataset cleanup (model/CLIP/SAE to CPU, multi-round GC) (lines 374–416)
   - Sweep summary save (lines 418–426)
3. **Duplicated CLIP/ImageNet logic** — both `run_single_experiment` and `run_parameter_sweep` independently configure CLIP settings for ImageNet, with slightly different prompt templates ("a photo of a {cls}" vs article-aware prompts).
4. **`gc.collect()` / `torch.cuda.empty_cache()` scattered** throughout — cleanup logic appears in 4 different places with varying levels of aggressiveness.
5. **No clear reading order** — a newcomer can't tell why vanilla runs first, what the parameter grid means, or how results are structured without reading the whole function.

Planned refactor:
1. **`_build_pipeline_config(dataset_name, experiment_params, output_dir, current_mode, debug_mode)`** — consolidate PipelineConfig construction including ImageNet/CLIP setup. Single source of truth for config, eliminates duplication between `run_single_experiment` and `run_parameter_sweep`.
2. **`_build_experiment_grid(layer_combinations, kappa_values, gate_constructions, shuffle_decoder_options, clamp_max_values)`** — generate the list of experiment param dicts (vanilla baseline + all gated combinations) with their directory names. Makes the grid explicit and testable.
3. **`_gpu_cleanup(aggressive=False)`** — centralize GC + CUDA cache clearing. Replace the 4 scattered cleanup blocks.
4. **`_load_dataset_resources(dataset_name, layer_combinations, device)`** — extract model/SAE/CLIP loading from the per-dataset loop body. Returns `(model, clip_classifier, steering_resources)`.
5. **`_release_dataset_resources(model, clip_classifier, steering_resources)`** — extract the heavy teardown block.
6. **Simplify `run_single_experiment`** — receives a fully-built PipelineConfig instead of rebuilding it internally from an experiment_params dict.
7. **Simplify `run_parameter_sweep`** — becomes a clear loop: load resources → iterate experiments → release resources → save summary.

Hard constraints:
- No algorithm/metric behavior changes.
- Same output directory structure and file naming (tests depend on `layers_3_kappa_0.5_combined_clamp_10.0` naming).
- Signatures may change freely (tests catch breakage).
- All existing tests must pass.

### WP-08 Concrete Plan (Structural Move of Leaf Root Files)

Goal: Move 7 of 8 remaining root `.py` files into their target package locations. Update all imports. Delete root files. `pipeline.py` stays at root (deferred to WP-10).

Move table:

| Root file | Destination | New import path |
|---|---|---|
| `dataset_config.py` (230L) | `data/dataset_config.py` | `gradcamfaith.data.dataset_config` |
| `unified_dataloader.py` (206L) | `data/dataloader.py` | `gradcamfaith.data.dataloader` |
| `io_utils.py` (96L) | `data/io_utils.py` | `gradcamfaith.data.io_utils` |
| `setup.py` (71L) | `data/setup.py` | `gradcamfaith.data.setup` |
| `clip_classifier.py` (278L) | `models/clip_classifier.py` | `gradcamfaith.models.clip_classifier` |
| `faithfulness.py` (929L) | `experiments/faithfulness.py` | `gradcamfaith.experiments.faithfulness` |
| `saco.py` (746L) | `experiments/saco.py` | `gradcamfaith.experiments.saco` |

Import rewrite sites (~25 total):
- `dataset_config` — 15 sites: `data/__init__.py:2`, `data/prepare.py:18,195,257`, `models/load.py:14`, `experiments/sweep.py:30`, `experiments/sae_train.py:13`, `experiments/case_studies.py:72,1108`, plus internal imports in moving files (`unified_dataloader:16`, `faithfulness:19,303`, `saco:264`, `setup:45`), plus `pipeline.py:23`
- `clip_classifier` — 2 sites: `models/load.py:51` (conditional), `pipeline.py:397` (conditional)
- `unified_dataloader` — 2 sites: `experiments/case_studies.py:74`, `pipeline.py:28`
- `io_utils` — 1 site: `pipeline.py:19`
- `faithfulness` — 1 site: `pipeline.py:24`
- `saco` — 1 site: `pipeline.py:25`
- `setup` — 3 sites: `pipeline.py:26`, `tests/test_smoke_contracts.py:17,43`, `tests/test_integration_fresh_env.py:32`

`__init__.py` updates:
- `data/__init__.py` — re-export `get_dataset_config`, `DatasetConfig`, key configs from new `dataset_config` location
- `models/__init__.py` — add `clip_classifier` re-exports (keep existing lazy `__getattr__` for pipeline until WP-10)

Hard constraints:
- All test pass after the move.
- `pipeline.py` stays at root — its internal imports updated to point at new package paths, but the file itself is not moved.
- No behavior changes. Structural move only.

### WP-09 Concrete Plan (Cleanup and Consolidation)

Goal: Clean up after WP-08 — remove dead re-exports, promote conditional imports to top-level, define clean `__init__.py` public APIs.

Concrete actions:
1. **`data/setup.py`**: Remove the 14 re-export lines (`download_with_progress`, `extract_tar_gz`, `download_hyperkvasir`, etc.). These were re-exported for backward compatibility when `setup.py` was a root shim. After migration, callers go directly to `data.download` / `data.prepare`. Keep only `main()`.
2. **Conditional imports → top-level**: `saco.py:264` has conditional `from dataset_config import get_dataset_config` inside a function body. `faithfulness.py:303` has the same. Now that all files are in the package, these can become top-level imports (no circular dependency risk).
3. **`data/__init__.py` public API**: Define a clean re-export surface: `get_dataset_config`, `DatasetConfig`, `create_dataloader`, `get_single_image_loader`, `convert_dataset`.
4. **Evaluate `io_utils` placement**: Contains cache/results I/O used only by `pipeline.py`. Decision: stays in `data/` for now. When pipeline decomposes in WP-10, cache functions naturally move with the classify module.
5. **Evaluate naming**: `dataloader.py` (from `unified_dataloader.py`) is clean. `io_utils.py` could become `cache.py` but minimal churn preferred — defer.

Hard constraints:
- No behavior changes. Cleanup only.
- All tests pass.

### WP-10 Concrete Plan (Pipeline Breakup)

Goal: Decompose root `pipeline.py` (436 lines, 5 functions) into focused package modules. Remove dead code. Resolve the circular dependency in `models/__init__.py`. Delete root `pipeline.py`. Zero root `.py` files remain.

Current `pipeline.py` function inventory (pre-decomposition analysis):
- `prepare_dataset_if_needed` (lines 34-60) — data preparation check + `convert_dataset` call. Pure data logic.
- `classify_single_image` (lines 63-107) — **DEAD CODE**: defined but never called anywhere in the codebase.
- `save_attribution_bundle_to_files` (lines 110-129) — save `.npy` attribution arrays. Helper for `classify_explain_single_image`.
- `classify_explain_single_image` (lines 132-211) — per-image classification + attribution + caching. Core per-image logic.
- `run_unified_pipeline` (lines 214-435) — full orchestrator: prepare data, loop images, accumulate debug, save CSV, run faithfulness, run SaCo, extract SaCo stats.
- Re-exports (lines 30-31): `load_model_for_dataset`, `load_steering_resources` — pure pass-throughs, canonical code already in `models/`.

Known bugs/dead code in current `pipeline.py`:
- Line 271: dead expression `config.classify.use_clip if hasattr(config.classify, 'use_clip') else False` — result never assigned.
- `classify_single_image`: never called. Remove entirely.

Decomposition:

| Function | Action | Destination | Rationale |
|---|---|---|---|
| `classify_single_image` | **Delete** | — | Dead code, never called |
| `prepare_dataset_if_needed` | **Move** | `data/setup.py` | Pure data preparation logic |
| `save_attribution_bundle_to_files` | **Move** | `experiments/classify.py` (new) | Per-image I/O helper |
| `classify_explain_single_image` | **Move** | `experiments/classify.py` (new) | Per-image classification + attribution |
| `run_unified_pipeline` (lines 214-413) | **Move** | `experiments/pipeline.py` (new) | Orchestrator — loop, debug accumulation, CSV save, faithfulness call |
| SaCo result extraction (lines 414-431) | **Move into `saco.py`** | `experiments/saco.py` | Belongs with SaCo logic; expose as utility (e.g. `extract_saco_summary`) called by orchestrator |
| Dead expression (line 271) | **Delete** | — | Bug: result of ternary never assigned |
| Re-exports (lines 30-31) | **Delete** | — | Callers import directly from `models.load` / `models.sae_resources` |

Debug accumulation (lines 301-391, ~90 lines inside `run_unified_pipeline`):
- Stays inline in `experiments/pipeline.py` for now — it's orchestration bookkeeping.
- **Noted as future cleanup**: extract debug accumulation + save into a helper when debug mode gets more complex or when we add new debug channels.

Circular dependency resolution:
- Current cycle: `pipeline.py → models.load → models/__init__ → pipeline.py` (via lazy `__getattr__`)
- Fix: Remove `__getattr__` from `models/__init__.py`. `run_unified_pipeline` belongs in `experiments/`, not `models/`. After decomposition, `experiments/pipeline.py` imports from `models.load` (downward dependency, no cycle).

Import updates (all `from pipeline import` sites):
- `experiments/sweep.py:31` → `from gradcamfaith.experiments.pipeline import run_unified_pipeline` + `from gradcamfaith.models.load import load_model_for_dataset` + `from gradcamfaith.models.sae_resources import load_steering_resources`
- `experiments/sae_train.py:14` → `from gradcamfaith.models.load import load_model_for_dataset`
- `experiments/case_studies.py:73` → `from gradcamfaith.models.load import load_model_for_dataset` + `from gradcamfaith.models.sae_resources import load_steering_resources`
- `tests/test_smoke_contracts.py:16,42` → `from gradcamfaith.experiments.pipeline import run_unified_pipeline` + `from gradcamfaith.models.load import load_model_for_dataset` + `from gradcamfaith.models.sae_resources import load_steering_resources`
- `models/__init__.py` → remove lazy `__getattr__` entirely, replace with clean eager re-exports of `load_model_for_dataset`, `load_steering_resources`

New file: `experiments/classify.py` (~80 lines):
- `save_attribution_bundle_to_files` — save `.npy` arrays
- `classify_explain_single_image` — per-image classification + attribution + caching
- Imports: `gradcamfaith.data.io_utils`, `gradcamfaith.data.dataloader`, `gradcamfaith.data.dataset_config`, `gradcamfaith.core.attribution`, `gradcamfaith.core.types`

New file: `experiments/pipeline.py` (~200 lines):
- `run_unified_pipeline` — orchestrator (prepare, loop, debug accumulate, save CSV, faithfulness, SaCo)
- Imports: `gradcamfaith.data.*`, `gradcamfaith.experiments.classify`, `gradcamfaith.experiments.faithfulness`, `gradcamfaith.experiments.saco`, `gradcamfaith.models.clip_classifier`

Addition to `experiments/saco.py`:
- `extract_saco_summary(saco_analysis)` — extract overall/per-class/by-correctness SaCo stats from analysis output. Currently inlined in `run_unified_pipeline` lines 414-431.

Final dependency graph (no cycles):
```
core (types, config, gating, attribution)
  ↑
data (dataset_config, dataloader, download, prepare, setup, io_utils)
  ↑
models (load, sae_resources, clip_classifier)
  ↑
experiments (pipeline, classify, sweep, faithfulness, saco, ...)
```

Hard constraints:
- All tests pass.
- Zero root `.py` files remain.
- `models/__init__.py` has no `__getattr__` hack.
- No behavior changes (except dead code removal).

Known future cleanup (deferred):
- Debug accumulation block in `experiments/pipeline.py` (~90 lines) — extract into helper when debug mode evolves.
- `experiments/classify.py` uses `io_utils` from `data/` for caching — evaluate whether cache logic should move closer to classify when it becomes more complex.

### WP-11 Concrete Plan (Faithfulness Metric Decomposition)

Goal: Split monolithic `faithfulness.py` (929L) into focused modules with shared perturbation infrastructure. Compress `saco.py` (780L) by removing ceremony and deduplicating. No behavior changes. Target: ~1020L total down from ~1710L.

#### faithfulness.py decomposition

**Stays in `faithfulness.py`** (shared perturbation infra + orchestration, ~300L):

| Function | Role | Notes |
|---|---|---|
| `create_patch_mask` | Shared perturbation infra | Generalize to return (H, W) mask; callers broadcast channels as needed |
| `apply_baseline_perturbation` | Shared perturbation infra | Unchanged |
| `_predict_torch_model` → `predict_on_batch` | Shared perturbation infra | Rename for clarity |
| `convert_patch_attribution_to_image` + `_normalize_attribution_format` | Shared perturbation infra | **Merge** into single `normalize_patch_attribution` |
| `calc_faithfulness` | Orchestration | Simplify: kill `FaithfulnessEstimatorConfig`, directly instantiate estimators |
| `_run_estimator_trials` | Orchestration | **Inline** `_process_estimator_output` (called once, 10L) |
| `evaluate_faithfulness_for_results` | Orchestration | Unchanged |
| `_prepare_batch_data` | Data preparation | Unchanged |
| `evaluate_and_report_faithfulness` | Entry point | Unchanged (called by `experiments/pipeline.py`) |
| `_build_results_structure` | Reporting | Unchanged |
| `_print_faithfulness_summary` | Reporting | Unchanged |
| `_save_faithfulness_results` | Reporting | Unchanged |
| `_compute_statistics_from_scores` | Statistics | Unchanged |

**Delete from faithfulness.py:**

| Item | Action |
|---|---|
| `FaithfulnessEstimatorConfig` dataclass | Delete — inline direct instantiation in `calc_faithfulness` |
| `handle_array_values` | Delete — inline `.tolist()` at call site |
| `_process_estimator_output` | Delete — inline into `_run_estimator_trials` |
| `faithfulness_pixel_flipping` factory | Delete — absorb `patch_size` logic into `PatchPixelFlipping.__init__` |
| `faithfulness_correlation` factory | Delete — absorb `patch_size` logic into `FaithfulnessCorrelation.__init__` |

**Move to `experiments/pixel_flipping.py`** (new, ~100L):

| Item | Notes |
|---|---|
| `PatchPixelFlipping` class | Including `__call__` and `evaluate_batch` |
| `patch_size` auto-detection | Absorb from deleted `faithfulness_pixel_flipping` factory into `__init__` |
| Imports | `create_patch_mask`, `apply_baseline_perturbation`, `predict_on_batch` from `faithfulness` |

**Move to `experiments/faithfulness_correlation.py`** (new, ~120L):

| Item | Notes |
|---|---|
| `FaithfulnessCorrelation` class | Including `__call__`, `evaluate_batch`, `_compute_spearman_correlation` |
| `patch_size` auto-detection | Absorb from deleted `faithfulness_correlation` factory into `__init__` |
| Imports | `create_patch_mask`, `apply_baseline_perturbation`, `predict_on_batch` from `faithfulness` |

#### saco.py compression (780L → ~500L)

| Current | Action |
|---|---|
| `ImageData` dataclass | **Remove** — pass components directly |
| `BinnedPerturbationData` dataclass | **Remove** — return tuple `(bins, perturbed_tensors)` |
| `BinImpactResult` dataclass | **Remove** — return tuple `(bin_results, saco_score, bin_biases)` |
| `create_spatial_mask_for_bin` | **Replace** — call `faithfulness.create_patch_mask` with `bin_info.patch_indices` |
| `batched_model_inference` | **Keep** — different interface from `predict_on_batch` (returns full prediction dicts, not just target-class scores) |
| `analyze_key_attribution_patterns` | **Inline** into `run_binned_saco_analysis` — near no-op (filters correct predictions, renames column) |
| `run_binned_attribution_analysis` (wrapper) | **Merge** body of `run_binned_saco_analysis` into this function. Keep `run_binned_attribution_analysis` as the public name (used by `experiments/pipeline.py`). Delete `run_binned_saco_analysis`. |
| `apply_binned_perturbation` | **Keep** — PIL-based perturbation with dataset-specific transforms is genuinely different from numpy `apply_baseline_perturbation`. Adapt to accept numpy (H, W) mask from shared `create_patch_mask` instead of torch tensor. |

#### Import updates

- `experiments/pixel_flipping.py` → imports `create_patch_mask`, `apply_baseline_perturbation`, `predict_on_batch` from `gradcamfaith.experiments.faithfulness`
- `experiments/faithfulness_correlation.py` → same imports from `gradcamfaith.experiments.faithfulness`
- `experiments/saco.py` → imports `create_patch_mask` from `gradcamfaith.experiments.faithfulness`
- `experiments/pipeline.py` → **no change** (still imports `evaluate_and_report_faithfulness` from `faithfulness` and `run_binned_attribution_analysis` from `saco`)
- `experiments/faithfulness.py` → imports `PatchPixelFlipping` from `pixel_flipping` and `FaithfulnessCorrelation` from `faithfulness_correlation` (in `calc_faithfulness`)

#### Expected line counts

| File | Before | After |
|---|---|---|
| `faithfulness.py` | 929L | ~300L |
| `pixel_flipping.py` | — | ~100L |
| `faithfulness_correlation.py` | — | ~120L |
| `saco.py` | 780L | ~500L |
| **Total** | **1709L** | **~1020L** |

Hard constraints:
- All tests pass.
- No behavior changes — same metrics, same outputs, same file formats.
- Public entry points unchanged: `evaluate_and_report_faithfulness` (faithfulness.py), `run_binned_attribution_analysis` (saco.py).
- `experiments/pipeline.py` imports are not affected.

### WP-12 Concrete Plan (SaCo Simplification)

Goal: Make saco.py readable and concise. The core idea is simple — bin patches by attribution, perturb each bin, check that higher-attribution bins cause bigger confidence drops — but the current code obscures this with dense matrix math, a PIL perturbation round-trip, unused fields, and reporting mixed into computation. Target: ~250L down from 429L.

**Accepted behavior change:** Switching from PIL-based perturbation (perturb original image → retransform) to tensor-level perturbation (perturb already-transformed tensor via shared `apply_baseline_perturbation`) will cause minor numerical differences in SaCo scores. The metric semantics are identical (mean-value replacement on masked patches). This is approved by the maintainer.

#### Step 1: Simplify `calculate_saco_vectorized_with_bias` → `calculate_saco`

Current code (lines 60-86) uses dense matrix operations with unclear variable names (`violation_weights`, `is_faithful`, `weights`, `weights_upper`). The bias computation is unused downstream.

**Rewrite to:**
```python
def calculate_saco(attributions, confidence_drops):
    """Pairwise concordance: do higher-attribution bins cause bigger drops?

    For each pair (i, j) where attr[i] > attr[j], check if drop[i] > drop[j].
    SaCo = (concordant_pairs - discordant_pairs) / total_pairs.
    """
    n = len(attributions)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            attr_diff = attributions[i] - attributions[j]
            drop_diff = confidence_drops[i] - confidence_drops[j]
            if attr_diff * drop_diff > 0:
                concordant += abs(attr_diff)
            elif attr_diff * drop_diff < 0:
                discordant += abs(attr_diff)
    total = concordant + discordant
    return (concordant - discordant) / total if total > 0 else 0.0
```

Changes:
- Remove `bin_bias` computation entirely (never used downstream).
- Rename to `calculate_saco` (no `_vectorized_with_bias` — it's neither vectorized in the useful sense nor returning bias anymore).
- Rename `confidence_impacts` → `confidence_drops` for clarity.
- Keep the weighted formulation (weighted by `abs(attr_diff)`) to match current behavior exactly.
- Simple loop is clearer than the matrix approach for 20 bins (20×20 = 400 comparisons — trivial).

#### Step 2: Unify perturbation path — delete `apply_binned_perturbation`

Current `apply_binned_perturbation` (lines 142-177) does:
1. Compute PIL image mean intensity
2. Create a gray PIL layer
3. Convert numpy mask to PIL mask, paste gray, get composite PIL image
4. Re-run dataset-specific transforms to get a tensor

Replace with shared `apply_baseline_perturbation` from `faithfulness.py` which does:
1. `np.where(mask, image.mean(), image)` on the already-transformed `(C, H, W)` numpy array

This means `create_binned_perturbations` takes the cached tensor (`_cached_tensor`) instead of a PIL image. The PIL image open, `ImageStat`, and `get_dataset_config` transform pipeline are all deleted.

**Concrete changes:**
- Delete `apply_binned_perturbation` function entirely.
- Delete `from PIL import Image, ImageStat` (PIL no longer needed).
- Delete `from gradcamfaith.data.dataset_config import get_dataset_config` (no longer needed for transforms).
- `create_binned_perturbations` signature changes: takes `image_tensor` (numpy C,H,W) instead of `pil_image` + `dataset_name` + `perturbation_method`. Uses `apply_baseline_perturbation(image_tensor, mask, "mean")` from faithfulness.py and wraps result with `torch.from_numpy(...).float()`.
- `load_image_and_attributions` returns `(cached_tensor, raw_attributions, confidence, class_idx)` instead of `(pil_image, ...)`. Uses `_cached_tensor` (already available). Falls back to opening+transforming only if no cache.
- `calculate_binned_saco_for_image` updated to pass tensor through the new flow.

#### Step 3: Simplify `batched_model_inference`

Current version (lines 37-53) returns full prediction dicts with `probabilities` tensor, `predicted_class_idx`, and `confidence`. But callers only need `confidence` and `predicted_class_idx`.

**Rewrite to:**
```python
def _classify_batch(model, batch_tensor, device):
    """Run inference, return list of (predicted_class_idx, confidence)."""
    model.eval()
    with torch.no_grad():
        logits = model(batch_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        idxs = torch.argmax(probs, dim=1)
        confs = probs[torch.arange(len(idxs)), idxs]
    return list(zip(idxs.cpu().tolist(), confs.cpu().tolist()))
```

Changes:
- Rename to `_classify_batch` (private, descriptive).
- Return tuples instead of dicts — callers destructure directly.
- Remove `batch_size` parameter — all current callers pass the full batch at once (`batch_size=len(batch_tensor)`). The outer batching loop was dead code.
- Remove `probabilities` from return (never used).

#### Step 4: Simplify `measure_bin_impacts`

Current version (lines 221-241) builds result dicts with 7 fields including `bin_attribution_bias` (added later by `compute_saco_from_impacts`), `class_changed` (never read), and `confidence_delta_abs` (never read).

**Rewrite to return parallel arrays:**
```python
def _measure_bin_drops(bins, perturbed_tensors, original_confidence, model, device):
    """Measure confidence drop when each bin is perturbed. Returns numpy array of drops."""
    batch = torch.stack(perturbed_tensors)
    preds = _classify_batch(model, batch, device)
    return np.array([original_confidence - conf for _, conf in preds])
```

This replaces `measure_bin_impacts` + `compute_saco_from_impacts` chain. The SaCo computation becomes:
```python
attributions = np.array([b.mean_attribution for b in bins])
drops = _measure_bin_drops(bins, perturbed_tensors, confidence, model, device)
saco_score = calculate_saco(attributions, drops)
```

#### Step 5: Remove dead fields and simplify data flow

| Dead field/code | Where | Action |
|---|---|---|
| `bin_attribution_bias` | `compute_saco_from_impacts` → bin result dicts | Delete (never read by any consumer) |
| `class_changed` | `measure_bin_impacts` result dicts | Delete (never read) |
| `confidence_delta_abs` | `measure_bin_impacts` result dicts | Delete (never read) |
| `probabilities` in `batched_model_inference` return | Return dicts | Delete (never read) |
| `probabilities` in `_analyze_faithfulness_vs_correctness` | DataFrame column | Delete (never read) |
| `compute_saco_from_impacts` function | Standalone function | Delete — inline 3-line SaCo call into `calculate_binned_saco_for_image` |
| `measure_bin_impacts` function | Standalone function | Replace with `_measure_bin_drops` (simpler return type) |
| `debug` parameter on `calculate_binned_saco_for_image` | Function signature | Delete (single print statement, never passed as True from any caller) |

#### Step 6: Simplify reporting in `run_binned_attribution_analysis`

Current reporting block (lines 348-391) mixes post-hoc analysis, summary printing, and file I/O into one function. The "attribution patterns" concept is just "correct predictions with SaCo scores" — a one-line DataFrame filter, not worth naming.

**Simplify to:**
- `_get_or_compute_binned_results` returns `(saco_scores_by_image: dict, bin_results_df: DataFrame)` — the DataFrame for CSV saving, the dict for downstream use.
- `_analyze_faithfulness_vs_correctness` stays but drops `probabilities` column.
- The "attribution patterns" block (lines 361-366) deletes entirely — it's a filtered view that's saved as CSV but never used programmatically.
- The double-save of `bin_results` (lines 381-388 save it once as `{ds_name}_bin_results.csv` and once as `analysis_bin_results_binned_{timestamp}.csv`) reduces to a single save.

#### Step 7: Keep `BinInfo` and `extract_saco_summary` unchanged

- `BinInfo` dataclass has 7 meaningful fields and clear semantics — keep.
- `extract_saco_summary` is the public API used by `experiments/pipeline.py` — keep unchanged.

#### Expected result

| Item | Before (429L) | After (~250L) |
|---|---|---|
| Core algorithm (`calculate_saco`) | 27L dense matrix | ~15L clear loop |
| Perturbation (`apply_binned_perturbation`) | 36L PIL round-trip | 0L (deleted, uses shared `apply_baseline_perturbation`) |
| Model inference (`batched_model_inference`) | 17L returning dicts | ~8L returning tuples |
| Per-bin measurement | 21L + 16L (two functions) | ~5L (one helper) |
| Per-image pipeline | 25L | ~20L |
| Dataset-level analysis | 70L | ~50L |
| Post-hoc / reporting | 50L | ~30L |
| Binning | 43L | 43L (unchanged) |
| `extract_saco_summary` | 15L | 15L (unchanged) |
| **Total** | **429L** | **~250L** |

#### Import changes

- Add: `from gradcamfaith.experiments.faithfulness import apply_baseline_perturbation` (in addition to existing `create_patch_mask`)
- Delete: `from PIL import Image, ImageStat`
- Delete: `from gradcamfaith.data.dataset_config import get_dataset_config`
- Keep: `from gradcamfaith.core.types import ClassificationResult` (used for type context)

#### Hard constraints

- All tests pass.
- Public entry points unchanged: `run_binned_attribution_analysis(config, vit_model, original_results, device, n_bins)`, `extract_saco_summary(saco_analysis)`.
- `experiments/pipeline.py` imports not affected.
- CSV output format may lose columns (`class_changed`, `confidence_delta_abs`, `bin_attribution_bias`) — these are unused analysis artifacts.
- SaCo score values will differ slightly from pre-WP-12 due to perturbation path change. This is accepted.

### WP-13 Concrete Plan (I/O Cleanup)

Goal: Declutter the main pipeline code flow by removing dead I/O writes, consolidating debug outputs into a gated `debug/` subdirectory, and removing the brittle SaCo CSV cache. No behavior changes to metrics or analysis.

**Principle:** The main pipeline produces only what's consumed downstream. Everything else is gated behind `debug_mode` and written to `output_dir/debug/`.

#### I/O Audit Summary

**Always saved (consumed by downstream code):**

| What | Format | Written by | Read by |
|------|--------|-----------|---------|
| `.npy` attributions (per image) | npy | `classify.py:37,39` | `faithfulness.py:295`, `saco.py:138` |
| JSON cache (per ClassificationResult) | JSON | `io_utils.py:47` | `io_utils.py:31` |
| `faithfulness_stats_*.json` | JSON | `faithfulness.py:392` | `comparison.py:39`, `case_studies.py:288` |

**Never consumed (remove or gate behind debug):**

| What | Format | Written by | Action |
|------|--------|-----------|--------|
| `results_{dataset}_unified.csv` | CSV | `pipeline.py:181` | Move to debug |
| `faithfulness_scores_*.npz` | npz | `faithfulness.py:399` | Delete (data already in JSON) |
| `{ds_name}_bin_results.csv` | CSV | `saco.py:284` | Move to debug |
| `analysis_*_binned_*.csv` (3 timestamped files) | CSV | `saco.py:287` | Delete entirely |
| SaCo CSV cache load (`use_cached_perturbed`) | CSV | `saco.py:217-222` | Delete (brittle) |

#### Config changes

**`core/config.py`:**
- Remove: `use_cached_perturbed: str = ""` from `FileConfig`
- Consolidate debug flags: single `debug_mode` on `BoostingConfig` gates ALL debug I/O (gate/feature data collection, classification CSV, SaCo bin CSV, debug .npz). Removed redundant `save_debug_outputs` from `FileConfig`.

#### Changes by file

**`experiments/pipeline.py`:**
- Gate `save_classification_results_to_csv` behind `debug_mode`
- Write to `output_dir/debug/classification_results.csv` instead of `output_dir/results_{dataset}_unified.csv`
- Extract ~60 lines of debug accumulation (per-image loop body) into `_accumulate_debug_info` helper
- Extract debug saving (classification CSV + gate/attribution .npz) into `_save_debug_outputs` helper
- Consolidate debug .npz output from `output_dir/debug_data/` → `output_dir/debug/`

**`experiments/faithfulness.py` (`_save_faithfulness_results`):**
- Delete the `.npz` save — data already in JSON (`mean_scores`, `std_scores` as lists)
- Clean up function signature (remove unused `faithfulness_results` and `class_labels` params)
- Keep JSON save unchanged (consumed by `comparison.py:39`, `case_studies.py:288`)

**`experiments/saco.py` (`run_binned_attribution_analysis`):**
- Remove `use_cached_perturbed` cache load in `_get_or_compute_binned_results`
- Gate `bin_results_df.to_csv()` behind `debug_mode`, write to `output_dir/debug/saco_bin_results.csv`
- Delete the timestamped analysis CSV loop entirely
- Remove unused `datetime` import

**`experiments/sweep.py`:**
- Remove `pipeline_config.file.use_cached_perturbed = ""` — field no longer exists

**`experiments/case_studies.py`:**
- Rename `debug_data/` path constant → `debug/` (5 sites: extraction writer, existence check, `load_debug_data`, `load_activation_data`) — consistent with pipeline.py's new output path

#### Hard constraints
- No metric or algorithm changes.
- `.npy` attributions always saved (hard requirement).
- Faithfulness JSON always saved (consumed by `comparison.py`, `case_studies.py`).
- No changes to what `run_unified_pipeline` returns.
- All tests pass.

#### Expected outcome
- Main pipeline flow has 3 write types (npy, JSON cache, faithfulness JSON) instead of ~8
- Debug outputs consolidated in `output_dir/debug/` subdirectory (was split across `debug_data/` and root output dir)
- `run_unified_pipeline` main loop is scannable — debug accumulation extracted to helpers
- `use_cached_perturbed` config field and SaCo CSV cache mechanism removed
- I/O code in saco.py and faithfulness.py significantly shorter

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
- **WP-06C (gating boundary refactor)**: Extracted 4 private helpers from `core/gating.py`: `_extract_decoder` (SAE decoder dispatch), `_compute_patch_scores` (4-branch gate construction), `_collect_gate_debug_info` (sparse feature debug collection), `_apply_gate_to_cam` (CLS-aware gate application + delta computation). Public function bodies reduced to clear sequential steps. Gate equivalence: max_diff == 0.0. All 11 tests pass.
- **Intermediate (golden-value regression test)**: Added `test_imagenet_golden_faithfulness_values` to `tests/test_integration_fresh_env.py`. Full-stack test runs imagenet val split, subset=500, seed=123, combined gate, layer [3], kappa=0.5, clamp_max=10.0 via `run_parameter_sweep`. Asserts exact golden values for SaCo (mean/std), PixelFlipping (count/mean/median), and FaithfulnessCorrelation (count/mean/median). Explicit cleanup via `shutil.rmtree` in `finally` block. Gated behind `GRADCAMFAITH_RUN_FULL_STACK=1` + CUDA.
- **WP-06D (sweep readability rewrite)**: Extracted 8 helpers from `experiments/sweep.py`: `_gpu_cleanup` (centralized GC/CUDA), `_print_gpu_memory` (GPU memory reporting), `_build_pipeline_config` (PipelineConfig construction), `_build_experiment_grid` (vanilla + gated param grid), `_build_imagenet_clip_prompts` (article-aware CLIP prompts), `_load_dataset_resources` (model/CLIP/SAE loading), `_release_dataset_resources` (teardown), `_summarize_result` (result extraction). Added `SweepConfig` dataclass matching `run_parameter_sweep` parameters 1:1 for `**asdict(cfg)` unpacking. `main()` simplified to config-first pattern. Public signatures unchanged. All 14 tests pass. Deferred: resource lifecycle helpers still in sweep.py — should move to `models/` in WP-07.
- **WP-07 (legacy removal and thorough refactor)**: Deleted 8 root compatibility wrappers (`main.py`, `transmm.py`, `feature_gradient_gating.py`, `config.py`, `data_types.py`, `comparsion.py`, `analysis_feature_case_studies.py`, `sae.py`). Removed deprecated `transmm_prisma_enhanced` and `generate_attribution_prisma_enhanced` from `core/attribution.py` (along with unused `import warnings` and `HookedSAEViT` import). Updated 4 root files to package imports: `pipeline.py` (`config`→`gradcamfaith.core.config`, `data_types`→`gradcamfaith.core.types`), `faithfulness.py` (same), `io_utils.py` (`data_types`→`gradcamfaith.core.types`), `saco.py` (same). Rewrote `test_smoke_contracts.py` to package-only imports (removed all root wrapper import tests, added `SweepConfig` to import check). Rewrote `test_attribution_boundary_contracts.py` (removed `test_legacy_wrappers_deprecated`, added `test_deprecated_shims_removed`). All 14 tests pass. Resource extraction from sweep.py deferred to future WP.
- **WP-08 (structural move of root files)**: Migrated 7 root files into package: `dataset_config.py`→`data/dataset_config.py`, `unified_dataloader.py`→`data/dataloader.py` (renamed), `io_utils.py`→`data/io_utils.py`, `setup.py`→`data/setup.py`, `clip_classifier.py`→`models/clip_classifier.py`, `faithfulness.py`→`experiments/faithfulness.py`, `saco.py`→`experiments/saco.py`. Updated ~25 import sites across `src/gradcamfaith/`, `pipeline.py`, and `tests/`. Updated `data/__init__.py` to re-export from new `dataset_config` location. Only `pipeline.py` remains at root. All 14 tests pass.
- **WP-09 (cleanup and consolidation)**: Removed 14 dead re-exports from `data/setup.py` — kept only imports actually used by `main()` plus `convert_dataset` (used by `pipeline.py`). Promoted conditional `get_dataset_config` imports to top-level in `experiments/saco.py` and removed redundant inline import in `experiments/faithfulness.py`. Updated `pip install` hint to `uv add` in `data/setup.py`. All 14 tests pass.
- **WP-10 (pipeline breakup)**: Decomposed root `pipeline.py` (436 lines) into focused package modules. Created `experiments/pipeline.py` (orchestrator: `run_unified_pipeline`), `experiments/classify.py` (per-image: `save_attribution_bundle_to_files`, `classify_explain_single_image`). Added `extract_saco_summary` to `experiments/saco.py` (extracted from inline SaCo result extraction). Moved `prepare_dataset_if_needed` to `data/setup.py`. Deleted dead code: `classify_single_image` (never called), line 271 dead expression (ternary result never assigned), re-exports of `load_model_for_dataset`/`load_steering_resources` (callers now import directly). Replaced `models/__init__.py` lazy `__getattr__` with clean eager re-exports — resolves circular dependency `pipeline → models.load → models/__init__ → pipeline`. Updated 4 consumer import sites: `experiments/sweep.py`, `experiments/sae_train.py`, `experiments/case_studies.py`, `tests/test_smoke_contracts.py`. Updated `test_attribution_boundary_contracts.py` to check `experiments/classify.py` instead of root `pipeline`. Root `pipeline.py` deleted. Zero root `.py` files remain. All 14 tests pass. Known future cleanup: debug accumulation block (~90 lines) in `experiments/pipeline.py` — extract into helper when debug mode evolves.
- **WP-11 (faithfulness metric decomposition)**: Split `faithfulness.py` (929L) into 3 files: `faithfulness.py` (401L, shared perturbation infra + orchestration + reporting), `pixel_flipping.py` (90L, `PatchPixelFlipping` class), `faithfulness_correlation.py` (112L, `FaithfulnessCorrelation` class). Compressed `saco.py` (780→429L): removed 3 dataclasses (`ImageData`, `BinnedPerturbationData`, `BinImpactResult`), replaced with tuple returns; deduplicated `create_spatial_mask_for_bin` by importing shared `create_patch_mask` from faithfulness.py; inlined single-use functions (`analyze_key_attribution_patterns`, `run_binned_saco_analysis`); made `_analyze_faithfulness_vs_correctness` private. Shared perturbation infra: `create_patch_mask` (returns numpy H,W mask), `apply_baseline_perturbation` (handles broadcasting), `predict_on_batch`, `normalize_patch_attribution`. Factory functions absorbed into class `__init__`. Total LOC: 1709→1032 (40% reduction). All 14 tests pass.
- **WP-12 (SaCo simplification + perturbation bugfix)**: Simplified `saco.py` (429→324L). **Bugfix: spatial misalignment in SaCo perturbation** — old `apply_binned_perturbation` applied the 224x224 patch mask at the original PIL image resolution (e.g., 500x375), then resize+center-crop back to 224x224 distorted patch boundaries. Diagnostic confirmed: 489 unmasked pixels changed (max diff 0.44), masked fill non-uniform (std=0.18/channel from resize interpolation blending). Fixed by perturbing directly on the cached (C,224,224) tensor using shared `apply_baseline_perturbation` — exact patch grid alignment, uniform fill, no PIL round-trip. Golden SaCo values updated (gated mean: 0.2633→0.3217, std: 0.4044→0.4138). Rewrote `calculate_saco_vectorized_with_bias` → `calculate_saco`: clear pairwise loop using signed `attr_diff` weights with descending sort (required for correct sign). Removed bias computation (never used). Replaced `batched_model_inference` → `_classify_batch`: returns numpy arrays instead of dicts. Replaced `measure_bin_impacts` + `compute_saco_from_impacts` → `_measure_bin_drops`. Removed 5 dead fields, deleted attribution patterns block, removed double-save, renamed `_analyze_faithfulness_vs_correctness` → `_join_saco_with_correctness`. All tests pass including full-stack golden value test.

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
  - **WP-06D (sweep readability rewrite)**:
    - rewrite `experiments/sweep.py` for clarity and separation of concerns
    - extract config construction, experiment grid generation, resource loading/teardown, and GPU cleanup into focused helpers
    - eliminate duplicated ImageNet/CLIP setup between `run_single_experiment` and `run_parameter_sweep`
    - preserve output directory structure, file naming, and metric behavior
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

### WP-07 Legacy Removal (done)
- Status: **done** (commit `860e6bc` on `wp/WP-07-legacy-removal`, pending integration).
- Completed: removed 8 root compatibility wrappers, deleted deprecated attribution shims, updated all imports and tests to package paths.
- Deferred to WP-08/09/10: root canonical code migration, pipeline decomposition, resource lifecycle extraction.

### WP-08/09/10: Complete Root File Migration and Pipeline Breakup
- Goal: migrate all remaining root `.py` files into the package, clean up post-migration, and decompose `pipeline.py` into focused modules. Zero root `.py` files when done.
- In scope:
  - **WP-08 (structural move)**: move 7 leaf root files into package, update ~25 import sites, delete root files. See WP-08 Concrete Plan above.
  - **WP-09 (cleanup)**: remove dead re-exports from `setup.py`, promote conditional imports, define clean `__init__.py` APIs. See WP-09 Concrete Plan above.
  - **WP-10 (pipeline breakup)**: decompose `pipeline.py` into `experiments/pipeline.py` + `experiments/classify.py`, move `prepare_dataset_if_needed` to `data/setup.py`, resolve `models/__init__.py` circular dependency. See WP-10 Concrete Plan above.
- Hard constraints:
  - All algorithm/metric behavior stays the same.
  - All existing tests must pass.
  - Final dependency graph: `core → data → models → experiments` (no cycles).
- Depends on: WP-07.
- Acceptance checks: all tests pass, zero root `.py` files, no `__getattr__` hacks, clean `ruff check`.
- Deliverables: fully packaged codebase, clean dependency graph, decomposed pipeline.

## Workpackage Review Checklist
- Scope respected (`in scope` only, no silent expansion).
- Behavior parity maintained unless explicitly approved.
- `uv`-only commands and validation evidence included.
- Compatibility path preserved (root script calls still valid unless approved change).
- `AGENTS.md` updated with outcomes, decisions, and next goal proposal.
- Reviewer decision recorded: `accepted`, `accepted with follow-ups`, or `rework requested`.

## Immediate Next Steps (Concrete)
1. Complete WP-13 (I/O cleanup).
2. Assess next priorities with maintainer:
   - Sweep compression (resource lifecycle extraction to `models/`, deferred from WP-06D).
   - `case_studies.py` and `comparison.py` compression.
   - Debug accumulation cleanup in `experiments/pipeline.py` (~90 lines).
   - Paper preparation infrastructure (frozen configs, reproduction entrypoints).
   - WP-06E (unused-argument + dead-surface cleanup).

## Done Criteria for This Rework
- Core method code is isolated from experiment orchestration.
- Experiment scripts are grouped and discoverable.
- A newcomer can run one example and one experiment with documented `uv` commands.
- `AGENTS.md` reflects current structure and next planned steps at all times.
