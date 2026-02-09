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
- WP-13 status: `done and accepted`
- WP-14 status: `done and accepted`
- WP-15 status: `done and accepted`
- WP-16 status: `done and accepted`
- WP-17 status: `in progress — Parts 1-7 implemented, function splitting done, runtime data format mismatch discovered (pre-existing). Not yet committed.`
- WP-18 status: `planned`
- What happened most recently: `WP-17 implementation revealed that case_studies.py depends on data that the pipeline never saves: (1) per-image SaCo scores as CSV, (2) sparse feature debug data. Also found results.json only has SaCo overview while faithfulness_stats only has FC+PF. Debug NPZ saves 4 unused fields and misses 3 needed fields. WP-18 addresses all of this.`
- Reviewer decision: `WP-17 implementation approved, WP-18 plan needed before commit.`
- What should happen next: `implement WP-18 on wp/WP-17-analysis-cleanup branch (same branch, extends WP-17).`
- Immediate next task (concrete): `implement WP-18 data consolidation changes.`
- Immediate validation for that task: `uv run pytest — all pass. Manual: verify faithfulness_stats JSON contains SaCo, results.json contains all 3 metrics, debug NPZ contains only needed fields.`
- Known blockers/risks now: `WP-18 changes output file formats — comparison.py and case_studies.py loaders must update in lockstep. Existing experiment data uses old format.`
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

### WP-14 Concrete Plan (Config Cleanup + Patch Size Constant + Batch Loop Dedup)

Goal: Reduce confusion in the core code path by cleaning up config dead weight, extracting the scattered patch-size magic into a single utility, and deduplicating the identical batch loop in the two faithfulness metric classes. No behavior changes.

#### Part 1: `core/config.py` cleanup

**1a. Remove redundant `mode_dir` property (lines 20-22)**

`mode_dir` and `output_dir` are **identical** — both return `self.base_pipeline_dir / self.current_mode`. Three internal callers use `mode_dir` (`data_dir`, `vit_inputs_dir`, `perturbed_dir`, `mask_dir`). Change them to use `output_dir` instead and delete `mode_dir`.

Concrete changes:
- Delete `mode_dir` property (lines 20-22)
- `data_dir`: `self.mode_dir` → `self.output_dir` (line 30)
- `vit_inputs_dir`: `self.mode_dir` → `self.output_dir` (line 44)
- `perturbed_dir`: `self.mode_dir` → `self.output_dir` (line 48)
- `mask_dir`: `self.mode_dir` → `self.output_dir` (line 52)
- `directories` list: `self.mode_dir` → `self.output_dir` (line 59) — this was a duplicate entry anyway since `self.output_dir` is already in the list at position 0
- `directory_map`: remove `"mode_dir": self.mode_dir` entry (line 67)

Verify: grep entire codebase for `mode_dir` — no external consumers.

**1b. Consolidate `directories` and `directory_map` into one**

`directories` (list) and `directory_map` (dict) maintain the same paths in parallel. Replace both with a single `directory_map` property. The `directories` property becomes `list(self.directory_map.values())`. On `PipelineConfig`, same delegation.

Concrete changes in `FileConfig`:
- Keep `directory_map` property (lines 62-74), remove the `"mode_dir"` entry (from 1a)
- Rewrite `directories` property to: `return list(self.directory_map.values())`

`PipelineConfig` (lines 179-187): same pattern — `directories` delegates to `self.file.directories`, `directory_map` delegates to `self.file.directory_map`. Both already do this. No change needed.

**1c. Replace `set_dataset` mutation with factory pattern**

`set_dataset()` (line 76-79) mutates `dataset_name` and `base_pipeline_dir` in place. Called from 5 sites: `minimal_run.py:123,138,162`, `case_studies.py:87`, `sweep.py:110,207`.

Replace with `@classmethod` factory:
```python
@classmethod
def for_dataset(cls, dataset_name: str, **kwargs) -> 'FileConfig':
    return cls(
        dataset_name=dataset_name,
        base_pipeline_dir=Path(f"./data/{dataset_name}_unified/results"),
        **kwargs,
    )
```

Update all call sites from:
```python
pipeline_config.file.set_dataset(dataset_name)
```
To:
```python
pipeline_config.file = FileConfig.for_dataset(dataset_name)
```

Note: some callers set other FileConfig fields before `set_dataset`. Check each site to preserve field values. If a caller sets `current_mode` before `set_dataset`, it must pass `current_mode=...` to `for_dataset()` instead.

Call sites (exhaustive):
- `experiments/sweep.py:110` — `_build_pipeline_config`: sets `current_mode` on line 105, then calls `set_dataset`. Must pass `current_mode` to factory.
- `experiments/sweep.py:207` — `_load_dataset_resources`: only calls `set_dataset`. Simple replacement.
- `experiments/case_studies.py:87` — `run_case_study_analysis`: only calls `set_dataset`. Simple replacement.
- `examples/minimal_run.py:123,138,162` — three calls. Check if `current_mode` or other fields are set before. If so, pass them to factory.

Delete `set_dataset` method after all sites updated.

**1d. Fix `analysis` field declaration (line 121)**

`analysis = False` is a **class attribute**, not a dataclass field (missing type annotation). It works at runtime but is invisible to dataclass machinery (`asdict`, `__init__`). Change to proper field:
```python
analysis: bool = False
```

This has callers: `pipeline.py:148`, `sweep.py:108`, `minimal_run.py:141,165`. All set it to `True`. No behavior change — it already works as a class attribute; making it a proper field formalizes it.

**1e. Leave `kappa` alone**

`kappa` on `BoostingConfig` IS used — `sweep.py:121` writes it from experiment params, and `sweep.py:157` uses it in experiment directory naming. The comment says "sweep metadata only" which is correct. No action.

#### Part 2: Extract patch size utility

The expression `32 if n_patches == 49 else 16` appears in **5 locations**:
- `experiments/faithfulness.py:116` — `normalize_patch_attribution`
- `experiments/faithfulness_correlation.py:27` — `FaithfulnessCorrelation.__init__`
- `experiments/pixel_flipping.py:29` — `PatchPixelFlipping.__init__`
- `experiments/saco.py:183` — `_saco_for_image` (uses `model.cfg.patch_size` with fallback 16)
- `experiments/case_studies.py:500` — uses `224 // patches_per_side` (different form, same logic)

Create a single utility function in `experiments/faithfulness.py` (where the shared perturbation infra already lives):

```python
def patch_size_for_n_patches(n_patches: int) -> int:
    """Derive patch pixel size from number of patches.

    Standard ViT patch grids: 196 patches → 16px, 49 patches → 32px.
    """
    return 32 if n_patches == 49 else 16
```

Replace at each site:
- `faithfulness.py:116` → `patch_size = patch_size_for_n_patches(n_patches)` (local call)
- `faithfulness_correlation.py:27` → import and call `patch_size_for_n_patches`
- `pixel_flipping.py:29` → import and call `patch_size_for_n_patches`
- `saco.py:183` — currently reads from `model.cfg.patch_size` with fallback 16. This is actually better (reads from model metadata). Leave as-is, no change.
- `case_studies.py:500` — derives from `patches_per_side`. Leave as-is (different computation path).

Net: 3 call sites consolidated + 1 definition. `saco.py` and `case_studies.py` already derive patch size from model/geometry and don't need the utility.

#### Part 3: Deduplicate faithfulness batch loop

`PatchPixelFlipping.__call__` (pixel_flipping.py:33-50) and `FaithfulnessCorrelation.__call__` (faithfulness_correlation.py:32-49) are **character-for-character identical**:

```python
def __call__(self, model, x_batch, y_batch, a_batch, device=None, batch_size=256):
    x_batch = np.asarray(x_batch)
    y_batch = np.asarray(y_batch)
    a_batch = np.asarray(a_batch)
    scores = []
    for start in range(0, len(x_batch), batch_size):
        end = min(start + batch_size, len(x_batch))
        scores.extend(
            self.evaluate_batch(
                model=model,
                x_batch=x_batch[start:end],
                y_batch=y_batch[start:end],
                a_batch=a_batch[start:end],
                device=device,
            )
        )
    return scores
```

Both classes share the same interface: `__init__` sets `n_patches`, `patch_size`, `perturb_baseline` and class-specific params; `__call__` batches; `evaluate_batch` does the work.

Extract a base class in `experiments/faithfulness.py` (where shared infra lives):

```python
class _BatchedFaithfulnessMetric:
    """Base class for faithfulness metrics with batched evaluation."""

    def __call__(self, model, x_batch, y_batch, a_batch, device=None, batch_size=256):
        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)
        a_batch = np.asarray(a_batch)
        scores = []
        for start in range(0, len(x_batch), batch_size):
            end = min(start + batch_size, len(x_batch))
            scores.extend(
                self.evaluate_batch(
                    model=model,
                    x_batch=x_batch[start:end],
                    y_batch=y_batch[start:end],
                    a_batch=a_batch[start:end],
                    device=device,
                )
            )
        return scores

    def evaluate_batch(self, model, x_batch, y_batch, a_batch, device=None):
        raise NotImplementedError
```

Then:
- `PatchPixelFlipping` inherits `_BatchedFaithfulnessMetric`, keeps `__init__` and `evaluate_batch`, deletes `__call__`
- `FaithfulnessCorrelation` inherits `_BatchedFaithfulnessMetric`, keeps `__init__`, `evaluate_batch`, `_compute_spearman_correlation`, deletes `__call__`

Import changes:
- `pixel_flipping.py` adds: `from gradcamfaith.experiments.faithfulness import _BatchedFaithfulnessMetric`
- `faithfulness_correlation.py` adds: `from gradcamfaith.experiments.faithfulness import _BatchedFaithfulnessMetric`

#### Hard constraints
- No behavior changes. Same metrics, same outputs, same file formats.
- All tests pass (`uv run pytest tests/`).
- Public API signatures unchanged: `run_unified_pipeline`, `run_parameter_sweep`, `run_single_experiment`, etc.
- `PipelineConfig.directories` and `PipelineConfig.directory_map` still work (same return types).

#### Expected outcome
- `config.py`: ~10 fewer lines, no redundant properties, no mutable setter, `analysis` properly declared
- Patch size magic consolidated: 1 definition replaces 3 inline expressions
- Batch loop: 1 base class replaces 2 identical `__call__` methods (~18 lines each)
- Total net: ~-40 lines, significantly less confusion in core path

---

### WP-15 Concrete Plan (Models Cleanup + Sweep Refactor)

Goal: Fix structural issues in `models/` (silent failures, misplaced adapter, duplicate assignment) and improve `sweep.py` readability (split config builder, make vanilla baseline explicit). No behavior changes.

#### Part 1: `models/load.py` cleanup

**1a. Remove duplicate `clip_model_name` assignment**

Lines 41 and 53 both compute `clip_model_name = config.classify.clip_model_name if config else "openai/clip-vit-base-patch32"`. Assign once at line 41, delete the duplicate at line 53.

**1b. Add validation after model load**

After both the CLIP path and the ViT path, assert the returned model has the expected interface attributes (`cfg`, `cfg.n_layers`, `cfg.patch_size`). Raise clear `ValueError` with model type and missing attribute if not.

#### Part 2: `models/clip_classifier.py` — move `CLIPModelWrapper` to experiments

`CLIPModelWrapper` (lines 208-279) is a `torch.nn.Module` adapter that wraps `CLIPClassifier` to look like a standard PyTorch model. It's only imported by `experiments/pipeline.py:143`. It belongs in the experiments layer, not models.

Concrete changes:
- Move `CLIPModelWrapper` class and `create_clip_classifier_for_waterbirds` factory to a new file `experiments/adapters.py`
- Update `experiments/pipeline.py:143` import: `from gradcamfaith.experiments.adapters import CLIPModelWrapper`
- Update `experiments/sweep.py` if it imports `create_clip_classifier_for_waterbirds` (check)
- `models/load.py:51` imports `create_clip_classifier_for_waterbirds` — update to `from gradcamfaith.experiments.adapters import create_clip_classifier_for_waterbirds`

Wait — this creates a **cycle**: `models/load.py` → `experiments/adapters.py` → potentially back to models. Check: does `create_clip_classifier_for_waterbirds` import from models? If so, it cannot move.

**Resolution**: Only move `CLIPModelWrapper` (which has no models/ imports). Keep `create_clip_classifier_for_waterbirds` in `models/clip_classifier.py` (it's a model factory, correctly placed). Move only `CLIPModelWrapper` to `experiments/adapters.py`.

#### Part 3: `models/sae_resources.py` — fail explicitly on missing layers

Current behavior: if a requested layer's SAE file doesn't exist, the function prints a warning and continues, returning a partial dict. Downstream code that expects all layers will crash with a confusing `KeyError`.

Change: after the loading loop, check if any requested layers are missing from `resources`. If so, raise `FileNotFoundError` listing the missing layers and their expected paths.

#### Part 4: `experiments/sweep.py` — split `_build_pipeline_config`

`_build_pipeline_config` (lines 92-126) does three things:
1. Base config construction (PipelineConfig, set dataset, set mode)
2. CLIP configuration (if imagenet)
3. Boosting configuration (layers, gate, clamp_max, etc.)

Split into:
- `_build_pipeline_config` — keeps steps 1+2+3 but calls helpers
- `_configure_clip_for_imagenet(pipeline_config, dataset_name)` — extracted CLIP setup (lines 113-117)
- `_configure_boosting(pipeline_config, experiment_params)` — extracted boosting setup (lines 119-124)

Also make the vanilla baseline explicit in `_build_experiment_grid`:
- Add docstring noting that the first experiment is always the vanilla baseline with `enable_feature_gradients=False`
- Current behavior is correct but undocumented

#### Hard constraints
- No behavior changes. Same metrics, same outputs, same experiment names.
- All tests pass.
- Dependency graph preserved: `core → data → models → experiments` (no new cycles).
- Public API signatures unchanged.

#### Expected outcome
- `models/load.py`: no duplicate assignment, validated model interface
- `CLIPModelWrapper` correctly placed in experiments layer
- `sae_resources.py` fails fast on missing layers (no silent partial returns)
- `sweep.py` config builder decomposed into readable pieces

---

### WP-16 Concrete Plan (Remove Waterbirds Dataset Support)

Goal: Remove all waterbirds-specific code paths, configs, and hardcoded dataset name checks. Waterbirds was never actually used and adds dead code throughout the stack. Rename the waterbirds-named CLIP factory to a generic name since it is used by ImageNet too. No behavior changes for the three supported datasets (imagenet, covidquex, hyperkvasir).

#### Inventory of waterbirds references (5 files, ~100 lines to remove/edit)

| File | What | Action |
|---|---|---|
| `data/dataset_config.py:67-70` | `create_waterbirds_transform()` | Delete |
| `data/dataset_config.py:74` | Comment "same as waterbirds" in `create_imagenet_transform` | Rewrite comment |
| `data/dataset_config.py:149-164` | `WATERBIRDS_CONFIG` constant | Delete |
| `data/dataset_config.py:215` | `"waterbirds": WATERBIRDS_CONFIG` registry entry | Delete |
| `data/prepare.py:4` | Module docstring mentions "Waterbirds" | Remove from list |
| `data/prepare.py:190-246` | `prepare_waterbirds()` function | Delete |
| `data/prepare.py:336` | `'waterbirds': prepare_waterbirds` converter entry | Delete |
| `models/clip_classifier.py:160-205` | `create_clip_classifier_for_waterbirds()` | Rename → `create_clip_classifier`, remove waterbirds default prompts, require `class_names` param |
| `models/load.py:45` | `dataset_config.name == "waterbirds"` hardcode in `use_clip` | Remove — CLIP usage is config-driven via `config.classify.use_clip` |
| `models/load.py:64-66` | Import and call `create_clip_classifier_for_waterbirds` | Update to `create_clip_classifier` |
| `models/sae_resources.py:20` | Docstring mentions "waterbirds" | Remove from example list |
| `models/sae_resources.py:26-27` | `"waterbirds"` in CLIP SAE condition + comment | Remove `"waterbirds"` from list, update comment to say "CLIP" |

#### Part 1: `data/dataset_config.py` — delete waterbirds config

- Delete `create_waterbirds_transform()` (lines 67-70).
- Rewrite `create_imagenet_transform` docstring: remove "(same as waterbirds)" → "CLIP preprocessing".
- Delete `WATERBIRDS_CONFIG` block (lines 149-164).
- Delete `"waterbirds": WATERBIRDS_CONFIG` from `DATASET_CONFIGS` registry.

#### Part 2: `data/prepare.py` — delete waterbirds converter

- Update module docstring: remove "Waterbirds" from the dataset list.
- Delete `prepare_waterbirds()` function (lines 190-246).
- Delete `'waterbirds': prepare_waterbirds` from `convert_dataset` converter map.

#### Part 3: `models/clip_classifier.py` — rename factory to generic

- Rename `create_clip_classifier_for_waterbirds` → `create_clip_classifier`.
- Change parameter `custom_prompts: Optional[List[str]] = None` → `class_names: List[str]` (required, no default).
- Remove the waterbirds default fallback (`if custom_prompts is None: class_names = ["landbird", "waterbird"]`).
- Update docstring: "Create a CLIP classifier" (remove "specifically for Waterbirds dataset").
- Remove "Default prompts for waterbirds" comment.

#### Part 4: `models/load.py` — remove hardcoded waterbirds check

- Line 45: change `use_clip = (config and config.classify.use_clip) or dataset_config.name == "waterbirds"` → `use_clip = config and config.classify.use_clip`.
- Line 64: update import `create_clip_classifier_for_waterbirds` → `create_clip_classifier`.
- Line 66-70: update call to `create_clip_classifier(...)` with `class_names=config.classify.clip_text_prompts` (prompts are now required, always provided from config).

#### Part 5: `models/sae_resources.py` — remove waterbirds from SAE path

- Line 20: remove "waterbirds" from docstring example list.
- Line 26: change `dataset_name in ["waterbirds", "imagenet"]` → `dataset_name == "imagenet"`.
- Line 27: update comment from "Use CLIP Vanilla B-32 SAE for waterbirds" → "Use CLIP Vanilla B-32 SAE".

#### Hard constraints
- No behavior changes for imagenet, covidquex, hyperkvasir.
- All tests pass.
- Dependency graph preserved.
- Zero "waterbird" references in `src/` after completion.

#### Expected outcome
- 3 supported datasets: imagenet, covidquex, hyperkvasir.
- CLIP classifier factory is generic (no dataset-specific name or defaults).
- CLIP usage is purely config-driven (no hardcoded dataset name checks).
- ~100 lines of dead code removed.

---

### WP-17 Concrete Plan (Analysis Cleanup: comparison.py + case_studies.py)

Goal: Remove dead code, fix a latent WP-16 bug, separate SAE extraction from analysis, rename conflated functions, compress repetitive formatting, and rewrite case_studies.py internals for clarity. No behavior changes to analysis outputs.

Current state: `comparison.py` (583 lines, 14 functions), `case_studies.py` (1294 lines, 15 functions). Total: 1877 lines.

#### Audit findings

**comparison.py**:
- `cohens_d()` (line 21) is **dead** — never called. The effect size is computed inline in `calculate_statistical_comparison` from summary stats.
- `print_detailed_results`, `identify_best_performers`, `identify_best_overall_performers` share ~80% of their metric-formatting logic (~170 lines of near-duplicate print code).
- `save_results` hardcodes output filenames to CWD.
- Otherwise clean: stats engine is correct, `main()` is a clean orchestrator.

**case_studies.py**:
- **Two unrelated responsibilities**: SAE activation extraction (lines 23-244, GPU inference) + case study analysis (lines 247-1294, post-hoc analysis & visualization).
- **Latent WP-16 bug**: `_extract_sae_activations` line 88-90 sets `use_clip=True` but never sets `clip_text_prompts`. After WP-16, `create_clip_classifier` requires `class_names: List[str]` — this will crash at runtime when `None` is passed. Additionally, the CLIP classifier is created but never used — only the vision model is needed for hook-based residual capture.
- **Dead function**: `visualize_case_study()` (lines 661-847, 135 lines) creates a single multi-panel figure but is never called by `run_case_study_analysis`. Replaced by `save_case_study_individual_images`.
- **Hardcoded dataset**: `run_case_study_analysis` hardcodes `dataset = 'imagenet'` (line 1091) instead of accepting it as parameter. The `__main__` block extracts activations for `covidquex` but the analysis forces `imagenet` — a mismatch.
- **Name collision**: `compute_composite_improvement` (z-score normalization, per-image) vs comparison.py's `calculate_composite_improvement` (simple average of percent improvements, from summary stats). Same concept, different semantics, confusing names.

#### Part 1: Delete dead code

- Delete `cohens_d()` from comparison.py (never called).
- Delete `visualize_case_study()` from case_studies.py (replaced by `save_case_study_individual_images`, not called by orchestrator). ~135 lines removed.

#### Part 2: Fix latent WP-16 bug in SAE extraction

`_extract_sae_activations` creates a CLIP config without setting `clip_text_prompts`:
```python
temp_config.classify.use_clip = True
temp_config.classify.clip_model_name = "open-clip:..."
# clip_text_prompts left as None → crash in create_clip_classifier
```

Fix: set `clip_text_prompts` from dataset config class names (matching sweep.py pattern):
```python
if use_clip:
    temp_config.classify.use_clip = True
    temp_config.classify.clip_model_name = "open-clip:laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K"
    dataset_cfg = get_dataset_config(dataset_name)
    temp_config.classify.clip_text_prompts = [f"a photo of a {cls}" for cls in dataset_cfg.class_names]
```

Note: the extraction only needs the vision model (for hook-based residual capture). The CLIP classifier (text encoder + text embeddings) is created but never used. This is wasteful but not broken. A future optimization could add a model-only loading path, but that's out of scope for this WP.

#### Part 3: Move SAE extraction to models/

Move `extract_sae_activations_if_needed` + `_extract_sae_activations` → new `models/sae_extraction.py`.

Rationale:
- This is GPU inference code: loads models via `load_model_for_dataset`, loads SAEs via `load_steering_resources`, runs forward passes with hooks, handles checkpointing.
- It belongs in the models layer, not in an analysis file.
- case_studies.py will import from `gradcamfaith.models.sae_extraction`.
- ~220 lines moved out of case_studies.py.

#### Part 4: Rename conflated functions

- comparison.py `calculate_composite_improvement(row, metrics)` → `_average_percent_improvement(row, metrics)` — make private (only used by `identify_best_overall_performers`), name reflects what it does (averages percent improvements from summary stats).
- case_studies.py `compute_composite_improvement(vanilla_df, gated_df)` → `compute_per_image_improvement(vanilla_df, gated_df)` — name reflects that it works per-image with z-score normalization, distinct from the summary-stat version.

#### Part 5: Parameterize hardcoded values

- `run_case_study_analysis`: add `dataset: str` parameter instead of hardcoding `dataset = 'imagenet'` (line 1091). Propagate to path construction.
- comparison.py `save_results`: accept `output_dir: Path` parameter instead of hardcoding CWD filenames.

#### Part 6: Compress comparison.py print formatting

Extract shared `_format_metric_comparison(row, metric)` helper returning a formatted string block. Deduplicate the metric-formatting loops in `print_detailed_results`, `identify_best_performers`, `identify_best_overall_performers`.

Lower priority — these are research output functions that may need per-use tweaking. Only deduplicate the clearly shared core (metric name, treatment/vanilla values, percent improvement, significance stars).

#### Part 7: Rewrite case_studies.py internals for clarity

After Parts 1-5, case_studies.py still has ~950 lines of tangled analysis+visualization code. Rewrite the internal structure:

**7a. Extract `_patch_grid(n_patches)` helper**

Patch grid geometry (`n_patches`, `patches_per_side`, `patch_size`) is recomputed from debug data in 4 separate places: `find_dominant_features_in_image`, (deleted) `visualize_case_study`, `save_case_study_individual_images`, and inline in various loops. Extract once:

```python
def _patch_grid(n_patches: int) -> Tuple[int, int]:
    """Derive patch grid geometry. Returns (patches_per_side, patch_size)."""
    side = int(np.sqrt(n_patches))
    return side, 224 // side
```

**7b. Extract `_save_attribution_overlay(img, attr, path, patch_highlight=None, title=None)`**

The pattern "normalize attribution → imshow image → imshow heatmap overlay → optional rectangle → save" appears 3 times in `save_case_study_individual_images` (vanilla, gated, each prototype). Extract to a single helper. This replaces ~50 lines of repeated plt code with 3 one-liner calls.

**7c. Unify boost/suppress logic in `find_dominant_features_in_image`**

Currently duplicated with inverted sign (lines 541-556):
```python
if final_delta > 0:
    pos_contribs = [(i, c) for i, c in enumerate(patch_contributions) if c > 0]
    ...
    best_idx, best_contrib = max(pos_contribs, key=lambda x: x[1])
    role = "BOOST"
else:
    neg_contribs = [(i, c) for i, c in enumerate(patch_contributions) if c < 0]
    ...
    best_idx, best_contrib = min(neg_contribs, key=lambda x: x[1])
    role = "SUPPRESS"
```

Unify to:
```python
matching = [(i, c) for i, c in enumerate(contribs) if np.sign(c) == np.sign(final_delta)]
if not matching:
    skipped_reasons['no_matching_direction'] += 1
    continue
best_idx, best_contrib = max(matching, key=lambda x: abs(x[1]))
role = "BOOST" if final_delta > 0 else "SUPPRESS"
```

**7d. Extract `_resolve_prototype_path(debug_idx, debug_to_image_idx, path_mapping, image_dir)` helper**

The dual image path resolution (simple `get_image_path` vs `prototype_path_mapping` dict) is inlined in both `save_case_study_individual_images` and the (deleted) `visualize_case_study`. Extract to one helper that encapsulates the branching.

**7e. Untangle `run_case_study_analysis` prototype source setup**

Lines 1107-1178 build prototype source configuration with deeply nested if/else, metadata file loading, and dict construction. Extract to:
```python
def _load_prototype_source(validation_activations_path, gated_faithfulness, val_image_dir):
    """Load and configure prototype image source (validation or test set).
    Returns (debug_to_image_idx, image_dir, path_mapping).
    """
```

This reduces the main orchestrator to a clean sequence: load results → setup prototype source → per-layer loop → save combined.

**7f. Replace inline z-score lambda**

```python
z_score = lambda series: (series - series.mean()) / (series.std() + 1e-9)
```

Replace with a named `_zscore(series)` function for readability.

#### Hard constraints
- No behavior changes to analysis outputs (same stats, same visualizations, same CSVs).
- All tests pass.
- Dependency graph preserved: `core → data → models → experiments`.
- `models/sae_extraction.py` imports only from models/ and below (no cycle risk).

#### Expected outcome
- comparison.py: ~583 → ~450 lines (dead code removed, print helpers compressed).
- case_studies.py: ~1294 → ~750 lines (SAE extraction out, dead viz out, rewritten internals).
- New `models/sae_extraction.py`: ~230 lines (extracted from case_studies).
- Total: ~1877 → ~1430 lines.
- Latent CLIP bug fixed.
- Clear separation: models/ handles extraction, experiments/ handles analysis.
- case_studies.py internals are readable: single-responsibility helpers, no duplicated geometry/visualization/path logic.

---

### WP-18 Concrete Plan (Experiment Output Consolidation + Debug Data Cleanup)

Goal: Unify all three faithfulness metrics (SaCo, FaithfulnessCorrelation, PixelFlipping) into two canonical output files, and trim debug NPZ to only what downstream analysis actually uses. Fix the data format mismatch that prevents `case_studies.py` from running on current experiment outputs.

#### Motivation

Current experiment output has three problems:
1. **Fragmented metrics**: `results.json` only has SaCo summary; `faithfulness_stats_*.json` only has FC + PF detail. Neither file gives a complete picture.
2. **Missing data**: `case_studies.py` expects `analysis_faithfulness_correctness_*.csv` (per-image SaCo + classification metadata) — but no code ever writes this file. The data IS computed in `saco.py:263` (`faithfulness_correctness` DataFrame) but discarded by the pipeline.
3. **Debug NPZ mismatch**: `_accumulate_debug_info` saves 4 fields (`gate_values`, `contribution_sum`, `total_contribution_magnitude`, `patch_attribution_deltas`). `case_studies.py` needs 4 different fields (`sparse_indices`, `sparse_activations`, `sparse_contributions`, `patch_attribution_deltas`). Only `patch_attribution_deltas` overlaps. The sparse data IS generated by `gating.py:96-114` but dropped by the pipeline.

#### Output file specification (after WP-18)

**File 1: `results.json`** — experiment overview with all 3 metrics in uniform format.

```json
{
  "dataset": "imagenet",
  "timestamp": "2026-02-09T18:01:44",
  "experiment_params": { ... },
  "subset_size": 10000,
  "random_seed": 123,
  "status": "success",
  "n_images": 10000,
  "metrics": {
    "SaCo": {
      "mean": 0.3036, "std": 0.3980, "n_samples": 10000,
      "per_class": { ... }, "by_correctness": { ... }
    },
    "FaithfulnessCorrelation": {
      "mean": 0.2743, "std": 0.2492, "n_samples": 10000
    },
    "PixelFlipping": {
      "mean": 0.0523, "std": 0.0312, "n_samples": 10000
    }
  }
}
```

The old `saco_results` key is replaced by `metrics` containing all 3. Each metric has at minimum `mean`, `std`, `n_samples`. SaCo additionally has `per_class` and `by_correctness` (from `extract_saco_summary`). FC and PF additionally have `median`, `min`, `max` (from `_compute_statistics_from_scores`).

**File 2: `test/faithfulness_stats_<timestamp>.json`** — detailed per-image scores for all 3 metrics + per-image classification metadata.

```json
{
  "dataset": "imagenet",
  "metrics": {
    "SaCo": {
      "overall": { "count": 10000, "mean": 0.3036, ... },
      "mean_scores": [0.45, 0.12, ...],
      "n_trials": 1
    },
    "FaithfulnessCorrelation": {
      "overall": { "count": 10000, "mean": 0.2743, ... },
      "by_class": { ... },
      "mean_scores": [0.31, 0.28, ...],
      "std_scores": [0.05, 0.03, ...],
      "n_trials": 3
    },
    "PixelFlipping": {
      "overall": { "count": 10000, "mean": 0.0523, ... },
      "by_class": { ... },
      "mean_scores": [0.06, 0.04, ...],
      "std_scores": [0.0, 0.0, ...],
      "n_trials": 1
    }
  },
  "images": [
    {
      "filename": "path/to/img_000015.jpeg",
      "predicted_class": "tench",
      "predicted_idx": 0,
      "true_class": "tench",
      "is_correct": true,
      "confidence": 0.95
    }
  ]
}
```

The existing `class_labels` array is replaced by the richer `images` array (from `saco.py`'s `_join_saco_with_correctness` data). SaCo per-image scores are added as `metrics.SaCo.mean_scores`. The `images` array has one entry per image, in processing order (same indexing as `mean_scores` arrays).

**File 3: `test/debug/layer_<N>_debug.npz`** (only when `debug_mode=True`) — sparse feature data per layer.

Keys (all kept):
- `patch_attribution_deltas` — `np.ndarray`, shape `[n_images, n_patches]`. Per-patch signed attribution change from gating. Used by `case_studies.py:find_dominant_features_in_image` and for `_patch_grid` geometry derivation.
- `sparse_indices` — `np.ndarray` of object arrays, shape `[n_images][n_patches][variable]`. Per-patch active SAE feature indices. Used by `case_studies.py:build_feature_activation_index` and `find_dominant_features_in_image`.
- `sparse_activations` — same shape. Per-patch SAE feature activations. Used by `case_studies.py:build_feature_activation_index`.
- `sparse_contributions` — same shape. Per-patch SAE feature contributions (activation × gradient). Used by `case_studies.py:find_dominant_features_in_image`.

Keys removed (not used by any downstream consumer):
- `gate_values` — per-image gate multiplier arrays. Not read by case_studies.py or comparison.py.
- `contribution_sum` — per-image aggregate scores (s_t). Not read by any consumer.
- `total_contribution_magnitude` — per-image |contribution| sums. Not read by any consumer.
- `sparse_gradients` — per-patch feature gradients. Loaded by `load_debug_data` but never accessed by any analysis function.

#### Debug NPZ audit (complete)

What `gating.py:_collect_gate_debug_info` generates:

| Key in gating.py | Type | Used by case_studies.py? | Keep in NPZ? |
|---|---|---|---|
| `gate_values` | `np.ndarray [n_patches]` | No | **Remove** |
| `contribution_sum` | `np.ndarray [n_patches]` | No | **Remove** |
| `total_contribution_magnitude` | `np.ndarray [n_patches]` | No | **Remove** |
| `mean_gate` | `float` | No | Already not collected |
| `std_gate` | `float` | No | Already not collected |
| `sparse_features_indices` | `list[np.ndarray]` per patch | Yes (`build_feature_activation_index`, `find_dominant_features_in_image`) | **Add** as `sparse_indices` |
| `sparse_features_activations` | `list[np.ndarray]` per patch | Yes (`build_feature_activation_index`) | **Add** as `sparse_activations` |
| `sparse_features_gradients` | `list[np.ndarray]` per patch | No (loaded but never accessed) | **Skip** |
| `sparse_features_contributions` | `list[np.ndarray]` per patch | Yes (`find_dominant_features_in_image`) | **Add** as `sparse_contributions` |

What `apply_feature_gradient_gating` generates (outer debug_info):

| Key | Used by case_studies.py? | Keep? |
|---|---|---|
| `patch_attribution_deltas` | Yes | **Keep** (already collected) |
| `combined_gate` | No | Already not collected |
| `cam_delta` | No | Already not collected |

comparison.py does NOT use debug NPZ at all. It reads only `results.json` and `faithfulness_stats_*.json`.

#### Changes by file

**1. `experiments/pipeline.py`** — the central coordinator

Current flow:
```
classify images → [optionally accumulate debug] → evaluate_and_report_faithfulness → run_binned_attribution_analysis → extract_saco_summary → return (results, saco_results)
```

New flow:
```
classify images → [optionally accumulate debug] → compute faithfulness (no save) → compute SaCo → build unified stats → save unified faithfulness_stats → return (results, unified_metrics)
```

Concrete changes:

a) **Replace `evaluate_and_report_faithfulness` call** with `evaluate_faithfulness_for_results` + `_build_results_structure` (both already exist as lower-level functions in faithfulness.py). This skips the save that `evaluate_and_report_faithfulness` does. Import the lower-level functions instead.

b) **After SaCo, inject SaCo into the faithfulness results dict**:
```python
# Build SaCo stats matching faithfulness format
fc_df = saco_analysis["faithfulness_correctness"]
saco_scores = fc_df["saco_score"].values
faithfulness_data['metrics']['SaCo'] = {
    'overall': {
        'count': len(saco_scores),
        'mean': float(np.nanmean(saco_scores)),
        'std': float(np.nanstd(saco_scores)),
    },
    'mean_scores': saco_scores.tolist(),
    'n_trials': 1,
}
```

c) **Add per-image classification metadata** to the faithfulness stats:
```python
faithfulness_data['images'] = fc_df[['filename', 'predicted_class', 'predicted_idx', 'true_class', 'is_correct', 'confidence']].to_dict('records')
```

d) **Save the unified faithfulness stats** using a new `_save_unified_faithfulness_stats` helper (replaces the save that was in `evaluate_and_report_faithfulness`).

e) **Return unified metric summary** instead of just `saco_results`. Build from faithfulness data:
```python
unified_metrics = {
    'SaCo': extract_saco_summary(saco_analysis),
    'FaithfulnessCorrelation': {
        'mean': fc_data['overall']['mean'],
        'std': fc_data['overall']['std'],
        'n_samples': fc_data['overall']['count'],
    },
    'PixelFlipping': { ... same ... },
}
```

f) **Rework `_accumulate_debug_info`** — collect sparse feature data, drop unused fields:
```python
def _accumulate_debug_info(debug_data_per_layer, debug_info):
    for layer_idx, layer_debug in debug_info.items():
        feature_gating = layer_debug.get('feature_gating', {})
        if layer_idx not in debug_data_per_layer:
            debug_data_per_layer[layer_idx] = {
                'patch_attribution_deltas': [],
                'sparse_indices': [],
                'sparse_activations': [],
                'sparse_contributions': [],
            }
        buf = debug_data_per_layer[layer_idx]
        buf['patch_attribution_deltas'].append(
            layer_debug.get('patch_attribution_deltas', np.array([])))
        buf['sparse_indices'].append(
            feature_gating.get('sparse_features_indices', []))
        buf['sparse_activations'].append(
            feature_gating.get('sparse_features_activations', []))
        buf['sparse_contributions'].append(
            feature_gating.get('sparse_features_contributions', []))
```

g) **Rework `_save_debug_outputs`** — save new keys, use `allow_pickle=True` for ragged sparse arrays:
```python
np.savez_compressed(
    debug_dir / f"layer_{layer_idx}_debug.npz",
    patch_attribution_deltas=np.array(layer_data['patch_attribution_deltas']),
    sparse_indices=np.array(layer_data['sparse_indices'], dtype=object),
    sparse_activations=np.array(layer_data['sparse_activations'], dtype=object),
    sparse_contributions=np.array(layer_data['sparse_contributions'], dtype=object),
)
```

**2. `experiments/faithfulness.py`** — expose lower-level compute, remove standalone save

Current `evaluate_and_report_faithfulness`:
```python
def evaluate_and_report_faithfulness(config, model, device, classification_results):
    faithfulness_results, class_labels = evaluate_faithfulness_for_results(...)
    results = _build_results_structure(config, faithfulness_results, class_labels)
    _print_faithfulness_summary(results['metrics'])
    _save_faithfulness_results(config, results)
    return results
```

Change: make `_build_results_structure` and `_print_faithfulness_summary` importable (remove leading underscore, or keep private and add a new public function).

Simplest approach: add a `compute_faithfulness` function that does everything EXCEPT save:
```python
def compute_faithfulness(config, model, device, classification_results):
    """Compute faithfulness metrics. Returns results dict (does not save)."""
    faithfulness_results, class_labels = evaluate_faithfulness_for_results(
        config, model, device, classification_results,
    )
    results = _build_results_structure(config, faithfulness_results, class_labels)
    _print_faithfulness_summary(results['metrics'])
    return results
```

Keep `evaluate_and_report_faithfulness` as a thin wrapper (backward compat for any direct callers):
```python
def evaluate_and_report_faithfulness(config, model, device, classification_results):
    """Compute faithfulness metrics and save to disk."""
    results = compute_faithfulness(config, model, device, classification_results)
    _save_faithfulness_results(config, results)
    return results
```

Pipeline.py switches to calling `compute_faithfulness` instead.

**3. `experiments/sweep.py`** — unified `results.json`

`run_single_experiment` currently writes:
```python
config_dict['saco_results'] = saco_results
```

Change to write the unified metrics dict returned by pipeline:
```python
results, unified_metrics = run_unified_pipeline(...)
config_dict['metrics'] = unified_metrics
```

The `saco_results` key is replaced by `metrics` containing all 3. The rest of `results.json` (dataset, timestamp, experiment_params, status, n_images) stays unchanged.

**4. `experiments/case_studies.py`** — update loaders for new format

a) **`load_faithfulness_results`**: Rewrite to read from `faithfulness_stats_*.json` instead of the (never-existing) CSV:
```python
def load_faithfulness_results(path: Path) -> pd.DataFrame:
    """Load per-image faithfulness results from experiment directory."""
    faithfulness_json = list(path.glob("faithfulness_stats_*.json"))
    if not faithfulness_json:
        raise FileNotFoundError(f"No faithfulness stats JSON found in {path}")

    with open(faithfulness_json[0], 'r') as f:
        stats = json.load(f)

    # Build DataFrame from images metadata
    images = stats.get('images', [])
    df = pd.DataFrame(images)

    # Add per-image metric scores
    for metric_name, metric_data in stats.get('metrics', {}).items():
        if 'mean_scores' in metric_data:
            scores = metric_data['mean_scores']
            if metric_name == 'SaCo':
                df['saco_score'] = scores
            else:
                df[metric_name] = scores

    # Extract image index from filename
    if 'filename' in df.columns:
        df['image_idx'] = df['filename'].str.extract(r'_(\d+)\.(?:jpeg|png)$')[0].astype(int)
    else:
        df['image_idx'] = range(len(df))

    return df
```

b) **`load_debug_data`**: Update to match new NPZ keys (drop `sparse_gradients` and `gate_values`):
```python
debug_data[layer_idx] = {
    'sparse_indices': data['sparse_indices'],
    'sparse_activations': data['sparse_activations'],
    'sparse_contributions': data['sparse_contributions'],
    'patch_attribution_deltas': data['patch_attribution_deltas'],
}
```

**5. `experiments/comparison.py`** — update `extract_metrics` for new `results.json` format

`extract_metrics` currently reads SaCo from `results.json:saco_results` and FC/PF from `faithfulness_stats_*.json:metrics`. Update to read all 3 from `results.json:metrics`:

```python
def extract_metrics(data: Dict[str, Any]) -> Dict[str, float]:
    metrics = {}
    results_metrics = data['results'].get('metrics', {})

    # SaCo (from unified metrics)
    saco = results_metrics.get('SaCo', data['results'].get('saco_results', {}))
    metrics['saco_mean'] = saco['mean']
    metrics['saco_std'] = saco['std']
    metrics['saco_n'] = saco['n_samples']

    # FC and PF (from unified metrics, with fallback to faithfulness_stats)
    for metric_key, prefix in [('FaithfulnessCorrelation', 'faithfulness_correlation'),
                                ('PixelFlipping', 'pixelflipping')]:
        source = results_metrics.get(metric_key, {})
        if not source:
            # Fallback: read from faithfulness_stats for old-format experiments
            source = data['faithfulness']['metrics'][metric_key]['overall']
        metrics[f'{prefix}_mean'] = source['mean']
        metrics[f'{prefix}_std'] = source['std']
        metrics[f'{prefix}_n'] = source.get('n_samples', source.get('count', 0))

    return metrics
```

No backward compatibility with old `saco_results` format — old experiment data must be re-run.

#### Import changes

| File | Add | Remove |
|---|---|---|
| `pipeline.py` | `from gradcamfaith.experiments.faithfulness import compute_faithfulness` | `from gradcamfaith.experiments.faithfulness import evaluate_and_report_faithfulness` |
| `faithfulness.py` | (expose `compute_faithfulness`) | (keep `evaluate_and_report_faithfulness` as wrapper) |
| `sweep.py` | — | — |
| `case_studies.py` | — | — |
| `comparison.py` | — | — |

#### Hard constraints
- No metric algorithm changes. Same SaCo, FC, PF computations.
- All tests pass (`uv run pytest tests/`).
- Dependency graph preserved: `core → data → models → experiments`.
- Debug NPZ is only saved when `debug_mode=True` (no change to gating).
- No backward compat with old `saco_results` format — old experiments must be re-run.

#### Validation
- `uv run pytest tests/` — all pass.
- Manual: run sweep with `debug_mode=True`, verify:
  - `results.json` has `metrics` key with all 3 metrics.
  - `faithfulness_stats_*.json` has `metrics.SaCo.mean_scores`, `images` array.
  - `debug/layer_*_debug.npz` has keys `patch_attribution_deltas`, `sparse_indices`, `sparse_activations`, `sparse_contributions` — and does NOT have `gate_values`, `contribution_sum`, `total_contribution_magnitude`.
- Manual: run `case_studies.py` against the new output — no `FileNotFoundError`.

#### Expected outcome
- Two canonical output files give complete picture: `results.json` (overview), `faithfulness_stats_*.json` (detail).
- `case_studies.py` works against current experiment data format.
- Debug NPZ contains only what case_studies.py actually reads (~3× less data per image).
- `comparison.py` works with both old and new `results.json` format.

---

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
- **WP-16 (remove waterbirds dataset support)**: Deleted `WATERBIRDS_CONFIG`, `create_waterbirds_transform`, `prepare_waterbirds` function, and all registry/converter entries. Renamed `create_clip_classifier_for_waterbirds` → `create_clip_classifier` with required `class_names` parameter (no waterbirds default fallback). Removed hardcoded `dataset_config.name == "waterbirds"` CLIP check in `models/load.py` — CLIP usage now purely config-driven via `config.classify.use_clip`. Removed `"waterbirds"` from SAE path condition in `sae_resources.py`. Zero waterbird references remain in `src/`. 3 supported datasets: imagenet, covidquex, hyperkvasir. All 14 tests pass.
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
