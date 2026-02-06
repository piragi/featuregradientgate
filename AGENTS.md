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
- `src/gradcamfaith/` currently contains only cache artifacts and no tracked source files.
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
- Last successful commit reflected here: `none yet (tracking starts with rework commits on program branch)`
- What happened most recently: `added config-first research interface policy for concurrent paper release workflow`
- What should happen next: `assign WP-01 to a coder and start package bootstrap on src/gradcamfaith/core`
- Immediate next task (concrete): `WP-01: move transmm/feature_gradient_gating + config/data_types into src/gradcamfaith/core with root-level compatibility wrappers`
- Immediate validation for that task: `uv run python -c "import transmm, feature_gradient_gating, config, data_types"`
- Known blockers/risks now: `large monolithic files and mixed responsibilities increase migration risk if moved in big batches`
- Decision log pointer: `all accepted structural decisions must be appended in this section`

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
6. Compatibility cleanup: keep shims until new paths are stable, then prune gradually.

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
- Out of scope: changing metrics definitions or experiment semantics.
- Depends on: WP-01.
- Acceptance checks: `uv run python -c "from pipeline import run_unified_pipeline"` remains valid.
- Acceptance checks: one small subset run succeeds with existing command path.
- Deliverables: decomposed modules and compatibility imports with updated docs.

### WP-03 Data Setup Split
- Goal: split downloading and conversion concerns now mixed in `setup.py`.
- In scope: create `data/download.py` and `data/prepare.py`; keep root `setup.py` as compatibility CLI.
- In scope: retain current dataset support (`imagenet`, `hyperkvasir`, `covidquex`, `waterbirds`).
- Out of scope: changing dataset content logic beyond structural extraction.
- Depends on: WP-01.
- Acceptance checks: `uv run setup.py` still resolves and starts the same top-level flow.
- Deliverables: separated modules, unchanged CLI behavior, AGENTS update noting boundaries.

### WP-04 Experiments Migration
- Goal: move experiment drivers into `src/gradcamfaith/experiments`.
- In scope: migrate `main.py`, `sae.py`, `comparsion.py`, `analysis_feature_case_studies.py`.
- In scope: fix typo path by introducing `comparison.py` while preserving `comparsion.py` compatibility.
- In scope: preserve typed in-file config patterns where they accelerate research iteration.
- Out of scope: changing experiment objective functions or metrics interpretation.
- Depends on: WP-01 and WP-02.
- Acceptance checks: root entry scripts still callable via compatibility wrappers.
- Deliverables: package experiment modules + stable root shims.

### WP-05 Example Path
- Goal: add one clear newcomer path for understanding and running the method.
- In scope: `src/gradcamfaith/examples/minimal_run.py` and `src/gradcamfaith/examples/README.md`.
- In scope: include exact `uv` commands, expected output artifacts, and an example config-first run.
- Out of scope: full benchmark scripts.
- Depends on: WP-02 minimum.
- Acceptance checks: example imports and runs on a tiny subset with documented command.
- Deliverables: runnable example and documentation.

### WP-06 Compatibility and Cleanup
- Goal: tighten structure after migrations while preserving legacy branch compatibility promises.
- In scope: audit wrappers, imports, and deprecated paths; remove only what is explicitly approved.
- In scope: update `README.md` and `AGENTS.md` to final structure and command paths.
- Out of scope: dropping legacy wrappers without maintainer sign-off.
- Depends on: WP-01 through WP-05.
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
1. Assign `WP-01` to one coder with branch name `wp/WP-01-core-package-bootstrap`.
2. Require one commit for the workpackage and include validation output summary in the PR description.
3. Review against `Workpackage Review Checklist`, then update `Feature Tracker (Living)` with accepted result and next assignment.

## Done Criteria for This Rework
- Core method code is isolated from experiment orchestration.
- Experiment scripts are grouped and discoverable.
- A newcomer can run one example and one experiment with documented `uv` commands.
- `AGENTS.md` reflects current structure and next planned steps at all times.
