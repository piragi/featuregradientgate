# WP-06A Core Responsibility Map and Equivalence Contract

## Document Purpose
This document is the required audit artifact for `WP-06A`.
It defines:
- the current responsibility map of `core/attribution.py` and `core/gating.py`
- overlap findings and readability risks
- the no-logic-change equivalence contract for follow-up slices (`WP-06B` and `WP-06C`)

This document must be updated in each accepted WP-06 slice.

## Metadata

| Field | Value |
|---|---|
| Branch | `wp/WP-06-clarity-cleanup` |
| Slice | `WP-06A` |
| Commit SHA | `8f4ae89` (audit), `01925e0` (kappa/clamp_max follow-up) |
| Author | coder |
| Reviewer | (maintainer) |
| Date (UTC) | 2026-02-07 |

## Scope
### In Scope
- `src/gradcamfaith/core/attribution.py`
- `src/gradcamfaith/core/gating.py`

### Out of Scope
- algorithmic changes
- metric definition changes
- output path or artifact schema changes
- CLI surface changes

## Public Interface Contract
The following public functions are compatibility-critical and must remain import-stable unless explicitly approved:

### `src/gradcamfaith/core/attribution.py`
- `transmm_prisma_enhanced`
- `generate_attribution_prisma_enhanced`

### `src/gradcamfaith/core/gating.py`
- `compute_feature_gradient_gate`
- `apply_feature_gradient_gating`

## Current Function Inventory

### Attribution Module Inventory
| Function | Primary Responsibility Tag | Secondary Responsibility Tags | Inputs Used? | Overlap Candidate | Notes |
|---|---|---|---|---|---|
| `apply_gradient_gating_to_cam` | `ADAPTER` | `I/O_PACKING` | `device` unused (see ledger) | yes — paired with `apply_feature_gradient_gating` | Extracts per-layer hook data from dicts, unpacks `PipelineConfig` into flat config dict, delegates to `gating.apply_feature_gradient_gating`. Adapter role is valid but undocumented. |
| `compute_layer_attribution` | `ORCHESTRATOR` | — | all used | no | Iterates all transformer layers, computes gradient-weighted attention CAMs via `avg_heads`, conditionally applies gating at configured layers, accumulates self-attention rollout matrix. |
| `avg_heads` | `PURE_MATH` | — | all used | no | `grad * cam`, clamp positive, mean across heads. 5 lines, clean. |
| `apply_self_attention_rules` | `PURE_MATH` | — | all used | no | Single `torch.matmul` for self-attention propagation rule. 2 lines, clean. |
| `run_model_forward_backward` | `I/O_PACKING` | `HOOK_MANAGEMENT` | all used | no | Runs forward pass (direct or via clip_classifier), constructs one-hot target, runs backward, returns prediction dict. |
| `setup_hooks` | `HOOK_MANAGEMENT` | — | all used | no | Creates hook functions and hook name lists for vit_prisma hook API. Returns storage dicts for activations, gradients, residuals. |
| `transmm_prisma_enhanced` | `ENTRYPOINT` | `ORCHESTRATOR`, `NORMALIZATION` | all used | yes — paired with `generate_attribution_prisma_enhanced` | Main entrypoint: hooks -> forward/backward -> layer attribution -> numpy reshape/interpolate/normalize. Returns 4-tuple `(prediction_dict, attribution_map, raw_patches, debug_info)`. Post-processing (reshape/interpolate/normalize, L236-248) is output packaging mixed into orchestration. |
| `generate_attribution_prisma_enhanced` | `ADAPTER` | `I/O_PACKING` | all used | yes — wraps `transmm_prisma_enhanced` | Thin wrapper: reads `config.classify.boosting.debug_mode`, calls `transmm_prisma_enhanced`, repackages 4-tuple into a flat dict with keys `predictions`, `attribution_positive`, `raw_attribution`, `debug_info`. Only external caller is `pipeline.py`. |

### Gating Module Inventory
| Function | Primary Responsibility Tag | Secondary Responsibility Tags | Inputs Used? | Overlap Candidate | Notes |
|---|---|---|---|---|---|
| `compute_feature_gradient_gate` | `PURE_MATH` | `DEBUG_PACKING`, `NORMALIZATION` | all used (`kappa` removed, `clamp_max` wired in) | no | Implements full gate formula: decoder projection -> contribution scoring (4 gate_construction modes) -> MAD normalization -> `clamp_max ** tanh(s_norm)` mapping. Debug block is ~40 lines (~30% of function) with per-patch loop. |
| `apply_feature_gradient_gating` | `ADAPTER` | `I/O_PACKING`, `DEBUG_PACKING` | all used | no — clean boundary with `compute_feature_gradient_gate` | Entry point: unpacks config dict -> resolves SAE decoder matrix -> delegates to `compute_feature_gradient_gate` -> applies gate to CAM tensor (CLS-aware) -> computes attribution deltas -> compiles debug info dict. |

## Explicit Overlap Analysis

### `transmm_prisma_enhanced` vs `generate_attribution_prisma_enhanced`
1. Responsibility of `transmm_prisma_enhanced`:
Full orchestration of the TransLRP attribution pipeline: input preparation, hook wiring, model forward/backward, layer-by-layer attribution computation (including optional feature gradient gating), and output post-processing (numpy conversion, spatial interpolation, min-max normalization). Returns a 4-tuple: `(prediction_dict, attribution_map_2d, raw_patch_map_1d, debug_info_per_layer)`.

2. Responsibility of `generate_attribution_prisma_enhanced`:
Thin adapter that reads a single config field (`config.classify.boosting.debug_mode`), delegates to `transmm_prisma_enhanced`, and repackages the 4-tuple output into a flat dict with named keys (`predictions`, `attribution_positive`, `raw_attribution`, `debug_info`). Conditionally strips debug info when `debug=False`.

3. Current overlap and why it increases cognitive load:
The overlap is minimal and intentional (adapter pattern). However, the cognitive load comes from **undocumented role differentiation**: a reader encountering both functions sees two nearly identical signatures and docstrings, with no indication that one is a thin wrapper around the other. The names (`transmm_prisma_enhanced` vs `generate_attribution_prisma_enhanced`) don't signal the wrapper/core distinction. `transmm_prisma_enhanced` is never called from outside `attribution.py`, making it effectively internal despite being export-stable.

4. Proposed boundary (no logic change):
Keep both functions and make the wrapper role explicit:
- Add a one-line docstring annotation to `generate_attribution_prisma_enhanced` stating it wraps `transmm_prisma_enhanced` into a dict interface for pipeline consumption.
- Add a note to `transmm_prisma_enhanced` that it is the core implementation, called only through `generate_attribution_prisma_enhanced`.
- Consider extracting the post-processing block (L236-248: reshape, interpolate, normalize) from `transmm_prisma_enhanced` into a private helper `_postprocess_attribution` to separate orchestration from output formatting.

5. Proposed naming roles:
- `transmm_prisma_enhanced`: Core implementation — hooks, forward/backward, attribution computation
- `generate_attribution_prisma_enhanced`: Dict adapter for pipeline consumption

### `compute_feature_gradient_gate` vs `apply_feature_gradient_gating`
1. Responsibility of `compute_feature_gradient_gate`:
Core gate computation: takes raw tensors (residual gradient, SAE codes, decoder matrix), computes per-patch scalar scores via one of four gate_construction modes, normalizes via MAD, and maps to gate multipliers via `10 ** tanh(s_norm)`. Also collects debug info (sparse feature extraction).

2. Responsibility of `apply_feature_gradient_gating`:
Entry point adapter: unpacks config dict into named parameters, resolves SAE decoder matrix from SAE object (handles `W_dec` vs `decoder` attribute), delegates to `compute_feature_gradient_gate`, then applies the resulting gate vector to the attention CAM tensor (CLS-token aware), computes attribution deltas, and assembles debug output dict.

3. Current overlap and why it increases cognitive load:
No algorithmic overlap — the boundary is clean and correct. However, `apply_feature_gradient_gating` mixes five distinct sub-responsibilities (config unpacking, decoder resolution, delegation, CAM application, debug compilation) in a flat sequence, making it harder to scan. The function is 100 lines but each sub-block is small (<10 lines).

4. Proposed boundary (no logic change):
Extract debug collection from `compute_feature_gradient_gate` into a private helper `_collect_gate_debug_info` to make the hot path (gate computation) immediately visible. Optionally split `apply_feature_gradient_gating` sub-blocks into named helpers, though the individual blocks are already small.

## Unused and Reserved Parameter Ledger

| File:Function | Parameter | State | Proposed Action | Rationale |
|---|---|---|---|---|
| `core/attribution.py:apply_gradient_gating_to_cam` | `device` | `unused` | `remove_internal` | Function never references it. Device is obtained from `cam_pos_avg.device` downstream. `apply_gradient_gating_to_cam` is internal (only called by `compute_layer_attribution`), so removing the parameter has no external API impact. Caller already has `device` available and doesn't need to pass it. |
| `core/gating.py:compute_feature_gradient_gate` | `kappa` | **RESOLVED: removed** | removed per maintainer decision | Removed from `compute_feature_gradient_gate` signature and full config chain. Retained in `PipelineConfig` as sweep metadata only (used for experiment naming in sweep.py/comparison.py). |
| `core/gating.py:compute_feature_gradient_gate` | `clamp_max` | **RESOLVED: wired in** | wired into active formula per maintainer decision | Formula changed from hardcoded `10 ** tanh(s_norm)` to `clamp_max ** tanh(s_norm)` with default 10.0. Numeric equivalence verified: max_diff == 0.0 with default value. |
| `core/gating.py:compute_feature_gradient_gate` | `residual` | `conditionally_used` | no action needed | Only used when `gate_construction == "no_SAE"` (L77-80). This is correct behavior — `no_SAE` mode uses raw `residual * residual_grad` instead of SAE-decomposed contributions. |

**Resolved**: Maintainer decided to remove `kappa` (not used in formula) and wire `clamp_max` into the active gate formula with default 10. ARG001 findings reduced from 5 to 3.

## Refactor Slice Plan (No Logic Change)

| Slice | Target Files | Planned Changes | Public Signature Impact | Risk | Validation |
|---|---|---|---|---|---|
| `WP-06B` | `core/attribution.py` | (1) Annotate adapter/core roles in docstrings for `transmm_prisma_enhanced` and `generate_attribution_prisma_enhanced`. (2) Extract post-processing (reshape/interpolate/normalize) from `transmm_prisma_enhanced` into `_postprocess_attribution`. (3) Remove unused `device` parameter from internal `apply_gradient_gating_to_cam` and its call site. | none (only internal function touched) | low | Signature check + import smoke + numeric equivalence on synthetic gate tensor |
| `WP-06C` | `core/gating.py` | (1) Extract debug collection from `compute_feature_gradient_gate` into `_collect_gate_debug_info`. (2) Optionally extract score construction dispatch into `_compute_patch_scores`. | none | low | Signature check + import smoke + numeric equivalence on synthetic gate tensor |
| `WP-06D` | `experiments/sweep.py`, `experiments/case_studies.py` | Split oversized orchestration functions into small private helpers. Preserve external behavior and output locations. | none | low | Signature check + import smoke |
| `WP-06E` | `data/download.py` and all touched modules | Clear remaining `ARG001`/`ARG002` findings: `description` in `download_from_gdrive`, `models_dir` in `download_imagenet`. Remove or annotate intentionally retained compatibility parameters. | depends on findings | low | `uvx ruff check --select ARG001,ARG002` clean |

## Equivalence Contract
All WP-06 implementation slices must satisfy this contract.

### 1. Import and Signature Stability
Run and record:

```bash
uv run python -c "import inspect; from transmm import transmm_prisma_enhanced, generate_attribution_prisma_enhanced; from feature_gradient_gating import compute_feature_gradient_gate, apply_feature_gradient_gating; print(inspect.signature(transmm_prisma_enhanced)); print(inspect.signature(generate_attribution_prisma_enhanced)); print(inspect.signature(compute_feature_gradient_gate)); print(inspect.signature(apply_feature_gradient_gating))"
```

Baseline result (WP-06A):
```
transmm_prisma_enhanced:
(model_prisma: vit_prisma.models.base_vit.HookedViT, input_tensor: torch.Tensor, config: gradcamfaith.core.config.PipelineConfig, idx_to_class: Dict[int, str], device: Optional[torch.device] = None, img_size: int = 224, steering_resources: Optional[Dict[int, Dict[str, Any]]] = None, enable_feature_gradients: bool = True, feature_gradient_layers: Optional[List[int]] = None, clip_classifier: Optional[Any] = None, debug: bool = False) -> Tuple[Dict[str, Any], numpy.ndarray, numpy.ndarray, Dict[int, Dict[str, Any]]]

generate_attribution_prisma_enhanced:
(model: vit_prisma.models.base_vit.HookedSAEViT, input_tensor: torch.Tensor, config: gradcamfaith.core.config.PipelineConfig, idx_to_class: Dict[int, str], device: Optional[torch.device] = None, steering_resources: Optional[Dict[int, Dict[str, Any]]] = None, enable_feature_gradients: bool = True, feature_gradient_layers: Optional[List[int]] = None, clip_classifier: Optional[Any] = None) -> Dict[str, Any]

compute_feature_gradient_gate:
(residual_grad: torch.Tensor, residual: Optional[torch.Tensor], sae_codes: torch.Tensor, sae_decoder: torch.Tensor, clamp_max: float = 10.0, gate_construction: str = 'combined', shuffle_decoder: bool = False, shuffle_decoder_seed: int = 12345, active_feature_threshold: float = 0.1, debug: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]

apply_feature_gradient_gating:
(cam_pos_avg: torch.Tensor, residual_grad: torch.Tensor, residual: Optional[torch.Tensor], sae_codes: torch.Tensor, sae: Any, config: Optional[Dict[str, Any]] = None, debug: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]
```

### 2. Synthetic Equivalence Harness (Core)
For each slice touching `core/attribution.py` or `core/gating.py`:
- run pre/post comparison on fixed-seed synthetic tensors
- report:
  - max absolute diff
  - mean absolute diff
  - shape equality

Template:

| Metric | Value |
|---|---|
| Max absolute diff | (fill per slice) |
| Mean absolute diff | (fill per slice) |
| Shape equal | (fill per slice) |

Decision threshold (default):
- require exact equality for deterministic branches when possible
- if minor floating differences appear, justify and cap tolerance explicitly

Synthetic harness code:
```python
import torch
from gradcamfaith.core.gating import compute_feature_gradient_gate

torch.manual_seed(42)
n_patches, d_model, n_features = 196, 768, 3072

residual_grad = torch.randn(n_patches, d_model)
sae_codes = torch.relu(torch.randn(n_patches, n_features))
sae_decoder = torch.randn(d_model, n_features)

gate, _ = compute_feature_gradient_gate(
    residual_grad=residual_grad, residual=None,
    sae_codes=sae_codes, sae_decoder=sae_decoder,
    gate_construction="combined", debug=False,
)
print(f"gate shape: {gate.shape}, mean: {gate.mean():.6f}, std: {gate.std():.6f}")
# Save gate tensor for post-refactor comparison
torch.save(gate, "/tmp/wp06_gate_baseline.pt")
```

### 3. Lint Checks for Unused Arguments
Run:

```bash
uvx ruff check src/gradcamfaith/core src/gradcamfaith/experiments src/gradcamfaith/data --select ARG001,ARG002
```

Baseline result (WP-06A, 5 findings -> 3 after kappa/clamp_max fix):
```
src/gradcamfaith/core/attribution.py:20:76  ARG001 Unused function argument: `device`
src/gradcamfaith/data/download.py:39:59     ARG001 Unused function argument: `description`
src/gradcamfaith/data/download.py:112:39    ARG001 Unused function argument: `models_dir`
```

### 4. Path Smokes
Run and record:

```bash
uv run python -c "from main import run_parameter_sweep; from transmm import generate_attribution_prisma_enhanced; from feature_gradient_gating import apply_feature_gradient_gating; print('smoke-ok')"
uv run python -c "from gradcamfaith.core.attribution import generate_attribution_prisma_enhanced; from gradcamfaith.core.gating import apply_feature_gradient_gating; print('pkg-smoke-ok')"
```

Result (WP-06A):
```
smoke-ok
pkg-smoke-ok
```

**Note**: The root smoke check initially failed due to a circular import in `src/gradcamfaith/models/__init__.py` (introduced by WP-05 fix). The eager `from pipeline import run_unified_pipeline` in `models/__init__.py` triggered a circular dependency: `pipeline.py` -> `gradcamfaith.models.load` -> `gradcamfaith.models.__init__` -> `pipeline` (still loading). Fixed by converting to a lazy `__getattr__` re-export. This fix is included in the WP-06A commit.

## Review Checklist for WP-06A
- `[x]` Function inventory complete for attribution and gating modules.
- `[x]` Overlap analysis explicitly answered for both function pairs.
- `[x]` Unused/reserved parameter ledger complete.
- `[x]` Slice plan (`WP-06B` to `WP-06E`) filled with concrete planned changes.
- `[x]` Equivalence contract with baseline values recorded.
- `[ ]` Reviewer decision recorded in `AGENTS.md`.

## Reviewer Decision
One of:
- `accepted`
- `accepted with follow-ups`
- `rework requested`

Decision:
(pending review)

Notes:
(pending review)
