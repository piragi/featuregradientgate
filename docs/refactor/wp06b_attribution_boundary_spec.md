# WP-06B-R3 Clean Attribution API Specification (`compute_attribution`)

## Status
- Maintainer decision date: `2026-02-08` (updated)
- Implementation status: `done — awaiting review`
- Target branch: `wp/WP-06B-attribution-boundary-refactor`

## Why This Addendum Exists
WP-06B v1 improved clarity, but wrapper/core boundaries in `core/attribution.py` remain partially mixed.
This addendum defines a stricter boundary for coder implementation before moving to WP-06C.

## Core Decision
`compute_attribution` becomes the single canonical API and orchestrator in `src/gradcamfaith/core/attribution.py`.

Legacy attribution entrypoints from pre-refactor code are no longer signature-stable requirements.
They may be:
- removed, or
- retained as temporary deprecation shims (without signature guarantees)

## Hard Constraints
- No algorithm/metric behavior changes.
- No output naming/path changes.
- Logical equivalence must be preserved for attribution outputs.
- All in-repo call sites must be migrated to the new canonical API.
- If temporary shims are kept, they must be clearly marked deprecated.

## Required Function Topology

### 1. `compute_attribution` (new orchestrator)
Implement/upgrade `compute_attribution` in `src/gradcamfaith/core/attribution.py` as the only orchestrator:

```python
def compute_attribution(
    model_prisma: HookedViT,
    input_tensor: torch.Tensor,
    config: PipelineConfig,
    idx_to_class: Dict[int, str],
    device: Optional[torch.device] = None,
    img_size: int = 224,
    steering_resources: Optional[Dict[int, Dict[str, Any]]] = None,
    enable_feature_gradients: bool = True,
    feature_gradient_layers: Optional[List[int]] = None,
    clip_classifier: Optional[Any] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    ...
```

Responsibility:
- full orchestration only (device/input prep, hooks, forward/backward, layer attribution, post-processing)
- return canonical dict output with keys:
  - `predictions`
  - `attribution_positive`
  - `raw_attribution`
  - `debug_info`

### 2. Legacy entrypoints (`transmm_prisma_enhanced`, `generate_attribution_prisma_enhanced`)
- No longer required as stable public contract.
- Preferred clean rewrite outcome: remove both from active internal call paths.
- If retained temporarily:
  - wrappers only
  - emit deprecation warning
  - delegate directly to `compute_attribution`
  - contain zero orchestration logic

### 3. Existing helper split remains
- Keep `_postprocess_attribution` as output-format helper used by `compute_attribution`.
- Keep `apply_gradient_gating_to_cam` as adapter into `core/gating.py`.

## Separation Rules (Enforced)
- `compute_attribution` is the only place that may:
  - resolve default device
  - move `input_tensor` to device
  - add batch dimension when needed
- Deprecated wrappers (if any) must not repeat those steps.

## Naming and Docstring Contract
- `compute_attribution`: docstring must call it the "orchestrator" explicitly.
- Deprecated wrappers (if retained): docstrings must include `DEPRECATED` and migration target `compute_attribution`.

## Required Validation Evidence

### Signature and import stability
```bash
uv run python -c "from gradcamfaith.core.attribution import compute_attribution; print('compute-attribution-ok')"
```

### In-repo call-site migration check
```bash
rg -n "transmm_prisma_enhanced|generate_attribution_prisma_enhanced" pipeline.py src/
```

Expected:
- no active internal call sites to legacy entrypoints in production paths
- if matches exist, they must be explicit compatibility shims/tests only

### Wrapper contract tests (must add)
Create `tests/test_attribution_boundary_contracts.py` with at least:
- `test_compute_attribution_exists`
- `test_compute_attribution_output_contract`
- `test_pipeline_uses_compute_attribution`
- `test_legacy_wrappers_deprecated_or_absent`

Use monkeypatching to assert call-site delegation where needed (no heavy model execution required).

### Equivalence check
- Fixed-seed synthetic equivalence before/after refactor.
- Report:
  - max absolute diff
  - mean absolute diff
  - shape equality

### Standard smoke
```bash
uv run pytest tests/test_smoke_contracts.py tests/test_sweep_reproducibility.py tests/test_attribution_boundary_contracts.py
uv run python scripts/validation/git_workflow_audit.py
```

## Out of Scope for This Slice
- Refactoring `core/gating.py` internals (WP-06C).
- Changing experiment logic.
- Any CLI changes.

## Handoff Requirements
- Before/after function map for `src/gradcamfaith/core/attribution.py`.
- Call-site migration proof (`pipeline.py` and package modules use `compute_attribution`).
- If legacy wrappers remain: deprecation note + explicit removal plan (target slice/date).
- Equivalence summary with numeric results.
- Exact commands run and pass/fail outcomes.
