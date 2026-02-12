# `simulation.py` + `rk23.py` Rewrite Plan

This is a refactor plan only. No behavior changes should be introduced while executing these tasks.

## Scope

Files in scope:
- `src/diffusion_sim/simulation.py`
- `src/diffusion_sim/integrators/rk23.py`

Primary goal:
- Make long functions readable and separable by responsibility.

Non-goals:
- Changing scientific model equations.
- Changing public CLI/config semantics.
- Performance optimization beyond neutral refactor.

## Definition Of Done

- Long orchestration and integration functions are split into named helpers with clear inputs/outputs.
- Public APIs remain backward compatible.
- Existing behavior is preserved (within current numeric tolerance).
- Tests cover the refactored control flow sufficiently to protect future changes.

## Task Breakdown

## Phase 0: Baseline Safety Net

- [ ] Task 0.1: Record baseline behavior
  - Run targeted tests for config, simulation flow, and integrators.
  - Save one deterministic reference run artifact (fixed seeds, small N) for output comparison.
- [ ] Task 0.2: Add/strengthen tests before refactor
  - Add a smoke test for `simulation.py` full orchestration path.
  - Add/extend a test for `rk23.py` output contract (shapes, monotonic time, metadata fields).
  - Add resume-path test (charged and non-charged cases if feasible).

Deliverable:
- Green baseline tests + one reproducible reference run.

## Phase 1: `simulation.py` Responsibility Split

- [ ] Task 1.1: Map responsibilities in the current long function(s)
  - Identify blocks for init/resume, integrator dispatch, batching/checkpointing, finalize/save.
- [ ] Task 1.2: Extract pure helpers first
  - Config normalization/derived values helper.
  - Resume validation helper.
  - Integrator argument assembly helper.
- [ ] Task 1.3: Extract side-effect helpers
  - Resume loading helper.
  - HDF5/NPZ save+metadata helper.
  - Checkpoint/update helper.
- [ ] Task 1.4: Rebuild top-level flow as staged pipeline
  - `prepare -> init_or_resume -> run -> finalize`.
  - Keep current public entrypoint names/signatures.
- [ ] Task 1.5: Keep data contracts explicit
  - Use typed lightweight containers (`dataclass`/typed dict) for shared state if argument lists become too long.

Deliverable:
- `simulation.py` entry flow is short and readable; internal logic moved to named helpers.

## Phase 2: `rk23.py` Integration Loop Decomposition

- [ ] Task 2.1: Define internal solver state structure
  - Group mutable loop state (`t`, `dt`, `r`, counters, diagnostics) in one internal state object.
- [ ] Task 2.2: Extract step lifecycle helpers
  - Propose step.
  - Evaluate error norm.
  - Accept/reject decision.
  - Step-size update policy.
- [ ] Task 2.3: Extract physics/noise helpers
  - Deterministic drift evaluation wrapper.
  - Diffusion kick application wrapper.
- [ ] Task 2.4: Extract output/sampling helpers
  - Sample/save decision (accepted step vs `save_every` behavior).
  - Metrics update (`energy`, `std`) and append.
- [ ] Task 2.5: Flatten nested branches
  - Replace deep nesting with early-continue/early-return where possible.

Deliverable:
- RK23 main function becomes a readable coordinator over small helper calls.

## Phase 3: Contract Validation And Parity

- [ ] Task 3.1: Compare refactor output vs baseline artifacts
  - Positions: tolerance check.
  - Energy/std trends: no structural regression.
  - Metadata and saved arrays: expected keys/shapes preserved.
- [ ] Task 3.2: Resume and batching parity checks
  - Ensure resumed run continuation produces consistent concatenated outputs.
- [ ] Task 3.3: Backward compatibility check
  - Existing CLI/config workflows still run without required argument changes.

Deliverable:
- Documented parity check results with pass/fail notes.

## Phase 4: Cleanup And Documentation

- [ ] Task 4.1: Remove dead branches and duplicated code discovered during extraction.
- [ ] Task 4.2: Add concise docstrings for non-trivial helper functions.
- [ ] Task 4.3: Add module-level flow note at top of each file (short, high-level).
- [ ] Task 4.4: Update relevant developer docs/tests references.

Deliverable:
- Cleaner modules + maintainable docs/test coverage for new structure.

## Suggested PR/Commit Slicing

1. Safety-net tests + baseline artifact generation.
2. `simulation.py` pure helper extraction.
3. `simulation.py` side-effect helper extraction + staged flow.
4. `rk23.py` state + step lifecycle helper extraction.
5. `rk23.py` sampling/diagnostics extraction + nesting cleanup.
6. Parity checks, cleanup, and docs.

## Risk List (And Mitigation)

- Risk: subtle behavior drift in adaptive stepping.
  - Mitigation: freeze baseline outputs + tolerance-based regression checks.
- Risk: resume path regressions.
  - Mitigation: dedicated resume tests with deterministic seeds.
- Risk: hidden coupling between simulation orchestration and integrator internals.
  - Mitigation: explicit adapter/helper boundaries and contract tests.

## Execution Notes

- Prefer small, reversible edits with test pass after each task.
- Avoid API renames until parity is proven.
- If a helper extraction changes behavior, revert that chunk and retry with smaller moves.
