# Diffusion Project Review and Refactor Plan

## Findings (ordered by severity)
- [High] DOP853 final_positions is the last sampled state at t0 + (steps - 1) * dt, while rk2/rk4/rk23 return the state at t0 + steps * dt. This mismatch can skew solver comparisons and downstream analysis; see `nbody_core/dop853/__init__.py:47` and `nbody_core/dop853/__init__.py:84`.
- [High] Potential energy divides by k, so k = 0 yields division by zero and invalid energy values; see `nbody_core/common.py:46`.
- [Medium] `run_fixedstep_chunked` can infinite loop if called with `chunk_steps <= 0` (m becomes 0, idx never advances); see `simulate.py:115`.
- [Low] Metadata always writes `density_stride=1` even when a different stride is used, so saved results misreport density sampling; see `simulate.py:368`.
- [Low] DOP853 chunk with m = 1 uses `t_span = (t, t)` and can skip integration for that chunk; see `nbody_core/dop853/__init__.py:49`.

## Flow map (modules and responsibilities)
- `simulate.py`
  - CLI and config parsing.
  - Initial conditions via `init_positions_jittered_disk`.
  - Dispatch to integrators (rk2, rk4, rk23, dop853).
  - Optional post-process density and save to npz.
- `nbody_core/common.py`
  - Physics: overdamped velocity from pairwise power-law repulsion.
  - Metrics: potential energy, radial std.
  - Initial condition sampler (jittered disk + optional min distance).
- `nbody_core/rk2/__init__.py`
  - Fixed-step RK2 and diffusion noise helpers.
- `nbody_core/rk4/__init__.py`
  - Fixed-step RK4 integrator.
- `nbody_core/rk23/__init__.py`
  - Adaptive RK23/RK45 wrapper using SciPy, optional sampling, progress callback.
- `nbody_core/dop853/__init__.py`
  - DOP853 integration via SciPy `solve_ivp`, chunked for progress.
- `density_voronoi.py`
  - Voronoi-based density per particle, and radii computation.
- `npz_io.py`
  - Load saved runs and parse metadata.
- `visualize.py`
  - Plot energy/std/density and create animations.
- `compare_solvers.py`
  - Compare solver outputs, plot divergence, create videos.
- `tests/test_rk23.py`
  - Basic sanity checks for RK23.

## Code review (quality and smells)
- Core simulation logic, CLI, IO, and plotting are mixed in single files; makes reuse/testing harder and increases surface area for bugs.
- Fixed-step integrators duplicate metric computation loops; would benefit from a shared "record/metrics" utility.
- Diffusion code paths are split between `run_rk2_loop` (random walk) and `run_rk2_loop_with_noise` (precomputed noise), but only one path is used in practice.
- Adaptive (RK23) and fixed-step paths return different time grids and final state semantics; cross-solver comparisons require careful interpretation.
- Density computation is heavy (Voronoi per step) and sits in the main simulation flow instead of a clear post-process phase with its own config and progress controls.
- Some minor text encoding artifacts in comments/docstrings (likely copy/paste); keep ASCII for portability.

## Physics logic review (step by step)
- Initialization: `init_positions_jittered_disk` samples area-uniform radii and angles with jitter, which matches a uniform disk. The optional min distance is a reasonable packing constraint but makes the distribution no longer strictly uniform; that is fine if intentional.
- Forces: `compute_velocity_overdamped` implements
  - v_i = sum_j (v0 * l^(k+1)) * (r_i - r_j) / |r_i - r_j|^(k+2)
  - This matches overdamped motion with potential U ~ (v0 * l^(k+1)) / (k * r^k) for k != 0 and positive v0 (repulsion).
- Softening: `r_floor` is a hard floor on distance. This makes the force non-smooth at r = r_floor and the potential non-differentiable there. That is numerically stable but not physically smooth; consider a softening kernel if you want continuous forces.
- Energy: energy formula matches the above potential for k != 0. For k = 0, the correct potential is logarithmic, so the current energy expression is invalid.
- Integration:
  - RK4/RK2 deterministic updates are standard.
  - Diffusion is added as an additive noise term after the RK2 step. That is closer to an Euler-Maruyama update than a true stochastic RK2. If you want accuracy for SDEs, consider a dedicated SDE integrator or switch to Euler-Maruyama for diffusion.
- Metrics: radial std is computed from particle radii about the origin, not the center of mass. If the center of mass drifts, this is a measure of spread around the origin rather than spread around the mean position.
- Density: Voronoi-based density is a reasonable proxy, but boundary cells are unbounded and set to 0, which will bias averages if you include edge particles.

## Refactor plan (no code changes, design only)

### Proposed file layout
- `src/diffusion_sim/`
  - `__init__.py` (public API)
  - `config.py` (dataclasses and validation)
  - `init_conditions.py` (disk sampling, optional min distance)
  - `forces.py` (pairwise force and potential)
  - `metrics.py` (energy, std, diagnostics)
  - `integrators/`
    - `base.py` (common interface and record helpers)
    - `rk2.py`
    - `rk4.py`
    - `rk23.py`
    - `dop853.py`
  - `simulation.py` (top-level run_simulation orchestration)
  - `postprocess/`
    - `density_voronoi.py`
  - `io/`
    - `npz.py`
  - `cli.py` (argument parsing, entry point)
  - `plotting.py` (plots and animations)
- `scripts/`
  - `compare_solvers.py` (moved from root)
- `tests/`
  - `test_integrators.py`
  - `test_physics.py`
  - `test_io.py`

### What to rewrite or remove
- Rewrite `simulate.py` into `simulation.py` (pure library) + `cli.py` (argument parsing and file output only).
- Remove or deprecate `run_rk2_loop` random-walk path and keep a single diffusion interface that always uses precomputed noise for reproducibility.
- Normalize integrator outputs so all methods return:
  - `positions` sampled at the same semantic times
  - `times` aligned across methods
  - `final_positions` at the same definition of final time
- Centralize metric computation in `metrics.py` and call it from a shared record helper to avoid duplication.
- Move plotting and comparison scripts under `scripts/` or `tools/` so they do not look like core library modules.

### Suggested cleanup and validation
- Add parameter validation in `config.py` (dt > 0, steps > 0, n > 1, k != 0 unless special-case log potential, r_floor >= 0).
- Add explicit support for k = 0 if you need it (log potential, 1/r force).
- Add a "metric_stride" and "record_stride" to reduce runtime for very long runs.
- Provide a consistent definition for "final_positions" across integrators.

## Optimization ideas
- Short-term (CPU):
  - Use a neighbor list or cell list with a cutoff if the force decays fast enough (large k). That reduces O(n^2) to near O(n) per step.
  - If long-range is required, consider Barnes-Hut or fast multipole methods to reduce complexity.
  - Compute energy/std at a lower cadence (stride) to reduce per-step cost.
- GPU (RTX A2000):
  - If N is large and steps are many, consider moving the full integration to GPU using CuPy or numba.cuda so you do not pay CPU-GPU transfer per step.
  - A direct O(n^2) kernel can still be too slow; GPU acceleration helps but scales better with a cutoff or approximation.
- Numerical stability:
  - Consider a smooth softening kernel (Plummer-type) to avoid force discontinuity at r_floor.
  - For diffusion, use a proper SDE integrator if accuracy matters (Euler-Maruyama or Heun for SDE).

## Testing gaps
- No tests for rk2/rk4/dop853 parity or for time-grid alignment across solvers.
- No tests for energy scaling with k or for k = 0 behavior.
- No tests for initialization uniformity or min distance constraints.
- No tests for density post-processing edge cases (duplicate points, collinear sets).
