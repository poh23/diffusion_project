# Project Reorganization Plan

This plan targets three fronts: (1) duplication cleanup, (2) CLI/pipeline + argument design, and (3) batch saving + resumability. It is written so you can mark up what you want implemented next.

## 1) Duplication Cleanup (Remove wrappers, keep CLI stable)

Goal: remove legacy compatibility wrappers (e.g., `simulate.py`) and call the package entry points directly, while keeping the same CLI experience.

Plan
- Inventory wrappers and redundant entry points
  - Identify wrapper scripts at repo root (e.g., `simulate.py`, `visualize.py`, any legacy CLIs).
  - Confirm which ones can be removed without breaking workflows or imports.
- Ensure the CLI remains the canonical entry point
  - Keep `diffusion_sim.cli:main` as the only supported entry.
  - Ensure `python -m diffusion_sim.cli` (and the installed console script `diffusion-sim`) remain the expected interface.
- Update docs and scripts
  - Replace mentions of wrapper scripts in README and examples.
  - Remove wrapper files after references are updated.
- Confirm package entry points
  - If needed, add or verify console script entry in `pyproject.toml`.

Open questions
- Which wrappers are you okay removing besides `simulate.py`?
- Do you rely on any wrapper in notebooks or batch scripts that should be preserved as a stub/redirect?

## 2) CLI/Pipeline + Argument Reorganization

Goal: simplify config/CLI for RK23 (with/without diffusion) and occasional dop853 runs, remove density-from-sim options, and eliminate redundant time arguments.

Proposed parameter model (high level)
- Common physical parameters: `n_particles, k, v0, l, r_floor, init_radius, seed`
- Solver selection: `method: rk23 | dop853` (remove rk2/rk4 from CLI/config)
- Integration horizon:
  - Replace `dt` + `steps` with `t_duration` (total integration time).
- Output sampling:
  - `save_every` (uniform sampling interval). If omitted: record at accepted steps.
  - Drop `sample_count` to avoid ambiguity.
- Progress reporting:
  - Use a tqdm progress bar (remove `status_every_steps` / `status_every_sec` arguments).
- Diffusion settings (rk23):
  - `diffusion` (bool), `diffusion_coeff`, `diffusion_noise_var`, `diffusion_seed`
- Remove density flags from simulation config/CLI (leave post-process CLI).
 - Fix tqdm progress display for rk23 runs.

Plan
- Define the new config schema and migration strategy
  - Keep JSON key mapping for backward compatibility (old keys mapped → new keys).
  - Emit warnings for deprecated keys.
- Update validation logic
  - Enforce required fields for each method.
  - Enforce `diffusion` only for rk23.
- Update CLI + docs to reflect new schema
  - Simplify the CLI help and README examples.
- Update pipeline scripts (if any) to use `t_end` and `sample_dt`.

Open questions (resolved)
- Use `t_duration` (relative time from `t0`) for adaptive solvers.
- Remove rk2/rk4 from CLI/config (keep code available).
- Keep `sample_dt` only; no `sample_count`.

Example config (new schema)
```json
{
  "method": "rk23",
  "n_particles": 1000,
  "k": 3.0,
  "v0": 1.0,
  "l": 1.0,
  "r_floor": 1e-7,
  "init_radius": 1.0,
  "seed": 1,
  "t0": 0.0,
  "t_duration": 5.0,
  "rtol": 1e-3,
  "atol": 1e-3,
  "first_step": null,
  "max_step_global": null,
  "eta": 100.0,
  "recompute_every": 200,
  "save_every": 1e-3,
  "diffusion": true,
  "diffusion_coeff": 1.0,
  "diffusion_noise_var": 1.0,
  "diffusion_seed": 1,
  "out_format": "h5"
}
```

## 3) Batch Saving + Resumability

Goal: enable long runs to be saved in batches and resume from the last saved state.

Proposed behavior
- Add batching by simulation time:
  - `save_every` (seconds of simulated time between saves) or `save_every_steps` (accepted steps).
  - Each batch saves positions/metrics for its time window.
- Add resumable runs:
  - `resume_from` path (HDF5/NPZ) or `--resume` flag with auto-detect of latest chunk.
  - Store current state (positions + time + RNG state + solver status).
- Output format:
  - Prefer HDF5 for appendable chunked datasets.
  - For NPZ, save per-chunk files (e.g., `run_0001.npz`, `run_0002.npz`).

Plan
- Choose a batch format (recommended: HDF5 with extendable datasets)
- Implement periodic save hook in RK23 loop
  - Trigger on time threshold: `t_curr - last_save_t >= save_every`
  - Persist state required to resume (positions, time, RNG state, solver stats)
- Implement resume:
  - Load last saved state and continue integration to target `t_end`.
  - Ensure diffusion RNG reproducibility when resuming.
- Update CLI/config:
  - `save_every` (time), `save_dir`, `run_id`, `resume_from`
  - Optionally `max_wall_time` to stop and save.

Open questions
- Do you want a single growing file or per-chunk files?
- Should resume require the exact same config hash?
- Do you want deterministic reproducibility across resume boundaries?

## Next Step

Tell me:
- Which wrappers to remove or keep.
- Your preferred new parameter list (I can propose a concrete schema).
- Your preference for batch saving format and resume behavior.
