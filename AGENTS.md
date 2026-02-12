# Project Overview

This repository simulates a two-population interacting particle system in 2D with overdamped dynamics and optional Brownian diffusion. It is used to study emergent behavior under power-law interactions with population-dependent charges, and to support future experiments with alternative repulsive laws (for example exponential forms).

Main execution path: `diffusion_sim.cli` (installed command: `diffusion-sim`).

# Scientific Model

## Equations

State is particle positions `r_i(t)` in 2D.

Overdamped dynamics:

`v_i = mu * F_i`

Current implementation (effective velocity update):

`v_i = sum_{j!=i} coupling * (q_i * q_j) * r_ij / (|r_ij|^(k+2))`

with:
- `r_ij = r_i - r_j`
- `coupling = v0 * l^(k+1)`
- minimum distance floor `r_floor` applied via `|r_ij|_eff = max(|r_ij|, r_floor)`.

Stochastic diffusion increment per step:

`dx = sqrt(2 * D * dt) * noise`

where `noise` is Gaussian with configured variance (`diffusion_noise_var`).

## Potential Energy

Pair contribution:
- if `k == 0`: `PE_ij = -coupling * (q_i * q_j) * log(|r_ij|)`
- else: `PE_ij = coupling * (q_i * q_j) / (k * |r_ij|^k)`

## Assumptions

- Dimensionless units.
- Mixed deterministic + stochastic dynamics.
- Pairwise additive interactions.
- Two charge populations at present (extensible later).
- No external fields currently.
- No explicit boundary conditions currently.
- No geometric constraints beyond `r_floor`.
- Forces are currently reciprocal.

## Typical Regimes

- Small test: `N ~= 300`
- Typical runs: `N ~= 700-1000`
- Large runs: `N ~= 1000-3000`

## Invariants / Expected Trends

- No strict energy conservation (overdamped + optional noise).
- Practical sanity check: energy vs time should decay without spikes.
- Standard deviation trend checks:
- Without diffusion: expected scaling near `t^(1/(k+2))`
- With diffusion at large separation: expected scaling near `t^(1/2)`

# Numerical Methods

## Integrators

Supported:
- `rk23` (default)
- `dop853`
- legacy fixed-step `rk2` / `rk4` implementations (deprecated)

Default is `rk23` because diffusion is integrated in the RK23 workflow and adaptive error order is usually not the dominant limitation once stochastic kicks are included.

## Time Stepping and Tolerances

- `rk23` and `dop853` run with adaptive internal steps.
- Defaults: `rtol=1e-6`, `atol=1e-6`.
- `save_every` controls output sampling.
- `dop853` requires `save_every`.
- `rk23` can run without `save_every` (accepted-step output).

## Reproducibility

- Initialization uses `seed`.
- Diffusion uses a separate `diffusion_seed`.
- Goal is reproducible runs when seeds + config are fixed.

## Known Gaps

- No robust in-run instability detector yet.
- Blow-ups are typically diagnosed post-run from energy/trajectory behavior.

# Repo Tour

- `src/diffusion_sim/cli.py`: main CLI entry point and argument/config merge.
- `src/diffusion_sim/config.py`: `SimulationConfig`, file loading, validation.
- `src/diffusion_sim/simulation.py`: orchestration of initialization, integration, saving, resume.
- `src/diffusion_sim/forces.py`: overdamped interaction kernel.
- `src/diffusion_sim/integrators/`: solver wrappers (`rk23`, `dop853`, plus legacy fixed-step).
- `src/diffusion_sim/io/`: NPZ/HDF5 save/load and HDF5 batch/resume helpers.
- `src/diffusion_sim/plotting.py`: plotting/animation utilities.
- `src/diffusion_sim/postprocess/density_voronoi.py`: density post-processing.
- `tests/`: existing tests (`config`, `io`, `integrators`, `rk23`) but coverage is incomplete.
- notebooks (`*analysis.ipynb`, etc.): exploratory analysis and plotting workflows.

# How To Run

## Prerequisites

- Python `>=3.11`
- `uv` recommended (project standard)

## Environment + Dependencies (`uv`)

From repo root:

```bash
uv sync
```

Install with dev extras (notebooks):

```bash
uv sync --group dev
```

## Minimal Demo

```bash
uv run diffusion-sim --method rk23 --n-particles 100 --t-duration 0.5 --save-every 1e-3
```

Equivalent module form:

```bash
uv run python -m diffusion_sim.cli --method rk23 --n-particles 100 --t-duration 0.5 --save-every 1e-3
```

## Config-Driven Run

```bash
uv run diffusion-sim --config config_rk23.json
```

## Full Simulation Pattern

Large/long runs are usually launched by tuning JSON config files and running:

```bash
uv run diffusion-sim --config <your_config.json>
```

For HDF5 batching/resume:

```bash
uv run diffusion-sim --config config_rk23.json --out data/run.h5 --batch-every 0.5 --target-batch-mb 64
uv run diffusion-sim --resume-from data/run.h5 --t-duration 10.0
```

## Tests

```bash
uv run pytest
```

## Reproducing Figures

No single scripted figure-reproduction command exists yet. Current workflow:
- run simulation via CLI,
- load output in notebook,
- use functions from `src/diffusion_sim/plotting.py`.

## Runtime Expectations

- Large runs can take several hours.

# Configuration

Configuration source order:
1. `SimulationConfig` defaults
2. JSON file via `--config`
3. CLI overrides

Primary parameters:
- System: `n_particles`, `k`, `v0`, `l`, `r_floor`, `init_radius`
- Populations: `charge_values`, `charge_counts`
- Time/integration: `t0`, `t_duration`, `method`, `save_every`, `rtol`, `atol`
- RK23 options: `first_step`, `max_step_global`, `eta`, `recompute_every`
- Diffusion: `diffusion`, `diffusion_coeff`, `diffusion_seed`, `diffusion_noise_var`
- Output/checkpointing: `out_format`, `batch_every`, `target_batch_mb`, `max_wall_time`, `resume_from`, `resume_force`

Config files are JSON; several examples exist in repo root (for example `config_rk23.json`, `config_rk23_charged_small.json`).

# Outputs & Diagnostics

## Saved State

Typical fields in simulation output:
- `positions` shape `(T, N, 2)` (`float64`)
- `times`
- `energy`
- `std`
- `final_positions`
- `charges`
- optional metadata and resume stats

Formats:
- primary: HDF5 (`.h5` / `.hdf5`)
- also supported: NPZ (`.npz`)

Sampling frequency is controlled by `save_every`.

Default naming pattern (user convention):

`data/YYYYMMDD/<method>_N<N>_t<T>_k<k>_rtol<rtol>_q<charge1>_<charge2>_n<num_charge1>_<num_charge2>_atol<atol>_save<save_every>_Diff<Diffusion_const>.hdf5`

## Metadata

Metadata typically records simulation parameters, tolerances, seeds, diffusion settings, batching/resume fields, completion state, and elapsed runtime.

## Plotting / Animation Dependencies

Used tools include:
- `matplotlib`
- `scipy.ndimage.gaussian_filter1d`
- `imageio_ffmpeg`
- `matplotlib.animation.FuncAnimation`
- `matplotlib.animation.FFMpegWriter`

# Validation

Current practical checks:
- energy vs time should decay without spikes,
- standard deviation trend should be physically plausible for regime (diffusive vs non-diffusive),
- compare selected runs to `dop853` reference; target per-particle positional agreement around `1e-3` under chosen RK23 settings.

Reference behavior available:
- one-population, no-diffusion self-similar solution.

Known limitation:
- no complete formal correctness oracle yet for two-population + diffusion regimes.

# Common Pitfalls

- Forgetting to set `save_every` with `dop853` (required).
- Setting only one of `charge_values` / `charge_counts` (must set both).
- `charge_counts` not summing to `n_particles`.
- Using batching/resume options with non-HDF5 output format.
- Assuming in-run blow-up detection exists; currently most diagnosis is post-run.
- Relying on undocumented defaults for long runs instead of an explicit JSON config.

# Contributing

No strict team style policy is currently enforced. Recommended baseline for maintainability:

- Use `ruff` + `black` for lint/format.
- Prefer type hints for public functions and config/data interfaces.
- Add docstrings on non-trivial APIs.
- Add/extend tests in `tests/` for behavior changes.

Suggested commit style:
- small, focused commits
- imperative subject lines (`Add RK23 diffusion seed validation`)

## Adding New Physics

To add a new force law:
1. Implement force and matching energy expression.
2. Thread parameters through `SimulationConfig` and CLI.
3. Keep diagnostics (`energy`, `std`) compatible.
4. Add targeted tests and one reproducible example config.

To add a new observable:
1. Compute in simulation/postprocess path.
2. Save in NPZ/HDF5 consistently.
3. Expose in plotting/notebook workflow.
4. Add at least one validation test or regression check.
