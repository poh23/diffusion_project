diffusion-project
=================

Simulation arguments and parameters
-----------------------------------

This project runs simulations via `simulate.py`. You can configure runs with
command-line arguments or a JSON config file.

Command-line usage
------------------

Basic:

```bash
python simulate.py --method rk4 --n-particles 100 --steps 500 --dt 1e-3
```

Installed CLI (after `pip install -e .`):

```bash
diffusion-sim --method rk4 --n-particles 100 --steps 500 --dt 1e-3
```

Config file:

```bash
python simulate.py --config config.json
```

Core arguments
--------------

- `--config`: Path to a JSON config file (matches `SimulationConfig` fields).
- `--out`: Output `.npz` path (default: `data/YYYYMMDD/<method>_N<N>_steps<steps>_dt<dt>_k<k>_rtol<rtol>_atol<atol>.npz`). Use `.h5` or `.hdf5` to save in HDF5 format.
- `--out-format`: Controls the default output suffix when `--out` is not provided (`npz`, `h5`, or `hdf5`). Can also be set in JSON config as `out_format`.
- `--n-particles`: Number of particles (`n_particles`).
- `--k`: Power-law exponent.
- `--v0`: Coupling prefactor.
- `--l`: Length scale (usually `1.0`).
- `--r-floor`: Hard distance floor for interactions.
- `--init-radius`: Initial disk radius for particle placement.
- `--dt`: Time step (fixed-step methods) or requested sampling interval for RK23 when sampling is enabled.
- `--steps`: Number of steps (fixed-step) or sample count for RK23 when sampling is enabled.
- `--t0`: Initial time.
- `--method`: One of `rk2`, `rk4`, `rk23`, `dop853`.
- `--seed`: RNG seed for initialization.

Progress / chunking
-------------------

- `--chunk-steps`: Chunk size for progress updates.
- `--print-every-chunks`: Print progress every N chunks (`0` to disable).

Adaptive tolerances
-------------------

- `--rtol`: Relative tolerance (rk23/dop853).
- `--atol`: Absolute tolerance (rk23/dop853).

RK23 adaptive options
---------------------

- `--first-step`: Initial step size guess.
- `--max-step-global`: Global max step size.
- `--eta`: Safety factor for distance-based max step cap.
- `--recompute-every`: Recompute distance-based cap every N accepted steps.
- `--rk23-sample-dt`: Sample interval for RK23 output (optional).
- `--rk23-sample-count`: Number of samples for RK23 output (optional).
- `--rk23-status-every-steps`: Update rk23 progress status every N accepted steps.
- `--rk23-status-every-sec`: Update rk23 progress status every N seconds.

If `rk23_sample_dt` and `rk23_sample_count` are not set, RK23 returns data at
accepted solver steps (no interpolation).

Density post-processing
-----------------------

- `--compute-density`: Compute Voronoi density/radii after integration.
- `--density-print-every`: Print progress every N steps during density computation.
- `--density-stride`: Compute density every N steps (default 1 = every step).

Standalone density CLI
----------------------

After `pip install -e .`, you can run:

```bash
diffusion-density data/20260120/your_run.npz --suffix _with_density --print-every 10
```

To overwrite in place:

```bash
diffusion-density data/20260120/your_run.npz --overwrite
```

To compute density every N steps:

```bash
diffusion-density data/20260120/your_run.npz --stride 10
```

Config file fields (JSON)
-------------------------

All CLI options map to fields in `SimulationConfig`. Example:

```json
{
  "n_particles": 100,
  "k": 3.0,
  "v0": 1.0,
  "l": 1.0,
  "r_floor": 1e-12,
  "dt": 1e-3,
  "steps": 500,
  "method": "rk23",
  "rtol": 1e-6,
  "atol": 1e-6,
  "rk23_sample_dt": 1e-3,
  "rk23_sample_count": 500
}
```

Notes
-----

- `k=0` uses the logarithmic potential energy (limit of the power-law form).
- For RK23, sampling is optional and controlled by `rk23_sample_dt` and
  `rk23_sample_count`.

Future tasks
------------

- Add an optional post-step stochastic callback for RK23 (apply a random process after each accepted deterministic step).
