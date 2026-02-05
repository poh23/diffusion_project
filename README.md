diffusion-project
=================

Simulation arguments and parameters
-----------------------------------

This project runs simulations via the package CLI. You can configure runs with
command-line arguments or a JSON config file.

Command-line usage
------------------

Basic (module):

```bash
python -m diffusion_sim.cli --method rk23 --n-particles 100 --t-duration 0.5 --save-every 1e-3
```

Installed CLI (after `pip install -e .`):

```bash
diffusion-sim --method rk23 --n-particles 100 --t-duration 0.5 --save-every 1e-3
```

Config file:

```bash
python -m diffusion_sim.cli --config config.json
```

Core arguments
--------------

- `--config`: Path to a JSON config file (matches `SimulationConfig` fields).
- `--out`: Output `.npz` path (default: `data/YYYYMMDD/<method>_N<N>_t<T>_k<k>_rtol<rtol>_atol<atol>_save<save_every>.npz`). Use `.h5` or `.hdf5` to save in HDF5 format.
- `--out-format`: Controls the default output suffix when `--out` is not provided (`npz`, `h5`, or `hdf5`). Can also be set in JSON config as `out_format`.
- `--n-particles`: Number of particles (`n_particles`).
- `--k`: Power-law exponent.
- `--v0`: Coupling prefactor.
- `--l`: Length scale (usually `1.0`).
- `--r-floor`: Hard distance floor for interactions.
- `--init-radius`: Initial disk radius for particle placement.
- `--t0`: Initial time.
- `--t-duration`: Total integration time.
- `--save-every`: Sample interval for output (optional for RK23; required for DOP853). If omitted, RK23 records accepted steps.
- `--method`: One of `rk23`, `dop853`.
- `--seed`: RNG seed for initialization.

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
- `--diffusion`: Enable stochastic diffusion step (rk23 only).
- `--diffusion-coeff`: Diffusion constant `D` used in the stochastic step.
- `--diffusion-seed`: RNG seed for the diffusion term.
- `--diffusion-noise-var`: Variance of the Gaussian noise used in the diffusion step.

If `save_every` is not set, RK23 returns data at accepted solver steps (no interpolation).
When diffusion is enabled, interpolation is disabled; if `save_every` is set it
records the first accepted step at or after each interval (no interpolation).

RK23 diffusion example:

```bash
python -m diffusion_sim.cli --method rk23 --t-duration 0.5 --save-every 1e-3 --diffusion --diffusion-coeff 0.05 --diffusion-seed 123
```

Batching / resume (HDF5 only)
-----------------------------

- `--batch-every`: Sim-time interval between batch flushes.
- `--target-batch-mb`: Target batch size in MB.
- `--max-wall-time`: Stop after this many seconds (checkpoint and exit).
- `--resume-from`: Resume from an existing HDF5 file.
- `--resume-force`: Resume even if config mismatch.

Example (batch + resume):

```bash
diffusion-sim --config config_rk23.json --out data/run.h5 --batch-every 0.5 --target-batch-mb 64
diffusion-sim --resume-from data/run.h5 --t-duration 10.0
```

Density post-processing
-----------------------

Density is computed as a post-process step using the standalone CLI below.

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

HDF5 files are also supported:

```bash
diffusion-density data/20260121/your_run.h5 --stride 10
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
  "t_duration": 0.5,
  "save_every": 1e-3,
  "method": "rk23",
  "rtol": 1e-6,
  "atol": 1e-6,
  "diffusion": true,
  "diffusion_coeff": 0.05,
  "diffusion_seed": 123,
  "diffusion_noise_var": 1.0,
  "batch_every": 0.5,
  "target_batch_mb": 64.0
}
```

Notes
-----

- `k=0` uses the logarithmic potential energy (limit of the power-law form).
- For RK23, sampling is optional and controlled by `save_every`.
- For DOP853, `save_every` is required to define the output sampling grid.
- RK23 diffusion adds a post-step stochastic displacement when `--diffusion` is enabled.
- Batching/resume is supported only with `out_format` set to `h5` or `hdf5`.

Future tasks
------------

- (none)
