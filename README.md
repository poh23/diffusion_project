diffusion-project
=================

`diffusion-project` simulates a 2D overdamped interacting-particle system with
optional Brownian diffusion. The code supports the original one-population
workflow and the newer two-population workflow, both through the same CLI.

The main entry point is `diffusion_sim.cli`, exposed as the installed command
`diffusion-sim`.

Features
--------

- Overdamped pairwise dynamics with power-law interactions.
- Optional stochastic diffusion in the `rk23` workflow.
- One-population runs with uniform charge.
- Two-population runs with user-defined charge values and counts.
- Adaptive solvers: `rk23` and `dop853`.
- Output to NPZ or HDF5, with HDF5 batching and resume support.
- Post-processing CLIs for density and energy diagnostics.

Installation
------------

Project-standard setup uses `uv`:

```bash
uv sync
```

For notebooks and other dev tools:

```bash
uv sync --group dev
```

If you prefer the installed CLI outside `uv run`, install the package in your
environment first:

```bash
pip install -e .
```

Command-line usage
------------------

With `uv`:

```bash
uv run diffusion-sim --method rk23 --n-particles 100 --t-duration 0.5 --save-every 1e-3
```

Module form:

```bash
uv run python -m diffusion_sim.cli --method rk23 --n-particles 100 --t-duration 0.5 --save-every 1e-3
```

Installed CLI form:

```bash
diffusion-sim --method rk23 --n-particles 100 --t-duration 0.5 --save-every 1e-3
```

Run One-Population Simulations
------------------------------

The current code treats one-population runs as the default mode. Do not provide
`charge_values` or `charge_counts`. Internally, the simulation assigns every
particle charge `1.0`.

Example with `uv run`:

```bash
uv run diffusion-sim --method rk23 --n-particles 350 --t-duration 5.0 --save-every 1e-4
```

Equivalent module form:

```bash
uv run python -m diffusion_sim.cli --method rk23 --n-particles 350 --t-duration 5.0 --save-every 1e-4
```

One-population JSON config example:

```json
{
  "method": "rk23",
  "n_particles": 350,
  "k": 3.0,
  "v0": 1.0,
  "l": 1.0,
  "r_floor": 1e-7,
  "init_radius": 0.1,
  "t_duration": 5.0,
  "save_every": 1e-4,
  "rtol": 1e-3,
  "atol": 1e-3,
  "diffusion": true,
  "diffusion_coeff": 5.0,
  "diffusion_seed": 1,
  "out_format": "h5"
}
```

Run from config:

```bash
uv run diffusion-sim --config config_rk23.json
uv run python -m diffusion_sim.cli --config config_rk23.json
```

Run Two-Population Simulations
------------------------------

Two-population mode is enabled only when both `charge_values` and
`charge_counts` are provided. The current code requires exactly two charge
values and exactly two counts, and the counts must sum to `n_particles`.

CLI example:

```bash
uv run diffusion-sim --method rk23 --n-particles 1000 --t-duration 10.0 --save-every 1e-7 --charge-values 1.0 1.0 --charge-counts 900 100
```

Equivalent module form:

```bash
uv run python -m diffusion_sim.cli --method rk23 --n-particles 1000 --t-duration 10.0 --save-every 1e-7 --charge-values 1.0 1.0 --charge-counts 900 100
```

Two-population JSON config example:

```json
{
  "method": "rk23",
  "n_particles": 1000,
  "k": -1.0,
  "v0": 1.0,
  "l": 1.0,
  "r_floor": 1e-7,
  "init_radius": 0.5,
  "t_duration": 10.0,
  "save_every": 1e-7,
  "max_step_global": 1e-3,
  "seed": 1,
  "eta": 100.0,
  "rtol": 1e-3,
  "atol": 1e-3,
  "charge_values": [1.0, 1.0],
  "charge_counts": [900, 100],
  "diffusion": false,
  "out_format": "h5"
}
```

Run from config:

```bash
uv run diffusion-sim --config config_rk23_two_species.json
uv run python -m diffusion_sim.cli --config config_rk23_two_species.json
```

Core Arguments
--------------

- `--config`: Path to a JSON config file matching `SimulationConfig`.
- `--out`: Output `.npz`, `.h5`, or `.hdf5` path.
- `--out-format`: Default output format when `--out` is not provided.
- `--n-particles`: Number of particles.
- `--k`: Power-law exponent.
- `--v0`: Coupling prefactor.
- `--l`: Length scale.
- `--r-floor`: Minimum effective pair distance.
- `--init-radius`: Initial disk radius for particle placement.
- `--t0`: Initial simulation time.
- `--t-duration`: Total integration time.
- `--save-every`: Sample interval for saved output.
- `--save-every-steps`: For `rk23`, save every N accepted steps.
- `--method`: `rk23` or `dop853`.
- `--seed`: RNG seed for initialization.
- `--charge-values`: Two positive charge values for two-population runs.
- `--charge-counts`: Two population sizes for two-population runs.

Adaptive and Diffusion Options
------------------------------

- `--rtol`: Relative tolerance.
- `--atol`: Absolute tolerance.
- `--first-step`: Initial `rk23` step guess.
- `--max-step-global`: Global maximum step size for `rk23`.
- `--eta`: Safety factor for the `rk23` distance cap.
- `--recompute-every`: Recompute the `rk23` distance cap every N accepted steps.
- `--no-interpolation`: Disable interpolation when sampling `rk23` with `--save-every`.
- `--diffusion`: Enable stochastic diffusion for `rk23`.
- `--diffusion-coeff`: Diffusion constant `D`.
- `--diffusion-seed`: RNG seed for the diffusion term.
- `--diffusion-noise-var`: Variance of the Gaussian diffusion noise.

Notes:

- `dop853` requires `--save-every`.
- `save_every` and `save_every_steps` are mutually exclusive.
- Diffusion is supported only with `rk23`.
- When diffusion is enabled, interpolation-based sampling is disabled.

Batching and Resume
-------------------

These options are supported only with HDF5 output.

- `--batch-every`: Simulated-time interval between flushes.
- `--target-batch-mb`: Target batch size in MB.
- `--max-wall-time`: Stop after this many wall-clock seconds and checkpoint.
- `--resume-from`: Resume from an existing HDF5 run.
- `--resume-force`: Resume even if config metadata differs.

Example:

```bash
uv run diffusion-sim --config config_rk23.json --out data/run.h5 --batch-every 0.5 --target-batch-mb 64
uv run diffusion-sim --resume-from data/run.h5 --t-duration 10.0
```

Post-processing CLIs
--------------------

Density:

```bash
uv run diffusion-density data/20260120/your_run.npz --suffix _with_density --print-every 10
uv run diffusion-density data/20260120/your_run.h5 --stride 10
```

Energy:

```bash
uv run diffusion-energy data/20260120/your_run.h5
```

External Potential
------------------

An optional one-body external potential can be added through the JSON config.
Currently supported:

- `external_potential: "harmonic"`
- `external_potential_params["omega"]` required
- `external_potential_params["center"]` optional, defaults to `[0.0, 0.0]`

For the harmonic case, the potential is:

`V(x) = omega * |x - center|^2`

If `center` is omitted, it defaults to `[0.0, 0.0]`.

Example:

```json
{
  "external_potential": "harmonic",
  "external_potential_params": {
    "omega": 0.5,
    "center": [0.0, 0.0]
  }
}
```

Project Structure
-----------------

```text
.
|-- AGENTS.md
|-- README.md
|-- Q&A.md
|-- pyproject.toml
|-- uv.lock
|-- config_rk23.json
|-- config_rk23_two_species.json
|-- src/
|   `-- diffusion_sim/
|       |-- __init__.py
|       |-- cli.py
|       |-- config.py
|       |-- density_cli.py
|       |-- energy_cli.py
|       |-- external_potentials.py
|       |-- forces.py
|       |-- init_conditions.py
|       |-- metrics.py
|       |-- integrators/
|       |   |-- __init__.py
|       |   |-- dop853.py
|       |   |-- rk23.py
|       |   |-- rk2.py
|       |   `-- rk4.py
|       |-- io/
|       |   |-- __init__.py
|       |   |-- h5.py
|       |   |-- h5_batch.py
|       |   `-- npz.py
|       |-- plotting/
|       |   |-- __init__.py
|       |   |-- animation_video/
|       |   |-- density_profile_plots/
|       |   |-- energy_metrics/
|       |   |-- mean_radial_separation_plots/
|       |   |-- mixing_inner_core_structure/
|       |   |-- radial_comparison/
|       |   |-- shared_helpers/
|       |   |-- std_diagnostics/
|       |   `-- wasserstein_radial_separation_plots/
|       |-- postprocess/
|       |   |-- __init__.py
|       |   |-- density_voronoi.py
|       |   `-- energy_components.py
|       `-- simulation/
|           |-- __init__.py
|           |-- helpers.py
|           |-- nonstream.py
|           `-- streaming.py
|-- tests/
|   |-- test_cli_sweep.py
|   |-- test_config.py
|   |-- test_density_cli.py
|   |-- test_external_potentials.py
|   |-- test_integrators.py
|   |-- test_io.py
|   |-- test_plotting_radial_wasserstein.py
|   `-- test_rk23.py
|-- scripts/
|   |-- __init__.py
|   `-- compare_solvers.py
|-- data/
|   `-- refactor_baseline/
|       `-- rk23_charged_small_baseline_fingerprint.json
|-- graphs/
|-- videos/
`-- notebooks and analysis files
    |-- integrator_k_comparison.ipynb
    |-- single_pop_analysis.ipynb
    |-- two_pop_analysis_overview.ipynb
    |-- two_pop_analysis_k3_variants.ipynb
    |-- two_pop_analysis_high_charge.ipynb
    |-- two_pop_analysis_power_laws.ipynb
    |-- two_pop_analysis_k0.ipynb
    `-- two_pop_analysis_k_comparison.ipynb
```

Directory guide:

- `src/diffusion_sim/`: main package code for simulation, IO, plotting, and post-processing.
- `src/diffusion_sim/integrators/`: solver implementations and wrappers.
- `src/diffusion_sim/io/`: NPZ/HDF5 save, load, batching, and resume helpers.
- `src/diffusion_sim/simulation/`: orchestration code for streaming and non-streaming runs.
- `src/diffusion_sim/plotting/`: plotting and diagnostic modules grouped by analysis type.
- `src/diffusion_sim/postprocess/`: offline analysis helpers used by CLIs and plotting.
- `tests/`: regression and behavior tests for config, IO, integrators, plotting, and CLI features.
- `scripts/`: ad hoc utility scripts.
- `data/`: sample or generated run outputs and saved baselines.
- `graphs/`: generated figures.
- `videos/`: generated animations.
- notebooks: exploratory analysis and figure-generation workflows.

Notebooks
---------

The repository uses Jupytext for the newer two-population analysis notebooks.
Those notebooks are paired as `.ipynb` and `.py` files in `py:percent` format.

Current paired notebook set:

- `two_pop_analysis_overview.ipynb` / `two_pop_analysis_overview.py`: overview notebook for the smaller and baseline two-population runs, including early `N=100`, `N=350`, and initial `N=700` examples.
- `two_pop_analysis_k3_variants.ipynb` / `two_pop_analysis_k3_variants.py`: `k=3` two-population runs comparing different seeds, diffusion strengths, and initial-radius variations.
- `two_pop_analysis_high_charge.ipynb` / `two_pop_analysis_high_charge.py`: high-charge-ratio cases such as `q=20` and `q=100`, with energy, standard deviation, density, and saved-video views.
- `two_pop_analysis_power_laws.ipynb` / `two_pop_analysis_power_laws.py`: comparisons across different power-law exponents, mainly `k=1`, `k=0`, and `k=-1`.

Other notebooks currently in the repo:

- `single_pop_analysis.ipynb`: exploratory analysis for one-population simulations.
- `integrator_k_comparison.ipynb`: integrator and parameter-comparison notebook.
- `two_pop_analysis_k0.ipynb`: older focused notebook for the `k=0` two-population case.
- `two_pop_analysis_k_comparison.ipynb`: older large notebook comparing several `k` values in two-population runs.

If you add a new paired notebook, keep the `.py` and `.ipynb` files together and
prefer committing the `.py` as the canonical review surface.

Downloading Simulation Outputs and Videos
-----------------------------------------

Generated simulation files and rendered videos are stored on the lab machine:

- host: `faulkner.tau.ac.il`
- user: `naomi`
- project path: `~/Documents/maya/diffusion_project`

Relevant remote directories:

- `~/Documents/maya/diffusion_project/data/`: previous simulation outputs
- `~/Documents/maya/diffusion_project/videos/`: rendered `.mp4` files

Typical usage:

- Copy the whole `data/` or `videos/` directory to your computer if you want a full local mirror.
- Or copy only the specific files you need for a given analysis notebook.
- The notebook cells usually show the exact `data/...` or `videos/...` path used by that analysis, so you can search by the filename referenced in the cell and transfer only those files.

Example `scp` commands:

```bash
scp -r naomi@faulkner.tau.ac.il:~/Documents/maya/diffusion_project/data ./data
scp -r naomi@faulkner.tau.ac.il:~/Documents/maya/diffusion_project/videos ./videos
```

To copy a single file instead:

```bash
scp naomi@faulkner.tau.ac.il:~/Documents/maya/diffusion_project/videos/<your_video>.mp4 ./videos/
scp naomi@faulkner.tau.ac.il:~/Documents/maya/diffusion_project/data/<your_run>.h5 ./data/
```

Testing
-------

Run the test suite with:

```bash
uv run pytest
```

Notes
-----

- `k = 0` uses the logarithmic potential-energy form.
- Two-population mode currently supports exactly two populations, not an arbitrary
  number of species.
- If `charge_values` and `charge_counts` are omitted, the run is treated as a
  one-population simulation with uniform unit charge.
- Output is written to NPZ by default unless `--out-format` or an HDF5 output
  path is used.
