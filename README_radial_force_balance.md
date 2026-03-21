# Radial Force-Balance Analysis (k=0)

Script: `analysis/plot_radial_force_balance.py`

Purpose:
- Computes empirical species-resolved radial interaction force profiles in self-similar coordinates.
- Plots:
  - `f1(r) = <e_r · F>` for species 1
  - `f2(r) = <e_r · F>` for species 2
  - reference line `beta*r` with `beta=1/2`
- Saves binned data to CSV.

## Inputs expected
- Simulation output file (`.h5`, `.hdf5`, or `.npz`) containing:
  - `positions` with shape `(T, N, 2)`
  - `times` with shape `(T,)`
  - `charges` with shape `(N,)`
  - metadata containing at least `k` (or pass `--k`)

## Default behavior
- Uses **last 20 frames** (`--last-n-frames 20`) for late-time averaging.
- Assumes `k=0` and errors otherwise.
- In `--use-rescaled-coords auto` mode, positions are treated as physical coordinates unless metadata explicitly indicates rescaled coordinates.

## Example commands

Late-time average over the last 50 frames:

```bash
uv run python analysis/plot_radial_force_balance.py \
  --input data/20260310/k0/your_run.h5 \
  --last-n-frames 50 \
  --bins 80 \
  --output graphs/radial_force_balance.png \
  --csv-output graphs/radial_force_balance.csv \
  --use-rescaled-coords auto
```

Single frame analysis:

```bash
uv run python analysis/plot_radial_force_balance.py \
  --input data/20260310/k0/your_run.h5 \
  --frame -1 \
  --bins 60 \
  --use-rescaled-coords no
```

If your file already stores self-similar coordinates `y`, force this mode:

```bash
uv run python analysis/plot_radial_force_balance.py \
  --input data/20260310/k0/your_run.h5 \
  --last-n-frames 30 \
  --use-rescaled-coords yes
```
