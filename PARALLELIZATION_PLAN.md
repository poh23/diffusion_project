# Practical Parallelization Plan (Simplified)

This repo already has the right “physics shape” for parallelization: every step mostly boils down to a big pairwise sum (all-to-all interactions) and a few simple reductions. The only tricky part is doing it without ever building an `N×N` matrix in memory.

## 1) What Runs the Simulation (Where to Look)

- Run command: `diffusion-sim ...` (defined in `pyproject.toml`).
- CLI entry: `src/diffusion_sim/cli.py` calls `src/diffusion_sim/simulation.py:run_simulation`.
- Two integrator paths:
- RK23 adaptive: `src/diffusion_sim/integrators/rk23.py:run_rk23_dynamic` (SciPy `RK23`/`RK45`).
- DOP853 adaptive: `src/diffusion_sim/integrators/dop853.py:run_dop853_chunked` (SciPy `solve_ivp(method="DOP853")`).
- The expensive physics kernel (the “force”):
- `src/diffusion_sim/forces.py:compute_velocity_overdamped` (Numba, parallel; O(N^2)).
- Expensive diagnostics:
- `src/diffusion_sim/metrics.py:compute_energy_numba` (Numba, parallel; O(N^2)).
- `src/diffusion_sim/metrics.py:compute_std_numba` (O(N)).

## 2) The Only Thing That Really Dominates Runtime

### Pairwise interaction sum (already Numba-parallel)

For each particle `i`, the code does:

```text
v[i] = sum over j!=i of  f(r[i] - r[j]) * q[i] * q[j]
```

This is O(N^2) work and it happens many times because adaptive solvers call the RHS a lot.

### Extra O(N^2) work you may be paying for without realizing it

- Energy: also O(N^2) per time it is computed.
- RK23 “min pairwise distance” cap: `_min_pairwise_distance_floor` in `src/diffusion_sim/integrators/rk23.py` is O(N^2) and runs every `recompute_every` accepted steps (when `k > -1`).

If you want “practical speedups”, the biggest wins are often from doing fewer O(N^2) things, not from making one O(N^2) kernel 5% faster.

## 3) Practical Parallelization Map (What’s Easy vs Hard)

1. Pairwise force accumulation (`compute_velocity_overdamped`)
- Already parallel on CPU using `numba.prange`.
- Parallel unit: particles `i` (each thread owns one `i` and loops over all `j`).
- No complicated synchronization needed (each thread writes a different `v[i]`).

2. Integrator bookkeeping (SciPy RK23 / solve_ivp)
- Mostly serial control logic on CPU.
- Calls the force kernel repeatedly; that’s where almost all time goes.

3. Metrics
- `compute_energy_numba` is parallel, but still O(N^2).
- `compute_std_numba` is cheap (O(N)).

4. IO
- Writing `.npz` or growing HDF5 datasets can dominate if you save too frequently.

## 4) “NN Force Matrix” Idea (Simple Guidance)

Thinking of interactions as an `N×N` matrix is a good mental model, but you usually should not build the matrix explicitly.

- Good: conceptual model, small-N debugging.
- Bad: memory blows up fast if you store `dx`, `dy`, `dist`, `factor`, etc.

Rule of thumb: keep the current streaming approach (compute contributions and immediately add into `v[i]`), or use tiling (compute a small block of pairs at a time and reduce immediately). Avoid allocating anything shaped `(N,N)`.

## 5) Practical Advice You Can Apply Without Becoming a GPU Engineer

### A) Reduce how often you do O(N^2) work

- If you don’t need energy at every saved frame, compute it less often (or in postprocessing).
- If RK23 is active and `k > -1`, the min-distance step cap can be expensive:
- Increase `recompute_every` if stability allows.
- If you can tolerate a slightly smaller global step, rely more on `max_step_global` and less on frequent O(N^2) min-distance recomputation.

### B) Reduce output cost

- Increase `--save-every` if you don’t need dense trajectories.
- Prefer streaming HDF5 (`--out ...h5` with `--batch-every` or `--target-batch-mb`) for long runs to avoid giant in-memory histories.

### C) Stay CPU-first unless you are ready to change the solver

Right now the solvers are SciPy-based (CPU). That means:

- Keeping the force on CPU (Numba) is the “easy practical path” and already parallel.
- Moving only the force to GPU while leaving the solver on CPU usually loses to CPU↔GPU transfer overhead, because the solver calls the RHS many times.

## 6) GPU Options (Only If You’re Willing to Simplify the Integrator)

If you want a GPU win that is realistic to implement, the usual practical approach is:

- Switch to a fixed-step loop (e.g., RK2/RK4) so the entire loop can live on GPU.
- Keep `r`, `charges`, and `v` on GPU for the whole run.
- Never build `(N,N)` arrays; use tiling/reduction.

Most approachable ecosystems for that:

- CuPy: closest to NumPy mental model.
- PyTorch: also fine for tensor ops; handy if you ever want learned interactions later.

If you want to keep adaptive SciPy RK23/DOP853, treat GPU as “long-term / hard” in this repo.

## 7) Validation Checklist (Practical)

- Small run, fixed seeds:
- Use `--seed` and (if diffusion) `--diffusion-seed`.
- Compare `final_positions` and a few snapshots between versions.
- Sanity checks:
- No NaNs/Infs in `positions`.
- Repulsive cases should not show sudden particle collapse at the start.
- Expectation management:
- If you parallelize differently (or later use GPU), don’t expect bitwise-identical results; accept small floating-point differences.
