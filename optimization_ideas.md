# Optimization Ideas (Not Implemented)

This document lists potential optimizations to consider later. None of these are implemented yet.

## Algorithmic
- Add a short-range cutoff with a cell list / Verlet list when k is large enough that forces decay quickly.
- Use Barnes-Hut (quadtree) or FMM for long-range interactions to reduce O(n^2) to O(n log n) or better.
- Exploit symmetry in pairwise forces to cut work roughly in half (compute i<j once, accumulate both i and j).
- Introduce a split integration path: exact pairwise for near neighbors and approximate far-field for the rest.

## Numerical strategy
- Provide a metric stride (compute energy/std every N steps) to reduce per-step overhead.
- Offer an option to record positions at a lower cadence and interpolate for visualization.
- Use adaptive step control consistently across solvers (fixed-step runs could estimate stability limits).
- Consider a smooth softening kernel (Plummer or spline) to avoid discontinuities and allow larger dt.

## CPU performance
- Reduce temporary allocations in hot loops (reuse arrays, preallocate hist buffers).
- Switch inner loops to a structure-of-arrays layout to improve cache locality.
- Collapse loops to eliminate repeated sqrt calls where possible (use dist_sq with power laws carefully).
- For large N, experiment with Numba `parallel=False` vs `parallel=True` to match CPU topology.
- Batch compute energies using vectorized numpy if N is small (to reduce Numba overhead).

## GPU acceleration
- Port pairwise force and energy kernels to CuPy or numba.cuda.
- Keep data resident on the GPU across steps to avoid PCIe transfers.
- Use fused kernels (force + update) to minimize global memory traffic.
- Consider a GPU-based cell list for cutoff interactions.

## I/O and storage
- Store outputs in chunks (append per chunk) to avoid holding all steps in memory.
- Use float32 for history arrays when precision allows, and keep float64 for accumulation.
- Save metrics separately from trajectories for lighter analysis runs.

## Workflow
- Add a lightweight benchmarking harness to track steps/sec vs N and method.
- Include a profiling mode that reports time spent in force, metrics, and I/O.
- Make RNG seeding explicit and deterministic across chunks and integrators.
