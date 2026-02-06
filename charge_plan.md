# Charge Populations Plan

Goal: add two charge populations with configurable magnitudes and ratios, update forces/energy, initialization, storage, and plotting.

## Plan

1) Configuration + data model
- Add config fields:
  - `charge_values` (e.g., `[q1, q2]`)
  - `charge_ratio` (e.g., fraction of population 1, or `[n1, n2]`)
  - `charge_seed` (optional, for randomized assignment if needed)
- Store charges in the simulation output:
  - `meta["charges"]` for per-particle array
  - Persist in NPZ/HDF5 as `charges` dataset/array

2) Initialization (mixed populations)
- Extend initialization to create a `charges` array alongside positions:
  - Decide how to split counts (deterministic vs stochastic according to ratio)
  - Shuffle assignments to ensure spatial mixing (same positions generator)
- Ensure deterministic behavior with `seed`/`charge_seed`.

3) Force + energy updates
- Modify overdamped velocity calculation:
  - Use Coulomb-like scaling: each pair contributes a factor `q_i * q_j`
  - Force form (schematic): `v_i = sum_j coupling * (q_i * q_j) * r_ij / (|r_ij|^(k+2))`
- Modify energy calculation accordingly.
- Keep r_floor handling and singularity protection consistent.

4) Plotting / visualization
- Add charge-aware coloring:
  - Assign a fixed color per charge value.
  - Update scatter/animation helpers to use color map by charge.
- Ensure legends reflect charge populations.

5) Tests / documentation
- Add simple tests:
  - Charges array length matches N.
  - Two populations created in expected ratio.
  - Energy/force respects charges (sanity check).
- Update README with new config keys and example.

## Open Questions

Resolved:

1) Force law: use Coulomb-like scaling `q_i * q_j`.
2) Ratios: explicit counts `[n1, n2]`.
3) Charges: keep both positive for now.
4) Initialization mixing: randomized with the same seed as positions.
5) Output: store `charges` array in both NPZ and HDF5.
6) Plotting colors: fixed palette (e.g., blue/red).
