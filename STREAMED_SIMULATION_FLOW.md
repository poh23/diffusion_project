# Streamed Simulation Flow (Technical)

This document describes the runtime flow for a **streamed simulation** (HDF5 batching/resume path).

Primary code paths:
- `src/diffusion_sim/cli.py`
- `src/diffusion_sim/simulation/__init__.py`
- `src/diffusion_sim/simulation/streaming.py`
- `src/diffusion_sim/simulation/helpers.py`
- `src/diffusion_sim/io/h5_batch.py`
- `src/diffusion_sim/integrators/rk23.py`
- `src/diffusion_sim/integrators/dop853.py`

---

## 1) Entry and Mode Selection

1. CLI parses args and builds `SimulationConfig`:
   - `src/diffusion_sim/cli.py`
2. CLI computes output path (`--out`, `--resume-from`, or auto-generated default).
3. CLI calls:
   - `run_simulation(**config.__dict__, out_path=out_path)`
4. `run_simulation` validates config and decides streaming mode:
   - `is_streaming_mode(config)` in `src/diffusion_sim/simulation/helpers.py`
5. If streaming mode is true, it dispatches to:
   - `run_streaming_simulation(config, out_path=out_path)` in `src/diffusion_sim/simulation/streaming.py`

Streaming mode is activated if any of these are set:
- `batch_every`
- `target_batch_mb`
- `max_wall_time`
- `resume_from`

---

## 2) Output Path + Config Fingerprint

`run_streaming_simulation` starts with:

1. `_resolve_stream_out_path(...)`
   - If `resume_from` is set, that path is authoritative.
   - If no `resume_from` and no explicit `out_path`, raises `ValueError`.
2. `config_hash = config_fingerprint(config)`
   - Hash of a subset of simulation-defining fields.
   - Used to detect resume mismatch.

Why hash exists:
- On resume, saved `config_hash` in HDF5 attrs is compared to current one.
- Mismatch raises unless `resume_force=True`.

---

## 3) State Initialization (Resume vs Fresh)

`_initialize_stream_state(...)` chooses one of two branches.

### Resume branch (`_initialize_stream_from_resume`)

1. Open existing HDF5 file.
2. Read `h5.attrs["config_hash"]` and optional `charges` dataset.
3. Validate config hash (unless `resume_force`).
4. Load resume state via `load_h5_resume_state(path)`:
   - Preferred source: `/resume` group (`t_current`, `y_current`, `rng_state`, `completed`).
   - Fallback: last frame from `times/positions`.
5. Early-completed short-circuit:
   - If run already completed to requested horizon, return a minimal result immediately (no integration).

### Fresh branch (`_initialize_stream_from_fresh`)

1. Initialize positions (`init_positions_jittered_disk`) from `seed`.
2. Initialize charges (`init_charges`) from `charge_values/charge_counts`, or ones if uncharged.

### Common post-init checks

1. Resolve charges with `_resolve_stream_charges(...)` (guard against missing charges in charged resume).
2. Compute `t_end = t0 + t_duration`.
3. Early short-circuit if `t_start_sim >= t_end`.

Returned runtime state keys:
- `resumed`
- `rng_state`
- `t_start_sim`
- `t_end`
- `r0`
- `charges`

---

## 4) Open HDF5 Writer + Dataset Contracts

1. Compute chunk length via `_compute_stream_chunk_len(config)`:
   - Default `1024`
   - If `target_batch_mb` is set:
     - `bytes_per_frame = (2 * n_particles * 8) + (3 * 8)`
     - `chunk_len = floor(target_bytes / bytes_per_frame)`, at least 1
2. Open writer via `_open_stream_writer(...)` -> `open_h5_batch(...)`

`open_h5_batch(...)` guarantees these datasets exist:
- `positions`: `(T, N, 2)` growable
- `times`: `(T,)` growable
- `energy`: `(T,)` growable
- `std`: `(T,)` growable
- `final_positions`: `(N, 2)` fixed

Streaming layer also ensures `charges` dataset exists and matches (or uses `resume_force` override).

---

## 5) Runtime Coordination Objects

Initialized in `run_streaming_simulation`:

- `last_state = {"t": t_start_sim, "r": r0.copy()}`
- `last_stats = {}`
- `wall_start = perf_counter()`

Coordinator helpers:

1. `_BatchAccumulator`
   - Buffers `(time, position, energy, std)` records.
   - Flush trigger policy:
     - `t_curr >= next_batch_t` (if `batch_every`)
     - buffered frames >= `target_frames` (if `target_batch_mb`)
   - On flush:
     - appends buffered chunk via `append_h5_batch(...)`
     - calls `_FlushCallback`

2. `_FlushCallback`
   - Writes resume snapshot (`write_h5_resume_state`)
   - Writes metadata snapshot (`write_h5_meta(meta, config_hash=...)`)
   - `h5.flush()`

3. `tqdm` progress bar
   - Total is simulation duration.
   - Initial offset supports resumed runs.

4. `_StopCondition`
   - Stops integration if wall-clock exceeds `max_wall_time`.
   - Tracks `stopped_early`.

---

## 6) Solver Branches

Dispatch via `_run_stream_method(...)`.

### RK23 branch (`_run_rk23_stream`)

1. Create progress callback (`_RK23ProgressCallback`):
   - Updates progress bar by accepted-step delta time.
   - Updates `last_state`.
2. Create record hook (`_RK23RecordHook`):
   - Pushes each recorded sample into `_BatchAccumulator`.
3. Compute `sample_count` from `save_every` (`compute_sample_count`).
4. Run `run_rk23_dynamic(...)` with:
   - `record=False` (do not return large in-memory arrays)
   - `record_hook` active (stream records into HDF5 buffer)
   - `stop_condition` active
   - `diffusion_rng_state` set from resume when available
5. Return:
   - `r_final`
   - RK23 stats dict (`nfev`, `n_steps`, status/message, and optional diffusion RNG state)

### DOP853 branch (`_run_dop853_stream`)

1. Compute target step count from current time to `t_end`.
2. Apply `skip_first` if resumed and first sample is already present.
3. Create record hook (`_Dop853RecordHook`):
   - Updates `last_state`, progress bar, and accumulator with chunked arrays.
4. Adapt stop callback signature via `_Dop853StopAdapter`.
5. Run `run_dop853_chunked(...)` with `return_arrays=False`.
6. Return:
   - `r_final`
   - Empty stats dict

---

## 7) Finalization

`_finalize_stream_run(...)` performs deterministic shutdown:

1. Force final accumulator flush.
2. Compute completion flag:
   - `completed = (last_state["t"] >= t_end) and not stopped_early`
3. Persist final resume snapshot:
   - `write_h5_resume_state(..., completed=completed)`
4. Write `final_positions`.
5. Build final metadata and write:
   - `write_h5_meta(..., config_hash=...)`
6. Flush + close HDF5, close progress bar.
7. Return lightweight result:
   - empty arrays for histories
   - `final_positions`, `charges`, `meta`, `saved_path`

CLI behavior:
- If returned dict has `saved_path`, CLI exits without re-saving (streaming path already persisted to HDF5).

---

## 8) HDF5 Metadata + Resume Schema

From `src/diffusion_sim/io/h5_batch.py`:

- HDF5 root attrs:
  - `meta_json`: serialized metadata
  - `config_hash`: config fingerprint for resume safety

- `resume` group:
  - attrs:
    - `t_current`
    - `completed`
    - optional `n_steps`, `nfev`
    - optional serialized `rng_state`
  - dataset:
    - `y_current` (last position state)

---

## 9) Failure and Safety Semantics

- Missing output path in streaming mode -> immediate `ValueError`.
- Resume file not found -> `FileNotFoundError`.
- Config hash mismatch on resume -> `ValueError` unless `resume_force=True`.
- Charges mismatch in resume file -> `ValueError` unless `resume_force=True`.
- Charged resume without stored `charges` -> `ValueError`.
- Max wall time reached -> controlled early stop with persisted checkpoint state.

---

## 10) Reproducibility Notes (Streaming)

- Initial condition reproducibility:
  - Controlled by `seed`.
- Diffusion reproducibility (RK23):
  - Controlled by `diffusion_seed` on fresh runs.
  - On resume, diffusion RNG state is restored from stored `rng_state` in resume metadata (if present).

---

## 11) Minimal Pseudocode

```text
run_streaming_simulation(config, out_path):
  out_path <- resolve path
  config_hash <- fingerprint(config)
  state <- init from resume or fresh
  if early return condition: return saved_result

  h5 <- open writer + validate charges
  init accumulator, flush callback, progress, stop condition

  (r_final, method_stats) <- run RK23 or DOP853 branch
  merge method_stats into last_stats

  flush remaining buffers
  write final resume state + final_positions + meta_json + config_hash
  close h5
  return saved_result(saved_path, final_positions, meta)
```
