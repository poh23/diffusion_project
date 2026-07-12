import time
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

from ..config import SimulationConfig
from ..integrators.dop853 import run_dop853_chunked
from ..integrators.rk23 import run_rk23_dynamic
from ..io.h5_batch import (
    append_h5_batch,
    load_h5_resume_state,
    open_h5_batch,
    write_h5_meta,
    write_h5_resume_state,
)
from .helpers import (
    build_saved_result,
    build_stream_meta,
    compute_sample_count,
    config_fingerprint,
    init_charges,
    init_fresh_positions_and_charges,
)


class _BatchAccumulator:
    def __init__(self, *, writer, batch_every, target_batch_mb, n_particles, t0, start_t, on_flush=None):
        self.writer = writer
        self.batch_every = batch_every
        self.target_batch_mb = target_batch_mb
        self.on_flush = on_flush
        self.next_batch_t = None
        if batch_every is not None:
            n_intervals = int(np.floor((start_t - t0) / batch_every))
            self.next_batch_t = t0 + (n_intervals + 1) * batch_every
        bytes_per_frame = (2 * n_particles * 8) + (6 * 8)
        self.target_frames = None
        if target_batch_mb is not None:
            target_bytes = target_batch_mb * 1024 * 1024
            self.target_frames = max(1, int(target_bytes // bytes_per_frame))
        self._times = []
        self._positions = []
        self._energy = []
        self._energy_aa = []
        self._energy_ab = []
        self._energy_bb = []
        self._std = []

    def append(self, t, r, energy, std, energy_aa, energy_ab, energy_bb):
        self._times.append(float(t))
        self._positions.append(r.copy())
        self._energy.append(float(energy))
        self._energy_aa.append(float(energy_aa))
        self._energy_ab.append(float(energy_ab))
        self._energy_bb.append(float(energy_bb))
        self._std.append(float(std))
        if self._should_flush(float(t)):
            self.flush(float(t))

    def append_batch(self, times, positions, energy, std, energy_aa, energy_ab, energy_bb):
        if len(times) == 0:
            return
        self._times.extend([float(t) for t in times])
        self._positions.extend([p.copy() for p in positions])
        self._energy.extend([float(e) for e in energy])
        self._energy_aa.extend([float(e) for e in energy_aa])
        self._energy_ab.extend([float(e) for e in energy_ab])
        self._energy_bb.extend([float(e) for e in energy_bb])
        self._std.extend([float(s) for s in std])
        last_t = float(times[-1])
        if self._should_flush(last_t):
            self.flush(last_t)

    def _should_flush(self, t_curr):
        if self.batch_every is not None and self.next_batch_t is not None:
            if t_curr >= self.next_batch_t:
                return True
        if self.target_frames is not None and len(self._times) >= self.target_frames:
            return True
        return False

    def flush(self, t_curr):
        if not self._times:
            return
        times = np.asarray(self._times, dtype=np.float64)
        positions = np.asarray(self._positions, dtype=np.float64)
        energy = np.asarray(self._energy, dtype=np.float64)
        energy_aa = np.asarray(self._energy_aa, dtype=np.float64)
        energy_ab = np.asarray(self._energy_ab, dtype=np.float64)
        energy_bb = np.asarray(self._energy_bb, dtype=np.float64)
        std = np.asarray(self._std, dtype=np.float64)
        append_h5_batch(
            self.writer,
            positions,
            times,
            energy,
            std,
            energy_aa=energy_aa,
            energy_ab=energy_ab,
            energy_bb=energy_bb,
        )
        if self.on_flush is not None:
            self.on_flush(t_curr)
        self._times.clear()
        self._positions.clear()
        self._energy.clear()
        self._energy_aa.clear()
        self._energy_ab.clear()
        self._energy_bb.clear()
        self._std.clear()
        if self.next_batch_t is not None:
            while self.next_batch_t <= t_curr:
                self.next_batch_t += self.batch_every


class _StopCondition:
    def __init__(self, *, max_wall_time, wall_start):
        self.max_wall_time = max_wall_time
        self.wall_start = wall_start
        self.stopped_early = False

    def __call__(self, t, r, solver):
        if self.max_wall_time is None:
            return False
        if (time.perf_counter() - self.wall_start) >= self.max_wall_time:
            self.stopped_early = True
            return True
        return False


class _Dop853StopAdapter:
    def __init__(self, stop_condition):
        self.stop_condition = stop_condition

    def __call__(self, t, y):
        return self.stop_condition(t, y, None)


class _FlushCallback:
    def __init__(
        self,
        *,
        h5,
        last_state,
        last_stats,
        wall_start,
        config,
        out_path,
        resumed,
        config_hash,
    ):
        self.h5 = h5
        self.last_state = last_state
        self.last_stats = last_stats
        self.wall_start = wall_start
        self.config = config
        self.out_path = out_path
        self.resumed = resumed
        self.config_hash = config_hash

    def __call__(self, t_curr):
        write_h5_resume_state(
            self.h5,
            t_current=self.last_state["t"],
            y_current=self.last_state["r"],
            rng_state=self.last_stats.get("rng_state"),
            stats=self.last_stats,
            completed=False,
        )
        elapsed = time.perf_counter() - self.wall_start
        meta_snapshot = build_stream_meta(
            self.config,
            out_path=self.out_path,
            elapsed_sec=elapsed,
            completed=False,
            rk23_stats=self.last_stats if self.config.method == "rk23" else None,
            resumed_flag=self.resumed,
        )
        write_h5_meta(self.h5, meta_snapshot, config_hash=self.config_hash)
        self.h5.flush()


class _RK23ProgressCallback:
    def __init__(self, *, pbar, last_state, t_start):
        self.pbar = pbar
        self.last_state = last_state
        self.last_step_t = t_start

    def __call__(self, t, r, solver):
        step_dt = t - self.last_step_t
        if step_dt > 0.0:
            self.pbar.update(step_dt)
        self.last_step_t = t
        self.last_state["t"] = t
        self.last_state["r"] = r.copy()


class _RK23RecordHook:
    def __init__(self, accumulator):
        self.accumulator = accumulator

    def __call__(self, t, r, energy, std, energy_aa, energy_ab, energy_bb):
        self.accumulator.append(t, r, energy, std, energy_aa, energy_ab, energy_bb)


class _Dop853RecordHook:
    def __init__(self, *, pbar, last_state, accumulator, t_start):
        self.pbar = pbar
        self.last_state = last_state
        self.accumulator = accumulator
        self.last_progress_t = t_start

    def __call__(self, times, positions, energy, std, energy_aa, energy_ab, energy_bb):
        if len(times) == 0:
            return
        self.last_state["t"] = float(times[-1])
        self.last_state["r"] = positions[-1].copy()
        delta_t = self.last_state["t"] - self.last_progress_t
        if delta_t > 0.0:
            self.pbar.update(delta_t)
        self.last_progress_t = self.last_state["t"]
        self.accumulator.append_batch(times, positions, energy, std, energy_aa, energy_ab, energy_bb)


def _resolve_stream_out_path(config: SimulationConfig, out_path) -> Path:
    if out_path is None and config.resume_from is None:
        raise ValueError("out_path is required for batching/resume runs.")
    if config.resume_from is not None:
        return Path(config.resume_from)
    return Path(out_path)


def _resolve_stream_charges(config: SimulationConfig, charges: np.ndarray | None) -> np.ndarray:
    if charges is not None:
        return charges
    if config.charge_values is None and config.charge_counts is None:
        return np.ones(config.n_particles, dtype=np.float64)
    raise ValueError("Resume file missing charges dataset for charged simulation.")


def _build_early_stream_result(
    config: SimulationConfig,
    *,
    out_path: Path,
    final_positions: np.ndarray,
    charges: np.ndarray,
    resumed_flag: bool,
) -> dict:
    meta = build_stream_meta(
        config,
        out_path=out_path,
        elapsed_sec=0.0,
        completed=True,
        rk23_stats=None,
        resumed_flag=resumed_flag,
    )
    return build_saved_result(
        config,
        out_path=out_path,
        final_positions=final_positions,
        charges=charges,
        meta=meta,
    )


def _initialize_stream_from_resume(
    config: SimulationConfig,
    *,
    out_path: Path,
    config_hash: str,
):
    if not out_path.exists():
        raise FileNotFoundError(out_path)
    with h5py.File(out_path, "r") as h5:
        saved_hash = h5.attrs.get("config_hash")
        if isinstance(saved_hash, bytes):
            saved_hash = saved_hash.decode("utf-8")
        charges = h5["charges"][()] if "charges" in h5 else None
    if saved_hash is not None and saved_hash != config_hash and not config.resume_force:
        raise ValueError("Config mismatch for resume (use resume_force to override).")

    resume_state = load_h5_resume_state(out_path)
    if charges is None and config.charge_values is not None and config.charge_counts is not None:
        raise ValueError("Resume file missing charges dataset for charged simulation.")

    if resume_state["completed"] and resume_state["t_current"] >= config.t0 + config.t_duration:
        final_charges = _resolve_stream_charges(config, charges)
        early = _build_early_stream_result(
            config,
            out_path=out_path,
            final_positions=resume_state["y_current"],
            charges=final_charges,
            resumed_flag=True,
        )
        return None, early

    state = dict(
        resumed=True,
        rng_state=resume_state["rng_state"],
        t_start_sim=resume_state["t_current"],
        r0=resume_state["y_current"],
        charges=charges,
    )
    return state, None


def _initialize_stream_from_fresh(config: SimulationConfig):
    rng = np.random.default_rng(config.seed)
    r0, charges = init_fresh_positions_and_charges(config, rng)
    return dict(
        resumed=False,
        rng_state=None,
        t_start_sim=config.t0,
        r0=r0,
        charges=charges,
    )


def _initialize_stream_state(config: SimulationConfig, *, out_path: Path, config_hash: str):
    if config.resume_from is not None:
        state, early = _initialize_stream_from_resume(
            config,
            out_path=out_path,
            config_hash=config_hash,
        )
        if early is not None:
            return None, early
    else:
        state = _initialize_stream_from_fresh(config)

    charges = _resolve_stream_charges(config, state["charges"])
    t_end = config.t0 + config.t_duration
    if state["t_start_sim"] >= t_end:
        early = _build_early_stream_result(
            config,
            out_path=out_path,
            final_positions=state["r0"],
            charges=charges,
            resumed_flag=state["resumed"],
        )
        return None, early

    state = dict(state)
    state["t_end"] = t_end
    state["charges"] = charges
    return state, None


def _compute_stream_chunk_len(config: SimulationConfig) -> int:
    chunk_len = 1024
    if config.target_batch_mb is not None:
        bytes_per_frame = (2 * config.n_particles * 8) + (6 * 8)
        target_bytes = config.target_batch_mb * 1024 * 1024
        chunk_len = max(1, int(target_bytes // bytes_per_frame))
    return chunk_len


def _open_stream_writer(
    config: SimulationConfig,
    *,
    out_path: Path,
    resumed: bool,
    charges: np.ndarray,
    chunk_len: int,
):
    h5 = open_h5_batch(out_path, config.n_particles, resume=resumed, chunk_len=chunk_len)
    if "charges" not in h5:
        h5.create_dataset("charges", data=charges, dtype=np.float64)
    else:
        stored = h5["charges"][()]
        if stored.shape != charges.shape or not np.allclose(stored, charges):
            if not config.resume_force:
                raise ValueError("Charges mismatch for resume (use resume_force to override).")
            h5["charges"][...] = charges
    return h5


def _run_rk23_stream(
    config: SimulationConfig,
    *,
    state: dict,
    pbar,
    accumulator,
    last_state: dict,
    stop_condition,
):
    progress_callback = _RK23ProgressCallback(
        pbar=pbar,
        last_state=last_state,
        t_start=state["t_start_sim"],
    )
    record_hook = _RK23RecordHook(accumulator)
    sample_count = compute_sample_count(config.t_duration, config.save_every)
    metric_every = int(config.save_every_steps) if config.save_every_steps is not None else 1

    r_final, _, _, _, _, _, _, _, rk23_stats = run_rk23_dynamic(
        state["r0"],
        config.k,
        config.v0,
        config.l,
        config.r_floor,
        charges=state["charges"],
        population_values=config.charge_values,
        t_span=(state["t_start_sim"], state["t_end"]),
        rtol=config.rtol,
        atol=config.atol,
        first_step=config.first_step,
        max_step_global=config.max_step_global,
        eta=config.eta,
        recompute_every=config.recompute_every,
        metric_every=metric_every,
        sample_dt=config.save_every,
        sample_count=sample_count,
        sample_t0=config.t0,
        interpolate_sampling=config.interpolate_sampling,
        callback=progress_callback,
        record_hook=record_hook,
        record=False,
        method="RK23",
        return_stats=True,
        diffusion=config.diffusion,
        diffusion_coeff=config.diffusion_coeff,
        diffusion_seed=config.diffusion_seed,
        diffusion_rng_state=state["rng_state"],
        diffusion_noise_var=config.diffusion_noise_var,
        stop_condition=stop_condition,
        external_potential=config.external_potential,
        external_potential_params=config.external_potential_params,
    )
    return r_final, (rk23_stats or {})


def _run_dop853_stream(
    config: SimulationConfig,
    *,
    state: dict,
    pbar,
    accumulator,
    last_state: dict,
    stop_condition,
):
    dt = config.save_every
    steps_total = int(np.floor((state["t_end"] - state["t_start_sim"]) / dt)) + 1
    skip_first = state["resumed"] and steps_total > 0
    steps = max(0, steps_total - (1 if skip_first else 0))
    record_hook = _Dop853RecordHook(
        pbar=pbar,
        last_state=last_state,
        accumulator=accumulator,
        t_start=state["t_start_sim"],
    )
    dop853_stop = _Dop853StopAdapter(stop_condition)

    r_final, _, _, _, _, _, _, _ = run_dop853_chunked(
        state["r0"],
        config.k,
        config.v0,
        config.l,
        config.r_floor,
        charges=state["charges"],
        dt=dt,
        steps=steps,
        t0=state["t_start_sim"],
        rtol=config.rtol,
        atol=config.atol,
        chunk_steps=config.chunk_steps,
        record_hook=record_hook,
        return_arrays=False,
        skip_first=skip_first,
        stop_condition=dop853_stop,
        population_values=config.charge_values,
        external_potential=config.external_potential,
        external_potential_params=config.external_potential_params,
    )
    return r_final, {}


def _run_stream_method(
    config: SimulationConfig,
    *,
    state: dict,
    pbar,
    accumulator,
    last_state: dict,
    stop_condition,
):
    if config.method == "rk23":
        return _run_rk23_stream(
            config,
            state=state,
            pbar=pbar,
            accumulator=accumulator,
            last_state=last_state,
            stop_condition=stop_condition,
        )
    if config.method == "dop853":
        return _run_dop853_stream(
            config,
            state=state,
            pbar=pbar,
            accumulator=accumulator,
            last_state=last_state,
            stop_condition=stop_condition,
        )
    raise ValueError("method must be 'rk23' or 'dop853'")


def _finalize_stream_run(
    config: SimulationConfig,
    *,
    out_path: Path,
    config_hash: str,
    state: dict,
    r_final: np.ndarray,
    last_state: dict,
    last_stats: dict,
    h5,
    accumulator,
    pbar,
    stop_condition,
    wall_start: float,
):
    accumulator.flush(last_state["t"])
    completed = (last_state["t"] >= state["t_end"]) and not stop_condition.stopped_early

    write_h5_resume_state(
        h5,
        t_current=last_state["t"],
        y_current=last_state["r"],
        rng_state=last_stats.get("rng_state"),
        stats=last_stats,
        completed=completed,
    )
    h5["final_positions"][...] = r_final
    elapsed = time.perf_counter() - wall_start

    meta = build_stream_meta(
        config,
        out_path=out_path,
        elapsed_sec=elapsed,
        completed=completed,
        rk23_stats=last_stats if config.method == "rk23" else None,
        resumed_flag=state["resumed"],
    )
    write_h5_meta(h5, meta, config_hash=config_hash)
    h5.flush()
    h5.close()
    pbar.close()

    print(f"Saved: {out_path}")
    print(f"Meta: {meta}")

    return build_saved_result(
        config,
        out_path=out_path,
        final_positions=r_final,
        charges=state["charges"],
        meta=meta,
    )


def run_streaming_simulation(config: SimulationConfig, *, out_path) -> dict:
    out_path = _resolve_stream_out_path(config, out_path)
    config_hash = config_fingerprint(config)
    state, early_result = _initialize_stream_state(config, out_path=out_path, config_hash=config_hash)
    if early_result is not None:
        return early_result

    h5 = _open_stream_writer(
        config,
        out_path=out_path,
        resumed=state["resumed"],
        charges=state["charges"],
        chunk_len=_compute_stream_chunk_len(config),
    )

    last_state = dict(t=state["t_start_sim"], r=state["r0"].copy())
    last_stats = {}
    wall_start = time.perf_counter()

    on_flush = _FlushCallback(
        h5=h5,
        last_state=last_state,
        last_stats=last_stats,
        wall_start=wall_start,
        config=config,
        out_path=out_path,
        resumed=state["resumed"],
        config_hash=config_hash,
    )
    accumulator = _BatchAccumulator(
        writer=h5,
        batch_every=config.batch_every,
        target_batch_mb=config.target_batch_mb,
        n_particles=config.n_particles,
        t0=config.t0,
        start_t=state["t_start_sim"],
        on_flush=on_flush,
    )

    pbar = tqdm(
        total=config.t_duration,
        initial=max(0.0, state["t_start_sim"] - config.t0),
        desc=config.method,
        unit="t",
    )
    stop_condition = _StopCondition(max_wall_time=config.max_wall_time, wall_start=wall_start)

    r_final, method_stats = _run_stream_method(
        config,
        state=state,
        pbar=pbar,
        accumulator=accumulator,
        last_state=last_state,
        stop_condition=stop_condition,
    )
    last_stats.update(method_stats or {})

    return _finalize_stream_run(
        config,
        out_path=out_path,
        config_hash=config_hash,
        state=state,
        r_final=r_final,
        last_state=last_state,
        last_stats=last_stats,
        h5=h5,
        accumulator=accumulator,
        pbar=pbar,
        stop_condition=stop_condition,
        wall_start=wall_start,
    )
