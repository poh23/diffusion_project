import hashlib
import json
import time
from pathlib import Path
from dataclasses import replace

import numpy as np
from tqdm.auto import tqdm
import h5py

from .config import DEFAULT_R_FLOOR, SimulationConfig, validate_config
from .init_conditions import init_positions_jittered_disk
from .integrators.rk23 import run_rk23_dynamic
from .integrators.dop853 import run_dop853_chunked
from .io.h5_batch import (
    append_h5_batch,
    load_h5_resume_state,
    open_h5_batch,
    write_h5_meta,
    write_h5_resume_state,
)


def _auto_chunk_steps(steps: int, *, target_updates: int = 100, min_chunk: int = 100, max_chunk: int = 5000) -> int:
    """
    Choose a chunk size to get roughly target_updates progress updates without making chunks too small.
    """
    if steps <= 0:
        return min_chunk
    chunk = max(min_chunk, steps // max(1, target_updates))
    return int(max(1, min(chunk, max_chunk)))


def _config_fingerprint(config: SimulationConfig) -> str:
    payload = dict(
        n_particles=config.n_particles,
        k=config.k,
        v0=config.v0,
        l=config.l,
        r_floor=config.r_floor,
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
        first_step=config.first_step,
        max_step_global=config.max_step_global,
        eta=config.eta,
        recompute_every=config.recompute_every,
        diffusion=config.diffusion,
        diffusion_coeff=config.diffusion_coeff,
        diffusion_noise_var=config.diffusion_noise_var,
        save_every=config.save_every,
        t0=config.t0,
    )
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        bytes_per_frame = (2 * n_particles * 8) + (3 * 8)
        self.target_frames = None
        if target_batch_mb is not None:
            target_bytes = target_batch_mb * 1024 * 1024
            self.target_frames = max(1, int(target_bytes // bytes_per_frame))
        self._times = []
        self._positions = []
        self._energy = []
        self._std = []

    def append(self, t, r, energy, std):
        self._times.append(float(t))
        self._positions.append(r.copy())
        self._energy.append(float(energy))
        self._std.append(float(std))
        if self._should_flush(float(t)):
            self.flush(float(t))

    def append_batch(self, times, positions, energy, std):
        if len(times) == 0:
            return
        self._times.extend([float(t) for t in times])
        self._positions.extend([p.copy() for p in positions])
        self._energy.extend([float(e) for e in energy])
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
        std = np.asarray(self._std, dtype=np.float64)
        append_h5_batch(self.writer, positions, times, energy, std)
        if self.on_flush is not None:
            self.on_flush(t_curr)
        self._times.clear()
        self._positions.clear()
        self._energy.clear()
        self._std.clear()
        if self.next_batch_t is not None:
            while self.next_batch_t <= t_curr:
                self.next_batch_t += self.batch_every


# ----------------------------
# Public API
# ----------------------------

def run_simulation(
    *,
    n_particles=10,
    k=1.0,
    v0=1.0,
    l=1.0,
    method="rk23",  # "rk23" or "dop853"
    seed=0,
    r_floor=DEFAULT_R_FLOOR,
    init_radius=1.0,
    t0=0.0,
    t_duration=1.0,
    save_every=None,
    # progress control
    chunk_steps=5000,
    # batching / resume
    batch_every=None,
    target_batch_mb=None,
    max_wall_time=None,
    resume_from=None,
    resume_force=False,
    # dop853 tolerances
    rtol=1e-6,
    atol=1e-6,
    # rk23 adaptive settings
    first_step=None,
    max_step_global=np.inf,
    eta=0.05,
    recompute_every=10,
    diffusion=False,
    diffusion_coeff=0.0,
    diffusion_seed=None,
    diffusion_noise_var=1.0,
    out_format="npz",
    out_path=None,
):
    config = SimulationConfig(
        n_particles=n_particles,
        k=k,
        v0=v0,
        l=l,
        r_floor=r_floor,
        init_radius=init_radius,
        t0=t0,
        t_duration=t_duration,
        save_every=save_every,
        method=method,
        seed=seed,
        chunk_steps=chunk_steps,
        batch_every=batch_every,
        target_batch_mb=target_batch_mb,
        max_wall_time=max_wall_time,
        resume_from=resume_from,
        resume_force=resume_force,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_step_global=max_step_global,
        eta=eta,
        recompute_every=recompute_every,
        diffusion=diffusion,
        diffusion_coeff=diffusion_coeff,
        diffusion_seed=diffusion_seed,
        diffusion_noise_var=diffusion_noise_var,
        out_format=out_format,
    )

    validate_config(config)

    streaming = (
        config.batch_every is not None
        or config.target_batch_mb is not None
        or config.max_wall_time is not None
        or config.resume_from is not None
    )

    if streaming:
        if out_path is None and config.resume_from is None:
            raise ValueError("out_path is required for batching/resume runs.")

        if config.resume_from is not None:
            out_path = Path(config.resume_from)
        else:
            out_path = Path(out_path)

        config_hash = _config_fingerprint(config)
        resumed = False
        rng_state = None
        t_start_sim = config.t0

        if config.resume_from is not None:
            if not out_path.exists():
                raise FileNotFoundError(out_path)
            with h5py.File(out_path, "r") as h5:
                saved_hash = h5.attrs.get("config_hash")
                if isinstance(saved_hash, bytes):
                    saved_hash = saved_hash.decode("utf-8")
            if saved_hash is not None and saved_hash != config_hash and not config.resume_force:
                raise ValueError("Config mismatch for resume (use resume_force to override).")

            resume_state = load_h5_resume_state(out_path)
            if resume_state["completed"] and resume_state["t_current"] >= config.t0 + config.t_duration:
                return dict(
                    positions=np.empty((0, config.n_particles, 2), dtype=np.float64),
                    times=np.empty(0, dtype=np.float64),
                    energy=np.empty(0, dtype=np.float64),
                    std=np.empty(0, dtype=np.float64),
                    final_positions=resume_state["y_current"],
                    density=None,
                    radii=None,
                    meta=dict(
                        n_particles=config.n_particles,
                        k=config.k,
                        v0=config.v0,
                        l=config.l,
                        t_duration=config.t_duration,
                        save_every=config.save_every,
                        method=config.method,
                        seed=config.seed,
                        r_floor=config.r_floor,
                        init_radius=config.init_radius,
                        t0=config.t0,
                        rtol=config.rtol,
                        atol=config.atol,
                        diffusion=config.diffusion,
                        diffusion_coeff=config.diffusion_coeff,
                        diffusion_seed=config.diffusion_seed,
                        diffusion_noise_var=config.diffusion_noise_var,
                        batch_every=config.batch_every,
                        target_batch_mb=config.target_batch_mb,
                        max_wall_time=config.max_wall_time,
                        resume_from=str(out_path),
                        resumed=True,
                        completed=True,
                        out_format=config.out_format,
                        elapsed_sec=0.0,
                    ),
                    saved_path=str(out_path),
                )

            r0 = resume_state["y_current"]
            t_start_sim = resume_state["t_current"]
            rng_state = resume_state["rng_state"]
            resumed = True
        else:
            r0 = init_positions_jittered_disk(
                config.n_particles,
                radius=config.init_radius,
                seed=config.seed,
            )

        t_end = config.t0 + config.t_duration
        if t_start_sim >= t_end:
            return dict(
                positions=np.empty((0, config.n_particles, 2), dtype=np.float64),
                times=np.empty(0, dtype=np.float64),
                energy=np.empty(0, dtype=np.float64),
                std=np.empty(0, dtype=np.float64),
                final_positions=r0,
                density=None,
                radii=None,
                meta=dict(
                    n_particles=config.n_particles,
                    k=config.k,
                    v0=config.v0,
                    l=config.l,
                    t_duration=config.t_duration,
                    save_every=config.save_every,
                    method=config.method,
                    seed=config.seed,
                    r_floor=config.r_floor,
                    init_radius=config.init_radius,
                    t0=config.t0,
                    rtol=config.rtol,
                    atol=config.atol,
                    diffusion=config.diffusion,
                    diffusion_coeff=config.diffusion_coeff,
                    diffusion_seed=config.diffusion_seed,
                    diffusion_noise_var=config.diffusion_noise_var,
                    batch_every=config.batch_every,
                    target_batch_mb=config.target_batch_mb,
                    max_wall_time=config.max_wall_time,
                    resume_from=str(out_path),
                    resumed=resumed,
                    completed=True,
                    out_format=config.out_format,
                    elapsed_sec=0.0,
                ),
                saved_path=str(out_path),
            )

        chunk_len = 1024
        if config.target_batch_mb is not None:
            bytes_per_frame = (2 * config.n_particles * 8) + (3 * 8)
            target_bytes = config.target_batch_mb * 1024 * 1024
            chunk_len = max(1, int(target_bytes // bytes_per_frame))

        h5 = open_h5_batch(out_path, config.n_particles, resume=resumed, chunk_len=chunk_len)

        last_state = dict(t=t_start_sim, r=r0.copy())
        last_stats = {}

        def _on_flush(t_curr):
            write_h5_resume_state(
                h5,
                t_current=last_state["t"],
                y_current=last_state["r"],
                rng_state=last_stats.get("rng_state"),
                stats=last_stats,
                completed=False,
            )
            h5.flush()

        accumulator = _BatchAccumulator(
            writer=h5,
            batch_every=config.batch_every,
            target_batch_mb=config.target_batch_mb,
            n_particles=config.n_particles,
            t0=config.t0,
            start_t=t_start_sim,
            on_flush=_on_flush,
        )

        pbar = tqdm(
            total=config.t_duration,
            initial=max(0.0, t_start_sim - config.t0),
            desc=config.method,
            unit="t",
        )
        wall_start = time.perf_counter()
        stopped_early = False

        def _stop_condition(t, r, solver):
            nonlocal stopped_early
            if config.max_wall_time is None:
                return False
            if (time.perf_counter() - wall_start) >= config.max_wall_time:
                stopped_early = True
                return True
            return False

        if config.method == "rk23":
            last_step_t = t_start_sim

            def _rk23_progress(t, r, solver):
                nonlocal last_step_t
                step_dt = t - last_step_t
                if step_dt > 0.0:
                    pbar.update(step_dt)
                last_step_t = t
                last_state["t"] = t
                last_state["r"] = r.copy()

            def _record_hook(t, r, energy, std):
                accumulator.append(t, r, energy, std)

            sample_count = None
            if config.save_every is not None:
                sample_count = int(np.floor(config.t_duration / config.save_every)) + 1

            r_final, _, _, _, _, rk23_stats = run_rk23_dynamic(
                r0,
                config.k,
                config.v0,
                config.l,
                config.r_floor,
                (t_start_sim, t_end),
                rtol=config.rtol,
                atol=config.atol,
                first_step=config.first_step,
                max_step_global=config.max_step_global,
                eta=config.eta,
                recompute_every=config.recompute_every,
                sample_dt=config.save_every,
                sample_count=sample_count,
                sample_t0=config.t0,
                callback=_rk23_progress,
                record_hook=_record_hook,
                record=False,
                method="RK23",
                return_stats=True,
                diffusion=config.diffusion,
                diffusion_coeff=config.diffusion_coeff,
                diffusion_seed=config.diffusion_seed,
                diffusion_rng_state=rng_state,
                diffusion_noise_var=config.diffusion_noise_var,
                stop_condition=_stop_condition,
            )
            last_stats = rk23_stats or {}
        elif config.method == "dop853":
            dt = config.save_every
            steps_total = int(np.floor((t_end - t_start_sim) / dt)) + 1
            skip_first = resumed and steps_total > 0
            steps = max(0, steps_total - (1 if skip_first else 0))
            last_progress_t = t_start_sim

            def _record_hook(times, positions, energy, std):
                nonlocal last_progress_t
                if len(times) == 0:
                    return
                last_state["t"] = float(times[-1])
                last_state["r"] = positions[-1].copy()
                delta_t = last_state["t"] - last_progress_t
                if delta_t > 0.0:
                    pbar.update(delta_t)
                last_progress_t = last_state["t"]
                accumulator.append_batch(times, positions, energy, std)

            r_final, _, _, _, _ = run_dop853_chunked(
                r0,
                config.k,
                config.v0,
                config.l,
                config.r_floor,
                dt,
                steps,
                t_start_sim,
                rtol=config.rtol,
                atol=config.atol,
                chunk_steps=config.chunk_steps,
                record_hook=_record_hook,
                return_arrays=False,
                skip_first=skip_first,
                stop_condition=lambda t, y: _stop_condition(t, y, None),
            )
        else:
            raise ValueError("method must be 'rk23' or 'dop853'")

        accumulator.flush(last_state["t"])
        completed = (last_state["t"] >= t_end) and not stopped_early

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

        meta = dict(
            n_particles=config.n_particles,
            k=config.k,
            v0=config.v0,
            l=config.l,
            t_duration=config.t_duration,
            save_every=config.save_every,
            method=config.method,
            seed=config.seed,
            r_floor=config.r_floor,
            init_radius=config.init_radius,
            t0=config.t0,
            rtol=config.rtol,
            atol=config.atol,
            first_step=config.first_step if config.method == "rk23" else None,
            max_step_global=config.max_step_global if config.method == "rk23" else None,
            eta=config.eta if config.method == "rk23" else None,
            recompute_every=config.recompute_every if config.method == "rk23" else None,
            diffusion=config.diffusion if config.method == "rk23" else None,
            diffusion_coeff=config.diffusion_coeff if config.method == "rk23" else None,
            diffusion_seed=config.diffusion_seed if config.method == "rk23" else None,
            diffusion_noise_var=config.diffusion_noise_var if config.method == "rk23" else None,
            rk23_stats=last_stats if config.method == "rk23" else None,
            batch_every=config.batch_every,
            target_batch_mb=config.target_batch_mb,
            max_wall_time=config.max_wall_time,
            resume_from=str(out_path) if resumed else None,
            resumed=resumed,
            completed=completed,
            out_format=config.out_format,
            elapsed_sec=elapsed,
        )
        write_h5_meta(h5, meta, config_hash=config_hash)
        h5.flush()
        h5.close()
        pbar.close()

        print(f"Saved: {out_path}")
        print(f"Meta: {meta}")

        return dict(
            positions=np.empty((0, config.n_particles, 2), dtype=np.float64),
            times=np.empty(0, dtype=np.float64),
            energy=np.empty(0, dtype=np.float64),
            std=np.empty(0, dtype=np.float64),
            final_positions=r_final,
            density=None,
            radii=None,
            meta=meta,
            saved_path=str(out_path),
        )

    r0 = init_positions_jittered_disk(
        config.n_particles,
        radius=config.init_radius,
        seed=config.seed,
    )

    sample_count = None
    if config.save_every is not None:
        sample_count = int(np.floor(config.t_duration / config.save_every)) + 1

    # If chunk_steps <= 0, auto-select for nicer tqdm updates
    if config.chunk_steps <= 0 and sample_count is not None:
        target = 80 if config.method == "dop853" else 100
        config = replace(
            config,
            chunk_steps=_auto_chunk_steps(sample_count, target_updates=target, min_chunk=100, max_chunk=10000),
        )

    t_start = time.perf_counter()
    rk23_stats = None

    if config.method == "rk23":
        t_span = (config.t0, config.t0 + config.t_duration)
        pbar = tqdm(total=config.t_duration, desc="rk23", unit="t")
        last_step_t = config.t0

        def _rk23_progress(t, r, solver):
            nonlocal last_step_t
            step_dt = t - last_step_t
            if step_dt > 0.0:
                pbar.update(step_dt)
            last_step_t = t

        r_final, r_hist, pe_hist, std_hist, t_hist, rk23_stats = run_rk23_dynamic(
            r0,
            config.k,
            config.v0,
            config.l,
            config.r_floor,
            t_span,
            rtol=config.rtol,
            atol=config.atol,
            first_step=config.first_step,
            max_step_global=config.max_step_global,
            eta=config.eta,
            recompute_every=config.recompute_every,
            sample_dt=config.save_every,
            sample_count=sample_count,
            sample_t0=config.t0,
            callback=_rk23_progress,
            record=True,
            method="RK23",
            return_stats=True,
            diffusion=config.diffusion,
            diffusion_coeff=config.diffusion_coeff,
            diffusion_seed=config.diffusion_seed,
            diffusion_noise_var=config.diffusion_noise_var,
        )
        pbar.close()
    elif config.method == "dop853":
        dt = config.save_every
        steps = sample_count
        r_final, r_hist, pe_hist, std_hist, t_hist = run_dop853_chunked(
            r0,
            config.k,
            config.v0,
            config.l,
            config.r_floor,
            dt,
            steps,
            config.t0,
            rtol=config.rtol,
            atol=config.atol,
            chunk_steps=config.chunk_steps,
        )
    else:
        raise ValueError("method must be 'rk23' or 'dop853'")

    elapsed = time.perf_counter() - t_start

    density = None
    radii = None

    meta = dict(
        n_particles=config.n_particles,
        k=config.k,
        v0=config.v0,
        l=config.l,
        t_duration=config.t_duration,
        save_every=config.save_every,
        method=config.method,
        seed=config.seed,
        r_floor=config.r_floor,
        init_radius=config.init_radius,
        t0=config.t0,
        rtol=config.rtol if config.method in ("dop853", "rk23") else None,
        atol=config.atol if config.method in ("dop853", "rk23") else None,
        first_step=config.first_step if config.method == "rk23" else None,
        max_step_global=config.max_step_global if config.method == "rk23" else None,
        eta=config.eta if config.method == "rk23" else None,
        recompute_every=config.recompute_every if config.method == "rk23" else None,
        diffusion=config.diffusion if config.method == "rk23" else None,
        diffusion_coeff=config.diffusion_coeff if config.method == "rk23" else None,
        diffusion_seed=config.diffusion_seed if config.method == "rk23" else None,
        diffusion_noise_var=config.diffusion_noise_var if config.method == "rk23" else None,
        rk23_stats=rk23_stats if config.method == "rk23" else None,
        chunk_steps=config.chunk_steps,
        batch_every=config.batch_every,
        target_batch_mb=config.target_batch_mb,
        max_wall_time=config.max_wall_time,
        resume_from=None,
        resumed=False,
        completed=True,
        out_format=config.out_format,
        elapsed_sec=elapsed,
        steps_per_sec=(len(t_hist) / elapsed) if elapsed > 0 else None,
    )

    return dict(
        positions=r_hist,
        times=t_hist,
        energy=pe_hist,
        std=std_hist,
        final_positions=r_final,
        density=density,
        radii=radii,
        meta=meta,
    )
