import time
from dataclasses import replace

import numpy as np
from tqdm.auto import tqdm

from ..config import SimulationConfig
from ..init_conditions import init_positions_jittered_disk
from ..integrators.dop853 import run_dop853_chunked
from ..integrators.rk23 import run_rk23_dynamic
from .helpers import (
    auto_chunk_steps,
    build_nonstream_meta,
    compute_sample_count,
    init_charges,
)


class _RK23ProgressCallback:
    def __init__(self, *, pbar, t_start):
        self.pbar = pbar
        self.last_step_t = t_start

    def __call__(self, t, r, solver):
        step_dt = t - self.last_step_t
        if step_dt > 0.0:
            self.pbar.update(step_dt)
        self.last_step_t = t


def run_nonstream_simulation(config: SimulationConfig) -> dict:
    rng = np.random.default_rng(config.seed)
    r0 = init_positions_jittered_disk(
        config.n_particles,
        radius=config.init_radius,
        rng=rng,
    )
    charges = init_charges(config.n_particles, config.charge_values, config.charge_counts, rng)

    sample_count = compute_sample_count(config.t_duration, config.save_every)

    # If chunk_steps <= 0, auto-select for nicer tqdm updates
    if config.chunk_steps <= 0 and sample_count is not None:
        target = 80 if config.method == "dop853" else 100
        config = replace(
            config,
            chunk_steps=auto_chunk_steps(sample_count, target_updates=target, min_chunk=100, max_chunk=10000),
        )

    t_start = time.perf_counter()
    rk23_stats = None

    if config.method == "rk23":
        t_span = (config.t0, config.t0 + config.t_duration)
        pbar = tqdm(total=config.t_duration, desc="rk23", unit="t")
        progress_callback = _RK23ProgressCallback(pbar=pbar, t_start=config.t0)

        r_final, r_hist, pe_hist, std_hist, t_hist, rk23_stats = run_rk23_dynamic(
            r0,
            config.k,
            config.v0,
            config.l,
            config.r_floor,
            charges=charges,
            t_span=t_span,
            rtol=config.rtol,
            atol=config.atol,
            first_step=config.first_step,
            max_step_global=config.max_step_global,
            eta=config.eta,
            recompute_every=config.recompute_every,
            sample_dt=config.save_every,
            sample_count=sample_count,
            sample_t0=config.t0,
            callback=progress_callback,
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
            charges=charges,
            dt=dt,
            steps=steps,
            t0=config.t0,
            rtol=config.rtol,
            atol=config.atol,
            chunk_steps=config.chunk_steps,
        )
    else:
        raise ValueError("method must be 'rk23' or 'dop853'")

    elapsed = time.perf_counter() - t_start

    meta = build_nonstream_meta(
        config,
        elapsed_sec=elapsed,
        rk23_stats=rk23_stats,
        t_hist_len=len(t_hist),
    )

    return dict(
        positions=r_hist,
        times=t_hist,
        energy=pe_hist,
        std=std_hist,
        final_positions=r_final,
        charges=charges,
        density=None,
        radii=None,
        meta=meta,
    )
