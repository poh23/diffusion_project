import time
from dataclasses import replace

import numpy as np
from tqdm.auto import tqdm

from .config import DEFAULT_R_FLOOR, SimulationConfig, validate_config
from .init_conditions import init_positions_jittered_disk
from .integrators.rk2 import run_rk2_loop
from .integrators.rk4 import run_rk4_loop
from .integrators.rk23 import run_rk23_dynamic
from .integrators.dop853 import run_dop853_chunked
from .postprocess.density_voronoi import compute_density_and_radius_series

# ----------------------------
# Fixed-step (Numba) RK2/RK4 with progress via chunking
# ----------------------------

def _run_fixedstep_integrator(config: SimulationConfig, r0):
    return run_fixedstep_chunked(
        r0,
        config.k,
        config.v0,
        config.l,
        config.r_floor,
        config.dt,
        config.steps,
        config.t0,
        method=config.method,
        chunk_steps=config.chunk_steps,
        print_every_chunks=config.print_every_chunks,
    )


def run_fixedstep_chunked(
    r0,
    k,
    v0,
    l,
    r_floor,
    dt,
    steps,
    t0,
    method="rk4",
    chunk_steps=5000,
    print_every_chunks=1,
):
    """
    Runs rk4/rk2 in chunks and prints progress between chunks.
    """
    n = r0.shape[0]
    positions = np.empty((steps, n, 2), dtype=np.float64)
    times = np.empty(steps, dtype=np.float64)
    energy = np.empty(steps, dtype=np.float64)
    std = np.empty(steps, dtype=np.float64)

    r = r0.copy()
    t = t0
    idx = 0
    show_progress = print_every_chunks is None or print_every_chunks > 0
    pbar = tqdm(total=steps, desc=method, unit="step", disable=not show_progress)

    while idx < steps:
        m = min(chunk_steps, steps - idx)

        if method == "rk4":
            r_final, r_hist, pe_hist, std_hist, t_hist = run_rk4_loop(
                r, k, v0, l, r_floor, dt, m, t
            )
        elif method == "rk2":
            r_final, r_hist, pe_hist, std_hist, t_hist = run_rk2_loop(
                r, k, v0, l, r_floor, dt, m, t
            )
        else:
            raise ValueError("method must be 'rk4' or 'rk2'")

        positions[idx : idx + m] = r_hist
        times[idx : idx + m] = t_hist
        energy[idx : idx + m] = pe_hist
        std[idx : idx + m] = std_hist

        r = r_final
        t += m * dt
        idx += m

        if show_progress:
            pbar.update(m)

    if show_progress:
        pbar.close()
    return r, positions, energy, std, times


def _auto_chunk_steps(steps: int, *, target_updates: int = 100, min_chunk: int = 100, max_chunk: int = 5000) -> int:
    """
    Choose a chunk size to get roughly target_updates progress updates without making chunks too small.
    """
    if steps <= 0:
        return min_chunk
    chunk = max(min_chunk, steps // max(1, target_updates))
    return int(max(1, min(chunk, max_chunk)))


# ----------------------------
# Public API
# ----------------------------

def run_simulation(
    *,
    n_particles=10,
    k=1.0,
    v0=1.0,
    l=1.0,
    dt=0.01,
    steps=200,
    method="rk4",  # "rk4" or "rk2" or "rk23" or "dop853"
    seed=0,
    r_floor=DEFAULT_R_FLOOR,
    init_radius=1.0,
    t0=0.0,
    # progress control
    chunk_steps=5000,
    print_every_chunks=1,
    # dop853 tolerances
    rtol=1e-6,
    atol=1e-6,
    # rk23 adaptive settings
    first_step=None,
    max_step_global=np.inf,
    eta=0.05,
    recompute_every=10,
    rk23_sample_dt=None,
    rk23_sample_count=None,
    rk23_status_every_steps=50,
    rk23_status_every_sec=1.0,
    diffusion=False,
    diffusion_coeff=0.0,
    diffusion_seed=None,
    diffusion_noise_var=1.0,
    compute_density=False,
    density_print_every=None,
    density_stride=1,
    out_format="npz",
):
    config = SimulationConfig(
        n_particles=n_particles,
        k=k,
        v0=v0,
        l=l,
        r_floor=r_floor,
        init_radius=init_radius,
        dt=dt,
        steps=steps,
        t0=t0,
        method=method,
        seed=seed,
        chunk_steps=chunk_steps,
        print_every_chunks=print_every_chunks,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_step_global=max_step_global,
        eta=eta,
        recompute_every=recompute_every,
        rk23_sample_dt=rk23_sample_dt,
        rk23_sample_count=rk23_sample_count,
        rk23_status_every_steps=rk23_status_every_steps,
        rk23_status_every_sec=rk23_status_every_sec,
        diffusion=diffusion,
        diffusion_coeff=diffusion_coeff,
        diffusion_seed=diffusion_seed,
        diffusion_noise_var=diffusion_noise_var,
        compute_density=compute_density,
        density_print_every=density_print_every,
        density_stride=density_stride,
        out_format=out_format,
    )

    validate_config(config)

    r0 = init_positions_jittered_disk(
        config.n_particles,
        radius=config.init_radius,
        seed=config.seed,
    )

    # If chunk_steps <= 0, auto-select for nicer tqdm updates
    if config.chunk_steps <= 0:
        if config.method == "dop853":
            config = replace(
                config,
                chunk_steps=_auto_chunk_steps(config.steps, target_updates=80, min_chunk=500, max_chunk=10000),
            )
        else:
            config = replace(
                config,
                chunk_steps=_auto_chunk_steps(config.steps, target_updates=100, min_chunk=100, max_chunk=5000),
            )

    t_start = time.perf_counter()
    rk23_stats = None

    if config.method in ("rk4", "rk2"):
        r_final, r_hist, pe_hist, std_hist, t_hist = _run_fixedstep_integrator(config, r0)
    elif config.method == "rk23":
        t_span = (config.t0, config.t0 + config.steps * config.dt)
        show_progress = config.print_every_chunks is None or config.print_every_chunks > 0
        pbar = None
        last_progress = 0
        step_counter = 0
        last_step_t = config.t0
        last_status_wall = time.perf_counter()
        if show_progress:
            total = config.rk23_sample_count if config.rk23_sample_count is not None else None
            pbar = tqdm(total=total, desc="rk23", unit="step")

        def _rk23_progress(t, r, solver):
            nonlocal last_progress, step_counter, last_step_t, last_status_wall
            if pbar is None:
                return
            step_counter += 1
            step_dt = t - last_step_t
            last_step_t = t
            if config.rk23_sample_dt is not None and config.rk23_sample_dt > 0.0:
                current = int(np.floor((t - config.t0) / config.rk23_sample_dt)) + 1
                if config.rk23_sample_count is not None:
                    current = min(current, config.rk23_sample_count)
                delta = current - last_progress
                if delta > 0:
                    pbar.update(delta)
                    last_progress = current
            else:
                pbar.update(1)

            status_steps = config.rk23_status_every_steps
            status_sec = config.rk23_status_every_sec
            if (status_steps and status_steps > 0) or (status_sec and status_sec > 0.0):
                now = time.perf_counter()
                if (status_steps and step_counter % status_steps == 0) or (
                    status_sec and (now - last_status_wall) >= status_sec
                ):
                    pbar.set_postfix(t=f"{t:.3g}", dt=f"{step_dt:.2e}")
                    last_status_wall = now

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
            sample_dt=config.rk23_sample_dt,
            sample_count=config.rk23_sample_count,
            callback=_rk23_progress if show_progress else None,
            record=True,
            method="RK23",
            return_stats=True,
            diffusion=config.diffusion,
            diffusion_coeff=config.diffusion_coeff,
            diffusion_seed=config.diffusion_seed,
            diffusion_noise_var=config.diffusion_noise_var,
        )
        if pbar is not None:
            pbar.close()
    elif config.method == "dop853":
        r_final, r_hist, pe_hist, std_hist, t_hist = run_dop853_chunked(
            r0,
            config.k,
            config.v0,
            config.l,
            config.r_floor,
            config.dt,
            config.steps,
            config.t0,
            rtol=config.rtol,
            atol=config.atol,
            chunk_steps=config.chunk_steps,
            print_every_chunks=config.print_every_chunks,
        )
    else:
        raise ValueError("method must be 'rk4', 'rk2', 'rk23', or 'dop853'")

    elapsed = time.perf_counter() - t_start

    density = None
    radii = None
    if config.compute_density:
        # stride > 1: compute densities on subsampled steps to save time
        if config.density_stride > 1:
            r_hist_for_density = r_hist[:: config.density_stride]
        else:
            r_hist_for_density = r_hist

        density_raw, radii_raw = compute_density_and_radius_series(
            r_hist_for_density,
            print_every=config.density_print_every,
        )

        if config.density_stride > 1:
            # expand back to full length with NaNs for skipped steps to keep alignment
            total_steps = len(r_hist)
            n = r_hist.shape[1]
            density = np.full((total_steps, n), np.nan, dtype=np.float64)
            radii = np.full((total_steps, n), np.nan, dtype=np.float64)
            density[:: config.density_stride] = density_raw
            radii[:: config.density_stride] = radii_raw
        else:
            density = density_raw
            radii = radii_raw

    meta = dict(
        n_particles=config.n_particles,
        k=config.k,
        v0=config.v0,
        l=config.l,
        dt=config.dt,
        steps=config.steps,
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
        rk23_sample_dt=config.rk23_sample_dt if config.method == "rk23" else None,
        rk23_sample_count=config.rk23_sample_count if config.method == "rk23" else None,
        rk23_status_every_steps=config.rk23_status_every_steps if config.method == "rk23" else None,
        rk23_status_every_sec=config.rk23_status_every_sec if config.method == "rk23" else None,
        diffusion=config.diffusion if config.method == "rk23" else None,
        diffusion_coeff=config.diffusion_coeff if config.method == "rk23" else None,
        diffusion_seed=config.diffusion_seed if config.method == "rk23" else None,
        diffusion_noise_var=config.diffusion_noise_var if config.method == "rk23" else None,
        rk23_stats=rk23_stats if config.method == "rk23" else None,
        chunk_steps=config.chunk_steps,
        density_method="voronoi_2d" if config.compute_density else None,
        density_stride=config.density_stride if config.compute_density else None,
        out_format=config.out_format,
        elapsed_sec=elapsed,
        steps_per_sec=(config.steps / elapsed) if elapsed > 0 else None,
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
