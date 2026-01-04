import json
import time
from dataclasses import dataclass

import numpy as np

from nbody_core import (
    init_positions_jittered_disk,
    run_rk2_loop,
    run_rk2_loop_with_noise,
    run_rk4_loop,
    run_dop853_chunked,
)

DEFAULT_SOFTENING = 1e-1


@dataclass
class SimulationConfig:
    # system/physics
    n_particles: int = 10
    k: float = 1.0
    v0: float = 1.0
    l: float = 1.0
    softening: float = DEFAULT_SOFTENING
    # integration / time
    dt: float = 0.01
    steps: int = 200
    t0: float = 0.0
    method: str = "rk4"  # "rk4" or "rk2" or "dop853"
    # diffusion (rk2 only)
    D: float = 0.0
    var_chi: float = 1.0
    seed: int = 0
    # progress / chunking
    chunk_steps: int = 5000
    print_every_chunks: int = 1
    # dop853 tolerances
    rtol: float = 1e-6
    atol: float = 1e-6

# ----------------------------
# Progress printing
# ----------------------------
def _progress(i_done, total, t_start, prefix="Progress"):
    frac = i_done / total if total else 1.0
    elapsed = time.perf_counter() - t_start
    rate = i_done / elapsed if elapsed > 0 else 0.0
    eta = (total - i_done) / rate if rate > 0 else float("inf")

    pct = 100.0 * frac
    eta_str = "?" if eta == float("inf") else f"{eta:,.1f}s"
    print(
        f"{prefix}: {pct:6.2f}%  ({i_done}/{total})  "
        f"elapsed={elapsed:,.1f}s  ETA={eta_str}  rate={rate:,.1f} steps/s"
    )


# ----------------------------
# Fixed-step (Numba) RK2/RK4 with progress via chunking
# ----------------------------
def _run_fixedstep_integrator(config: SimulationConfig, r0):
    return run_fixedstep_chunked(
        r0,
        config.k,
        config.v0,
        config.l,
        config.softening,
        config.dt,
        config.steps,
        config.t0,
        method=config.method,
        D=config.D,
        var_chi=config.var_chi,
        seed=config.seed,
        chunk_steps=config.chunk_steps,
        print_every_chunks=config.print_every_chunks,
    )


def run_fixedstep_chunked(
    r0,
    k,
    v0,
    l,
    softening,
    dt,
    steps,
    t0,
    method="rk4",
    D=0.0,          # diffusion constant
    var_chi=1.0,    # Var(chi) per coordinate (usually 1)
    seed=0,
    chunk_steps=5000,
    print_every_chunks=1,
):
    """
    Runs rk4/rk2 in chunks and prints progress between chunks.

    NOTE on seeds:
    - seed controls reproducibility overall.
    - for rk2 random-walk, we pass a per-chunk seed = seed + idx so runs are reproducible.
    """
    n = r0.shape[0]
    positions = np.empty((steps, n, 2), dtype=np.float64)
    times = np.empty(steps, dtype=np.float64)
    energy = np.empty(steps, dtype=np.float64)
    std = np.empty(steps, dtype=np.float64)

    r = r0.copy()
    t = t0
    idx = 0
    chunk_idx = 0

    t_start = time.perf_counter()

    while idx < steps:
        m = min(chunk_steps, steps - idx)

        if method == "rk4":
            r_final, r_hist, pe_hist, std_hist, t_hist = run_rk4_loop(
                r, k, v0, l, softening, dt, m, t
            )
        elif method == "rk2":
            if D > 0.0:
                # dx = sqrt(2*D*dt) * chi, with Var(chi)=var_chi
                rng = np.random.default_rng((seed + idx) & 0xFFFFFFFF)

                chi = rng.standard_normal(size=(m, n, 2)).astype(np.float64)
                if var_chi != 1.0:
                    chi *= np.sqrt(var_chi)

                noise = (np.sqrt(2.0 * D * dt) * chi).astype(np.float64)

                r_final, r_hist, pe_hist, std_hist, t_hist = run_rk2_loop_with_noise(
                    r, k, v0, l, softening, dt, m, t, noise
                )
            else:
                # no diffusion
                seed_numba = (seed + idx) & 0xFFFFFFFF
                r_final, r_hist, pe_hist, std_hist, t_hist = run_rk2_loop(
                    r, k, v0, l, softening, dt, m, t,
                    random_walk_std=0.0, seed=seed_numba
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
        chunk_idx += 1

        if (chunk_idx % print_every_chunks == 0) or (idx == steps):
            _progress(idx, steps, t_start, prefix=f"{method} progress")

    return r, positions, energy, std, times


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
    method="rk4",  # "rk4" or "rk2" or "dop853"
    D=0.0,        # diffusion constant (used for rk2)
    var_chi=1.0,  # Var(chi) per coordinate
    seed=0,
    softening=DEFAULT_SOFTENING,
    t0=0.0,
    # progress control
    chunk_steps=5000,
    print_every_chunks=1,
    # dop853 tolerances
    rtol=1e-6,
    atol=1e-6,
):
    config = SimulationConfig(
        n_particles=n_particles,
        k=k,
        v0=v0,
        l=l,
        softening=softening,
        dt=dt,
        steps=steps,
        t0=t0,
        method=method,
        D=D,
        var_chi=var_chi,
        seed=seed,
        chunk_steps=chunk_steps,
        print_every_chunks=print_every_chunks,
        rtol=rtol,
        atol=atol,
    )

    r0 = init_positions_jittered_disk(config.n_particles, seed=config.seed)

    t_start = time.perf_counter()

    if config.method in ("rk4", "rk2"):
        r_final, r_hist, pe_hist, std_hist, t_hist = _run_fixedstep_integrator(config, r0)
    elif config.method == "dop853":
        r_final, r_hist, pe_hist, std_hist, t_hist = run_dop853_chunked(
            r0,
            config.k,
            config.v0,
            config.l,
            config.softening,
            config.dt,
            config.steps,
            config.t0,
            rtol=config.rtol,
            atol=config.atol,
            chunk_steps=config.chunk_steps,
            print_every_chunks=config.print_every_chunks,
        )
    else:
        raise ValueError("method must be 'rk4', 'rk2', or 'dop853'")

    elapsed = time.perf_counter() - t_start

    meta = dict(
        n_particles=config.n_particles,
        k=config.k,
        v0=config.v0,
        l=config.l,
        dt=config.dt,
        steps=config.steps,
        method=config.method,
        D=config.D if config.method == "rk2" else None,
        var_chi=config.var_chi if config.method == "rk2" else None,
        seed=config.seed,
        softening=config.softening,
        t0=config.t0,
        rtol=config.rtol if config.method == "dop853" else None,
        atol=config.atol if config.method == "dop853" else None,
        chunk_steps=config.chunk_steps,
        elapsed_sec=elapsed,
        steps_per_sec=(config.steps / elapsed) if elapsed > 0 else None,
    )

    return dict(
        positions=r_hist,
        times=t_hist,
        energy=pe_hist,
        std=std_hist,
        final_positions=r_final,
        meta=meta,
    )


def save_npz(out_path, sim_dict):
    meta_json = json.dumps(sim_dict["meta"])
    np.savez_compressed(
        out_path,
        positions=sim_dict["positions"],
        times=sim_dict["times"],
        energy=sim_dict["energy"],
        std=sim_dict["std"],
        final_positions=sim_dict["final_positions"],
        meta_json=np.array(meta_json, dtype=object),
    )
    print(f"Saved: {out_path}")
    print(f"Meta: {sim_dict['meta']}")


if __name__ == "__main__":
    sim = run_simulation(
        n_particles=100,
        k=3.0,
        v0=1.0,
        l=1.0,
        dt=1e-3,
        steps=500,
        softening=1e-1,
        method="rk2",
        seed=1,
        D=1.0,          # example diffusion constant
        var_chi=1.0,    # standard normal
        # progress
        chunk_steps=100000,          # prints more often for small steps; for big runs set 2000-5000
        print_every_chunks=1,
    )
    save_npz("data/31-12-2025_k3_rk2_w_diff/rk2_w_random_walk_std1.0_k3.0_N100_steps500_dt1e-3_softening1e-1_new.npz", sim)
