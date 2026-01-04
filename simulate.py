import json
import time
import numpy as np
from scipy.integrate import solve_ivp

from nbody_core import (
    init_positions_on_circle,
    init_positions_jittered_disk,
    run_rk2_loop,
    run_rk2_loop_with_noise,
    run_rk4_loop,
    compute_velocity_overdamped,
    compute_energy_numba,
    compute_std_numba,
)

DEFAULT_SOFTENING = 1e-1


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
# DOP853 (chunked for progress)
# ----------------------------
def run_dop853_chunked(
    r0,
    k,
    v0,
    l,
    softening,
    dt,
    steps,
    t0,
    rtol=1e-6,
    atol=1e-6,
    chunk_steps=5000,
    print_every_chunks=1,
):
    """
    Chunked DOP853 integration so we can print progress.
    We sample the solution on a dt grid (t_eval).
    """
    n = r0.shape[0]
    positions = np.empty((steps, n, 2), dtype=np.float64)
    times = np.empty(steps, dtype=np.float64)
    energy = np.empty(steps, dtype=np.float64)
    std = np.empty(steps, dtype=np.float64)

    def rhs(t, y):
        r = y.reshape((n, 2))
        vel = compute_velocity_overdamped(r, k, v0, l, softening)
        return vel.reshape(-1)

    y = r0.reshape(-1).copy()
    t = t0
    idx = 0
    chunk_idx = 0

    t_start = time.perf_counter()

    while idx < steps:
        m = min(chunk_steps, steps - idx)
        t_eval = t + dt * np.arange(m)
        t_span = (t_eval[0], t_eval[-1])

        sol = solve_ivp(
            fun=rhs,
            t_span=t_span,
            y0=y,
            t_eval=t_eval,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"DOP853 failed: {sol.message}")

        r_hist = sol.y.T.reshape((m, n, 2))
        t_hist = sol.t

        positions[idx : idx + m] = r_hist
        times[idx : idx + m] = t_hist

        # metrics
        for i in range(m):
            energy[idx + i] = compute_energy_numba(r_hist[i], k, v0, l, softening)
            std[idx + i] = compute_std_numba(r_hist[i])

        # prepare next chunk
        y = sol.y[:, -1].copy()
        t += m * dt
        idx += m
        chunk_idx += 1

        if (chunk_idx % print_every_chunks == 0) or (idx == steps):
            _progress(idx, steps, t_start, prefix="dop853 progress")

    r_final = positions[-1]
    return r_final, positions, energy, std, times


# ----------------------------
# Fixed-step (Numba) RK2/RK4 with progress via chunking
# ----------------------------
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
    r0 = init_positions_jittered_disk(n_particles, seed=seed)

    t_start = time.perf_counter()

    if method in ("rk4", "rk2"):
        r_final, r_hist, pe_hist, std_hist, t_hist = run_fixedstep_chunked(
            r0,
            k,
            v0,
            l,
            softening,
            dt,
            steps,
            t0,
            method=method,
            D=D,          # example diffusion constant
            var_chi=var_chi,
            seed=seed,
            chunk_steps=chunk_steps,
            print_every_chunks=print_every_chunks,
        )
    elif method == "dop853":
        r_final, r_hist, pe_hist, std_hist, t_hist = run_dop853_chunked(
            r0,
            k,
            v0,
            l,
            softening,
            dt,
            steps,
            t0,
            rtol=rtol,
            atol=atol,
            chunk_steps=chunk_steps,
            print_every_chunks=print_every_chunks,
        )
    else:
        raise ValueError("method must be 'rk4', 'rk2', or 'dop853'")

    elapsed = time.perf_counter() - t_start

    meta = dict(
        n_particles=n_particles,
        k=k,
        v0=v0,
        l=l,
        dt=dt,
        steps=steps,
        method=method,
        D=D if method == "rk2" else None,
        var_chi=var_chi if method == "rk2" else None,
        seed=seed,
        softening=softening,
        t0=t0,
        rtol=rtol if method == "dop853" else None,
        atol=atol if method == "dop853" else None,
        chunk_steps=chunk_steps,
        elapsed_sec=elapsed,
        steps_per_sec=(steps / elapsed) if elapsed > 0 else None,
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
        method="rk2",
        seed=1,
        D=1.0,          # example diffusion constant
        var_chi=1.0,    # standard normal
        # progress
        chunk_steps=100000,          # prints more often for small steps; for big runs set 2000-5000
        print_every_chunks=1,
    )
    save_npz("data/31-12-2025_k3_rk2_w_diff/rk2_w_random_walk_std1.0_k3.0_N100_steps500_dt1e-3_softening1e-1_new.npz", sim)