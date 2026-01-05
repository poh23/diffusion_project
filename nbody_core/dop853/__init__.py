import time
import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

from ..common import compute_velocity_overdamped, compute_energy_numba, compute_std_numba

__all__ = ["run_dop853_chunked"]


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
    t_start = time.perf_counter()
    show_progress = print_every_chunks is None or print_every_chunks > 0
    pbar = tqdm(total=steps, desc="dop853", unit="step", disable=not show_progress)

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

        if show_progress:
            pbar.update(m)

    r_final = positions[-1]
    if show_progress:
        pbar.close()
    return r_final, positions, energy, std, times
