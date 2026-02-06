import numpy as np
from scipy.integrate import solve_ivp
from tqdm.auto import tqdm

from ..forces import compute_velocity_overdamped
from ..metrics import compute_energy_numba, compute_std_numba

__all__ = ["run_dop853_chunked"]


def run_dop853_chunked(
    r0,
    k,
    v0,
    l,
    r_floor,
    dt,
    steps,
    t0,
    rtol=1e-6,
    atol=1e-6,
    chunk_steps=5000,
    record_hook=None,
    return_arrays=True,
    skip_first=False,
    stop_condition=None,
    charges=None,
):
    """
    Chunked DOP853 integration so we can print progress.
    We sample the solution on a dt grid (t_eval).
    """
    n = r0.shape[0]
    if charges is None:
        charges = np.ones(n, dtype=np.float64)
    if return_arrays:
        positions = np.empty((steps, n, 2), dtype=np.float64)
        times = np.empty(steps, dtype=np.float64)
        energy = np.empty(steps, dtype=np.float64)
        std = np.empty(steps, dtype=np.float64)
    else:
        positions = np.empty((0, n, 2), dtype=np.float64)
        times = np.empty(0, dtype=np.float64)
        energy = np.empty(0, dtype=np.float64)
        std = np.empty(0, dtype=np.float64)

    def rhs(t, y):
        r = y.reshape((n, 2))
        vel = compute_velocity_overdamped(r, k, v0, l, r_floor, charges)
        return vel.reshape(-1)

    y = r0.reshape(-1).copy()
    t = t0
    idx = 0
    pbar = tqdm(total=steps, desc="dop853", unit="step")

    while idx < steps:
        m = min(chunk_steps, steps - idx)
        if skip_first and idx == 0:
            t_eval = t + dt * np.arange(1, m + 1)
        else:
            t_eval = t + dt * np.arange(m)
        t_span = (t, t + m * dt)

        sol = solve_ivp(
            fun=rhs,
            t_span=t_span,
            y0=y,
            t_eval=t_eval,
            method="DOP853",
            rtol=rtol,
            atol=atol,
            dense_output=True,
        )

        if not sol.success:
            raise RuntimeError(f"DOP853 failed: {sol.message}")

        r_hist = sol.y.T.reshape((m, n, 2))
        t_hist = sol.t

        # metrics
        energy_chunk = np.empty(m, dtype=np.float64)
        std_chunk = np.empty(m, dtype=np.float64)
        for i in range(m):
            energy_chunk[i] = compute_energy_numba(r_hist[i], k, v0, l, r_floor, charges)
            std_chunk[i] = compute_std_numba(r_hist[i])

        if return_arrays:
            positions[idx : idx + m] = r_hist
            times[idx : idx + m] = t_hist
            energy[idx : idx + m] = energy_chunk
            std[idx : idx + m] = std_chunk

        if record_hook is not None:
            record_hook(t_hist, r_hist, energy_chunk, std_chunk)

        # prepare next chunk
        if sol.sol is None:
            y = sol.y[:, -1].copy()
        else:
            y = sol.sol(t_span[1]).reshape(-1).copy()
        t += m * dt
        idx += m

        pbar.update(m)

        if stop_condition is not None:
            if stop_condition(t, y):
                break

    if return_arrays and idx < steps:
        positions = positions[:idx]
        times = times[:idx]
        energy = energy[:idx]
        std = std[:idx]

    r_final = y.reshape((n, 2))
    pbar.close()
    return r_final, positions, energy, std, times
