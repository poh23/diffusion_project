import numpy as np
from numba import njit

from ..forces import compute_velocity_overdamped
from ..metrics import compute_energy_numba, compute_std_numba

__all__ = ["rk2_step_numba", "run_rk2_loop"]


@njit(fastmath=True)
def rk2_step_numba(r, k, v0, l, r_floor, dt):
    k1 = compute_velocity_overdamped(r, k, v0, l, r_floor)
    k2 = compute_velocity_overdamped(r + 0.5 * dt * k1, k, v0, l, r_floor)
    return r + dt * k2


@njit(fastmath=True)
def run_rk2_loop(r0, k, v0, l, r_floor, dt, steps, t0):
    n = len(r0)
    r_hist = np.zeros((steps, n, 2))
    pe_hist = np.zeros(steps)
    std_hist = np.zeros(steps)
    t_hist = np.zeros(steps)

    r = r0.copy()
    t = t0

    for i in range(steps):
        r_hist[i] = r
        pe_hist[i] = compute_energy_numba(r, k, v0, l, r_floor)
        std_hist[i] = compute_std_numba(r)
        t_hist[i] = t

        r = rk2_step_numba(r, k, v0, l, r_floor, dt)
        t += dt

    return r, r_hist, pe_hist, std_hist, t_hist
