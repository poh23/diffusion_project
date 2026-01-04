import numpy as np
from numba import njit

from ..common import compute_velocity_overdamped, compute_energy_numba, compute_std_numba

__all__ = ["rk4_step_numba", "run_rk4_loop"]


@njit(fastmath=True)
def rk4_step_numba(r, k, v0, l, softening, dt):
    k1 = compute_velocity_overdamped(r, k, v0, l, softening)
    k2 = compute_velocity_overdamped(r + 0.5 * dt * k1, k, v0, l, softening)
    k3 = compute_velocity_overdamped(r + 0.5 * dt * k2, k, v0, l, softening)
    k4 = compute_velocity_overdamped(r + dt * k3, k, v0, l, softening)
    return r + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@njit(fastmath=True)
def run_rk4_loop(r0, k, v0, l, softening, dt, steps, t0):
    n = len(r0)
    r_hist = np.zeros((steps, n, 2))
    pe_hist = np.zeros(steps)
    std_hist = np.zeros(steps)
    t_hist = np.zeros(steps)

    r = r0.copy()
    t = t0

    for i in range(steps):
        r_hist[i] = r
        pe_hist[i] = compute_energy_numba(r, k, v0, l, softening)
        std_hist[i] = compute_std_numba(r)
        t_hist[i] = t
        r = rk4_step_numba(r, k, v0, l, softening, dt)
        t += dt

    return r, r_hist, pe_hist, std_hist, t_hist
