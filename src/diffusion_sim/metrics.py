import numpy as np
from numba import njit, prange

__all__ = ["compute_energy_numba", "compute_std_numba"]


@njit(fastmath=True, parallel=True)
def compute_energy_numba(r, k, v0, l, r_floor):
    n = len(r)
    pe = 0.0
    coupling = v0 * (l ** (k + 1))

    for i in prange(n):
        local_pe = 0.0
        for j in range(i + 1, n):
            dx = r[i, 0] - r[j, 0]
            dy = r[i, 1] - r[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq)
            dist_eff = dist if dist > r_floor else r_floor
            local_pe += coupling / (k * (dist_eff ** k))
        pe += local_pe
    return pe


@njit(fastmath=True)
def compute_std_numba(r):
    n = len(r)
    sum_r = 0.0
    sum_sq = 0.0

    for i in range(n):
        radius = np.sqrt(r[i, 0] * r[i, 0] + r[i, 1] * r[i, 1])
        sum_r += radius
        sum_sq += radius * radius

    mean = sum_r / n
    var = (sum_sq / n) - mean * mean
    if var < 0.0:
        var = 0.0
    return np.sqrt(var)
