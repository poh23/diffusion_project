import numpy as np
from numba import njit, prange

__all__ = ["compute_velocity_overdamped"]


@njit(fastmath=True, parallel=True)
def compute_velocity_overdamped(r, k, v0, l, r_floor):
    n = len(r)
    vel = np.zeros_like(r)
    coupling = v0 * (l ** (k + 1))

    for i in prange(n):
        for j in range(n):
            if i != j:
                dx = r[i, 0] - r[j, 0]
                dy = r[i, 1] - r[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq)
                dist_eff = dist if dist > r_floor else r_floor
                factor = coupling / (dist_eff ** (k + 2))
                vel[i, 0] += factor * dx
                vel[i, 1] += factor * dy
    return vel
