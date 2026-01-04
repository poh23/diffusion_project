import numpy as np
from numba import njit, prange

__all__ = [
    "compute_velocity_overdamped",
    "compute_energy_numba",
    "compute_std_numba",
    "init_positions_jittered_disk",
]


@njit(fastmath=True, parallel=True)
def compute_velocity_overdamped(r, k, v0, l, softening):
    n = len(r)
    vel = np.zeros_like(r)
    coupling = v0 * (l ** (k + 1))

    for i in prange(n):
        for j in range(n):
            if i != j:
                dx = r[i, 0] - r[j, 0]
                dy = r[i, 1] - r[j, 1]
                dist_sq = dx * dx + dy * dy
                soft_dist = np.sqrt(dist_sq + softening * softening)

                factor = coupling / (soft_dist ** (k + 2))
                vel[i, 0] += factor * dx
                vel[i, 1] += factor * dy
    return vel


@njit(fastmath=True, parallel=True)
def compute_energy_numba(r, k, v0, l, softening):
    n = len(r)
    pe = 0.0
    coupling = v0 * (l ** (k + 1))

    for i in prange(n):
        local_pe = 0.0
        for j in range(i + 1, n):
            dx = r[i, 0] - r[j, 0]
            dy = r[i, 1] - r[j, 1]
            dist_sq = dx * dx + dy * dy
            soft_dist = np.sqrt(dist_sq + softening * softening)
            local_pe += coupling / (k * (soft_dist ** k))
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


def init_positions_jittered_disk(n, radius=1.0, jitter=0.05, seed=None):
    """
    Sample points uniformly over a disk (area-uniform), with a small angular/radial jitter.
    """
    rng = np.random.default_rng(seed)

    # area-uniform radius: r = R * sqrt(u)
    u = rng.uniform(0.0, 1.0, size=n)
    r = radius * np.sqrt(u)

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)

    if jitter and jitter > 0.0:
        theta += rng.uniform(-jitter, jitter, size=n)
        r *= 1.0 + rng.uniform(-jitter, jitter, size=n)
        r = np.clip(r, 0.0, radius)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])
