import numpy as np
from numba import njit, prange

__all__ = [
    "compute_velocity_overdamped",
    "compute_energy_numba",
    "compute_std_numba",
    "init_positions_jittered_disk",
]


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


def init_positions_jittered_disk(n, radius=1.0, jitter=0.05, seed=None, min_dist=None, max_attempts=None):
    """
    Sample points uniformly over a disk (area-uniform), with a small angular/radial jitter.
    Enforces a minimum pair distance when min_dist > 0, and raises if packing is too dense.
    """
    rng = np.random.default_rng(seed)

    if min_dist is None:
        min_dist = 0.5 * radius / np.sqrt(max(1, n))

    if min_dist <= 0.0:
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

    # Hex packing upper bound for disk area.
    area = np.pi * radius * radius
    max_n = int(np.floor(area / (0.5 * np.sqrt(3.0) * min_dist * min_dist)))
    if n > max_n:
        raise ValueError(
            f"Too many particles for radius={radius} and min_dist={min_dist}. "
            f"Max is about {max_n}; increase radius or reduce min_dist."
        )

    if max_attempts is None:
        max_attempts = max(10000, n * 200)

    positions = np.empty((n, 2), dtype=np.float64)
    min_dist_sq = min_dist * min_dist
    cell_size = min_dist
    inv_cell = 1.0 / cell_size
    grid = {}

    count = 0
    attempts = 0
    while count < n:
        if attempts >= max_attempts:
            raise ValueError(
                f"Failed to place {n} particles with min_dist={min_dist} "
                f"within {max_attempts} attempts; increase radius or reduce min_dist."
            )
        attempts += 1

        u = rng.uniform(0.0, 1.0)
        r = radius * np.sqrt(u)
        theta = rng.uniform(0.0, 2.0 * np.pi)

        if jitter and jitter > 0.0:
            theta += rng.uniform(-jitter, jitter)
            r *= 1.0 + rng.uniform(-jitter, jitter)
            r = float(np.clip(r, 0.0, radius))

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        gx = int(np.floor((x + radius) * inv_cell))
        gy = int(np.floor((y + radius) * inv_cell))

        ok = True
        for nx in range(gx - 1, gx + 2):
            for ny in range(gy - 1, gy + 2):
                for idx in grid.get((nx, ny), ()):
                    dx = x - positions[idx, 0]
                    dy = y - positions[idx, 1]
                    if (dx * dx + dy * dy) < min_dist_sq:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            positions[count, 0] = x
            positions[count, 1] = y
            grid.setdefault((gx, gy), []).append(count)
            count += 1

    return positions
