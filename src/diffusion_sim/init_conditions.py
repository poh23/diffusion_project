import numpy as np

__all__ = ["init_positions_jittered_disk", "init_charges_two_populations"]


def init_positions_jittered_disk(
    n,
    radius=1.0,
    inner_radius=0.0,
    jitter=0.05,
    seed=None,
    min_dist=None,
    max_attempts=None,
    rng=None,
):
    """
    Sample points uniformly over a disk or annulus (area-uniform), with a small angular/radial jitter.
    Enforces a minimum pair distance when min_dist > 0, and raises if packing is too dense.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    inner_radius = float(inner_radius)
    radius = float(radius)
    if inner_radius < 0.0 or radius <= inner_radius:
        raise ValueError("radius must be greater than inner_radius >= 0")

    if min_dist is None:
        equivalent_radius = np.sqrt(max(0.0, radius * radius - inner_radius * inner_radius))
        min_dist = 0.5 * equivalent_radius / np.sqrt(max(1, n))

    if min_dist <= 0.0:
        # area-uniform radius: r = R * sqrt(u)
        u = rng.uniform(0.0, 1.0, size=n)
        r = np.sqrt(inner_radius * inner_radius + u * (radius * radius - inner_radius * inner_radius))
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)

        if jitter and jitter > 0.0:
            theta += rng.uniform(-jitter, jitter, size=n)
            r *= 1.0 + rng.uniform(-jitter, jitter, size=n)
            r = np.clip(r, inner_radius, radius)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y])

    # Hex packing upper bound for disk/annulus area.
    area = np.pi * (radius * radius - inner_radius * inner_radius)
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
        r = np.sqrt(inner_radius * inner_radius + u * (radius * radius - inner_radius * inner_radius))
        theta = rng.uniform(0.0, 2.0 * np.pi)

        if jitter and jitter > 0.0:
            theta += rng.uniform(-jitter, jitter)
            r *= 1.0 + rng.uniform(-jitter, jitter)
            r = float(np.clip(r, inner_radius, radius))

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


def init_charges_two_populations(n, values, counts, rng=None):
    """
    Initialize two positive charge populations and mix them uniformly.
    """
    if rng is None:
        rng = np.random.default_rng()

    if len(values) != 2 or len(counts) != 2:
        raise ValueError("values and counts must have length 2")
    if counts[0] + counts[1] != n:
        raise ValueError("counts must sum to n")

    charges = np.empty(n, dtype=np.float64)
    charges[: counts[0]] = values[0]
    charges[counts[0] :] = values[1]
    perm = rng.permutation(n)
    return charges[perm]
