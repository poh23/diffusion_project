import numpy as np
from numba import njit, prange


# ----------------------------
# Physics kernels (Numba)
# ----------------------------

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


# ----------------------------
# Integrators (Numba)
# ----------------------------

@njit(fastmath=True)
def rk4_step_numba(r, k, v0, l, softening, dt):
    k1 = compute_velocity_overdamped(r, k, v0, l, softening)
    k2 = compute_velocity_overdamped(r + 0.5 * dt * k1, k, v0, l, softening)
    k3 = compute_velocity_overdamped(r + 0.5 * dt * k2, k, v0, l, softening)
    k4 = compute_velocity_overdamped(r + dt * k3, k, v0, l, softening)
    return r + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@njit(fastmath=True)
def rk2_step_numba(r, k, v0, l, softening, dt):
    k1 = compute_velocity_overdamped(r, k, v0, l, softening)
    k2 = compute_velocity_overdamped(r + 0.5 * dt * k1, k, v0, l, softening)
    return r + dt * k2


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

def make_diffusion_noise(steps, n, dt, D, seed=0, var_chi=1.0, dist="normal", dtype=np.float64):
    """
    Build noise array with:
        noise = sqrt(2*D*dt) * chi
    where Var(chi)=var_chi per coordinate.

    Parameters
    ----------
    D : float
        Diffusion constant.
    var_chi : float
        Variance of chi entries (per coordinate).
        - If dist="normal", chi~N(0,1) => var_chi should be 1.
    dist : "normal" | "uniform"
        Distribution used for chi.
    """
    rng = np.random.default_rng(seed)

    if D <= 0.0:
        return np.zeros((steps, n, 2), dtype=dtype)

    if dist == "normal":
        # chi ~ N(0,1) => var=1
        chi = rng.standard_normal(size=(steps, n, 2)).astype(dtype)
        chi_var = 1.0

    elif dist == "uniform":
        # Uniform(-a,a) has var = a^2/3 -> choose a so var matches var_chi
        a = np.sqrt(3.0 * var_chi)
        chi = rng.uniform(-a, a, size=(steps, n, 2)).astype(dtype)
        chi_var = var_chi

    else:
        raise ValueError("dist must be 'normal' or 'uniform'")

    # rescale chi to have variance = var_chi (in case dist="normal" but user gave var_chi != 1)
    if chi_var > 0.0 and var_chi > 0.0:
        chi *= np.sqrt(var_chi / chi_var)

    sigma = np.sqrt(2.0 * D * dt)
    noise = sigma * chi
    return noise


def run_rk2_loop_diffusion(r0, k, v0, l, softening, dt, steps, t0,
                           D, seed=0, var_chi=1.0, dist="normal"):
    """
    Fast RK2 + diffusion using precomputed noise:
        dx = sqrt(2*D*dt)*chi
    """
    n = len(r0)
    noise = make_diffusion_noise(steps, n, dt, D, seed=seed, var_chi=var_chi, dist=dist, dtype=np.float64)
    return run_rk2_loop_with_noise(r0, k, v0, l, softening, dt, steps, t0, noise)


@njit(fastmath=True)
def run_rk2_loop_with_noise(r0, k, v0, l, softening, dt, steps, t0, noise):
    """
    noise shape: (steps, n, 2)
    This avoids calling RNG inside numba (much faster).
    """
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

        r = rk2_step_numba(r, k, v0, l, softening, dt)
        r += noise[i]   # <<<< diffusion increment, no RNG calls here

        t += dt

    return r, r_hist, pe_hist, std_hist, t_hist



@njit(fastmath=True)
def run_rk2_loop(r0, k, v0, l, softening, dt, steps, t0, random_walk_std=0.0, seed=0):
    """
    random_walk_std interpreted as *per-step* displacement std.
    """
    np.random.seed(seed)

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

        r = rk2_step_numba(r, k, v0, l, softening, dt)

        if random_walk_std > 0.0:
            for p in range(n):
                r[p, 0] += np.random.normal(0.0, random_walk_std)
                r[p, 1] += np.random.normal(0.0, random_walk_std)

        t += dt

    return r, r_hist, pe_hist, std_hist, t_hist


def init_positions_on_circle(n, jitter=0.1, radius=1.0, seed=None):
    rng = np.random.default_rng(seed)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    angles = angles + rng.random(n) * jitter
    r = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
    return r

def init_positions_jittered_disk(n, radius=1.0, jitter=0.05, seed=None):
    rng = np.random.default_rng(seed)

    # choose number of radial layers ~ sqrt(n)
    n_r = int(np.sqrt(n))
    if n_r < 1:
        n_r = 1

    # radii of layers (area-uniform layer boundaries)
    layer_edges = np.linspace(0.0, 1.0, n_r + 1)
    r_layers = radius * np.sqrt(0.5 * (layer_edges[:-1] + layer_edges[1:]))

    pts = []
    for i, r0 in enumerate(r_layers):
        # allocate points per layer proportional to circumference ~ r
        # (avoid 0 at center)
        weight = max(r0, 1e-6)
        pts.append(weight)

    pts = np.array(pts)
    counts = np.maximum(1, np.round(n * pts / pts.sum()).astype(int))

    # fix total count to exactly n
    while counts.sum() > n:
        counts[np.argmax(counts)] -= 1
    while counts.sum() < n:
        counts[np.argmax(pts)] += 1

    xy = []
    for r0, m in zip(r_layers, counts):
        angles = np.linspace(0.0, 2.0*np.pi, m, endpoint=False)
        angles += rng.uniform(-jitter, jitter, size=m)
        rr = r0 * (1.0 + rng.uniform(-jitter, jitter, size=m))
        xy.append(np.column_stack([rr*np.cos(angles), rr*np.sin(angles)]))

    xy = np.vstack(xy)[:n]
    return xy

