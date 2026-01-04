import numpy as np
from numba import njit

from ..common import compute_velocity_overdamped, compute_energy_numba, compute_std_numba

__all__ = [
    "rk2_step_numba",
    "make_diffusion_noise",
    "run_rk2_loop_diffusion",
    "run_rk2_loop_with_noise",
    "run_rk2_loop",
]


@njit(fastmath=True)
def rk2_step_numba(r, k, v0, l, softening, dt):
    k1 = compute_velocity_overdamped(r, k, v0, l, softening)
    k2 = compute_velocity_overdamped(r + 0.5 * dt * k1, k, v0, l, softening)
    return r + dt * k2


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
        r += noise[i]   # diffusion increment, no RNG calls here

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
