import numpy as np
from numba import njit
from scipy.integrate import RK23, RK45

from ..common import compute_velocity_overdamped, compute_energy_numba, compute_std_numba

__all__ = ["run_rk23_dynamic"]


@njit(fastmath=True)
def _min_pairwise_distance_floor(r, r_floor):
    n = len(r)
    if n < 2:
        return r_floor

    min_dist_sq = 1.0e308
    for i in range(n):
        for j in range(i + 1, n):
            dx = r[i, 0] - r[j, 0]
            dy = r[i, 1] - r[j, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

    if min_dist_sq >= 1.0e308:
        return r_floor

    dist = np.sqrt(min_dist_sq)
    return dist if dist > r_floor else r_floor


def run_rk23_dynamic(
    r0,
    k,
    v0,
    l=1.0,
    r_floor=0.0,
    t_span=(0.0, 1.0),
    *,
    rtol=1e-6,
    atol=1e-6,
    first_step=None,
    max_step_global=np.inf,
    eta=0.05,
    recompute_every=10,
    metric_every=1,
    sample_dt=None,
    sample_count=None,
    callback=None,
    record=True,
    method="RK23",
    return_stats=False,
):
    """
    Adaptive RK23 (or RK45) integrator with optional callback and recording.

    Parameters
    ----------
    r0 : (N,2) ndarray
        Initial positions.
    k, v0, l : float
        Interaction parameters; coupling = v0 * l**(k+1).
    r_floor : float
        Hard floor for pair distances to avoid singularities.
    t_span : (t0, tf)
        Integration time span.
    rtol, atol : float
        Solver tolerances.
    first_step : float or None
        Optional initial step size guess.
    max_step_global : float
        Global maximum step size for the solver.
    eta : float
        Safety factor for dynamic max_step cap (k > -1).
    recompute_every : int
        Recompute distance-based cap every N accepted steps (and step 1).
    metric_every : int
        Record metrics/positions every N accepted steps (1 = every step).
    sample_dt : float or None
        If set, record on a uniform time grid with spacing sample_dt using interpolation.
    sample_count : int or None
        Number of samples to record when sample_dt is set. If None, it is derived from t_span.
    callback : callable or None
        Called after each accepted step: callback(t, r, solver).
    record : bool
        If True, return time/position/metric histories (accepted steps or sampled grid).
    method : "RK23" | "RK45"
        Integrator class to use (default RK23).
    return_stats : bool
        If True, also return a stats dict with nfev and step counts.
    """
    n = r0.shape[0]
    t0, tf = t_span
    y0 = r0.reshape(-1).copy()
    max_step = np.inf if max_step_global is None else max_step_global
    recompute_every = max(1, int(recompute_every))
    metric_every = max(1, int(metric_every))
    coupling = v0 * (l ** (k + 1))
    tiny = 1e-300
    using_sampling = sample_dt is not None

    if using_sampling:
        if sample_dt <= 0.0:
            raise ValueError("sample_dt must be > 0")
        if sample_count is None:
            sample_count = int(np.floor((tf - t0) / sample_dt)) + 1
        if sample_count < 0:
            raise ValueError("sample_count must be >= 0")

    def rhs(t, y):
        r = y.reshape((n, 2))
        vel = compute_velocity_overdamped(r, k, v0, l, r_floor)
        return vel.reshape(-1)

    method_upper = str(method).upper()
    if method_upper == "RK23":
        solver_cls = RK23
    elif method_upper == "RK45":
        solver_cls = RK45
    else:
        raise ValueError("method must be 'RK23' or 'RK45'")
    solver = solver_cls(
        rhs,
        t0,
        y0,
        tf,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_step=max_step,
    )

    if using_sampling and record:
        times = t0 + sample_dt * np.arange(sample_count, dtype=np.float64)
        positions = np.empty((sample_count, n, 2), dtype=np.float64)
        energy = np.empty(sample_count, dtype=np.float64)
        std = np.empty(sample_count, dtype=np.float64)
        sample_idx = 0
    else:
        times = []
        positions = []
        energy = []
        std = []

    if record and not using_sampling:
        r_view = solver.y.reshape((n, 2))
        times.append(solver.t)
        positions.append(r_view.copy())
        energy.append(compute_energy_numba(r_view, k, v0, l, r_floor))
        std.append(compute_std_numba(r_view))

    step_count = 0
    last_d_min = None
    last_msg = None
    t_prev = solver.t
    y_prev = solver.y.copy()

    if using_sampling and record:
        while sample_idx < sample_count and times[sample_idx] <= t_prev:
            r_sample = y_prev.reshape((n, 2))
            positions[sample_idx] = r_sample
            energy[sample_idx] = compute_energy_numba(r_sample, k, v0, l, r_floor)
            std[sample_idx] = compute_std_numba(r_sample)
            sample_idx += 1

    while solver.status == "running":
        if k > -1:
            if step_count == 0 or step_count % recompute_every == 0:
                r_view = solver.y.reshape((n, 2))
                last_d_min = _min_pairwise_distance_floor(r_view, r_floor)

            if last_d_min is not None:
                dt_cap = eta * (last_d_min ** (k + 2)) / ((n - 1) * abs(coupling) + tiny)
                solver.max_step = min(max_step, dt_cap)

        last_msg =solver.step()
        if solver.status == "failed":
            raise RuntimeError(f"{method_upper} failed at t={solver.t}: {last_msg}")

        step_count += 1
        r_view = solver.y.reshape((n, 2))
        t_curr = solver.t
        y_curr = solver.y

        if callback is not None:
            callback(solver.t, r_view, solver)

        if record and using_sampling:
            while sample_idx < sample_count and times[sample_idx] <= t_curr:
                t_sample = times[sample_idx]
                if t_curr > t_prev:
                    alpha = (t_sample - t_prev) / (t_curr - t_prev)
                else:
                    alpha = 0.0
                y_sample = y_prev + alpha * (y_curr - y_prev)
                r_sample = y_sample.reshape((n, 2))
                positions[sample_idx] = r_sample
                energy[sample_idx] = compute_energy_numba(r_sample, k, v0, l, r_floor)
                std[sample_idx] = compute_std_numba(r_sample)
                sample_idx += 1
        elif record and (step_count % metric_every == 0):
            times.append(solver.t)
            positions.append(r_view.copy())
            energy.append(compute_energy_numba(r_view, k, v0, l, r_floor))
            std.append(compute_std_numba(r_view))

        t_prev = t_curr
        y_prev = y_curr.copy()

    r_final = solver.y.reshape((n, 2)).copy()

    if record and using_sampling:
        times = times[:sample_idx]
        positions = positions[:sample_idx]
        energy = energy[:sample_idx]
        std = std[:sample_idx]
    elif record:
        times = np.asarray(times, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        energy = np.asarray(energy, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
    else:
        times = np.empty(0, dtype=np.float64)
        positions = np.empty((0, n, 2), dtype=np.float64)
        energy = np.empty(0, dtype=np.float64)
        std = np.empty(0, dtype=np.float64)

    stats = dict(
        nfev=solver.nfev,
        n_steps=step_count,
        status=solver.status,
        message=last_msg,
    )

    if return_stats:
        return r_final, positions, energy, std, times, stats
    return r_final, positions, energy, std, times
