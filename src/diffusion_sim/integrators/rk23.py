import numpy as np
from numba import njit
from scipy.integrate import RK23, RK45

from ..forces import compute_velocity_overdamped
from ..metrics import compute_energy_numba, compute_std_numba

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
    charges=None,
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
    sample_t0=None,
    callback=None,
    record_hook=None,
    record=True,
    method="RK23",
    return_stats=False,
    diffusion=False,
    diffusion_coeff=0.0,
    diffusion_seed=None,
    diffusion_rng_state=None,
    diffusion_noise_var=1.0,
    stop_condition=None,
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
    charges : (N,) ndarray or None
        Per-particle charge values. None defaults to all ones.
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
        When diffusion is enabled, interpolation is disabled and the first accepted step
        at or after each sample_dt interval is recorded instead.
    sample_count : int or None
        Number of samples to record when sample_dt is set. If None, it is derived from t_span.
    sample_t0 : float or None
        Reference time for the sampling grid when sample_dt is set. Defaults to t_span[0].
    callback : callable or None
        Called after each accepted step: callback(t, r, solver).
    record_hook : callable or None
        Called on each recorded sample: record_hook(t, r, energy, std).
    record : bool
        If True, return time/position/metric histories (accepted steps or sampled grid).
    method : "RK23" | "RK45"
        Integrator class to use (default RK23).
    return_stats : bool
        If True, also return a stats dict with nfev and step counts.
    diffusion : bool
        If True, add a stochastic displacement after each accepted step.
    diffusion_coeff : float
        Diffusion constant D used in the stochastic step.
    diffusion_seed : int or None
        RNG seed for the diffusion term. None uses non-deterministic entropy.
    diffusion_rng_state : dict or None
        RNG state to restore for the diffusion term (overrides diffusion_seed).
    diffusion_noise_var : float
        Variance of the Gaussian noise used in the diffusion step (per component).
    stop_condition : callable or None
        If provided, called as stop_condition(t, r, solver). If it returns True, integration stops.
    """
    n = r0.shape[0]
    t0, tf = t_span
    y0 = r0.reshape(-1).copy()
    max_step = np.inf if max_step_global is None else max_step_global
    recompute_every = max(1, int(recompute_every))
    metric_every = max(1, int(metric_every))
    if charges is None:
        charges = np.ones(n, dtype=np.float64)
    coupling = v0 * (l ** (k + 1))
    charge_scale = np.max(np.abs(charges)) if n > 0 else 1.0
    coupling *= charge_scale * charge_scale
    tiny = 1e-300
    use_diffusion = bool(diffusion) and diffusion_coeff != 0.0 and diffusion_noise_var != 0.0
    using_sampling = sample_dt is not None and not use_diffusion
    record_schedule_dt = sample_dt if use_diffusion and sample_dt is not None else None
    base_sample_t0 = t0 if sample_t0 is None else sample_t0
    if record_schedule_dt is not None:
        n_intervals = int(np.floor((t0 - base_sample_t0) / record_schedule_dt))
        next_save_t = base_sample_t0 + (n_intervals + 1) * record_schedule_dt
    else:
        next_save_t = None
    if use_diffusion:
        diffusion_rng = np.random.default_rng(diffusion_seed)
        if diffusion_rng_state is not None:
            diffusion_rng.bit_generator.state = diffusion_rng_state
    else:
        diffusion_rng = None
    diffusion_noise_scale = np.sqrt(diffusion_noise_var) if diffusion_noise_var > 0.0 else 0.0
    do_record = record or record_hook is not None

    if sample_dt is not None:
        if sample_dt <= 0.0:
            raise ValueError("sample_dt must be > 0")
    if using_sampling:
        if sample_count is None:
            sample_count = int(np.floor((tf - t0) / sample_dt)) + 1
        if sample_count < 0:
            raise ValueError("sample_count must be >= 0")

    def rhs(t, y):
        r = y.reshape((n, 2))
        vel = compute_velocity_overdamped(r, k, v0, l, r_floor, charges)
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
        times = base_sample_t0 + sample_dt * np.arange(sample_count, dtype=np.float64)
        positions = np.empty((sample_count, n, 2), dtype=np.float64)
        energy = np.empty(sample_count, dtype=np.float64)
        std = np.empty(sample_count, dtype=np.float64)
        sample_idx = 0
    else:
        if using_sampling and do_record:
            times = base_sample_t0 + sample_dt * np.arange(sample_count, dtype=np.float64)
            sample_idx = 0
        else:
            times = []
            sample_idx = None
        positions = []
        energy = []
        std = []

    if do_record and not using_sampling:
        r_view = solver.y.reshape((n, 2))
        e0 = compute_energy_numba(r_view, k, v0, l, r_floor, charges)
        s0 = compute_std_numba(r_view)
        if record:
            times.append(solver.t)
            positions.append(r_view.copy())
            energy.append(e0)
            std.append(s0)
        if record_hook is not None:
            record_hook(solver.t, r_view.copy(), e0, s0)

    step_count = 0
    last_d_min = None
    last_msg = None
    t_prev = solver.t
    y_prev = solver.y.copy()

    if using_sampling and do_record:
        while sample_idx < sample_count and times[sample_idx] <= t_prev:
            r_sample = y_prev.reshape((n, 2))
            e_sample = compute_energy_numba(r_sample, k, v0, l, r_floor, charges)
            s_sample = compute_std_numba(r_sample)
            if record:
                positions[sample_idx] = r_sample
                energy[sample_idx] = e_sample
                std[sample_idx] = s_sample
            if record_hook is not None:
                record_hook(times[sample_idx], r_sample.copy(), e_sample, s_sample)
            sample_idx += 1

    while solver.status == "running":
        if k > -1:
            if step_count == 0 or step_count % recompute_every == 0:
                r_view = solver.y.reshape((n, 2))
                last_d_min = _min_pairwise_distance_floor(r_view, r_floor)

            if last_d_min is not None:
                dt_cap = eta * (last_d_min ** (k + 2)) / ((n - 1) * abs(coupling) + tiny)
                solver.max_step = min(max_step, dt_cap)

        last_msg = solver.step()
        if solver.status == "failed":
            raise RuntimeError(f"{method_upper} failed at t={solver.t}: {last_msg}")

        step_count += 1
        r_view = solver.y.reshape((n, 2))
        t_curr = solver.t
        y_curr = solver.y
        dt_step = t_curr - t_prev

        if use_diffusion:
            if dt_step > 0.0:
                noise = diffusion_rng.normal(0.0, 1.0, size=(n, 2))
                if diffusion_noise_scale != 1.0:
                    noise *= diffusion_noise_scale
                r_view += np.sqrt(2.0 * diffusion_coeff * dt_step) * noise

        if callback is not None:
            callback(solver.t, r_view, solver)

        if do_record and using_sampling:
            while sample_idx < sample_count and times[sample_idx] <= t_curr:
                t_sample = times[sample_idx]
                if t_curr > t_prev:
                    alpha = (t_sample - t_prev) / (t_curr - t_prev)
                else:
                    alpha = 0.0
                y_sample = y_prev + alpha * (y_curr - y_prev)
                r_sample = y_sample.reshape((n, 2))
                e_sample = compute_energy_numba(r_sample, k, v0, l, r_floor, charges)
                s_sample = compute_std_numba(r_sample)
                if record:
                    positions[sample_idx] = r_sample
                    energy[sample_idx] = e_sample
                    std[sample_idx] = s_sample
                if record_hook is not None:
                    record_hook(t_sample, r_sample.copy(), e_sample, s_sample)
                sample_idx += 1
        elif do_record:
            should_record = False
            if record_schedule_dt is not None:
                if t_curr >= next_save_t:
                    should_record = True
            elif step_count % metric_every == 0:
                should_record = True

            if should_record:
                e_now = compute_energy_numba(r_view, k, v0, l, r_floor, charges)
                s_now = compute_std_numba(r_view)
                if record:
                    times.append(solver.t)
                    positions.append(r_view.copy())
                    energy.append(e_now)
                    std.append(s_now)
                if record_hook is not None:
                    record_hook(solver.t, r_view.copy(), e_now, s_now)
                if next_save_t is not None:
                    while next_save_t <= t_curr:
                        next_save_t += record_schedule_dt

        if stop_condition is not None:
            if stop_condition(t_curr, r_view, solver):
                solver.status = "finished"
                break

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
    if use_diffusion and diffusion_rng is not None:
        stats["rng_state"] = diffusion_rng.bit_generator.state

    if return_stats:
        return r_final, positions, energy, std, times, stats
    return r_final, positions, energy, std, times
