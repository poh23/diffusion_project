import numpy as np
from numba import njit
from scipy.integrate import RK23, RK45

from ..forces import compute_velocity_overdamped
from ..metrics import compute_energy_components, compute_std_numba

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


def _select_solver_class(method: str):
    method_upper = str(method).upper()
    if method_upper == "RK23":
        return method_upper, RK23
    if method_upper == "RK45":
        return method_upper, RK45
    raise ValueError("method must be 'RK23' or 'RK45'")


def _compute_metrics(r_view, k, v0, l, r_floor, charges, population_values):
    energy, energy_aa, energy_ab, energy_bb = compute_energy_components(
        r_view,
        k,
        v0,
        l,
        r_floor,
        charges,
        population_values=population_values,
    )
    std = compute_std_numba(r_view)
    return energy, std, energy_aa, energy_ab, energy_bb


def _should_record_now(step_count, t_curr, next_save_t, record_schedule_dt, metric_every):
    if record_schedule_dt is not None:
        return t_curr >= next_save_t
    return step_count % metric_every == 0


def _advance_next_save_t(next_save_t, record_schedule_dt, t_curr):
    while next_save_t <= t_curr:
        next_save_t += record_schedule_dt
    return next_save_t


def _initialize_diffusion(
    *,
    use_diffusion: bool,
    diffusion_seed,
    diffusion_rng_state,
    diffusion_noise_var: float,
):
    if use_diffusion:
        diffusion_rng = np.random.default_rng(diffusion_seed)
        if diffusion_rng_state is not None:
            diffusion_rng.bit_generator.state = diffusion_rng_state
    else:
        diffusion_rng = None
    diffusion_noise_scale = np.sqrt(diffusion_noise_var) if diffusion_noise_var > 0.0 else 0.0
    return diffusion_rng, diffusion_noise_scale


def _initialize_sampling(
    *,
    n: int,
    t0: float,
    tf: float,
    sample_dt,
    sample_count,
    sample_t0,
    use_diffusion: bool,
    interpolate_sampling: bool,
    do_record: bool,
    record: bool,
):
    using_sampling = sample_dt is not None and not use_diffusion and interpolate_sampling
    record_schedule_dt = (
        sample_dt if sample_dt is not None and (use_diffusion or not interpolate_sampling) else None
    )
    base_sample_t0 = t0 if sample_t0 is None else sample_t0

    if record_schedule_dt is not None:
        n_intervals = int(np.floor((t0 - base_sample_t0) / record_schedule_dt))
        next_save_t = base_sample_t0 + (n_intervals + 1) * record_schedule_dt
    else:
        next_save_t = None

    if sample_dt is not None:
        if sample_dt <= 0.0:
            raise ValueError("sample_dt must be > 0")
    if using_sampling:
        if sample_count is None:
            sample_count = int(np.floor((tf - t0) / sample_dt)) + 1
        if sample_count < 0:
            raise ValueError("sample_count must be >= 0")

    if using_sampling and record:
        times = base_sample_t0 + sample_dt * np.arange(sample_count, dtype=np.float64)
        positions = np.empty((sample_count, n, 2), dtype=np.float64)
        energy = np.empty(sample_count, dtype=np.float64)
        energy_aa = np.empty(sample_count, dtype=np.float64)
        energy_ab = np.empty(sample_count, dtype=np.float64)
        energy_bb = np.empty(sample_count, dtype=np.float64)
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
        energy_aa = []
        energy_ab = []
        energy_bb = []
        std = []

    return dict(
        using_sampling=using_sampling,
        record_schedule_dt=record_schedule_dt,
        next_save_t=next_save_t,
        sample_count=sample_count,
        sample_idx=sample_idx,
        times=times,
        positions=positions,
        energy=energy,
        energy_aa=energy_aa,
        energy_ab=energy_ab,
        energy_bb=energy_bb,
        std=std,
    )


def _record_nonsampling_initial_point(
    *,
    do_record: bool,
    using_sampling: bool,
    solver,
    n: int,
    k,
    v0,
    l,
    r_floor,
    charges,
    population_values,
    record: bool,
    record_hook,
    times,
    positions,
    energy,
    energy_aa,
    energy_ab,
    energy_bb,
    std,
):
    if not do_record or using_sampling:
        return
    r_view = solver.y.reshape((n, 2))
    e0, s0, e0_aa, e0_ab, e0_bb = _compute_metrics(
        r_view, k, v0, l, r_floor, charges, population_values
    )
    if record:
        times.append(solver.t)
        positions.append(r_view.copy())
        energy.append(e0)
        energy_aa.append(e0_aa)
        energy_ab.append(e0_ab)
        energy_bb.append(e0_bb)
        std.append(s0)
    if record_hook is not None:
        record_hook(solver.t, r_view.copy(), e0, s0, e0_aa, e0_ab, e0_bb)


def _record_sampling_points_between(
    *,
    sample_idx: int,
    sample_count: int,
    times,
    t_prev: float,
    t_curr: float,
    y_prev,
    y_curr,
    n: int,
    k,
    v0,
    l,
    r_floor,
    charges,
    population_values,
    record: bool,
    record_hook,
    positions,
    energy,
    energy_aa,
    energy_ab,
    energy_bb,
    std,
):
    while sample_idx < sample_count and times[sample_idx] <= t_curr:
        t_sample = times[sample_idx]
        if t_curr > t_prev:
            alpha = (t_sample - t_prev) / (t_curr - t_prev)
        else:
            alpha = 0.0
        y_sample = y_prev + alpha * (y_curr - y_prev)
        r_sample = y_sample.reshape((n, 2))
        e_sample, s_sample, e_sample_aa, e_sample_ab, e_sample_bb = _compute_metrics(
            r_sample, k, v0, l, r_floor, charges, population_values
        )
        if record:
            positions[sample_idx] = r_sample
            energy[sample_idx] = e_sample
            energy_aa[sample_idx] = e_sample_aa
            energy_ab[sample_idx] = e_sample_ab
            energy_bb[sample_idx] = e_sample_bb
            std[sample_idx] = s_sample
        if record_hook is not None:
            record_hook(
                t_sample,
                r_sample.copy(),
                e_sample,
                s_sample,
                e_sample_aa,
                e_sample_ab,
                e_sample_bb,
            )
        sample_idx += 1
    return sample_idx


def _update_dynamic_max_step(
    *,
    solver,
    step_count: int,
    recompute_every: int,
    last_d_min,
    k,
    r_floor,
    eta: float,
    n: int,
    coupling: float,
    tiny: float,
    max_step: float,
):
    if k <= -1:
        return last_d_min

    if step_count == 0 or step_count % recompute_every == 0:
        r_view = solver.y.reshape((n, 2))
        last_d_min = _min_pairwise_distance_floor(r_view, r_floor)

    if last_d_min is not None:
        dt_cap = eta * (last_d_min ** (k + 2)) / ((n - 1) * abs(coupling) + tiny)
        solver.max_step = min(max_step, dt_cap)

    return last_d_min


def _apply_diffusion_step(
    r_view,
    *,
    use_diffusion: bool,
    dt_step: float,
    diffusion_rng,
    diffusion_noise_scale: float,
    diffusion_coeff: float,
    n: int,
):
    if not use_diffusion or dt_step <= 0.0:
        return
    noise = diffusion_rng.normal(0.0, 1.0, size=(n, 2))
    if diffusion_noise_scale != 1.0:
        noise *= diffusion_noise_scale
    r_view += np.sqrt(2.0 * diffusion_coeff * dt_step) * noise


def _record_nonsampling_step(
    *,
    step_count: int,
    t_curr: float,
    next_save_t,
    record_schedule_dt,
    metric_every: int,
    solver,
    r_view,
    k,
    v0,
    l,
    r_floor,
    charges,
    population_values,
    record: bool,
    record_hook,
    times,
    positions,
    energy,
    energy_aa,
    energy_ab,
    energy_bb,
    std,
):
    should_record = _should_record_now(
        step_count,
        t_curr,
        next_save_t,
        record_schedule_dt,
        metric_every,
    )

    if should_record:
        e_now, s_now, e_now_aa, e_now_ab, e_now_bb = _compute_metrics(
            r_view, k, v0, l, r_floor, charges, population_values
        )
        if record:
            times.append(solver.t)
            positions.append(r_view.copy())
            energy.append(e_now)
            energy_aa.append(e_now_aa)
            energy_ab.append(e_now_ab)
            energy_bb.append(e_now_bb)
            std.append(s_now)
        if record_hook is not None:
            record_hook(solver.t, r_view.copy(), e_now, s_now, e_now_aa, e_now_ab, e_now_bb)
        if next_save_t is not None:
            next_save_t = _advance_next_save_t(next_save_t, record_schedule_dt, t_curr)

    return next_save_t


def _finalize_recorded_arrays(
    *,
    record: bool,
    using_sampling: bool,
    sample_idx,
    times,
    positions,
    energy,
    energy_aa,
    energy_ab,
    energy_bb,
    std,
    n: int,
):
    if record and using_sampling:
        return (
            positions[:sample_idx],
            energy[:sample_idx],
            energy_aa[:sample_idx],
            energy_ab[:sample_idx],
            energy_bb[:sample_idx],
            std[:sample_idx],
            times[:sample_idx],
        )
    if record:
        return (
            np.asarray(positions, dtype=np.float64),
            np.asarray(energy, dtype=np.float64),
            np.asarray(energy_aa, dtype=np.float64),
            np.asarray(energy_ab, dtype=np.float64),
            np.asarray(energy_bb, dtype=np.float64),
            np.asarray(std, dtype=np.float64),
            np.asarray(times, dtype=np.float64),
        )
    return (
        np.empty((0, n, 2), dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
        np.empty(0, dtype=np.float64),
    )


def _build_stats(*, solver, step_count: int, last_msg, use_diffusion: bool, diffusion_rng):
    stats = dict(
        nfev=solver.nfev,
        n_steps=step_count,
        status=solver.status,
        message=last_msg,
    )
    if use_diffusion and diffusion_rng is not None:
        stats["rng_state"] = diffusion_rng.bit_generator.state
    return stats


def _build_rhs(*, n: int, k, v0, l, r_floor, charges):
    def rhs(t, y):
        r = y.reshape((n, 2))
        vel = compute_velocity_overdamped(r, k, v0, l, r_floor, charges)
        return vel.reshape(-1)

    return rhs


def _setup_rk23_run(
    r0,
    k,
    v0,
    l,
    r_floor,
    charges,
    t_span,
    *,
    rtol,
    atol,
    first_step,
    max_step_global,
    eta,
    recompute_every,
    metric_every,
    sample_dt,
    sample_count,
    sample_t0,
    interpolate_sampling,
    record,
    record_hook,
    method,
    diffusion,
    diffusion_coeff,
    diffusion_seed,
    diffusion_rng_state,
    diffusion_noise_var,
    population_values,
):
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
    diffusion_rng, diffusion_noise_scale = _initialize_diffusion(
        use_diffusion=use_diffusion,
        diffusion_seed=diffusion_seed,
        diffusion_rng_state=diffusion_rng_state,
        diffusion_noise_var=diffusion_noise_var,
    )
    do_record = record or record_hook is not None

    sampling_state = _initialize_sampling(
        n=n,
        t0=t0,
        tf=tf,
        sample_dt=sample_dt,
        sample_count=sample_count,
        sample_t0=sample_t0,
        use_diffusion=use_diffusion,
        interpolate_sampling=interpolate_sampling,
        do_record=do_record,
        record=record,
    )

    rhs = _build_rhs(n=n, k=k, v0=v0, l=l, r_floor=r_floor, charges=charges)
    method_upper, solver_cls = _select_solver_class(method)
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

    times = sampling_state["times"]
    positions = sampling_state["positions"]
    energy = sampling_state["energy"]
    std = sampling_state["std"]

    _record_nonsampling_initial_point(
        do_record=do_record,
        using_sampling=sampling_state["using_sampling"],
        solver=solver,
        n=n,
        k=k,
        v0=v0,
        l=l,
        r_floor=r_floor,
        charges=charges,
        population_values=population_values,
        record=record,
        record_hook=record_hook,
        times=times,
        positions=positions,
        energy=energy,
        energy_aa=sampling_state["energy_aa"],
        energy_ab=sampling_state["energy_ab"],
        energy_bb=sampling_state["energy_bb"],
        std=std,
    )

    t_prev = solver.t
    y_prev = solver.y.copy()
    sample_idx = sampling_state["sample_idx"]
    if sampling_state["using_sampling"] and do_record:
        sample_idx = _record_sampling_points_between(
            sample_idx=sample_idx,
            sample_count=sampling_state["sample_count"],
            times=times,
            t_prev=t_prev,
            t_curr=t_prev,
            y_prev=y_prev,
            y_curr=y_prev,
            n=n,
            k=k,
            v0=v0,
            l=l,
            r_floor=r_floor,
            charges=charges,
            population_values=population_values,
            record=record,
            record_hook=record_hook,
            positions=positions,
            energy=energy,
            energy_aa=sampling_state["energy_aa"],
            energy_ab=sampling_state["energy_ab"],
            energy_bb=sampling_state["energy_bb"],
            std=std,
        )

    return dict(
        n=n,
        k=k,
        v0=v0,
        l=l,
        r_floor=r_floor,
        charges=charges,
        population_values=population_values,
        method_upper=method_upper,
        solver=solver,
        max_step=max_step,
        eta=eta,
        recompute_every=recompute_every,
        metric_every=metric_every,
        coupling=coupling,
        tiny=tiny,
        use_diffusion=use_diffusion,
        diffusion_coeff=diffusion_coeff,
        diffusion_rng=diffusion_rng,
        diffusion_noise_scale=diffusion_noise_scale,
        do_record=do_record,
        using_sampling=sampling_state["using_sampling"],
        record_schedule_dt=sampling_state["record_schedule_dt"],
        next_save_t=sampling_state["next_save_t"],
        sample_count=sampling_state["sample_count"],
        sample_idx=sample_idx,
        times=times,
        positions=positions,
        energy=energy,
        energy_aa=sampling_state["energy_aa"],
        energy_ab=sampling_state["energy_ab"],
        energy_bb=sampling_state["energy_bb"],
        std=std,
        t_prev=t_prev,
        y_prev=y_prev,
        step_count=0,
        last_d_min=None,
        last_msg=None,
    )


def _integrate_rk23_loop(runtime: dict, *, record: bool, callback, record_hook, stop_condition):
    solver = runtime["solver"]

    while solver.status == "running":
        runtime["last_d_min"] = _update_dynamic_max_step(
            solver=solver,
            step_count=runtime["step_count"],
            recompute_every=runtime["recompute_every"],
            last_d_min=runtime["last_d_min"],
            k=runtime["k"],
            r_floor=runtime["r_floor"],
            eta=runtime["eta"],
            n=runtime["n"],
            coupling=runtime["coupling"],
            tiny=runtime["tiny"],
            max_step=runtime["max_step"],
        )

        runtime["last_msg"] = solver.step()
        if solver.status == "failed":
            raise RuntimeError(f"{runtime['method_upper']} failed at t={solver.t}: {runtime['last_msg']}")

        runtime["step_count"] += 1
        r_view = solver.y.reshape((runtime["n"], 2))
        t_curr = solver.t
        y_curr = solver.y
        dt_step = t_curr - runtime["t_prev"]

        _apply_diffusion_step(
            r_view,
            use_diffusion=runtime["use_diffusion"],
            dt_step=dt_step,
            diffusion_rng=runtime["diffusion_rng"],
            diffusion_noise_scale=runtime["diffusion_noise_scale"],
            diffusion_coeff=runtime["diffusion_coeff"],
            n=runtime["n"],
        )

        if callback is not None:
            callback(solver.t, r_view, solver)

        if runtime["do_record"] and runtime["using_sampling"]:
            runtime["sample_idx"] = _record_sampling_points_between(
                sample_idx=runtime["sample_idx"],
                sample_count=runtime["sample_count"],
                times=runtime["times"],
                t_prev=runtime["t_prev"],
                t_curr=t_curr,
                y_prev=runtime["y_prev"],
                y_curr=y_curr,
                n=runtime["n"],
                k=runtime["k"],
                v0=runtime["v0"],
                l=runtime["l"],
                r_floor=runtime["r_floor"],
                charges=runtime["charges"],
                population_values=runtime["population_values"],
                record=record,
                record_hook=record_hook,
                positions=runtime["positions"],
                energy=runtime["energy"],
                energy_aa=runtime["energy_aa"],
                energy_ab=runtime["energy_ab"],
                energy_bb=runtime["energy_bb"],
                std=runtime["std"],
            )
        elif runtime["do_record"]:
            runtime["next_save_t"] = _record_nonsampling_step(
                step_count=runtime["step_count"],
                t_curr=t_curr,
                next_save_t=runtime["next_save_t"],
                record_schedule_dt=runtime["record_schedule_dt"],
                metric_every=runtime["metric_every"],
                solver=solver,
                r_view=r_view,
                k=runtime["k"],
                v0=runtime["v0"],
                l=runtime["l"],
                r_floor=runtime["r_floor"],
                charges=runtime["charges"],
                population_values=runtime["population_values"],
                record=record,
                record_hook=record_hook,
                times=runtime["times"],
                positions=runtime["positions"],
                energy=runtime["energy"],
                energy_aa=runtime["energy_aa"],
                energy_ab=runtime["energy_ab"],
                energy_bb=runtime["energy_bb"],
                std=runtime["std"],
            )

        if stop_condition is not None:
            if stop_condition(t_curr, r_view, solver):
                solver.status = "finished"
                break

        runtime["t_prev"] = t_curr
        runtime["y_prev"] = y_curr.copy()

    runtime["r_final"] = solver.y.reshape((runtime["n"], 2)).copy()
    return runtime


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
    interpolate_sampling=True,
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
    population_values=None,
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
    interpolate_sampling : bool
        If False, disable interpolation even when diffusion is off and sample_dt is set.
        In that mode, recording uses the first accepted step at or after each sample time.
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
    runtime = _setup_rk23_run(
        r0,
        k,
        v0,
        l,
        r_floor,
        charges,
        t_span,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_step_global=max_step_global,
        eta=eta,
        recompute_every=recompute_every,
        metric_every=metric_every,
        sample_dt=sample_dt,
        sample_count=sample_count,
        sample_t0=sample_t0,
        interpolate_sampling=interpolate_sampling,
        record=record,
        record_hook=record_hook,
        method=method,
        diffusion=diffusion,
        diffusion_coeff=diffusion_coeff,
        diffusion_seed=diffusion_seed,
        diffusion_rng_state=diffusion_rng_state,
        diffusion_noise_var=diffusion_noise_var,
        population_values=population_values,
    )

    runtime = _integrate_rk23_loop(
        runtime,
        record=record,
        callback=callback,
        record_hook=record_hook,
        stop_condition=stop_condition,
    )

    positions, energy, energy_aa, energy_ab, energy_bb, std, times = _finalize_recorded_arrays(
        record=record,
        using_sampling=runtime["using_sampling"],
        sample_idx=runtime["sample_idx"],
        times=runtime["times"],
        positions=runtime["positions"],
        energy=runtime["energy"],
        energy_aa=runtime["energy_aa"],
        energy_ab=runtime["energy_ab"],
        energy_bb=runtime["energy_bb"],
        std=runtime["std"],
        n=runtime["n"],
    )

    stats = _build_stats(
        solver=runtime["solver"],
        step_count=runtime["step_count"],
        last_msg=runtime["last_msg"],
        use_diffusion=runtime["use_diffusion"],
        diffusion_rng=runtime["diffusion_rng"],
    )

    if return_stats:
        return runtime["r_final"], positions, energy, energy_aa, energy_ab, energy_bb, std, times, stats
    return runtime["r_final"], positions, energy, energy_aa, energy_ab, energy_bb, std, times
