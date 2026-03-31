from ..config import DEFAULT_R_FLOOR, SimulationConfig, validate_config
from .helpers import is_streaming_mode
from .nonstream import run_nonstream_simulation
from .streaming import run_streaming_simulation


def run_simulation(
    *,
    n_particles=10,
    k=1.0,
    v0=1.0,
    l=1.0,
    method="rk23",  # "rk23" or "dop853"
    seed=0,
    r_floor=DEFAULT_R_FLOOR,
    init_radius=1.0,
    charge_values=None,
    charge_counts=None,
    t0=0.0,
    t_duration=1.0,
    save_every=None,
    save_every_steps=None,
    # progress control
    chunk_steps=5000,
    # batching / resume
    batch_every=None,
    target_batch_mb=None,
    max_wall_time=None,
    resume_from=None,
    resume_force=False,
    # dop853 tolerances
    rtol=1e-6,
    atol=1e-6,
    # rk23 adaptive settings
    first_step=None,
    max_step_global=float("inf"),
    eta=0.05,
    recompute_every=10,
    interpolate_sampling=True,
    diffusion=False,
    diffusion_coeff=0.0,
    diffusion_seed=None,
    diffusion_noise_var=1.0,
    external_potential=None,
    external_potential_params=None,
    out_format="npz",
    out_path=None,
):
    config = SimulationConfig(
        n_particles=n_particles,
        k=k,
        v0=v0,
        l=l,
        r_floor=r_floor,
        init_radius=init_radius,
        charge_values=charge_values,
        charge_counts=charge_counts,
        t0=t0,
        t_duration=t_duration,
        save_every=save_every,
        save_every_steps=save_every_steps,
        method=method,
        seed=seed,
        chunk_steps=chunk_steps,
        batch_every=batch_every,
        target_batch_mb=target_batch_mb,
        max_wall_time=max_wall_time,
        resume_from=resume_from,
        resume_force=resume_force,
        rtol=rtol,
        atol=atol,
        first_step=first_step,
        max_step_global=max_step_global,
        eta=eta,
        recompute_every=recompute_every,
        interpolate_sampling=interpolate_sampling,
        diffusion=diffusion,
        diffusion_coeff=diffusion_coeff,
        diffusion_seed=diffusion_seed,
        diffusion_noise_var=diffusion_noise_var,
        external_potential=external_potential,
        external_potential_params=external_potential_params,
        out_format=out_format,
    )
    validate_config(config)

    if is_streaming_mode(config):
        return run_streaming_simulation(config, out_path=out_path)
    return run_nonstream_simulation(config)
