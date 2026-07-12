import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import SimulationConfig
from ..init_conditions import init_charges_two_populations, init_positions_jittered_disk


def auto_chunk_steps(steps: int, *, target_updates: int = 100, min_chunk: int = 100, max_chunk: int = 5000) -> int:
    """
    Choose a chunk size to get roughly target_updates progress updates without making chunks too small.
    """
    if steps <= 0:
        return min_chunk
    chunk = max(min_chunk, steps // max(1, target_updates))
    return int(max(1, min(chunk, max_chunk)))


def config_fingerprint(config: SimulationConfig) -> str:
    payload = dict(
        n_particles=config.n_particles,
        k=config.k,
        v0=config.v0,
        l=config.l,
        r_floor=config.r_floor,
        method=config.method,
        rtol=config.rtol,
        atol=config.atol,
        first_step=config.first_step,
        max_step_global=config.max_step_global,
        eta=config.eta,
        recompute_every=config.recompute_every,
        interpolate_sampling=config.interpolate_sampling,
        diffusion=config.diffusion,
        diffusion_coeff=config.diffusion_coeff,
        diffusion_noise_var=config.diffusion_noise_var,
        external_potential=config.external_potential,
        external_potential_params=config.external_potential_params,
        save_every=config.save_every,
        save_every_steps=config.save_every_steps,
        t0=config.t0,
    )
    if config.charge_values is not None:
        payload["charge_values"] = config.charge_values
    if config.charge_counts is not None:
        payload["charge_counts"] = config.charge_counts
    if config.init_radii is not None:
        payload["init_radii"] = config.init_radii
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def init_charges(n_particles, charge_values, charge_counts, rng):
    if charge_values is None and charge_counts is None:
        return np.ones(n_particles, dtype=np.float64)
    if charge_values is None or charge_counts is None:
        raise ValueError("charge_values and charge_counts must be set together")
    values = (float(charge_values[0]), float(charge_values[1]))
    counts = (int(charge_counts[0]), int(charge_counts[1]))
    return init_charges_two_populations(n_particles, values, counts, rng=rng)


def init_positions_by_charge_radii(charges, init_radii, rng):
    charges = np.asarray(charges, dtype=np.float64)
    unique = np.unique(charges)
    if unique.size != 2:
        raise ValueError(
            f"Expected exactly two charge populations for init_radii, found {unique.size}: "
            f"{unique.tolist()}"
        )

    low_charge = float(unique[0])
    high_charge = float(unique[1])
    low_mask = np.isclose(charges, low_charge)
    high_mask = np.isclose(charges, high_charge)

    positions = np.empty((charges.size, 2), dtype=np.float64)
    positions[low_mask] = init_positions_jittered_disk(
        int(np.count_nonzero(low_mask)),
        radius=float(init_radii["low"]),
        rng=rng,
    )
    positions[high_mask] = init_positions_jittered_disk(
        int(np.count_nonzero(high_mask)),
        radius=float(init_radii["high"]),
        rng=rng,
    )
    return positions


def init_fresh_positions_and_charges(config: SimulationConfig, rng):
    if config.init_radii is None:
        positions = init_positions_jittered_disk(
            config.n_particles,
            radius=config.init_radius,
            rng=rng,
        )
        charges = init_charges(config.n_particles, config.charge_values, config.charge_counts, rng)
        return positions, charges

    charges = init_charges(config.n_particles, config.charge_values, config.charge_counts, rng)
    positions = init_positions_by_charge_radii(charges, config.init_radii, rng)
    return positions, charges


def is_streaming_mode(config: SimulationConfig) -> bool:
    return (
        config.batch_every is not None
        or config.target_batch_mb is not None
        or config.max_wall_time is not None
        or config.resume_from is not None
    )


def compute_sample_count(t_duration: float, save_every: float | None) -> int | None:
    if save_every is None:
        return None
    return int(np.floor(t_duration / save_every)) + 1


def build_stream_meta(
    config: SimulationConfig,
    *,
    out_path: Path,
    elapsed_sec: float,
    completed: bool,
    rk23_stats,
    resumed_flag: bool,
) -> dict:
    return dict(
        n_particles=config.n_particles,
        k=config.k,
        v0=config.v0,
        l=config.l,
        t_duration=config.t_duration,
        save_every=config.save_every,
        save_every_steps=config.save_every_steps,
        method=config.method,
        seed=config.seed,
        r_floor=config.r_floor,
        init_radius=config.init_radius,
        init_radii=config.init_radii,
        charge_values=config.charge_values,
        charge_counts=config.charge_counts,
        t0=config.t0,
        rtol=config.rtol,
        atol=config.atol,
        first_step=config.first_step if config.method == "rk23" else None,
        max_step_global=config.max_step_global if config.method == "rk23" else None,
        eta=config.eta if config.method == "rk23" else None,
        recompute_every=config.recompute_every if config.method == "rk23" else None,
        interpolate_sampling=config.interpolate_sampling if config.method == "rk23" else None,
        diffusion=config.diffusion if config.method == "rk23" else None,
        diffusion_coeff=config.diffusion_coeff if config.method == "rk23" else None,
        diffusion_seed=config.diffusion_seed if config.method == "rk23" else None,
        diffusion_noise_var=config.diffusion_noise_var if config.method == "rk23" else None,
        external_potential=config.external_potential,
        external_potential_params=config.external_potential_params,
        rk23_stats=rk23_stats if config.method == "rk23" else None,
        batch_every=config.batch_every,
        target_batch_mb=config.target_batch_mb,
        max_wall_time=config.max_wall_time,
        resume_from=str(out_path) if resumed_flag else None,
        resumed=resumed_flag,
        completed=completed,
        out_format=config.out_format,
        elapsed_sec=elapsed_sec,
    )


def build_nonstream_meta(
    config: SimulationConfig,
    *,
    elapsed_sec: float,
    rk23_stats,
    t_hist_len: int,
) -> dict:
    return dict(
        n_particles=config.n_particles,
        k=config.k,
        v0=config.v0,
        l=config.l,
        t_duration=config.t_duration,
        save_every=config.save_every,
        save_every_steps=config.save_every_steps,
        method=config.method,
        seed=config.seed,
        r_floor=config.r_floor,
        init_radius=config.init_radius,
        init_radii=config.init_radii,
        charge_values=config.charge_values,
        charge_counts=config.charge_counts,
        t0=config.t0,
        rtol=config.rtol if config.method in ("dop853", "rk23") else None,
        atol=config.atol if config.method in ("dop853", "rk23") else None,
        first_step=config.first_step if config.method == "rk23" else None,
        max_step_global=config.max_step_global if config.method == "rk23" else None,
        eta=config.eta if config.method == "rk23" else None,
        recompute_every=config.recompute_every if config.method == "rk23" else None,
        interpolate_sampling=config.interpolate_sampling if config.method == "rk23" else None,
        diffusion=config.diffusion if config.method == "rk23" else None,
        diffusion_coeff=config.diffusion_coeff if config.method == "rk23" else None,
        diffusion_seed=config.diffusion_seed if config.method == "rk23" else None,
        diffusion_noise_var=config.diffusion_noise_var if config.method == "rk23" else None,
        external_potential=config.external_potential,
        external_potential_params=config.external_potential_params,
        rk23_stats=rk23_stats if config.method == "rk23" else None,
        chunk_steps=config.chunk_steps,
        batch_every=config.batch_every,
        target_batch_mb=config.target_batch_mb,
        max_wall_time=config.max_wall_time,
        resume_from=None,
        resumed=False,
        completed=True,
        out_format=config.out_format,
        elapsed_sec=elapsed_sec,
        steps_per_sec=(t_hist_len / elapsed_sec) if elapsed_sec > 0 else None,
    )


def build_saved_result(
    config: SimulationConfig,
    *,
    out_path: Path,
    final_positions: np.ndarray,
    charges: np.ndarray,
    meta: dict,
) -> dict:
    return dict(
        positions=np.empty((0, config.n_particles, 2), dtype=np.float64),
        times=np.empty(0, dtype=np.float64),
        energy=np.empty(0, dtype=np.float64),
        energy_aa=np.empty(0, dtype=np.float64),
        energy_ab=np.empty(0, dtype=np.float64),
        energy_bb=np.empty(0, dtype=np.float64),
        std=np.empty(0, dtype=np.float64),
        final_positions=final_positions,
        charges=charges,
        density=None,
        radii=None,
        meta=meta,
        saved_path=str(out_path),
    )
